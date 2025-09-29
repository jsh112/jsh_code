#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
두 장(기준/레이저) 이미지 차분으로 '레이저 같은' 변화만 엄선해 표시
- 채널 투표제(Gray / HSV-V / R) + 컬러 게이트(기본 red)
- 소형 영역 + 원형성 + 국소 대비(prominence) 필터
- 상위 점수 후보만 표시(topk)

키:
  'b'  기준 프레임 캡처
  'l'  레이저 프레임 캡처+분석
  's'  결과 저장
  'q'  종료

예)
  단일캠: python laser_diff_detect_strict.py --cam 0 --laser red --min_votes 2 --topk 1
  듀얼캠: python laser_diff_detect_strict.py --left 0 --right 1 --laser red
"""

import argparse, os, time
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ---------------------------
# 파라미터 구조체
# ---------------------------
class Params:
    def __init__(self,
                 k: float = 3.0,
                 min_votes: int = 2,
                 laser: str = "red",     # red / green / any
                 delta_r: int = 25,      # 레이저 컬러 게이트 강도(R-G, R-B)
                 delta_g: int = 25,      # green일 때 G-R/B
                 area_min: int = 1,      # 후보 최소 면적(px)
                 area_max: int = 40,     # 후보 최대 면적(px)
                 circ_min: float = 0.55, # 원형성 임계(4πA/P²)
                 prom_min: float = 18.0, # 국소 대비(중심-주변 평균)
                 topk: int = 1):
        self.k = k
        self.min_votes = min_votes
        self.laser = laser
        self.delta_r = delta_r
        self.delta_g = delta_g
        self.area_min = area_min
        self.area_max = area_max
        self.circ_min = circ_min
        self.prom_min = prom_min
        self.topk = topk

# ---------------------------
# 카메라 유틸
# ---------------------------
def set_manual_exposure(cap: cv2.VideoCapture, exposure: Optional[float]=None, gain: Optional[float]=None):
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(gain))

def warmup_and_grab(cap: cv2.VideoCapture, warmup=6) -> np.ndarray:
    ok, frame = False, None
    for _ in range(warmup):
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("카메라 프레임 획득 실패")
    return frame

# ---------------------------
# 차분/마스크
# ---------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def diffs(base: np.ndarray, test: np.ndarray):
    g0, g1 = to_gray(base), to_gray(test)
    d_gray = cv2.absdiff(g1, g0)

    hsv0 = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)
    d_v = cv2.absdiff(hsv1[:,:,2], hsv0[:,:,2])

    d_r = cv2.absdiff(test[:,:,2], base[:,:,2])
    return d_gray, d_v, d_r

def robust_thresh(img: np.ndarray, k: float, min_thr: int=5) -> np.ndarray:
    mean, std = float(np.mean(img)), float(np.std(img))
    T = max(min_thr, mean + k*std)
    _, m = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return m

def color_gate(test: np.ndarray, params: Params) -> np.ndarray:
    b, g, r = test[:,:,0].astype(np.int16), test[:,:,1].astype(np.int16), test[:,:,2].astype(np.int16)
    if params.laser.lower() == "red":
        gate = ((r - g) >= params.delta_r) & ((r - b) >= params.delta_r)
    elif params.laser.lower() == "green":
        gate = ((g - r) >= params.delta_g) & ((g - b) >= params.delta_g)
    else:
        # any: 색 조건 패스(전부 True)
        h, w = r.shape
        gate = np.ones((h, w), dtype=bool)
    return gate.astype(np.uint8)*255

def build_mask(base: np.ndarray, test: np.ndarray, params: Params) -> np.ndarray:
    d_gray, d_v, d_r = diffs(base, test)
    m_gray = robust_thresh(d_gray, params.k)
    m_v    = robust_thresh(d_v,    params.k)
    m_r    = robust_thresh(d_r,    params.k)

    # 투표제: 세 마스크 중 min_votes 이상이어야 통과
    votes = (m_gray>0).astype(np.uint8) + (m_v>0).astype(np.uint8) + (m_r>0).astype(np.uint8)
    mv = (votes >= params.min_votes).astype(np.uint8)*255

    # 컬러 게이트
    cg = color_gate(test, params)
    mask = cv2.bitwise_and(mv, cg)

    # 작은 잡음 제거(열림), 1~2px 시각화용 미세 팽창은 나중에
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask, (d_gray, d_v, d_r)

# ---------------------------
# 후보 추출/평가
# ---------------------------
def component_circularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    per  = cv2.arcLength(cnt, True)
    if per <= 1e-6: return 0.0
    return float(4.0*np.pi*area/(per*per))

def local_prominence(diff_gray: np.ndarray, pt: Tuple[int,int], win: int=7) -> float:
    x, y = pt
    h, w = diff_gray.shape
    r = win//2
    x0, x1 = max(0, x-r), min(w-1, x+r)
    y0, y1 = max(0, y-r), min(h-1, y+r)
    patch = diff_gray[y0:y1+1, x0:x1+1].astype(np.float32)
    if patch.size == 0: return 0.0
    center = patch[r if (y1-y0)>=r else 0, r if (x1-x0)>=r else 0]
    # 주변 평균(중심 제외)
    mask = np.ones_like(patch, dtype=bool)
    mask[r if (y1-y0)>=r else 0, r if (x1-x0)>=r else 0] = False
    neigh = patch[mask]
    if neigh.size == 0: return float(center)
    return float(center - np.mean(neigh))

def pick_candidates(mask: np.ndarray,
                    test: np.ndarray,
                    d_gray: np.ndarray,
                    d_v: np.ndarray,
                    d_r: np.ndarray,
                    params: Params):
    num, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    cand = []
    for i in range(1, num):
        x,y,w,h,a = stats[i]
        if a < params.area_min or a > params.area_max:
            continue
        # 컨투어/원형성
        m = (lbl == i).astype(np.uint8)*255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        circ = component_circularity(c)
        if circ < params.circ_min:
            continue

        cx, cy = int(round(cent[i][0])), int(round(cent[i][1]))
        if cx<0 or cy<0 or cx>=W or cy>=H:
            continue

        prom = local_prominence(d_gray, (cx, cy), win=7)

        # 점수: 국소대비 + 가중치*채널차분
        score = prom + 0.2*float(d_v[cy, cx]) + 0.3*float(d_r[cy, cx]) + 10.0*circ
        if prom < params.prom_min:
            continue

        cand.append({
            "bbox": (x,y,w,h),
            "center": (cx, cy),
            "area": a,
            "circ": circ,
            "prom": prom,
            "score": score
        })

    # 점수 상위 topk
    cand.sort(key=lambda z: -z["score"])
    return cand[:max(1, params.topk)]

# ---------------------------
# 시각화
# ---------------------------

def put_text_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, 
                     font_scale=0.5, text_color=(255,255,255), bg_color=(0,0,0),
                     thickness=1, pad=3):
    """
    텍스트 가독성을 높이기 위해 반투명 배경박스를 깔고 텍스트를 그린다.
    """
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # 배경 박스 좌표
    box_tl = (x - pad, y - th - baseline - pad)
    box_br = (x + tw + pad, y + baseline + pad)
    # 이미지 경계 처리
    H, W = img.shape[:2]
    box_tl = (max(0, box_tl[0]), max(0, box_tl[1]))
    box_br = (min(W-1, box_br[0]), min(H-1, box_br[1]))

    # 반투명 배경
    overlay = img.copy()
    cv2.rectangle(overlay, box_tl, box_br, bg_color, -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # 텍스트
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def overlay_result(test: np.ndarray, mask: np.ndarray, cands: List[dict],
                   title: str = "Laser Diff Detection",
                   show_global_count: bool = True,
                   point_label: str = "LASER"):
    """
    - 검출 마스크 빨간 히트맵으로 오버레이
    - 각 후보 중심에 마커 + 원형 링 표시
    - 각 후보 옆에 인덱스/좌표/점수 텍스트 라벨 추가
    """
    vis = test.copy()

    # 1) 히트맵 오버레이(빨강)
    kernel = np.ones((2,2), np.uint8)
    vis_mask = cv2.dilate(mask, kernel, iterations=1)
    color_mask = np.zeros_like(test)
    color_mask[vis_mask > 0] = (0, 0, 255)  # BGR: red
    vis = cv2.addWeighted(vis, 1.0, color_mask, 0.45, 0)

    # 2) 후보 표시
    for i, c in enumerate(cands):
        (cx, cy) = c["center"]
        (x, y, w, h) = c["bbox"]
        score = c.get("score", 0.0)

        # 중심 마커 + 얇은 링(시각적 강조)
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 0), 1, cv2.LINE_AA)

        # 바운딩 박스(점 주변 시각화)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)

        # 3) 점 라벨 텍스트 (인덱스/좌표/점수)
        label = f"{point_label} #{i}  ({cx},{cy})  S={score:.1f}"
        # 텍스트 위치: 박스 왼쪽 위 살짝 위로
        tx, ty = x, max(0, y - 8)
        put_text_with_bg(vis, label, (tx, ty), font_scale=0.55, 
                         text_color=(255,255,255), bg_color=(0,0,0), thickness=1)

    # 4) 상단 제목 + 전체 개수
    if show_global_count:
        header = f"{title}  |  candidates: {len(cands)}"
    else:
        header = title
    put_text_with_bg(vis, header, (10, 24), font_scale=0.7, 
                     text_color=(255,255,255), bg_color=(30,30,30), thickness=2)

    return vis

# ---------------------------
# 분석 파이프
# ---------------------------
def analyze_once(base: np.ndarray, test: np.ndarray, params: Params):
    mask, (d_gray, d_v, d_r) = build_mask(base, test, params)
    cands = pick_candidates(mask, test, d_gray, d_v, d_r, params)
    vis = overlay_result(
    test, mask, cands,
    title="Laser Diff Detection",
    show_global_count=True,
    point_label="LASER")
    return vis, cands

def ensure_dir(p: str):
    if not os.path.isdir(p): os.makedirs(p, exist_ok=True)

# ---------------------------
# 실행 루프 (단일/듀얼)
# ---------------------------
def run_single(cam: int, params: Params):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError(f"카메라 열기 실패: {cam}")
    set_manual_exposure(cap)

    base = None
    last = None
    print("[단일] 'b' 기준, 'l' 분석, 's' 저장, 'q' 종료")
    while True:
        ok, frame = cap.read()
        if not ok: continue
        disp = frame.copy()
        if base is not None:
            cv2.putText(disp, "BASE SET", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
        cv2.imshow("Live - Single", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            base = warmup_and_grab(cap, warmup=6)
            print("기준 프레임 캡처 완료")
        elif k == ord('l'):
            if base is None:
                print("먼저 'b'로 기준 프레임 캡처")
                continue
            test = warmup_and_grab(cap, warmup=3)

            # 분석 & 표시
            vis, cands = analyze_once(base, test, params)
            last = vis
            cv2.imshow("Diff Result - Single", vis)
            if cands:
                print(f"선정 후보 {len(cands)}개 (topk={params.topk}) → centers: {[c['center'] for c in cands]}")
            else:
                print("레이저로 볼 만한 변화 없음")

            # ★ 중요: 현재 프레임을 다음 비교를 위한 '새 기준'으로 업데이트
            base = test.copy()
            print("기준 프레임(베이스) 갱신 완료.")
        elif k == ord('s'):
            if last is not None:
                ensure_dir("diff_out")
                p = os.path.join("diff_out", f"single_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(p, last)
                print("저장:", p)
            else:
                print("저장할 결과 없음")
        elif k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_dual(left: int, right: int, params: Params):
    capL = cv2.VideoCapture(left,  cv2.CAP_DSHOW)
    capR = cv2.VideoCapture(right, cv2.CAP_DSHOW)
    if not (capL.isOpened() and capR.isOpened()): raise RuntimeError("듀얼 카메라 열기 실패")
    set_manual_exposure(capL); set_manual_exposure(capR)

    baseL = baseR = None
    lastL = lastR = None
    print("[듀얼] 'b' 기준, 'l' 분석, 's' 저장, 'q' 종료")
    while True:
        okL, frmL = capL.read()
        okR, frmR = capR.read()
        if not (okL and okR): continue
        live = np.hstack([frmL, frmR])
        if baseL is not None and baseR is not None:
            cv2.putText(live, "BASE L/R SET", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
        cv2.imshow("Live - Dual [L|R]", live)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            baseL = warmup_and_grab(capL, 6)
            baseR = warmup_and_grab(capR, 6)
            print("양쪽 기준 캡처 완료")
        elif k == ord('l'):
            if baseL is None or baseR is None:
                print("먼저 'b'로 기준 캡처")
                continue
            testL = warmup_and_grab(capL, 3)
            testR = warmup_and_grab(capR, 3)

            # 분석 & 표시
            visL, cL = analyze_once(baseL, testL, params)
            visR, cR = analyze_once(baseR, testR, params)
            lastL, lastR = visL, visR
            cv2.imshow("Diff Result - Dual [L|R]", np.hstack([visL, visR]))
            print(f"[L] {len(cL)}개 { [c['center'] for c in cL] }  /  [R] {len(cR)}개 { [c['center'] for c in cR] }")

            # ★ 중요: 현재 프레임을 다음 비교를 위한 '새 기준'으로 업데이트
            baseL = testL.copy()
            baseR = testR.copy()
            print("기준 프레임(좌/우) 갱신 완료.")

        elif k == ord('s'):
            if lastL is not None and lastR is not None:
                ensure_dir("diff_out")
                ts = time.strftime('%Y%m%d_%H%M%S')
                pL = os.path.join("diff_out", f"dual_left_{ts}.png")
                pR = os.path.join("diff_out", f"dual_right_{ts}.png")
                cv2.imwrite(pL, lastL); cv2.imwrite(pR, lastR)
                print("저장:\n ", pL, "\n ", pR)
            else:
                print("저장할 결과 없음")
        elif k == ord('q'):
            break
    capL.release(); capR.release()
    cv2.destroyAllWindows()

# ---------------------------
# 엔트리포인트
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="두 장 차분으로 레이저 같은 변화만 검출/표시(엄격 필터)")
    ap.add_argument("--cam", type=int, default=None, help="단일 카메라 인덱스")
    ap.add_argument("--left", type=int, default=None, help="듀얼: 좌 카메라 인덱스")
    ap.add_argument("--right", type=int, default=None, help="듀얼: 우 카메라 인덱스")
    ap.add_argument("--k", type=float, default=3.0, help="적응 임계값 계수(크게=보수적)")
    ap.add_argument("--min_votes", type=int, default=2, help="채널 투표제 최소 득표 수(1~3)")
    ap.add_argument("--laser", type=str, default="red", choices=["red","green","any"], help="레이저 색 가정")
    ap.add_argument("--delta_r", type=int, default=25, help="red 색 게이트 강도(R-G, R-B)")
    ap.add_argument("--delta_g", type=int, default=25, help="green 색 게이트 강도(G-R, G-B)")
    ap.add_argument("--area_min", type=int, default=1, help="최소 면적(px)")
    ap.add_argument("--area_max", type=int, default=40, help="최대 면적(px)")
    ap.add_argument("--circ_min", type=float, default=0.55, help="원형성 임계(0~1)")
    ap.add_argument("--prom_min", type=float, default=18.0, help="국소 대비 임계")
    ap.add_argument("--topk", type=int, default=1, help="상위 몇 개만 남길지")
    return ap.parse_args()

def main():
    args = parse_args()
    params = Params(k=args.k, min_votes=args.min_votes, laser=args.laser,
                    delta_r=args.delta_r, delta_g=args.delta_g,
                    area_min=args.area_min, area_max=args.area_max,
                    circ_min=args.circ_min, prom_min=args.prom_min,
                    topk=args.topk)

    if args.cam is not None and (args.left is not None or args.right is not None):
        raise SystemExit("단일(--cam)과 듀얼(--left/--right) 동시 지정 불가")

    if args.cam is not None:
        run_single(args.cam, params)
    else:
        if args.left is None or args.right is None:
            raise SystemExit("듀얼 모드에는 --left, --right가 모두 필요합니다(또는 --cam).")
        run_dual(args.left, args.right, params)

if __name__ == "__main__":
    main()
