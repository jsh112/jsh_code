#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
두 장(기준/레이저) 이미지 차분으로 '레이저 같은' 변화만 엄선해 표시 (강화판)
- 모드:
  1) color: 채널 투표제(Gray / HSV-V / R) + 컬러 게이트(기본 red), 비율상승/양의차분/정적빨강 억제/LoG 점강조
  2) gray_tophat: 색 배제, 그레이 White Top-Hat + 양의차분 + 자동임계 + NMS(지역최대)
- 공통: 소형 영역 + 원형성 + 국소 대비(prominence) 필터, 점수화 + topk
- 'l' 누를 때마다 현재 프레임을 새 기준으로 갱신(슬라이딩 비교)
- 옵션: b 누르기 직전 지정 모니터에 검정 화면 잠깐 띄워 SNR↑ (blank_on_b)
- 옵션: 카메라 노출/게인 명시 고정(exp/gain)
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
                 # 모드
                 mode: str = "color",   # "color" or "gray_tophat"
                 spot_radius: int = 3,  # gray_tophat 모드에서 점 크기 반지름(px)

                 # 통계 임계
                 k: float = 3.0,
                 min_votes: int = 2,

                 # 레이저 색 (color 모드에서만 유효)
                 laser: str = "red",     # red / green / any
                 delta_r: int = 25,      # red 색 게이트 강도(R-G, R-B)
                 delta_g: int = 25,      # green일 때 G-R/B

                 # 양의 차분 스레시홀드 (color 모드)
                 pos_thr_gray: float = 1.5,
                 pos_thr_v: float = 1.5,
                 pos_thr_r: float = 1.5,

                 # R 비율 상승 기준 (color 모드)
                 min_rg_rise: float = 1.25,
                 min_rb_rise: float = 1.25,

                 # 정적 빨강 억제 & 강한 증가 허용 (color 모드)
                 static_red_margin: int = 20,
                 strong_r_increase_thr: int = 12,

                 # 점형(LoG) 필터 (color 모드에서 선택)
                 use_spotness: bool = True,
                 spot_sigma: float = 1.0,
                 spot_thr: int = 8,

                 # 후보 필터(공통)
                 area_min: int = 1,
                 area_max: int = 40,
                 circ_min: float = 0.55,
                 prom_min: float = 18.0,

                 # 출력 후보 수
                 topk: int = 1,

                 # --- 블랭크 스크린 옵션 ---
                 blank_on_b: bool = False,
                 blank_ms: int = 300,
                 blank_x: int = 0,
                 blank_y: int = 0,
                 blank_w: int = 1920,
                 blank_h: int = 1080,

                 # --- 카메라 노출/게인 명시 (선택) ---
                 exposure: Optional[float] = None,
                 gain: Optional[float] = None,
                 ):
        self.mode = mode
        self.spot_radius = spot_radius

        self.k = k
        self.min_votes = min_votes
        self.laser = laser
        self.delta_r = delta_r
        self.delta_g = delta_g

        self.pos_thr_gray = pos_thr_gray
        self.pos_thr_v = pos_thr_v
        self.pos_thr_r = pos_thr_r

        self.min_rg_rise = min_rg_rise
        self.min_rb_rise = min_rb_rise

        self.static_red_margin = static_red_margin
        self.strong_r_increase_thr = strong_r_increase_thr

        self.use_spotness = use_spotness
        self.spot_sigma = spot_sigma
        self.spot_thr = spot_thr

        self.area_min = area_min
        self.area_max = area_max
        self.circ_min = circ_min
        self.prom_min = prom_min
        self.topk = topk

        self.blank_on_b = blank_on_b
        self.blank_ms = blank_ms
        self.blank_x = blank_x
        self.blank_y = blank_y
        self.blank_w = blank_w
        self.blank_h = blank_h

        self.exposure = exposure
        self.gain = gain

# ---------------------------
# 카메라/화면 유틸
# ---------------------------
def set_manual_exposure(cap: cv2.VideoCapture, exposure: Optional[float]=None, gain: Optional[float]=None):
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 일부 드라이버에서 0.25=수동
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

def show_black_fullscreen(x: int, y: int, w: int, h: int, hold_ms: int = 300, win_name: str = "BLANK"):
    """
    (x,y) 위치에 w×h 크기의 검정 창을 띄우고 hold_ms ms 유지.
    보조 모니터(프로젝터)가 있다면 그 좌표/해상도로 지정.
    """
    black = np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, x, y)
    cv2.resizeWindow(win_name, w, h)
    cv2.imshow(win_name, black)
    cv2.waitKey(1)
    t0 = time.time()
    while (time.time() - t0) * 1000.0 < hold_ms:
        cv2.waitKey(1)
    cv2.destroyWindow(win_name)

# ---------------------------
# 차분/마스크 공통 유틸
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
        h, w = r.shape
        gate = np.ones((h, w), dtype=bool)
    return gate.astype(np.uint8)*255

def posdiff(img_now: np.ndarray, img_base: np.ndarray, thr: float) -> np.ndarray:
    """양의 차분(현재-기준)만 남김"""
    d = cv2.subtract(img_now, img_base)  # 음수 clamp
    _, m = cv2.threshold(d, float(thr), 255, cv2.THRESH_BINARY)
    return m

# ---------------------------
# 색 기반(color)용 보조 유틸
# ---------------------------
def ratio_rise_mask(r_now, g_now, b_now, r_base, g_base, b_base,
                    min_rg_rise=1.2, min_rb_rise=1.2) -> np.ndarray:
    eps = 1.0
    rg_now = (r_now.astype(np.float32)+eps)/(g_now.astype(np.float32)+eps)
    rb_now = (r_now.astype(np.float32)+eps)/(b_now.astype(np.float32)+eps)
    rg_base = (r_base.astype(np.float32)+eps)/(g_base.astype(np.float32)+eps)
    rb_base = (r_base.astype(np.float32)+eps)/(b_base.astype(np.float32)+eps)
    m = (rg_now >= rg_base*min_rg_rise) & (rb_now >= rb_base*min_rb_rise)
    return (m.astype(np.uint8))*255

def spotness_mask(gray_img: np.ndarray, sigma=1.0, thr=8) -> np.ndarray:
    """작고 동그란 블롭(레이저 점) 강조: LoG 응답 기반"""
    blur = cv2.GaussianBlur(gray_img, (0,0), sigma)
    lap  = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    lap  = cv2.convertScaleAbs(lap)
    _, m = cv2.threshold(lap, thr, 255, cv2.THRESH_BINARY)
    return m

# ---------------------------
# 색 기반(color) 마스크 빌더
# ---------------------------
def build_mask(base: np.ndarray, test: np.ndarray, params: Params):
    # 절대차분
    d_gray, d_v, d_r = diffs(base, test)

    # 양의 차분 (현재 > 이전)
    g0, g1 = to_gray(base), to_gray(test)
    hsv0 = cv2.cvtColor(base, cv2.COLOR_BGR2HSV); hsv1 = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)
    v0, v1 = hsv0[:,:,2], hsv1[:,:,2]
    r0, g0c, b0 = base[:,:,2], base[:,:,1], base[:,:,0]
    r1, g1c, b1 = test[:,:,2], test[:,:,1], test[:,:,0]

    m_gray_pos = posdiff(g1, g0, thr=params.pos_thr_gray)
    m_v_pos    = posdiff(v1, v0, thr=params.pos_thr_v)
    m_r_pos    = posdiff(r1, r0, thr=params.pos_thr_r)

    # R비율 상승
    m_ratio = ratio_rise_mask(r1, g1c, b1, r0, g0c, b0,
                              min_rg_rise=params.min_rg_rise,
                              min_rb_rise=params.min_rb_rise)

    # 통계 임계
    m_gray = robust_thresh(d_gray, params.k)
    m_v    = robust_thresh(d_v,    params.k)
    m_r    = robust_thresh(d_r,    params.k)

    # 투표
    votes = (m_gray_pos>0).astype(np.uint8) + (m_v_pos>0).astype(np.uint8) + (m_r_pos>0).astype(np.uint8) \
          + (m_ratio>0).astype(np.uint8) \
          + (m_gray>0).astype(np.uint8) + (m_v>0).astype(np.uint8) + (m_r>0).astype(np.uint8)
    mv = (votes >= max(2, params.min_votes)).astype(np.uint8)*255

    # 색 게이트 + 정적 빨강 억제(단, 강한 R증가는 허용)
    cg = color_gate(test, params)
    static_red = (((r0.astype(int) - np.maximum(g0c.astype(int), b0.astype(int))) >=
                  max(params.static_red_margin, params.delta_r-5))).astype(np.uint8)*255
    strong_increase = posdiff(r1, r0, thr=params.strong_r_increase_thr)
    not_static_red_or_strong = cv2.bitwise_or(cv2.bitwise_not(static_red), strong_increase)

    mask = cv2.bitwise_and(mv, cg)
    mask = cv2.bitwise_and(mask, not_static_red_or_strong)

    # 점형(LoG) 응답과 AND (옵션)
    if params.use_spotness:
        m_spot = spotness_mask(g1, sigma=params.spot_sigma, thr=params.spot_thr)
        mask = cv2.bitwise_and(mask, m_spot)

    # 소형 잡음 제거
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask, (d_gray, d_v, d_r)

# ---------------------------
# 색 배제(gray_tophat) 유틸
# ---------------------------
def white_tophat(gray: np.ndarray, radius: int) -> np.ndarray:
    k = 2*radius + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

def auto_thresh(img: np.ndarray) -> np.ndarray:
    # 값 분포가 빈약하면 0 반환
    if np.count_nonzero(img) < 10:
        return np.zeros_like(img, dtype=np.uint8)
    try:
        v = img[img > 0]
        if v.size < 10:
            return np.zeros_like(img, dtype=np.uint8)
        _, m = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if np.count_nonzero(m) == 0:
            mean, std = float(np.mean(v)), float(np.std(v))
            T = max(1, int(mean + 3*std))
            _, m = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
        return m
    except Exception:
        mean, std = float(np.mean(img)), float(np.std(img))
        T = max(1, int(mean + 3*std))
        _, m = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
        return m

def nms_peaks(img: np.ndarray, win: int = 5) -> np.ndarray:
    # 지역 최대값만 남김 (비최대 억제)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dil = cv2.dilate(img, kernel)
    peaks = (img == dil) & (img > 0)
    return (peaks.astype(np.uint8))*255

# ---------------------------
# 색 배제(gray_tophat) 마스크 빌더
# ---------------------------
def build_mask_gray_tophat(base: np.ndarray, test: np.ndarray, params: Params):
    # 그레이
    g0, g1 = to_gray(base), to_gray(test)

    # 작은 밝은 점 강조 (색 무관)
    th0 = white_tophat(g0, params.spot_radius)
    th1 = white_tophat(g1, params.spot_radius)

    # 양의 차분: 현재가 더 밝아진 부분만
    pos = cv2.subtract(th1, th0)

    # 자동 임계 + NMS
    m = auto_thresh(pos)
    if np.count_nonzero(m) > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
        peaks = nms_peaks(pos, win=max(3, 2*params.spot_radius+1))
        m = cv2.bitwise_and(m, peaks)

    # 참고용 채널(후속 점수/라벨에 사용)
    d_gray = cv2.absdiff(g1, g0)
    d_v = d_gray.copy()
    d_r = d_gray.copy()
    return m, (d_gray, d_v, d_r)

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
    cx = r if (x1-x0)>=r else 0
    cy = r if (y1-y0)>=r else 0
    center = patch[cy, cx]
    mask = np.ones_like(patch, dtype=bool)
    mask[cy, cx] = False
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

        # 점수: 국소대비 + V/R 증감 + 원형성 + R비율(현재 프레임)
        eps = 1.0
        r_now, g_now, b_now = float(test[cy, cx, 2]), float(test[cy, cx, 1]), float(test[cy, cx, 0])
        ratio_boost = (r_now+eps)/(((g_now+b_now)/2.0)+eps)

        score = (1.0 * prom) + (0.3 * float(d_v[cy, cx])) + (0.5 * float(d_r[cy, cx])) \
                + (10.0 * circ) + (4.0 * ratio_boost)

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

    cand.sort(key=lambda z: -z["score"])
    return cand[:max(1, params.topk)]

# ---------------------------
# 시각화
# ---------------------------
def paint_transparent_rect(img, x, y, w, h, color=(0, 0, 255), alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, 
                     font_scale=0.5, text_color=(255,255,255), bg_color=(0,0,0),
                     thickness=1, pad=3):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    box_tl = (x - pad, y - th - baseline - pad)
    box_br = (x + tw + pad, y + baseline + pad)
    H, W = img.shape[:2]
    box_tl = (max(0, box_tl[0]), max(0, box_tl[1]))
    box_br = (min(W-1, box_br[0]), min(H-1, box_br[1]))
    overlay = img.copy()
    cv2.rectangle(overlay, box_tl, box_br, bg_color, -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def overlay_result(test: np.ndarray, mask: np.ndarray, cands: List[dict],
                   title: str = "Laser Diff Detection",
                   show_global_count: bool = True,
                   point_label: str = "LASER",
                   cand_prefix: str = "cand"):
    """
    - 전체 마스크는 연한 빨강 오버레이
    - '최종 후보'는 bbox를 더 진한 빨강 반투명으로 하이라이트 + green cross
    - 후보 텍스트 라벨: cand1, cand2 ... (cand_prefix 변경 가능)
    """
    vis = test.copy()

    # 1) 1차 후보(마스크) → 연한 빨강 히트맵
    kernel = np.ones((2,2), np.uint8)
    vis_mask = cv2.dilate(mask, kernel, iterations=1)
    color_mask = np.zeros_like(test)
    color_mask[vis_mask > 0] = (0, 0, 255)  # BGR red
    vis = cv2.addWeighted(vis, 1.0, color_mask, 0.35, 0)

    # 2) 최종 후보 강조
    for i, c in enumerate(cands):
        (cx, cy) = c["center"]
        (x, y, w, h) = c["bbox"]
        score = c.get("score", 0.0)

        # 후보 bbox 반투명 빨강
        paint_transparent_rect(vis, x, y, w, h, color=(0,0,255), alpha=0.45)

        # 중심 마커 + 얇은 원
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 0), 1, cv2.LINE_AA)

        # cand 라벨
        cand_name = f"{cand_prefix}{i+1}"
        label = f"{cand_name}  ({cx},{cy})  S={score:.1f}"
        tx, ty = x, max(0, y - 8)
        put_text_with_bg(vis, label, (tx, ty), font_scale=0.55,
                         text_color=(255,255,255), bg_color=(0,0,0), thickness=1)

    # 3) 상단 헤더
    header = f"{title}  |  candidates: {len(cands)}" if show_global_count else title
    put_text_with_bg(vis, header, (10, 24), font_scale=0.7,
                     text_color=(255,255,255), bg_color=(30,30,30), thickness=2)

    return vis

# ---------------------------
# 분석 파이프
# ---------------------------
def analyze_once(base: np.ndarray, test: np.ndarray, params: Params):
    if params.mode == "gray_tophat":
        mask, (d_gray, d_v, d_r) = build_mask_gray_tophat(base, test, params)
    else:
        mask, (d_gray, d_v, d_r) = build_mask(base, test, params)
    cands = pick_candidates(mask, test, d_gray, d_v, d_r, params)
    vis = overlay_result(test, mask, cands,
                         title=f"Laser Diff Detection [{params.mode}]",
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
    set_manual_exposure(cap, params.exposure, params.gain)

    base = None
    last = None
    print("[단일] 'b' 기준, 'l' 분석(끝나면 새 기준으로 갱신), 's' 저장, 'q' 종료")
    while True:
        ok, frame = cap.read()
        if not ok: continue
        disp = frame.copy()
        if base is not None:
            cv2.putText(disp, "BASE SET", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
        cv2.imshow("Live - Single", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            if params.blank_on_b:
                show_black_fullscreen(params.blank_x, params.blank_y,
                                      params.blank_w, params.blank_h,
                                      hold_ms=max(100, params.blank_ms))
            base = warmup_and_grab(cap, warmup=6)
            print("기준 프레임 캡처 완료")
        elif k == ord('l'):
            if base is None:
                print("먼저 'b'로 기준 프레임 캡처")
                continue
            test = warmup_and_grab(cap, warmup=3)
            vis, cands = analyze_once(base, test, params)
            last = vis
            cv2.imshow("Diff Result - Single", vis)
            if cands:
                print(f"선정 후보 {len(cands)}개 (topk={params.topk}) → centers: {[c['center'] for c in cands]}")
            else:
                print("레이저로 볼 만한 변화 없음")
            base = test.copy()  # 슬라이딩 기준 갱신
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
    set_manual_exposure(capL, params.exposure, params.gain)
    set_manual_exposure(capR, params.exposure, params.gain)

    baseL = baseR = None
    lastL = lastR = None
    print("[듀얼] 'b' 기준, 'l' 분석(끝나면 새 기준으로 갱신), 's' 저장, 'q' 종료")
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
            if params.blank_on_b:
                show_black_fullscreen(params.blank_x, params.blank_y,
                                      params.blank_w, params.blank_h,
                                      hold_ms=max(100, params.blank_ms))
            baseL = warmup_and_grab(capL, 6)
            baseR = warmup_and_grab(capR, 6)
            print("양쪽 기준 캡처 완료")
        elif k == ord('l'):
            if baseL is None or baseR is None:
                print("먼저 'b'로 기준 캡처")
                continue
            testL = warmup_and_grab(capL, 3)
            testR = warmup_and_grab(capR, 3)
            visL, cL = analyze_once(baseL, testL, params)
            visR, cR = analyze_once(baseR, testR, params)
            lastL, lastR = visL, visR
            cv2.imshow("Diff Result - Dual [L|R]", np.hstack([visL, visR]))
            print(f"[L] {len(cL)}개 { [c['center'] for c in cL] }  /  [R] {len(cR)}개 { [c['center'] for c in cR] }")
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
    ap = argparse.ArgumentParser(description="두 장 차분으로 레이저 같은 변화만 검출/표시(강화판)")

    # 모드
    ap.add_argument("--mode", type=str, default="color",
                    choices=["color","gray_tophat"], help="색 기반 or 색 배제(탑햇) 모드")
    ap.add_argument("--spot_radius", type=int, default=3, help="점 크기 반지름(px), gray_tophat 모드용")

    # 카메라
    ap.add_argument("--cam", type=int, default=None, help="단일 카메라 인덱스")
    ap.add_argument("--left", type=int, default=None, help="듀얼: 좌 카메라 인덱스")
    ap.add_argument("--right", type=int, default=None, help="듀얼: 우 카메라 인덱스")

    # 기본 임계 파라미터
    ap.add_argument("--k", type=float, default=3.0, help="통계 임계 k(크면 보수적)")
    ap.add_argument("--min_votes", type=int, default=2, help="투표제 최소 득표 수(1~7)")

    # 색 (color 모드)
    ap.add_argument("--laser", type=str, default="red", choices=["red","green","any"], help="레이저 색")
    ap.add_argument("--delta_r", type=int, default=25, help="red 컬러게이트 강도")
    ap.add_argument("--delta_g", type=int, default=25, help="green 컬러게이트 강도")

    # 양의 차분 (color 모드)
    ap.add_argument("--pos_thr_gray", type=float, default=2)
    ap.add_argument("--pos_thr_v", type=float, default=2)
    ap.add_argument("--pos_thr_r", type=float, default=2)

    # 비율 상승 (color 모드)
    ap.add_argument("--min_rg_rise", type=float, default=1.25)
    ap.add_argument("--min_rb_rise", type=float, default=1.25)

    # 정적 빨강 억제 (color 모드)
    ap.add_argument("--static_red_margin", type=int, default=20)
    ap.add_argument("--strong_r_increase_thr", type=int, default=12)

    # 점형(LoG) (color 모드)
    ap.add_argument("--use_spotness", action="store_true", help="LoG 점형 필터 사용(켜면 강함)")
    ap.add_argument("--spot_sigma", type=float, default=1.0)
    ap.add_argument("--spot_thr", type=int, default=8)

    # 후보 필터(공통)
    ap.add_argument("--area_min", type=int, default=1)
    ap.add_argument("--area_max", type=int, default=40)
    ap.add_argument("--circ_min", type=float, default=0.55)
    ap.add_argument("--prom_min", type=float, default=18.0)

    ap.add_argument("--topk", type=int, default=1)

    # 블랭크 스크린
    ap.add_argument("--blank_on_b", action="store_true", help="b 누르기 전 검정 화면 잠깐 띄우기")
    ap.add_argument("--blank_ms", type=int, default=300, help="검정 유지 시간(ms)")
    ap.add_argument("--blank_x", type=int, default=0)
    ap.add_argument("--blank_y", type=int, default=0)
    ap.add_argument("--blank_w", type=int, default=1920)
    ap.add_argument("--blank_h", type=int, default=1080)

    # 노출/게인
    ap.add_argument("--exp", type=float, default=None, help="카메라 노출값(드라이버 단위)")
    ap.add_argument("--gain", type=float, default=None, help="카메라 게인")

    return ap.parse_args()

def main():
    args = parse_args()
    params = Params(
        mode=args.mode, spot_radius=args.spot_radius,
        k=args.k, min_votes=args.min_votes, laser=args.laser,
        delta_r=args.delta_r, delta_g=args.delta_g,
        pos_thr_gray=args.pos_thr_gray, pos_thr_v=args.pos_thr_v, pos_thr_r=args.pos_thr_r,
        min_rg_rise=args.min_rg_rise, min_rb_rise=args.min_rb_rise,
        static_red_margin=args.static_red_margin, strong_r_increase_thr=args.strong_r_increase_thr,
        use_spotness=args.use_spotness, spot_sigma=args.spot_sigma, spot_thr=args.spot_thr,
        area_min=args.area_min, area_max=args.area_max,
        circ_min=args.circ_min, prom_min=args.prom_min,
        topk=args.topk,
        blank_on_b=args.blank_on_b, blank_ms=args.blank_ms,
        blank_x=args.blank_x, blank_y=args.blank_y,
        blank_w=args.blank_w, blank_h=args.blank_h,
        exposure=args.exp, gain=args.gain
    )

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

# py laser_diff_detect_black.py --left 1 --right 2 --mode gray_tophat --spot_radius 3 --min_votes 1 --k 2.0 --prom_min 6 --circ_min 0.30 --area_max 150 --topk 2 --blank_on_b --blank_ms 350 --blank_x 1920 --blank_y 0 --blank_w 1920 --blank_h 1080 --exp -6 --gain 0
