#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
route_recorder.py — 숙련자 등반 경로 캡처 스크립트 (v3.2, 2025-09-23)

목적(미니멀 라벨러):
- 숙련자가 실제로 사용한 홀드들을 '순서대로' CSV로 저장
- **평가 규칙(350ms)은 기록하지 않음** — 수집은 좌표·순서에만 집중
- MediaPipe Pose로 손/발 랜드마크를 추적하고, "홀드 폴리곤 내부에 연속 N프레임(기본 4)" 있으면 자동 저장
- 손/발은 **자동 판별**되어 CSV의 `contact_part`에 기록
- 요청 반영: **hold_id 포함**(초기 병합/인덱싱에서 부여된 공통 ID)

CSV 스키마(default):
seq, hold_id, cxL, cyL, cxR, cyR, X_mm, Y_mm, Z_mm, contact_part, timestamp
  - seq: 경로 내 순서(0부터)
  - hold_id: 초기 10프레임 병합/인덱싱 결과의 공통 ID(세션별로 달라질 수 있음)
  - (cxL,cyL)/(cxR,cyR): 좌/우 카메라에서의 홀드 중심점 픽셀 좌표(세그먼트 컨투어 모멘트)
  - (X_mm,Y_mm,Z_mm): 삼각측량한 3D 좌표(mm, LEFT 카메라 좌표계)
  - contact_part: hand|foot (MediaPipe 랜드마크로 자동 판별)
  - timestamp: 행 저장 시각(Unix epoch). --no_timestamp로 비활성화 가능

사용 예시:
  python route_recorder.py                                 # 실행 시 색상 선택 메뉴 + routes/ 자동 저장
  python route_recorder.py --colors "orange,blue"          # 여러 색 허용
  python route_recorder.py --colors all --label expertA    # 전체 허용 + 파일명 라벨
  python route_recorder.py --parts both --confirm_frames 3 # 손/발 모두 사용 + 디바운스 프레임 조정

주의/설계 철학:
- **hold_id는 편의용**(즉시 재생에 유리). 다른 세션에서 재생할 땐 YOLO 결과가 달라질 수 있으므로,
  combined_test.py에서는 3D/2D 근접 매칭으로 보정하는 로직을 추가 권장.
- yaw/pitch는 재생 시(최신 보정값) 계산을 권장.
"""

import cv2
import numpy as np
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime

# ===== 경로 설정 =====
REPO_ROOT = Path(__file__).resolve().parent  # 스크립트 위치 기준(리포 루트로 사용)
ROUTES_DIR = REPO_ROOT / "routes"
ROUTES_DIR.mkdir(parents=True, exist_ok=True)

# 스테레오/모델 기본 경로(환경에 맞게 수정)
NPZ_PATH   = REPO_ROOT / "calib_out/old_camera_same/stereo/stereo_params_scaled.npz"
MODEL_PATH = REPO_ROOT / "best_6.pt"

# 카메라 인덱스
CAM1_INDEX = 1   # LEFT (필요 시 --left_cam 로 덮어쓰기)
CAM2_INDEX = 2   # RIGHT (필요 시 --right_cam 로 덮어쓰기)

# 디스플레이/입력 스왑 여부
SWAP_INPUT   = False
SWAP_DISPLAY = False

WINDOW_NAME = "Route Recorder (L|R)"
THRESH_MASK = 0.7
ROW_TOL_Y   = 30

# ---- 색상 매핑 (YOLO 클래스명 매핑) ----
COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}
COLOR_ALIAS = {
    'red':'Hold_Red','orange':'Hold_Orange','yellow':'Hold_Yellow','green':'Hold_Green',
    'blue':'Hold_Blue','purple':'Hold_Purple','pink':'Hold_Pink','white':'Hold_White',
    'black':'Hold_Black','gray':'Hold_Gray','grey':'Hold_Gray','lime':'Hold_Lime','sky':'Hold_Sky',
}

# ===== 유틸 =====
def parse_colors_arg(colors_arg:str|None):
    """--colors 문자열을 파싱하여 허용 클래스명 리스트 반환. None 또는 all이면 None(=전체 허용)."""
    if not colors_arg:
        human = ", ".join(sorted(COLOR_ALIAS.keys()))
        print(f"🎨 선택 가능한 색상: {human}")
        s = input("✅ 원하는 홀드 색상 입력 (쉼표구분, 엔터=전체): ").strip()
        if not s:
            print("→ 전체 클래스 허용"); return None
        colors_arg = s
    s = colors_arg.strip().lower()
    if s in {"all","*","전체","모두"}:
        print("→ 전체 클래스 허용"); return None
    allow = []
    for tok in [t.strip() for t in s.split(',') if t.strip()]:
        mapped = COLOR_ALIAS.get(tok)
        if mapped is None:
            print(f"⚠️ 색상 '{tok}' 인식 실패 → 무시")
        else:
            allow.append(mapped)
    allow = sorted(set(allow))
    if not allow:
        print("→ 유효한 색상이 없어 전체 허용으로 전환"); return None
    print("🎯 허용 클래스:", ", ".join(allow))
    return allow


def load_stereo(npz_path:Path):
    S = np.load(str(npz_path), allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    M  = np.array([0.5*Tx, 0.0, 0.0], dtype=np.float64)  # 정보용
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H), B, M)


def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("카메라를 열 수 없습니다. 인덱스/연결 확인.")
    return cap1, cap2


def rectify(frame, mx, my, size):
    W, H = size
    if (frame.shape[1], frame.shape[0]) != (W, H):
        frame = cv2.resize(frame, (W, H))
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)


def extract_holds_with_indices(frame_bgr, model, allow_classes:list[str]|None, mask_thresh=0.7, row_tol=50):
    h, w = frame_bgr.shape[:2]
    res = model(frame_bgr)[0]
    holds = []
    if res.masks is None: return []
    masks = res.masks.data; boxes = res.boxes; names = model.names
    for i in range(masks.shape[0]):
        cls_id = int(boxes.cls[i].item()); class_name = names[cls_id]
        if (allow_classes is not None) and (class_name not in allow_classes):
            continue
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"]) 
        holds.append({
            "class_name": class_name,
            "color": COLOR_MAP.get(class_name,(255,255,255)),
            "contour": contour, "center": (cx, cy),
            "conf": float(boxes.conf[i].item()),
        })
    if not holds: return []
    enriched = [{"cx": h_["center"][0], "cy": h_["center"][1], **h_} for h_ in holds]
    enriched.sort(key=lambda h: h["cy"])  # y 정렬
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol: cur.append(h_)
        else: rows.append(cur); cur = [h_]
    rows.append(cur)
    final_sorted = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])  # x 정렬
        final_sorted.extend(row)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted


def merge_holds_by_center(holds_lists, merge_dist_px=18):
    merged = []
    for holds in holds_lists:
        for h in holds:
            h = {k: v for k, v in h.items()}
            h.pop("hold_index", None)
            assigned = False
            for m in merged:
                dx = h["center"][0] - m["center"][0]
                dy = h["center"][1] - m["center"][1]
                if (dx*dx + dy*dy) ** 0.5 <= merge_dist_px:
                    area_h = cv2.contourArea(h["contour"])
                    area_m = cv2.contourArea(m["contour"])
                    if (area_h > area_m) or (abs(area_h - area_m) < 1e-6 and h.get("conf",0) > m.get("conf",0)):
                        m.update(h)
                    assigned = True
                    break
            if not assigned:
                merged.append(h)
    return merged


def assign_indices(holds, row_tol=50):
    if not holds: return []
    enriched = [{"cx": h["center"][0], "cy": h["center"][1], **h} for h in holds]
    enriched.sort(key=lambda h: h["cy"])  # y 정렬
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol: cur.append(h_)
        else: rows.append(cur); cur = [h_]
    rows.append(cur)
    final_sorted = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])  # x 정렬
        final_sorted.extend(row)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted


def triangulate_xy(P1, P2, ptL, ptR):
    xl = np.array(ptL, dtype=np.float64).reshape(2,1)
    xr = np.array(ptR, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, xl, xr)
    X  = (Xh[:3] / Xh[3]).reshape(3)
    return X


def part_name_from_idx(landmark_idx:int)->str:
    if landmark_idx in (15,16):
        return "hand"
    else:
        return "foot"


# ===== 메인 =====

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",   default=str(NPZ_PATH))
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--colors", default=None, help="쉼표구분 색상(예: orange,blue) / all | 전체 | *")
    ap.add_argument("--parts",  default="both", choices=["hands","feet","both"], help="접촉 판단에 사용할 신체 부위")
    ap.add_argument("--label",  default=None, help="파일명 라벨(선택)")
    ap.add_argument("--confirm_frames", type=int, default=4, help="자동 저장을 위한 연속 프레임 수(스침 방지 디바운스)")
    ap.add_argument("--no_timestamp", action="store_true", help="CSV에 timestamp 필드를 저장하지 않음")
    ap.add_argument("--left_cam", type=int, default=None, help="왼쪽 카메라 인덱스(기본 1)")
    ap.add_argument("--right_cam", type=int, default=None, help="오른쪽 카메라 인덱스(기본 2)")
    ap.add_argument("--show_init", action="store_true", help="초기 10프레임 YOLO 단계에서도 미리보기 표시")
    args = ap.parse_args()

    # 지연 로드
    from ultralytics import YOLO
    import mediapipe as mp

    # ------- 스테레오/모델 로드 -------
    npz_path = Path(args.npz); model_path = Path(args.model)
    if not npz_path.exists():
        raise FileNotFoundError(f"스테레오 파라미터 파일 없음: {npz_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO 모델 파일 없음: {model_path}")

    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(npz_path)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

    # 색상 설정
    allow_classes = parse_colors_arg(args.colors)  # None이면 전체 허용

    # 카메라 열기
    # 카메라 인덱스 확정
    left_idx  = args.left_cam  if args.left_cam  is not None else CAM1_INDEX
    right_idx = args.right_cam if args.right_cam is not None else CAM2_INDEX
    capL_idx, capR_idx = (right_idx, left_idx) if SWAP_INPUT else (left_idx, right_idx)
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)

    # YOLO 로드
    model = YOLO(str(model_path))

    # ------- 초기 10프레임: 홀드 검출 병합 -------
    print(f"[Init] First 10 frames: YOLO seg & merge ...")
    L_sets, R_sets = [], []
    for _ in range(2):
        cap1.read(); cap2.read()
    for k in range(10):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            raise SystemExit("초기 프레임 캡처 실패")
        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)
        L_sets.append(extract_holds_with_indices(Lr_k, model, allow_classes, THRESH_MASK, ROW_TOL_Y))
        R_sets.append(extract_holds_with_indices(Rr_k, model, allow_classes, THRESH_MASK, ROW_TOL_Y))
        if args.show_init:
            vis_init = np.hstack([Rr_k, Lr_k]) if SWAP_DISPLAY else np.hstack([Lr_k, Rr_k])
            cv2.putText(vis_init, f"Initializing YOLO: {k+1}/10", (20, 40), 0, 1.0, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis_init, f"Initializing YOLO: {k+1}/10", (20, 40), 0, 1.0, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, vis_init)
            cv2.waitKey(1)
        print(f"  - frame {k+1}/10: L={len(L_sets[-1])}  R={len(R_sets[-1])}")

    holdsL = assign_indices(merge_holds_by_center(L_sets, 18), ROW_TOL_Y)
    holdsR = assign_indices(merge_holds_by_center(R_sets, 18), ROW_TOL_Y)
    if not holdsL or not holdsR:
        raise SystemExit("[Warn] 한쪽 또는 양쪽에서 홀드 검출 실패")

    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        raise SystemExit("[Warn] 좌/우 공통 hold_index가 없습니다.")

    # ------- MediaPipe Pose -------
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # 부위별 사용 랜드마크 인덱스
    LM = {
        "hands": [15, 16],               # 손목 (left_wrist, right_wrist)
        "feet":  [31, 32, 27, 28],       # 발끝(index toe), 발목
    }
    if args.parts == "hands":
        lm_used = LM["hands"]
    elif args.parts == "feet":
        lm_used = LM["feet"]
    else:
        lm_used = LM["hands"] + LM["feet"]

    # ------- 렌더/입력 루프 -------
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    route = []  # list[dict]

    # 접촉 누적(연속 프레임 카운트)
    contact = {hid: {"inside": False, "frames": 0, "part": None} for hid in common_ids}

    def save_csv(path:Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["seq","hold_id","cxL","cyL","cxR","cyR","X_mm","Y_mm","Z_mm","contact_part"]
        if not args.no_timestamp:
            fieldnames.append("timestamp")
        with open(path, "w", newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(route)
        print(f"[Save] {path}  (rows={len(route)})")

    def append_step(hid:int, trigger_part:str):
        Lh = idxL.get(hid); Rh = idxR.get(hid)
        if not (Lh and Rh):
            print(f"[Skip] ID{hid} 좌/우 매칭 불가"); return
        X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
        row = dict(
            seq=len(route), hold_id=int(hid),
            cxL=Lh["center"][0], cyL=Lh["center"][1],
            cxR=Rh["center"][0], cyR=Rh["center"][1],
            X_mm=float(X[0]), Y_mm=float(X[1]), Z_mm=float(X[2]),
            contact_part=trigger_part,
        )
        if not args.no_timestamp:
            row["timestamp"] = time.time()
        route.append(row)
        print(f"[Route] + seq {row['seq']}  ID{hid} ({trigger_part})  X=({row['X_mm']:.1f},{row['Y_mm']:.1f},{row['Z_mm']:.1f})")

    # 기본 출력 경로
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"route_{ts}{('_'+args.label) if args.label else ''}.csv"
    out_path = ROUTES_DIR / fname

    while True:
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            print("[Warn] 프레임 읽기 실패"); break
        Lr = rectify(f1, map1x, map1y, size)
        Rr = rectify(f2, map2x, map2y, size)
        vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])

        # 홀드 오버레이
        for side, holds in (("L", holdsL), ("R", holdsR)):
            xoff = (W if SWAP_DISPLAY else 0) if side=="L" else (0 if SWAP_DISPLAY else W)
            for h in holds:
                cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)
                cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                cx, cy = h["center"]
                cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                tag = f"ID:{h['hold_index']}"
                cv2.putText(vis, tag, (cx+xoff-10, cy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, (cx+xoff-10, cy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

        # MediaPipe (왼쪽 프레임 기준)
        result = pose.process(cv2.cvtColor(Lr, cv2.COLOR_BGR2RGB))
        coordsL = {}
        if result.pose_landmarks:
            hL, wL = Lr.shape[:2]
            for idx in lm_used:
                lm = result.pose_landmarks.landmark[idx]
                x, y = float(lm.x*wL), float(lm.y*hL)
                coordsL[idx] = (x, y)
                # draw
                left_xoff = (W if SWAP_DISPLAY else 0)
                cv2.circle(vis, (int(x)+left_xoff, int(y)), 6, (0,0,255), -1)

        # 접촉 프레임 카운트 갱신
        if coordsL:
            for hid in common_ids:
                inside_any = False
                used_part = None
                if hid in idxL:
                    cnt = idxL[hid]["contour"]
                    for lm_idx, xy in coordsL.items():
                        if cv2.pointPolygonTest(cnt, xy, False) >= 0:
                            inside_any = True
                            used_part = part_name_from_idx(lm_idx)
                            break
                c = contact[hid]
                if inside_any:
                    c["inside"] = True
                    c["frames"] += 1
                    c["part"] = used_part or c["part"]
                    if c["frames"] >= args.confirm_frames:
                        # 마지막 추가와 동일 좌표 중복 방지
                        if (not route) or (abs(route[-1]["cxL"] - idxL[hid]["center"][0]) > 1 or abs(route[-1]["cyL"] - idxL[hid]["center"][1]) > 1):
                            append_step(hid, trigger_part=c["part"] or "unknown")
                        # 떠날 때까지 재기록 방지
                        c["frames"] = 10**9
                else:
                    c["inside"] = False
                    c["frames"] = 0
                    c["part"] = None

        # HUD
        cv2.putText(vis, f"Route len: {len(route)}   (M:add  Z:undo  R:reset  S:save  Q:quit)   parts={args.parts}  confirm_frames={args.confirm_frames}", (20, 28), 0, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"Route len: {len(route)}   (M:add  Z:undo  R:reset  S:save  Q:quit)   parts={args.parts}  confirm_frames={args.confirm_frames}", (20, 28), 0, 0.7, (0,255,255), 1, cv2.LINE_AA)
        
        cv2.imshow(WINDOW_NAME, vis)
        
        # 키 입력
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            candidates = [(hid, c["frames"]) for hid,c in contact.items() if c["inside"]]
            if not candidates:
                print("[Info] 접촉 중인 홀드를 찾지 못했습니다.")
            else:
                candidates.sort(key=lambda t: -t[1])
                hid = candidates[0][0]
                c = contact[hid]
                append_step(hid, trigger_part=c["part"] or "manual")
        elif k == ord('z') and route:
            removed = route.pop()
            print(f"[Undo] seq {removed['seq']} 제거 (ID{removed['hold_id']})")
        elif k == ord('r'):
            route.clear(); print("[Reset] route cleared")
        elif k == ord('s'):
            save_csv(out_path)

    # 종료 처리
    cap1.release(); cap2.release(); cv2.destroyAllWindows()
    # 자동 저장
    if route:
        save_csv(out_path)


if __name__ == "__main__":
    main()
