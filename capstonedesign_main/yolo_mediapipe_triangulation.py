#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo YOLOv8n-Seg (first 10 frames merged) + MediaPipe-on-Left → Live overlay (mm)
+ Laser-origin yaw/pitch per hold (LEFT-camera-based) + auto-target ID0 (absolute command)

- 시작 시 첫 10프레임에서 YOLO 세그 → 프레임 간 중복 병합 → y행/x정렬로 hold_index 부여
- 좌/우 공통 hold_index 쌍만 삼각측량 → X(mm), |X−L|, d_line, yaw/pitch(레이저 원점=LEFT기준) 계산
- 시작 시: ID 0(없으면 가장 작은 ID) 자동 선택 → 명령 각 산출(보정 반영) → (옵션) 시리얼 송신
- 라이브: 레티파이 프레임을 화면에 표시할 때 좌/우 스왑 옵션(SWAP_DISPLAY)으로 UI 정렬
- 저장: grip_records.csv 만 저장
"""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp
import csv
import math

# ========= 사용자 설정 =========
NPZ_PATH       = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\calib_out\old_camera_same\stereo\stereo_params_scaled.npz"
MODEL_PATH     = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\best_6.pt"

CAM1_INDEX     = 1   # 물리 카메라 인덱스(왼쪽)
CAM2_INDEX     = 2   # 물리 카메라 인덱스(오른쪽)

# 입력(캡처) 좌/우가 보정(P1/P2)과 뒤집혔다면 True로 (정석 해결)
SWAP_INPUT     = False

# 화면(UI)만 좌/우 바꿔서 표시할지 (오버레이/텍스트 오프셋 자동 정합)
SWAP_DISPLAY   = False

WINDOW_NAME    = "Rectified L | R  (10f merged, LEFT-origin O; MP Left, Auto-ID0)"
SHOW_GRID      = False
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30
SELECTED_COLOR = None    # 예: 'orange' (None이면 콘솔 입력/엔터=전체)

SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"

CSV_GRIPS_PATH  = "grip_records.csv"   # ✅ 그립 기록만 저장
TOUCH_THRESHOLD = 10                   # 연속 프레임 수(>= 이면 채색)

# ---- 레이저 원점(=조준 기준점) 오프셋 (LEFT 카메라 원점 기준, cm) ----
# 실측: 왼쪽 카메라 중심 기준 왼쪽 1.85cm, 위 8cm, 카메라보다 3.3cm 뒤
LASER_OFFSET_CM_LEFT = 1.85   # '왼쪽'은 x 음(-) 처리
LASER_OFFSET_CM_UP   = 8.0    # '위쪽'은 y 음(-) 처리
LASER_OFFSET_CM_FWD  = -3.3   # 전방 +, 뒤쪽 - → 뒤 3.3cm 이므로 -3.3
Y_UP_IS_NEGATIVE = True       # 위가 -y

# ---- “ID0로 조준”을 실제로 보낼지 옵션 ----
SEND_SERIAL      = False           # True로 바꾸면 시리얼 송신
SERIAL_PORT      = "COM6"          # 보드 포트
SERIAL_BAUD      = 115200

# 간단 오프셋 보정(현장 튜닝)
YAW_OFFSET_DEG   = 0.0
PITCH_OFFSET_DEG = 0.0

# (선택) 2x2 선형 보정 모델 사용 여부
USE_LINEAR_CAL = False
A11, A12, B1 = 1.0, 0.0, 0.0    # yaw_cmd = A11*yaw_est + A12*pitch_est + B1
A21, A22, B2 = 0.0, 1.0, 0.0    # pitch_cmd = A21*yaw_est + A22*pitch_est + B2

# (선택) 서보 각→PWM(us) 맵 — 하드웨어에 맞게 수정
SERVO = {
    "YAW_MIN_DEG":   -90.0, "YAW_MAX_DEG":   90.0, "YAW_MIN_US": 1000, "YAW_MAX_US": 2000,
    "PITCH_MIN_DEG": -45.0, "PITCH_MAX_DEG": 45.0, "PITCH_MIN_US":1000, "PITCH_MAX_US":2000,
}

# (선택) 프리뷰 최대 폭
PREVIEW_MAX_W = None  # 예: 1280

# ==== 초기 YOLO 프레임 수 & 병합 기준 ====
INIT_DET_FRAMES   = 10          # ✅ 첫 10프레임 사용
CENTER_MERGE_PX   = 18          # ✅ 프레임 간 동일 홀드로 간주할 중심거리(px)
# ==============================

# YOLO 클래스 컬러 (BGR)
COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}
ALL_COLORS = {
    'red':'Hold_Red','orange':'Hold_Orange','yellow':'Hold_Yellow','green':'Hold_Green',
    'blue':'Hold_Blue','purple':'Hold_Purple','pink':'Hold_Pink','white':'Hold_White',
    'black':'Hold_Black','gray':'Hold_Gray','lime':'Hold_Lime','sky':'Hold_Sky',
}

# ---------- 유틸 ----------
def ask_color_and_map_to_class(all_colors_dict):
    print("🎨 선택 가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("✅ 원하는 홀드 색상 입력(엔터=전체): ").strip().lower()
    if not s:
        print("→ 전체 클래스 사용"); return None
    mapped = all_colors_dict.get(s)
    if mapped is None:
        print(f"⚠️ '{s}' 는 유효하지 않은 색상입니다. 전체 클래스 사용")
        return None
    print(f"🎯 선택된 클래스: {mapped}")
    return mapped

def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    M  = np.array([0.5*Tx, 0.0, 0.0], dtype=np.float64)  # 중점(정보용)
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

def extract_holds_with_indices(frame_bgr, model, selected_class_name=None,
                               mask_thresh=0.7, row_tol=50):
    h, w = frame_bgr.shape[:2]
    res = model(frame_bgr)[0]
    holds = []
    if res.masks is None: return []
    masks = res.masks.data; boxes = res.boxes; names = model.names
    print(f"[dbg] masks={tuple(res.masks.data.shape)} | frame={(h,w)}")
    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea)
        cls_id = int(boxes.cls[i].item()); conf = float(boxes.conf[i].item())
        class_name = names[cls_id]
        if (selected_class_name is not None) and (class_name != selected_class_name):
            continue
        Mom = cv2.moments(contour)
        if Mom["m00"] == 0: continue
        cx = int(Mom["m10"]/Mom["m00"]); cy = int(Mom["m01"]/Mom["m00"])
        holds.append({"class_name": class_name, "color": COLOR_MAP.get(class_name,(255,255,255)),
                      "contour": contour, "center": (cx, cy), "conf": conf})
    if not holds: return []
    enriched = [{"cx": h_["center"][0], "cy": h_["center"][1], **h_} for h_ in holds]
    enriched.sort(key=lambda h: h["cy"])
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol: cur.append(h_)
        else: rows.append(cur); cur = [h_]
    rows.append(cur)
    final_sorted = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])
        final_sorted.extend(row)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted

def merge_holds_by_center(holds_lists, merge_dist_px=18):
    merged = []
    for holds in holds_lists:
        for h in holds:
            h = {k: v for k, v in h.items()}  # shallow copy
            h.pop("hold_index", None)         # 인덱스는 최종에 재부여
            assigned = False
            for m in merged:
                dx = h["center"][0] - m["center"][0]
                dy = h["center"][1] - m["center"][1]
                if (dx*dx + dy*dy) ** 0.5 <= merge_dist_px:
                    # 대표 갱신 기준: 면적 우선, 비슷하면 conf 큰 것
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
    if not holds:
        return []
    enriched = [{"cx": h["center"][0], "cy": h["center"][1], **h} for h in holds]
    enriched.sort(key=lambda h: h["cy"])
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol: cur.append(h_)
        else: rows.append(cur); cur = [h_]
    rows.append(cur)
    final_sorted = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])
        final_sorted.extend(row)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted

def triangulate_xy(P1, P2, ptL, ptR):
    xl = np.array(ptL, dtype=np.float64).reshape(2,1)
    xr = np.array(ptR, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, xl, xr)
    X  = (Xh[:3] / Xh[3]).reshape(3)  # [X,Y,Z] (mm)
    return X

def draw_grid(img):
    h, w = img.shape[:2]; step = max(20, h//20)
    for y in range(0, h, step):
        cv2.line(img, (0,y), (w-1,y), (0,255,0), 1, cv2.LINE_AA)

def yaw_pitch_from_X(X, O, y_up_is_negative=True):
    v = X - O
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw   = np.degrees(np.arctan2(vx, vz))
    pitch = np.degrees(np.arctan2((-vy if y_up_is_negative else vy), np.hypot(vx, vz)))
    return yaw, pitch

def angle_between(v1, v2):
    a = np.linalg.norm(v1); b = np.linalg.norm(v2)
    if a == 0 or b == 0: return 0.0
    cosang = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def wrap_deg(d): return (d + 180.0) % 360.0 - 180.0

def imshow_scaled(win, img, maxw=None):
    if not maxw: cv2.imshow(win, img); return
    h, w = img.shape[:2]
    if w > maxw:
        s = maxw / w
        img = cv2.resize(img, (int(w*s), int(h*s)))
    cv2.imshow(win, img)

def deg_to_us(angle, min_deg, max_deg, min_us, max_us):
    angle = float(np.clip(angle, min_deg, max_deg))
    return int(np.interp(angle, [min_deg, max_deg], [min_us, max_us]))

def xoff_for(side, W, swap):
    # side: "L" 또는 "R" (왼쪽 카메라/오른쪽 카메라 프레임)
    if side == "L":
        return (W if swap else 0)
    else:
        return (0 if swap else W)

# ---------- 메인 ----------
def main():
    # 경로 검사
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일이 없습니다: {p}")

    # 준비
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

    # 레이저 원점 O = LEFT 카메라 원점 L + (왼 1.85cm, 위 8cm, 뒤 3.3cm)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 왼쪽 카메라가 원점
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = L + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, LEFT-based) = {O}")

    # 색상 필터 선택
    if SELECTED_COLOR is not None:
        sc = SELECTED_COLOR.strip().lower()
        selected_class_name = ALL_COLORS.get(sc)
        if selected_class_name is None:
            print(f"[Filter] SELECTED_COLOR='{SELECTED_COLOR}' 인식 실패. 콘솔에서 선택합니다.")
            selected_class_name = ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 선택 클래스(상수): {selected_class_name}")
    else:
        selected_class_name = ask_color_and_map_to_class(ALL_COLORS)

    # 카메라 & 모델
    capL_idx, capR_idx = CAM1_INDEX, CAM2_INDEX
    if SWAP_INPUT:
        capL_idx, capR_idx = capR_idx, capL_idx  # 입력을 스왑하여 보정 좌/우와 일치
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)
    model = YOLO(str(MODEL_PATH))

    # ====== 초기 10프레임 수집 & YOLO → 병합 ======
    print(f"[Init] First {INIT_DET_FRAMES} frames: YOLO seg & merge ...")
    L_sets, R_sets = [], []
    # 워밍업 (옵션)
    for _ in range(2):
        cap1.read(); cap2.read()

    for k in range(INIT_DET_FRAMES):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡처 실패")
        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)
        holdsL_k = extract_holds_with_indices(Lr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        holdsR_k = extract_holds_with_indices(Rr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        L_sets.append(holdsL_k); R_sets.append(holdsR_k)
        print(f"  - frame {k+1}/{INIT_DET_FRAMES}: L={len(holdsL_k)}  R={len(holdsR_k)}")

    # 프레임 간 중복 병합 후 최종 인덱스 재부여
    holdsL = assign_indices(merge_holds_by_center(L_sets, CENTER_MERGE_PX), ROW_TOL_Y)
    holdsR = assign_indices(merge_holds_by_center(R_sets, CENTER_MERGE_PX), ROW_TOL_Y)

    if not holdsL or not holdsR:
        cap1.release(); cap2.release()
        print("[Warn] 한쪽 또는 양쪽에서 홀드가 검출되지 않았습니다.")
        return

    # index → hold 맵 & 공통 ID
    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        print("[Warn] 좌/우 공통 hold_index가 없습니다.")
    else:
        print(f"[Info] 매칭된 홀드 쌍 수: {len(common_ids)}")

    # 매칭 결과 사전 계산(3D, 거리, 각도) — LEFT 원점 기반
    matched_results = []
    for hid in common_ids:
        Lh = idxL[hid]; Rh = idxR[hid]
        X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
        d_left  = float(np.linalg.norm(X - L))            # LEFT 기준 거리
        d_line  = float(np.hypot(X[1], X[2]))
        yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
        matched_results.append({
            "hid": hid,
            "Lcx": Lh["center"][0], "Lcy": Lh["center"][1],
            "Rcx": Rh["center"][0], "Rcy": Rh["center"][1],
            "color": Lh["color"],
            "X": X, "d_left": d_left, "d_line": d_line,
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
        })

    # 연속 인덱스 각도차 (정보용)
    by_id = {mr["hid"]: mr for mr in matched_results}
    max_id = max(by_id) if by_id else -1
    angle_deltas = []
    for i in range(max_id):
        if (i in by_id) and (i+1 in by_id):
            a = by_id[i]; b = by_id[i+1]
            dyaw   = wrap_deg(b["yaw_deg"]   - a["yaw_deg"])
            dpitch = wrap_deg(b["pitch_deg"] - a["pitch_deg"])
            v1 = a["X"] - O; v2 = b["X"] - O
            d3d = angle_between(v1, v2)
            angle_deltas.append((i, i+1, dyaw, dpitch, d3d))

    print("\n[ΔAngles] (i -> i+1):  Δyaw(deg), Δpitch(deg), 3D_angle(deg)")
    for i, j, dyaw, dpitch, d3d in angle_deltas:
        print(f"  {i:>2}→{j:<2} :  {dyaw:+6.2f}°, {dpitch:+6.2f}°, {d3d:6.2f}°")

    # ====== ⬇️ 여기서 '시작하면 0번 인덱스로 조준' 처리됨 ⬇️ ======
    target_id = 0 if 0 in by_id else (min(by_id.keys()) if by_id else None)
    first_target = by_id.get(target_id) if target_id is not None else None

    yaw_cmd = pitch_cmd = None
    if first_target:
        yaw_est   = first_target["yaw_deg"]
        pitch_est = first_target["pitch_deg"]

        if USE_LINEAR_CAL:
            yaw_cmd   = A11*yaw_est + A12*pitch_est + B1
            pitch_cmd = A21*yaw_est + A22*pitch_est + B2
        else:
            yaw_cmd   = yaw_est   + YAW_OFFSET_DEG
            pitch_cmd = pitch_est + PITCH_OFFSET_DEG

        print(f"\n[FIRST TARGET] ID{first_target['hid']}: "
              f"yaw_est={yaw_est:.2f}°, pitch_est={pitch_est:.2f}°  "
              f"-> cmd=({yaw_cmd:.2f}°, {pitch_cmd:.2f}°)")

        if SEND_SERIAL:
            try:
                import serial, time as _t
                yaw_us   = deg_to_us(yaw_cmd,   SERVO['YAW_MIN_DEG'],   SERVO['YAW_MAX_DEG'],
                                               SERVO['YAW_MIN_US'],    SERVO['YAW_MAX_US'])
                pitch_us = deg_to_us(pitch_cmd, SERVO['PITCH_MIN_DEG'], SERVO['PITCH_MAX_DEG'],
                                               SERVO['PITCH_MIN_US'],  SERVO['PITCH_MAX_US'])
                ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
                _t.sleep(0.5)
                ser.write(f"Y{yaw_us}\n".encode())
                ser.write(f"P{pitch_us}\n".encode())
                ser.close()
                print(f"[Serial] Sent: Y={yaw_us}us, P={pitch_us}us")
            except Exception as e:
                print(f"[Serial ERROR] {e}")
    else:
        print("[FIRST TARGET] 선택할 수 있는 타겟이 없습니다.")
    # ====== ⬆️ 여기까지가 '자동 ID0 조준' 로직 ⬆️ ======

    # ==== MediaPipe Pose (왼쪽 카메라 전용) ====
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    important_landmarks = {
        "left_index": 15, "right_index": 16, "left_heel": 29,
        "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32,
    }
    hand_parts = {"left_index", "right_index"}

    # 터치 기록 상태
    grip_records = []     # [part, hold_id, cx, cy]
    already_grabbed = {}  # key=(name, hold_index) → True
    touch_counters = {}   # key=(name, hold_index) → 연속 프레임 카운트

    # 비디오 저장
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))

    # 라이브 루프
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time(); frame_idx = 0

    while True:
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            print("[Warn] 프레임 읽기 실패"); break

        Lr = rectify(f1, map1x, map1y, size)
        Rr = rectify(f2, map2x, map2y, size)

        # 화면 결합(표시만 스왑 옵션)
        vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])

        if SHOW_GRID:
            draw_grid(vis[:, :W]); draw_grid(vis[:, W:])

        # 병합된 10프레임 결과(holdsL/holdsR)를 계속 그림
        for side, holds in (("L", holdsL), ("R", holdsR)):
            xoff = xoff_for(side, W, SWAP_DISPLAY)
            for h in holds:
                cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)
                cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                cx, cy = h["center"]
                cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                cv2.putText(vis, f"ID:{h['hold_index']}", (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, f"ID:{h['hold_index']}", (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

        # 3D/각도 텍스트 + FIRST 표시
        y0 = 30
        for mr in matched_results:
            X = mr["X"]
            base = (f"ID{mr['hid']}  X=({X[0]:.1f},{X[1]:.1f},{X[2]:.1f})mm  "
                    f"|X-L|={mr['d_left']:.1f}  d_line={mr['d_line']:.1f}  "
                    f"yaw={mr['yaw_deg']:.1f}°  pitch={mr['pitch_deg']:.1f}°")
            txt = "[FIRST] " + base if (first_target and mr["hid"] == first_target["hid"]) else base
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
            y0 += 22

        # 연속 인덱스 각도차(상위 5줄)
        y1 = y0 + 8
        for k in range(min(5, len(angle_deltas))):
            i, j, dyaw, dpitch, d3d = angle_deltas[k]
            t2 = f"Δ({i}->{j}): yaw={dyaw:+.1f}°, pitch={dpitch:+.1f}°, 3D={d3d:.1f}°"
            cv2.putText(vis, t2, (20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, t2, (20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
            y1 += 22

        # MediaPipe Pose: 왼쪽만 (객체 재사용)
        image_rgb = cv2.cvtColor(Lr, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        pose_landmarks = result.pose_landmarks
        if pose_landmarks:
            hL, wL = Lr.shape[:2]
            coords = {}
            for name, idx in important_landmarks.items():
                lm = pose_landmarks.landmark[idx]
                coords[name] = (lm.x * wL, lm.y * hL)
            left_xoff = xoff_for("L", W, SWAP_DISPLAY)
            for name, (x, y) in coords.items():
                joint_color = (0, 0, 255) if name in hand_parts else (0, 255, 0)
                cv2.circle(vis, (int(x)+left_xoff, int(y)), 5, joint_color, -1)
                cv2.putText(vis, f"{name}: ({int(x)}, {int(y)})",
                            (int(x)+left_xoff+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
            for name, (x, y) in coords.items():
                for i, hold in enumerate(holdsL):
                    inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0
                    key = (name, i)
                    if inside:
                        touch_counters[key] = touch_counters.get(key, 0) + 1
                        if touch_counters[key] >= TOUCH_THRESHOLD:
                            cnt_shifted = hold["contour"] + np.array([[[left_xoff, 0]]], dtype=hold["contour"].dtype)
                            cv2.drawContours(vis, [cnt_shifted], -1, hold["color"], thickness=cv2.FILLED)
                            if not already_grabbed.get(key):
                                cx, cy = hold["center"]
                                grip_records.append([name, i, cx, cy])
                                already_grabbed[key] = True
                    else:
                        touch_counters[key] = 0

        # FPS & 출력
        t_now = time.time(); fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
        cv2.putText(vis, f"FPS: {fps:.1f}  (YOLO first-10 merged; MP left, LEFT-origin)", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f}  (YOLO first-10 merged; MP left, LEFT-origin)", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

        imshow_scaled(WINDOW_NAME, vis, PREVIEW_MAX_W)
        if SAVE_VIDEO:
            out.write(vis)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 정리
    cap1.release(); cap2.release()
    if SAVE_VIDEO:
        out.release(); print(f"[Info] 저장 완료: {OUT_PATH}")
    cv2.destroyAllWindows()

    # ✅ 그립 기록만 저장
    with open(CSV_GRIPS_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["part", "hold_id", "cx", "cy"])
        writer.writerows(grip_records)
    print(f"[Info] 그립 CSV: {CSV_GRIPS_PATH} (행 수: {len(grip_records)})")

if __name__ == "__main__":
    main()
