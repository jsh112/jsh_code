#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo YOLOv8n-Seg (first 10 frames merged) + MediaPipe-on-Left → Live overlay (mm)
+ Laser-origin yaw/pitch per hold (LEFT-camera-based)
+ ✅ DualServoController 연동 + Δ테이블(dyaw, dpitch) 기반 상대 이동:
    - 시작 각도(--pitch/--yaw)를 '베이스'로 설정
    - 다음 홀드로 갈 때:  yaw_next   = cur_yaw   - dyaw
                        pitch_next = cur_pitch + dpitch
    - Mediapipe 손이 '현재 타깃 홀드'에 10프레임 이상 들어오면 자동으로 다음 홀드로 이동
    - 수동 넘김: 실행 중 'n'
    - 레이저 on/off 옵션 지원
"""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp
import csv
import math
import argparse

# ========= 사용자 설정 (네 환경 경로 그대로 둠) =========
NPZ_PATH       = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\calib_out\old_camera_same\stereo\stereo_params_scaled.npz"
MODEL_PATH     = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\best_6.pt"

CAM1_INDEX     = 1   # 왼쪽 카메라
CAM2_INDEX     = 2   # 오른쪽 카메라

SWAP_INPUT     = False   # 입력 좌/우가 보정과 뒤집혔으면 True
SWAP_DISPLAY   = False   # 화면 표시만 좌/우 바꿈

WINDOW_NAME    = "Rectified L | R  (10f merged; MP Left; Δ-Relative Servo)"
SHOW_GRID      = False
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30
SELECTED_COLOR = None    # 예: 'orange' (None=전체)

# 자동 진행 파라미터
TOUCH_THRESHOLD = 10     # 손가락 in-polygon 연속 프레임 수
ADV_COOLDOWN    = 0.5    # 중복 트리거 방지 (sec)

# 저장 옵션
SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"
CSV_GRIPS_PATH = "grip_records.csv"

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 1.85
LASER_OFFSET_CM_UP   = 8.0
LASER_OFFSET_CM_FWD  = -3.3
Y_UP_IS_NEGATIVE     = True  # 위가 -y

# 간단 오프셋/선형 보정 (필요시 사용)
YAW_OFFSET_DEG   = 0.0
PITCH_OFFSET_DEG = 0.0
USE_LINEAR_CAL   = False
A11, A12, B1     = 1.0, 0.0, 0.0
A21, A22, B2     = 0.0, 1.0, 0.0

# ======== Servo controller import (stub fallback) ========
try:
    from servo_control import DualServoController
    HAS_SERVO = True
except Exception:
    HAS_SERVO = False
    class DualServoController:
        def __init__(self, *a, **k): print("[Servo] (stub) controller unavailable")
        def set_angles(self, pitch=None, yaw=None): print(f"[Servo] (stub) set_angles: P={pitch}, Y={yaw}")
        def center(self): print("[Servo] (stub) center")
        def query(self): print("[Servo] (stub) query"); return ""
        def laser_on(self): print("[Servo] (stub) laser_on")
        def laser_off(self): print("[Servo] (stub) laser_off")
        def close(self): pass

# ==== 유틸/수학 ====
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
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
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

def xoff_for(side, W, swap):
    return (W if swap else 0) if side=="L" else (0 if swap else W)

def apply_calibration(yaw_est, pitch_est):
    if USE_LINEAR_CAL:
        yaw_cmd   = A11*yaw_est + A12*pitch_est + B1
        pitch_cmd = A21*yaw_est + A22*pitch_est + B2
    else:
        yaw_cmd   = yaw_est   + YAW_OFFSET_DEG
        pitch_cmd = pitch_est + PITCH_OFFSET_DEG
    return yaw_cmd, pitch_cmd

def send_servo_angles(ctl, yaw_cmd, pitch_cmd):
    try:
        print(f"[Servo] send: yaw={yaw_cmd:.2f}°, pitch={pitch_cmd:.2f}°")
        ctl.set_angles(pitch_cmd, yaw_cmd)  # (pitch, yaw) 순서
    except Exception as e:
        print(f"[Servo ERROR] {e}")

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM7", help="서보 보드 포트 (예: COM4)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--center", action="store_true", help="시작 시 1회 센터 이동")
    ap.add_argument("--pitch", type=float, required=True, help="초기 pitch 각도 (베이스)")
    ap.add_argument("--yaw",   type=float, required=True, help="초기 yaw 각도 (베이스)")
    ap.add_argument("--laser_on",  action="store_true", help="시작 시 레이저 ON")
    ap.add_argument("--laser_off", action="store_true", help="시작 시 레이저 OFF")
    ap.add_argument("--no_auto_advance", action="store_true", help="손 인식 자동 넘김 비활성화")
    args = ap.parse_args()

    # 경로 검사
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일이 없습니다: {p}")

    # 스테레오 로드
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

    # 레이저 원점 O (LEFT 기준)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = L + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, LEFT-based) = {O}")

    # 색상 선택
    if SELECTED_COLOR is not None:
        sc = SELECTED_COLOR.strip().lower()
        selected_class_name = ALL_COLORS.get(sc)
        if selected_class_name is None:
            print(f"[Filter] SELECTED_COLOR='{SELECTED_COLOR}' 인식 실패. 콘솔에서 선택")
            selected_class_name = ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 선택 클래스(상수): {selected_class_name}")
    else:
        selected_class_name = ask_color_and_map_to_class(ALL_COLORS)

    # 카메라 & 모델
    capL_idx, capR_idx = CAM1_INDEX, CAM2_INDEX
    if SWAP_INPUT:
        capL_idx, capR_idx = capR_idx, capL_idx
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)
    model = YOLO(str(MODEL_PATH))

    # ====== 초기 10프레임 수집 & YOLO → 병합 ======
    print(f"[Init] First 10 frames: YOLO seg & merge ...")
    L_sets, R_sets = [], []
    for _ in range(2):
        cap1.read(); cap2.read()  # 워밍업

    for k in range(10):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡처 실패")
        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)
        holdsL_k = extract_holds_with_indices(Lr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        holdsR_k = extract_holds_with_indices(Rr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        L_sets.append(holdsL_k); R_sets.append(holdsR_k)
        print(f"  - frame {k+1}/10: L={len(holdsL_k)}  R={len(holdsR_k)}")

    holdsL = assign_indices(merge_holds_by_center(L_sets, 18), ROW_TOL_Y)
    holdsR = assign_indices(merge_holds_by_center(R_sets, 18), ROW_TOL_Y)
    if not holdsL or not holdsR:
        cap1.release(); cap2.release()
        print("[Warn] 한쪽 또는 양쪽에서 홀드가 검출되지 않았습니다.")
        return

    # 공통 ID
    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        print("[Warn] 좌/우 공통 hold_index가 없습니다.")
        return
    print(f"[Info] 매칭된 홀드 쌍 수: {len(common_ids)}")

    # 3D/각도 계산
    matched_results = []
    for hid in common_ids:
        Lh = idxL[hid]; Rh = idxR[hid]
        X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
        d_left  = float(np.linalg.norm(X - L))
        d_line  = float(np.hypot(X[1], X[2]))
        yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
        matched_results.append({
            "hid": hid, "color": Lh["color"],
            "X": X, "d_left": d_left, "d_line": d_line,
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
        })

    # ===== 연속 인덱스 각도차(정보용) & Δ맵 =====
    by_id  = {mr["hid"]: mr for mr in matched_results}
    max_id = max(by_id) if by_id else -1
    angle_deltas = []
    next_id_map  = {}   # i -> j
    delta_from_id = {}  # i -> (dyaw, dpitch)
    for i in range(max_id):
        if (i in by_id) and (i+1 in by_id):
            a = by_id[i]; b = by_id[i+1]
            dyaw   = wrap_deg(b["yaw_deg"]   - a["yaw_deg"])
            dpitch = wrap_deg(b["pitch_deg"] - a["pitch_deg"])
            v1 = a["X"] - O; v2 = b["X"] - O
            d3d = angle_between(v1, v2)
            angle_deltas.append((i, i+1, dyaw, dpitch, d3d))
            next_id_map[i]  = i+1
            delta_from_id[i] = (dyaw, dpitch)

    print("\n[ΔAngles] (i -> i+1):  Δyaw(deg), Δpitch(deg), 3D_angle(deg)")
    for i, j, dyaw, dpitch, d3d in angle_deltas:
        print(f"  {i:>2}→{j:<2} :  {dyaw:+6.2f}°, {dpitch:+6.2f}°, {d3d:6.2f}°")

    # ===== Servo 초기화 & 베이스 각도 설정 =====
    ctl = DualServoController(args.port, args.baud) if HAS_SERVO else DualServoController()
    auto_advance_enabled = (not args.no_auto_advance)

    cur_pitch = float(args.pitch)
    cur_yaw   = float(args.yaw)

    try:
        if args.center:
            print(ctl.center())
        if args.laser_on:
            ctl.laser_on()
        if args.laser_off:
            ctl.laser_off()
        # 시작 베이스 각도 1회 세팅
        ctl.set_angles(cur_pitch, cur_yaw)
        print(f"[Init Servo] base yaw={cur_yaw:.2f}°, pitch={cur_pitch:.2f}°")
    except Exception as e:
        print(f"[Servo Init ERROR] {e}")

    # ===== 타깃 진행 상태 =====
    current_target_id = 0 if 0 in by_id else (min(by_id.keys()) if by_id else None)
    last_advanced_time = 0.0

    # ==== MediaPipe Pose ====
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    important_landmarks = {"left_index": 15, "right_index": 16}
    hand_parts = set(important_landmarks.keys())

    # 터치 기록
    grip_records = []
    already_grabbed = {}
    touch_counters = {}

    # 비디오 저장
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))

    # 화면 루프
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()

    while True:
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            print("[Warn] 프레임 읽기 실패"); break

        Lr = rectify(f1, map1x, map1y, size)
        Rr = rectify(f2, map2x, map2y, size)
        vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])
        if SHOW_GRID:
            draw_grid(vis[:, :W]); draw_grid(vis[:, W:])

        # 병합된 홀드 오버레이
        for side, holds in (("L", holdsL), ("R", holdsR)):
            xoff = xoff_for(side, W, SWAP_DISPLAY)
            for h in holds:
                cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)
                cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                cx, cy = h["center"]
                cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                tag = f"ID:{h['hold_index']}"
                if (current_target_id is not None) and (h["hold_index"] == current_target_id):
                    tag = "[TARGET] " + tag
                cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)
        
        # --- 홀드별 3D 좌표/깊이 출력 ---
        y_info = 60   # 출력 시작 Y 좌표 (적당히 조절 가능)
        for mr in matched_results:
            X = mr["X"]
            depth = X[2]   # Z 값 (mm)
            txt3d = (f"ID{mr['hid']} : X=({X[0]:.1f}, {X[1]:.1f}, {X[2]:.1f}) mm "
                     f" | depth(Z)={depth:.1f} mm")
            cv2.putText(vis, txt3d, (20, y_info),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, txt3d, (20, y_info),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            y_info += 18

        # Δ→NEXT 및 현재각도 표시
        y0 = 28
        if current_target_id in by_id:
            if current_target_id in delta_from_id:
                dyaw, dpitch = delta_from_id[current_target_id]
                nxt = next_id_map[current_target_id]
                txt = (f"[Δ→NEXT] ID{current_target_id}→ID{nxt}  "
                       f"Δyaw={dyaw:+.1f}°, Δpitch={dpitch:+.1f}°  "
                       f"[now yaw={cur_yaw:.1f}°, pitch={cur_pitch:.1f}°]")
            else:
                mr = by_id[current_target_id]
                txt = (f"[LAST] ID{mr['hid']}  yaw={mr['yaw_deg']:.1f}°, pitch={mr['pitch_deg']:.1f}°  "
                       f"[now yaw={cur_yaw:.1f}°, pitch={cur_pitch:.1f}°]")
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1, cv2.LINE_AA)

        # MediaPipe 손 검출(왼쪽 프레임 기반 → 왼쪽 오프셋만 표시)
        result = pose.process(cv2.cvtColor(Lr, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks and (current_target_id in idxL):
            hL, wL = Lr.shape[:2]
            coords = {}
            for name, idx in important_landmarks.items():
                lm = result.pose_landmarks.landmark[idx]
                coords[name] = (lm.x * wL, lm.y * hL)

            left_xoff = xoff_for("L", W, SWAP_DISPLAY)
            for name, (x, y) in coords.items():
                joint_color = (0, 0, 255) if name in hand_parts else (0, 255, 0)
                cv2.circle(vis, (int(x)+left_xoff, int(y)), 6, joint_color, -1)
                cv2.putText(vis, f"{name}:({int(x)},{int(y)})",
                            (int(x)+left_xoff+6, int(y)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

            # 현재 타깃 홀드 polygon만 터치 판정
            hold = idxL[current_target_id]
            now = time.time()
            for name, (x, y) in coords.items():
                inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0
                key = (name, current_target_id)
                if inside:
                    touch_counters[key] = touch_counters.get(key, 0) + 1
                    if (auto_advance_enabled and
                        touch_counters[key] >= TOUCH_THRESHOLD and
                        now - last_advanced_time > ADV_COOLDOWN):

                        # 1) 그립 기록 1회
                        if not already_grabbed.get(key):
                            cx, cy = hold["center"]
                            grip_records.append([name, current_target_id, cx, cy])
                            already_grabbed[key] = True

                        # 2) 다음 홀드가 존재하면 Δ로 이동
                        if current_target_id in delta_from_id:
                            dyaw, dpitch = delta_from_id[current_target_id]
                            target_yaw   = cur_yaw   - dyaw     # yaw는 빼기
                            target_pitch = cur_pitch + dpitch   # pitch는 더하기
                            send_servo_angles(ctl, target_yaw, target_pitch)
                            cur_yaw, cur_pitch = target_yaw, target_pitch
                            nxt = next_id_map[current_target_id]
                            print(f"[Auto-Advance] {current_target_id}→{nxt}  (dyaw={dyaw:+.2f}, dpitch={dpitch:+.2f})")
                            current_target_id = nxt
                            last_advanced_time = now
                        else:
                            print("[Auto-Advance] 더 이상 다음 홀드가 없습니다.")
                else:
                    touch_counters[key] = 0

        # FPS
        t_now = time.time()
        fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
        cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                    (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                    (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

        imshow_scaled(WINDOW_NAME, vis, None)
        if SAVE_VIDEO:
            if 'out' not in locals() or out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))
            out.write(vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('n') and (current_target_id in delta_from_id):
            # 수동 다음
            dyaw, dpitch = delta_from_id[current_target_id]
            target_yaw   = cur_yaw   - dyaw
            target_pitch = cur_pitch + dpitch
            send_servo_angles(ctl, target_yaw, target_pitch)
            cur_yaw, cur_pitch = target_yaw, target_pitch
            current_target_id  = next_id_map[current_target_id]
            print(f"[Manual Next] moved with Δ (dyaw={dyaw:+.2f}, dpitch={dpitch:+.2f})")

    # 정리
    cap1.release(); cap2.release()
    if SAVE_VIDEO and out is not None:
        out.release(); print(f"[Info] 저장 완료: {OUT_PATH}")
    cv2.destroyAllWindows()
    try:
        ctl.close()
    except:
        pass

    # 그립 기록 저장
    with open(CSV_GRIPS_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["part", "hold_id", "cx", "cy"])
        writer.writerows(grip_records)
    print(f"[Info] 그립 CSV: {CSV_GRIPS_PATH} (행 수: {len(grip_records)})")

if __name__ == "__main__":
    main()
