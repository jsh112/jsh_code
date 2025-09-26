#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo YOLOv8n-Seg (first 10 frames merged) + MediaPipe-on-Left → Live overlay (mm)
+ Laser-origin yaw/pitch per hold (LEFT-camera-based)
+ ✅ DualServoController 연동 + Δ테이블(dyaw, dpitch) 기반 상대 이동
+ ✅ MediaPipe 모듈(import from B_Mediapipe)
+ ✅ (NEW) 웹 기반 색상 선택 지원
+ ✅ (NEW) 잡은 홀드(성공한 홀드)를 화면에서 칠해주기(반투명 표시)
    - 네트워크 비활성화 환경이면 --no_web 사용(키보드 입력 대체 방식)

사용 예시:
  python A_main.py --port COM8 --baud 115200 --pitch 90 --yaw 90
  python A_main.py --port COM8 --baud 115200 --pitch 90 --yaw 90 --no_web  # 콘솔로 색 선택

필요 파일(같은 폴더): B_Mediapipe.py, servo_control.py, A_web.py
사전 설치: pip install ultralytics opencv-python mediapipe flask(선택)`
"""

import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import cv2
import argparse

# === MediaPipe 모듈 ===
from Climb_Mediapipe import PoseTracker, TouchCounter, draw_pose_points

# === (NEW) 웹 모듈 - 색상 선택 ===
_USE_WEB = True
try:
    from A_web import choose_color_via_web
except Exception:
    _USE_WEB = False
    def choose_color_via_web(*a, **k):
        raise RuntimeError("color_web 모듈(A_web)이 로드되지 않았습니다.")

# ========= 사용자 환경 경로 =========
NPZ_PATH       = r"/home/jojang/Desktop/climbing/stereo_params_scaled.npz"
MODEL_PATH     = r"/home/jojang/Desktop/climbing/best_5.pt"

CAM1_INDEX     = 0   # 왼쪽 카메라
CAM2_INDEX     = 1   # 오른쪽 카메라

SWAP_INPUT     = False   # 입력 좌/우 스왑
SWAP_DISPLAY   = False   # 화면 표시 좌/우 스왑

WINDOW_NAME    = "Rectified L | R  (10f merged; MP Left; Δ-Relative Servo + WEB)"
SHOW_GRID      = False
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30
SELECTED_COLOR = None    # 예: 'orange' (None=전체)

# 자동 진행(터치→다음 홀드) 관련
TOUCH_THRESHOLD = 10     # in-polygon 연속 프레임 임계(기본 10)
ADV_COOLDOWN    = 0.5    # 연속 넘김 방지 쿨다운(sec)

# 저장 옵션
SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"
CSV_GRIPS_PATH = "grip_records.csv"

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 1.85
LASER_OFFSET_CM_UP   = 8.0
LASER_OFFSET_CM_FWD  = -3.3
Y_UP_IS_NEGATIVE     = True  # 위 방향이 -y인 좌표계면 True

# 각도 보정/선형 캘리브레이션(필요시 사용)
YAW_OFFSET_DEG   = 0.0
PITCH_OFFSET_DEG = 0.0
USE_LINEAR_CAL   = False
A11, A12, B1     = 1.0, 0.0, 0.0
A21, A22, B2     = 0.0, 1.0, 0.0

# === 서보 기준(중립 90/90) & 부호/스케일 ===
BASE_YAW_DEG   = 90.0   # 서보 중립
BASE_PITCH_DEG = 90.0   # 서보 중립
YAW_SIGN       = -1.0   # 반대로 가면 -1.0
PITCH_SIGN     = +1.0   # 반대로 가면 -1.0
YAW_SCALE      = 1.0    # 필요시 감도 미세조정
PITCH_SCALE    = 1.0

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

# ==== 색상 맵 ====
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

def _sanitize_label(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in ("_", "-"))

def ask_color_and_map_to_class(all_colors_dict):
    print("가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("필터할 색상 입력(엔터=전체): ").strip().lower()
    if not s:
        print("→ 전체 표시 사용")
        return None, "all"   # (모델클래스=None, 파일라벨="all")
    mapped = all_colors_dict.get(s)
    if mapped is None:
        print(f"입력 '{s}' 은(는) 유효하지 않은 색입니다. 전체 표시 사용")
        return None, "all"
    print(f"선택된 클래스명: {mapped}")
    return mapped, s        # (모델클래스, 파일라벨)

def to_servo_cmd(yaw_opt_deg, pitch_opt_deg):
    """
    광학각(카메라 전방 +Z 기준의 yaw/pitch, 단위 °) -> 서보 명령각(°)
    '90/90이 정면'이 되도록 중립 오프셋을 더해준다.
    """
    y = BASE_YAW_DEG   + YAW_SIGN   * (YAW_SCALE   * yaw_opt_deg)
    p = BASE_PITCH_DEG + PITCH_SIGN * (PITCH_SCALE * pitch_opt_deg)
    # 안전 클램프(필요하면 유지/수정)
    y = max(0.0, min(180.0, y))
    p = max(0.0, min(180.0, p))
    return y, p

# === (NEW) CSV에서 경로 순서 로드 ===
def load_route_ids_from_csv(path):
    route_ids = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "hold_id" in row:
                    try:
                        hid = int(row["hold_id"])
                        route_ids.append(hid)
                    except:
                        pass
    except FileNotFoundError:
        print(f"[Warn] 경로 CSV '{path}' 없음 → 인덱스 순서 사용")
    return route_ids

def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    M  = np.array([0.5*Tx, 0.0, 0.0], dtype=np.float64)  # 기준점(시각화시)
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H), B, M)

def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_V4L2)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_V4L2)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("카메라 오픈 실패. 연결/권한 확인.")
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

def send_servo_angles(ctl, yaw_cmd, pitch_cmd):
    try:
        print(f"[Servo] send: yaw={yaw_cmd:.2f}°, pitch={pitch_cmd:.2f}°")
        ctl.set_angles(pitch_cmd, yaw_cmd)  # (pitch, yaw) 순서
    except Exception as e:
        print(f"[Servo ERROR] {e}")

# ---------- 메인 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--pitch", type=float, default=None, help="초기 각도(옵션). 미지정 시 첫 타깃으로 자동 조준")
    ap.add_argument("--yaw",   type=float, default=None, help="초기 각도(옵션). 미지정 시 첫 타깃으로 자동 조준")
    ap.add_argument("--laser_on",  action="store_true")
    ap.add_argument("--laser_off", action="store_true")
    ap.add_argument("--no_auto_advance", action="store_true")
    ap.add_argument("--no_web", action="store_true")
    args = ap.parse_args()

    # 경로 검증
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

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

    # ================= 색상 필터 선택 =================
    selected_class_name  = None     # 모델 클래스명 (예: Hold_Green)
    selected_color_label = "all"    # 파일명 라벨 (예: green, orange, all)

    # 1) 웹에서 색상 선택(가능하고 --no_web가 아닐 때)
    if (not args.no_web) and _USE_WEB:
        try:
            chosen = choose_color_via_web(
                all_colors=list(ALL_COLORS.keys()),
                defaults={"port": args.port, "baud": args.baud}
            )  # ""이면 전체
            if chosen:
                mapped = ALL_COLORS.get(chosen)
                if mapped is None:
                    print(f"[Filter] 웹 선택 '{chosen}' 무효 → 전체 표시")
                else:
                    print(f"[Filter] 웹 선택: {chosen} → {mapped}")
                    selected_class_name  = mapped
                    selected_color_label = chosen.lower()
            else:
                print("[Filter] 웹에서 전체 선택")
        except Exception as e:
            print(f"[Filter] 웹 선택 실패 → 콘솔 대체: {e}")

    # 2) 고정 설정 값
    if (selected_class_name is None) and (SELECTED_COLOR is not None):
        sc = SELECTED_COLOR.strip().lower()
        mapped = ALL_COLORS.get(sc)
        if mapped is None:
            print(f"[Filter] SELECTED_COLOR='{SELECTED_COLOR}' 무효 → 콘솔에서 선택")
            selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 고정 선택 클래스: {mapped}")
            selected_class_name  = mapped
            selected_color_label = sc

    # 3) 콘솔 입력 대체
    if selected_class_name is None and (args.no_web or not _USE_WEB):
        selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)

    # === 여기서 색상 라벨에 맞춰 CSV 파일명 생성 ===
    csv_label = _sanitize_label(selected_color_label) if selected_color_label else "all"
    CSV_GRIPS_PATH = f"grip_records_{csv_label}.csv"
    print(f"[Info] 경로 CSV: {CSV_GRIPS_PATH}")

    # 카메라 & 모델
    capL_idx, capR_idx = CAM1_INDEX, CAM2_INDEX
    if SWAP_INPUT:
        capL_idx, capR_idx = capR_idx, capL_idx
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)
    model = YOLO(str(MODEL_PATH))

    # ====== 초기 10프레임: YOLO seg & merge ======
    print(f"[Init] First 10 frames: YOLO seg & merge ...")
    L_sets, R_sets = [], []
    for _ in range(2):
        cap1.read(); cap2.read()  # 워밍업

    for k in range(3):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡쳐 실패")
        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)
        holdsL_k = extract_holds_with_indices(Lr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        holdsR_k = extract_holds_with_indices(Rr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        L_sets.append(holdsL_k); R_sets.append(holdsR_k)
        print(f"  - frame {k+1}/3: L={len(holdsL_k)}  R={len(holdsR_k)}")

    holdsL = assign_indices(merge_holds_by_center(L_sets, 18), ROW_TOL_Y)
    holdsR = assign_indices(merge_holds_by_center(R_sets, 18), ROW_TOL_Y)
    if not holdsL or not holdsR:
        cap1.release(); cap2.release()
        print("[Warn] 왼/오 프레임에서 홀드가 검출되지 않았습니다.")
        return

    # 공통 ID
    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        print("[Warn] 좌/우 공통 hold_index가 없습니다.")
        return
    print(f"[Info] 공통 홀드 개수: {len(common_ids)}")

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

    # ===== Δ 테이블 구성 (CSV 순서 기반) =====
    by_id  = {mr["hid"]: mr for mr in matched_results}
    route_ids = load_route_ids_from_csv(CSV_GRIPS_PATH)
    if not route_ids:
        route_ids = sorted(by_id.keys())

    next_id_map   = {}
    delta_from_id = {}
    angle_deltas  = []

    for i in range(len(route_ids)-1):
        a_id, b_id = route_ids[i], route_ids[i+1]
        if (a_id in by_id) and (b_id in by_id):
            a, b = by_id[a_id], by_id[b_id]
            dyaw   = wrap_deg(b["yaw_deg"]   - a["yaw_deg"])
            dpitch = wrap_deg(b["pitch_deg"] - a["pitch_deg"])
            v1 = a["X"] - O; v2 = b["X"] - O
            d3d = angle_between(v1, v2)
            angle_deltas.append((a_id, b_id, dyaw, dpitch, d3d))
            next_id_map[a_id]   = b_id
            delta_from_id[a_id] = (dyaw, dpitch)

    print("[ΔAngles] (CSV order):")
    for a_id, b_id, dyaw, dpitch, d3d in angle_deltas:
        print(f"  {a_id}->{b_id}: Δyaw={dyaw:+.2f}°, Δpitch={dpitch:+.2f}°, angle={d3d:.2f}°")

    # ===== Servo 초기화 & 초기 자동 조준 (NEW) =====
    ctl = DualServoController(args.port, args.baud) if HAS_SERVO else DualServoController()

    # 1) 첫 타깃 선택 (CSV 첫 항목 우선, 없으면 by_id 중 최소 ID)
    if route_ids:
        current_target_id = route_ids[0]
    else:
        current_target_id = min(by_id.keys()) if by_id else None

    # 2) 첫 타깃 절대각 계산 → 캘리브레이션 적용
    if current_target_id is not None:
        mr0 = by_id[current_target_id]               # {'yaw_deg','pitch_deg', ...}
        auto_yaw, auto_pitch = mr0["yaw_deg"], mr0["pitch_deg"]
        yaw_cmd0, pitch_cmd0 = to_servo_cmd(auto_yaw, auto_pitch)
    else:
        yaw_cmd0, pitch_cmd0 = 0.0, 0.0              # 안전 기본값

    # 3) 사용자 각도 제공 여부 체크 → 없으면 자동 조준 사용
    pitch_arg = getattr(args, "pitch", None)         # argparse에 --pitch/--yaw 없더라도 안전
    yaw_arg   = getattr(args, "yaw",   None)
    user_angles_provided = (pitch_arg is not None) and (yaw_arg is not None)

    if user_angles_provided:
        cur_pitch, cur_yaw = float(pitch_arg), float(yaw_arg)
        print(f"[Init-Point] Using user angles: yaw={cur_yaw:.2f}°, pitch={cur_pitch:.2f}°")
    else:
        cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
        print(f"[Init-Point] Auto to ID{current_target_id}: yaw={cur_yaw:.2f}°, pitch={cur_pitch:.2f}°")

    # 4) 실제 서보 반영 + 자동진행 플래그
    ctl.set_angles(cur_pitch, cur_yaw)
    auto_advance_enabled = (not args.no_auto_advance)

    # ==== MediaPipe Pose ====
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)
    touch = TouchCounter(threshold_frames=TOUCH_THRESHOLD, cooldown_sec=ADV_COOLDOWN)

    # 그립 기록 및 '칠해지기' 표시 상태
    filled_ids = set()   # 성공 처리된(채워진) 홀드 ID
    blocked_state = {}   # (part, hold_id)별 차폐 상태

    # 비디오 저장
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))

    # 화면
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W, H)
    t_prev = time.time()

    # (NEW) 프레임 내 연쇄 넘김 방지 디바운스 타임스탬프
    last_advanced_time = 0.0

    try:
        while True:
            ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
            if not (ok1 and ok2):
                print("[Warn] 프레임 캡쳐 실패"); break

            Lr = rectify(f1, map1x, map1y, size)
            Rr = rectify(f2, map2x, map2y, size)
            vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])
            if SHOW_GRID:
                draw_grid(vis[:, :W]); draw_grid(vis[:, W:])

            # 검출 결과 그리기(성공 홀드는 반투명 칠하기)
            for side, holds in (("L", holdsL), ("R", holdsR)):
                xoff = xoff_for(side, W, SWAP_DISPLAY)
                for h in holds:
                    cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)

                    if h["hold_index"] in filled_ids:
                        overlay = vis.copy()
                        cv2.drawContours(overlay, [cnt_shifted], -1, h["color"], thickness=-1)
                        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

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

            # --- 디버그 3D 좌표/깊이 표시 ---
            y_info = 60
            for mr in matched_results:
                X = mr["X"]
                depth = X[2]
                txt3d = (f"ID{mr['hid']} : X=({X[0]:.1f}, {X[1]:.1f}, {X[2]:.1f}) mm "
                         f" | depth(Z)={depth:.1f} mm")
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                y_info += 18

            # NEXT 텍스트 및 현재 각도 표시
            y0 = 28
            if current_target_id in by_id:
                if current_target_id in delta_from_id:
                    dyaw, dpitch = delta_from_id[current_target_id]
                    nxt = next_id_map[current_target_id]
                    txt = (f"[NEXT] ID{current_target_id}→ID{nxt}  "
                           f"Δyaw={dyaw:+.1f}°, Δpitch={dpitch:+.1f}°  "
                           f"[now yaw={cur_yaw:.1f}°, pitch={cur_pitch:.1f}°]")
                else:
                    mr = by_id[current_target_id]
                    txt = (f"[LAST] ID{mr['hid']}  yaw={mr['yaw_deg']:.1f}°, pitch={mr['pitch_deg']:.1f}°  "
                           f"[now yaw={cur_yaw:.1f}°, pitch={cur_pitch:.1f}°]")
                cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1, cv2.LINE_AA)

            # === MediaPipe 포즈 추정 & 표시 ===
            coords = pose.process(Lr)
            left_xoff = xoff_for("L", W, SWAP_DISPLAY)
            draw_pose_points(vis, coords, offset_x=left_xoff)

            # === (핵심 수정) 판정 스냅샷 ID 사용 & 프레임당 1회만 타깃 변경 ===
            if coords and (current_target_id in idxL):
                # 스냅샷 ID
                tid = current_target_id
                hold = idxL[tid]

                # 터치 판정은 tid 기준으로 고정
                triggered, parts = touch.check(hold["contour"], coords, tid, now=time.time())

                # 관절 루프
                advanced_this_frame = False
                for name, (x, y) in coords.items():
                    key = (name, tid)  # 항상 tid 사용
                    inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0
                    cx, cy = hold["center"]

                    if inside:
                        # 성공 파트
                        if name in pose.success_parts:
                            # tid(스냅샷) 기준으로 아직 기록 안 했으면 기록
                            filled_ids.add(tid)

                             # 자동 넘김(프레임당 1회 제한 + 쿨다운)
                            now_t = time.time()
                            if (auto_advance_enabled and tid in delta_from_id
                                and (now_t - last_advanced_time) > ADV_COOLDOWN
                                and not advanced_this_frame):

                                dyaw, dpitch = delta_from_id[tid]
                                target_yaw   = cur_yaw   - dyaw
                                target_pitch = cur_pitch + dpitch
                                send_servo_angles(ctl, target_yaw, target_pitch)
                                cur_yaw, cur_pitch = target_yaw, target_pitch

                                current_target_id  = next_id_map[tid]  # ← 변경은 tid 기준
                                last_advanced_time = now_t
                                advanced_this_frame = True

                                # 같은 프레임에서 추가 판정/넘김 방지
                                break

                        # 차폐 파트
                        elif name in pose.blocking_parts:
                            if not blocked_state.get(key, False):
                                print(f"[BLOCKED] Hold ID={tid}, Pos=({cx},{cy}), blocked by {name}")
                                blocked_state[key] = True
                    else:
                        blocked_state[key] = False

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
                # (수동) 다음 이동도 프레임 내 연쇄를 막기 위해 즉시 다음 프레임으로
                filled_ids.add(current_target_id)
                dyaw, dpitch = delta_from_id[current_target_id]
                target_yaw   = cur_yaw   - dyaw
                target_pitch = cur_pitch + dpitch
                send_servo_angles(ctl, target_yaw, target_pitch)
                cur_yaw, cur_pitch = target_yaw, target_pitch
                current_target_id  = next_id_map[current_target_id]
                print(f"[Manual Next] moved with Δ (dyaw={dyaw:+.2f}, dpitch={dpitch:+.2f})")
                # 같은 프레임 추가 판정 방지 → 다음 루프로 즉시 진행
                continue

    finally:
        cap1.release(); cap2.release()
        if SAVE_VIDEO and out is not None:
            out.release(); print(f"[Info] 저장 완료: {OUT_PATH}")
        cv2.destroyAllWindows()
        try: pose.close()
        except: pass
        try: ctl.close()
        except: pass

if __name__ == "__main__":
    main()
