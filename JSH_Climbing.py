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
from test_angle import point_to_motor_angles, angle_to_pwm

# === (NEW) 웹 모듈 - 색상 선택 ===
_USE_WEB = True
try:
    from A_web import choose_color_via_web
except Exception:
    _USE_WEB = False
    def choose_color_via_web(*a, **k):
        raise RuntimeError("color_web 모듈(A_web)이 로드되지 않았습니다.")

# ========= 사용자 환경 경로 =========
NPZ_PATH       = r"/home/jsh/Desktop/JSH_CODE/jsh_code/stereo_params_scaled.npz"
MODEL_PATH     = r"/home/jsh/Desktop/JSH_CODE/jsh_code/best_5.pt"

CAM1_INDEX     = 1   # 왼쪽 카메라
CAM2_INDEX     = 0   # 오른쪽 카메라

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

# ---------- 장세환의 추가 코드 --------------
import struct

def send_servo_angles(yaw_pwm, pitch_pwm):
    yaw_pwm = int(yaw_pwm)
    pitch_pwm = int(pitch_pwm)
    data = struct.pack('<HH', yaw_pwm, pitch_pwm)
    ser.write(data)
    print(f"[Send] yaw_pwm={yaw_pwm}, pitch_pwm={pitch_pwm}")

def test_triangulate(ptsL, ptsR, P1, P2):
    """
    좌우 공통 포인트를 이용해서 3D 좌표를 계산하는 테스트 함수.
    
    ptsL, ptsR : Nx2 float32 좌표 (이미지 좌표)
    P1, P2     : 3x4 스테레오 카메라 프로젝션 행렬
    """
    # cv2.triangulatePoints는 homogeneous 좌표 반환 (4xN)
    pts4D = cv2.triangulatePoints(P1, P2, ptsL.T, ptsR.T)  # shape: 4xN
    pts3D = pts4D[:3] / pts4D[3]  # 정규화: X = x/w, Y = y/w, Z = z/w

    pts3D = pts3D.T  # Nx3
    for i, X in enumerate(pts3D):
        print(f"Point {i}: X={X[0]:.2f} mm, Y={X[1]:.2f} mm, Z={X[2]:.2f} mm")
    
    return pts3D

def test_triangulate_and_draw(ptsL, ptsR, P1, P2, imgL, imgR):
    """
    좌우 공통 포인트를 이용해 3D 좌표 계산 후, 좌/우 이미지 위에 표시.
    
    ptsL, ptsR : Nx2 float32 좌표 (이미지 좌표)
    P1, P2     : 3x4 스테레오 카메라 프로젝션 행렬
    imgL, imgR : 좌/우 이미지 (BGR)
    """
    # 3D 좌표 계산
    pts4D = cv2.triangulatePoints(P1, P2, ptsL.T, ptsR.T)  # shape: 4xN
    pts3D = (pts4D[:3] / pts4D[3]).T  # Nx3

    # 좌측 이미지에 표시
    imgL_draw = imgL.copy()
    imgR_draw = imgR.copy()

    for i, X in enumerate(pts3D):
        print(f"Point {i}: X={X[0]:.2f} mm, Y={X[1]:.2f} mm, Z={X[2]:.2f} mm")
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # 좌측 이미지
        ptL = tuple(ptsL[i].astype(int))
        cv2.circle(imgL_draw, ptL, 5, color, -1)
        cv2.putText(imgL_draw, f"{i}", (ptL[0]+5, ptL[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 우측 이미지: 3D -> 이미지 평면 투영
        X_h = np.hstack([X, 1.0])  # homogeneous
        proj = P2 @ X_h  # 3x1
        proj /= proj[2]
        ptR = tuple(proj[:2].astype(int))
        cv2.circle(imgR_draw, ptR, 5, color, -1)
        cv2.putText(imgR_draw, f"{i}", (ptR[0]+5, ptR[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return pts3D, imgL_draw, imgR_draw

def reprojection_error(ptsL, ptsR, pts3D, P1, P2):
    errs_L = []
    errs_R = []
    reproj_coords = []  # 추가

    for i in range(len(pts3D)):
        X = np.append(pts3D[i], 1)  # homogeneous
        proj_L = P1 @ X
        proj_L /= proj_L[2]
        proj_R = P2 @ X
        proj_R /= proj_R[2]

        errL = np.linalg.norm(proj_L[:2] - ptsL[i])
        errR = np.linalg.norm(proj_R[:2] - ptsR[i])
        errs_L.append(errL)
        errs_R.append(errR)

        reproj_coords.append((proj_L[0], proj_L[1], proj_R[0], proj_R[1]))  # 저장

        print(f"Point {i}: errL={errL:.2f}px, errR={errR:.2f}px")

    print(f"[Summary] Avg error: Left={np.mean(errs_L):.2f}px, Right={np.mean(errs_R):.2f}px")
    return errs_L, errs_R, reproj_coords

def angle_to_pwm(angle_deg, min_ms=1.0, max_ms=2.0, min_deg=0, max_deg=180):
    ms = min_ms + (angle_deg - min_deg) * (max_ms - min_ms) / (max_deg - min_deg)
    return ms

def save_reproj_right_image(f_right, map2x, map2y, size, reproj_coords, common_ids, filename="reproj_right.png"):
    """
    우측 카메라에 재투영한 3D 좌표를 이미지에 점으로 표시하고 PNG로 저장.

    Args:
        f_right (np.ndarray): 우측 원본 프레임
        map2x, map2y: 우측 카메라 rectification 맵
        size (tuple): (W, H) 이미지 크기
        reproj_coords (list of tuples): (pxL, pyL, pxR, pyR) 재투영 좌표 리스트
        common_ids (list): 홀드 ID 리스트 (reproj_coords 순서와 동일)
        filename (str): 저장할 PNG 파일 이름
    """
    W, H = size
    # 우측 이미지 rectification
    vis_right = rectify(f_right, map2x, map2y, size).copy()

    # 각 점 표시
    for i, hid in enumerate(common_ids):
        _, _, pxR, pyR = reproj_coords[i]
        cv2.circle(vis_right, (int(pxR), int(pyR)), 6, (0,0,255), -1)       # 빨강 점
        cv2.putText(vis_right, f"{hid}", (int(pxR)+5, int(pyR)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # 이미지 저장
    cv2.imwrite(filename, vis_right)
    print(f"[Saved] Reprojected right image -> {filename}")

# ---------- 장세환의 추가 코드 --------------
def project_point(P, X):
    """
    3D 점 X를 카메라 행렬 P로 투영
    P : 3x4 projection matrix
    X : 3D point [X, Y, Z]
    return : (px, py) pixel 좌표
    """
    X_h = np.hstack([X, 1.0])          # 동차좌표
    x_proj = P @ X_h
    x_proj /= x_proj[2]
    return int(x_proj[0]), int(x_proj[1])

def draw_camera_origin(img, P, Z_dummy=3000, color=(0,0,255), label="Camera origin"):
    """
    이미지에 카메라 중앙점(0,0,Z_dummy)을 찍음
    """
    origin_3D = np.array([0.0, 0.0, Z_dummy])
    px, py = project_point(P, origin_3D)
    cv2.circle(img, (px, py), 6, color, -1)
    cv2.putText(img, label, (px+5, py-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
# ---------- 메인 ----------

def main():

    # 경로 검증
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    # 아두이노 연결
    import serial
    import struct
    import time

    # 아두이노 시리얼 초기화
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    time.sleep(2)  # 아두이노 리셋 대기

    def send_servo_angles(yaw_pwm, pitch_pwm):
        """yaw/pitch PWM 값을 아두이노로 전송"""
        data = struct.pack('<HH', yaw_pwm, pitch_pwm)  # 2바이트씩 little-endian
        ser.write(data)


    # 스테레오 로드
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

    # 레이저 원점 O (LEFT 기준)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dx = LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = L + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, LEFT-based) = {O}")

    # ================= 색상 필터 선택 =================
    # 하드코딩 예: 전체 표시
    selected_class_name  = 'Hold_Green'     
    selected_color_label = "all"
    csv_label = _sanitize_label(selected_color_label)
    print(f"[Info] Color filter: {selected_color_label}")

    # 카메라 & 모델
    cap1, cap2 = open_cams(CAM1_INDEX, CAM2_INDEX, size)
    model = YOLO(str(MODEL_PATH))

    # ====== 초기 10프레임: YOLO seg & merge ======
    print(f"[Init] First 3 frames: YOLO seg & merge ...")
    L_sets, R_sets = [], []
    for _ in range(2):  # 워밍업
        cap1.read(); cap2.read()

    for k in range(3):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡쳐 실패")
        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)
        # Lr_k_vis = draw_camera_origin(Lr_k.copy(), P1, origin_3D=O)
        # Rr_k_vis = draw_camera_origin(Rr_k.copy(), P2, origin_3D=O)


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

    # 1. 좌표 추출
    ptsL = np.array([idxL[hid]["center"] for hid in common_ids], dtype=np.float32)
    ptsR = np.array([idxR[hid]["center"] for hid in common_ids], dtype=np.float32)
    print(f"ptsL is {ptsL}")
    print(f"ptsR is {ptsR}")
    # 2. Fundamental matrix 계산 (테스트용)
    # 좌우 수평 오차 계산 (Fundamental matrix 없이)
    y_errors = np.abs(ptsL[:,1] - ptsR[:,1])
    print(f"[Epipolar check] 평균 Y 오차: {np.mean(y_errors):.2f}px, 최대: {np.max(y_errors):.2f}px")

    pts3D = test_triangulate(ptsL, ptsR, P1, P2)
    # pts3D로 이제 각도 계산
    for i, point in enumerate(pts3D):
        # 1. 레이저 원점 기준으로 pitch, yaw 각도 계산
        pitch_deg, yaw_deg = point_to_motor_angles(point, O)
    
        pitch_pwm = angle_to_pwm(pitch, min_angle=-30, max_angle=30, min_pwm=1000, max_pwm=2000)
        yaw_pwm   = angle_to_pwm(yaw,   min_angle=-30, max_angle=30, min_pwm=1000, max_pwm=2000)

        
        # 3. 값 확인
        print(f"Point {i}: Pitch={pitch_deg:.2f}° ({pitch_pwm}us), Yaw={yaw_deg:.2f}° ({yaw_pwm}us)")
        
        send_servo_angles(yaw_pwm, pitch_pwm)
        # 서보가 목표 위치에 도달할 시간을 잠깐 기다릴 수도 있음
        time.sleep(0.5)
    
    errs_L, errs_R, reproj_coords = reprojection_error(ptsL, ptsR, pts3D, P1, P2)
    print(f"errs_L is {errs_L}")
    print(f"errs_R is {errs_R}")

    ok2, f2 = cap2.read()  # 우측 카메라 프레임
    save_reproj_right_image(f2, map2x, map2y, size, reproj_coords, common_ids, filename="right_reproj.png")

    
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

    # ===== Delta 테이블 (순서=hold_index 순) =====
    by_id  = {mr["hid"]: mr for mr in matched_results}
    route_ids = sorted(by_id.keys())
    next_id_map   = {}
    delta_from_id = {}
    angle_deltas  = []

    for i in range(len(route_ids)-1):
        a_id, b_id = route_ids[i], route_ids[i+1]
        a, b = by_id[a_id], by_id[b_id]
        dyaw   = wrap_deg(b["yaw_deg"]   - a["yaw_deg"])
        dpitch = wrap_deg(b["pitch_deg"] - a["pitch_deg"])
        v1 = a["X"] - O; v2 = b["X"] - O
        d3d = angle_between(v1, v2)
        angle_deltas.append((a_id, b_id, dyaw, dpitch, d3d))
        next_id_map[a_id]   = b_id
        delta_from_id[a_id] = (dyaw, dpitch)

    # print("[ΔAngles] (hold_index order):")
    # for a_id, b_id, dyaw, dpitch, d3d in angle_deltas:
    #     print(f"  {a_id}->{b_id}: Δyaw={dyaw:+.2f}°, Δpitch={dpitch:+.2f}°, angle={d3d:.2f}°")
    

    # ===== Servo 초기화 & 초기 조준 =====
    # ctl = DualServoController() if not HAS_SERVO else DualServoController(args.port, args.baud)
    current_target_id = route_ids[0]
    mr0 = by_id[current_target_id]
    yaw_cmd0, pitch_cmd0 = to_servo_cmd(mr0["yaw_deg"], mr0["pitch_deg"])
    cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
    # ctl.set_angles(cur_pitch, cur_yaw)
    auto_advance_enabled = True

    # ==== MediaPipe Pose ====
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)
    touch = TouchCounter(threshold_frames=TOUCH_THRESHOLD, cooldown_sec=ADV_COOLDOWN)
    filled_ids = set()
    blocked_state = {}

    # 화면
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W, H)
    t_prev = time.time()
    last_advanced_time = 0.0

    try:
        while True:
            ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
            if not (ok1 and ok2):
                print("[Warn] 프레임 캡쳐 실패"); break

            Lr = rectify(f1, map1x, map1y, size)
            Rr = rectify(f2, map2x, map2y, size)
            vis = np.hstack([Lr, Rr])

            if SHOW_GRID: 
                draw_grid(vis[:, :W]); 
                draw_grid(vis[:, W:])
            
            # 3D 좌표 시각화 (좌/우 이미지에 원 그리기)
            for i, hid in enumerate(common_ids):
                # 원래 좌/우 이미지 좌표 (YOLO로 검출된 홀드 중심)
                cxL, cyL = ptsL[i].astype(int)
                cxR, cyR = ptsR[i].astype(int)

                # 우측 이미지 x-offset
                cxR_vis = cxR + W

                # 빨강: 원래 YOLO 중심
                cv2.circle(vis, (cxL, cyL), 1, (0, 0, 255), -1)       # 좌측
                cv2.circle(vis, (cxR_vis, cyR), 1, (0, 0, 255), -1)   # 우측

                # 녹색: 재투영 좌표
                pxL, pyL, pxR, pyR = reproj_coords[i]  # reprojection_error 또는 직접 계산
                cv2.circle(vis, (int(pxL), int(pyL)), 2, (0, 255, 0), 2)       # 좌측
                cv2.circle(vis, (int(pxR) + W, int(pyR)), 2, (0, 255, 0), 2)   # 우측

                # ID 텍스트
                cv2.putText(vis, f"{hid}", (cxL-10, cyL-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


            # 검출 홀드 그리기
            for side, holds in (("L", holdsL), ("R", holdsR)):
                xoff = xoff_for(side, W, SWAP_DISPLAY)
                for h in holds:
                    cnt_shifted = h["contour"] + np.array([[[xoff,0]]], dtype=h["contour"].dtype)
                    if h["hold_index"] in filled_ids:
                        overlay = vis.copy()
                        cv2.drawContours(overlay, [cnt_shifted], -1, h["color"], -1)
                        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
                    cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                    cx, cy = h["center"]
                    cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                    tag = f"ID:{h['hold_index']}"
                    if h["hold_index"] == current_target_id: tag = "[TARGET] " + tag
                    cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

            # MediaPipe
            coords = pose.process(Lr)
            draw_pose_points(vis, coords, offset_x=0)

            # 터치 판정 & 자동 진행
            if coords and (current_target_id in idxL):
                tid = current_target_id
                hold = idxL[tid]
                triggered, parts = touch.check(hold["contour"], coords, tid, now=time.time())
                for name, (x, y) in coords.items():
                    key = (name, tid)
                    inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0
                    if inside and name in pose.success_parts and tid not in filled_ids:
                        filled_ids.add(tid)
                        now_t = time.time()
                        if tid in delta_from_id and (now_t - last_advanced_time) > ADV_COOLDOWN:
                            dyaw, dpitch = delta_from_id[tid]
                            target_yaw   = cur_yaw - dyaw
                            target_pitch = cur_pitch + dpitch
                            # send_servo_angles(ctl, target_yaw, target_pitch)
                            cur_yaw, cur_pitch = target_yaw, target_pitch
                            current_target_id = next_id_map[tid]
                            last_advanced_time = now_t
                            break

            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-6); t_prev = t_now
            cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap1.release(); cap2.release()
        cv2.destroyAllWindows()
        try: pose.close()
        except: pass
        try: print("hi")# ctl.close()
        except: pass

# def main():
#     # 경로 검증
#     for p in (NPZ_PATH, MODEL_PATH):
#         if not Path(p).exists():
#             raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

#     # 스테레오 로드
#     map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
#     W, H = size
#     print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

#     # 레이저 원점 O (LEFT 기준)
#     L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
#     dx = -LASER_OFFSET_CM_LEFT * 10.0
#     dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
#     dz = LASER_OFFSET_CM_FWD * 10.0
#     O  = L + np.array([dx, dy, dz], dtype=np.float64)
#     print(f"[Laser] Origin O (mm, LEFT-based) = {O}")

#     # ================= 색상 필터 선택 =================
#     # 하드코딩 예: 전체 표시
#     selected_class_name  = 'Hold_Green'     
#     selected_color_label = "all"
#     csv_label = _sanitize_label(selected_color_label)
#     print(f"[Info] Color filter: {selected_color_label}")

#     # 카메라 & 모델
#     cap1, cap2 = open_cams(CAM1_INDEX, CAM2_INDEX, size)
#     model = YOLO(str(MODEL_PATH))

#     # ====== 초기 10프레임: YOLO seg & merge ======
#     print(f"[Init] First 3 frames: YOLO seg & merge ...")
#     L_sets, R_sets = [], []
#     for _ in range(2):  # 워밍업
#         cap1.read(); cap2.read()

#     for k in range(3):
#         ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
#         if not (ok1 and ok2):
#             cap1.release(); cap2.release()
#             raise SystemExit("초기 프레임 캡쳐 실패")
#         Lr_k = rectify(f1, map1x, map1y, size)
#         Rr_k = rectify(f2, map2x, map2y, size)
#         holdsL_k = extract_holds_with_indices(Lr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
#         holdsR_k = extract_holds_with_indices(Rr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
#         L_sets.append(holdsL_k); R_sets.append(holdsR_k)
#         print(f"  - frame {k+1}/3: L={len(holdsL_k)}  R={len(holdsR_k)}")

#     holdsL = assign_indices(merge_holds_by_center(L_sets, 18), ROW_TOL_Y)
#     holdsR = assign_indices(merge_holds_by_center(R_sets, 18), ROW_TOL_Y)
#     if not holdsL or not holdsR:
#         cap1.release(); cap2.release()
#         print("[Warn] 왼/오 프레임에서 홀드가 검출되지 않았습니다.")
#         return

#     # 공통 ID
#     idxL = {h["hold_index"]: h for h in holdsL}
#     idxR = {h["hold_index"]: h for h in holdsR}
#     common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
#     if not common_ids:
#         print("[Warn] 좌/우 공통 hold_index가 없습니다.")
#         return
#     print(f"[Info] 공통 홀드 개수: {len(common_ids)}")

#     # 1. 좌표 추출
#     ptsL = np.array([idxL[hid]["center"] for hid in common_ids], dtype=np.float32)
#     ptsR = np.array([idxR[hid]["center"] for hid in common_ids], dtype=np.float32)
#     print(f"ptsL is {ptsL}")
#     print(f"ptsR is {ptsR}")
#     # 2. Fundamental matrix 계산 (테스트용)
#     # 좌우 수평 오차 계산 (Fundamental matrix 없이)
#     y_errors = np.abs(ptsL[:,1] - ptsR[:,1])
#     print(f"[Epipolar check] 평균 Y 오차: {np.mean(y_errors):.2f}px, 최대: {np.max(y_errors):.2f}px")

#     pts3D = test_triangulate(ptsL, ptsR, P1, P2)
#     errs_L, errs_R = reprojection_error(ptsL, ptsR, pts3D, P1, P2)  # 여기서 한 번만
#     print(f"errs_L is {errs_L}")
#     print(f"errs_R is {errs_R}")
    
#     # 3D/각도 계산
#     matched_results = []
#     for hid in common_ids:
#         Lh = idxL[hid]; Rh = idxR[hid]
#         X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
#         d_left  = float(np.linalg.norm(X - L))
#         d_line  = float(np.hypot(X[1], X[2]))
#         yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
#         matched_results.append({
#             "hid": hid, "color": Lh["color"],
#             "X": X, "d_left": d_left, "d_line": d_line,
#             "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
#         })

#     # ===== Delta 테이블 (순서=hold_index 순) =====
#     by_id  = {mr["hid"]: mr for mr in matched_results}
#     route_ids = sorted(by_id.keys())
#     next_id_map   = {}
#     delta_from_id = {}
#     angle_deltas  = []

#     for i in range(len(route_ids)-1):
#         a_id, b_id = route_ids[i], route_ids[i+1]
#         a, b = by_id[a_id], by_id[b_id]
#         dyaw   = wrap_deg(b["yaw_deg"]   - a["yaw_deg"])
#         dpitch = wrap_deg(b["pitch_deg"] - a["pitch_deg"])
#         v1 = a["X"] - O; v2 = b["X"] - O
#         d3d = angle_between(v1, v2)
#         angle_deltas.append((a_id, b_id, dyaw, dpitch, d3d))
#         next_id_map[a_id]   = b_id
#         delta_from_id[a_id] = (dyaw, dpitch)

#     print("[ΔAngles] (hold_index order):")
#     for a_id, b_id, dyaw, dpitch, d3d in angle_deltas:
#         print(f"  {a_id}->{b_id}: Δyaw={dyaw:+.2f}°, Δpitch={dpitch:+.2f}°, angle={d3d:.2f}°")
    

#     # ===== Servo 초기화 & 초기 조준 =====
#     # ctl = DualServoController() if not HAS_SERVO else DualServoController(args.port, args.baud)
#     current_target_id = route_ids[0]
#     mr0 = by_id[current_target_id]
#     yaw_cmd0, pitch_cmd0 = to_servo_cmd(mr0["yaw_deg"], mr0["pitch_deg"])
#     cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
#     # ctl.set_angles(cur_pitch, cur_yaw)
#     auto_advance_enabled = True

#     # ==== MediaPipe Pose ====
#     pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)
#     touch = TouchCounter(threshold_frames=TOUCH_THRESHOLD, cooldown_sec=ADV_COOLDOWN)
#     filled_ids = set()
#     blocked_state = {}

#     # 화면
#     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(WINDOW_NAME, W, H)
#     t_prev = time.time()
#     last_advanced_time = 0.0

#     try:
#         while True:
#             ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
#             if not (ok1 and ok2):
#                 print("[Warn] 프레임 캡쳐 실패"); break

#             Lr = rectify(f1, map1x, map1y, size)
#             Rr = rectify(f2, map2x, map2y, size)
#             vis = np.hstack([Lr, Rr])

#             if SHOW_GRID: 
#                 draw_grid(vis[:, :W]); 
#                 draw_grid(vis[:, W:])
            
#             # 3D 좌표 시각화 (좌/우 이미지에 원 그리기)
#             for i, hid in enumerate(common_ids):
#                 cxL, cyL = ptsL[i].astype(int)
#                 cxR, cyR = ptsR[i].astype(int) + W  # 우측 이미지는 x-offset 필요
#                 cv2.circle(vis, (cxL, cyL), 5, (0, 0, 255), -1)  # 빨강: 좌
#                 cv2.circle(vis, (cxR, cyR), 5, (0, 0, 255), -1)  # 빨강: 우
#                 cv2.putText(vis, f"{hid}", (cxL-10, cyL-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

#             # 검출 홀드 그리기
#             for side, holds in (("L", holdsL), ("R", holdsR)):
#                 xoff = xoff_for(side, W, SWAP_DISPLAY)
#                 for h in holds:
#                     cnt_shifted = h["contour"] + np.array([[[xoff,0]]], dtype=h["contour"].dtype)
#                     if h["hold_index"] in filled_ids:
#                         overlay = vis.copy()
#                         cv2.drawContours(overlay, [cnt_shifted], -1, h["color"], -1)
#                         vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
#                     cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
#                     cx, cy = h["center"]
#                     cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
#                     tag = f"ID:{h['hold_index']}"
#                     if h["hold_index"] == current_target_id: tag = "[TARGET] " + tag
#                     cv2.putText(vis, tag, (cx+xoff-10, cy+26),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
#                     cv2.putText(vis, tag, (cx+xoff-10, cy+26),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

#             # MediaPipe
#             coords = pose.process(Lr)
#             draw_pose_points(vis, coords, offset_x=0)

#             # 터치 판정 & 자동 진행
#             if coords and (current_target_id in idxL):
#                 tid = current_target_id
#                 hold = idxL[tid]
#                 triggered, parts = touch.check(hold["contour"], coords, tid, now=time.time())
#                 for name, (x, y) in coords.items():
#                     key = (name, tid)
#                     inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0
#                     if inside and name in pose.success_parts and tid not in filled_ids:
#                         filled_ids.add(tid)
#                         now_t = time.time()
#                         if tid in delta_from_id and (now_t - last_advanced_time) > ADV_COOLDOWN:
#                             dyaw, dpitch = delta_from_id[tid]
#                             target_yaw   = cur_yaw - dyaw
#                             target_pitch = cur_pitch + dpitch
#                             # send_servo_angles(ctl, target_yaw, target_pitch)
#                             cur_yaw, cur_pitch = target_yaw, target_pitch
#                             current_target_id = next_id_map[tid]
#                             last_advanced_time = now_t
#                             break

#             # FPS
#             t_now = time.time()
#             fps = 1.0 / max(t_now - t_prev, 1e-6); t_prev = t_now
#             cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
#                         (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

#             cv2.imshow(WINDOW_NAME, vis)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         cap1.release(); cap2.release()
#         cv2.destroyAllWindows()
#         try: pose.close()
#         except: pass
#         try: print("hi")# ctl.close()
#         except: pass

if __name__ == "__main__":
    main()
