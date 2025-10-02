#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C_Climbing.py — 사용자 등반 모드 (CSV 경로 + 3D 삼각측량 + 레이저 초기보정 + 서보 이동)

변경/핵심 흐름:
1) 시작: (90,90) 센터 → 레이저 OFF
2) 레이저 OFF 상태에서 BEFORE 프레임 1장 자동 캡처
3) YOLO 10프레임 병합(OFF 유지)
4) 레이저 ON → AFTER 프레임 1장 자동 캡처
5) C_laser_diff_detect 로 좌/우 레이저 픽셀 찾기 → 3D(X) 및 yaw/pitch 계산
6) (90,90) 원점으로 간주한 레이저 각과 첫 홀드 절대각의 차(Δyaw,Δpitch)를 계산
7) 5초 카운트다운 후 첫 홀드로 한 번에 이동
8) 이후 Mediapipe 터치 시 CSV 순서대로 Δyaw/Δpitch 상대 이동 (A_Climbing 로직 반영)
9) 마지막 홀드 성공 시 레이저 OFF
10) 필수 로그: [ΔAngles] (CSV order) … 반드시 출력
11) 화면: 홀드별 3D/깊이 텍스트, 레이저 diff 오버레이(좌/우) 창 표시

주의:
- HTTP 스팸 로그 억제: werkzeug 로거를 WARNING 이상으로 낮추고 import C_web
"""

import os, sys, time, csv, argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 외부 모듈
from Climb_Mediapipe import PoseTracker, TouchCounter, draw_pose_points
from C_laser_diff_detect import find_laser_point_pair  # 레이저 diff 탐지
from C_web import choose_meta_via_web, meta_to_csv_filename, update_state, consume_flags

# ===== Servo =====
try:
    from servo_control import DualServoController
    HAS_SERVO = True
except Exception:
    HAS_SERVO = False
    class DualServoController:
        def __init__(self, *a, **k): print("[Servo] (stub) controller unavailable")
        def set_angles(self, pitch=None, yaw=None): print(f"[Servo] (stub) set_angles: P={pitch}, Y={yaw}")
        def center(self): print("[Servo] (stub) center")
        def query(self): print("[Servo] (stub) query"); return {"pitch":90,"yaw":90,"laser":0}
        def laser_on(self): print("[Servo] (stub) laser_on")
        def laser_off(self): print("[Servo] (stub) laser_off")
        def close(self): pass

# ===== 사용자/환경 설정 =====
MODEL_PATH = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\best_6.pt"
NPZ_PATH   = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\calib_out\old_camera_same\stereo\stereo_params_scaled.npz"

LEFT_CAM, RIGHT_CAM = 1, 2
INIT_MERGE_FRAMES = 10
ROW_TOL_Y = 30
TOUCH_THRESHOLD = 10
ADV_COOLDOWN = 0.5
ROUTES_DIR = Path("./routes"); ROUTES_DIR.mkdir(parents=True, exist_ok=True)

# 레이저 원점(LEFT 기준) 오프셋 & 좌표계 방향
LASER_OFFSET_CM_LEFT = 1.85
LASER_OFFSET_CM_UP   = 8.0
LASER_OFFSET_CM_FWD  = -3.3
Y_UP_IS_NEGATIVE     = True

# 서보 기준/부호/스케일 (절대각 변환)
BASE_YAW_DEG   = 90.0
BASE_PITCH_DEG = 90.0
YAW_SIGN       = -1.0
PITCH_SIGN     = +1.0
YAW_SCALE      = 1.0
PITCH_SCALE    = 1.0

# (상대 이동 시 커맨드 업데이트 부호)
DELTA_YAW_CMD_SIGN   = -1.0   # cur_yaw   += SIGN * dyaw
DELTA_PITCH_CMD_SIGN = +1.0   # cur_pitch += SIGN * dpitch

# 색/표시
ALLOWED_COLORS = ["black","blue","gray","green","lime","orange","pink","purple","red","sky","white","yellow"]
COLOR_CANON = {"grey":"gray"}
BGR = {
    "black":(20,20,20), "blue":(255,0,0), "gray":(150,150,150), "green":(0,255,0),
    "lime":(50,255,128), "orange":(0,165,255), "pink":(203,192,255), "purple":(204,50,153),
    "red":(0,0,255), "sky":(255,255,0), "white":(255,255,255), "yellow":(0,255,255)
}

# ===== 보조 함수 =====
def _align32(x: int) -> int:
    return int((x + 31) // 32) * 32

def _extract_color_token(cls_name: str) -> str:
    n = cls_name.strip().lower()
    for sep in ("_", "-", "/", " "):
        if sep in n: n = n.split(sep)[-1]
    return COLOR_CANON.get(n, n)

def _mask_to_polygon(mask: np.ndarray):
    if mask.dtype != np.uint8: mask = mask.astype(np.uint8)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20: return None
    eps = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1,2)
    return approx

def _assign_hold_ids(centers):
    if not centers: return []
    pts = [(i,cx,cy) for i,(cx,cy,_,_) in enumerate(centers)]
    pts.sort(key=lambda t:(round(t[2]/ROW_TOL_Y), t[1]))
    idmap={}
    for new_id,(orig,_,_) in enumerate(pts): idmap[orig]=new_id
    out=[]
    for idx,(cx,cy,color,poly) in enumerate(centers):
        out.append((idmap[idx], cx, cy, color, poly))
    return out

def _infer_batched(model, imgs, infer_hw, conf=0.25, iou=0.5):
    return model.predict(imgs, verbose=False, conf=conf, iou=iou, imgsz=infer_hw)

def _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, selected_color):
    merge_L, merge_R = [], []
    for i in range(INIT_MERGE_FRAMES):
        okL, fL = capL.read(); okR, fR = capR.read()
        if not (okL and okR): break
        if map1L is not None:
            fL = cv2.remap(fL, map1L, map2L, cv2.INTER_LINEAR)
            fR = cv2.remap(fR, map1R, map2R, cv2.INTER_LINEAR)
        res_list = _infer_batched(model, [fL, fR], infer_hw)
        for side_idx, res in enumerate(res_list):
            side_merge = merge_L if side_idx==0 else merge_R
            if getattr(res, "masks", None) is None: continue
            mks = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros(len(mks), dtype=int)
            names = res.names if hasattr(res, "names") else {}
            dets=[]
            for k, mk in enumerate(mks):
                mask = (mk>0.5).astype(np.uint8)*255
                ys,xs = np.where(mask>0)
                if xs.size==0: continue
                cx,cy = int(xs.mean()), int(ys.mean())
                cname = names.get(int(cls[k]), str(int(cls[k])))
                tok = _extract_color_token(cname)
                if tok not in ALLOWED_COLORS: continue
                if (selected_color is not None) and (tok != selected_color): continue
                dets.append((mask, tok, cx, cy))
            if not side_merge:
                side_merge = [(m.copy(), c, x, y) for (m,c,x,y) in dets]
            else:
                new=[]
                for (m0,c0,x0,y0) in side_merge:
                    comb=m0.copy()
                    for (m1,c1,x1,y1) in dets:
                        if c0==c1 and abs(x0-x1)<40 and abs(y0-y1)<40:
                            comb = cv2.bitwise_or(comb, m1)
                    new.append((comb,c0,x0,y0))
                side_merge = new
            if side_idx==0: merge_L = side_merge
            else: merge_R = side_merge

        # 진행 표시만 가볍게
        vis = fL.copy()
        cv2.putText(vis, f"Init YOLO merge {i+1}/{INIT_MERGE_FRAMES}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Left (Climb)", vis); cv2.imshow("Right (Climb)", fR)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return [], []

    def to_holds(side_merge):
        centers=[]
        for (mask, tok, cx, cy) in side_merge:
            poly = _mask_to_polygon(mask)
            if poly is None: continue
            centers.append((cx,cy,tok,poly))
        return _assign_hold_ids(centers)
    return to_holds(merge_L), to_holds(merge_R)

# ===== 스테레오/리맵/P행렬 로드 =====
def load_stereo_full(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H), B)

def rectify(img, mapx, mapy, size):
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# ===== 기하 =====
def triangulate_X(P1, P2, ptL, ptR):
    xl = np.array(ptL, dtype=np.float64).reshape(2,1)
    xr = np.array(ptR, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, xl, xr)
    X  = (Xh[:3] / Xh[3]).reshape(3)
    return X

def triangulate_xy(P1, P2, centerL, centerR):
    return triangulate_X(P1, P2, centerL, centerR)

def angle_between(v1, v2):
    a = np.linalg.norm(v1); b = np.linalg.norm(v2)
    if a < 1e-9 or b < 1e-9: return 0.0
    c = float(np.dot(v1, v2) / (a*b))
    c = max(-1.0, min(1.0, c))
    return np.degrees(np.arccos(c))

def yaw_pitch_from_X(X, O, y_up_is_negative=True):
    v = X - O
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw   = np.degrees(np.arctan2(vx, vz))
    pitch = np.degrees(np.arctan2((-vy if y_up_is_negative else vy), np.hypot(vx, vz)))
    return yaw, pitch

def wrap_deg(d): return (d + 180.0) % 360.0 - 180.0

def to_servo_cmd(yaw_opt_deg, pitch_opt_deg):
    y = BASE_YAW_DEG   + YAW_SIGN   * (YAW_SCALE   * yaw_opt_deg)
    p = BASE_PITCH_DEG + PITCH_SIGN * (PITCH_SCALE * pitch_opt_deg)
    y = max(0.0, min(180.0, y))
    p = max(0.0, min(180.0, p))
    return y, p

def send_servo_angles(ctl: DualServoController, yaw_cmd, pitch_cmd):
    yaw_cmd   = max(0.0, min(180.0, float(yaw_cmd)))
    pitch_cmd = max(0.0, min(180.0, float(pitch_cmd)))
    print(f"[Servo] set_angles -> yaw={yaw_cmd:.2f}°, pitch={pitch_cmd:.2f}°")
    ctl.set_angles(pitch_cmd, yaw_cmd)

# ===== 표시 =====
def draw_3d_debug_overlay(img, matched_results, x=20, y=60, dy=18):
    yy = y
    for mr in matched_results:
        X = mr["X"]; depth = float(X[2])
        line = (f"ID{mr['hid']} : X=({X[0]:.1f}, {X[1]:.1f}, {X[2]:.1f}) mm | depth(Z)={depth:.1f} mm")
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)
        yy += dy

# ===== 3D/Δ 계산 (공통) =====
def compute_3d_and_deltas(idxL, idxR, P1, P2, O, csv_rows):
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    matched_results = []
    for hid in common_ids:
        Lh, Rh = idxL[hid], idxR[hid]
        X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
        d_left  = float(np.linalg.norm(X - np.array([0,0,0], dtype=np.float64)))
        d_line  = float(np.hypot(X[1], X[2]))
        yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
        matched_results.append({
            "hid": hid, "color": Lh["color"],
            "X": X, "d_left": d_left, "d_line": d_line,
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
        })

    route_ids = [int(r["hold_id"]) for r in csv_rows] if csv_rows else []
    if not route_ids:
        route_ids = common_ids[:]

    by_id  = {mr["hid"]: mr for mr in matched_results}

    next_id_map, delta_from_id, angle_deltas = {}, {}, []
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

    # ===== 필수 디버그 출력 =====
    print("[ΔAngles] (CSV order):")
    for a_id, b_id, dyaw, dpitch, d3d in angle_deltas:
        print(f"  {a_id}->{b_id}: Δyaw={dyaw:+.2f}°, Δpitch={dpitch:+.2f}°, angle={d3d:.2f}°")

    return matched_results, by_id, route_ids, next_id_map, delta_from_id, angle_deltas

# ===== CLI =====
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", "--port", dest="port", default=None, help="Serial port, e.g., COM7 or /dev/ttyUSB0")
    ap.add_argument("--baud", dest="baud", type=int, default=115200)
    ap.add_argument("--pitch", dest="pitch", type=float, default=None, help="override initial pitch command (0-180)")
    ap.add_argument("--yaw",   dest="yaw",   type=float, default=None, help="override initial yaw command (0-180)")
    ap.add_argument("--no-auto-advance", action="store_true", help="disable automatic advancing on touch")
    return ap.parse_args()

def main():
    args = parse_args()
    # 0) 컨트롤러 생성(포트 미지정/실패 시 더미 폴백)
    class _DummyCtl:
        def set_angles(self, *a, **k): print("[Servo-Dummy] set_angles", a, k)
        def center(self): print("[Servo-Dummy] center")
        def query(self): return {"raw":"STATE 90 90 0", "pitch":90, "yaw":90, "laser":0}
        def laser_on(self): print("[Servo-Dummy] laser_on")
        def laser_off(self): print("[Servo-Dummy] laser_off")
        def close(self): pass
    try:
        if HAS_SERVO and (args.port is not None):
            ctl = DualServoController(args.port, args.baud)
        else:
            print("[Servo] 포트 미지정 → Dummy 모드로 실행합니다. (--serial COMx 로 실제 포트 지정 가능)")
            ctl = _DummyCtl()
    except Exception as e:
        print(f"[Servo] 컨트롤러 초기화 실패 → Dummy 모드로 전환 ({e})")
        ctl = _DummyCtl()

    # 1) 웹 메타 선택
    update_state(mode="climb")
    model = YOLO(MODEL_PATH)
    try:
        class_names = list(model.names.values())
    except Exception:
        class_names = None
    meta = choose_meta_via_web(yolo_class_names=class_names)
    csv_name = meta_to_csv_filename(meta)
    CSV_PATH = Path("./routes") / csv_name
    SELECTED = None if meta["color"] == "all" else meta["color"]
    print(f"[Info] 사용자 등반 모드, 메타={meta} → CSV: {CSV_PATH}")

    # 2) CSV 로드
    if not CSV_PATH.exists():
        print("[Error] CSV 없음. 경로 저장 모드에서 먼저 저장하세요.")
        black = np.zeros((480,640,3), np.uint8)
        cv2.putText(black, "No route CSV", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow("Left (Climb)", black); cv2.imshow("Right (Climb)", black)
        cv2.waitKey(1500); cv2.destroyAllWindows(); return

    csv_rows=[]
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_rows.append(row)
    if not csv_rows:
        print("[Error] CSV가 비었습니다."); return

    # 3) 스테레오/리맵/P 행렬
    map1x, map1y, map2x, map2y, P1, P2, size, baseline = load_stereo_full(NPZ_PATH)
    W, H = int(size[0]), int(size[1])
    infer_hw = (_align32(H), _align32(W))
    print(f"[Stereo] size={(W,H)}, baseline~{baseline:.2f} mm")

    # 레이저 원점 O
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = L + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, LEFT-based) = {O.astype(int)}")

    # 4) 카메라
    capL = cv2.VideoCapture(LEFT_CAM, cv2.CAP_DSHOW if os.name=='nt' else 0)
    capR = cv2.VideoCapture(RIGHT_CAM, cv2.CAP_DSHOW if os.name=='nt' else 0)
    for c in (capL, capR):
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        c.set(cv2.CAP_PROP_FPS, 30)
    if not (capL.isOpened() and capR.isOpened()):
        print("[Error] 카메라 오픈 실패"); return

    # 5) 포즈/터치
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=0)
    touch = TouchCounter(threshold_frames=TOUCH_THRESHOLD, cooldown_sec=ADV_COOLDOWN)

    # ===== (A) 초기 센터 & 레이저 OFF & BEFORE 프레임 캡처 =====
    try:
        ctl.center()
        ctl.laser_off()
    except Exception as e:
        print(f"[Laser] 초기화 중 예외(무시): {e}")
    cur_yaw, cur_pitch = 90.0, 90.0  # (90,90) 원점으로 간주
    time.sleep(0.25)

    okL0, fL0 = capL.read(); okR0, fR0 = capR.read()
    if not (okL0 and okR0):
        print("[Error] BEFORE 프레임 캡처 실패"); return
    beforeL = rectify(fL0, map1x, map1y, size)
    beforeR = rectify(fR0, map2x, map2y, size)
    print("[LaserDiff] BEFORE frame captured (laser OFF).")

    # ===== (B) YOLO 10프레임 병합 (레이저 OFF 상태로 유지) =====
    print("[Info] 초기 YOLO 병합 중...(레이저 OFF 유지)")
    holdsL_list, holdsR_list = _yolo_once_merge(model, capL, capR, map1x, map1y, map2x, map2y, infer_hw, SELECTED)
    idxL = {h[0]: {"hold_index":h[0], "center":(h[1],h[2]), "color":BGR.get(h[3],(0,255,255)), "contour":np.array(h[4],dtype=np.int32)} for h in holdsL_list}
    idxR = {h[0]: {"hold_index":h[0], "center":(h[1],h[2]), "color":BGR.get(h[3],(0,255,255)), "contour":np.array(h[4],dtype=np.int32)} for h in holdsR_list}

    # 3D/Δ 테이블
    matched_results, by_id, route_ids, next_id_map, delta_from_id, angle_deltas = \
        compute_3d_and_deltas(idxL, idxR, P1, P2, O, csv_rows)
    if not route_ids:
        print("[Warn] route_ids 비어 있음(공통 ID 없음)."); return

    # ===== (C) 레이저 ON & AFTER 프레임 캡처 & 레이저 3D/각도 계산 =====
    yaw_laser = pitch_laser = None
    try:
        ctl.laser_on()
        print("[Laser] ON (for initial diff)")
        time.sleep(0.30)
    except Exception as e:
        print(f"[Laser] ON 실패(무시): {e}")

    okL1, fL1 = capL.read(); okR1, fR1 = capR.read()
    if not (okL1 and okR1):
        print("[Error] AFTER 프레임 캡처 실패")
        # 그래도 진행은 가능: 아래에서 자동 조준으로 진입
    else:
        afterL = rectify(fL1, map1x, map1y, size)
        afterR = rectify(fR1, map2x, map2y, size)
        ptL, ptR, dbg = find_laser_point_pair(beforeL, beforeR, afterL, afterR)
        print(f"[LaserDiff] ptL={ptL}, ptR={ptR}")

        # 디버그 오버레이 표시 (좌/우)
        try:
            grid_top = np.hstack([dbg["overlayL"], dbg["overlayR"]])
            cv2.imshow("LaserDiff Overlay (L|R)", grid_top)
        except Exception:
            pass

        if ptL is not None and ptR is not None:
            Xlaser = triangulate_xy(P1, P2, ptL, ptR)
            yaw_laser, pitch_laser = yaw_pitch_from_X(Xlaser, O, Y_UP_IS_NEGATIVE)
            print(f"[Laser 3D] X=({Xlaser[0]:.1f}, {Xlaser[1]:.1f}, {Xlaser[2]:.1f}) mm  "
                  f"yaw={yaw_laser:.2f}°, pitch={pitch_laser:.2f}°")
        else:
            print("[LaserDiff] 레이저 점 검출 실패 → 자동조준 경로로 진행")

    # === (D) 5초 대기 → (90,90) 원점에서 첫 홀드로 Δ각 이동 ===
    first_id = route_ids[0] if route_ids else (min(by_id.keys()) if by_id else None)
    if first_id is not None and (first_id in by_id) and (yaw_laser is not None) and (pitch_laser is not None):
        mr0 = by_id[first_id]
        dyaw0   = wrap_deg(mr0["yaw_deg"]   - yaw_laser)
        dpitch0 = wrap_deg(mr0["pitch_deg"] - pitch_laser)
        print(f"[Init-Rel] Laser(90,90 원점)→ID{first_id}  Δyaw={dyaw0:+.2f}°, Δpitch={dpitch0:+.2f}°")
        print("[Init-Rel] 5초 뒤 첫 홀드로 이동합니다...")
        for s in range(5, 0, -1):
            print(f"  {s}...", end="\r", flush=True)
            time.sleep(1.0)
        print("  GO!          ")

        target_yaw   = cur_yaw   + DELTA_YAW_CMD_SIGN   * dyaw0
        target_pitch = cur_pitch + DELTA_PITCH_CMD_SIGN * dpitch0
        print(f"[Init-Rel] cmd(yaw,pitch)=({target_yaw:.2f},{target_pitch:.2f}) 로 이동")
        send_servo_angles(ctl, target_yaw, target_pitch)
        cur_yaw, cur_pitch = target_yaw, target_pitch
        current_target_id  = first_id
    else:
        # 레이저 실패 → 기존 자동조준
        if (first_id is not None) and (first_id in by_id):
            mr0 = by_id[first_id]
            auto_yaw, auto_pitch = mr0["yaw_deg"], mr0["pitch_deg"]
            yaw_cmd0, pitch_cmd0 = to_servo_cmd(auto_yaw, auto_pitch)
        else:
            yaw_cmd0, pitch_cmd0 = 90.0, 90.0

        pitch_arg = getattr(args, "pitch", None)
        yaw_arg   = getattr(args, "yaw",   None)
        if (pitch_arg is not None) and (yaw_arg is not None):
            cur_pitch, cur_yaw = float(pitch_arg), float(yaw_arg)
            print(f"[Init-Point] Using user angles: yaw={cur_yaw:.2f}°, pitch={cur_pitch:.2f}°")
        else:
            cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
            print(f"[Init-Point] Auto to ID{first_id}: yaw={cur_yaw:.2f}°, pitch={cur_pitch:.2f}°")
        send_servo_angles(ctl, cur_yaw, cur_pitch)
        current_target_id = first_id

    auto_advance_enabled = (not args.no_auto_advance)

    # ===== 메인 루프 =====
    filled_ids = set()
    t_prev = time.time()
    last_advanced_time = 0.0

    try:
        while True:
            okL, frameL = capL.read(); okR, frameR = capR.read()
            if not (okL and okR):
                print("[Warn] 프레임 수신 실패"); break

            Lr = rectify(frameL, map1x, map1y, size)
            Rr = rectify(frameR, map2x, map2y, size)

            visL, visR = Lr.copy(), Rr.copy()

            # 현재 타깃 표시
            if (current_target_id is not None) and (current_target_id in idxL):
                h = idxL[current_target_id]
                cv2.drawContours(visL, [h["contour"]], -1, (0,255,255), 3)
                cx, cy = h["center"]
                cv2.putText(visL, f"[TARGET] ID:{current_target_id}", (int(cx)-40, int(cy)-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

            # 완료 홀드 표시(반투명 칠하기)
            for hid, h in idxL.items():
                contour = h["contour"]
                if hid in filled_ids:
                    overlay = visL.copy()
                    cv2.drawContours(overlay, [contour], -1, h["color"], thickness=-1)
                    visL = cv2.addWeighted(overlay, 0.45, visL, 0.55, 0)
                cv2.drawContours(visL, [contour], -1, h["color"], 2)
                cx, cy = h["center"]
                cv2.putText(visL, f"ID:{hid}", (int(cx)-12, int(cy)+24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, h["color"], 2, cv2.LINE_AA)

            # 3D 오버레이
            draw_3d_debug_overlay(visL, matched_results, x=20, y=60, dy=18)

            # 포즈
            coords = pose.process(Lr)
            if coords: draw_pose_points(visL, coords, offset_x=0)

            # 터치 판정 → 다음 타깃 이동
            if coords and (current_target_id is not None) and (current_target_id in idxL):
                tid = current_target_id
                hold = idxL[tid]
                triggered, parts = touch.check(hold["contour"].astype(np.float32), coords, tid, now=time.time())

                if triggered:
                    filled_ids.add(tid)
                    # 마지막 홀드?
                    is_last = (tid not in delta_from_id)
                    if auto_advance_enabled and (not is_last) and ((time.time() - last_advanced_time) > ADV_COOLDOWN):
                        dyaw, dpitch = delta_from_id[tid]
                        target_yaw   = cur_yaw   + DELTA_YAW_CMD_SIGN   * dyaw
                        target_pitch = cur_pitch + DELTA_PITCH_CMD_SIGN * dpitch
                        send_servo_angles(ctl, target_yaw, target_pitch)
                        cur_yaw, cur_pitch = target_yaw, target_pitch
                        current_target_id  = next_id_map[tid]
                        last_advanced_time = time.time()
                    elif is_last:
                        print("[Route] 마지막 홀드 성공 → 레이저 OFF")
                        try: ctl.laser_off()
                        except Exception: pass
                        # break  # 원하면 종료

            # HUD
            t_now = time.time(); fps = 1.0 / max(t_now - t_prev, 1e-6); t_prev = t_now
            cv2.putText(visL, f"{meta['sector']} · {meta['level']} · {meta['color']}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(visL, f"route: {len(filled_ids)}/{len(route_ids)}  (q:quit, r:reset, y:rescan)", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2, cv2.LINE_AA)
            cv2.putText(visL, f"FPS: {fps:.1f}", (10, visL.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)

            update_state(records_count=len(filled_ids),
                         last_record=["target", int(current_target_id) if current_target_id is not None else -1, -1, -1],
                         fps=fps, mode="climb")

            # 웹 버튼
            flags = consume_flags()
            if flags.get("reset"):
                filled_ids.clear(); last_advanced_time = 0.0
                # 초기 보정 유지: 필요하면 레이저 상태 유지
                if (route_ids and (route_ids[0] in by_id)):
                    tid0 = route_ids[0]
                    yaw_abs, pitch_abs = by_id[tid0]["yaw_deg"], by_id[tid0]["pitch_deg"]
                    yaw_cmd0, pitch_cmd0 = to_servo_cmd(yaw_abs, pitch_abs)
                    cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
                    send_servo_angles(ctl, cur_yaw, cur_pitch)
                    current_target_id = tid0
            if flags.get("rescan"):
                holdsL_list, holdsR_list = _yolo_once_merge(model, capL, capR, map1x, map1y, map2x, map2y, infer_hw, SELECTED)
                idxL = {h[0]: {"hold_index":h[0], "center":(h[1],h[2]), "color":BGR.get(h[3],(0,255,255)), "contour":np.array(h[4],dtype=np.int32)} for h in holdsL_list}
                idxR = {h[0]: {"hold_index":h[0], "center":(h[1],h[2]), "color":BGR.get(h[3],(0,255,255)), "contour":np.array(h[4],dtype=np.int32)} for h in holdsR_list}
                matched_results, by_id, route_ids, next_id_map, delta_from_id, angle_deltas = \
                    compute_3d_and_deltas(idxL, idxR, P1, P2, O, csv_rows)
            if flags.get("stop"):
                break

            # 표시
            cv2.imshow("Left (Climb)", visL); cv2.imshow("Right (Climb)", visR)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('r'):
                filled_ids.clear(); last_advanced_time = 0.0
                if (route_ids and (route_ids[0] in by_id)):
                    tid0 = route_ids[0]
                    yaw_abs, pitch_abs = by_id[tid0]["yaw_deg"], by_id[tid0]["pitch_deg"]
                    yaw_cmd0, pitch_cmd0 = to_servo_cmd(yaw_abs, pitch_abs)
                    cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
                    send_servo_angles(ctl, cur_yaw, cur_pitch)
                    current_target_id = tid0
            elif k == ord('y'):
                holdsL_list, holdsR_list = _yolo_once_merge(model, capL, capR, map1x, map1y, map2x, map2y, infer_hw, SELECTED)
                idxL = {h[0]: {"hold_index":h[0], "center":(h[1],h[2]), "color":BGR.get(h[3],(0,255,255)), "contour":np.array(h[4],dtype=np.int32)} for h in holdsL_list}
                idxR = {h[0]: {"hold_index":h[0], "center":(h[1],h[2]), "color":BGR.get(h[3],(0,255,255)), "contour":np.array(h[4],dtype=np.int32)} for h in holdsR_list}
                matched_results, by_id, route_ids, next_id_map, delta_from_id, angle_deltas = \
                    compute_3d_and_deltas(idxL, idxR, P1, P2, O, csv_rows)

    finally:
        try: capL.release(); capR.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        try: pose.close()
        except: pass
        try:
            ctl.laser_off()  # 안전 OFF
        except Exception:
            pass
        try: ctl.close()
        except: pass


if __name__ == "__main__":
    main()
