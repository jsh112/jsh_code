#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
climbing_by_routing.py — CSV 기반 타깃 추적 & 350ms 접촉 시점 알림 (v1.0, 2025-09-23)

기능 개요
- route_recorder.py가 만든 CSV(순서, 홀드ID, 좌표, 3D 등)를 불러와 '경로'로 사용
- 현재 프레임에서 YOLO 세그먼트를 수행하고, CSV의 (cxL,cyL)과 가장 가까운 홀드를 타깃으로 매칭
- MediaPipe Pose로 손(wrist) 랜드마크가 '타깃 홀드 폴리곤 내부'에 연속 confirm_ms(기본 350ms) 이상 있으면
  그 시점(벽시계 시간 & 경과 시간)을 터미널에 출력 → 정량지표 1 측정용 시작 이벤트
- 시각화: 좌/우 카메라 합성, 타깃 홀드 강조, 손 랜드마크 점 표시, HUD
- 각 타깃 홀드에 대해 CSV의 (X_mm,Y_mm,Z_mm)로 yaw/pitch(도) 계산하여 함께 출력
- (옵션) --serial 지정 시 해당 시점에 yaw/pitch를 시리얼로 전송(프로토콜 예시: "G {yaw} {pitch}\n")

권장 실행 예시
  python climbing_by_routing.py --csv routes/route_20250923_153012_expertA.csv --show_init
  python climbing_by_routing.py --csv routes/route_20250923_153012_expertA.csv --serial COM5 --baud 115200

조작
  n : 다음 홀드로 건너뛰기
  p : 이전 홀드로 이동
  g : 현재 타깃의 yaw/pitch를 즉시 전송(시리얼 사용 시)
  q : 종료
"""

import cv2
import csv
import time
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ====== 환경 기본 경로 (필요시 수정) ======
REPO_ROOT = Path(__file__).resolve().parent
NPZ_PATH   = REPO_ROOT / "calib_out/old_camera_same/stereo/stereo_params_scaled.npz"
MODEL_PATH = REPO_ROOT / "best_6.pt"

# 카메라 기본 인덱스(필요시 인자 덮어쓰기)
CAM1_INDEX = 1  # LEFT
CAM2_INDEX = 2  # RIGHT

# 디스플레이/입력 스왑
SWAP_INPUT   = False
SWAP_DISPLAY = False

WINDOW_NAME = "Climb by Routing (L|R)"
THRESH_MASK = 0.7
ROW_TOL_Y   = 30
MATCH_TOL_PX = 28  # CSV (cxL,cyL)와 현재 프레임 홀드 매칭 허용 거리

# ---- YOLO 클래스 색상 (combined_test/route_recorder와 동일 맵) ----
COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}

# ===== 유틸 =====
def load_stereo(npz_path:Path):
    S = np.load(str(npz_path), allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H), B)

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

def extract_holds(frame_bgr, model, mask_thresh=0.7, row_tol=50):
    """YOLO Seg → 컨투어/센터/색/확률 포함 홀드 목록 리턴"""
    h, w = frame_bgr.shape[:2]
    res = model(frame_bgr)[0]
    holds = []
    if res.masks is None:
        return []
    masks = res.masks.data; boxes = res.boxes; names = model.names
    for i in range(masks.shape[0]):
        cls_id = int(boxes.cls[i].item()); class_name = names[cls_id]
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
    if not holds:
        return []
    # y-row 후 x-sort (안정적 태깅용)
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
    # 인덱스 부여(정보용)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted

def yaw_pitch_from_X(X_mm: float, Y_mm: float, Z_mm: float):
    """LEFT 카메라 원점 기준 yaw/pitch(도). yaw: +좌향, pitch: +위로 가도록 일반적 정의."""
    # yaw = atan2(X, Z), pitch = -atan2(Y, hypot(X,Z))  (화면 위가 +Y라면 기구축에 맞게 부호 조정 가능)
    yaw  = math.degrees(math.atan2(X_mm, max(1e-6, Z_mm)))
    pitch= math.degrees(-math.atan2(Y_mm, math.hypot(X_mm, Z_mm)))
    return yaw, pitch

def read_route_csv(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            try:
                rows.append({
                    "seq": int(row["seq"]),
                    "hold_id": int(row["hold_id"]),
                    "cxL": float(row["cxL"]), "cyL": float(row["cyL"]),
                    "cxR": float(row["cxR"]), "cyR": float(row["cyR"]),
                    "X_mm": float(row["X_mm"]), "Y_mm": float(row["Y_mm"]), "Z_mm": float(row["Z_mm"]),
                    "contact_part": row.get("contact_part",""),
                    "timestamp": float(row["timestamp"]) if "timestamp" in row and row["timestamp"] else None,
                })
            except Exception as e:
                print(f"[Warn] CSV row {i} parse 실패: {e}")
    rows.sort(key=lambda d: d["seq"])
    return rows

def find_match_by_csv_center(holds, csv_cx, csv_cy, tol_px=MATCH_TOL_PX):
    """현재 프레임 홀드들 중 CSV (cxL,cyL)와 가장 가까운 홀드 반환(거리<=tol_px). 없으면 None."""
    best, best_d2 = None, None
    for h in holds:
        cx, cy = h["center"]
        d2 = (cx - csv_cx)**2 + (cy - csv_cy)**2
        if best is None or d2 < best_d2:
            best, best_d2 = h, d2
    if best is None:
        return None
    if best_d2 is None or best_d2 > tol_px**2:
        return None
    return best

def maybe_open_serial(port: str|None, baud: int):
    if not port:
        return None
    try:
        import serial
    except Exception:
        print("[Warn] pyserial 미설치. `pip install pyserial` 후 사용하세요.")
        return None
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
        print(f"[Serial] Opened {port} @ {baud}")
        time.sleep(0.5)
        return ser
    except Exception as e:
        print(f"[Serial] Open 실패: {e}")
        return None

def send_goto(ser, yaw_deg, pitch_deg):
    if ser is None:
        print(f"[GOTO] yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°  (시리얼 미사용)")
        return
    try:
        cmd = f"G {yaw_deg:.2f} {pitch_deg:.2f}\n"
        ser.write(cmd.encode("utf-8"))
        print(f"[Serial] -> {cmd.strip()}")
    except Exception as e:
        print(f"[Serial] 전송 실패: {e}")

# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="route_recorder.py가 생성한 경로 CSV")
    ap.add_argument("--npz",   default=str(NPZ_PATH))
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--left_cam",  type=int, default=None)
    ap.add_argument("--right_cam", type=int, default=None)
    ap.add_argument("--confirm_ms", type=int, default=350, help="연속 접촉 판정 시간(ms)")
    ap.add_argument("--match_px", type=int, default=MATCH_TOL_PX, help="CSV (cxL,cyL)과 현재 홀드 매칭 허용 px")
    ap.add_argument("--show_init", action="store_true", help="초기 프레임에서도 미리보기 표시")
    ap.add_argument("--serial", default=None, help="시리얼 포트 (예: COM5, /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    # 지연 로드
    from ultralytics import YOLO
    import mediapipe as mp

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV가 없습니다: {csv_path}")
    npz_path = Path(args.npz); model_path = Path(args.model)
    if not npz_path.exists():
        raise FileNotFoundError(f"스테레오 파라미터 없음: {npz_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO 모델 없음: {model_path}")

    route = read_route_csv(csv_path)
    if not route:
        raise SystemExit("[Error] CSV에서 유효한 경로를 읽지 못했습니다.")
    N = len(route)
    print(f"[Info] Loaded CSV route: {csv_path.name} (steps={N})")

    map1x, map1y, map2x, map2y, P1, P2, size, B = load_stereo(npz_path)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

    left_idx  = args.left_cam  if args.left_cam  is not None else CAM1_INDEX
    right_idx = args.right_cam if args.right_cam is not None else CAM2_INDEX
    capL_idx, capR_idx = (right_idx, left_idx) if SWAP_INPUT else (left_idx, right_idx)
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)

    model = YOLO(str(model_path))

    # MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    hand_indices = [15, 16]  # wrists

    # 시리얼
    ser = maybe_open_serial(args.serial, args.baud)

    # 초기 프레임 약간 버림
    for _ in range(2):
        cap1.read(); cap2.read()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # 필요시 최상단

    # 상태
    target_i = 0
    contact_ms = 0.0
    inside_prev = False
    t_prev = time.time()
    t_run0 = t_prev
    announced = False  # 현재 타깃에 대해 1회만 알림

    # 초기 YOLO 준비용 샘플
    if args.show_init:
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if ok1 and ok2:
            Lr0 = rectify(f1, map1x, map1y, size)
            Rr0 = rectify(f2, map2x, map2y, size)
            vis0 = np.hstack([Rr0, Lr0]) if SWAP_DISPLAY else np.hstack([Lr0, Rr0])
            cv2.putText(vis0, "Initializing...", (20, 40), 0, 1.0, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis0, "Initializing...", (20, 40), 0, 1.0, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, vis0); cv2.waitKey(1)

    while True:
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            print("[Warn] 프레임 읽기 실패"); break
        Lr = rectify(f1, map1x, map1y, size)
        Rr = rectify(f2, map2x, map2y, size)
        vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])

        # 현재 타깃 CSV 정보
        tgt = route[target_i]
        csv_cx, csv_cy = tgt["cxL"], tgt["cyL"]
        X_mm, Y_mm, Z_mm = tgt["X_mm"], tgt["Y_mm"], tgt["Z_mm"]
        yaw_deg, pitch_deg = yaw_pitch_from_X(X_mm, Y_mm, Z_mm)

        # YOLO 세그 → 현재 프레임 홀드들
        holdsL = extract_holds(Lr, model, THRESH_MASK, ROW_TOL_Y)

        # CSV (cxL,cyL)와 가장 가까운 홀드를 타깃으로 매칭
        matched = find_match_by_csv_center(holdsL, csv_cx, csv_cy, tol_px=args.match_px)

        # MediaPipe (왼쪽)
        result = pose.process(cv2.cvtColor(Lr, cv2.COLOR_BGR2RGB))
        coordsL = {}
        if result.pose_landmarks:
            hL, wL = Lr.shape[:2]
            for idx in hand_indices:
                lm = result.pose_landmarks.landmark[idx]
                x, y = float(lm.x*wL), float(lm.y*hL)
                coordsL[idx] = (x, y)
                # draw landmark
                left_xoff = (W if SWAP_DISPLAY else 0)
                cv2.circle(vis, (int(x)+left_xoff, int(y)), 6, (0,0,255), -1)

        # 타깃 홀드 표시 + 접촉 판정
        inside_now = False
        if matched is not None:
            # 타깃 강조
            xoffL = (W if SWAP_DISPLAY else 0)
            cnt_shifted = matched["contour"] + np.array([[[xoffL, 0]]], dtype=matched["contour"].dtype)
            cv2.drawContours(vis, [cnt_shifted], -1, (0,255,255), 3)  # 타깃=노란색 강조
            cx, cy = matched["center"]
            cv2.circle(vis, (cx+xoffL, cy), 5, (255,255,255), -1)
            tag = f"TGT seq:{tgt['seq']} yaw:{yaw_deg:.1f} pitch:{pitch_deg:.1f}"
            cv2.putText(vis, tag, (cx+xoffL-10, cy-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, tag, (cx+xoffL-10, cy-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            # 손이 폴리곤 내부?
            if coordsL:
                for lm_idx, xy in coordsL.items():
                    if cv2.pointPolygonTest(matched["contour"], xy, False) >= 0:
                        inside_now = True
                        break

        # 시간 누적 (연속 접촉만)
        t_now = time.time()
        dt_ms = (t_now - t_prev) * 1000.0
        t_prev = t_now

        if inside_now:
            # 연속 접촉만 인정
            contact_ms = (contact_ms + dt_ms) if inside_prev else dt_ms
        else:
            contact_ms = 0.0
        inside_prev = inside_now

        # 임계 도달 시, 1회 알림 (정량지표 1 시작 시점)
        if (not announced) and (contact_ms >= args.confirm_ms) and matched is not None:
            wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            t_elapsed_ms = (t_now - t_run0) * 1000.0
            print(
                f"[CONTACT>= {args.confirm_ms}ms] wall={wall}  elapsed={t_elapsed_ms:.0f}ms  "
                f"seq={tgt['seq']}  hold_id={tgt['hold_id']}  "
                f"yaw={yaw_deg:.2f}°  pitch={pitch_deg:.2f}°"
            )
            # (옵션) 이 순간에 바로 GOTO 전송
            send_goto(ser, yaw_deg, pitch_deg)
            announced = True  # 같은 타깃에 대해 중복 알림 방지

        # HUD
        hud1 = (f"Route CSV: {csv_path.name} | step {target_i+1}/{N} "
                f"(seq={tgt['seq']}, id={tgt['hold_id']}) "
                f"yaw={yaw_deg:.1f} pitch={pitch_deg:.1f}")
        hud2 = (f"Contact: {int(contact_ms)} ms / thresh {args.confirm_ms} ms   "
                f"(n=next, p=prev, g=goto, q=quit)")
        cv2.putText(vis, hud1, (20, 28), 0, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, hud1, (20, 28), 0, 0.7, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, hud2, (20, 56), 0, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, hud2, (20, 56), 0, 0.7, (255,255,255), 1, cv2.LINE_AA)

        # 디버그: CSV 타깃 위치도 점으로
        xoffL = (W if SWAP_DISPLAY else 0)
        cv2.circle(vis, (int(csv_cx)+xoffL, int(csv_cy)), 5, (255,0,255), 2)  # 보라: CSV 타깃 중심

        # 화면 출력/키 입력
        cv2.imshow(WINDOW_NAME, vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('n'):
            if target_i < N-1:
                target_i += 1
                contact_ms = 0.0; inside_prev = False; announced = False
                print(f"[Step] → Next: seq={route[target_i]['seq']} (#{target_i+1}/{N})")
            else:
                print("[Info] 마지막 스텝입니다.")
        elif k == ord('p'):
            if target_i > 0:
                target_i -= 1
                contact_ms = 0.0; inside_prev = False; announced = False
                print(f"[Step] ← Prev: seq={route[target_i]['seq']} (#{target_i+1}/{N})")
            else:
                print("[Info] 첫 스텝입니다.")
        elif k == ord('g'):
            # 현재 타깃 각도 전송(수동 트리거)
            send_goto(ser, yaw_deg, pitch_deg)

    # 종료
    cap1.release(); cap2.release(); cv2.destroyAllWindows()
    if ser is not None:
        try: ser.close()
        except: pass

if __name__ == "__main__":
    main()
