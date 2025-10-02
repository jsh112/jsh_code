#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo YOLOv8n-Seg (first 10 frames merged) + MediaPipe-on-Left → Live overlay
+ Laser-origin yaw/pitch per hold (LEFT-camera-based)
+ ✅ MediaPipe 모듈(import from A_Mediapipe)
+ ✅ (NEW) 웹 기반 색상 선택 지원 (A_web.py 없는 경우 콘솔 입력)
+ ✅ (NEW) '잡고 있을 때만' 꽉 채움 (놓으면 즉시 해제)
+ ✅ (NEW) 선택한 색상 이름으로 CSV 분리 저장: grip_records_<color>.csv (전체=all)

사전 설치: pip install ultralytics opencv-python mediapipe flask(선택)
필요 파일(같은 폴더): A_Mediapipe.py, (선택) A_web.py
"""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import argparse

# === MediaPipe 모듈 ===
from Save_Mediapipe import PoseTracker, draw_pose_points
# (참고) 기존 TouchCounter는 사용하지 않고 프레임별 streak 직접 관리

# === (NEW) 웹 모듈 - 색상 선택 ===
_USE_WEB = True
try:
    from A_web import choose_color_via_web
except Exception:
    _USE_WEB = False
    def choose_color_via_web(*a, **k):
        raise RuntimeError("A_web 모듈이 로드되지 않았습니다.")

# ========= 사용자 환경 경로 =========
NPZ_PATH       = r"C:\Users\PC\Desktop\Segmentation_Hold\stereo_params_scaled.npz"
MODEL_PATH     = r"C:\Users\PC\Desktop\Segmentation_Hold\best_5.pt"

CAM1_INDEX     = 1   # 왼쪽 카메라
CAM2_INDEX     = 2   # 오른쪽 카메라

SWAP_INPUT     = False   # 입력 좌/우 스왑
SWAP_DISPLAY   = False   # 창 표시 좌/우 스왑

WINDOW_NAME    = "Rectified L | R  (10f merged; MP Left; LIVE Fill)"
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30
SELECTED_COLOR = None    # 예: 'orange' (None=전체)

# 연속 프레임 터치 기준 (잡았다 판정)
TOUCH_THRESHOLD = 10     # 연속 프레임 임계

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

def ask_color_and_map_to_class(all_colors_dict):
    """콘솔에서 색상 라벨을 받아 (모델 클래스명, 파일명용 라벨) 반환"""
    print("가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("필터할 색상 입력(엔터=전체): ").strip().lower()
    if not s:
        print("→ 전체 표시 사용")
        return None, "all"
    mapped = all_colors_dict.get(s)
    if mapped is None:
        print(f"입력 '{s}' 은(는) 유효하지 않은 색입니다. 전체 표시 사용")
        return None, "all"
    print(f"선택된 클래스명: {mapped}")
    return mapped, s  # (모델 클래스명, 사람 라벨)

def _sanitize_label(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in ("_", "-"))

def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    M  = np.array([0.5*Tx, 0.0, 0.0], dtype=np.float64)
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H), B, M)

def open_cams(idx1, idx2, size):
    W, H = size
    # Windows면 CAP_DSHOW, Linux면 CAP_V4L2 권장
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_DSHOW)
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
    if res.masks is None:
        return []
    masks = res.masks.data; boxes = res.boxes; names = model.names
    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        cls_id = int(boxes.cls[i].item()); conf = float(boxes.conf[i].item())
        class_name = names[cls_id]
        if (selected_class_name is not None) and (class_name != selected_class_name):
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        holds.append({"class_name": class_name, "color": COLOR_MAP.get(class_name,(255,255,255)),
                      "contour": contour, "center": (cx, cy), "conf": conf})
    if not holds:
        return []
    # y-행, x-정렬 → hold_index 부여
    enriched = [{"cx": h_["center"][0], "cy": h_["center"][1], **h_} for h_ in holds]
    enriched.sort(key=lambda h: h["cy"])
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol:
            cur.append(h_)
        else:
            rows.append(cur); cur = [h_]
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
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol:
            cur.append(h_)
        else:
            rows.append(cur); cur = [h_]
    rows.append(cur)
    final_sorted = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])
        final_sorted.extend(row)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted

def imshow_scaled(win, img, maxw=None):
    if not maxw:
        cv2.imshow(win, img); return
    h, w = img.shape[:2]
    if w > maxw:
        s = maxw / w
        img = cv2.resize(img, (int(w*s), int(h*s)))
    cv2.imshow(win, img)

def xoff_for(side, W, swap):
    return (W if swap else 0) if side=="L" else (0 if swap else W)

# ---------- 메인 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_web", action="store_true", help="웹 색상 선택 비활성화(콘솔 입력)")
    args = ap.parse_args()

    # 경로 검증
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    # 스테레오 로드
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

    # ================= 색상 필터 선택 =================
    selected_class_name = None
    selected_color_label = "all"   # 파일명용 사람 라벨

    # 1) 웹 선택
    if (not args.no_web) and _USE_WEB:
        try:
            chosen = choose_color_via_web(
                all_colors=list(ALL_COLORS.keys()),
                defaults={}
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

    # 2) 고정 설정
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

    # 3) 콘솔 입력
    if selected_class_name is None and (args.no_web or not _USE_WEB):
        selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)

    # CSV 파일명(색상별 분리)
    csv_label = _sanitize_label(selected_color_label) if selected_color_label else "all"
    CSV_GRIPS_PATH = f"grip_records_{csv_label}.csv"
    print(f"[Info] 그립 CSV 파일: {CSV_GRIPS_PATH}")

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

    for k in range(10):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡쳐 실패")
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

    # ==== MediaPipe Pose ====
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)

    # 기록 상태
    grip_records = []
    already_logged = set()  # ("part", hold_id) 임계 최초 도달만 기록
    touch_streak = {}       # dict: (part_name, hold_id) -> int (연속 프레임 수)

    # 화면
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()

    try:
        while True:
            ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
            if not (ok1 and ok2):
                print("[Warn] 프레임 캡쳐 실패"); break

            Lr = rectify(f1, map1x, map1y, size)
            Rr = rectify(f2, map2x, map2y, size)
            vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])

            # === 프레임 단위 라이브 채움 집합 ===
            live_filled_ids = set()

            # === MediaPipe 포즈 추정 & 표시 ===
            coords = pose.process(Lr)
            left_xoff = xoff_for("L", W, SWAP_DISPLAY)
            draw_pose_points(vis, coords, offset_x=left_xoff)

            # === 라이브 터치 판정 ===
            if coords:
                # 이번 프레임 접촉 목록
                current_touched = set()  # {(part_name, hold_id)}
                for part_name, (px, py) in coords.items():
                    for hid, hold in idxL.items():
                        if cv2.pointPolygonTest(hold["contour"], (px, py), False) >= 0:
                            current_touched.add((part_name, hid))

                # streak 갱신 + 라이브 채움 + CSV 한 번 기록
                for key in current_touched:
                    touch_streak[key] = touch_streak.get(key, 0) + 1
                    part_name, hid = key
                    if touch_streak[key] >= TOUCH_THRESHOLD:
                        live_filled_ids.add(hid)
                        if key not in already_logged:
                            cx, cy = idxL[hid]["center"]
                            grip_records.append([part_name, hid, cx, cy])
                            already_logged.add(key)

                # 이번 프레임에 닿지 않은 키는 0으로 리셋 → 접촉 끊기면 즉시 해제
                for key in list(touch_streak.keys()):
                    if key not in current_touched:
                        touch_streak[key] = 0
            else:
                # 포즈가 아예 안 잡히면 모두 해제
                for key in list(touch_streak.keys()):
                    touch_streak[key] = 0

            # === 드로잉: 라이브 접촉 중인 홀드만 꽉 채움 ===
            for side, holds in (("L", holdsL), ("R", holdsR)):
                xoff = xoff_for(side, W, SWAP_DISPLAY)
                for h in holds:
                    cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)
                    cx, cy = h["center"]

                    if h["hold_index"] in live_filled_ids:
                        cv2.drawContours(vis, [cnt_shifted], -1, h["color"], thickness=cv2.FILLED)

                    # 외곽선/라벨(항상)
                    cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                    cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                    tag = f"ID:{h['hold_index']}"
                    cv2.putText(vis, tag, (cx + xoff - 10, cy + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, tag, (cx + xoff - 10, cy + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
            cv2.putText(vis, f"FPS: {fps:.1f}",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"FPS: {fps:.1f}",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            imshow_scaled(WINDOW_NAME, vis, None)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    finally:
        cap1.release(); cap2.release()
        cv2.destroyAllWindows()
        try: pose.close()
        except: pass

    # 그립 기록 CSV 저장 (색상별 파일명)
    with open(CSV_GRIPS_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["part", "hold_id", "cx", "cy"])
        writer.writerows(grip_records)
    print(f"[Info] 그립 CSV 저장 완료: {CSV_GRIPS_PATH} (총 {len(grip_records)}개)")

if __name__ == "__main__":
    main()
