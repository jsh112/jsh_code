#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo YOLOv8n-Seg (first-frame only) → Live overlay + triangulation (mm)

- 시작 시: 좌/우 첫 프레임만 YOLO 세그 → 컨투어/중심 추출 → y행 정렬 → x정렬 → hold_index 부여
- 공통 hold_index 쌍으로 3D(X,Y,Z, mm) 계산(|X−M|, d_line 포함) → 고정 결과 저장
- 이후: 카메라에서 프레임 계속 읽고(레티파이) → "첫 프레임에서 얻은 컨투어/중심/ID/3D"만 오버레이 표시
"""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========= 사용자 설정 =========
NPZ_PATH       = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\calib_out\20250909_173042\stereo\stereo_params.npz"
MODEL_PATH     = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\best_6.pt"  # YOLO 세그 모델 경로
CAM1_INDEX     = 1
CAM2_INDEX     = 2
WINDOW_NAME    = "Rectified L | R  (YOLO on first frame → live overlay)"
SHOW_GRID      = True
THRESH_MASK    = 0.7     # 세그 마스크 이진화 임계
ROW_TOL_Y      = 30      # 같은 행으로 묶을 y 오차(px)
SELECTED_COLOR = None    # 예: 'orange' (None이면 실행 시 콘솔에서 입력받거나 엔터=전체)
SAVE_VIDEO     = False   # True면 결과를 mp4로 저장
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"
# ==============================

# YOLO 클래스 컬러 (BGR)
COLOR_MAP = {
    'Hold_Red':     (0, 0, 255),
    'Hold_Orange':  (0, 165, 255),
    'Hold_Yellow':  (0, 255, 255),
    'Hold_Green':   (0, 255, 0),
    'Hold_Blue':    (255, 0, 0),
    'Hold_Purple':  (204, 50, 153),
    'Hold_Pink':    (203, 192, 255),
    'Hold_Lime':    (50, 255, 128),
    'Hold_Sky':     (255, 255, 0),
    'Hold_White':   (255, 255, 255),
    'Hold_Black':   (30, 30, 30),
    'Hold_Gray':    (150, 150, 150),
}

# 선택 색상 → YOLO 클래스 이름 매핑
ALL_COLORS = {
    'red': 'Hold_Red',
    'orange': 'Hold_Orange',
    'yellow': 'Hold_Yellow',
    'green': 'Hold_Green',
    'blue': 'Hold_Blue',
    'purple': 'Hold_Purple',
    'pink': 'Hold_Pink',
    'white': 'Hold_White',
    'black': 'Hold_Black',
    'gray': 'Hold_Gray',
    'lime': 'Hold_Lime',
    'sky': 'Hold_Sky',
}

def ask_color_and_map_to_class(all_colors_dict):
    print("🎨 선택 가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("✅ 원하는 홀드 색상 입력(엔터=전체): ").strip().lower()
    if not s:
        print("→ 전체 클래스 사용")
        return None
    mapped = all_colors_dict.get(s)
    if mapped is None:
        print(f"⚠️ '{s}' 는 유효하지 않은 색상입니다. 전체 클래스 사용")
        return None
    print(f"🎯 선택된 클래스: {mapped}")
    return mapped

def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]
    K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]
    P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    # Tx, baseline, 중앙점 M
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    M  = np.array([0.5*Tx, 0.0, 0.0], dtype=np.float64)
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H), B, M, Tx)

def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
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
    """
    YOLO 세그 → (가장 큰 외곽) 컨투어, 중심(cx,cy), 클래스, conf
    → y행 정렬 → 행 내 x정렬 → hold_index 부여
    반환: 리스트(dict): {class_name, color, contour, center, conf, hold_index}
    """
    h, w = frame_bgr.shape[:2]
    res = model(frame_bgr)[0]
    holds = []
    if res.masks is None:
        return []

    masks = res.masks.data
    boxes = res.boxes
    names = model.names

    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        class_name = names[cls_id]

        if (selected_class_name is not None) and (class_name != selected_class_name):
            continue

        Mmom = cv2.moments(contour)
        if Mmom["m00"] == 0:
            continue
        cx = int(Mmom["m10"] / Mmom["m00"])
        cy = int(Mmom["m01"] / Mmom["m00"])

        holds.append({
            "class_name": class_name,
            "color": COLOR_MAP.get(class_name, (255,255,255)),
            "contour": contour,
            "center": (cx, cy),
            "conf": conf,
        })

    if not holds:
        return []

    # ---- 인덱스 부여: y행 정렬 → 행 내 x정렬 ----
    enriched = [{"cx": h_["center"][0], "cy": h_["center"][1], **h_} for h_ in holds]
    enriched.sort(key=lambda h: h["cy"])  # y로 전체 정렬

    rows = []
    cur_row = [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur_row[0]["cy"]) < row_tol:
            cur_row.append(h_)
        else:
            rows.append(cur_row)
            cur_row = [h_]
    rows.append(cur_row)

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
    h, w = img.shape[:2]
    step = max(20, h//20)
    for y in range(0, h, step):
        cv2.line(img, (0,y), (w-1,y), (0,255,0), 1, cv2.LINE_AA)

def main():
    # 경로 검사(선택)
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일이 없습니다: {p}")

    # ---- 준비 ----
    map1x, map1y, map2x, map2y, P1, P2, size, B, M, Tx = load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

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
    cap1, cap2 = open_cams(CAM1_INDEX, CAM2_INDEX, size)
    model = YOLO(str(MODEL_PATH))

    # ---- 첫 프레임 캡처 & 레티파이 & YOLO ----
    ok1, f1 = cap1.read()
    ok2, f2 = cap2.read()
    if not (ok1 and ok2):
        cap1.release(); cap2.release()
        raise SystemExit("첫 프레임 캡처 실패")

    Lr0 = rectify(f1, map1x, map1y, size)
    Rr0 = rectify(f2, map2x, map2y, size)

    holdsL = extract_holds_with_indices(Lr0, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
    holdsR = extract_holds_with_indices(Rr0, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)

    if not holdsL or not holdsR:
        cap1.release(); cap2.release()
        print("[Warn] 한쪽 또는 양쪽에서 홀드가 검출되지 않았습니다.")
        return

    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        print("[Warn] 좌/우 공통 hold_index가 없습니다. (정렬 기준 상이 가능)")
    else:
        print(f"[Info] 매칭된 홀드 쌍 수: {len(common_ids)}")

    # ---- 매칭 결과 사전 계산(3D, 텍스트) ----
    # 결과를 고정시켜 이후 프레임에 계속 그린다.
    matched_results = []  # [{hid, Lcx, Lcy, Rcx, Rcy, color, X, d_mid, d_line}, ...]
    for hid in common_ids:
        L = idxL[hid]
        R = idxR[hid]
        X = triangulate_xy(P1, P2, L["center"], R["center"])
        d_mid  = float(np.linalg.norm(X - M))            # 중앙점 M까지
        d_line = float(np.hypot(X[1], X[2]))             # 베이스라인까지 최단거리
        matched_results.append({
            "hid": hid,
            "Lcx": L["center"][0], "Lcy": L["center"][1],
            "Rcx": R["center"][0], "Rcy": R["center"][1],
            "color": L["color"],  # 좌/우 동일 클래스라고 가정
            "X": X, "d_mid": d_mid, "d_line": d_line
        })

    # 비디오 저장 준비
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))

    # ---- 라이브 루프 (YOLO는 안 돌림, 오버레이만) ----
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    while True:
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            print("[Warn] 프레임 읽기 실패")
            break

        Lr = rectify(f1, map1x, map1y, size)
        Rr = rectify(f2, map2x, map2y, size)
        vis = np.hstack([Lr, Rr])

        if SHOW_GRID:
            draw_grid(vis[:, :W])
            draw_grid(vis[:, W:])

        # 첫 프레임에서 얻은 컨투어/ID/중심을 계속 그린다
        for side, holds in (("L", holdsL), ("R", holdsR)):
            xoff = 0 if side == "L" else W
            for h in holds:
                cnt = h["contour"]
                cnt_shifted = cnt + np.array([[[xoff, 0]]], dtype=cnt.dtype)
                cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                cx, cy = h["center"]
                cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                cv2.putText(vis, f"ID:{h['hold_index']}", (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, f"ID:{h['hold_index']}", (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

        # 매칭된 쌍 보조선 + 3D 텍스트 반복 표기
        y0 = 30
        for mr in matched_results:
            # 중심 연결선
            cv2.line(vis, (mr["Lcx"], mr["Lcy"]), (mr["Rcx"]+W, mr["Rcy"]), (200,200,200), 1, cv2.LINE_AA)
            X = mr["X"]
            txt = (f"ID{mr['hid']}  X=({X[0]:.1f},{X[1]:.1f},{X[2]:.1f})mm  "
                   f"|X-M|={mr['d_mid']:.1f}  d_line={mr['d_line']:.1f}")
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
            y0 += 22

        # FPS 표시
        t_now = time.time()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        cv2.putText(vis, f"FPS: {fps:.1f}  (YOLO first-frame only)", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f}  (YOLO first-frame only)", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, vis)
        if out is not None:
            out.write(vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap1.release(); cap2.release()
    if out is not None:
        out.release()
        print(f"[Info] 저장 완료: {OUT_PATH}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
