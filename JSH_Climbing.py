import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv

# ================== 사용자 환경 ==================
NPZ_PATH = r"/home/jojang/Desktop/climbing/stereo_params_scaled.npz"
MODEL_PATH = r"/home/jojang/Desktop/climbing/best_5.pt"

CAM1_INDEX = 0
CAM2_INDEX = 1

SWAP_INPUT     = False   # 입력 좌/우 스왑
SWAP_DISPLAY   = False   # 화면 표시 좌/우 스왑

CSV_GRIPS_PATH = "grip_records.csv"

COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}
ROW_TOL_Y      = 30
THRESH_MASK    = 0.7

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 1.85
LASER_OFFSET_CM_UP   = 8.0
LASER_OFFSET_CM_FWD  = -3.3
Y_UP_IS_NEGATIVE     = True  # 위 방향이 -y인 좌표계면 True

# ================== 유틸 함수 ==================
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
    W,H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_V4L2)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_V4L2)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("카메라 오픈 실패")
    return cap1, cap2

def rectify(frame, mx, my, size):
    W,H = size
    if (frame.shape[1], frame.shape[0]) != (W,H):
        frame = cv2.resize(frame, (W,H))
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)

def extract_holds(frame, model, mask_thresh=0.7, row_tol=50):
    res = model(frame)[0]
    holds = []
    if res.masks is None: return []
    masks = res.masks.data; boxes = res.boxes; names = model.names
    h, w = frame.shape[:2]
    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea)
        cls_id = int(boxes.cls[i].item())
        class_name = names[cls_id]
        M = cv2.moments(contour)
        if M["m00"]==0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        holds.append({
            "class_name": class_name,
            "color": COLOR_MAP.get(class_name,(255,255,255)),
            "contour": contour,
            "center": (cx,cy),
            "conf": float(boxes.conf[i]) if hasattr(boxes,'conf') else 0.0
        })
    # row 정렬
    if not holds: return []
    enriched = [{"cx":h_["center"][0],"cy":h_["center"][1],**h_} for h_ in holds]
    enriched.sort(key=lambda h: h["cy"])
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol: cur.append(h_)
        else: rows.append(cur); cur = [h_]
    rows.append(cur)
    final = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])
        final.extend(row)
    return final

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
                if (dx*dx + dy*dy)**0.5 <= merge_dist_px:
                    area_h = cv2.contourArea(h["contour"])
                    area_m = cv2.contourArea(m["contour"])
                    if (area_h > area_m) or (abs(area_h - area_m)<1e-6 and h.get("conf",0)>m.get("conf",0)):
                        m.update(h)
                    assigned = True
                    break
            if not assigned:
                merged.append(h)
    # 인덱스 부여
    for idx,h in enumerate(merged):
        h["hold_index"] = idx
    return merged

def triangulate_xy(P1,P2,ptL,ptR):
    xl = np.array(ptL,dtype=np.float64).reshape(2,1)
    xr = np.array(ptR,dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1,P2,xl,xr)
    X = (Xh[:3]/Xh[3]).reshape(3)
    return X

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

def yaw_pitch_from_X(X, O, y_up_is_negative=True):
    v = X - O
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw   = np.degrees(np.arctan2(vx, vz))
    pitch = np.degrees(np.arctan2((-vy if y_up_is_negative else vy), np.hypot(vx, vz)))
    return yaw, pitch

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

def wrap_deg(d): return (d + 180.0) % 360.0 - 180.0

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

def angle_between(v1, v2):
    a = np.linalg.norm(v1); b = np.linalg.norm(v2)
    if a == 0 or b == 0: return 0.0
    cosang = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


# ================== 메인 ==================
def main():
    if not Path(NPZ_PATH).exists() or not Path(MODEL_PATH).exists():
        raise FileNotFoundError("NPZ 또는 모델 파일 없음")
    
    # 스테레오 로드
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
    W,H = size
    cap1, cap2 = open_cams(CAM1_INDEX, CAM2_INDEX, size)
    model = YOLO(str(MODEL_PATH))

    # 초기 프레임
    ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
    Lr = rectify(f1, map1x, map1y, size)
    Rr = rectify(f2, map2x, map2y, size)

    holdsL = extract_holds_with_indices(Lr, model, selected_class_name="green")
    holdsR = extract_holds_with_indices(Rr, model, selected_class_name="green")

    # 픽셀 거리 기반 좌/우 매칭
    matched_pairs = []
    max_dist_px = 20
    row_tol = 10  # y 좌표 허용 오차
    for Lh in holdsL:
        best_R = None
        best_dist = max_dist_px
        for Rh in holdsR:
            if abs(Lh["center"][1] - Rh["center"][1]) > row_tol:
                continue  # 같은 row가 아니면 후보 제외
            dist = np.linalg.norm(np.array(Lh["center"]) - np.array(Rh["center"]))
            if dist < best_dist:
                best_dist = dist
                best_R = Rh
        if best_R is not None:
            matched_pairs.append((Lh, best_R))
    # 3D 좌표 계산
    for Lh, Rh in matched_pairs:
        X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
        print(f"Matched {Lh['class_name']} at 3D {X}")

if __name__=="__main__":
    main()
