import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import cv2
import argparse

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

# ====== 스테레오 로드 ======
def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H))

def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_V4L2)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_V4L2)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("카메라 오픈 실패")
    return cap1, cap2

def rectify(frame, mx, my, size):
    W, H = size
    if (frame.shape[1], frame.shape[0]) != (W, H):
        frame = cv2.resize(frame, (W, H))
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
        holds.append({"class_name": class_name, "color": COLOR_MAP.get(class_name,(255,255,255)),
                      "contour": contour, "center": (cx,cy)})
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
    for idx,h_ in enumerate(final): h_["hold_index"]=idx
    return final

def merge_holds(Lh,Rh,merge_dist_px=18):
    merged = []
    for h in Lh+Rh:
        h = {k:v for k,v in h.items()}
        h.pop("hold_index",None)
        assigned = False
        for m in merged:
            dx = h["center"][0]-m["center"][0]; dy = h["center"][1]-m["center"][1]
            if (dx*dx+dy*dy)**0.5 <= merge_dist_px:
                assigned=True; break
        if not assigned: merged.append(h)
    return merged

def triangulate_xy(P1,P2,ptL,ptR):
    xl = np.array(ptL,dtype=np.float64).reshape(2,1)
    xr = np.array(ptR,dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1,P2,xl,xr)
    X = (Xh[:3]/Xh[3]).reshape(3)
    return X

def save_holds_3d_to_csv(holdsL, holdsR, P1, P2, csv_path):
    """좌/우 홀드 매칭 후 3D 좌표 계산하고 CSV 저장"""
    rows = []
    for hL in holdsL:
        hid = hL["hold_index"]
        hR = next((x for x in holdsR if x["hold_index"] == hid), None)
        if hR:
            X = triangulate_xy(P1, P2, hL["center"], hR["center"])
            row = {
                "hold_id": hid,
                "x_mm": X[0],
                "y_mm": X[1],
                "z_mm": X[2],
                "color": hL["class_name"]
            }
            rows.append(row)

    # CSV 저장
    fieldnames = ["hold_id", "x_mm", "y_mm", "z_mm", "color"]
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows

# ====== 메인 ======
def main():
    # 카메라 관련
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists(): raise FileNotFoundError(f"{p} 없음")
    map1x,map1y,map2x,map2y,P1,P2,size = load_stereo(NPZ_PATH)
    W,H = size
    cap1,cap2 = open_cams(CAM1_INDEX,CAM2_INDEX,size)
    model = YOLO(str(MODEL_PATH))
    cv2.namedWindow("Stereo YOLO 3D", cv2.WINDOW_NORMAL)

    rows = []

    while True:
        ok1,f1 = cap1.read(); ok2,f2 = cap2.read()
        if not (ok1 and ok2): break

        Lr = rectify(f1,map1x,map1y,size)
        Rr = rectify(f2,map2x,map2y,size)
        holdsL = extract_holds(Lr,model)
        holdsR = extract_holds(Rr,model)

        # 화면 표시
        vis = np.hstack([Lr,Rr])
        for side, holds in (("L",holdsL),("R",holdsR)):
            xoff = 0 if side=="L" else W
            for h in holds:
                cnt_shifted = h["contour"] + np.array([[[xoff,0]]],dtype=h["contour"].dtype)
                cv2.drawContours(vis,[cnt_shifted],-1,h["color"],2)
                cx,cy = h["center"]
                cv2.circle(vis,(cx+xoff,cy),4,(255,255,255),-1)

        # 3D 좌표 계산 + 화면에 표시 + CSV 저장
        for hL in holdsL:
            hid = hL["hold_index"]
            hR = next((x for x in holdsR if x["hold_index"] == hid), None)
            if hR:
                X = triangulate_xy(P1, P2, hL["center"], hR["center"])
                # 터미널 출력
                print(f"ID {hid}  X={X[0]:.1f}, Y={X[1]:.1f}, Z={X[2]:.1f} mm  Color={hL['class_name']}")
                # 화면 표시
                cx, cy = hL["center"]
                cv2.putText(vis, f"ID{hid}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                # CSV 기록용
                rows.append({
                    "hold_id": hid,
                    "x_mm": X[0],
                    "y_mm": X[1],
                    "z_mm": X[2],
                    "color": hL["class_name"]
                })
        # CSV 저장

    fieldnames = ["hold_id", "x_mm", "y_mm", "z_mm", "color"]
    with open(CSV_GRIPS_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장 완료: {CSV_GRIPS_PATH}")
    cap1.release(); cap2.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
