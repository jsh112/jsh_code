import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ================== 사용자 환경 ==================
NPZ_PATH = r"/home/jojang/Desktop/climbing/stereo_params_scaled.npz"
MODEL_PATH = r"/home/jojang/Desktop/climbing/best_5.pt"
CAM1_INDEX = 0
CAM2_INDEX = 1
SWAP_INPUT = False
ROW_TOL_Y = 10        # y 좌표 row 허용 오차
MAX_DIST_PX = 20      # 좌/우 matching 최대 거리

COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}

# ================== 유틸 함수 ==================
def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1,D1,R1,P1,(W,H),cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2,D2,R2,P2,(W,H),cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, P1, P2, (W,H)

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

def extract_holds_with_indices(frame_bgr, model, selected_class_name=None, mask_thresh=0.7):
    h,w = frame_bgr.shape[:2]
    res = model(frame_bgr)[0]
    holds = []
    if res.masks is None: return []
    masks = res.masks.data; boxes = res.boxes; names = model.names
    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea)
        cls_id = int(boxes.cls[i].item()); conf = float(boxes.conf[i].item())
        class_name = names[cls_id]
        if selected_class_name and class_name != selected_class_name:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        holds.append({"class_name": class_name, "color": COLOR_MAP.get(class_name,(255,255,255)),
                      "contour": contour, "center": (cx,cy), "conf": conf})
    return holds

def match_holds(holdsL, holdsR, max_dist_px=30, row_tol=30):
    matched_pairs = []
    used_R = set()
    for Lh in holdsL:
        best_R = None
        best_dist = max_dist_px
        for i,Rh in enumerate(holdsR):
            if i in used_R: continue
            if abs(Lh["center"][1]-Rh["center"][1]) > row_tol: continue
            dist = np.linalg.norm(np.array(Lh["center"]) - np.array(Rh["center"]))
            if dist < best_dist:
                best_dist = dist
                best_R = (i,Rh)
        if best_R:
            idx,Rh = best_R
            used_R.add(idx)
            matched_pairs.append((Lh,Rh))
    return matched_pairs

def triangulate_xy(P1,P2,ptL,ptR):
    xl = np.array(ptL,dtype=np.float64).reshape(2,1)
    xr = np.array(ptR,dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1,P2,xl,xr)
    X = (Xh[:3]/Xh[3]).reshape(3)
    return X

def match_holds(holdsL, holdsR, max_dist_px=20, row_tol=10):
    matched_pairs = []
    used_R = set()
    for Lh in holdsL:
        best_R = None
        best_dist = max_dist_px
        for Rh in holdsR:
            if id(Rh) in used_R:
                continue  # 이미 매칭된 R 제외
            if abs(Lh["center"][1] - Rh["center"][1]) > row_tol:
                continue  # 같은 row가 아니면 제외
            dist = np.linalg.norm(np.array(Lh["center"]) - np.array(Rh["center"]))
            if dist < best_dist:
                best_dist = dist
                best_R = Rh
        if best_R is not None:
            matched_pairs.append((Lh, best_R))
            used_R.add(id(best_R))
    return matched_pairs

# ================== 메인 ==================
def main():
    if not Path(NPZ_PATH).exists() or not Path(MODEL_PATH).exists():
        raise FileNotFoundError("NPZ 또는 모델 파일 없음")

    map1x,map1y,map2x,map2y,P1,P2,size = load_stereo(NPZ_PATH)
    W,H = size
    cap1, cap2 = open_cams(CAM1_INDEX,CAM2_INDEX,size)
    model = YOLO(str(MODEL_PATH))

    MAX_DIST_PX = 20
    ROW_TOL_Y = 10

    while True:
        ok1,f1 = cap1.read(); ok2,f2 = cap2.read()
        if not (ok1 and ok2): break

        Lr = rectify(f1,map1x,map1y,size)
        Rr = rectify(f2,map2x,map2y,size)

        holdsL = extract_holds_with_indices(Lr, model, "green")
        holdsR = extract_holds_with_indices(Rr, model, "green")

        print("L holds:", [h["center"] for h in holdsL])
        print("R holds:", [h["center"] for h in holdsR])

        matched_pairs = match_holds(holdsL, holdsR, MAX_DIST_PX, ROW_TOL_Y)
        print("Matched:", len(matched_pairs))

        # 3D 계산 & 화면 표시
        for Lh,Rh in matched_pairs:
            X = triangulate_xy(P1,P2,Lh["center"],Rh["center"])
            cv2.circle(Lr,Lh["center"],5,(0,255,0),-1)
            cv2.circle(Rr,Rh["center"],5,(0,255,0),-1)
            cv2.putText(Lr,f"{X.round(1)}", (Lh["center"][0]+5,Lh["center"][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
            print(f"Matched {Lh['class_name']} at 3D {X}")

        cv2.imshow("Left", Lr)
        cv2.imshow("Right", Rr)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

    cap1.release(); cap2.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
