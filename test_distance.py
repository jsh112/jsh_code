import cv2
import numpy as np

# ==== 스테레오 캘리브레이션/맵 불러오기 ====
NPZ_PATH = "/home/jsh/Desktop/JSH_CODE/jsh_code/stereo_params_scaled.npz"

def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W,H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W,H), cv2.CV_32FC1)
    return (map1x, map1y, map2x, map2y, P1, P2, (W,H))

map1x, map1y, map2x, map2y, P1, P2, size = load_stereo(NPZ_PATH)
W, H = size

# ==== 카메라 열기 ====
capL = cv2.VideoCapture(0, cv2.CAP_V4L2)
capR = cv2.VideoCapture(1, cv2.CAP_V4L2)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, W); capL.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, W); capR.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not capL.isOpened() or not capR.isOpened():
    raise SystemExit("카메라 오픈 실패")

# ==== 삼각측량 함수 ====
def triangulate_xy(P1, P2, ptL, ptR):
    xl = np.array(ptL, dtype=np.float64).reshape(2,1)
    xr = np.array(ptR, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, xl, xr)
    X = (Xh[:3] / Xh[3]).reshape(3)
    return X  # mm 단위

# ==== 마우스 클릭 이벤트 ====
click_pts = {"L": None, "R": None}

def mouse_callback(event, x, y, flags, param):
    side = param
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pts[side] = (x, y)
        print(f"[Click] {side}-cam: {x}, {y}")
        
        # 왼쪽/오른쪽 둘 다 찍혔으면 3D 좌표 계산
        if click_pts["L"] and click_pts["R"]:
            X = triangulate_xy(P1, P2, click_pts["L"], click_pts["R"])
            print(f"-> 3D 좌표 (mm): X={X[0]:.2f}, Y={X[1]:.2f}, Z={X[2]:.2f}")
            # 좌표 초기화
            click_pts["L"], click_pts["R"] = None, None

cv2.namedWindow("Left")
cv2.namedWindow("Right")
cv2.setMouseCallback("Left", mouse_callback, "L")
cv2.setMouseCallback("Right", mouse_callback, "R")

# ==== 메인 루프 ====
while True:
    okL, fL = capL.read(); okR, fR = capR.read()
    if not (okL and okR):
        print("프레임 캡쳐 실패"); break

    # Rectify
    fLr = cv2.remap(fL, map1x, map1y, cv2.INTER_LINEAR)
    fRr = cv2.remap(fR, map2x, map2y, cv2.INTER_LINEAR)

    cv2.imshow("Left", fLr)
    cv2.imshow("Right", fRr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release(); capR.release()
cv2.destroyAllWindows()
