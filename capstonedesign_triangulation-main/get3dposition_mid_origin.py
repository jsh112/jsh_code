# triangulate_click_midpoint.py  (원점=중앙점 옵션 추가)
import cv2, numpy as np, time

# ========= 사용자 설정 =========
NPZ_PATH    = r"calib_out\20250909_173042\stereo\stereo_params.npz"
CAM1_INDEX  = 1
CAM2_INDEX  = 2
WINDOW_NAME = "Rectified: cam1 | cam2  (Click L then R; r:reset, o:toggle origin, q:quit)"
SHOW_GRID   = True
ORIGIN_MODE = "mid"   # "cam1" 또는 "mid" 중 선택 (기본: 중앙점 원점)
# ==============================

# 추가해야할 것
# 카메라 좌, 우 클릭하면 나오는 좌표 txt 파일로 각각 저장하기
S = np.load(NPZ_PATH, allow_pickle=True)
K1, D1 = S["K1"], S["D1"]
K2, D2 = S["K2"], S["D2"]
R1, R2 = S["R1"], S["R2"]
P1, P2 = S["P1"], S["P2"]
Q      = S["Q"]
W, H   = [int(x) for x in S["image_size"]]  # (w,h)

map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)

# OpenCV: P2[0,3] = -f * Tx  →  Tx = -P2[0,3]/f
Tx = -P2[0,3] / P2[0,0]
B  = float(abs(Tx))           # 베이스라인 길이(mm)
M  = np.array([0.5*Tx, 0.0, 0.0], dtype=np.float64)  # 중앙점( cam1좌표계 )

cam1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(CAM2_INDEX, cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, W); cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, W); cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cam1.isOpened() or not cam2.isOpened():
    raise SystemExit("카메라를 열 수 없습니다.")

ptL = None
ptR = None

def on_mouse(event, x, y, flags, userdata):
    global ptL, ptR
    if event != cv2.EVENT_LBUTTONDOWN: return
    w = userdata["w"]
    if x < w:  ptL = (x, y)
    else:      ptR = (x - w, y)

def triangulate_point(ptL, ptR):
    xl = np.array(ptL, dtype=np.float64).reshape(2,1)
    xr = np.array(ptR, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, xl, xr)  # 4x1
    X  = (Xh[:3] / Xh[3]).reshape(3)            # cam1 원점 좌표
    return X

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse, {"w": W})

last_info = None
print(f"[Info] loaded {NPZ_PATH}, image_size={(W,H)}, baseline={B:.1f}mm, origin={ORIGIN_MODE}")

# NEW: 키 입력 디바운스(자동반복 방지) – 'o' 토글이 0.2초에 한 번만 반응하도록
last_toggle_ts = 0.0
TOGGLE_DEBOUNCE_SEC = 0.2

while True:
    ok1, f1 = cam1.read(); ok2, f2 = cam2.read()
    if not (ok1 and ok2): break
    if (f1.shape[1], f1.shape[0]) != (W,H): f1 = cv2.resize(f1, (W,H))
    if (f2.shape[1], f2.shape[0]) != (W,H): f2 = cv2.resize(f2, (W,H))

    Lr = cv2.remap(f1, map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(f2, map2x, map2y, cv2.INTER_LINEAR)
    vis = np.hstack([Lr, Rr])

    if SHOW_GRID:
        step = max(20, H//20)
        for y in range(0, H, step):
            cv2.line(vis, (0,y), (vis.shape[1]-1,y), (0,255,0), 1, cv2.LINE_AA)

    if ptL is not None:
        cv2.drawMarker(vis, ptL, (0,255,255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    if ptR is not None:
        cv2.drawMarker(vis, (ptR[0]+W, ptR[1]), (0,255,255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

    if ptL is not None and ptR is not None:
        X_cam1 = triangulate_point(ptL, ptR)
        # --- 원점 선택 ---
        if ORIGIN_MODE == "mid":
            Xo = X_cam1 - M          # 중앙점을 원점으로 이동
            origin_label = "midpoint"
        else:
            Xo = X_cam1              # cam1 원점(기존)
            origin_label = "cam1"

        # 선(베이스라인)의 최단거리: YZ 평면 거리(레티파이 좌표에선 baseline이 x축)
        d_line = float(np.hypot(Xo[1], Xo[2]))
        # 원점(선택)에 대한 벡터 크기
        d_origin = float(np.linalg.norm(Xo))

        text = f"origin={origin_label} | X=({Xo[0]:.1f},{Xo[1]:.1f},{Xo[2]:.1f})mm | |X|={d_origin:.1f}mm | dist(line)={d_line:.1f}mm"
        last_info = text
        cv2.putText(vis, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)
    elif last_info is not None:
        cv2.putText(vis, last_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, last_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)

    # 상태표시(카메라 위치; ORIGIN_MODE가 mid일 때 직관적)
    cam1_pos = -M   # mid 원점 기준 cam1 좌표
    cam2_pos = +M   # mid 원점 기준 cam2 좌표
    info2 = f"cam1={tuple(v.round(1) for v in cam1_pos)} mm, cam2={tuple(v.round(1) for v in cam2_pos)} mm (mid-frame)"
    cv2.putText(vis, info2, (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, info2, (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,200), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, vis)
    k = (cv2.waitKeyEx(1) & 0xFF)
    if    k == ord('q'):
        break
    elif  k == ord('r'):
        ptL = ptR = last_info = None
    elif  k == ord('o'):
        # NEW: 디바운스 + 토글 메시지 (무한 출력/자동반복 방지)
        now = time.time()
        if now - last_toggle_ts > TOGGLE_DEBOUNCE_SEC:
            ORIGIN_MODE = "cam1" if ORIGIN_MODE=="mid" else "mid"
            print(f"[toggle] origin -> {ORIGIN_MODE}")
            last_toggle_ts = now

cam1.release(); cam2.release(); cv2.destroyAllWindows()
