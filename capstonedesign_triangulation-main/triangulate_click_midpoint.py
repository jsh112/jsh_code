"""
실시간 클릭 삼각측량 + 베이스라인 중앙점까지 거리(mm)

- 좌/우 카메라 프레임을 보정/레티파이해서 좌우로 띄웁니다.
- 마우스로 '왼쪽 창에서 1번 클릭(점 A_L)', '오른쪽 창에서 1번 클릭(점 A_R)' 하면
  해당 대응점으로 삼각측량 → 3D 좌표 X(mm)를 구하고,
  두 카메라 중심(C1,C2)의 '중앙점 M' 까지의 거리를 계산/표시합니다.

※ 반드시 이 보정 결과와 같은 해상도/포커스 상태에서 사용하세요.
"""

import cv2
import numpy as np
import time

# ========= 사용자 설정 =========
NPZ_PATH    = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\calib_out\20250915_145057\stereo\stereo_params_scaled.npz"  # 사용할 stereo_params 경로.
CAM1_INDEX  = 1     # cam1 장치 인덱스
CAM2_INDEX  = 2     # cam2 장치 인덱스
WINDOW_NAME = "Rectified: cam1 | cam2  (Click L then R; r:reset, q:quit)"
SHOW_GRID   = True  # 수평 보조선
# ==============================


# ---------- 로드 & 준비 ----------
S = np.load(NPZ_PATH, allow_pickle=True)
K1, D1 = S["K1"], S["D1"]
K2, D2 = S["K2"], S["D2"]
R1, R2 = S["R1"], S["R2"]
P1, P2 = S["P1"], S["P2"]
Q      = S["Q"]
W, H   = [int(x) for x in S["image_size"]]  # (w,h)

# rectification 맵
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)

# 베이스라인 길이/부호(레티파이 좌표계)
# OpenCV에서 P2[0,3] = -f * Tx  → Tx = -P2[0,3]/f
Tx = -P2[0,3] / P2[0,0]
B  = float(abs(Tx))          # 베이스라인 길이(mm)
Mx = 0.5 * Tx                # 중앙점 M = (Tx/2, 0, 0)  (cam1-rectified 좌표계)
M  = np.array([Mx, 0.0, 0.0], dtype=np.float64)

# ---------- 캡처 ----------
cam1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(CAM2_INDEX, cv2.CAP_DSHOW)
# 해상도 시도(드라이버가 안 받으면 밑에서 resize로 맞춤)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not cam1.isOpened() or not cam2.isOpened():
    raise SystemExit("카메라를 열 수 없습니다. 인덱스/연결을 확인하세요.")

# 클릭 상태 관리
ptL = None  # (x,y) on rectified left (cam1)
ptR = None  # (x,y) on rectified right (cam2)

def on_mouse(event, x, y, flags, userdata):
    global ptL, ptR
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    w = userdata["w"]
    if x < w:
        # 왼쪽 창 클릭 → cam1
        ptL = (x, y)
        print(f"[click] Left: {ptL}")
    else:
        # 오른쪽 창 클릭 → cam2
        ptR = (x - w, y)
        print(f"[click] Right: {ptR}")

def triangulate_point(ptL, ptR):
    """rectified 픽셀 좌표(각각 cam1, cam2) → 3D(mm)"""
    xl = np.array(ptL, dtype=np.float64).reshape(2,1)
    xr = np.array(ptR, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, xl, xr)  # 4x1
    # cam1-rectified 좌표계의 원점(=cam1 중심)
    X  = (Xh[:3] / Xh[3]).reshape(3)            # [X,Y,Z] in cam1-rectified coordinates
    return X

def draw_cross(img, p, color=(0,255,255)):
    if p is None: return
    x,y = int(p[0]), int(p[1])
    cv2.drawMarker(img, (x,y), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

print(f"[Info] Loaded: {NPZ_PATH}")
print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")
print("[Guide] 왼쪽 창에서 1번, 오른쪽 창에서 1번 클릭하세요. (r: reset, q: quit)")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse, {"w": W})

last_info = None  # 최근 결과 문자열

while True:
    ok1, f1 = cam1.read()
    ok2, f2 = cam2.read()
    if not (ok1 and ok2):
        print("[warn] 프레임 읽기 실패"); break

    # 해상도 보정(필요 시)
    if (f1.shape[1], f1.shape[0]) != (W,H):
        f1 = cv2.resize(f1, (W,H))
    if (f2.shape[1], f2.shape[0]) != (W,H):
        f2 = cv2.resize(f2, (W,H))

    # 레티파이 적용
    Lr = cv2.remap(f1, map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(f2, map2x, map2y, cv2.INTER_LINEAR)

    # UI 합치기
    vis = np.hstack([Lr, Rr])

    # 수평 보조선
    if SHOW_GRID:
        h = Lr.shape[0]
        step = max(20, h//20)
        for y in range(0, h, step):
            cv2.line(vis, (0,y), (vis.shape[1]-1, y), (0,255,0), 1, cv2.LINE_AA)

    # 클릭 마커
    draw_cross(vis, ptL, (0,255,255))
    if ptR is not None:
        draw_cross(vis, (ptR[0]+W, ptR[1]), (0,255,255))  # 오른쪽 패널 좌표로 이동

    # 두 점이 모두 있으면 계산
    if ptL is not None and ptR is not None:
        X = triangulate_point(ptL, ptR)        # 3D(mm). cam1 기준 x, y, z 좌표 였던 것. 
        d_mid = float(np.linalg.norm(X - M))   # 중앙점까지 거리
        # (참고) 선까지의 최단거리 = sqrt(Y^2 + Z^2) (레티파이 좌표계에서 baseline==x축)
        d_line = float(np.hypot(X[1], X[2]))

        last_info = f"X=({X[0]:.1f},{X[1]:.1f},{X[2]:.1f})mm | |X-M|={d_mid:.1f}mm | dist(line)={d_line:.1f}mm"
        # 화면에 표시
        cv2.putText(vis, last_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, last_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)

    elif last_info is not None:
        # 최근 값 유지 표시(참고용)
        cv2.putText(vis, last_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, last_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, vis)
    k = cv2.waitKeyEx(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('r'):
        ptL, ptR, last_info = None, None, None
        print("[reset] points cleared")

cam1.release(); cam2.release()
cv2.destroyAllWindows()
