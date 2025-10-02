# stereo_yolo_depth_RAWdisplay_firstframe_only.py
# - 화면: 진짜 RAW (왜곡 있는 그대로). => "원본 영상 그대로"
# - YOLO: 첫 프레임에만 RAW 프레임에 실행 (좌/우 각각)
# - 매칭: undistort한 '픽셀 좌표'로 Fundamental epipolar 거리 게이트
# - 깊이: undistort → '정규화 좌표'로 [I|0],[R|t] 삼각측량 (rectify 불필요)
# - 이후 프레임: YOLO 돌리지 않고, 첫 프레임에서 얻은 좌표쌍으로만 깊이 표시

import cv2
import numpy as np
from ultralytics import YOLO
import time

# ====================== 사용자 설정 ======================
NPZ_PATH     = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\calib_out\20250915_104820\stereo\stereo_params_scaled.npz"
CAM1_INDEX   = 1
CAM2_INDEX   = 2
MODEL_PATH   = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\best_6.pt"
CONF_THRES   = 0.5
EPIP_DIST_TH = 1.5      # 에피폴라 거리(px) 허용치 (1~3px 사이 조절)
SIZE_RATIO   = 0.6      # 박스 크기 유사도(0.6 ~ 1/0.6 허용)
DRAW_BOXES   = True
WINDOW_NAME  = "RAW Left | RAW Right  (q:quit)  [YOLO only on first frame]"
# ========================================================

# ---------------- 파라미터 로드 ----------------
S  = np.load(NPZ_PATH, allow_pickle=True)
K2, D2 = S["K1"], S["D1"]
K1, D1 = S["K2"], S["D2"]
R      = S["R"]                      # cam1 -> cam2 회전 (3x3)
T      = S["T"].reshape(3,1)         # cam1 -> cam2 이동 (3x1)  (단위: mm 권장)
W, H   = map(int, S["image_size"])   # (width, height)

BASELINE = float(np.linalg.norm(T))
print(f"[Info] Loaded {NPZ_PATH}")
print(f"[Info] image_size={(W,H)}, baseline≈{BASELINE:.1f} (unit = T's unit)")

# ----------------- Fundamental matrix (undistorted pixel domain) -----------------
def _skew(t):
    tx, ty, tz = t.ravel()
    return np.array([[0, -tz,  ty],
                     [tz,  0, -tx],
                     [-ty, tx,  0]], dtype=np.float64)

E = _skew(T) @ R                                    # Essential
F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)     # Fundamental (3x3)

# ---------- RAW픽셀 → undistorted 픽셀 / 정규화 좌표 변환 ----------
def rawpx_to_undist_px(pt_px, K, D):
    """RAW 픽셀 → undistorted '픽셀' 좌표 (P=K로 재투영)"""
    p = np.array([pt_px], np.float32).reshape(-1,1,2)
    q = cv2.undistortPoints(p, K, D, R=None, P=K)     # undistorted pixel
    return tuple(np.squeeze(q).astype(float))

def rawpx_to_normalized(pt_px, K, D):
    """RAW 픽셀 → undistorted '정규화' 좌표 ((x-cx)/fx,(y-cy)/fy)"""
    p = np.array([pt_px], np.float32).reshape(-1,1,2)
    n = cv2.undistortPoints(p, K, D, R=None, P=None)  # normalized
    return tuple(np.squeeze(n).astype(float))         # (nx, ny), z=1

# ---------------- 에피폴라 거리 (undistorted 픽셀 기준) ----------------
def epipolar_dist_undist_px(ptL_ud_px, ptR_ud_px):
    """ptL_ud_px, ptR_ud_px: undistorted '픽셀' 좌표쌍"""
    x  = np.array([ptL_ud_px[0], ptL_ud_px[1], 1.0], dtype=np.float64)
    xp = np.array([ptR_ud_px[0], ptR_ud_px[1], 1.0], dtype=np.float64)
    lp = F @ x
    a, b, c = lp
    return abs(a * xp[0] + b * xp[1] + c) / max(1e-9, np.hypot(a, b))

# ---------------- YOLO ----------------
model = YOLO(MODEL_PATH)

def detect_boxes_raw(img_raw):
    """
    RAW 프레임에서 YOLO 검출
    반환: [{cls,conf,cx,cy,w,h,x1,y1,x2,y2} ...]  (좌표는 RAW 픽셀)
    """
    out = model.predict(img_raw, conf=CONF_THRES, verbose=False)[0]
    res = []
    if out.boxes is None:
        return res
    for b in out.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = b[:6]
        w, h = x2 - x1, y2 - y1
        res.append({
            "cls": int(cls), "conf": float(conf),
            "cx": float((x1 + x2) / 2.0), "cy": float((y1 + y2) / 2.0),
            "w": float(w), "h": float(h),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        })
    return res

def good_pair_by_epi(L_raw, R_raw):
    """클래스/크기/에피폴라 거리/시차 부호 게이트 (에피폴라는 undistorted 픽셀에서 평가)"""
    if L_raw["cls"] != R_raw["cls"]:
        return False
    rh = (L_raw["h"] + 1e-6) / (R_raw["h"] + 1e-6)
    rw = (L_raw["w"] + 1e-6) / (R_raw["w"] + 1e-6)
    if not (SIZE_RATIO <= rh <= 1.0 / SIZE_RATIO): return False
    if not (SIZE_RATIO <= rw <= 1.0 / SIZE_RATIO): return False
    # RAW → undistorted 픽셀로 변환해 epipolar 거리 판정
    L_ud = rawpx_to_undist_px((L_raw["cx"], L_raw["cy"]), K1, D1)
    R_ud = rawpx_to_undist_px((R_raw["cx"], R_raw["cy"]), K2, D2)
    if epipolar_dist_undist_px(L_ud, R_ud) > EPIP_DIST_TH:
        return False
    # 좌->우 배치 가정: disparity > 0  (undistorted 픽셀 기준으로 판단)
    if (L_ud[0] - R_ud[0]) <= 0:
        return False
    return True

def match_boxes_firstframe(L_list_raw, R_list_raw):
    """첫 프레임용 간단 그리디 매칭 (에피폴라 거리 최소)"""
    pairs, usedR = [], set()
    for i, L in enumerate(L_list_raw):
        cands = []
        for j, Rb in enumerate(R_list_raw):
            if j in usedR: continue
            if not good_pair_by_epi(L, Rb): continue
            L_ud = rawpx_to_undist_px((L["cx"], L["cy"]), K1, D1)
            R_ud = rawpx_to_undist_px((Rb["cx"], Rb["cy"]), K2, D2)
            de = epipolar_dist_undist_px(L_ud, R_ud)
            cands.append((de, j))
        if not cands: continue
        cands.sort(key=lambda x: x[0])
        jbest = cands[0][1]
        pairs.append((i, jbest))
        usedR.add(jbest)
    return pairs

# ---------------- 삼각측량 ----------------
def triangulate_from_RAWpixels_once(ptL_raw_px, ptR_raw_px):
    """
    RAW 픽셀 좌표쌍을 입력받아:
    1) undistort → 정규화 좌표
    2) P1=[I|0], P2=[R|t]로 삼각측량 → cam1 좌표계 3D(단위=T)
    """
    nL = rawpx_to_normalized(ptL_raw_px, K1, D1)   # (nx, ny)
    nR = rawpx_to_normalized(ptR_raw_px, K2, D2)
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = np.hstack([R, T])
    Xh = cv2.triangulatePoints(P1, P2,
                               np.array(nL, dtype=np.float64).reshape(2,1),
                               np.array(nR, dtype=np.float64).reshape(2,1))
    X  = (Xh[:3] / Xh[3]).reshape(3)
    return X

def reprojection_rmse_one(X, K, Rm, tm, pt_raw_px):
    """3D점 X를 카메라로 투영 → RAW 픽셀과 오차 (undistort를 통해 비교)"""
    # undistorted 픽셀 기준으로 비교
    # 1) 이상 카메라 투영(정규화) 후 K 곱
    Xc = (Rm @ X.reshape(3,1) + tm).reshape(3)
    uv_ideal = (K @ (Xc / Xc[2])).ravel()[:2]  # undistorted pixel 예측치
    # 2) 관측 RAW 픽셀 → undistorted 픽셀
    obs_ud = rawpx_to_undist_px(pt_raw_px, K, D1 if K is K1 else D2)
    du = uv_ideal[0] - obs_ud[0]
    dv = uv_ideal[1] - obs_ud[1]
    return float(np.hypot(du, dv))

# ---------------- 카메라 오픈 (RAW 표시) ----------------
cam1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(CAM2_INDEX, cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cam1.isOpened() or not cam2.isOpened():
    raise SystemExit("카메라 열기 실패")

print("[Guide] YOLO runs ONLY on the first frame. Press q to quit.")

# ---------------- 첫 프레임: YOLO 1회 실행 ----------------
ok1, f1 = cam1.read(); ok2, f2 = cam2.read()
if not (ok1 and ok2):
    raise SystemExit("첫 프레임 캡처 실패")

# 드라이버가 다른 해상도로 줄 수 있어 강제 resize (캘리브 해상도와 맞춤)
if (f1.shape[1], f1.shape[0]) != (W, H): f1 = cv2.resize(f1, (W, H))
if (f2.shape[1], f2.shape[0]) != (W, H): f2 = cv2.resize(f2, (W, H))

L_boxes_raw = detect_boxes_raw(f1)   # RAW에서 검출
R_boxes_raw = detect_boxes_raw(f2)

pairs = match_boxes_firstframe(L_boxes_raw, R_boxes_raw)

# 매칭된 RAW 좌표쌍 보관 (그대로 고정하여 사용)
matched_pts_raw = []
for (i, j) in pairs:
    Lp = L_boxes_raw[i]; Rp = R_boxes_raw[j]
    matched_pts_raw.append(((Lp["cx"], Lp["cy"]), (Rp["cx"], Rp["cy"]), Lp, Rp))

print(f"[Info] YOLO done once. matched pairs = {len(matched_pts_raw)}")

# ---------------- 메인 루프: RAW 표시 + 깊이만 갱신 ----------------
fps_t0 = time.time(); fcount = 0
while True:
    ok1, f1 = cam1.read(); ok2, f2 = cam2.read()
    if not (ok1 and ok2):
        print("[warn] 프레임 읽기 실패")
        break

    # 해상도 강제 일치
    if (f1.shape[1], f1.shape[0]) != (W, H): f1 = cv2.resize(f1, (W, H))
    if (f2.shape[1], f2.shape[0]) != (W, H): f2 = cv2.resize(f2, (W, H))

    vis = np.hstack([f1.copy(), f2.copy()])  # RAW 화면 그대로

    # 깊이 계산(첫 프레임 좌표쌍으로만)
    for (ptL_raw, ptR_raw, Lp, Rp) in matched_pts_raw:
        X = triangulate_from_RAWpixels_once(ptL_raw, ptR_raw)   # cam1 좌표계, 단위=T
        # (선택) 재투영 오차(undistorted pixel 기준) 모니터링
        rmseL = reprojection_rmse_one(X, K1, np.eye(3), np.zeros((3,1)), ptL_raw)
        rmseR = reprojection_rmse_one(X, K2, R, T, ptR_raw)

        # 시각화: 원본 RAW 화면에 표시
        lc = (int(ptL_raw[0]), int(ptL_raw[1]))
        rc = (int(ptR_raw[0] + W), int(ptR_raw[1]))
        cv2.circle(vis, lc, 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, rc, 6, (0, 255, 0), -1, cv2.LINE_AA)

        label = f"Z={X[2]:.0f}  X={X[0]:.0f}  Y={X[1]:.0f} | reproj(px): L={rmseL:.2f} R={rmseR:.2f}"
        cv2.putText(vis, label, (lc[0]+10, lc[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (lc[0]+10, lc[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)

        if DRAW_BOXES:
            cv2.rectangle(vis, (int(Lp["x1"]), int(Lp["y1"])),
                               (int(Lp["x2"]), int(Lp["y2"])), (0,255,255), 1, cv2.LINE_AA)
            cv2.rectangle(vis, (int(Rp["x1"]+W), int(Rp["y1"])),
                               (int(Rp["x2"]+W), int(Rp["y2"])), (0,255,255), 1, cv2.LINE_AA)

    # FPS
    fcount += 1
    if fcount % 10 == 0:
        now = time.time()
        fps = 10.0 / (now - fps_t0 + 1e-9)
        fps_t0 = now
        cv2.putText(vis, f"FPS ~ {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"FPS ~ {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, vis)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()
