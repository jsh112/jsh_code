import cv2, numpy as np, os, glob, re, time
from pathlib import Path

# 길이 = 
# ============ 설정 ============
PAIRS_DIR   = "captures/pairs"  # 페어 이미지 폴더 (take_a_photo.py가 저장한 곳)
DATE_FILTER = "20250924"
PATTERN     = (9, 6)            # 체스보드 "내부 코너" 개수 
SQUARE_MM   = 26.00             # 한 칸 길이(단위는 mm)

RUN_SINGLE  = True              # cam1, cam2 각각 단일 카메라 보정 수행 여부
RUN_STEREO  = True              # 두 카메라 스테레오 보정 수행 여부
LEFT_IS_CAM2 = True             # 물리 좌=cam2, 우=cam1 환경일 때 보기용으로 좌우 스왑(수학적 결과엔 영향 없음)

MARGIN_PX   = 15                # 체스보드 코너 bounding box가 프레임 가장자리에서 떨어져야 하는 최소 마진(px)

# 실측 베이스라인(mm). None이면 스케일 보정 생략
MEASURED_BASELINE_MM = 372.00  # 예) 103.0  (mm)
# =============================
def ts():
    # 결과 디렉터리 이름 등에 쓰는 타임스탬프(YYYYmmdd_HHMMSS)
    return time.strftime('%Y%m%d_%H%M%S')

def ensure(p): os.makedirs(p, exist_ok=True)  # 디렉터리 생성 헬퍼

def list_files():
    # 페어 폴더에서 cam1/cam2 파일을 글롭으로 수집
    # *_cam1*.{jpg,png} / *_cam2*.{jpg,png} 형태면 모두 매칭
    pats1 = [f"{PAIRS_DIR}/{DATE_FILTER}*_cam1*.jpg",
             f"{PAIRS_DIR}/{DATE_FILTER}*_cam1*.jpeg",
             f"{PAIRS_DIR}/{DATE_FILTER}*_cam1*.png"]
    pats2 = [f"{PAIRS_DIR}/{DATE_FILTER}*_cam2*.jpg",
             f"{PAIRS_DIR}/{DATE_FILTER}*_cam2*.jpeg",
             f"{PAIRS_DIR}/{DATE_FILTER}*_cam2*.png"]
    cam1 = sorted(sum([glob.glob(p) for p in pats1], []))
    cam2 = sorted(sum([glob.glob(p) for p in pats2], []))
    return cam1, cam2

# _cam1_cb 같은 이름을 정확히 자르기
# "_cam1_" 또는 파일 끝("_cam1")에서 split하여 공통 pair id 추출
# 예: "20250905_153031_cam1_cb.jpg" → "20250905_153031"
def pair_id(path: str) -> str:
    name = Path(path).stem
    parts = re.split(r"_cam(?:1|2)(?:_|$)", name, maxsplit=1)
    return parts[0] if parts else name

def build_pairs(cam1_list, cam2_list):
    # cam1/cam2 각각에서 공통 pair id를 가진 파일쌍만 골라 정렬
    m1 = {pair_id(p): p for p in cam1_list}
    m2 = {pair_id(p): p for p in cam2_list}
    ids = sorted(set(m1.keys()) & set(m2.keys()))
    cam1_pairs = [m1[i] for i in ids]
    cam2_pairs = [m2[i] for i in ids]
    return ids, cam1_pairs, cam2_pairs

def make_object_points(pattern, square):
    # 체스보드 월드 좌표(평면 z=0) 생성: (cols*rows, 3)
    # 한 칸 길이(square)를 곱해 실제 단위(mm)로 스케일이 정해짐
    cols, rows = pattern # 체스보드 내부 코너 수 받고 
    objp = np.zeros((cols*rows, 3), np.float32) # cols*rows 개의 점을 담는 배열. 각 점은 (x, y, z) 3차원 좌표다
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) # 행렬 변환
    objp *= float(square)
    return objp

def draw_grid(img, corners, pattern):
    # 코너 시각화(행/열을 컬러 선으로 연결 + 점 표시) → 디버그/검수용
    cols, rows = pattern
    pts = corners.reshape(-1, 2)
    pal = [(255,0,0),(255,128,0),(255,255,0),(128,255,0),(0,255,0),
           (0,255,128),(0,255,255),(0,128,255),(0,0,255)]
    for x in range(cols):
        line = [pts[y*cols + x] for y in range(rows)]
        c = pal[x % len(pal)]
        for i in range(len(line)-1):
            cv2.line(img, tuple(np.int32(line[i])), tuple(np.int32(line[i+1])), c, 2, cv2.LINE_AA)
    for y in range(rows):
        line = [pts[y*cols + x] for x in range(cols)]
        c = pal[y % len(pal)]
        for i in range(len(line)-1):
            cv2.line(img, tuple(np.int32(line[i])), tuple(np.int32(line[i+1])), c, 2, cv2.LINE_AA)
    for p in pts:
        p = tuple(np.int32(p))
        cv2.circle(img, p, 4, (0,0,0), -1)
        cv2.circle(img, p, 2, (255,255,255), -1)
    return img

def corners_with_margin(img, pattern):
    # 체스보드 코너 검출 + 서브픽셀 보정 후,
    # 코너 bbox가 가장자리와 너무 가까우면(마진 미달) 품질상 제외
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return False, None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    xs = corners[:,0,0]; ys = corners[:,0,1]
    h, w = gray.shape
    if xs.min()<MARGIN_PX or ys.min()<MARGIN_PX or xs.max()>w-MARGIN_PX or ys.max()>h-MARGIN_PX:
        return False, corners, (w,h)   # 가장자리 너무 가까우면 품질상 제외
    return True, corners, (w,h)

def detect_set(paths, pattern, square, vis_dir, tag):
    # 파일 목록(paths)에서 코너 검출 수행
    # - 성공 이미지만 obj/img 리스트에 추가
    # - 시각화 이미지를 vis_dir에 저장(_cb: ok, _bad: 실패/마진 미달)
    obj_list, img_list, used, sizes = [], [], [], []
    objp = make_object_points(pattern, square)
    ensure(vis_dir)
    ok_cnt = edge_skip = fail = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            fail += 1
            continue
        ok, corners, size = corners_with_margin(img, pattern)
        if ok:
            obj_list.append(objp.copy()); img_list.append(corners); used.append(p); sizes.append(size); ok_cnt += 1
            vis = img.copy(); cv2.drawChessboardCorners(vis, pattern, corners, True); draw_grid(vis, corners, pattern)
            cv2.imwrite(os.path.join(vis_dir, Path(p).stem+"_cb.jpg"), vis)
        else:
            # 실패 or 가장자리 근접
            reason = "edge" if corners is not None else "nofind"
            if reason == "edge": edge_skip += 1
            fail += 1
            vis = img.copy()
            msg = "EDGE TOO CLOSE" if reason=="edge" else "NOT FOUND"
            cv2.putText(vis, msg, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(vis_dir, Path(p).stem+"_bad.jpg"), vis)
    size = sizes[0] if sizes else None
    print(f"[{tag}] ok={ok_cnt}, edge-skip={edge_skip}, fail={fail}")
    return obj_list, img_list, used, size

def calibrate_single(paths, out_dir, cam_tag):
    # 단일 카메라 보정: K(내부행렬), dist(왜곡계수), RMSE 저장 + undistort 샘플 출력
    ensure(out_dir); vis_dir = os.path.join(out_dir, "vis"); ensure(vis_dir)
    obj_pts, img_pts, used, img_size = detect_set(paths, PATTERN, SQUARE_MM, vis_dir, f"Single:{cam_tag}")
    if len(used) < 5:
        print(f"[Single:{cam_tag}] 유효 이미지 부족(>=5 권장)."); return None, None
    # 모델 단순화(기본 5-파라미터: k1,k2,p1,p2,k3)로 안정화
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
    # 재투영 오차(RMSE) 직접 계산하여 출력
    sse = 0.0; npts = 0
    for i in range(len(obj_pts)):
        proj,_ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_pts[i], proj, cv2.NORM_L2)
        sse += err*err; npts += len(proj)
    rmse = float(np.sqrt(sse/npts))
    print(f"[Single:{cam_tag}] RMSE={rmse:.4f}px\nK=\n{K}\ndist={dist.ravel()}")
    # 결과 저장(npz: 넘파이 아카이브). npz 확장자들은 나중에 캘리브레이션 할 때 가져가서 쓰면 되는 듯.
    np.savez(os.path.join(out_dir, "intrinsics.npz"),
             K=K, dist=dist, rmse=rmse, image_size=np.array(img_size), used_images=np.array(used))
    # 왜곡 보정 샘플 이미지
    und = cv2.undistort(cv2.imread(used[0]), K, dist)
    cv2.imwrite(os.path.join(out_dir, "undistort_sample.jpg"), und)
    return K, dist

def calibrate_stereo(ids, cam1_paths, cam2_paths, out_dir, measured_baseline=None):
    # 스테레오 보정: 공통 pair id로 정렬된 cam1/cam2 리스트를 받아
    # 코너 검출 → intrinsics 고정 stereoCalibrate → Rectify(Map) 생성/저장
    ensure(out_dir)
    vis1 = os.path.join(out_dir, "vis_cam1"); vis2 = os.path.join(out_dir, "vis_cam2")
    obj1, img1, used1, size1 = detect_set(cam1_paths, PATTERN, SQUARE_MM, vis1, "Stereo:cam1")
    obj2, img2, used2, size2 = detect_set(cam2_paths, PATTERN, SQUARE_MM, vis2, "Stereo:cam2")

    # 공통 id로 정렬 (코너 성공한 파일들 중 교집합만 사용)
    id1 = [pair_id(p) for p in used1]; id2 = [pair_id(p) for p in used2]
    inter = [i for i in ids if i in id1 and i in id2]
    if len(inter) < 5:
        print("[Stereo] 코너 성공 공통 페어 부족."); return

    idx1 = [id1.index(i) for i in inter]
    idx2 = [id2.index(i) for i in inter]
    obj  = [obj1[i]  for i in idx1]
    ptsL = [img1[i]  for i in idx1]
    ptsR = [img2[j]  for j in idx2]

    # 단일 보정으로 초기 intrinsics를 만들고,
    _, K1, D1, _, _ = cv2.calibrateCamera(obj, ptsL, size1, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(obj, ptsR, size2, None, None)
    # 스테레오 보정은 intrinsics를 고정(FIX_INTRINSIC)하고 R(회전),T(이동) 등 extrinsics 추정
    flags = cv2.CALIB_FIX_INTRINSIC
    crit  = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(obj, ptsL, ptsR, K1, D1, K2, D2, size1, criteria=crit, flags=flags)
    print(f"[Stereo] RMS={rms:.4f}  | pairs={len(inter)}")
    baseline = float(np.linalg.norm(T))
    print(f"[Stereo] Baseline ≈ {baseline:.2f} (SQUARE 단위; mm면 mm)")

    # Rectify(수평 에피폴라 정렬) → R1,R2(회전), P1,P2(투영행렬), Q(3D 재투영)
    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(K1,D1,K2,D2,size1,R,T,alpha=0)
    map1x,map1y = cv2.initUndistortRectifyMap(K1,D1,R1,P1,size1,cv2.CV_32FC1)
    map2x,map2y = cv2.initUndistortRectifyMap(K2,D2,R2,P2,size2,cv2.CV_32FC1)

    # 파라미터 전부 저장(npz): 후속 삼각측량/실시간 정합에서 바로 사용
    np.savez(os.path.join(out_dir, "stereo_params.npz"),
             K1=K1,D1=D1,K2=K2,D2=D2,R=R,T=T,E=E,F=F,
             R1=R1,R2=R2,P1=P1,P2=P2,Q=Q,image_size=np.array(size1),
             used_pair_ids=np.array(inter))

    # 미리보기 이미지 생성(첫 공통 페어로 remap). 원본꺼
    sid = inter[0]
    s1 = next(p for p in cam1_paths if pair_id(p)==sid)
    s2 = next(p for p in cam2_paths if pair_id(p)==sid)
    exL = cv2.imread(s1); exR = cv2.imread(s2)
    rectL = cv2.remap(exL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(exR, map2x, map2y, cv2.INTER_LINEAR)
    if LEFT_IS_CAM2: rectL, rectR = rectR, rectL  # 보기용 좌우 스왑(파라미터 자체는 그대로)
    vis = np.hstack([rectL, rectR])
    h, w = rectL.shape[:2]
    for y in range(0, h, max(20, h//20)):
        cv2.line(vis, (0,y), (w*2-1,y), (0,255,0), 1, cv2.LINE_AA)  # 수평 보조선
    cv2.imwrite(os.path.join(out_dir, "rectified_pair.jpg"), vis)
    print("[Stereo] rectified_pair.jpg 저장")

    # --- 실측 베이스라인으로 스케일 보정(옵션) ---
    if measured_baseline is not None:
        B_meas = float(measured_baseline)
        scale  = B_meas / baseline
        T2     = T * scale
        R1s,R2s,P1s,P2s,Qs,_,_ = cv2.stereoRectify(K1,D1,K2,D2,size1,R,T2,alpha=0)
        np.savez(os.path.join(out_dir, "stereo_params_scaled.npz"),
                 K1=K1,D1=D1,K2=K2,D2=D2,R=R,T=T2,E=E,F=F,
                 R1=R1s,R2=R2s,P1=P1s,P2=P2s,Q=Qs,image_size=np.array(size1),
                 used_pair_ids=np.array(inter),
                 baseline_before=baseline, baseline_measured=B_meas, scale_applied=scale)
        # 미리보기(스케일 보정본)
        map1x_s,map1y_s = cv2.initUndistortRectifyMap(K1,D1,R1s,P1s,size1,cv2.CV_32FC1)
        map2x_s,map2y_s = cv2.initUndistortRectifyMap(K2,D2,R2s,P2s,size1,cv2.CV_32FC1)
        rectL_s = cv2.remap(exL, map1x_s, map1y_s, cv2.INTER_LINEAR)
        rectR_s = cv2.remap(exR, map2x_s, map2y_s, cv2.INTER_LINEAR)
        if LEFT_IS_CAM2: rectL_s, rectR_s = rectR_s, rectL_s
        vis_s = np.hstack([rectL_s, rectR_s])
        for y in range(0, h, max(20, h//20)):
            cv2.line(vis_s, (0,y), (w*2-1,y), (0,255,0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, "rectified_pair_scaled.jpg"), vis_s)
        print(f"[Stereo] Baseline scaled: {baseline:.2f} → {np.linalg.norm(T2):.2f} (scale={scale:.6f})")
        print("[Stereo] stereo_params_scaled.npz 저장")

def main():
    # 출력 루트 폴더(타임스탬프) 준비
    out_root = os.path.join("calib_out", ts()); ensure(out_root)
    print(f"[Info] 결과 폴더: {out_root}")
    print(f"[Info] 찾는 폴더: {os.path.abspath(PAIRS_DIR)}")
    print(f"[Info] 패턴={PATTERN}, square={SQUARE_MM}")

    # 원본 cam1/cam2 파일 전체 스캔
    cam1_raw, cam2_raw = list_files()
    print(f"[Scan] cam1={len(cam1_raw)}, cam2={len(cam2_raw)}")

    # 🧭 디버그: pair id 샘플 3개 출력(정규식 파싱 확인용). 파일명이 쌍 잘 이루는지 확인용
    sample1 = [pair_id(p) for p in cam1_raw[:3]]
    sample2 = [pair_id(p) for p in cam2_raw[:3]]
    print(f"[Debug] cam1 first ids: {sample1}")
    print(f"[Debug] cam2 first ids: {sample2}")

    # 단일 보정은 원본 리스트로 바로 수행(교집합 없어도 가능)
    if RUN_SINGLE:
        calibrate_single(cam1_raw, os.path.join(out_root, "cam1"), "cam1")
        calibrate_single(cam2_raw, os.path.join(out_root, "cam2"), "cam2")

    # 스테레오 보정에 사용할 정확 페어 리스트 구성(공통 pair id 교집합)
    ids, cam1_pairs, cam2_pairs = build_pairs(cam1_raw, cam2_raw)
    print(f"[Pairs] 공통 pair_ids={len(ids)} (예: {ids[:5]}...)")

    if RUN_STEREO:
        calibrate_stereo(ids, cam1_pairs, cam2_pairs, os.path.join(out_root, "stereo"), measured_baseline=MEASURED_BASELINE_MM)

    print("\n[Done] 캘리브레이션 완료!")

if __name__ == "__main__":
    main()
