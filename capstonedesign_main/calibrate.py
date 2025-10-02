# calibrate.py
# 단일/스테레오 카메라 캘리브레이션 + 컬러 그리드 시각화 저장
import cv2
import numpy as np
import glob, os, argparse
from pathlib import Path

def parse_pattern(s):
    # "9x6" -> (9,6)
    a, b = s.lower().split('x')
    return (int(a), int(b))

def list_images(glob_patterns):
    files = []
    for pat in glob_patterns:
        files += glob.glob(pat)
    files = sorted(files)
    return files

def make_object_points(chessboard_size, square_size):
    # Z=0 평면상의 체스보드 코너 좌표
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp

def draw_colored_grid(img, corners, pattern, thickness=2):
    """행/열을 따라 무지개 색상으로 선을 그려줌 + 코너 작은 원"""
    h, w = img.shape[:2]
    cols, rows = pattern  # 주의: OpenCV 코너 순서는 x(열) 우선, y(행) 증가
    # corners shape: (N,1,2)
    pts = corners.reshape(-1, 2)

    # 색상 팔레트 (BGR)
    palette = [
        (255, 0, 0),(255, 64, 0),(255,128, 0),(255,192, 0),
        (255,255, 0),(192,255, 0),(128,255, 0),( 64,255, 0),
        (0,255, 0),(0,255, 64),(0,255,128),(0,255,192),
        (0,255,255),(0,192,255),(0,128,255),(0, 64,255),
        (0,  0,255)
    ]

    # 행(세로줄) 그리기: 각 x열마다 위→아래 연결
    for x in range(cols):
        line = [pts[y*cols + x] for y in range(rows)]
        color = palette[x % len(palette)]
        for i in range(len(line)-1):
            p1 = tuple(np.int32(line[i]))
            p2 = tuple(np.int32(line[i+1]))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # 열(가로줄) 그리기: 각 y행마다 좌→우 연결
    for y in range(rows):
        line = [pts[y*cols + x] for x in range(cols)]
        color = palette[y % len(palette)]
        for i in range(len(line)-1):
            p1 = tuple(np.int32(line[i]))
            p2 = tuple(np.int32(line[i+1]))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # 코너 점
    for p in pts:
        cv2.circle(img, tuple(np.int32(p)), 4, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(p)), 2, (255,255,255), -1, cv2.LINE_AA)
    return img

def detect_corners(image_paths, pattern, vis_dir=None):
    obj_points = []
    img_points = []
    used_images = []
    objp = make_object_points(pattern, 1.0)  # 단위 1.0, 최종 scale은 square_size에서 반영

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            obj_points.append(objp.copy())
            img_points.append(corners)
            used_images.append(path)

            if vis_dir:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern, corners, found)  # 기본 점 표시
                draw_colored_grid(vis, corners, pattern)                 # 컬러 그리드
                out = os.path.join(vis_dir, Path(path).stem + "_cb.jpg")
                cv2.imwrite(out, vis)
        else:
            if vis_dir:
                vis = img.copy()
                cv2.putText(vis, "NOT FOUND", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                out = os.path.join(vis_dir, Path(path).stem + "_no_cb.jpg")
                cv2.imwrite(out, vis)

    return obj_points, img_points, used_images, gray.shape[::-1] if len(image_paths)>0 else None

def calibrate_single(image_paths, pattern, square_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis_single")
    os.makedirs(vis_dir, exist_ok=True)

    obj_pts, img_pts, used, img_size = detect_corners(image_paths, pattern, vis_dir)
    if len(used) < 5:
        raise RuntimeError(f"코너 감지 성공 이미지가 너무 적습니다: {len(used)}장")

    # 객체점 스케일 적용
    obj_pts_scaled = [op * float(square_size) for op in obj_pts]

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_scaled, img_pts, img_size, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )

    # 평균 재투영 오차
    total_err, total_pts = 0.0, 0
    for i in range(len(obj_pts_scaled)):
        proj, _ = cv2.projectPoints(obj_pts_scaled[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_pts[i], proj, cv2.NORM_L2)
        total_err += err*err
        total_pts += len(proj)
    reproj_rmse = np.sqrt(total_err/total_pts)

    np.savez(os.path.join(out_dir, "intrinsics.npz"),
             K=K, dist=dist, reproj_rmse=reproj_rmse,
             image_size=img_size, used_images=np.array(used))
    print(f"[Single] 수렴={ret:.6f}, reproj_RMSE={reproj_rmse:.4f}px")
    print(f"[Single] K=\n{K}\n[Single] dist={dist.ravel()}")

    # 샘플 undistort 저장
    ex = cv2.imread(used[0])
    und = cv2.undistort(ex, K, dist)
    cv2.imwrite(os.path.join(out_dir, "undistort_sample.jpg"), und)

def calibrate_stereo(cam1_paths, cam2_paths, pattern, square_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir1 = os.path.join(out_dir, "vis_cam1")
    vis_dir2 = os.path.join(out_dir, "vis_cam2")
    os.makedirs(vis_dir1, exist_ok=True)
    os.makedirs(vis_dir2, exist_ok=True)

    # 짝 맞추기: 공통 prefix 기준(파일명 앞부분 동일)으로 매칭
    # 가장 간단히: cam1_paths와 cam2_paths를 같은 순서로 정렬했다고 가정
    n = min(len(cam1_paths), len(cam2_paths))
    cam1_paths, cam2_paths = cam1_paths[:n], cam2_paths[:n]

    obj_pts1, img_pts1, used1, img_size1 = detect_corners(cam1_paths, pattern, vis_dir1)
    obj_pts2, img_pts2, used2, img_size2 = detect_corners(cam2_paths, pattern, vis_dir2)

    # 두 카메라에서 코너가 모두 잡힌 "같은 인덱스"만 유지
    idx_ok = []
    names1 = [Path(p).stem for p in used1]
    names2 = [Path(p).stem for p in used2]
    set1, set2 = set(names1), set(names2)
    # 공통 이름(시간 스탬프 기반)이면 베스트. 단순화를 위해 교집합 사용
    common = sorted(list(set1.intersection(set2)))
    if len(common) < 5:
        print("[Stereo] 공통 감지 성공 페어가 적습니다. 순서 기반 매칭으로 시도합니다.")
        # 순서 기반 fallback
        k = min(len(obj_pts1), len(obj_pts2))
        idx_ok = list(range(k))
    else:
        idx_ok = [names1.index(n) for n in common]  # used1 기준 인덱스

    obj_pts = []
    img_pts_l, img_pts_r = [], []
    for i in idx_ok:
        obj_pts.append(obj_pts1[i] * float(square_size))
        img_pts_l.append(img_pts1[i])
        # used1의 동일 이름을 used2에서 찾아 매칭
        if len(common) >= 5:
            name = names1[i]
            j = names2.index(name)
        else:
            j = i
        img_pts_r.append(img_pts2[j])

    if len(obj_pts) < 5:
        raise RuntimeError(f"[Stereo] 유효한 페어 수가 너무 적습니다: {len(obj_pts)}")

    # 각 카메라 단일 보정으로 K,dist 초기값 추정
    _, K1, D1, r1, t1 = cv2.calibrateCamera(obj_pts, img_pts_l, img_size1, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    _, K2, D2, r2, t2 = cv2.calibrateCamera(obj_pts, img_pts_r, img_size2, None, None, flags=cv2.CALIB_RATIONAL_MODEL)

    # 스테레오 보정 (intrinsic 고정)
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_l, img_pts_r,
        K1, D1, K2, D2, img_size1,
        criteria=criteria, flags=flags
    )
    print(f"[Stereo] stereoCalibrate RMS={retval:.4f}")

    # 정합(레티피케이션)과 맵
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size1, R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size1, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size2, cv2.CV_32FC1)

    np.savez(os.path.join(out_dir, "stereo_params.npz"),
             K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             img_size=img_size1)

    # 시각화: 레티파이 샘플
    exL = cv2.imread(str(cam1_paths[0]))
    exR = cv2.imread(str(cam2_paths[0]))
    rectL = cv2.remap(exL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(exR, map2x, map2y, cv2.INTER_LINEAR)

    # 수평 라인 그려 정합 확인
    vis = np.hstack([rectL, rectR])
    h, w = rectL.shape[:2]
    for y in range(0, h, max(20, h//20)):
        cv2.line(vis, (0, y), (w*2-1, y), (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "rectified_pair.jpg"), vis)
    print("[Stereo] rectified_pair.jpg 저장")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single","stereo"], required=True, help="보정 모드")
    ap.add_argument("--pattern", default="9x6", help="내부 코너 예: 9x6")
    ap.add_argument("--square", type=float, default=26.0, help="체스보드 한 칸 길이(단위 mm 등)") # 26mm임
    ap.add_argument("--out", default="calib_out", help="결과 저장 폴더")
    ap.add_argument("--cam", choices=["cam1","cam2"], help="single 모드에서 어느 카메라인지")
    ap.add_argument("--pairs_dir", default="captures/pairs", help="페어 이미지 폴더")
    ap.add_argument("--cam1_glob", default=None, help="cam1 직접 글롭 패턴")
    ap.add_argument("--cam2_glob", default=None, help="cam2 직접 글롭 패턴")
    args = ap.parse_args()

    pattern = parse_pattern(args.pattern)

    if args.mode == "single":
        assert args.cam in ("cam1","cam2"), "single 모드에서는 --cam 필요"
        if args.cam1_glob or args.cam2_glob:
            if args.cam == "cam1":
                images = list_images([args.cam1_glob]) if args.cam1_glob else []
            else:
                images = list_images([args.cam2_glob]) if args.cam2_glob else []
        else:
            # 기본: pairs 폴더 중 하나만 선택해서 학습
            if args.cam == "cam1":
                images = list_images([f"{args.pairs_dir}/*_cam1.jpg", f"{args.pairs_dir}/*_cam1.png"])
            else:
                images = list_images([f"{args.pairs_dir}/*_cam2.jpg", f"{args.pairs_dir}/*_cam2.png"])

        print(f"[Single] 이미지 {len(images)}장")
        calibrate_single(images, pattern, args.square, os.path.join(args.out, args.cam))

    else:  # stereo
        if args.cam1_glob and args.cam2_glob:
            cam1_paths = list_images([args.cam1_glob])
            cam2_paths = list_images([args.cam2_glob])
        else:
            cam1_paths = list_images([f"{args.pairs_dir}/*_cam1.jpg", f"{args.pairs_dir}/*_cam1.png"])
            cam2_paths = list_images([f"{args.pairs_dir}/*_cam2.jpg", f"{args.pairs_dir}/*_cam2.png"])

        print(f"[Stereo] cam1 {len(cam1_paths)}장, cam2 {len(cam2_paths)}장")
        calibrate_stereo(cam1_paths, cam2_paths, pattern, args.square, os.path.join(args.out, "stereo"))

if __name__ == "__main__":
    main()
