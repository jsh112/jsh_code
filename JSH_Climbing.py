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

CSV_GRIPS_PATH = "grip_records.csv"

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
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W,H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W,H), cv2.CV_32FC1)
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

def save_holds_to_csv_cumulative(all_rows, csv_path="grip_records.csv"):
    if not all_rows: return
    header_fmt = "{:<8} {:>10} {:>10} {:>10} {:<15}"
    row_fmt = "{:<8} {:>10.1f} {:>10.1f} {:>10.1f} {:<15}"
    try:
        with open(csv_path,"w") as f:
            f.write(header_fmt.format("hold_id","x_mm","y_mm","z_mm","color")+"\n")
            for r in all_rows:
                f.write(row_fmt.format(r["hold_id"], r["x_mm"], r["y_mm"], r["z_mm"], r["color"])+"\n")
    except Exception as e:
        print(f"CSV 저장 실패: {e}")

# ------------------ 1. 좌우 홀드 매칭 ------------------
def match_holds_by_proximity(holdsL, holdsR, max_dist_px=20):
    """
    좌/우 홀드를 픽셀 거리 기준으로 매칭
    - holdsL, holdsR : extract_holds() 결과
    - max_dist_px : 허용 거리
    """
    matches = []
    for hl in holdsL:
        cxL, cyL = hl["center"]
        # R에서 가장 가까운 점 찾기
        best_match = None
        min_dist = max_dist_px
        for hr in holdsR:
            cxR, cyR = hr["center"]
            dist = np.hypot(cxL - cxR, cyL - cyR)
            if dist < min_dist:
                min_dist = dist
                best_match = hr
        if best_match is not None:
            matches.append({
                "L": hl,
                "R": best_match,
                "dist_px": min_dist
            })
    return matches

# ------------------ 2. 매칭 결과 시각화 ------------------
def visualize_matches(frameL, frameR, matches):
    """
    좌우 이미지에 매칭 점 표시
    - frameL, frameR : 좌/우 이미지
    - matches : match_holds_by_proximity 결과
    """
    vis = np.hstack([frameL.copy(), frameR.copy()])
    W = frameL.shape[1]

    for m in matches:
        cxL, cyL = m["L"]["center"]
        cxR, cyR = m["R"]["center"]
        color = m["L"].get("color", (255,255,255))
        
        # 좌측 표시
        cv2.circle(vis, (cxL, cyL), 5, color, -1)
        cv2.putText(vis, "L", (cxL, cyL-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 우측 표시 (가로 offset)
        cv2.circle(vis, (cxR+W, cyR), 5, color, -1)
        cv2.putText(vis, "R", (cxR+W, cyR-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 매칭 선
        cv2.line(vis, (cxL, cyL), (cxR+W, cyR), color, 1)

    return vis

# ------------------ 3. 매칭 실패 확인 ------------------
def report_unmatched_holds(holdsL, holdsR, matches):
    """
    매칭되지 않은 홀드 확인
    """
    matched_L = {m["L"]["center"] for m in matches}
    matched_R = {m["R"]["center"] for m in matches}

    unmatched_L = [h for h in holdsL if h["center"] not in matched_L]
    unmatched_R = [h for h in holdsR if h["center"] not in matched_R]

    print(f"좌측 미매칭 홀드: {len(unmatched_L)}개")
    print(f"우측 미매칭 홀드: {len(unmatched_R)}개")
    return unmatched_L, unmatched_R

# ------------------- 1. IOU 계산 -------------------
def iou_mask(mask1, mask2):
    """
    두 바이너리 마스크(mask1, mask2)의 IOU 계산
    mask1, mask2 : np.uint8 (0/255) 또는 bool
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0:
        return 0.0
    return intersection / union

# ------------------- 2. 좌우 홀드 매칭(IOU 기준) -------------------
def match_holds_by_iou(holdsL, holdsR, iou_thresh=0.2):
    """
    좌/우 홀드를 마스크 IOU 기준으로 매칭
    - holdsL, holdsR : extract_holds() 결과
    - iou_thresh : IOU 최소 기준
    """
    matches = []
    used_R = set()
    for hl in holdsL:
        best_match = None
        best_iou = 0
        maskL = cv2.resize((hl["contour_mask"] if "contour_mask" in hl else contour_to_mask(hl["contour"], hl.get("frame_shape"))),
                           (hl.get("frame_shape")[1], hl.get("frame_shape")[0])) if "frame_shape" in hl else None
        for idx, hr in enumerate(holdsR):
            if idx in used_R: continue
            maskR = cv2.resize((hr["contour_mask"] if "contour_mask" in hr else contour_to_mask(hr["contour"], hr.get("frame_shape"))),
                               (hr.get("frame_shape")[1], hr.get("frame_shape")[0])) if "frame_shape" in hr else None
            iou = iou_mask(maskL, maskR)
            if iou > best_iou:
                best_iou = iou
                best_match = hr
                best_idx = idx
        if best_iou >= iou_thresh and best_match is not None:
            matches.append({"L": hl, "R": best_match, "iou": best_iou})
            used_R.add(best_idx)
    return matches

# ------------------- 3. contour -> mask 변환 -------------------
def contour_to_mask(contour, shape):
    """
    단순히 contour를 mask로 변환
    - contour : cv2 contour
    - shape : (H,W)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask

# ------------------- 4. 매칭 결과 시각화 -------------------
def visualize_matches_iou(frameL, frameR, matches):
    vis = np.hstack([frameL.copy(), frameR.copy()])
    W = frameL.shape[1]
    for m in matches:
        cxL, cyL = m["L"]["center"]
        cxR, cyR = m["R"]["center"]
        color = m["L"].get("color", (255,255,255))

        cv2.circle(vis, (cxL, cyL), 5, color, -1)
        cv2.putText(vis, "L", (cxL, cyL-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.circle(vis, (cxR+W, cyR), 5, color, -1)
        cv2.putText(vis, "R", (cxR+W, cyR-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.line(vis, (cxL, cyL), (cxR+W, cyR), color, 1)
    return vis



# ================== 메인 ==================
def main():
    if not Path(NPZ_PATH).exists() or not Path(MODEL_PATH).exists():
        raise FileNotFoundError("NPZ 또는 모델 파일 없음")

    map1x,map1y,map2x,map2y,P1,P2,size = load_stereo(NPZ_PATH)
    W,H = size
    cap1,cap2 = open_cams(CAM1_INDEX,CAM2_INDEX,size)
    model = YOLO(str(MODEL_PATH))
    cv2.namedWindow("Stereo YOLO 3D", cv2.WINDOW_NORMAL)

    all_rows = []

    # while True:
    #     ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
    #     if not (ok1 and ok2): break

    #     Lr = rectify(f1, map1x, map1y, size)
    #     Rr = rectify(f2, map2x, map2y, size)

    #     holdsL = extract_holds(Lr, model)
    #     holdsR = extract_holds(Rr, model)

    #     # ---------------- 좌우 홀드 매칭 테스트 ----------------
    #     for h in holdsL: h["frame_shape"] = Lr.shape[:2]
    #     for h in holdsR: h["frame_shape"] = Rr.shape[:2]

    #     matches = match_holds_by_iou(holdsL, holdsR, iou_thresh=0.1)
    #     vis_matches = visualize_matches_iou(Lr, Rr, matches)
    #     cv2.imshow("Matches_IOU", vis_matches)
    #     # -------------------------------------------------------

    #     # 좌/우 병합 및 인덱스 유지 (화면 표시용)
    #     merged_holds = merge_holds_by_center([holdsL, holdsR])

    #     # 화면 표시
    #     vis = np.hstack([Lr, Rr])
    #     for h in merged_holds:
    #         cx, cy = h["center"]
    #         cv2.circle(vis, (cx, cy), 4, h["color"], -1)
    #         cv2.putText(vis, str(h["hold_index"]), (cx, cy-5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #     # 3D 좌표 계산 후 누적 (IOU 매칭 기준)
    #     for m in matches:
    #         hl = m["L"]
    #         hr = m["R"]
    #         hid = hl.get("hold_index", hl.get("center"))  # hold_index 없으면 center로 임시 ID
    #         X = triangulate_xy(P1, P2, hl["center"], hr["center"])
    #         all_rows.append({
    #             "hold_id": hid,
    #             "x_mm": X[0],
    #             "y_mm": X[1],
    #             "z_mm": X[2],
    #             "color": hl["class_name"]
    #         })

    #     # 누적 CSV 저장
    #     save_holds_to_csv_cumulative(all_rows, CSV_GRIPS_PATH)

    #     cv2.imshow("Stereo YOLO 3D", vis)
    #     if cv2.waitKey(1) & 0xFF == 27:  # ESC
    #         break
    while True:
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2): break

        Lr = rectify(f1, map1x, map1y, size)
        Rr = rectify(f2, map2x, map2y, size)

        holdsL = extract_holds(Lr, model)
        holdsR = extract_holds(Rr, model)

        # 각 hold에 frame_shape 추가 (mask 생성용)
        for h in holdsL: h["frame_shape"] = Lr.shape[:2]
        for h in holdsR: h["frame_shape"] = Rr.shape[:2]

        # IOU 기반 매칭
        matches = match_holds_by_iou(holdsL, holdsR, iou_thresh=0.1)
        
        # 매칭 결과만 시각화
        vis_matches = visualize_matches_iou(Lr, Rr, matches)
        cv2.imshow("Matches_IOU", vis_matches)

        # 좌/우 병합 화면 표시
        merged_holds = merge_holds_by_center([holdsL, holdsR])
        vis = np.hstack([Lr, Rr])
        for h in merged_holds:
            cx, cy = h["center"]
            cv2.circle(vis, (cx, cy), 4, h["color"], -1)
            cv2.putText(vis, str(h["hold_index"]), (cx, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Stereo YOLO 3D", vis)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break


    cap1.release(); cap2.release()
    cv2.destroyAllWindows()
    print(f"CSV 저장 완료: {CSV_GRIPS_PATH}")

if __name__=="__main__":
    main()
