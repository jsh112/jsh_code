#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C_Save_Route.py — 경로 저장 모드 (초기 YOLO 1회 병합→고정, 실시간은 MediaPipe만)
웹 패널(127.0.0.1:5002)에서:
- 메타 선택(섹터·난이도·색상)
- 실시간 상태(기록 수/마지막 기록/FPS)
- 버튼: 경로 저장 종료(= q), 기록 초기화(= r), YOLO 재스캔(= y)

저장 CSV:
- ./routes/{sector}_{level}_{color}.csv
- 헤더: part, hold_id, cx, cy
"""
import os, sys, time, csv
from pathlib import Path
import cv2, numpy as np
from collections import defaultdict

# ===== 외부 모듈 =====
from ultralytics import YOLO
from Save_Mediapipe import PoseTracker
from C_web import choose_meta_via_web, meta_to_csv_filename, update_state, consume_flags

# ===== 경로/기본 설정 =====
MODEL_PATH = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\best_6.pt"
NPZ_PATH   = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\calib_out\old_camera_same\stereo\stereo_params_scaled.npz"

LEFT_CAM, RIGHT_CAM = 1, 2
INIT_MERGE_FRAMES = 10
ROW_TOL_Y = 30
TOUCH_THRESHOLD = 10
ROUTES_DIR = Path("./routes"); ROUTES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_COLORS = [
    "black","blue","gray","green","lime","orange","pink","purple","red","sky","white","yellow"
]
COLOR_CANON = {"grey":"gray"}
BGR = {
    "black":(20,20,20), "blue":(255,0,0), "gray":(150,150,150), "green":(0,255,0),
    "lime":(50,255,128), "orange":(0,165,255), "pink":(203,192,255), "purple":(204,50,153),
    "red":(0,0,255), "sky":(255,255,0), "white":(255,255,255), "yellow":(0,255,255)
}

def _align32(x: int) -> int:
    return int((x + 31) // 32) * 32

def _extract_color_token(cls_name: str) -> str:
    n = cls_name.strip().lower()
    for sep in ("_", "-", "/", " "):
        if sep in n: n = n.split(sep)[-1]
    return COLOR_CANON.get(n, n)

def _mask_to_polygon(mask: np.ndarray):
    if mask.dtype != np.uint8: mask = mask.astype(np.uint8)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20: return None
    eps = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1,2)
    return approx

def _assign_hold_ids(centers):
    if not centers: return []
    pts = [(i,cx,cy) for i,(cx,cy,_,_) in enumerate(centers)]
    pts.sort(key=lambda t:(round(t[2]/ROW_TOL_Y), t[1]))  # y행→x
    idmap={}
    for new_id,(orig,_,_) in enumerate(pts): idmap[orig]=new_id
    out=[]
    for idx,(cx,cy,color,poly) in enumerate(centers):
        out.append((idmap[idx], cx, cy, color, poly))
    return out

def _draw(img, txt, org, col=(255,255,255)):
    cv2.putText(img, str(txt), org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

def draw_pose_points(img, coords, color=(0,255,255)):
    for name,(x,y) in coords.items():
        cv2.circle(img, (int(x), int(y)), 5, color, -1)
        cv2.putText(img, name, (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def _infer_batched(model, imgs, infer_hw, conf=0.25, iou=0.5):
    return model.predict(imgs, verbose=False, conf=conf, iou=iou, imgsz=infer_hw)

def _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, selected_color):
    merge_L, merge_R = [], []
    for i in range(INIT_MERGE_FRAMES):
        okL, fL = capL.read(); okR, fR = capR.read()
        if not (okL and okR): break
        if map1L is not None:
            fL = cv2.remap(fL, map1L, map2L, cv2.INTER_LINEAR)
            fR = cv2.remap(fR, map1R, map2R, cv2.INTER_LINEAR)

        res_list = _infer_batched(model, [fL, fR], infer_hw)
        for side_idx, res in enumerate(res_list):
            side_merge = merge_L if side_idx==0 else merge_R
            if getattr(res, "masks", None) is None: continue
            mks = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros(len(mks), dtype=int)
            names = res.names if hasattr(res, "names") else {}
            dets=[]
            for k, mk in enumerate(mks):
                mask = (mk>0.5).astype(np.uint8)*255
                ys,xs = np.where(mask>0)
                if xs.size==0: continue
                cx,cy = int(xs.mean()), int(ys.mean())
                cname = names.get(int(cls[k]), str(int(cls[k])))
                tok = _extract_color_token(cname)
                if tok not in ALLOWED_COLORS: continue
                if (selected_color is not None) and (tok != selected_color): continue
                dets.append((mask, tok, cx, cy))
            if not side_merge:
                side_merge = [(m.copy(), c, x, y) for (m,c,x,y) in dets]
            else:
                new=[]
                for (m0,c0,x0,y0) in side_merge:
                    comb=m0.copy()
                    for (m1,c1,x1,y1) in dets:
                        if c0==c1 and abs(x0-x1)<40 and abs(y0-y1)<40:
                            comb = cv2.bitwise_or(comb, m1)
                    new.append((comb,c0,x0,y0))
                side_merge = new
            if side_idx==0: merge_L = side_merge
            else: merge_R = side_merge

        vis = fL.copy()
        _draw(vis, f"Init YOLO merge {i+1}/{INIT_MERGE_FRAMES}", (10,30), (0,255,255))
        cv2.imshow("Left (Save Route)", vis); cv2.imshow("Right (Save Route)", fR)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return [], []

    def to_holds(side_merge):
        centers=[]
        for (mask, tok, cx, cy) in side_merge:
            poly = _mask_to_polygon(mask)
            if poly is None: continue
            centers.append((cx,cy,tok,poly))
        return _assign_hold_ids(centers)

    return to_holds(merge_L), to_holds(merge_R)

def _detect_size_from_npz(npz):
    for k in ("map1L","map2L","map1R","map2R"):
        if k in npz:
            h, w = npz[k].shape[:2]
            return (w, h)
    for k in ("img_size","image_size","size","wh","WH"):
        if k in npz:
            v = np.array(npz[k]).astype(int).ravel()
            if v.size >= 2:
                a,b = int(v[0]), int(v[1])
                return (max(a,b), min(a,b))
    if "W" in npz and "H" in npz: return (int(npz["W"]), int(npz["H"]))
    if "w" in npz and "h" in npz: return (int(npz["w"]), int(npz["h"]))
    return None

def load_stereo(npz_path: str):
    if not os.path.exists(npz_path):
        print(f"[Warn] NPZ 없음: {npz_path} → 보정 없이 진행")
        return (None, None, None, None, None, None)
    z = np.load(npz_path, allow_pickle=True)
    size = _detect_size_from_npz(z)
    if {"map1L","map2L","map1R","map2R"}.issubset(set(z.files)):
        if size: print(f"[Info] NPZ 해상도: {size[0]}x{size[1]} (precomputed remap 사용)")
        return (size[0] if size else None, size[1] if size else None, z["map1L"], z["map2L"], z["map1R"], z["map2R"])
    # 파라미터로 계산
    try:
        K1, D1 = z["K1"], z["D1"]; K2, D2 = z["K2"], z["D2"]
        R1, R2 = z["R1"], z["R2"]; P1, P2 = z["P1"], z["P2"]
    except KeyError:
        return (size[0] if size else None, size[1] if size else None, None, None, None, None)
    if not size: return (None, None, None, None, None, None)
    w, h = size
    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w,h), cv2.CV_16SC2)
    return (w, h, map1L, map2L, map1R, map2R)

def main():
    model = YOLO(MODEL_PATH)
    try:
        class_names = list(model.names.values())
    except Exception:
        class_names = None

    meta = choose_meta_via_web(yolo_class_names=class_names)
    csv_name = meta_to_csv_filename(meta)
    CSV_PATH = ROUTES_DIR / csv_name
    SELECTED = None if meta["color"] == "all" else meta["color"]
    print(f"[Info] 메타={meta} → 저장파일={CSV_PATH}")

    W, H, map1L, map2L, map1R, map2R = load_stereo(NPZ_PATH)
    if W is None or H is None: W, H = 1280, 720
    INFER_W, INFER_H = _align32(W), _align32(H)
    infer_hw = (INFER_H, INFER_W)

    capL = cv2.VideoCapture(LEFT_CAM, cv2.CAP_DSHOW if os.name=='nt' else 0)
    capR = cv2.VideoCapture(RIGHT_CAM, cv2.CAP_DSHOW if os.name=='nt' else 0)
    for c in (capL, capR):
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        c.set(cv2.CAP_PROP_FPS, 30)
    if not (capL.isOpened() and capR.isOpened()):
        print("[Error] 카메라 오픈 실패"); sys.exit(1)

    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=0)

    grip_records = []
    already_logged = set()           # ("part", hold_id)
    touch_streak = defaultdict(int)  # (part_name, hold_id) -> 연속프레임

    # 초기 YOLO 고정
    print("[Info] 초기 YOLO 병합 중...")
    holdsL, holdsR = _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, SELECTED)
    print(f"[Info] holdsL={len(holdsL)}, holdsR={len(holdsR)}")

    t_prev = time.time()
    try:
        while True:
            okL, frameL = capL.read(); okR, frameR = capR.read()
            if not (okL and okR): print("[Warn] 프레임 수신 실패"); break

            if map1L is not None:
                rectL = cv2.remap(frameL, map1L, map2L, cv2.INTER_LINEAR)
                rectR = cv2.remap(frameR, map1R, map2R, cv2.INTER_LINEAR)
            else:
                rectL, rectR = frameL, frameR

            visL = rectL.copy(); visR = rectR.copy()

            coords = pose.process(rectL)

            live_filled_ids = set(); current_touched = set()
            centers_by_id_L = {hid:(cx,cy) for (hid,cx,cy,_,_) in holdsL}

            if coords:
                for part_name, (px, py) in coords.items():
                    for hid, cx, cy, tok, poly in holdsL:
                        if cv2.pointPolygonTest(poly.astype(np.float32), (px, py), False) >= 0:
                            current_touched.add((part_name, hid))
                for key in current_touched:
                    touch_streak[key] += 1
                    part_name, hid = key
                    if touch_streak[key] >= TOUCH_THRESHOLD:
                        live_filled_ids.add(hid)
                        if key not in already_logged:
                            cxcy = centers_by_id_L.get(hid)
                            if cxcy:
                                cx, cy = cxcy
                                rec = [part_name, int(hid), int(cx), int(cy)]
                                grip_records.append(rec)
                                already_logged.add(key)
            # 떨어진 키는 0
            for key in list(touch_streak.keys()):
                if key not in current_touched:
                    touch_streak[key] = 0

            # 드로잉
            draw_pose_points(visL, coords, (0,255,255))
            for hid, cx, cy, tok, poly in holdsL:
                if hid in live_filled_ids:
                    cv2.fillPoly(visL, [poly.astype(np.int32)], BGR.get(tok,(200,200,200)))
                cv2.polylines(visL, [poly.astype(np.int32)], True, BGR.get(tok,(200,200,200)), 2)
                _draw(visL, f"ID:{hid}", (cx-10, cy+26), BGR.get(tok,(255,255,255)))
            for hid, cx, cy, tok, poly in holdsR:
                cv2.polylines(visR, [poly.astype(np.int32)], True, BGR.get(tok,(200,200,200)), 2)
                _draw(visR, f"ID:{hid}", (cx-10, cy+26), BGR.get(tok,(255,255,255)))

            _draw(visL, f"{meta['sector']} · {meta['level']} · {meta['color']}", (10,20))
            _draw(visL, f"records={len(grip_records)}  (웹: 종료/초기화/리스캔)", (10,46), (200,200,200))
            t_now = time.time(); fps = 1.0 / max(t_now - t_prev, 1e-6); t_prev = t_now
            _draw(visL, f"FPS: {fps:.1f}", (10, visL.shape[0]-10), (0,255,255))

            # ----- 웹 상태 갱신 -----
            last = grip_records[-1] if grip_records else None
            update_state(records_count=len(grip_records), last_record=last, fps=fps)

            # ----- 웹 버튼 신호 처리 -----
            flags = consume_flags()
            if flags.get("reset"):
                grip_records.clear(); already_logged.clear(); touch_streak.clear()
            if flags.get("rescan"):
                print("[Info] YOLO 재스캔...")
                holdsL, holdsR = _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, SELECTED)
                print(f"[Info] 재스캔 완료: holdsL={len(holdsL)}, holdsR={len(holdsR)}")
            if flags.get("stop"):
                break

            cv2.imshow("Left (Save Route)", visL)
            cv2.imshow("Right (Save Route)", visR)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break   # 키보드 폴백
            elif k == ord('r'):
                grip_records.clear(); already_logged.clear(); touch_streak.clear()
            elif k == ord('y'):
                print("[Info] YOLO 재스캔...")
                holdsL, holdsR = _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, SELECTED)
                print(f"[Info] 재스캔 완료: holdsL={len(holdsL)}, holdsR={len(holdsR)}")

    finally:
        try: capL.release(); capR.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        try: pose.close()
        except: pass

    # CSV 저장
    with open(CSV_PATH, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part","hold_id","cx","cy"])
        w.writerows(grip_records)
    print(f"[Info] CSV 저장 완료: {CSV_PATH} (총 {len(grip_records)}개)")
    print("[Done] C_Save_Route 종료")

if __name__ == "__main__":
    main()
