#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C_Climbing.py — 사용자 등반 모드 (CSV 경로 순서 진행)
- 웹(C_web)의 모드: "climb" / 메타(섹터·난이도·색상) 선택
- ./routes/{sector}_{level}_{color}.csv 가 있으면
    CSV의 hold_id, cx, cy 순서대로 '타깃'을 진행 (터치 성공 시 다음으로)
  없으면
    화면에 "아직 저장된 경로가 없습니다" 안내 후 종료
- YOLO는 시작 시 N프레임 병합으로 '한 번만' 실행하여 홀드 폴리곤 고정
- MediaPipe로 손/발 터치 판정(A_Climbing의 사용감 유지)
- 웹 패널에 진행상태/FPS 업데이트, 웹 버튼(stop/reset/rescan) 반영
- 키: q 종료, r 초기화, y YOLO 재스캔
"""
import os, sys, time, csv
from pathlib import Path
import cv2, numpy as np

from ultralytics import YOLO
from Climb_Mediapipe import PoseTracker, TouchCounter, draw_pose_points
from C_web import choose_meta_via_web, meta_to_csv_filename, update_state, consume_flags

# (옵션) 서보 — 연결 없으면 동작 없이 안전
try:
    from servo_control import DualServoController
except Exception:
    class DualServoController:
        def __init__(self, *a, **k): pass
        def set_angles(self, yaw=None, pitch=None): pass
        def laser_on(self): pass
        def laser_off(self): pass
        def close(self): pass

# ===== 설정 =====
# MODEL_PATH = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\best_6.pt"
# NPZ_PATH   = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\calib_out\old_camera_same\stereo\stereo_params_scaled.npz"

NPZ_PATH       = r"/home/jsh/Desktop/JSH_CODE/jsh_code/stereo_params_scaled.npz"
MODEL_PATH     = r"/home/jsh/Desktop/JSH_CODE/jsh_code/best_5.pt"

LEFT_CAM, RIGHT_CAM = 1, 2
INIT_MERGE_FRAMES = 10
ROW_TOL_Y = 30
TOUCH_THRESHOLD = 10
TOUCH_COOLDOWN = 0.5
ROUTES_DIR = Path("./routes"); ROUTES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_COLORS = ["black","blue","gray","green","lime","orange","pink","purple","red","sky","white","yellow"]
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
    # centers: [(cx,cy,color,poly), ...]
    if not centers: return []
    pts = [(i,cx,cy) for i,(cx,cy,_,_) in enumerate(centers)]
    pts.sort(key=lambda t:(round(t[2]/ROW_TOL_Y), t[1]))  # y행→x
    idmap={}
    for new_id,(orig,_,_) in enumerate(pts): idmap[orig]=new_id
    out=[]
    for idx,(cx,cy,color,poly) in enumerate(centers):
        out.append((idmap[idx], cx, cy, color, poly))
    return out  # [(hid,cx,cy,color,poly)]

def _infer_batched(model, imgs, infer_hw, conf=0.25, iou=0.5):
    return model.predict(imgs, verbose=False, conf=conf, iou=iou, imgsz=infer_hw)

def _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, selected_color):
    """초기 N프레임 YOLO → 병합 후 홀드 고정."""
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
        cv2.putText(vis, f"Init YOLO merge {i+1}/{INIT_MERGE_FRAMES}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Left (Climb)", vis); cv2.imshow("Right (Climb)", fR)
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

# ----- stereo NPZ -----
def _detect_size_from_npz(npz):
    for k in ("map1L","map2L","map1R","map2R"):
        if k in npz: return (npz[k].shape[1], npz[k].shape[0])
    for k in ("img_size","image_size","size","wh","WH"):
        if k in npz:
            v = np.array(npz[k]).astype(int).ravel()
            if v.size>=2: return (int(v[0]), int(v[1]))
    if "W" in npz and "H" in npz: return (int(npz["W"]), int(npz["H"]))
    if "w" in npz and "h" in npz: return (int(npz["w"]), int(npz["h"]))
    return None

def load_stereo(npz_path: str):
    if not os.path.exists(npz_path):
        return (None, None, None, None, None, None)
    z = np.load(npz_path, allow_pickle=True)
    size = _detect_size_from_npz(z)
    if {"map1L","map2L","map1R","map2R"}.issubset(set(z.files)):
        return (size[0] if size else None, size[1] if size else None, z["map1L"], z["map2L"], z["map1R"], z["map2R"])
    try:
        K1, D1 = z["K1"], z["D1"]; K2, D2 = z["K2"], z["D2"]
        R1, R2 = z["R1"], z["R2"]; P1, P2 = z["P1"], z["P2"]
    except KeyError:
        return (size[0] if size else None, size[1] if size else None, None, None, None, None)
    if not size: return (None, None, None, None, None, None)
    w,h = size
    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w,h), cv2.CV_16SC2)
    return (w, h, map1L, map2L, map1R, map2R)

# ===== 매핑/도형 유틸 =====
def circle_poly(cx, cy, r=28, n=16):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([cx + r*np.cos(t), cy + r*np.sin(t)], axis=1).astype(np.int32)
    return pts

def map_csv_holds_to_polys(csv_rows, holdsL, max_dist=50):
    """
    csv_rows: [{'part','hold_id','cx','cy'}, ...] (순서 유지)
    holdsL: [(hid,cx,cy,color,poly), ...] — YOLO 기반 현재 폴리곤
    return: list of dict [{id, center, poly, color}]
      - csv의 hold_id와 현재 hid가 다를 수 있어, (cx,cy) 최근접 매칭
      - 못 맞추면 원형 대체 폴리곤
    """
    centers = np.array([[cx,cy] for (_,cx,cy,_,_) in holdsL], dtype=np.float32) if holdsL else np.zeros((0,2), np.float32)
    out=[]
    for row in csv_rows:
        hid = int(row["hold_id"])
        cx  = float(row["cx"]); cy = float(row["cy"])
        chosen_poly = None; chosen_color = "white"
        if centers.shape[0] > 0:
            d2 = np.sum((centers - np.array([cx,cy], dtype=np.float32))**2, axis=1)
            j = int(np.argmin(d2)); dist = float(np.sqrt(d2[j]))
            if dist <= max_dist:
                # YOLO 폴리곤 사용
                h, hx, hy, col, poly = holdsL[j]
                chosen_poly = np.array(poly, dtype=np.int32)
                chosen_color = col
        if chosen_poly is None:
            chosen_poly = circle_poly(int(cx), int(cy), r=30, n=18)
        out.append({"id":hid, "center":(int(cx),int(cy)), "poly":chosen_poly, "color":chosen_color})
    return out

def main():
    # --- 모드 표시
    update_state(mode="climb")

    # YOLO/웹 메타
    model = YOLO(MODEL_PATH)
    try:
        class_names = list(model.names.values())
    except Exception:
        class_names = None
    meta = choose_meta_via_web(yolo_class_names=class_names)
    csv_name = meta_to_csv_filename(meta)
    CSV_PATH = ROUTES_DIR / csv_name
    SELECTED = None if meta["color"] == "all" else meta["color"]
    print(f"[Info] 사용자 등반 모드, 메타={meta} → 경로: {CSV_PATH}")

    # CSV 존재 확인
    if not CSV_PATH.exists():
        # 안내만 띄우고 종료
        dummy = np.zeros((480, 640, 3), np.uint8)
        msg = "아직 저장된 경로가 없습니다"
        cv2.putText(dummy, msg, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow("Left (Climb)", dummy); cv2.imshow("Right (Climb)", dummy)
        update_state(records_count=0, last_record=["no-route",-1,-1,-1], fps=0.0, mode="climb")
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
        return

    # CSV 로드(순서 유지)
    csv_rows=[]
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_rows.append(row)
    if not csv_rows:
        print("[Error] CSV가 비어있습니다."); return

    # 스테레오/캠/포즈
    W, H, map1L, map2L, map1R, map2R = load_stereo(NPZ_PATH)
    if W is None or H is None: W, H = 1280, 720
    infer_hw = (_align32(H), _align32(W))

    capL = cv2.VideoCapture(LEFT_CAM, cv2.CAP_DSHOW if os.name=='nt' else 0)
    capR = cv2.VideoCapture(RIGHT_CAM, cv2.CAP_DSHOW if os.name=='nt' else 0)
    for c in (capL, capR):
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        c.set(cv2.CAP_PROP_FPS, 30)
    if not (capL.isOpened() and capR.isOpened()):
        print("[Error] 카메라 오픈 실패"); return

    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=0)
    touch = TouchCounter(threshold_frames=TOUCH_THRESHOLD, cooldown_sec=TOUCH_COOLDOWN)

    # 초기 YOLO 병합(단 1회)
    print("[Info] 초기 YOLO 병합 중...(이후 YOLO OFF)")
    holdsL, holdsR = _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, SELECTED)
    print(f"[Info] holdsL={len(holdsL)}, holdsR={len(holdsR)}")

    # CSV (cx,cy) ↔ 현재 폴리곤 매핑
    route_items = map_csv_holds_to_polys(csv_rows, holdsL, max_dist=60)  # 넉넉히 60px
    route_ids = [ int(r["hold_id"]) if isinstance(r, dict) else int(r["id"]) for r in csv_rows ]
    cur_idx = 0
    filled = set()

    servo = DualServoController()  # 옵션

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

            visL, visR = rectL.copy(), rectR.copy()
            coords = pose.process(rectL)

            # 현재 타깃
            current = route_items[cur_idx] if cur_idx < len(route_items) else None
            if current is not None:
                poly = current["poly"]
                cx, cy = current["center"]
                # 강조
                cv2.polylines(visL, [poly], True, (0,255,255), 3)
                cv2.putText(visL, f"[TARGET] ID:{current['id']}", (cx-40, cy-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                # 터치 체크
                triggered, parts = touch.check(poly, coords or {}, current["id"], now=time.time())
                if triggered:
                    filled.add(current["id"])
                    cur_idx += 1
                    # (옵션) 서보 이동/레이저 등은 프로젝트 보정치에 맞춰 확장
            # 완료 여부
            done = (cur_idx >= len(route_items))

            # 성공 홀드 칠하기 + 라벨
            for item in route_items:
                contour = item["poly"]
                hid = item["id"]; cx,cy = item["center"]
                if hid in filled:
                    overlay = visL.copy()
                    cv2.fillPoly(overlay, [contour], BGR.get(item.get("color","white"), (200,200,200)))
                    visL = cv2.addWeighted(overlay, 0.45, visL, 0.55, 0)
                cv2.polylines(visL, [contour], True, BGR.get(item.get("color","white"), (200,200,200)), 2)
                cv2.putText(visL, f"ID:{hid}", (cx-12, cy+24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            BGR.get(item.get("color","white"), (255,255,255)), 2, cv2.LINE_AA)

            # 포즈 점
            if coords: draw_pose_points(visL, coords, offset_x=0)

            # HUD
            t_now = time.time(); fps = 1.0 / max(t_now - t_prev, 1e-6); t_prev = t_now
            cv2.putText(visL, f"{meta['sector']} · {meta['level']} · {meta['color']}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(visL, f"route: {min(cur_idx,len(route_items))}/{len(route_items)}  (q:quit, r:reset, y:rescan)", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2, cv2.LINE_AA)
            cv2.putText(visL, f"FPS: {fps:.1f}", (10, visL.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)

            # 웹 상태 업데이트 (모드=climb)
            last_label = ["target", int(current["id"]) if current else -1, -1, -1]
            update_state(records_count=len(filled), last_record=last_label, fps=fps, mode="climb")

            # 웹 버튼 처리
            flags = consume_flags()
            if flags.get("reset"):
                filled.clear(); cur_idx = 0; touch.reset_all()
            if flags.get("rescan"):
                holdsL, holdsR = _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, None if meta["color"]=="all" else meta["color"])
                route_items = map_csv_holds_to_polys(csv_rows, holdsL, max_dist=60)
            if flags.get("stop"):
                break

            # 표시/키
            cv2.imshow("Left (Climb)", visL); cv2.imshow("Right (Climb)", visR)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('r'):
                filled.clear(); cur_idx = 0; touch.reset_all()
            elif k == ord('y'):
                holdsL, holdsR = _yolo_once_merge(model, capL, capR, map1L, map2L, map1R, map2R, infer_hw, None if meta["color"]=="all" else meta["color"])
                route_items = map_csv_holds_to_polys(csv_rows, holdsL, max_dist=60)

            if done:
                cv2.putText(visL, "Route Complete!", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3, cv2.LINE_AA)
                cv2.imshow("Left (Climb)", visL); cv2.waitKey(800)
                break
    finally:
        try: capL.release(); capR.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        try: pose.close()
        except: pass
        try: servo.close()
        except: pass

if __name__ == "__main__":
    main()
