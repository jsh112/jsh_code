from pathlib import Path
import numpy as np
import argparse
# ---- User code ----
from make_module.climbing import StereoSystem
from make_module.find_laser import capture_once_and_return
# ---  User code ----
NPZ_PATH       = "/home/jsh/Desktop/JSH_CODE/jsh_code/stereo_params_scaled.npz"
MODEL_PATH     = "/home/jsh/Desktop/JSH_CODE/jsh_code/best_5.pt"

# === (NEW) 웹 모듈 - 색상 선택 ===
_USE_WEB = True

# 런타임 보정 오프셋(레이저 실측)
CAL_YAW_OFFSET   = 0.0
CAL_PITCH_OFFSET = 0.0

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 1.15
LASER_OFFSET_CM_UP   = 5.2
LASER_OFFSET_CM_FWD  = -0.6
Y_UP_IS_NEGATIVE     = True  # 위 방향이 -y인 좌표계면 True

def _capture_laser_and_rectify(stereo: StereoSystem,W, H):
    # ===== (NEW) 레이저 좌표 먼저 측정 (find_laser) =====
    try:
        laser_raw = capture_once_and_return(
            port="/dev/ttyUSB0",
            baud=115200,
            wait_s=2.0,
            settle_n=8,
            show_preview=True,           # 히트맵 확인하고 싶으면 True
            center_pitch=90.0,           # ← 필수: 시작 90/90
            center_yaw=90.0,
            servo_settle_s=0.5
        )
    except Exception as e:
        print(f"[A_Climbing] find_laser error: {e} → continue without laser")
        laser_raw = None

    if laser_raw is None:
        print("[A_Climbing] 레이저 좌표 취득 실패(취소/에러). 계속 진행.")
        return None

    # 원본 좌표(보정 전)
    cam0_raw = laser_raw["cam0"]  # 보통 LEFT(=CAM1_INDEX=1)
    cam1_raw = laser_raw["cam1"]  # 보통 RIGHT(=CAM2_INDEX=2)

    # npz에서 내부/왜곡/정렬행렬을 꺼내야 함
    K1, D1, R1, P1_ = stereo.K1,stereo.D1,stereo.R1, stereo.p1
    K2, D2, R2, P2_ = stereo.K2,stereo.D2,stereo.R2, stereo.p2

    # 원본→레티파이 좌표로 변환
    camL_rect = StereoSystem.raw_to_rectified_point(cam0_raw, K1, D1, R1, P1_) if cam0_raw else None
    camR_rect = StereoSystem.raw_to_rectified_point(cam1_raw, K2, D2, R2, P2_) if cam1_raw else None

    laser_px = {
        "left_rect":  camL_rect,   # Lr 좌표계
        "right_rect": camR_rect,   # Rr 좌표계
        "image_size": (W, H),
    }
    print(f"[A_Climbing] 레이저(원본): L={cam0_raw}, R={cam1_raw}")
    print(f"[A_Climbing] 레이저(레티파이): L={camL_rect}, R={camR_rect}")
    return laser_px

def _compute_laser_origin_left():
    # 레이저 원점 O (LEFT 기준)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = L + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, LEFT-based) = {O}")
    return L, O

def _choose_color(args):
    # ================= 색상 필터 선택 =================
    selected_class_name  = None     # 모델 클래스명 (예: Hold_Green)
    selected_color_label = "all"    # 파일명 라벨 (예: green, orange, all)

    # 1) 웹에서 색상 선택(가능하고 --no_web가 아닐 때)
    if (not args.no_web) and _USE_WEB:
        try:
            chosen = choose_color_via_web(
                all_colors=list(ALL_COLORS.keys()),
                defaults={"port": args.port, "baud": args.baud}
            )  # ""이면 전체
            if chosen:
                mapped = ALL_COLORS.get(chosen)
                if mapped is None:
                    print(f"[Filter] 웹 선택 '{chosen}' 무효 → 전체 표시")
                else:
                    print(f"[Filter] 웹 선택: {chosen} → {mapped}")
                    selected_class_name  = mapped
                    selected_color_label = chosen.lower()
            else:
                print("[Filter] 웹에서 전체 선택")
        except Exception as e:
            print(f"[Filter] 웹 선택 실패 → 콘솔 대체: {e}")

    # 2) 고정 설정 값
    if (selected_class_name is None) and (SELECTED_COLOR is not None):
        sc = SELECTED_COLOR.strip().lower()
        mapped = ALL_COLORS.get(sc)
        if mapped is None:
            print(f"[Filter] SELECTED_COLOR='{SELECTED_COLOR}' 무효 → 콘솔에서 선택")
            selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 고정 선택 클래스: {mapped}")
            selected_class_name  = mapped
            selected_color_label = sc

    # 3) 콘솔 입력 대체
    if selected_class_name is None and (args.no_web or not _USE_WEB):
        selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)

    # === 여기서 색상 라벨에 맞춰 CSV 파일명 생성 ===
    csv_label = _sanitize_label(selected_color_label) if selected_color_label else "all"
    CSV_GRIPS_PATH_dyn = f"grip_records_{csv_label}.csv"
    print(f"[Info] 경로 CSV: {CSV_GRIPS_PATH_dyn}")
    return selected_class_name, selected_color_label, CSV_GRIPS_PATH_dyn

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM15")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no_auto_advance", action="store_true")
    ap.add_argument("--no_web", action="store_true")
    return ap.parse_args()


def main():
    # def _verify_paths():
    # for p in (NPZ_PATH, MODEL_PATH):
    #     if not Path(p).exists():
    #         raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
    for p in (NPZ_PATH,MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    stereo = StereoSystem(NPZ_PATH)

    laser_px = _capture_laser_and_rectify(stereo, stereo.W, stereo.H)

    L, O = _compute_laser_origin_left()
    print(L, O)

    selected_class_name, selected_color_label, CSV_GRIPS_PATH_dyn = _choose_color(args)

if __name__ == "__main__":
    main()