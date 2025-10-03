from pathlib import Path
import numpy as np
import argparse
# ---- User code ----
from make_module.climbing import StereoSystem
from make_module.find_laser import capture_once_and_return
# ---  User code ----
NPZ_PATH = r"/home/jsh/Desktop/JSH_CODE/jsh_code/stereo_params_scaled.npz"
# MODEL_PATH = "sdfsf"

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

def main():
    # # ap = argparse.ArgumentParser()
    # # ap.add_argument("--port", default="COM15")
    # # ap.add_argument("--baud", type=int, default=115200)
    # # ap.add_argument("--no_auto_advance", action="store_true")
    # # ap.add_argument("--no_web", action="store_true")
    # # args = ap.parse_args()
    #
    # for p in (NPZ_PATH, MODEL_PATH):
    #     if not Path(p).exists():
    #         raise FileNotFoundError(f"Cannot find {p}")
    # # 클래스 선언
    # stereo = StereoSystem(NPZ_PATH)
    # try:
    #     laser_raw = capture_once_and_return(
    #         port=args.port,
    #         baud=args.baud,
    #         wait_s=2.0,
    #         settle_n=8,
    #         show_preview=True,           # 히트맵 확인하고 싶으면 True
    #         center_pitch=90.0,           # ← 필수: 시작 90/90
    #         center_yaw=90.0,
    #         servo_settle_s=0.5
    #     )
    # except Exception as e:
    #     print(f"[A_Climbing] find_laser error: {e} → continue without laser")
    #     laser_raw = None
    #
    # if laser_raw is None:
    #     print("[A_Climbing] 레이저 좌표 취득 실패(취소/에러). 계속 진행.")
    #     laser_px = None
    # else:
    #     # 원본 좌표(보정 전)
    #     cam0_raw = laser_raw["cam0"]  # 보통 LEFT(=CAM1_INDEX=1)
    #     cam1_raw = laser_raw["cam1"]  # 보통 RIGHT(=CAM2_INDEX=2)
    #
    #     # npz에서 내부/왜곡/정렬행렬을 꺼내야 함
    #     # --- 원래 이 부분을 stereo.__init__에서 처리하기로 함        # S = np.load(NPZ_PATH, allow_pickle=True)
    #     # K1, D1, R1, P1_ = S["K1"], S["D1"], S["R1"], S["P1"]
    #     # K2, D2, R2, P2_ = S["K2"], S["D2"], S["R2"], S["P2"]
    #
    #     camL_rect = stereo.raw_to_rectified_point(
    #         cam0_raw, stereo.K1, stereo.D1, stereo.R1, stereo.p1)
    #     camR_rect = stereo.raw_to_rectified_point(
    #         cam1_raw, stereo.K2, stereo.D2, stereo.R2, stereo.p2)
    # def _verify_paths():
    # for p in (NPZ_PATH, MODEL_PATH):
    #     if not Path(p).exists():
    #         raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
    for p in (NPZ_PATH,):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    stereo = StereoSystem(NPZ_PATH)

    laser_px = _capture_laser_and_rectify(stereo, stereo.W, stereo.H)

    L, O = _compute_laser_origin_left()
    print(L, O)

if __name__ == "__main__":
    main()