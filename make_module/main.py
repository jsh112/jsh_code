from pathlib import Path
import argparse
# ---- User code ----
from make_module.climbing import (StereoSystem)
from make_module.find_laser import capture_once_and_return
# ---  User code ----
NPZ_PATH = "sdf"
MODEL_PATH = "sdfsf"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM15")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no_auto_advance", action="store_true")
    ap.add_argument("--no_web", action="store_true")
    args = ap.parse_args()

    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"Cannot find {p}")
    # 클래스 선언
    stereo = StereoSystem(NPZ_PATH)
    try:
        laser_raw = capture_once_and_return(
            port=args.port,
            baud=args.baud,
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
        laser_px = None
    else:
        # 원본 좌표(보정 전)
        cam0_raw = laser_raw["cam0"]  # 보통 LEFT(=CAM1_INDEX=1)
        cam1_raw = laser_raw["cam1"]  # 보통 RIGHT(=CAM2_INDEX=2)

    # npz에서 내부/왜곡/정렬행렬을 꺼내야 함
    # --- 원래 이 부분을 stereo.__init__에서 처리하기로 함        # S = np.load(NPZ_PATH, allow_pickle=True)
    # K1, D1, R1, P1_ = S["K1"], S["D1"], S["R1"], S["P1"]
    # K2, D2, R2, P2_ = S["K2"], S["D2"], S["R2"], S["P2"]

    camL_rect = stereo.raw_to_rectified_point(
        cam0_raw, stereo.K1, stereo.D1, stereo.R1, stereo.P1)
    camR_rect = stereo.raw_to_rectified_point(
        cam1_raw, stereo.K1, stereo.D1, stereo.R1, stereo.P1)