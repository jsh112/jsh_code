# 카메라 파라미터 확인할 수 있는 코드

import numpy as np
data = np.load('calib_out/20250907_025321/stereo/stereo_params_scaled.npz', allow_pickle=True)

# 여긴 각 카메라들의 내부 파라미터
# K = data['K']          # 3x3. 카메라 행렬
# dist = data['dist']    # 왜곡 계수(길이는 모델에 따라 다름)
# rmse = data['rmse']    # 평균 재투영 오차(px)
# size = data['image_size']   # (w, h)
# used = data['used_images']  # 사용된 파일 목록

# 여긴 두 카메라를 합친거 확인
K1, K2 = data['K1'], data['K2']
R1, R2 = data['R1'], data['R2']

print(f"K1:{K1}" + "\n" + f"K2:{K2}")
print(f"R1:{R1}" + "\n" + f"R2:{R2}")
