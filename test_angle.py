import numpy as np

# 레이저 원점 O
O = np.array([-18.5, -80, -33], dtype=np.float64)

# Point 0
point0 = np.array([-99.85, -523.38, 3161.27], dtype=np.float64)
vec0 = point0 - O  # 레이저 좌표계로 변환

# Pitch (좌우, X/Z)
pitch0 = np.degrees(np.arctan2(vec0[0], vec0[2]))

# Yaw (위아래, Y/Z)
yaw0 = np.degrees(np.arctan2(vec0[1], vec0[2]))

print("Point 0 - Pitch:", pitch0, "Yaw:", yaw0)