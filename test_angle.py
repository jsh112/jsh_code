import numpy as np


def point_to_motor_angles(point, O):
    """
    point : np.array([X, Y, Z])  # mm, 카메라 좌표계
    O     : np.array([Xo, Yo, Zo]) # 레이저 원점 좌표
    Returns: pitch, yaw (degrees) # 모터 기준
    """
    vec = point - O  # 레이저 원점 기준 벡터

    # Pitch: 좌우 회전 (모터 기준: 좌+)
    # 좌표계 X가 오른쪽, Z가 앞으로 → 좌회전 +PWM, 우회전 -PWM
    pitch = np.degrees(np.arctan2(vec[0], vec[2]))  # X/Z
    # 모터 방향과 반대면 부호 반전 필요 시: pitch = -pitch

    # Yaw: 상하 회전 (모터 기준: 위+)
    # 좌표계 Y가 아래, Z가 앞으로 → 위+PWM, 아래-PWM
    yaw = np.degrees(np.arctan2(-vec[1], np.sqrt(vec[0]**2 + vec[2]**2)))

    return pitch, yaw

# 레이저 원점 O
O = np.array([18.5, -80, -33], dtype=np.float64)
# point0 = np.array([260.19, -566.05, 3660.13], dtype=np.float64)
# pitch0, yaw0 = point_to_motor_angles(point0, O)
# print("Point 0 - Pitch:", pitch0, "Yaw:", yaw0)

points = [
    np.array([260.19, -566.05, 3660.13]),
    np.array([143.58, -405.71, 3161.13]),
    np.array([276.53,  38.96, 3197.47]),
    np.array([160.82, 270.95, 3351.58])
]

for i, pt in enumerate(points):
    pitch, yaw = point_to_motor_angles(pt, O)
    print(f"Point {i} -> Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")