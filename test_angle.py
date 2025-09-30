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
point0 = np.array([279.99, -558.70, 3612.58], dtype=np.float64)

# PWM 변환 (하드코딩 테스트용)
def angles_to_pwm(pitch_deg, yaw_deg, k_pitch=12, k_yaw=12, center=1500):
    pwm_pitch = int(center + pitch_deg * k_pitch)
    pwm_yaw   = int(center + yaw_deg * k_yaw)
    
    # 서보 제한
    pwm_pitch = max(1200, min(1800, pwm_pitch))
    pwm_yaw   = max(1200, min(1800, pwm_yaw))
    return pwm_pitch, pwm_yaw

# 계산
pitch_deg, yaw_deg = point_to_motor_angles(point0, O)
pwm_pitch, pwm_yaw = angles_to_pwm(pitch_deg, yaw_deg)

print(f"Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}°")
print(f"PWM -> Pitch={pwm_pitch}us, Yaw={pwm_yaw}us")

# pitch0, yaw0 = point_to_motor_angles(point0, O)
# print("Point 0 - Pitch:", pitch0, "Yaw:", yaw0)

# points = [
#     np.array([261.45, -558.70, 3612.60]),
#     np.array([497.07,-417.25, 3477.08]),
#     np.array([629.89, 2-.31,3521.27]),
#     np.array([542.01, 239.52, 3659.99])
# ]

# for i, pt in enumerate(points):
#     pitch, yaw = point_to_motor_angles(pt, O)
#     print(f"Point {i} -> Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")