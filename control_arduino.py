import serial
import time
import math

def calc_yaw_pitch(target_x, target_y, target_z, offset_x=0, offset_y=0, offset_z=0):
    # 레이저 오프셋 보정
    x = target_x - offset_x
    y = target_y - offset_y
    z = target_z - offset_z
    import math
    yaw = math.degrees(math.atan2(x, z))
    pitch = math.degrees(math.atan2(y, z))
    return yaw, pitch

def angle_to_pwm(angle, angle_range=60, pwm_center=1500, pwm_range=500):
    pwm = pwm_center + (angle / angle_range) * pwm_range
    return int(max(1000, min(2000, pwm)))  # PWM 제한

def move_laser(serial_port, target_xyz, offset=(0,0,0)):
    yaw, pitch = calc_yaw_pitch(*target_xyz, *offset)
    pwm_yaw = angle_to_pwm(yaw)
    pwm_pitch = angle_to_pwm(pitch)
    cmd = f"YAW,{pwm_yaw},PITCH,{pwm_pitch}\n"
    serial_port.write(cmd.encode())
