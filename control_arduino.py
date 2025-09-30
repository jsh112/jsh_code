import serial
import struct
import time

# 아두이노 시리얼 연결
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # 아두이노 리셋 대기

def send_pwm(yaw, pitch):
    # 1000~2000 범위 체크
    yaw = max(1000, min(2000, yaw))
    pitch = max(1000, min(2000, pitch))

    # '<HH' : Little-endian, unsigned short 2개
    data = struct.pack('<HH', yaw, pitch)
    ser.write(data)

    # 아두이노에서 응답 읽기 (optional)
    if ser.in_waiting >= 4:
        resp = ser.read(4)
        r_yaw, r_pitch = struct.unpack('<HH', resp)
        print(f"[Arduino] PWM Yaw={r_yaw}, Pitch={r_pitch}")