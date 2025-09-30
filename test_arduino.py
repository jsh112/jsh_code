import serial
import time

# 시리얼 포트 연결 (/dev/ttyACM0는 실제 연결 포트 확인 필요)
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # 아두이노 초기화 대기

# 테스트용 PWM 값 (Yaw, Pitch)
test_values = [
    (1500, 1500),  # 중앙
    #(1200, 1500),  # Yaw 왼쪽
    #(1800, 1500),  # Yaw 오른쪽
    #(1500, 1200),  # Pitch 아래
    #(1500, 1800),  # Pitch 위
]

for yaw, pitch in test_values:
    cmd = f"YAW,{yaw},PITCH,{pitch}\n"
    ser.write(cmd.encode())
    print(f"Sent: {cmd.strip()}")
    # 아두이노에서 받은 응답 읽기
    if ser.in_waiting:
        response = ser.readline().decode().strip()
        print("Arduino:", response)
    time.sleep(2)  # 움직임 확인용 대기

ser.close()
