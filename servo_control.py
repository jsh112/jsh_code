# servo_control.py
import argparse, time, sys
import serial

class DualServoController:
    def __init__(self, port="/dev/ttyUSB0", baud=115200, timeout=1.0):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        # UNO는 포트 열면 리셋되므로 잠깐 대기
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        # 웜업: 상태 한 번 읽어보기
        self.query()

    def _send(self, line: str) -> str:
        if not line.endswith("\n"):
            line += "\n"
        self.ser.write(line.encode("ascii"))
        self.ser.flush()
        try:
            resp = self.ser.readline().decode("ascii", errors="ignore").strip()
        except Exception:
            resp = ""
        return resp

    # 기존 함수
    def set_angles(self, pitch=None, yaw=None):
        if pitch is not None and yaw is not None:
            return self._send(f"S {float(pitch)} {float(yaw)}")
        elif pitch is not None:
            return self._send(f"P {float(pitch)}")
        elif yaw is not None:
            return self._send(f"Y {float(yaw)}")
        else:
            return "ERR no angles"

    def center(self):
        return self._send("C")

    def query(self):
        return self._send("Q")

    # 레이저 제어 추가
    def laser_on(self):
        return self._send("R")

    def laser_off(self):
        return self._send("Z")

    def close(self):
        try:
            self.ser.close()
        except:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--pitch", type=float)
    ap.add_argument("--yaw", type=float)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="양 축 60↔120도 스윕 테스트")
    ap.add_argument("--laser_on", action="store_true", help="레이저 켜기")
    ap.add_argument("--laser_off", action="store_true", help="레이저 끄기")
    args = ap.parse_args()

    ctl = DualServoController(args.port, args.baud)

    try:
        if args.center:
            print(ctl.center())
        if args.pitch is not None or args.yaw is not None:
            print(ctl.set_angles(args.pitch, args.yaw))
        if args.sweep:
            for a in range(60, 121, 5):
                print(ctl.set_angles(a, a))
                time.sleep(0.05)
            for a in range(120, 59, -5):
                print(ctl.set_angles(a, a))
                time.sleep(0.05)
        if args.laser_on:
            print(ctl.laser_on())
        if args.laser_off:
            print(ctl.laser_off())

        print(ctl.query())
    finally:
        ctl.close()

if __name__ == "__main__":
    main()