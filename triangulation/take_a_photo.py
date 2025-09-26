import cv2
import numpy as np
import os
import time
import queue
import threading

# ==== 사용자 설정 ====
chessboard_size = (9, 6)     # 체스보드 "내부 코너" 개수. 나는 10 * 7 짜리니까 내부 코너 9 * 6임.
display_height = 480         # 보기용 리사이즈 높이
use_jpeg = True              # True면 JPG, False면 PNG
jpeg_quality = 95            # JPG 품질(높을수록 용량↑)
png_compression = 1          # PNG 압축(0~9, 낮을수록 빠름)
# ====================

def ensure_dirs(): 
    # cam1, cam2가 따로 잡으면 각각 cam1, cam2로 들어가지는데, 두 카메라에 체스보드가 동시에 잡히면 paris로 들어가짐.
    os.makedirs('captures/cam1', exist_ok=True) 
    os.makedirs('captures/cam2', exist_ok=True)
    os.makedirs('captures/pairs', exist_ok=True)

def timestamp():
    return time.strftime('%Y%m%d_%H%M%S') # 파일 이름 시간으로 

def resize_for_display(frame, height):
    # 미리보기 창 부하를 줄이기 위한 비율 리사이즈 (가로세로 비 유지)
    h, w = frame.shape[:2]
    new_w = int(w * (height / h))
    return cv2.resize(frame, (new_w, height))

def find_chessboard(gray, pattern_size):
    # 체스보드 코너 감지 + 서브픽셀 보정
    found, corners = cv2.findChessboardCorners( # 자동으로 코너 잡아주는 함수
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        # 코너 좌표 정밀화(서브픽셀). 캘리브레이션 품질에 매우 중요.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return found, corners

# ===== 비동기 저장 스레드 =====
class SaveWorker(threading.Thread):
    # 메인 스레드와 분리해 디스크 쓰기를 처리 -> 프레임 드랍/렉 방지
    def __init__(self, q: queue.Queue):
        super().__init__(daemon=True)
        self.q = q
        self.running = True

    def run(self):
        # 큐에서 (경로,영상,저장옵션)을 꺼내 파일로 기록
        while self.running:
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                path, img, use_jpg, jpg_q, png_c = item
                if use_jpg:
                    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
                else:
                    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, png_c])
                print(f"[saved] {path}")
            except Exception as e:
                print(f"[save error] {e}")
            finally:
                self.q.task_done()

def main():
    ensure_dirs()
    # 카메라 열고
    cam1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cam2 = cv2.VideoCapture(1, cv2.CAP_V4L2)

    # 카메라 좌, 우 해상도 다를 수 있는걸 여기서 고정해버려. 그럼 어차피 고정된 해상도의 사진이 저장되니까 캘리브레이션 돌리는데선 이 사진들만 쓰면 됨.
    W, H, FPS = 640, 480, 30
    for cam in (cam1, cam2):
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cam.set(cv2.CAP_PROP_FPS,          FPS)

    ret1, f1 = cam1.read(); ret2, f2 = cam2.read()
    print("cam1:", f1.shape, "cam2:", f2.shape)  # (H,W,3) 가 서로 같아야 함

    # (선택) 버퍼를 최소화해서 지연 줄이기 – 일부 드라이버만 지원
    cam1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam1.isOpened() or not cam2.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 저장 작업 큐 & 워커 시작
    save_q = queue.Queue(maxsize=128)
    worker = SaveWorker(save_q)
    worker.start()

    print("[키] 1: cam1 저장 | 2: cam2 저장 | b: 두 카메라 동시 저장 | c: 체스보드 감지 시 페어 저장 | q: 종료")
    ext = 'jpg' if use_jpeg else 'png'

    while True:
        # 두 카메라에서 프레임 읽기(동일 루프에서 읽어 동시성 최대화)
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if not ret1 or not ret2:
            print("프레임을 읽을 수 없습니다.")
            break

        # 보기용 리사이즈
        disp1 = resize_for_display(frame1, display_height)
        disp2 = resize_for_display(frame2, display_height)
        combined = np.hstack((disp2, disp2))  # 좌: cam1, 우: cam2

        # HUD: 안내 + 저장 큐 길이(현재 저장 작업이 얼마나 쌓였는지)
        hud = combined.copy()
        instr = "1: cam1 | 2: cam2 | b: BOTH | c: BOTH when chessboard | q: quit"
        cv2.putText(hud, instr, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(hud, instr, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        q_info = f"save-queue: {save_q.qsize()}"
        cv2.putText(hud, q_info, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(hud, q_info, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        # 키 입력 안내 문구가 포함된 미리보기 창
        cv2.imshow('Camera 1 (Left) | Camera 2 (Right) - Focus this window for keys', hud)

        # IME/한글 입력 환경에서의 키 인식 안정성을 높이기 위해 waitKeyEx 사용
        key = cv2.waitKeyEx(1) & 0xFF  # IME 환경에서도 더 안정적

        if key == ord('1'):
            # cam1 단독 저장 (현재 프레임을 그대로 디스크에)
            base = timestamp()
            path = f'captures/cam1/cam1_{base}.{ext}'
            try:
                save_q.put_nowait((path, frame1.copy(), use_jpeg, jpeg_quality, png_compression))
            except queue.Full:
                print("[warn] 저장 큐가 가득 찼습니다. 잠시 후 다시 시도하세요.")

        elif key == ord('2'):
            # cam2 단독 저장
            base = timestamp()
            path = f'captures/cam2/cam2_{base}.{ext}'
            try:
                save_q.put_nowait((path, frame2.copy(), use_jpeg, jpeg_quality, png_compression))
            except queue.Full:
                print("[warn] 저장 큐가 가득 찼습니다. 잠시 후 다시 시도하세요.")

        elif key == ord('b'):
            # 두 카메라 동시 저장(같은 타임스탬프 사용 → 페어 보장)
            base = timestamp()
            path1 = f'captures/pairs/{base}_cam1.{ext}'
            path2 = f'captures/pairs/{base}_cam2.{ext}'
            for p, img in [(path1, frame1), (path2, frame2)]:
                try:
                    save_q.put_nowait((p, img.copy(), use_jpeg, jpeg_quality, png_compression))
                except queue.Full:
                    print("[warn] 저장 큐가 가득 찼습니다. 잠시 후 다시 시도하세요.")

        elif key == ord('c'):
            # 두 카메라 모두에서 체스보드가 보이는 경우에만 페어 저장
            # 빠른 감지를 위해 다운스케일 이미지로 find → 성공 시 원본 프레임을 저장
            def quick_find(frame):
                scale = 0.5 if frame.shape[0] > 800 else 1.0
                small = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale))) if scale != 1.0 else frame
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                found, corners = find_chessboard(gray, chessboard_size)
                return found

            found1 = quick_find(frame1)
            found2 = quick_find(frame2)

            # HUD에 감지 상태 표시 (초록: 두 카메라 모두 OK / 빨강: 하나라도 실패)
            vis = combined.copy()
            status = f"Chessboard: cam1={'OK' if found1 else 'X'} | cam2={'OK' if found2 else 'X'}"
            cv2.putText(vis, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if (found1 and found2) else (0,0,255), 1, cv2.LINE_AA)
            cv2.imshow('Camera 1 (Left) | Camera 2 (Right) - Focus this window for keys', vis)

            if found1 and found2:
                # 둘 다 감지되면 페어 저장(파일명에 '_cb' 접미사)
                base = timestamp()
                path1 = f'captures/pairs/{base}_cam1_cb.{ext}'
                path2 = f'captures/pairs/{base}_cam2_cb.{ext}'
                for p, img in [(path1, frame1), (path2, frame2)]:
                    try:
                        save_q.put_nowait((p, img.copy(), use_jpeg, jpeg_quality, png_compression))
                    except queue.Full:
                        print("[warn] 저장 큐가 가득 찼습니다. 잠시 후 다시 시도하세요.")
                print("체스보드 감지 성공: 페어 저장 요청!")
            else:
                print("체스보드가 두 카메라 모두에서 보이지 않습니다.")

        elif key == ord('q'):
            # 종료
            break

    # 종료 처리: 장치/윈도우 닫기
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

    # 워커 정리(남은 저장 작업 마무리 시도)
    worker.running = False
    worker.join(timeout=0.5)

if __name__ == "__main__":
    main()
