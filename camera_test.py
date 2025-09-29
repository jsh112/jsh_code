# Jetson nano에서 사용
import cv2
import numpy as np

cam1 = cv2.VideoCapture(1)  # 아래 포트
cam2 = cv2.VideoCapture(2)  # 위 포트

if not cam1.isOpened() or not cam2.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# === 해상도/포맷/FPS 설정 ===
W, H, FPS = 1280, 720, 30
for cam in (cam1, cam2):
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 고해상도에서 안정적
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cam.set(cv2.CAP_PROP_FPS,          FPS)

# 창 생성 & 위치 지정(선택)
cv2.namedWindow('Cam1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
cv2.moveWindow('Cam1', 50, 50)
cv2.moveWindow('Cam2', 700, 50)
cv2.resizeWindow('Cam1', 1280, 720)  # 보기용 크기 (선택)
cv2.resizeWindow('Cam2', 1280, 720)

# 실제 적용 해상도 확인
ok1, f1 = cam1.read()
ok2, f2 = cam2.read()
if ok1 and ok2:
    print("Cam1 frame shape:", f1.shape)  # (H, W, C)
    print("Cam2 frame shape:", f2.shape)
else:
    print("초기 프레임을 읽지 못했습니다.")

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    if not ret1 or not ret2:
        print("프레임을 읽을 수 없습니다.")
        break

    # 각 프레임을 90도 반시계 회전 (요청 코드 유지)
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('Cam1', frame1)
    cv2.imshow('Cam2', frame2)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):  # q 또는 ESC로 종료
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()
