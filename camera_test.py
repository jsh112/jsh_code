import cv2

cam1 = cv2.VideoCapture(0)  # 아래 포트
cam2 = cv2.VideoCapture(1)  # 위 포트

if not cam1.isOpened() or not cam2.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 해상도/FPS 설정
W, H, FPS = 1280, 720, 30
for cam in (cam1, cam2):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cam.set(cv2.CAP_PROP_FPS, FPS)

# 프레임 확인
ok1, f1 = cam1.read()
ok2, f2 = cam2.read()
if ok1 and ok2:
    print("Cam1 frame shape:", f1.shape)
    print("Cam2 frame shape:", f2.shape)
else:
    print("초기 프레임을 읽지 못했습니다.")

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    if not ret1 or not ret2:
        print("프레임을 읽을 수 없습니다.")
        break

    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('Cam1', frame1)
    cv2.imshow('Cam2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()
