import cv2

def open_camera(index, width=640, height=480, fps=15):
    """
    V4L2 백엔드로 카메라 열기, 안정적인 해상도/FPS 자동 설정
    """
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cam.isOpened():
        print(f"Camera {index}를 열 수 없습니다.")
        return None, None, None, None

    # 안정적인 기본 설정
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, fps)

    # 실제 적용 해상도 확인
    ret, frame = cam.read()
    if not ret or frame is None:
        print(f"Camera {index} 초기 프레임 읽기 실패")
        return None, None, None, None

    H, W = frame.shape[:2]
    print(f"Camera {index} 실제 해상도: {W}x{H} @ {fps}fps")
    return cam, W, H, fps

# 카메라 열기
cam1, W1, H1, FPS1 = open_camera(0)
cam2, W2, H2, FPS2 = open_camera(1)

if cam1 is None or cam2 is None:
    exit()

# 창 생성
cv2.namedWindow('Cam1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cam1', W1, H1)
cv2.resizeWindow('Cam2', W2, H2)

while True:
    # 안정적 프레임 읽기
    cam1.grab()
    cam2.grab()
    ret1, frame1 = cam1.retrieve()
    ret2, frame2 = cam2.retrieve()

    if not ret1 or not ret2:
        print("프레임을 읽을 수 없습니다.")
        break

    # 필요 시 90도 회전
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('Cam1', frame1)
    cv2.imshow('Cam2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()
