# import cv2

# cam1 = cv2.VideoCapture(0)  # 아래 포트
# cam2 = cv2.VideoCapture(1)  # 위 포트

# if not cam1.isOpened() or not cam2.isOpened():
#     print("카메라를 열 수 없습니다.")
#     exit()

# # 해상도/FPS 설정
# W, H, FPS = 1280, 720, 30
# for cam in (cam1, cam2):
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
#     cam.set(cv2.CAP_PROP_FPS, FPS)

# # 프레임 확인
# ok1, f1 = cam1.read()
# ok2, f2 = cam2.read()
# if ok1 and ok2:
#     print("Cam1 frame shape:", f1.shape)
#     print("Cam2 frame shape:", f2.shape)
# else:
#     print("초기 프레임을 읽지 못했습니다.")

# while True:
#     ret1, frame1 = cam1.read()
#     ret2, frame2 = cam2.read()
#     if not ret1 or not ret2:
#         print("프레임을 읽을 수 없습니다.")
#         break

#     frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

#     cv2.imshow('Cam1', frame1)
#     cv2.imshow('Cam2', frame2)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam1.release()
# cam2.release()
# cv2.destroyAllWindows()


# import cv2

# # 카메라 포트 확인 후 사용 (Nano에서 CSI는 /dev/video0, USB는 /dev/video1 등)
# cam1 = cv2.VideoCapture(1)  # 아래 포트
# cam2 = cv2.VideoCapture(0)  # 위 포트

# if not cam1.isOpened() or not cam2.isOpened():
#     print("카메라를 열 수 없습니다.")
#     exit()

# # === 안정적 해상도/FPS 설정 ===
# W, H, FPS = 640, 480, 60  # 고해상도/30fps는 Nano USB 대역폭 문제 발생 가능
# for cam in (cam1, cam2):
#     # 지원되는 코덱 사용 (MJPG 안정적)
#     cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
#     cam.set(cv2.CAP_PROP_FPS, FPS)

# # 창 생성
# cv2.namedWindow('Cam1', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Cam1', W, H)
# cv2.resizeWindow('Cam2', W, H)

# # 초기 프레임 확인
# ok1, f1 = cam1.read()
# ok2, f2 = cam2.read()
# if ok1 and ok2:
#     print("Cam1 frame shape:", f1.shape)
#     print("Cam2 frame shape:", f2.shape)
# else:
#     print("초기 프레임을 읽지 못했습니다.")

# while True:
#     # grab & retrieve로 안정적 프레임 읽기
#     cam1.grab()
#     cam2.grab()
#     ret1, frame1 = cam1.retrieve()
#     ret2, frame2 = cam2.retrieve()

#     if not ret1 or not ret2:
#         print("프레임을 읽을 수 없습니다.")
#         break

#     # 필요 시 90도 회전
#     frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

#     cv2.imshow('Cam1', frame1)
#     cv2.imshow('Cam2', frame2)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam1.release()
# cam2.release()
# cv2.destroyAllWindows()

import cv2

def open_camera(index):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        print(f"Camera {index}를 열 수 없습니다.")
        return None

    # 지원 가능한 안정적 해상도/FPS 목록 (Nano USB 대역폭 고려)
    resolutions = [(640, 480), (800, 600), (1280, 720)]
    fps_list = [15, 20, 30]

    # 안정적인 설정 자동 적용
    for W, H in resolutions:
        for FPS in fps_list:
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            cam.set(cv2.CAP_PROP_FPS, FPS)

            ret, frame = cam.read()
            if ret and frame is not None:
                print(f"Camera {index} 성공: {W}x{H} @ {FPS}fps")
                return cam, W, H, FPS
    print(f"Camera {index} 안정적 설정 실패")
    return None, None, None, None

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

