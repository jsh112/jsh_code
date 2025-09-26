# 간단하게 카메라 켜는거부터 해본거

import cv2
import numpy as np

# 두 개 카메라 열기
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 내 노트북 기준 아래 포트
cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW) # 내 노트북 기준 위 포트

if not cam1.isOpened() or not cam2.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("프레임을 읽을 수 없습니다.")
        break

    # 두 카메라 해상도가 다를 수 있으니 크기 맞춰주기
    # height = min(frame1.shape[0], frame2.shape[0])
    # frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
    # frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))

    # 두 프레임을 가로로 합치기
    combined = np.hstack((frame1, frame2))

    # 합친 화면 출력
    # 오른쪽이 cam2, 왼쪽이 cam1
    cv2.imshow('Camera 1 & 2', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()
