import cv2
import numpy as np

# --- 스테레오 파라미터 불러오기 ---
data = np.load(r"C:\Users\user\Documents\캡스턴 디자인\triangulation\capstonedesign_triangulation\calib_out\old_camera_same\stereo\stereo_params_scaled.npz")
P1 = data['P1']
P2 = data['P2']
Q  = data['Q']

# 영상 해상도
W, H = 640, 480  # npz에 없으면 직접 지정

# 카메라 열기
cap_left = cv2.VideoCapture(0)   # CAM_LEFT
cap_right = cv2.VideoCapture(2)  # CAM_RIGHT


cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 노출 모드
cap_left.set(cv2.CAP_PROP_EXPOSURE, 50)        # 적절한 값
cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap_right.set(cv2.CAP_PROP_EXPOSURE, 50)


# 노이즈 제거용 커널
# kernel = np.ones((3,3), np.uint8)
kernel = np.ones((5,5), np.uint8) # blur 처리해서 노이즈 더 제거

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not (retL and retR):
        print("카메라 입력 실패")
        break

    # --- 레이저 포인트 검출 (빨간색 기준 HSV) ---
    hsvL = cv2.cvtColor(frameL, cv2.COLOR_BGR2HSV)
    hsvR = cv2.cvtColor(frameR, cv2.COLOR_BGR2HSV)

    # 빨간색은 HSV 두 범위 사용
    lower_red1 = np.array([0, 50, 200])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 200])
    upper_red2 = np.array([180, 255, 255])


    # lower_red1 = np.array([0, 150, 150])
    # upper_red1 = np.array([10, 255, 255])
    # lower_red2 = np.array([160, 150, 150])
    # upper_red2 = np.array([180, 255, 255])


    maskL1 = cv2.inRange(hsvL, lower_red1, upper_red1)
    maskL2 = cv2.inRange(hsvL, lower_red2, upper_red2)
    maskR1 = cv2.inRange(hsvR, lower_red1, upper_red1)
    maskR2 = cv2.inRange(hsvR, lower_red2, upper_red2)

    maskL = cv2.bitwise_or(maskL1, maskL2)
    maskR = cv2.bitwise_or(maskR1, maskR2)

    # 노이즈 제거
    # maskL = cv2.morphologyEx(maskL, cv2.MORPH_OPEN, kernel)
    # maskR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernel)

    maskL = cv2.morphologyEx(maskL, cv2.MORPH_CLOSE, kernel)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_CLOSE, kernel)



    # 가장 큰 blob 찾기
    cntsL, _ = cv2.findContours(maskL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsR, _ = cv2.findContours(maskR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntsL = [c for c in cntsL if cv2.contourArea(c) > 1]
    cntsR = [c for c in cntsR if cv2.contourArea(c) > 1]

    if cntsL and cntsR:
        cL = max(cntsL, key=cv2.contourArea)
        cR = max(cntsR, key=cv2.contourArea)

        xL, yL, wL, hL = cv2.boundingRect(cL)
        xR, yR, wR, hR = cv2.boundingRect(cR)

        cxL, cyL = xL + wL // 2, yL + hL // 2
        cxR, cyR = xR + wR // 2, yR + hR // 2

        # 시각화
        cv2.rectangle(frameL, (xL, yL), (xL + wL, yL + hL), (0, 255, 0), 2)
        cv2.circle(frameL, (cxL, cyL), 5, (0, 0, 255), -1)
        cv2.rectangle(frameR, (xR, yR), (xR + wR, yR + hR), (0, 255, 0), 2)
        cv2.circle(frameR, (cxR, cyR), 5, (0, 0, 255), -1)

        # --- 삼각측량법 ---
        ptsL = np.array([[cxL], [cyL]], dtype=np.float32)
        ptsR = np.array([[cxR], [cyR]], dtype=np.float32)
        points_4d = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
        points_3d = points_4d[:3] / points_4d[3]
        Z = points_3d[2][0]  # 깊이(mm)
        cv2.putText(frameL, f"Distance: {Z:.1f} mm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # --- 영상 표시 ---
    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Right Camera", frameR)
    cv2.imshow("Mask Left", maskL)
    cv2.imshow("Mask Right", maskR)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

