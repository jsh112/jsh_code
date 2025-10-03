import numpy as np
import cv2

class StereoSystem:
    def __init__(self, npz_path):
        S = np.load(npz_path,allow_pickle=True)
        self.K1, self.D1 = S["K1"], S["D1"]
        self.K2, self.D2 = S["K2"], S["D2"]
        self.R1, self.R2 = S["R1"], S["R2"]
        self._P1, self._P2 = S["P1"], S["P2"]
        self.W, self.H = [int(x) for x in S["image_size"]]
        self.size = (self.W, self.H)

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self._P1, self.size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self._P2, self.size, cv2.CV_32FC1)

        tx = -self._P2[0, 3] / -self._P2[0, 0]
        self.B = float(abs(tx))
        self.M = np.array([0.5 * tx, 0.0, 0.0], dtype=np.float64)  # 기준점(시각화시)
        print(f"[Info] image_size=({self.W},{self.H}), baseline~{self.B:.2f} mm")

    @property
    def p1(self):
        return self._P1

    @property
    def p2(self):
        return self._P2

    def open_cams(self, idx1, idx2):
        cap1 = cv2.VideoCapture(idx1, cv2.CAP_V4L2)
        cap2 = cv2.VideoCapture(idx2, cv2.CAP_V4L2)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)

        if not cap1.isOpened() or not cap2.isOpened():
            raise SystemExit("카메라 오픈 실패. 연결/권한 확인.")
        return cap1, cap2

    @staticmethod
    def raw_to_rectified_point(pt_xy,k,d,r,p):
        """
        원본(왜곡 포함) 픽셀 좌표 pt_xy -> 레티파이된 픽셀 좌표로 변환.
        K,D,R,P 는 stereo npz에서 읽은 해당 카메라 파라미터.
        """
        if pt_xy is None:
            return None
        # (x,y) -> (1,1,2) 형태로
        pts = np.array([[pt_xy]], dtype=np.float32)  # shape (1,1,2)
        # undistortPoints: 정규화 좌표로 보정 + R,P 적용하여 레티파이된 좌표로 변환
        # 결과 shape (1,1,2) 의 (x', y') 가 바로 레티파이된 픽셀 좌표
        rect = cv2.undistortPoints(pts, k, d, R=r, P=p)
        x_r, y_r = rect[0, 0, 0], rect[0, 0, 1]
        return (int(round(x_r)), int(round(y_r)))