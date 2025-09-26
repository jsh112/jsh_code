# mediapipe_pose_utils.py
import time
import cv2

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False



class PoseTracker:
    """
    - process(bgr) -> dict{name: (x,y)}  # 픽셀 좌표
    - close()
    """
    def __init__(self, min_detection_confidence=0.5, model_complexity=1):
        self.enabled = _HAS_MP
        if not self.enabled:
            self.pose = None
            return
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            model_complexity=model_complexity
        )
        # 기본 사용 랜드마크
        self.important_landmarks = {"left_index": 15, "right_index": 16}

    def process(self, bgr_image):
        """
        bgr_image: (H,W,3)
        return: dict{name->(x_px, y_px)}; mediapipe 미사용/실패 시 {}
        """
        if not self.enabled or self.pose is None or bgr_image is None:
            return {}
        h, w = bgr_image.shape[:2]
        res = self.pose.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        if not res or not res.pose_landmarks:
            return {}
        coords = {}
        for name, idx in self.important_landmarks.items():
            lm = res.pose_landmarks.landmark[idx]
            coords[name] = (lm.x * w, lm.y * h)
        return coords

    def close(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None


def draw_pose_points(vis, coords, offset_x=0, hand_names=("left_index", "right_index")):
    """
    vis: 표시용 이미지
    coords: dict{name:(x,y)}
    offset_x: 좌/우 합치기 시 왼쪽 프레임 x 오프셋
    """
    for name, (x, y) in coords.items():
        joint_color = (0, 0, 255) if name in hand_names else (0, 255, 0)
        cv2.circle(vis, (int(x) + offset_x, int(y)), 6, joint_color, -1)
        cv2.putText(vis, f"{name}:({int(x)},{int(y)})",
                    (int(x) + offset_x + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


class TouchCounter:
    """
    - 연속 프레임 터치 카운트 + 쿨다운 관리
    - check(contour, coords, target_id, now=None) -> (triggered, touched_parts)
      * triggered: True면 '터치 임계' 달성 (쿨다운 반영)
      * touched_parts: 이번 프레임에 in-polygon 판정된 파트 목록
    """
    def __init__(self, threshold_frames=10, cooldown_sec=0.5):
        self.threshold = int(max(1, threshold_frames))
        self.cooldown = float(max(0.0, cooldown_sec))
        self.counters = {}          # key=(part, target_id) -> count
        self.last_trigger_time = 0  # 마지막 트리거 시각(쿨다운)

    @staticmethod
    def _in_polygon(contour, pt):
        # pt: (x, y)
        return cv2.pointPolygonTest(contour, pt, False) >= 0

    def check(self, contour, coords, target_id, now=None):
        if now is None:
            now = time.time()
        touched_parts = []
        triggered = False

        for name, (x, y) in coords.items():
            key = (name, target_id)
            if self._in_polygon(contour, (x, y)):
                touched_parts.append(name)
                self.counters[key] = self.counters.get(key, 0) + 1
                if (self.counters[key] >= self.threshold and
                        now - self.last_trigger_time > self.cooldown):
                    triggered = True
                    self.last_trigger_time = now
            else:
                self.counters[key] = 0

        return triggered, touched_parts
