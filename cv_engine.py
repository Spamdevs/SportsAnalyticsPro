import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Resilient MediaPipe Pose Import
pose_detector = None
mp_pose = None
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        from mediapipe.python.solutions import pose as mp_pose
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
except Exception:
    pose_detector = None
    mp_pose = None

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Traditional CV: HOG + Linear SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

try:
    sift = cv2.SIFT_create()
except Exception:
    sift = None


# ==================== TRADITIONAL COMPUTER VISION ====================

def detect_hog_svm(frame):
    rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    detections = []
    for (x, y, w, h), weight in zip(rects, weights):
        if weight > 0.15:
            detections.append({
                'bbox': [float(x), float(y), float(x + w), float(y + h)],
                'confidence': float(min(1.0, weight)),
                'class': 'person',
                'model': 'HOG + SVM'
            })
    return detections


def detect_sift_keypoints(frame):
    if sift is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gray, None)
    return [{'x': float(kp.pt[0]), 'y': float(kp.pt[1]), 'size': float(kp.size)} for kp in keypoints[:50]]


# ==================== TRACKING ENGINES ====================

def calculate_iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    wh = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - wh
    return wh / (union + 1e-6)


class IOUTracker:
    def __init__(self, sigma_iou=0.3):
        self.sigma_iou = sigma_iou
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            bbox = det if isinstance(det, list) else det['bbox']
            best_match = None
            best_iou = self.sigma_iou
            for trk in self.tracks:
                iou = calculate_iou(bbox, trk['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = trk

            if best_match is not None:
                best_match['bbox'] = bbox
                best_match['hits'] += 1
                best_match['age'] = 0
                updated_tracks.append((best_match['id'], bbox, 'IOU'))
            else:
                new_trk = {'id': self.next_id, 'bbox': bbox, 'hits': 1, 'age': 0}
                self.next_id += 1
                self.tracks.append(new_trk)
                updated_tracks.append((new_trk['id'], bbox, 'IOU'))

        self.tracks = [t for t in self.tracks if t.get('age', 0) < 15]
        for t in self.tracks:
            t['age'] = t.get('age', 0) + 1
        return updated_tracks


class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]
        ], dtype=np.float32)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)

        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        x = float(bbox[0] + w / 2.0)
        y = float(bbox[1] + h / 2.0)

        # 2D Column vector of shape (7, 1) for OpenCV gemm compatibility
        self.kf.statePost = np.array([[x], [y], [w * h], [w / float(h + 1e-6)], [0.0], [0.0], [0.0]], dtype=np.float32)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        x = float(bbox[0] + w / 2.0)
        y = float(bbox[1] + h / 2.0)

        meas = np.array([[x], [y], [w * h], [w / float(h + 1e-6)]], dtype=np.float32)
        self.kf.correct(meas)

    def predict(self):
        state = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        vals = state.flatten()
        x, y, s, r = vals[0], vals[1], max(vals[2], 1.0), max(vals[3], 0.1)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        return [float(x - w / 2.0), float(y - h / 2.0), float(x + w / 2.0), float(y + h / 2.0)]

    def get_state(self):
        """Returns current bounding box WITHOUT advancing Kalman filter or resetting time_since_update."""
        vals = self.kf.statePost.flatten()
        x, y, s, r = vals[0], vals[1], max(vals[2], 1.0), max(vals[3], 0.1)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        return [float(x - w / 2.0), float(y - h / 2.0), float(x + w / 2.0), float(y + h / 2.0)]


class SportsAnalyticsTracker:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections(detections, trks)

        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.time_since_update < 1:
                ret.append((trk.id, d, trk))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret

    def associate_detections(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), tuple(range(len(detections))), ()
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = calculate_iou(det, trk)

        cost_matrix = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        unmatched_dets = [d for d in range(len(detections)) if d not in row_ind]
        unmatched_trks = [t for t in range(len(trackers)) if t not in col_ind]

        matched = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matched.append((r, c))

        return np.array(matched), tuple(unmatched_dets), tuple(unmatched_trks)


def compute_pitch_homography(frame_width, frame_height):
    src_points = np.array([
        [frame_width * 0.15, frame_height * 0.85],
        [frame_width * 0.85, frame_height * 0.85],
        [frame_width * 0.70, frame_height * 0.25],
        [frame_width * 0.30, frame_height * 0.25]
    ], dtype=np.float32)

    dst_points = np.array([
        [0.0, 0.0],
        [PITCH_LENGTH, 0.0],
        [PITCH_LENGTH, PITCH_WIDTH],
        [0.0, PITCH_WIDTH]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_points, dst_points)
    return H


def transform_to_pitch(x_pixel, y_pixel, H):
    pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H)
    px = float(dst[0][0][0])
    py = float(dst[0][0][1])
    return max(0.0, min(PITCH_LENGTH, px)), max(0.0, min(PITCH_WIDTH, py))


def extract_jersey_color_hsv(frame, bbox):
    h_img, w_img, _ = frame.shape
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w_img, int(bbox[2])), min(h_img, int(bbox[3]))
    box_h = y2 - y1
    if box_h <= 5 or (x2 - x1) <= 5:
        return np.array([0, 0, 0], dtype=np.float32)
    jersey_crop = frame[y1 + int(box_h * 0.15):y1 + int(box_h * 0.55), x1:x2]
    if jersey_crop.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    hsv_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
    return np.mean(hsv_crop.reshape(-1, 3), axis=0)


def recognize_player_activity(player_crop, speed_kmh):
    if speed_kmh > 22.0:
        return "sprinting"
    elif speed_kmh > 8.0:
        return "jogging"

    if pose_detector is not None and player_crop.size > 0:
        try:
            rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_crop)
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                left_ankle_y = lms[mp_pose.PoseLandmark.LEFT_ANKLE].y
                right_ankle_y = lms[mp_pose.PoseLandmark.RIGHT_ANKLE].y
                hip_y = (lms[mp_pose.PoseLandmark.LEFT_HIP].y + lms[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2.0
                if (left_ankle_y < hip_y + 0.18) or (right_ankle_y < hip_y + 0.18):
                    return "jumping"
        except Exception:
            pass

    return "standing" if speed_kmh < 3.0 else "jogging"
