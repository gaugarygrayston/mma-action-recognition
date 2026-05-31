"""
Feature extraction module for video movement classification.

This module extracts pose and movement features from labeled frame folders.
It uses person detection (SSD MobileNet) and MediaPipe Pose estimation to build
feature vectors for movement classification.

This code is compatible with a working Python 3.12 environment pinned to:
- mediapipe==0.10.13
- tensorflow==2.16.2
- protobuf==4.25.3
- opencv-contrib-python==4.11.0.86
- numpy==1.26.4

The extraction pipeline performs person detection, crops the person regions,
and extracts pose landmarks for the selected joints.
"""

import csv
import glob
import math
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Constants for detection and pose extraction
SSD_MODEL_DIR = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
POSE_MIN_CONFIDENCE = 0.2
DETECTION_SCORE_THRESHOLD = 0.2
MAX_PERSON_BOXES = 2

# Landmark indices for the joints we want to keep (COCO keypoints).
LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
KEYPOINT_NAMES = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
ANGLE_CONNECTIONS = [
    ('left_shoulder', 'left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow', 'right_wrist'),
    ('left_hip', 'left_shoulder', 'left_elbow'),
    ('right_hip', 'right_shoulder', 'right_elbow'),
    ('left_hip', 'left_knee', 'left_ankle'),
    ('right_hip', 'right_knee', 'right_ankle'),
    ('left_shoulder', 'left_hip', 'left_knee'),
    ('right_shoulder', 'right_hip', 'right_knee')
]

LABEL_MAP_FILE = 'data/class_dictionary.txt'

mp_pose = mp.solutions.pose


def load_ssd_model(model_dir=SSD_MODEL_DIR):
    """Load and return the saved SSD MobileNet model."""
    return tf.saved_model.load(model_dir)


def non_max_suppression(boxes, scores, iou_threshold=0.35):
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    scores = tf.convert_to_tensor(scores, dtype=tf.float32)
    indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        iou_threshold=iou_threshold,
        max_output_size=MAX_PERSON_BOXES,
        score_threshold=DETECTION_SCORE_THRESHOLD,
    )
    return tf.gather(boxes, indices).numpy()


def detect_people(frame, model, max_boxes=MAX_PERSON_BOXES):
    """Detect person boxes in an image frame using the SSD model."""
    resized = cv2.resize(frame, (320, 320))
    tensor = tf.convert_to_tensor(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), dtype=tf.uint8)
    tensor = tensor[tf.newaxis, ...]

    detections = model(tensor)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    person_boxes = [box for box, score, cls in zip(boxes, scores, classes)
                    if score >= DETECTION_SCORE_THRESHOLD and cls == 1]
    person_scores = [score for score, cls in zip(scores, classes)
                     if score >= DETECTION_SCORE_THRESHOLD and cls == 1]

    if not person_boxes:
        return []

    selected = non_max_suppression(np.array(person_boxes), np.array(person_scores))
    largest = sorted(selected, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    return largest[:max_boxes]


def mask_image_with_box(frame, box):
    """Return a masked image containing only the region inside the detected box."""
    mask = np.zeros_like(frame)
    h, w = frame.shape[:2]
    y_min, x_min, y_max, x_max = box
    top = int(y_min * h)
    left = int(x_min * w)
    bottom = int(y_max * h)
    right = int(x_max * w)
    mask[top:bottom, left:right] = frame[top:bottom, left:right]
    return mask


def estimate_pose(image):
    """Estimate pose landmarks for a single image crop."""
    with mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=POSE_MIN_CONFIDENCE) as pose:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pose.process(rgb).pose_landmarks


def filter_landmarks(landmarks):
    """Keep only the landmarks that are useful for our movement features."""
    if not landmarks:
        return []
    return [landmarks.landmark[idx] for idx in LANDMARK_INDICES]


def draw_pose_landmarks(image, landmarks, box=None, color=(0, 255, 0), radius=4, thickness=2):
    """Draw filtered pose points and limb segments back onto the full image."""
    if image is None or not landmarks:
        return image

    h, w = image.shape[:2]
    if box is None:
        x0, y0 = 0, 0
        crop_w, crop_h = w, h
    else:
        y_min, x_min, y_max, x_max = box
        x0 = int(x_min * w)
        y0 = int(y_min * h)
        crop_w = max(1, int(x_max * w) - x0)
        crop_h = max(1, int(y_max * h) - y0)

    points = []
    for landmark in landmarks:
        px = int(x0 + landmark.x * crop_w)
        py = int(y0 + landmark.y * crop_h)
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        points.append((px, py))
        cv2.circle(image, (px, py), radius, color, -1)

    pose_connections = [
        (0, 2), (2, 4),
        (1, 3), (3, 5),
        (0, 6), (1, 7),
        (6, 8), (8, 10),
        (7, 9), (9, 11),
    ]

    for start, end in pose_connections:
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], color, thickness, cv2.LINE_AA)

    return image


def are_poses_similar(pose1, pose2, threshold=0.1):
    """Compare two poses by landmark distance to avoid identity swapping."""
    if not pose1 or not pose2 or len(pose1) != len(pose2):
        return False

    similar = 0
    for a, b in zip(pose1, pose2):
        distance = math.hypot(a.x - b.x, a.y - b.y)
        if distance < threshold:
            similar += 1

    return similar / len(pose1) > 0.7


def calculate_angle(p1, p2, p3):
    """Compute the angle at point p2 formed by p1-p2-p3."""
    def v(a, b):
        return b[0] - a[0], b[1] - a[1]

    def dot(u, v2):
        return u[0] * v2[0] + u[1] * v2[1]

    def norm(u):
        return math.hypot(u[0], u[1])

    u = v(p2, p1)
    v2 = v(p2, p3)
    denom = norm(u) * norm(v2)
    if denom == 0:
        return 0.0

    cos_angle = max(min(dot(u, v2) / denom, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))


def load_label_map(label_map_file=LABEL_MAP_FILE):
    """Load the raw-to-simplified label mapping from the project text file."""
    if not os.path.exists(label_map_file):
        return {}
    content = Path(label_map_file).read_text(encoding='utf-8').strip()
    return eval(content)


def get_feature_header():
    header = ['Frame']
    for prefix in ('Person1', 'Person2'):
        for keypoint in KEYPOINT_NAMES:
            header.append(f'{prefix}_{keypoint}_x')
            header.append(f'{prefix}_{keypoint}_y')
    for keypoint in KEYPOINT_NAMES:
        header.append(f'Diff_{keypoint}_x')
        header.append(f'Diff_{keypoint}_y')
    for conn in ANGLE_CONNECTIONS:
        header.append(f'Angle_{conn[0]}_{conn[1]}_{conn[2]}_P1')
        header.append(f'Angle_{conn[0]}_{conn[1]}_{conn[2]}_P2')
    header.extend(['Label', 'Label_Simplified', 'Device'])
    return header


def build_feature_row(landmarks1, landmarks2, frame_id, raw_label, simplified_label, device):
    """Build a feature row from two pose landmark sets."""
    if len(landmarks1) != len(LANDMARK_INDICES) or len(landmarks2) != len(LANDMARK_INDICES):
        return None

    row = [frame_id]
    for landmarks in (landmarks1, landmarks2):
        for lm in landmarks:
            row.append(lm.x)
            row.append(lm.y)

    for lm1, lm2 in zip(landmarks1, landmarks2):
        row.append(abs(lm2.x - lm1.x))
        row.append(abs(lm2.y - lm1.y))

    for conn in ANGLE_CONNECTIONS:
        idx1 = KEYPOINT_NAMES.index(conn[0])
        idx2 = KEYPOINT_NAMES.index(conn[1])
        idx3 = KEYPOINT_NAMES.index(conn[2])

        p1 = (landmarks1[idx1].x, landmarks1[idx1].y)
        p2 = (landmarks1[idx2].x, landmarks1[idx2].y)
        p3 = (landmarks1[idx3].x, landmarks1[idx3].y)
        row.append(calculate_angle(p1, p2, p3))

        p1 = (landmarks2[idx1].x, landmarks2[idx1].y)
        p2 = (landmarks2[idx2].x, landmarks2[idx2].y)
        p3 = (landmarks2[idx3].x, landmarks2[idx3].y)
        row.append(calculate_angle(p1, p2, p3))

    row.extend([raw_label, simplified_label, device])
    return row


def extract_frame_features(image_path, model):
    """Extract joint and angle features from a single frame image."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    boxes = detect_people(frame, model)
    if len(boxes) < 2:
        return None

    masked1 = mask_image_with_box(frame, boxes[0])
    masked2 = mask_image_with_box(frame, boxes[1])
    landmarks1 = filter_landmarks(estimate_pose(masked1))
    landmarks2 = filter_landmarks(estimate_pose(masked2))

    return landmarks1, landmarks2


def require_images(directory):
    """Return a sorted list of image paths from a directory."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for pattern in extensions:
        files.extend(sorted(Path(directory).glob(pattern)))
    return [str(path) for path in files]


def extract_folder_to_csv(folder_path, output_csv_path, label=None, device='unknown', model=None):
    """Extract features from every image in a labeled frame folder."""
    if model is None:
        model = load_ssd_model()

    label_map = load_label_map()
    raw_label = label or os.path.basename(folder_path)
    simplified_label = label_map.get(raw_label, raw_label)

    image_paths = require_images(folder_path)
    if not image_paths:
        return 0

    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(get_feature_header())

        count = 0
        for image_path in image_paths:
            frame_id = Path(image_path).stem
            result = extract_frame_features(image_path, model)
            if result is None:
                continue
            landmarks1, landmarks2 = result
            row = build_feature_row(landmarks1, landmarks2, frame_id, raw_label, simplified_label, device)
            if row is not None:
                writer.writerow(row)
                count += 1

    return count


def extract_dataset_from_frame_root(source_root, output_csv_path, device='unknown'):
    """Walk a frame dataset root and extract a labeled CSV for every clip folder."""
    model = load_ssd_model()
    label_map = load_label_map()
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(get_feature_header())

        count = 0
        for root, dirs, files in os.walk(source_root):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not image_files:
                continue

            raw_label = os.path.basename(root)
            simplified_label = label_map.get(raw_label, raw_label)
            image_paths = sorted([os.path.join(root, f) for f in image_files])

            for image_path in image_paths:
                frame_id = Path(image_path).stem
                result = extract_frame_features(image_path, model)
                if result is None:
                    continue
                landmarks1, landmarks2 = result
                row = build_feature_row(landmarks1, landmarks2, frame_id, raw_label, simplified_label, device)
                if row is not None:
                    writer.writerow(row)
                    count += 1

    return count
