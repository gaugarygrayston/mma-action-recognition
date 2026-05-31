"""Annotate labeled frames with movement predictions from a saved sequence model.

This script loads a saved LSTM model and applies it to pairs of detected persons in
images from `data/sorted_frames_a13`. Annotated frames are written into
`results/labeled_frames_a13` by default.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.extraction import (
    build_feature_row,
    detect_people,
    draw_pose_landmarks,
    extract_frame_features,
    load_ssd_model,
    mask_image_with_box,
)

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'lstm_model.h5'
SOURCE_ROOT = Path(__file__).resolve().parent / 'data' / 'sorted_frames_a13'
OUTPUT_ROOT = Path(__file__).resolve().parent / 'results' / 'labeled_frames_a13'

SEQ_LENGTH = 16
LABEL_COLUMNS = ['ground', 'kick', 'not_engaged', 'punch', 'takedown']
FRAME_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def load_prediction_model():
    """Load the saved LSTM prediction model and disable training mode."""
    model = load_model(MODEL_PATH)
    model.trainable = False
    return model


def extract_feature_vector(image_path, ssd_model):
    """Read an image, detect persons, and build the numeric feature vector."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None, None

    boxes = detect_people(frame, ssd_model)
    if len(boxes) < 2:
        return None, None

    extracted = extract_frame_features(str(image_path), ssd_model)
    if extracted is None:
        return None, None

    person_boxes = boxes[:2]
    landmarks1, landmarks2 = extracted

    # Build a consistent feature vector for the model by reconstructing
    # the same feature order used during training.
    row = build_feature_row(
        landmarks1,
        landmarks2,
        image_path.stem,
        'unknown',
        'unknown',
        'a13',
    )
    if row is None:
        return None, None

    feature_vector = np.array(row[1:-3], dtype=np.float32)
    return feature_vector, (frame, person_boxes, [landmarks1, landmarks2])


def annotate_frame(frame, person_boxes, person_landmarks, pred_label, confidence):
    """Draw bounding boxes, pose skeletons, and the prediction label onto a frame."""
    annotated = frame.copy()

    box_colors = [(255, 0, 0), (0, 0, 255)]
    for idx, (box, landmarks) in enumerate(zip(person_boxes, person_landmarks)):
        y_min, x_min, y_max, x_max = box
        h, w = annotated.shape[:2]
        left = int(x_min * w)
        top = int(y_min * h)
        right = int(x_max * w)
        bottom = int(y_max * h)
        cv2.rectangle(annotated, (left, top), (right, bottom), box_colors[idx], 3)

        draw_pose_landmarks(
            annotated,
            landmarks,
            box=None,
            color=box_colors[idx],
            radius=4,
            thickness=2,
        )

    text = f"{pred_label} ({confidence:.1%})"
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 10
    box_x1 = 10
    box_y1 = 10
    box_x2 = 10 + text_size[0] + 2 * pad
    box_y2 = 10 + text_size[1] + baseline + 2 * pad
    cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        text,
        (box_x1 + pad, box_y2 - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return annotated


def main():
    """Process each frame sequence, predict movement, and save annotated output."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    ssd_model = load_ssd_model()
    lstm_model = load_prediction_model()

    group_buffers = defaultdict(list)
    processed = 0
    skipped = 0

    image_paths = sorted(SOURCE_ROOT.rglob('*'))
    image_paths = [path for path in image_paths if path.suffix.lower() in FRAME_EXTENSIONS]

    grouped_paths = defaultdict(list)
    for image_path in image_paths:
        relative_path = image_path.relative_to(SOURCE_ROOT)
        grouped_paths[str(relative_path.parent)].append(image_path)

    for group, paths in grouped_paths.items():
        paths = sorted(paths, key=lambda path: int(path.stem.split('_')[-1]))
        for image_path in paths:
            feature_vector, payload = extract_feature_vector(image_path, ssd_model)
            if feature_vector is None:
                skipped += 1
                continue

            buffer = group_buffers[group]
            buffer.append(feature_vector)
            window = buffer[-SEQ_LENGTH:]
            padded = np.zeros((SEQ_LENGTH, feature_vector.shape[0]), dtype=np.float32)
            padded[-len(window):] = np.vstack(window)
            seq = padded[np.newaxis, ...]

            probs = lstm_model.predict(seq, verbose=0)
            pred_idx = int(np.argmax(probs[0]))
            confidence = float(np.max(probs[0]))
            pred_label = LABEL_COLUMNS[pred_idx]

            frame, person_boxes, person_landmarks = payload
            annotated = annotate_frame(frame, person_boxes, person_landmarks, pred_label, confidence)

            rel_path = image_path.relative_to(SOURCE_ROOT)
            output_path = OUTPUT_ROOT / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

            if len(buffer) > SEQ_LENGTH:
                buffer.pop(0)

            processed += 1
            if processed % 100 == 0:
                print(f'Processed {processed} frames so far...')

    print(f'Annotated {processed} frames into {OUTPUT_ROOT}')
    print(f'Skipped {skipped} frames without two detected persons')


if __name__ == '__main__':
    main()
