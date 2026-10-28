#!/usr/bin/env python3
"""Live test: shoulder-to-wrist distance change using the existing IMX500 COCO-17 model.

Run in the same Raspberry Pi project environment as gait_recognition.py:
    python3 test_elbow_distance.py

This diagnostic shows shoulder-to-wrist distance in pixels and its change over
0.6 seconds. Decreasing distance can accompany a curl; increasing distance can
accompany straightening. Camera angle and whole-body movement affect the result.
"""
import sys
import time
from collections import deque

import cv2
import numpy as np
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_higherhrnet import postprocess_higherhrnet

POSE_MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
DETECTION_THRESHOLD = 0.3
MIN_KEYPOINT_CONFIDENCE = 0.3
WINDOW_SIZE_H_W = (480, 640)
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
COMPARE_SECONDS = 0.6
MIN_CHANGE_PIXELS = 8.0

imx500 = None
picam2 = None
drawer = None
last_boxes = last_scores = last_keypoints = None
distance_history = {"left": deque(), "right": deque()}


def valid_point(keypoints, index):
    point = keypoints[index]
    if len(point) < 3 or not np.all(np.isfinite(point[:3])):
        return None
    if float(point[2]) < MIN_KEYPOINT_CONFIDENCE:
        return None
    return np.asarray(point[:2], dtype=float)


def distance_and_trend(keypoints, side, now):
    shoulder_idx = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    wrist_idx = LEFT_WRIST if side == "left" else RIGHT_WRIST
    shoulder = valid_point(keypoints, shoulder_idx)
    wrist = valid_point(keypoints, wrist_idx)
    if shoulder is None or wrist is None:
        distance_history[side].clear()
        return None, None, "TRACKING LOST"

    distance = float(np.linalg.norm(shoulder - wrist))
    history = distance_history[side]
    history.append((now, distance))
    while history and now - history[0][0] > COMPARE_SECONDS * 3:
        history.popleft()

    old_sample = None
    for sample_time, sample_distance in history:
        if now - sample_time >= COMPARE_SECONDS:
            old_sample = (sample_time, sample_distance)
    if old_sample is None:
        return distance, None, "COLLECTING"

    delta = distance - old_sample[1]
    if delta <= -MIN_CHANGE_PIXELS:
        trend = "DISTANCE DECREASING"
    elif delta >= MIN_CHANGE_PIXELS:
        trend = "DISTANCE INCREASING"
    else:
        trend = "NO CLEAR CHANGE"
    return distance, delta, trend


def put_text(image, text, xy, scale=0.55, color=(255, 255, 255)):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def parse_outputs(metadata):
    global last_boxes, last_scores, last_keypoints
    outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=outputs, img_size=WINDOW_SIZE_H_W,
            img_w_pad=(0, 0), img_h_pad=(0, 0),
            detection_threshold=DETECTION_THRESHOLD,
            network_postprocess=True,
        )
        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.asarray(box) for box in boxes]
            last_scores = np.asarray(scores)
        else:
            last_boxes, last_scores, last_keypoints = None, None, None
    return last_boxes, last_scores, last_keypoints


def pre_callback(request: CompletedRequest):
    boxes, scores, poses = parse_outputs(request.get_metadata())
    now = time.monotonic()
    with MappedArray(request, "main") as mapped:
        image = mapped.array
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(
                image, boxes, scores, np.zeros(scores.shape), poses,
                DETECTION_THRESHOLD, DETECTION_THRESHOLD,
                request.get_metadata(), picam2, "main",
            )

        y = 28
        put_text(image, "SHOULDER-WRIST DISTANCE TEST (pixels)", (10, y), 0.62)
        y += 28
        if poses is None or len(poses) == 0:
            put_text(image, "No pose detected - stand fully in frame", (10, y))
            distance_history["left"].clear()
            distance_history["right"].clear()
            return

        keypoints = poses[int(np.argmax(scores))]
        for side in ("left", "right"):
            distance, delta, trend = distance_and_trend(keypoints, side, now)
            y += 27
            if distance is None:
                line = f"{side.upper()}: shoulder/wrist confidence too low"
                color = (255, 255, 255)
            elif delta is None:
                line = f"{side.upper()}: distance={distance:.0f}px | {trend}"
                color = (255, 255, 255)
            else:
                line = (f"{side.upper()}: distance={distance:.0f}px | "
                        f"change={delta:+.0f}px/{COMPARE_SECONDS:.1f}s | {trend}")
                color = ((80, 220, 80) if trend == "DISTANCE DECREASING" else
                         (80, 180, 255) if trend == "DISTANCE INCREASING" else
                         (255, 255, 255))
            put_text(image, line, (10, y), 0.48, color)
        y += 30
        put_text(image, "Try a slow curl, pause, then slowly straighten.", (10, y), 0.48)
        y += 23
        put_text(image, "Distance is a proxy, not proof of elbow flexion/extension.", (10, y), 0.43)


def main():
    global imx500, picam2, drawer
    print("Loading IMX500 HigherHRNet COCO-17 pose model...")
    imx500 = IMX500(POSE_MODEL_PATH)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Error: configured model is not a pose-estimation model.")
        sys.exit(1)
    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r", encoding="utf-8") as label_file:
            intrinsics.labels = label_file.read().splitlines()
    intrinsics.update_with_defaults()
    categories = [label for label in intrinsics.labels if label and label != "-"]
    drawer = COCODrawer(categories, imx500, needs_rescale_coords=False)
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
    )
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = pre_callback
    print(f"Distance test running; comparing over {COMPARE_SECONDS:.1f} seconds. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping camera...")
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()
