#!/usr/bin/env python3
"""IMX500 elbow test: detect flexion/extension and draw localized muscle guides.

Run from the Raspberry Pi project folder with its normal system Python:
    /usr/bin/python3 elbow_muscle_overlay_test.py

Important: COCO-17 detects joints, not muscle borders or which side of the arm
faces the camera. The short colored lines are proportion-scaled visual guides,
not precise anatomical segmentation. A 2-D pose alone cannot reliably decide
whether the anterior or posterior surface is visible.
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
# COCO-17 zero-based indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
COMPARE_SECONDS = 0.65
ANGLE_CHANGE_DEG = 12.0

# BGR colors
FLEXION_COLOR = (70, 100, 255)     # coral/red: biceps/brachialis guide
EXTENSION_COLOR = (255, 160, 50)   # blue/orange: triceps guide
ANCO_COLOR = (40, 210, 255)        # yellow: anconeus guide
WHITE = (245, 245, 245)

imx500 = None
picam2 = None
drawer = None
last_boxes = last_scores = last_keypoints = None
angle_history = {"left": deque(), "right": deque()}
last_trend = {"left": "COLLECTING", "right": "COLLECTING"}
last_trend_time = {"left": 0.0, "right": 0.0}


def point_for(kps, idx):
    p = kps[idx]
    if len(p) < 3 or not np.all(np.isfinite(p[:3])) or float(p[2]) < MIN_KEYPOINT_CONFIDENCE:
        return None
    return np.array(p[:2], dtype=float)


def elbow_angle(shoulder, elbow, wrist):
    """Angle at elbow in degrees: near 180 when straight; smaller when flexed."""
    a = shoulder - elbow
    b = wrist - elbow
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return None
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def movement_trend(kps, side, now):
    si, ei, wi = ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST) if side == "left"
                  else (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))
    shoulder, elbow, wrist = point_for(kps, si), point_for(kps, ei), point_for(kps, wi)
    if shoulder is None or elbow is None or wrist is None:
        angle_history[side].clear()
        last_trend[side] = "TRACKING LOST"
        return None, "TRACKING LOST", shoulder, elbow, wrist

    angle = elbow_angle(shoulder, elbow, wrist)
    if angle is None:
        return None, "TRACKING LOST", shoulder, elbow, wrist
    hist = angle_history[side]
    hist.append((now, angle))
    while hist and now - hist[0][0] > COMPARE_SECONDS * 3:
        hist.popleft()
    old = None
    for t, a in hist:
        if now - t >= COMPARE_SECONDS:
            old = a
            break
    if old is None:
        trend = "COLLECTING"
    else:
        delta = angle - old
        # Smaller internal elbow angle = flexion; larger = extension.
        if delta <= -ANGLE_CHANGE_DEG:
            trend = "ELBOW FLEXION"
            last_trend[side], last_trend_time[side] = trend, now
        elif delta >= ANGLE_CHANGE_DEG:
            trend = "ELBOW EXTENSION"
            last_trend[side], last_trend_time[side] = trend, now
        elif now - last_trend_time[side] < 1.0 and last_trend[side] not in ("COLLECTING", "TRACKING LOST"):
            trend = last_trend[side]
        else:
            trend = "NO CLEAR CHANGE"
    return angle, trend, shoulder, elbow, wrist


def put_text(img, text, xy, scale=0.48, color=WHITE):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_muscle_guide(img, shoulder, elbow, side, movement):
    """Draw a short, proportional guide over the upper arm, not a full muscle shape."""
    if shoulder is None or elbow is None or movement not in ("ELBOW FLEXION", "ELBOW EXTENSION"):
        return
    vec = elbow - shoulder
    length = float(np.linalg.norm(vec))
    if length < 20:
        return
    unit = vec / length
    perp = np.array([-unit[1], unit[0]])
    # Highlight only the middle section of the upper arm. The offset is a visual
    # convention; COCO keypoints cannot tell which anatomical surface is facing camera.
    start = shoulder + 0.22 * vec
    end = shoulder + 0.68 * vec
    offset = perp * max(3.0, min(9.0, length * 0.07))
    if movement == "ELBOW FLEXION":
        color = FLEXION_COLOR
        label = "Biceps brachii / brachialis"
        # A short line plus a small parallel accent to suggest a localized region.
        p1, p2 = start + offset, end + offset
    else:
        color = EXTENSION_COLOR
        label = "Triceps brachii"
        p1, p2 = start - offset, end - offset
    thickness = max(3, int(length * 0.035))
    cv2.line(img, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), color, thickness, cv2.LINE_AA)
    # Small endpoint dots make the highlighted segment easier to see.
    cv2.circle(img, tuple(np.round(p1).astype(int)), max(3, thickness), color, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(p2).astype(int)), max(3, thickness), color, -1, cv2.LINE_AA)
    mid = (p1 + p2) / 2
    label_x = int(mid[0] + (12 if side == "right" else -170))
    label_y = int(mid[1] - 8)
    label_x = max(6, min(img.shape[1] - 240, label_x))
    label_y = max(48, min(img.shape[0] - 8, label_y))
    # Leader from line midpoint to label anchor.
    anchor = np.array([label_x + (0 if side == "right" else 165), label_y - 4], dtype=float)
    cv2.line(img, tuple(np.round(mid).astype(int)), tuple(np.round(anchor).astype(int)), color, 1, cv2.LINE_AA)
    put_text(img, label, (label_x, label_y), 0.42, color)
    if movement == "ELBOW EXTENSION":
        # Anconeus is near the posterolateral elbow; approximate marker only.
        a = shoulder + 0.92 * vec - offset * 0.35
        cv2.circle(img, tuple(np.round(a).astype(int)), max(3, thickness - 1), ANCO_COLOR, -1, cv2.LINE_AA)
        put_text(img, "Anconeus (approx.)", (max(6, min(img.shape[1] - 190, int(a[0] + 10))),
                                             max(48, min(img.shape[0] - 8, int(a[1] + 18)))), 0.38, ANCO_COLOR)


def parse_outputs(metadata):
    global last_boxes, last_scores, last_keypoints
    outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=outputs, img_size=WINDOW_SIZE_H_W, img_w_pad=(0, 0), img_h_pad=(0, 0),
            detection_threshold=DETECTION_THRESHOLD, network_postprocess=True)
        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes, last_scores = [np.asarray(b) for b in boxes], np.asarray(scores)
        else:
            last_boxes = last_scores = last_keypoints = None
    return last_boxes, last_scores, last_keypoints


def pre_callback(request: CompletedRequest):
    boxes, scores, poses = parse_outputs(request.get_metadata())
    now = time.monotonic()
    with MappedArray(request, "main") as mapped:
        img = mapped.array
        if boxes is not None and len(boxes):
            drawer.annotate_image(img, boxes, scores, np.zeros(scores.shape), poses,
                                  DETECTION_THRESHOLD, DETECTION_THRESHOLD,
                                  request.get_metadata(), picam2, "main")
        put_text(img, "ELBOW MOVEMENT + LOCAL MUSCLE GUIDE TEST", (10, 25), 0.52)
        if poses is None or len(poses) == 0:
            put_text(img, "No pose detected - stand fully in frame", (10, 52), 0.45)
            return
        kps = poses[int(np.argmax(scores))]
        y = 52
        for side in ("left", "right"):
            angle, trend, shoulder, elbow, wrist = movement_trend(kps, side, now)
            if angle is None:
                status = f"{side.upper()}: {trend}"
            else:
                status = f"{side.upper()}: elbow angle {angle:.0f} deg | {trend}"
            put_text(img, status, (10, y), 0.43, FLEXION_COLOR if trend == "ELBOW FLEXION" else EXTENSION_COLOR if trend == "ELBOW EXTENSION" else WHITE)
            y += 23
            draw_muscle_guide(img, shoulder, elbow, side, trend)
        put_text(img, "Flexion: angle decreases | Extension: angle increases", (10, img.shape[0] - 30), 0.42)
        put_text(img, "Muscle lines are proportional guides, not exact muscle borders", (10, img.shape[0] - 10), 0.35)


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
        try:
            with open("assets/coco_labels.txt", "r", encoding="utf-8") as f:
                intrinsics.labels = f.read().splitlines()
        except FileNotFoundError:
            print("Could not find assets/coco_labels.txt. Run from your project folder or add that file.")
            sys.exit(1)
    intrinsics.update_with_defaults()
    categories = [label for label in intrinsics.labels if label and label != "-"]
    drawer = COCODrawer(categories, imx500, needs_rescale_coords=False)
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = pre_callback
    print("Elbow test running. Slowly bend, pause, then straighten your elbow. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping camera...")
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()
