"""
Live gait metrics overlay (Raspberry Pi + IMX500 AI Camera)
-------------------------------------------------------------

Opens a live camera preview. HigherHRNet pose estimation runs on-chip on
the IMX500; the Pi CPU just takes the 17 COCO keypoints per frame and
computes five gait metrics, drawing only the ones it currently has enough
confident, recent data to compute. Nothing is captured or saved.

METRICS (each independently gated -- see "why it might not show" below):

  1. Walking speed
     Midpoint of the two hips, tracked across the rolling window;
     distance travelled / elapsed time. Shown in px/s always; shown in
     m/s too if you set PIXELS_PER_METER (see SETTINGS).
     Why it might not show: hips not confidently visible across the window.

  2. Stride-time variability
     Detects repeated peaks ("foot strikes") in each ankle's vertical
     position, takes the time between consecutive strikes of the same
     foot as a stride duration, and reports 100 x SD / mean across all
     detected strides in the window.
     Why it might not show: fewer than 2 full strides detected yet --
     needs a few seconds of continuous, confidently-tracked walking.

  3. Step-length variability
     Forward (x-axis) displacement between successive foot-strike events
     (left-then-right or right-then-left), reported as mean length and
     100 x SD / mean. Most meaningful from a side-on camera view.
     Why it might not show: same as above -- needs several alternating
     foot-strike events.

  4. Arm-swing amplitude asymmetry
     For each arm, the peak-to-peak range of (wrist_x - shoulder_x) over
     the window approximates swing amplitude. Asymmetry is
     |right - left| / (0.5 x (right + left)) x 100.
     Why it might not show: a shoulder or wrist is off-screen or occluded.

  5. Arm-leg phase coordination
     Cross-correlates the right-wrist / left-ankle motion signals and the
     left-wrist / right-ankle signals over the window, finds the time lag
     of peak correlation, and expresses it as a phase offset in degrees
     of the gait cycle (using the stride duration from metric 2 as the
     cycle length). Near 0 deg suggests strong normal-gait coordination
     between opposite limbs; larger offsets suggest looser coupling.
     Why it might not show: depends on metric 2's stride duration being
     available, plus confident wrist + ankle tracking for ~1.5+ strides.

Calibration and camera-angle notes:
  - Pixel-based numbers are only as good as your camera setup. Set
    PIXELS_PER_METER below (measure a known distance in the frame, count
    the pixels) to get walking speed in m/s as well as px/s.
  - Stride length and arm-swing amplitude assume the person is moving
    roughly left-right across the frame (classic side-view gait camera
    placement). A front-facing camera will distort these.
  - None of this is clinically validated -- treat trends over repeated
    passes as more informative than any single number.
"""

import sys
import time
from collections import deque

import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

# -----------------------------
# SETTINGS
# -----------------------------

POSE_MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

DETECTION_THRESHOLD = 0.3          # person/keypoint detection confidence
MIN_KEYPOINT_CONFIDENCE = 0.3      # per-joint confidence to trust a keypoint
METRIC_WINDOW_SECONDS = 8.0        # rolling window -- needs a few strides in it
MIN_STRIKE_INTERVAL_SECONDS = 0.3  # ignore double-counted peaks closer than this

# Set this to enable metric speed/length metrics in real units, e.g.
# if 500 pixels in your frame = 1 metre, set PIXELS_PER_METER = 500.0
PIXELS_PER_METER = None

WINDOW_SIZE_H_W = (480, 640)       # must match your ISP output resolution

# COCO keypoint indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# -----------------------------
# GLOBALS
# -----------------------------

last_boxes = None
last_scores = None
last_keypoints = None

# Set by main.py to enable anatomical movement recognition.
anatomy_engine = None


# -----------------------------
# GAIT METRICS
# -----------------------------

class GaitMetricsTracker:
    """Rolling history of one person's keypoints; derives the five metrics
    above, returning only the ones it currently has enough data for."""

    def __init__(self, window_seconds=METRIC_WINDOW_SECONDS,
                 min_confidence=MIN_KEYPOINT_CONFIDENCE):
        self.window_seconds = window_seconds
        self.min_confidence = min_confidence
        self.history = deque()  # (timestamp, keypoints[17,3])

    def update(self, keypoints, timestamp):
        self.history.append((timestamp, keypoints))
        while self.history and timestamp - self.history[0][0] > self.window_seconds:
            self.history.popleft()

    # ---- shared helper: foot-strike detection ----

    def _detect_strikes(self, ankle_idx):
        """Local maxima in ankle image-y (foot at its lowest point), with a
        minimum time gap so noise doesn't get double-counted. Returns a
        list of (timestamp, (x, y))."""
        samples = [
            (t, kp[ankle_idx][:2]) for t, kp in self.history
            if kp[ankle_idx][2] >= self.min_confidence
        ]
        if len(samples) < 3:
            return []

        events = []
        last_t = -1e9
        for i in range(1, len(samples) - 1):
            t, xy = samples[i]
            y_prev = samples[i - 1][1][1]
            y_next = samples[i + 1][1][1]
            if xy[1] >= y_prev and xy[1] >= y_next and (t - last_t) >= MIN_STRIKE_INTERVAL_SECONDS:
                events.append((t, xy))
                last_t = t
        return events

    # ---- the five metrics ----

    def walking_speed(self):
        valid = [
            (t, kp) for t, kp in self.history
            if kp[LEFT_HIP][2] >= self.min_confidence and kp[RIGHT_HIP][2] >= self.min_confidence
        ]
        if len(valid) < 2:
            return None

        t0, kp0 = valid[0]
        t1, kp1 = valid[-1]
        dt = t1 - t0
        if dt <= 0:
            return None

        mid0 = (kp0[LEFT_HIP][:2] + kp0[RIGHT_HIP][:2]) / 2.0
        mid1 = (kp1[LEFT_HIP][:2] + kp1[RIGHT_HIP][:2]) / 2.0
        speed_px_s = np.linalg.norm(mid1 - mid0) / dt

        result = {"px_per_s": speed_px_s}
        if PIXELS_PER_METER:
            result["m_per_s"] = speed_px_s / PIXELS_PER_METER
        return result

    def stride_time_variability(self):
        left_events = self._detect_strikes(LEFT_ANKLE)
        right_events = self._detect_strikes(RIGHT_ANKLE)

        intervals = []
        for events in (left_events, right_events):
            times = [t for t, _ in events]
            intervals.extend(np.diff(times).tolist())

        if len(intervals) < 2:
            return None

        mean = float(np.mean(intervals))
        sd = float(np.std(intervals))
        if mean <= 0:
            return None

        return {"cv_pct": 100.0 * sd / mean, "mean_stride_s": mean, "n_strides": len(intervals)}

    def step_length_variability(self):
        left_events = self._detect_strikes(LEFT_ANKLE)
        right_events = self._detect_strikes(RIGHT_ANKLE)

        combined = sorted(
            [(t, xy, "L") for t, xy in left_events] + [(t, xy, "R") for t, xy in right_events],
            key=lambda e: e[0],
        )
        if len(combined) < 3:
            return None

        lengths = []
        for i in range(1, len(combined)):
            _, xy_prev, foot_prev = combined[i - 1]
            _, xy, foot = combined[i]
            if foot != foot_prev:  # alternating feet = one step
                lengths.append(abs(xy[0] - xy_prev[0]))

        if len(lengths) < 2:
            return None

        mean = float(np.mean(lengths))
        result = {
            "mean_step_length_px": mean,
            "cv_pct": (100.0 * float(np.std(lengths)) / mean) if mean > 0 else 0.0,
            "n_steps": len(lengths),
        }
        if PIXELS_PER_METER:
            result["mean_step_length_m"] = mean / PIXELS_PER_METER
        return result

    def arm_swing_asymmetry(self):
        def amplitude(shoulder_idx, wrist_idx):
            samples = [
                kp[wrist_idx][0] - kp[shoulder_idx][0]
                for _, kp in self.history
                if kp[wrist_idx][2] >= self.min_confidence and kp[shoulder_idx][2] >= self.min_confidence
            ]
            if len(samples) < 3:
                return None
            return max(samples) - min(samples)

        left_amp = amplitude(LEFT_SHOULDER, LEFT_WRIST)
        right_amp = amplitude(RIGHT_SHOULDER, RIGHT_WRIST)
        if left_amp is None or right_amp is None:
            return None

        denom = 0.5 * (left_amp + right_amp)
        if denom <= 0:
            return None

        return {
            "left_amp_px": left_amp,
            "right_amp_px": right_amp,
            "asymmetry_pct": abs(right_amp - left_amp) / denom * 100.0,
        }

    def arm_leg_phase_coordination(self, stride_period):
        if not stride_period or stride_period <= 0:
            return None

        times = [t for t, _ in self.history]
        if len(times) < 5:
            return None
        t0, t1 = times[0], times[-1]
        if (t1 - t0) < stride_period * 1.5:
            return None  # need at least ~1.5 gait cycles in the window

        grid = np.linspace(t0, t1, num=min(200, len(self.history) * 2))
        dt_grid = (grid[-1] - grid[0]) / (len(grid) - 1) if len(grid) > 1 else None
        if not dt_grid:
            return None

        def series(idx):
            pts = [(t, kp[idx][1]) for t, kp in self.history if kp[idx][2] >= self.min_confidence]
            if len(pts) < 5:
                return None
            pts_t = np.array([p[0] for p in pts])
            pts_v = np.array([p[1] for p in pts])
            return np.interp(grid, pts_t, pts_v)

        def phase_offset_deg(sig_a, sig_b):
            if sig_a is None or sig_b is None:
                return None
            a = sig_a - np.mean(sig_a)
            b = sig_b - np.mean(sig_b)
            if np.std(a) == 0 or np.std(b) == 0:
                return None

            corr = np.correlate(a, b, mode="full")
            lags = np.arange(-len(a) + 1, len(a))
            center = len(a) - 1
            max_lag_samples = max(1, int((stride_period / 2) / dt_grid))

            lo, hi = center - max_lag_samples, center + max_lag_samples + 1
            window_corr = corr[max(lo, 0):min(hi, len(corr))]
            window_lags = lags[max(lo, 0):min(hi, len(lags))]
            if len(window_corr) == 0:
                return None

            best_idx = int(np.argmax(window_corr))
            lag_time = window_lags[best_idx] * dt_grid
            return (lag_time / stride_period) * 360.0

        rw_la = phase_offset_deg(series(RIGHT_WRIST), series(LEFT_ANKLE))
        lw_ra = phase_offset_deg(series(LEFT_WRIST), series(RIGHT_ANKLE))

        result = {}
        if rw_la is not None:
            result["right_wrist_left_ankle_phase_deg"] = rw_la
        if lw_ra is not None:
            result["left_wrist_right_ankle_phase_deg"] = lw_ra
        return result or None

    def compute_all(self):
        """Returns an ordered list of (metric_name, dict) for every metric
        that could be computed right now -- metrics with no data are simply
        left out, so if only one is available, only one is returned."""
        out = []

        speed = self.walking_speed()
        if speed:
            out.append(("Walking speed", speed))

        stride_time = self.stride_time_variability()
        if stride_time:
            out.append(("Stride-time variability", stride_time))

        step_length = self.step_length_variability()
        if step_length:
            out.append(("Step-length variability", step_length))

        arm_swing = self.arm_swing_asymmetry()
        if arm_swing:
            out.append(("Arm-swing asymmetry", arm_swing))

        stride_period = stride_time["mean_stride_s"] if stride_time else None
        phase = self.arm_leg_phase_coordination(stride_period)
        if phase:
            out.append(("Arm-leg phase coordination", phase))

        return out


tracker = GaitMetricsTracker()


def format_metric_lines(name, values):
    """Turn one metric's dict into the text line(s) drawn on screen."""
    if name == "Walking speed":
        if "m_per_s" in values:
            return [f"Walking speed: {values['m_per_s']:.2f} m/s ({values['px_per_s']:.0f} px/s)"]
        return [f"Walking speed: {values['px_per_s']:.0f} px/s"]

    if name == "Stride-time variability":
        return [f"Stride-time CV: {values['cv_pct']:.1f}% (n={values['n_strides']}, "
                f"mean {values['mean_stride_s']:.2f}s)"]

    if name == "Step-length variability":
        if "mean_step_length_m" in values:
            return [f"Step length: {values['mean_step_length_m']:.2f} m, "
                    f"CV {values['cv_pct']:.1f}% (n={values['n_steps']})"]
        return [f"Step length: {values['mean_step_length_px']:.0f}px, "
                f"CV {values['cv_pct']:.1f}% (n={values['n_steps']})"]

    if name == "Arm-swing asymmetry":
        return [f"Arm-swing asymmetry: {values['asymmetry_pct']:.1f}% "
                f"(L {values['left_amp_px']:.0f}px, R {values['right_amp_px']:.0f}px)"]

    if name == "Arm-leg phase coordination":
        lines = []
        if "right_wrist_left_ankle_phase_deg" in values:
            lines.append(f"R-wrist/L-ankle phase: {values['right_wrist_left_ankle_phase_deg']:.0f} deg")
        if "left_wrist_right_ankle_phase_deg" in values:
            lines.append(f"L-wrist/R-ankle phase: {values['left_wrist_right_ankle_phase_deg']:.0f} deg")
        return lines

    return []


# -----------------------------
# POSE PARSING (mirrors picamera2's official HigherHRNet demo)
# -----------------------------

def ai_output_tensor_parse(metadata: dict):
    global last_boxes, last_scores, last_keypoints

    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)

    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=WINDOW_SIZE_H_W,
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=DETECTION_THRESHOLD,
            network_postprocess=True,
        )

        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)
        else:
            last_boxes, last_scores, last_keypoints = None, None, None

    return last_boxes, last_scores, last_keypoints


def draw_overlay(request: CompletedRequest, boxes, scores, keypoints, metric_lines, anatomy_lines=None, stream="main"):
    with MappedArray(request, stream) as m:
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(
                m.array, boxes, scores, np.zeros(scores.shape), keypoints,
                DETECTION_THRESHOLD, DETECTION_THRESHOLD,
                request.get_metadata(), picam2, stream,
            )

        import cv2
        # Deliberately show only one movement status line.
        text = (anatomy_lines or ["Wrist pronation: NOT DETECTED"])[0]
        cv2.putText(m.array, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(m.array, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)


def pre_callback(request: CompletedRequest):
    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())

    # This build intentionally displays only the requested movement.
    status = "Wrist pronation: NOT DETECTED"
    if keypoints is not None and len(keypoints) > 0:
        best_idx = int(np.argmax(scores))
        if anatomy_engine is not None:
            observations = anatomy_engine.update(keypoints[best_idx], time.time())
            sides = sorted({obs["side"] for obs in observations if obs["id"] == "wrist_pronation"})
            if sides:
                status = "Wrist pronation: DETECTED (" + ", ".join(sides) + ")"

    draw_overlay(request, boxes, scores, keypoints, [], [status])


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    print("Loading pose model...")
    imx500 = IMX500(POSE_MODEL_PATH)
    intrinsics = imx500.network_intrinsics

    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Model is not a pose estimation model")
        sys.exit()

    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    categories = [c for c in intrinsics.labels if c and c != "-"]
    drawer = COCODrawer(categories, imx500, needs_rescale_coords=False)

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
    )

    imx500.show_network_fw_progress_bar()

    # show_preview=True opens the live window; overlay is drawn in
    # pre_callback. Nothing is captured or saved to disk.
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = pre_callback

    print("Live gait metrics running. Ctrl+C to stop.")
    print(f"Rolling window: {METRIC_WINDOW_SECONDS}s -- give it a few seconds of "
          "continuous walking before metrics appear.")

    while True:
        time.sleep(0.5)
