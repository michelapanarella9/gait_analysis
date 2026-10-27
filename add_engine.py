import configparser
import math
import time
from collections import defaultdict, deque

import numpy as np

# COCO-17 keypoint indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12

KEYPOINTS = {
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
}


def load_movements(path="body_movements.txt"):
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
    )
    parser.read(path, encoding="utf-8")

    movements = []
    for section in parser.sections():
        item = dict(parser[section])
        item["id"] = section.strip()
        item["kind"] = item.get("kind", "static").strip().lower()
        item["rule"] = item.get("rule", "").strip().lower()

        for field in ("joints", "muscles", "bones", "nerves", "functions"):
            item[field] = [
                x.strip()
                for x in item.get(field, "").split(",")
                if x.strip()
            ]

        item["threshold"] = float(item.get("threshold", 0.6))
        item["min_duration"] = float(item.get("min_duration", 0.3))

        if not 0.0 <= item["threshold"] <= 1.0:
            raise ValueError(f"Invalid threshold in [{section}]")
        if item["min_duration"] < 0:
            raise ValueError(f"Invalid min_duration in [{section}]")

        movements.append(item)

    return movements


def joint_angle(a, b, c):
    """Angle ABC in degrees, with b as the vertex."""
    ba = np.asarray(a[:2], dtype=float) - np.asarray(b[:2], dtype=float)
    bc = np.asarray(c[:2], dtype=float) - np.asarray(b[:2], dtype=float)

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None

    cosine = np.dot(ba, bc) / denom
    return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))


class AnatomyEngine:
    def __init__(self, definitions, history_seconds=2.0,
                 min_keypoint_confidence=0.3):
        self.definitions = definitions
        self.history_seconds = history_seconds
        self.min_keypoint_confidence = min_keypoint_confidence

        # Each side and feature has an independent time history.
        self.history = defaultdict(deque)
        self.candidates = {}
        self.active = {}
        self.last_seen = {}

    def _point(self, kp, name):
        p = kp[KEYPOINTS[name]]
        if not np.all(np.isfinite(p)):
            return None
        if p[2] < self.min_keypoint_confidence:
            return None
        return p

    def _angle(self, kp, side):
        shoulder = self._point(kp, f"{side}_shoulder")
        elbow = self._point(kp, f"{side}_elbow")
        wrist = self._point(kp, f"{side}_wrist")

        if shoulder is None or elbow is None or wrist is None:
            return None

        return joint_angle(shoulder, elbow, wrist)

    def _rule_matches(self, rule, side, kp, now):
        if rule in ("wrist_above_elbow", "wrist_below_elbow"):
            wrist = self._point(kp, f"{side}_wrist")
            elbow = self._point(kp, f"{side}_elbow")

            if wrist is None or elbow is None:
                return None

            # Image y increases downward. This is not forearm rotation.
            above = wrist[1] < elbow[1]
            return above if rule == "wrist_above_elbow" else not above

        if rule in ("elbow_flexion", "elbow_extension"):
            angle = self._angle(kp, side)
            if angle is None:
                return None

            key = (side, "elbow_angle")
            samples = self.history[key]
            samples.append((now, angle))

            while samples and now - samples[0][0] > self.history_seconds:
                samples.popleft()

            # Compare early and late portions of the recent angle history.
            if len(samples) < 4 or now - samples[0][0] < 0.25:
                return False

            midpoint = len(samples) // 2
            early = float(np.median([v for _, v in list(samples)[:midpoint]]))
            late = float(np.median([v for _, v in list(samples)[midpoint:]]))
            delta = late - early

            if rule == "elbow_flexion":
                return delta < -5.0
            return delta > 5.0

        if rule == "shoulder_abduction":
            shoulder = self._point(kp, f"{side}_shoulder")
            elbow = self._point(kp, f"{side}_elbow")
            hip = self._point(kp, f"{side}_hip")

            if shoulder is None or elbow is None or hip is None:
                return None

            # Approximate angle relative to the image vertical axis.
            arm = np.asarray(elbow[:2]) - np.asarray(shoulder[:2])
            vertical = np.array([0.0, 1.0])
            denom = np.linalg.norm(arm)

            if denom < 1e-6:
                return None

            angle = math.degrees(math.acos(float(np.clip(
                np.dot(arm, vertical) / denom, -1, 1
            ))))

            key = (side, "shoulder_angle")
            samples = self.history[key]
            samples.append((now, angle))

            while samples and now - samples[0][0] > self.history_seconds:
                samples.popleft()

            if len(samples) < 4 or now - samples[0][0] < 0.25:
                return False

            midpoint = len(samples) // 2
            values = list(samples)
            early = float(np.median([v for _, v in values[:midpoint]]))
            late = float(np.median([v for _, v in values[midpoint:]]))
            return late - early > 5.0

        # Unsupported rules never produce a false positive.
        return None

    def update(self, keypoints, timestamp=None):
        now = time.monotonic() if timestamp is None else timestamp
        observations = []

        for definition in self.definitions:
            rule = definition["rule"]
            kind = definition["kind"]

            if kind == "unsupported":
                continue

            for side in ("left", "right"):
                # Keep each movement's state separate for each side.
                key = (definition["id"], side)
                match = self._rule_matches(rule, side, keypoints, now)

                if match is None or match is False:
                    self.candidates.pop(key, None)
                    self.active.pop(key, None)
                    self.last_seen.pop(key, None)
                    continue

                if key not in self.candidates:
                    self.candidates[key] = now

                self.last_seen[key] = now
                duration = now - self.candidates[key]

                if duration < definition["min_duration"]:
                    continue

                self.active[key] = True

                # Confidence here is the minimum confidence of required
                # joints, not a probability that the anatomical label is true.
                indices = [
                    KEYPOINTS[f"{side}_{joint}"]
                    for joint in definition["joints"]
                    if f"{side}_{joint}" in KEYPOINTS
                ]

                if not indices:
                    # Rules with no explicit index mapping use a conservative
                    # default. Add their required keypoints to the file/rule.
                    continue

                confidence = min(float(keypoints[i][2]) for i in indices)
                if confidence < definition["threshold"]:
                    continue

                observations.append({
                    "id": definition["id"],
                    "name": definition.get("name", definition["id"]),
                    "side": side,
                    "kind": kind,
                    "description": definition.get("description", ""),
                    "muscles": definition["muscles"],
                    "bones": definition["bones"],
                    "nerves": definition["nerves"],
                    "functions": definition["functions"],
                    "confidence": confidence,
                    "duration": duration,
                })

        # Unsupported movements remain visible as reference entries in the UI,
        # but are not reported as recognized.
        return observations