"""
Live gait + anatomy movement display for Raspberry Pi 5 with IMX500 AI Camera.
Keep this file with gait_recognition.py, add_engine.py, body_movements.txt.
"""
import time
from pathlib import Path

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer

import gait_recognition as gait
from add_engine import AnatomyEngine, load_movements


def main():
    project_dir = Path(__file__).resolve().parent
    definitions_path = project_dir / "body_movements.txt"
    model_path = Path(gait.POSE_MODEL_PATH)

    if not definitions_path.is_file():
        raise FileNotFoundError(f"Movement definitions not found: {definitions_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"IMX500 pose model not found: {model_path}. Check POSE_MODEL_PATH."
        )

    movements = load_movements(str(definitions_path))
    gait.anatomy_engine = AnatomyEngine(
        movements, min_keypoint_confidence=gait.MIN_KEYPOINT_CONFIDENCE
    )
    print("Movement enabled: Wrist pronation (wrist lower than elbow)")

    print("Loading IMX500 pose model...")
    gait.imx500 = IMX500(str(model_path))
    intrinsics = gait.imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        raise RuntimeError("Configured IMX500 model is not a pose-estimation model.")

    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        labels_path = project_dir / "assets" / "coco_labels.txt"
        if not labels_path.is_file():
            raise FileNotFoundError(
                f"Missing assets/coco_labels.txt required by the Picamera2 demo."
            )
        intrinsics.labels = labels_path.read_text(encoding="utf-8").splitlines()

    intrinsics.update_with_defaults()
    categories = [c for c in intrinsics.labels if c and c != "-"]
    gait.drawer = COCODrawer(categories, gait.imx500, needs_rescale_coords=False)
    gait.picam2 = Picamera2(gait.imx500.camera_num)
    config = gait.picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
    )

    gait.imx500.show_network_fw_progress_bar()
    gait.picam2.start(config, show_preview=True)
    gait.imx500.set_auto_aspect_ratio()
    gait.picam2.pre_callback = gait.pre_callback

    print("Live wrist-pronation display running. Ctrl+C to stop.")
    print("The label appears when the wrist stays lower than the elbow.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping camera...")
    finally:
        gait.picam2.stop()
        gait.picam2.close()


if __name__ == "__main__":
    main()
