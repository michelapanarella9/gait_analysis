RASPBERRY PI 5 — SIMPLE CAMERA MUSCLE-LABEL STARTER
===================================================

What this version does
----------------------
Uses the existing Raspberry Pi 5 + IMX500 HigherHRNet COCO-17 camera pipeline.
It does not install a new pose model or change the camera hardware setup.

It starts with these camera-keypoint rules:
- Elbow flexion: elbow angle is decreasing across recent frames.
- Elbow extension: elbow angle is increasing across recent frames.
- Shoulder abduction: the upper arm moves away from the image-vertical axis.
- Wrist below elbow: a simple positional cue only. It is NOT a direct measure
  of forearm pronation and does not claim muscle activity.

When elbow flexion/extension or shoulder abduction is detected, it draws a
crude line on the visible person and labels likely muscles. The lines are rough
anatomical teaching estimates, not muscle-boundary measurements. The camera
cannot directly detect electrical muscle activity (EMG).

Files
-----
main.py                 Starts the camera and anatomy engine
 gait_recognition.py    Existing IMX500 pose pipeline with overlay additions
add_engine.py            Movement rules and landmark confidence handling
body_movements.txt       Editable movement/muscle definitions

Installation
------------
1. Copy these files into the same folder on your Raspberry Pi as the working
   camera project. Keep any required existing assets folder, including
   assets/coco_labels.txt, alongside them.
2. Confirm gait.POSE_MODEL_PATH in gait_recognition.py points to your installed
   IMX500 HigherHRNet model. The default path in this project is:
   /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk
3. Use the Python environment where Picamera2, NumPy, and the Raspberry Pi
   camera libraries already work.
4. Run from that folder:
   python main.py
5. Stop with Ctrl+C.

Important
---------
- This package has been syntax-checked, but the camera itself cannot be tested
  away from your Raspberry Pi hardware.
- Use a clear, well-lit view where shoulders, elbows, and wrists are visible.
- Elbow flexion/extension is detected from movement over time; hold the
  movement in view and move at a moderate speed.
- The existing shoulder-abduction rule uses image geometry, so camera angle,
  body rotation, and occlusion can affect detection.
- To add movements, add a section to body_movements.txt only if its rule is
  already implemented in add_engine.py. Existing supported rules are
  wrist_above_elbow, wrist_below_elbow, elbow_flexion, elbow_extension, and
  shoulder_abduction. Shoulder adduction is not yet implemented.
