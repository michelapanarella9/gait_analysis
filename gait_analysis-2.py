"""
gait_analysis.py
=================

Implementation of the clinician-oriented 3D gait analysis (3DGA) method described in:

    Tanabe S, et al. "Clinical-oriented Three-dimensional Gait Analysis Method
    for Evaluating Gait Disorder." J Vis Exp. 2018 (PMC5931438).

The paper describes:
  1. A minimal 12-marker set (both acromia, iliac crests, hips, knees, ankles, toes)
     placed for treadmill-based 3D motion capture.
  2. Spatiotemporal parameters (step cycle time, gait speed, stance time, step width,
     cadence).
  3. A virtual center of gravity (COG), estimated from segment weight fractions:
         trunk = 0.66, thigh = 0.10 (each side), shank = 0.05 (each side),
         foot = 0.02 (each side)   [these sum to 1.0: 0.66 + 2*(0.10+0.05+0.02)]
  4. Sagittal-plane joint angles (hip, knee, ankle).
  5. A "Lissajous Overview Picture" (LOP): the trajectories of the 10 limb
     markers + the virtual COG, plotted pairwise in the horizontal (x-y),
     sagittal (y-z), and coronal (z-x) planes, averaged (mean +/- SD) over
     multiple gait cycles. This is the paper's signature "holistic, intuitive"
     output meant to replace long stacks of joint-angle-vs-time graphs.

IMPORTANT ASSUMPTIONS / WHAT YOU MAY NEED TO ADAPT
---------------------------------------------------
The paper's original workflow used proprietary 3D motion-capture software
(KinemaTracer) that already outputs synchronized, labeled, gap-filled marker
trajectories, and used a treadmill/instrumented-mat-based method for detecting
heel contact events. Since none of that hardware/software context is available
here, this script assumes:

  * INPUT: a CSV file with a `time` column (seconds) and, for each of the 12
    markers, three columns `<MARKER>_X`, `<MARKER>_Y`, `<MARKER>_Z` (millimeters),
    already in a lab coordinate system where:
        X = medial-lateral (left/right)
        Y = anterior-posterior (direction of walking)
        Z = vertical (superior-inferior)
    (this matches the x,y,z convention named explicitly in the paper).

  * MARKER LABELS (12 total, one per side unless noted):
        LACR, RACR   - left/right acromion
        LIC,  RIC    - left/right iliac crest
        LHIP, RHIP   - left/right hip (1/3 point from greater trochanter to ASIS)
        LKNE, RKNE   - left/right lateral knee epicondyle
        LANK, RANK   - left/right lateral malleolus (ankle)
        LTOE, RTOE   - left/right 5th metatarsal head (toe)

  * GAIT EVENT DETECTION: heel contact is approximated by local minima of the
    ankle marker's vertical (Z) velocity crossing zero while Z is near its
    minimum (i.e., the foot is at its lowest and momentarily stationary).
    This is a standard marker-based approximation used when force plates /
    an instrumented treadmill are not available, and is NOT the exact
    proprietary algorithm used in the original paper (which is not published
    in the article). Replace `detect_heel_strikes()` with your own
    force-plate- or instrumented-treadmill-based event detector if you have
    one, for better accuracy.

  * SEGMENT MIDPOINTS used to estimate COG (paper gives weights only, not
    exact anatomical landmarks for "thigh"/"lower thigh"/"foot" segments):
        trunk  midpoint = mean(LACR, RACR, LIC, RIC)
        thigh  midpoint = mean(HIP, KNEE) per side
        shank  midpoint = mean(KNEE, ANK) per side   ("lower thigh" in paper)
        foot   midpoint = mean(ANK, TOE) per side

Adjust the CONFIG section below (or your CSV headers) if your marker naming or
axis convention differs.

USAGE
-----
    python gait_analysis.py input_markers.csv --fs 100 --outdir ./gait_output

Produces:
  * <outdir>/spatiotemporal_parameters.csv
  * <outdir>/joint_angles.csv
  * <outdir>/lissajous_overview_picture.png
  * <outdir>/joint_angle_curves.png
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

MARKERS = [
    "LACR", "RACR", "LIC", "RIC",
    "LHIP", "RHIP", "LKNE", "RKNE",
    "LANK", "RANK", "LTOE", "RTOE",
]

# The paper generates the LOP from the 10 "limb" markers (excludes the two
# iliac crest markers, which are used mainly for trunk/COG reference) plus
# the virtual COG.
LOP_MARKERS = [
    "LACR", "RACR", "LHIP", "RHIP", "LKNE", "RKNE",
    "LANK", "RANK", "LTOE", "RTOE",
]

# Segment mass fractions (paper Table / protocol text). Trunk + 2*(thigh+shank+foot) = 1.0
SEG_WEIGHTS = {"trunk": 0.66, "thigh": 0.10, "shank": 0.05, "foot": 0.02}


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #

def load_marker_data(csv_path):
    """Load a CSV with a `time` column and `<MARKER>_X/_Y/_Z` columns per marker.

    Returns
    -------
    t : (N,) ndarray of time in seconds
    pos : dict[str, (N,3) ndarray] marker -> XYZ trajectory (mm)
    """
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("Input CSV must contain a 'time' column (seconds).")

    t = df["time"].to_numpy(dtype=float)
    pos = {}
    missing = []
    for m in MARKERS:
        cols = [f"{m}_X", f"{m}_Y", f"{m}_Z"]
        if not all(c in df.columns for c in cols):
            missing.append(m)
            continue
        pos[m] = df[cols].to_numpy(dtype=float)

    if missing:
        raise ValueError(
            f"Missing marker columns for: {missing}. "
            f"Expected columns like '{missing[0]}_X', '{missing[0]}_Y', '{missing[0]}_Z'."
        )
    return t, pos


# --------------------------------------------------------------------------- #
# VIRTUAL CENTER OF GRAVITY (COG)
# --------------------------------------------------------------------------- #

def compute_cog(pos):
    """Estimate whole-body COG trajectory from segment-weighted marker midpoints.

    trunk  = mean(LACR, RACR, LIC, RIC)
    thigh  = mean(HIP, KNE)      (per side)
    shank  = mean(KNE, ANK)      (per side)
    foot   = mean(ANK, TOE)      (per side)

    COG = w_trunk*trunk + sum_sides[ w_thigh*thigh + w_shank*shank + w_foot*foot ]
    """
    trunk = np.mean([pos["LACR"], pos["RACR"], pos["LIC"], pos["RIC"]], axis=0)

    def side_segments(hip, kne, ank, toe):
        thigh = (pos[hip] + pos[kne]) / 2.0
        shank = (pos[kne] + pos[ank]) / 2.0
        foot = (pos[ank] + pos[toe]) / 2.0
        return thigh, shank, foot

    l_thigh, l_shank, l_foot = side_segments("LHIP", "LKNE", "LANK", "LTOE")
    r_thigh, r_shank, r_foot = side_segments("RHIP", "RKNE", "RANK", "RTOE")

    cog = (
        SEG_WEIGHTS["trunk"] * trunk
        + SEG_WEIGHTS["thigh"] * (l_thigh + r_thigh)
        + SEG_WEIGHTS["shank"] * (l_shank + r_shank)
        + SEG_WEIGHTS["foot"] * (l_foot + r_foot)
    )
    return cog


# --------------------------------------------------------------------------- #
# GAIT EVENT DETECTION (heel strikes) -- see ASSUMPTIONS above
# --------------------------------------------------------------------------- #

def detect_heel_strikes(t, ankle_z, fs, min_cycle_time=0.6):
    """Approximate heel-strike times from ankle-marker vertical position minima.

    A minimum in the ankle Z-trajectory approximates the point where the foot
    is lowest (near heel contact on a treadmill, where forward progression is
    removed by the belt). Minima closer together than `min_cycle_time`
    seconds are merged (keeps only the deepest one) to avoid double-detecting
    noise.
    """
    from scipy.signal import argrelextrema

    minima_idx = argrelextrema(ankle_z, np.less_equal, order=max(1, int(0.15 * fs)))[0]
    # Collapse plateaus / near-duplicate minima into single events
    events = []
    for idx in minima_idx:
        if events and (t[idx] - t[events[-1]]) < min_cycle_time:
            if ankle_z[idx] < ankle_z[events[-1]]:
                events[-1] = idx
            continue
        events.append(idx)
    return np.array(events)


# --------------------------------------------------------------------------- #
# SPATIOTEMPORAL PARAMETERS
# --------------------------------------------------------------------------- #

def compute_spatiotemporal(t, pos, fs):
    """Compute step cycle time, cadence, stance time, step width, and gait speed.

    Gait speed is estimated from the belt/treadmill-relative forward (Y)
    excursion of the COG combined with step cycle time; if you recorded true
    overground speed (e.g., treadmill belt speed), prefer that value instead.
    """
    l_hs = detect_heel_strikes(t, pos["LANK"][:, 2], fs)
    r_hs = detect_heel_strikes(t, pos["RANK"][:, 2], fs)

    results = {}

    if len(l_hs) >= 2:
        step_cycle_l = np.diff(t[l_hs])
        results["step_cycle_time_L_mean_s"] = float(np.mean(step_cycle_l))
        results["step_cycle_time_L_sd_s"] = float(np.std(step_cycle_l))
    if len(r_hs) >= 2:
        step_cycle_r = np.diff(t[r_hs])
        results["step_cycle_time_R_mean_s"] = float(np.mean(step_cycle_r))
        results["step_cycle_time_R_sd_s"] = float(np.std(step_cycle_r))

    all_hs = np.sort(np.concatenate([l_hs, r_hs])) if len(l_hs) and len(r_hs) else np.array([])
    if len(all_hs) >= 2:
        step_times = np.diff(t[all_hs])
        results["cadence_steps_per_min"] = float(60.0 / np.mean(step_times))

        # Step width: mediolateral (X) distance between ankles at each heel strike
        step_widths = []
        for idx in all_hs:
            step_widths.append(abs(pos["LANK"][idx, 0] - pos["RANK"][idx, 0]))
        results["step_width_mean_mm"] = float(np.mean(step_widths))
        results["step_width_sd_mm"] = float(np.std(step_widths))

    # Stance time (heel strike to the following contralateral heel strike + own toe-off
    # is the rigorous definition; here we approximate stance as heel-strike to
    # heel-strike-of-same-side minus the time the *toe* marker is elevated,
    # i.e., time the ankle Z stays below its own 25th percentile).
    for side, ank_key in (("L", "LANK"), ("R", "RANK")):
        hs = l_hs if side == "L" else r_hs
        if len(hs) < 2:
            continue
        z = pos[ank_key][:, 2]
        thresh = np.percentile(z, 25)
        stance_durations = []
        for i in range(len(hs) - 1):
            seg = z[hs[i]:hs[i + 1]]
            below = np.where(seg <= thresh)[0]
            if len(below):
                stance_durations.append((below[-1] + 1) / fs)
        if stance_durations:
            results[f"stance_time_{side}_mean_s"] = float(np.mean(stance_durations))
            results[f"stance_time_{side}_sd_s"] = float(np.std(stance_durations))

    # Gait speed from COG forward (Y) displacement rate
    cog = compute_cog(pos)
    dy = np.diff(cog[:, 1])
    dt = np.diff(t)
    speed = np.abs(dy / dt) / 1000.0  # mm/s -> m/s
    results["gait_speed_mean_m_s"] = float(np.mean(speed))
    results["gait_speed_sd_m_s"] = float(np.std(speed))

    results["n_left_heel_strikes"] = int(len(l_hs))
    results["n_right_heel_strikes"] = int(len(r_hs))

    return results, l_hs, r_hs


# --------------------------------------------------------------------------- #
# JOINT ANGLES (sagittal plane: Y-Z)
# --------------------------------------------------------------------------- #

def _sagittal_angle(proximal, joint, distal):
    """Angle (degrees) at `joint` between vectors to `proximal` and `distal`,
    projected onto the sagittal (Y-Z) plane. 180 deg = fully extended."""
    v1 = proximal[:, [1, 2]] - joint[:, [1, 2]]
    v2 = distal[:, [1, 2]] - joint[:, [1, 2]]
    dot = np.einsum("ij,ij->i", v1, v2)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_joint_angles(pos):
    """Sagittal-plane hip, knee, and ankle angles for both sides.

    Hip angle:   trunk(acromion) - hip - knee
    Knee angle:  hip - knee - ankle
    Ankle angle: knee - ankle - toe
    Returned as 180 - angle so that 0 deg ~ neutral/extended and increasing
    values indicate flexion (a common clinical convention); adjust the sign
    convention to match your lab's if needed.
    """
    angles = {}
    for side, acr, hip, kne, ank, toe in (
        ("L", "LACR", "LHIP", "LKNE", "LANK", "LTOE"),
        ("R", "RACR", "RHIP", "RKNE", "RANK", "RTOE"),
    ):
        hip_angle = 180.0 - _sagittal_angle(pos[acr], pos[hip], pos[kne])
        knee_angle = 180.0 - _sagittal_angle(pos[hip], pos[kne], pos[ank])
        ankle_angle = 180.0 - _sagittal_angle(pos[kne], pos[ank], pos[toe])
        angles[f"hip_{side}"] = hip_angle
        angles[f"knee_{side}"] = knee_angle
        angles[f"ankle_{side}"] = ankle_angle
    return angles


# --------------------------------------------------------------------------- #
# GAIT-CYCLE NORMALIZATION (0-100%)
# --------------------------------------------------------------------------- #

def normalize_to_cycles(signal, event_indices, n_points=101):
    """Resample `signal` into `event_indices`-defined cycles, each stretched
    to n_points samples spanning 0-100% of the cycle. Returns a (n_cycles,
    n_points) array."""
    cycles = []
    x_new = np.linspace(0, 1, n_points)
    for i in range(len(event_indices) - 1):
        s, e = event_indices[i], event_indices[i + 1]
        if e - s < 3:
            continue
        seg = signal[s:e]
        x_old = np.linspace(0, 1, len(seg))
        cycles.append(np.interp(x_new, x_old, seg))
    return np.array(cycles) if cycles else np.empty((0, n_points))


# --------------------------------------------------------------------------- #
# LISSAJOUS OVERVIEW PICTURE (LOP)
# --------------------------------------------------------------------------- #

def build_lop(pos, cog, l_hs, r_hs, outpath):
    """Build the Lissajous Overview Picture: mean +/- SD trajectories of the
    10 limb markers + virtual COG, in the horizontal (X-Y), sagittal (Y-Z),
    and coronal (Z-X) planes, normalized over the gait cycle.
    """
    events = l_hs if len(l_hs) >= 2 else r_hs
    if len(events) < 2:
        print("Not enough heel-strike events detected to build a normalized LOP; "
              "plotting raw trajectories instead.")
        events = None

    planes = [("Horizontal (X-Y)", 0, 1), ("Sagittal (Y-Z)", 1, 2), ("Coronal (Z-X)", 2, 0)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    all_markers = {**{m: pos[m] for m in LOP_MARKERS}, "COG": cog}

    for ax, (title, ci, cj) in zip(axes, planes):
        for name, traj in all_markers.items():
            if events is not None:
                comp_i = normalize_to_cycles(traj[:, ci], events)
                comp_j = normalize_to_cycles(traj[:, cj], events)
                mean_i, mean_j = comp_i.mean(axis=0), comp_j.mean(axis=0)
            else:
                mean_i, mean_j = traj[:, ci], traj[:, cj]
            ax.plot(mean_i, mean_j, linewidth=1.2, label=name)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(["X (ML, mm)", "Y (AP, mm)", "Z (Vert, mm)"][ci])
        ax.set_ylabel(["X (ML, mm)", "Y (AP, mm)", "Z (Vert, mm)"][cj])

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle("Lissajous Overview Picture (LOP): marker trajectories, gait-cycle mean")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_joint_angle_curves(angles, l_hs, r_hs, outpath):
    """Plot mean +/- SD sagittal joint-angle curves over the normalized gait cycle."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    joints = ["hip", "knee", "ankle"]
    for ax, joint in zip(axes, joints):
        for side, events, color in (("L", l_hs, "tab:blue"), ("R", r_hs, "tab:red")):
            if len(events) < 2:
                continue
            cycles = normalize_to_cycles(angles[f"{joint}_{side}"], events)
            if cycles.size == 0:
                continue
            mean = cycles.mean(axis=0)
            sd = cycles.std(axis=0)
            x = np.linspace(0, 100, cycles.shape[1])
            ax.plot(x, mean, color=color, label=f"{side}")
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2)
        ax.set_title(f"{joint.capitalize()} angle")
        ax.set_xlabel("% Gait cycle")
        ax.set_ylabel("Flexion angle (deg)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# LIVE METRICS DASHBOARD (separate window, auto-refreshing)
# --------------------------------------------------------------------------- #

# Friendly labels + units for the numbers we show on the live dashboard.
_METRIC_DISPLAY = [
    ("gait_speed_mean_m_s", "Gait speed", "m/s"),
    ("cadence_steps_per_min", "Cadence", "steps/min"),
    ("step_cycle_time_L_mean_s", "Step cycle (L)", "s"),
    ("step_cycle_time_R_mean_s", "Step cycle (R)", "s"),
    ("stance_time_L_mean_s", "Stance time (L)", "s"),
    ("stance_time_R_mean_s", "Stance time (R)", "s"),
    ("step_width_mean_mm", "Step width", "mm"),
    ("n_left_heel_strikes", "Left heel strikes (n)", ""),
    ("n_right_heel_strikes", "Right heel strikes (n)", ""),
]


def _format_metrics_text(csv_path, fs, error=None):
    """Build the multi-line text block shown on the live dashboard."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines = [f"Live Gait Metrics  —  updated {timestamp}", "-" * 42]

    if error is not None:
        lines.append("")
        lines.append("No data yet / read error:")
        lines.append(f"  {error}")
        return "\n".join(lines)

    t, pos = load_marker_data(csv_path)
    st_params, l_hs, r_hs = compute_spatiotemporal(t, pos, fs)
    angles = compute_joint_angles(pos)

    for key, label, unit in _METRIC_DISPLAY:
        if key not in st_params:
            continue
        val = st_params[key]
        val_str = f"{val:.3f}" if isinstance(val, float) else f"{val}"
        lines.append(f"{label:<24s}{val_str:>10s} {unit}")

    lines.append("")
    lines.append("Peak flexion angle (deg):")
    for joint in ("hip", "knee", "ankle"):
        peaks = []
        for side in ("L", "R"):
            arr = angles.get(f"{joint}_{side}")
            if arr is not None and len(arr):
                peaks.append(f"{side}={np.nanmax(arr):.1f}")
        if peaks:
            lines.append(f"  {joint.capitalize():<8s} {'  '.join(peaks)}")

    lines.append("")
    lines.append(f"Samples loaded: {len(t)}")
    return "\n".join(lines)


def run_live_metrics_dashboard(csv_path, fs, interval_s=30.0):
    """Open a dedicated window that re-reads `csv_path` and refreshes the
    computed gait metrics every `interval_s` seconds.

    Intended for use on the Pi while your capture pipeline keeps
    writing/updating the same CSV file: each refresh reloads the file from
    disk and recomputes spatiotemporal parameters + peak joint angles, so the
    numbers track whatever is in the file at that moment. If the file is
    momentarily unreadable (e.g., being written to) or doesn't have enough
    strides yet, the window shows a "no data yet" message instead of crashing.
    """
    fig, ax = plt.subplots(figsize=(5.2, 6))
    fig.canvas.manager.set_window_title("Live Gait Metrics")
    ax.axis("off")
    text_artist = ax.text(
        0.03, 0.97, "Loading...", va="top", ha="left",
        family="monospace", fontsize=10, transform=ax.transAxes,
    )

    def update(_frame):
        try:
            text = _format_metrics_text(csv_path, fs)
        except Exception as exc:  # keep the dashboard alive across bad reads
            text = _format_metrics_text(csv_path, fs, error=str(exc))
        text_artist.set_text(text)
        return (text_artist,)

    # Keep a reference on the figure so the animation isn't garbage-collected.
    fig._live_metrics_animation = animation.FuncAnimation(
        fig, update, interval=interval_s * 1000, cache_frame_data=False
    )
    plt.show()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Path to marker trajectory CSV (see module docstring for format).")
    parser.add_argument("--fs", type=float, required=True, help="Sampling frequency in Hz.")
    parser.add_argument("--outdir", default="./gait_output", help="Output directory.")
    parser.add_argument(
        "--live", action="store_true",
        help="After running the analysis once, open a separate window showing "
             "key metrics that re-reads the input CSV and refreshes every "
             "--interval seconds (default 30).",
    )
    parser.add_argument(
        "--interval", type=float, default=30.0,
        help="Refresh interval in seconds for --live mode (default: 30).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    t, pos = load_marker_data(args.input_csv)
    cog = compute_cog(pos)

    st_params, l_hs, r_hs = compute_spatiotemporal(t, pos, args.fs)
    pd.Series(st_params).to_csv(os.path.join(args.outdir, "spatiotemporal_parameters.csv"), header=["value"])
    print("Spatiotemporal parameters:")
    for k, v in st_params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    angles = compute_joint_angles(pos)
    pd.DataFrame({**{"time": t}, **angles}).to_csv(
        os.path.join(args.outdir, "joint_angles.csv"), index=False
    )

    build_lop(pos, cog, l_hs, r_hs, os.path.join(args.outdir, "lissajous_overview_picture.png"))
    plot_joint_angle_curves(angles, l_hs, r_hs, os.path.join(args.outdir, "joint_angle_curves.png"))

    print(f"\nOutputs written to: {os.path.abspath(args.outdir)}")

    if args.live:
        print(f"\nOpening live metrics window (refreshing every {args.interval:.0f}s)... "
              f"close the window to exit.")
        run_live_metrics_dashboard(args.input_csv, args.fs, args.interval)


if __name__ == "__main__":
    sys.exit(main())
