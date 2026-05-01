#!/usr/bin/env python3
"""
Data Collection Script: Paired Human-Robot Trajectory Collection
================================================================

This script:
import select
import tty
import termios
1. Runs the trained BC vision policy on the Lift task
2. Pauses execution at ~15-20% of the trajectory (blackout injection)
3. Displays the frozen frame and robot state to the human operator
4. Records the human's ArUco marker hand trajectory (spacebar to start/stop)
5. Resumes the vision policy to complete the task
6. Saves the paired data: human trajectory (input) + robot remaining trajectory (output)

Usage:
    source ~/robomimic/bin/activate
    export DISPLAY=:99  # if using Xvfb
    python3 collect_paired_trajectories.py --n_trajectories 50

Requirements:
    - Camera calibration file (calibration.npz) with camera_matrix and dist_coeffs
    - ArUco marker ID 0, DICT_4X4_50, 8cm physical size
    - Trained BC model at the configured path
    - USB webcam accessible (update CAMERA_ID if needed)
"""

import argparse
import json
import os

def safe_imshow(window_name, frame):
    """Write-then-read workaround for ARM64 cv2.imshow crash."""
    cv2.imwrite("/tmp/_display_tmp.png", frame)
    img = cv2.imread("/tmp/_display_tmp.png")
    cv2.imshow(window_name, img)

import sys
import time
import numpy as np
import cv2
import h5py
from datetime import datetime

# ============================================================================
# CONFIGURATION — defaults are portable; override via CLI arguments
# ============================================================================

MODEL_PATH        = ""                                          # required — pass via --model_path
CALIBRATION_PATH  = os.path.join(os.getcwd(), "calibration.npz")
OUTPUT_DIR        = os.path.join(os.getcwd(), "paired_trajectories")
CAMERA_ID         = 0  # USB webcam device index

# ArUco configuration
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_SIZE_M = 0.08  # 8cm in meters

# Blackout injection range (fraction of trajectory)
BLACKOUT_MIN_FRAC = 0.15
BLACKOUT_MAX_FRAC = 0.20

# Rollout settings
MAX_HORIZON = 400  # max timesteps per rollout (robosuite Lift default)
CAPTURE_FPS = 30   # ArUco capture target framerate


# ============================================================================
# ROBOMIMIC / ROBOSUITE IMPORTS
# ============================================================================

def setup_robomimic_imports():
    """Add robomimic to path and import necessary modules."""
    robomimic_path = os.path.expanduser(
        "~/robomimic/lib/python3.8/site-packages"
    )
    if robomimic_path not in sys.path:
        sys.path.insert(0, robomimic_path)

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.algo import algo_factory
    import torch

    return FileUtils, EnvUtils, TorchUtils, algo_factory, torch


def load_policy_and_env(model_path):
    """
    Load the trained BC policy and create the robosuite environment
    from the saved checkpoint (same approach as run_trained_agent.py).
    """
    FileUtils, EnvUtils, TorchUtils, algo_factory, torch = setup_robomimic_imports()

    # Load checkpoint metadata
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=model_path)

    # Create environment from checkpoint config
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=True,  # needed to capture images
    )

    # Create and load the policy
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict,
        device=TorchUtils.get_torch_device(try_to_use_cuda=True),
    )
    policy.start_episode()

    return env, policy


# ============================================================================
# ARUCO POSE CAPTURE
# ============================================================================

class ArucoPoseCapture:
    """
    Captures 6-DOF pose of ArUco marker ID 0 from a stationary USB webcam.
    Uses OpenCV 4.13.0 older API (estimatePoseSingleMarkers).
    """

    def __init__(self, camera_id, calibration_path, marker_size_m):
        # Load camera calibration
        calib = np.load(calibration_path)
        self.camera_matrix = calib["camera_matrix"]
        self.dist_coeffs = calib["dist_coeffs"]

        # ArUco setup
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_size = marker_size_m

        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        # Set capture resolution to 1080p (not 4K — performance)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

        print(f"[ArUco] Camera opened: "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
              f"{int(self.cap.get(cv2.CAP_PROP_FPS))}fps")

    def get_pose(self):
        """
        Capture a single frame and return the 6-DOF pose of marker ID 0.

        Returns:
            pose: dict with keys 'rvec' (3,), 'tvec' (3,), 'timestamp'
                  or None if marker not detected
            frame: the captured BGR image (for display)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers (OpenCV 4.13.0 older API)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

        if ids is not None and MARKER_ID in ids.flatten():
            # Find the index of our marker
            idx = np.where(ids.flatten() == MARKER_ID)[0][0]

            # Estimate pose (older API — returns rvecs, tvecs directly)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[idx:idx+1],
                self.marker_size,
                self.camera_matrix,
                self.dist_coeffs,
            )

            rvec = rvecs[0].flatten()  # (3,) rotation vector
            tvec = tvecs[0].flatten()  # (3,) translation vector [x, y, z] in meters

            # Draw detection on frame for visual feedback
            pass  # drawDetectedMarkers disabled — ARM64
            # drawFrameAxes disabled — ARM64
            # disabled
            # disabled
            # disabled

            return {
                "rvec": rvec,
                "tvec": tvec,
                "timestamp": time.time(),
            }, frame
        else:
            return None, frame

    def release(self):
        self.cap.release()


def record_human_trajectory(aruco_capture):
    """
    Record the human's hand trajectory using ArUco marker tracking.
    Terminal-only version — no cv2 GUI (ARM64 compatibility).
    Press ENTER to start, ENTER again to stop.
    """
    import select, tty, termios

    print("\n" + "=" * 60)
    print("HUMAN GESTURE RECORDING")
    print("=" * 60)
    print("Position your hand with the ArUco marker visible to the camera.")
    print("Press ENTER to START recording your gesture.")
    print("Press ENTER again to STOP recording.")
    print("Press 'q' + ENTER to ABORT this trial.")
    print("=" * 60)

    input(">>> Press ENTER when ready to START recording...")
    print("[ArUco] Recording STARTED")

    recording = True
    trajectory = []
    aborted = False

    # Set terminal to non-blocking mode
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while recording:
            pose, frame = aruco_capture.get_pose()

            # Record if marker detected
            if pose is not None:
                trajectory.append(pose)

            # Print status every 10 samples
            if len(trajectory) % 10 == 0 and len(trajectory) > 0:
                pos = pose['tvec'] if pose is not None else None
                det = "DETECTED" if pose is not None else "NOT VISIBLE"
                if pos is not None:
                    print(f"\r  Samples: {len(trajectory)} | Marker: {det} | "
                          f"pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]", end="", flush=True)
                else:
                    print(f"\r  Samples: {len(trajectory)} | Marker: {det}", end="", flush=True)

            # Check for keypress (non-blocking)
            if select.select([sys.stdin], [], [], 0.001)[0]:
                ch = sys.stdin.read(1)
                if ch == '\n' or ch == '\r':
                    recording = False
                    print(f"\n[ArUco] Recording STOPPED — {len(trajectory)} samples captured")
                elif ch == 'q':
                    aborted = True
                    print("\n[ArUco] Trial ABORTED by user")
                    break

            time.sleep(1.0 / CAPTURE_FPS)  # throttle to target FPS
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    if aborted:
        return None

    if len(trajectory) < 5:
        print(f"[WARNING] Only {len(trajectory)} samples — too few. Discarding.")
        return None

    return trajectory

def run_single_collection(env, policy, aruco_capture, trial_idx):
    """
    Run one data collection trial:
    1. Reset env, run vision policy until blackout point
    2. Save frozen state
    3. Record human gesture
    4. Resume vision policy to completion
    5. Return paired data

    Returns:
        data dict or None if trial failed/aborted
    """
    FileUtils, EnvUtils, TorchUtils, algo_factory, torch = setup_robomimic_imports()

    # Reset environment
    obs = env.reset()
    policy.start_episode()

    # Determine blackout timestep (15-20% of horizon)
    blackout_frac = np.random.uniform(BLACKOUT_MIN_FRAC, BLACKOUT_MAX_FRAC)
    blackout_step = int(blackout_frac * MAX_HORIZON)

    print(f"\n{'='*60}")
    print(f"TRIAL {trial_idx}: Blackout at step {blackout_step} "
          f"({blackout_frac*100:.1f}% of horizon)")
    print(f"{'='*60}")

    # ---- Phase 1: Run vision policy BEFORE blackout ----
    pre_blackout_eef_poses = []   # robot EEF poses before blackout
    pre_blackout_actions = []     # actions taken before blackout

    for step in range(blackout_step):
        # Get action from vision policy
        action = policy(ob=obs)

        # Record robot state
        eef_pos = env.env.sim.data.site_xpos[
            env.env.sim.model.site_name2id("gripper0_right_grip_site")
        ].copy()
        eef_quat = None  # We store position; orientation can be added if needed

        pre_blackout_eef_poses.append(eef_pos)
        pre_blackout_actions.append(action.copy())

        # Step environment
        obs, reward, done, info = env.step(action)

        if done:
            print(f"  [Phase 1] Task completed BEFORE blackout at step {step}. Retrying.")
            return None  # task finished too early, skip this trial

    # ---- Blackout point: capture frozen state ----
    blackout_eef_pos = env.env.sim.data.site_xpos[
        env.env.sim.model.site_name2id("gripper0_right_grip_site")
    ].copy()

    # Get the target (cube) position at blackout
    # In Lift task, the cube body is named "cube"
    cube_pos = env.env.sim.data.body_xpos[
        env.env.sim.model.body_name2id("cube_main")
    ].copy()

    # Capture the frozen camera frame from the simulation
    # (this is what the human sees — the last frame before blackout)
    frozen_frame = env.render(
        mode="rgb_array",
        height=480,
        width=640,
        camera_name="agentview",
    )

    print(f"  Blackout EEF position: {blackout_eef_pos}")
    print(f"  Cube position: {cube_pos}")
    print(f"  Distance to cube: {np.linalg.norm(blackout_eef_pos - cube_pos):.4f}m")

    # Display frozen frame to the human
    frozen_bgr = cv2.cvtColor(frozen_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/tmp/frozen_frame.png", frozen_bgr)
    print("  Frozen frame saved to /tmp/frozen_frame.png — open it to see last view before blackout")
    safe_imshow("Frozen Frame (Last View Before Blackout)", frozen_bgr)
    # GUI disabled — ARM64 compatibility

    # ---- Phase 2: Record human gesture ----
    human_trajectory = record_human_trajectory(aruco_capture)

    # GUI disabled

    if human_trajectory is None:
        return None  # aborted or too few samples

    # ---- Phase 3: Resume vision policy AFTER blackout ----
    post_blackout_eef_poses = []
    post_blackout_actions = []
    task_success = False

    for step in range(blackout_step, MAX_HORIZON):
        # Vision policy continues (it still gets observations —
        # in the real scenario it wouldn't, but here we need ground truth)
        action = policy(ob=obs)

        # Record robot state (this is our ground truth output)
        eef_pos = env.env.sim.data.site_xpos[
            env.env.sim.model.site_name2id("gripper0_right_grip_site")
        ].copy()

        post_blackout_eef_poses.append(eef_pos)
        post_blackout_actions.append(action.copy())

        obs, reward, done, info = env.step(action)

        if done or env.env.is_success()["task"]:
            task_success = env.env.is_success()["task"]
            print(f"  [Phase 3] Task ended at step {step}. "
                  f"Success: {task_success}")
            break

    if not task_success:
        print(f"  [Phase 3] Task did NOT succeed. Still saving for training data.")

    # ---- Package the paired data ----
    # Convert human trajectory to numpy arrays
    human_tvecs = np.array([p["tvec"] for p in human_trajectory])  # (N, 3)
    human_rvecs = np.array([p["rvec"] for p in human_trajectory])  # (N, 3)
    human_timestamps = np.array([p["timestamp"] for p in human_trajectory])  # (N,)

    # Compute human deltas (relative to first pose)
    human_delta_tvecs = human_tvecs - human_tvecs[0:1]  # (N, 3) relative positions

    data = {
        # Metadata
        "trial_idx": trial_idx,
        "blackout_step": blackout_step,
        "blackout_frac": blackout_frac,
        "task_success": task_success,
        "timestamp": datetime.now().isoformat(),

        # Blackout state
        "blackout_eef_pos": blackout_eef_pos,          # (3,)
        "cube_pos_at_blackout": cube_pos,               # (3,)
        "frozen_frame": frozen_frame,                    # (480, 640, 3) uint8

        # Human trajectory (INPUT to mapping policy)
        "human_tvecs": human_tvecs,                      # (N_human, 3)
        "human_rvecs": human_rvecs,                      # (N_human, 3)
        "human_timestamps": human_timestamps,            # (N_human,)
        "human_delta_tvecs": human_delta_tvecs,          # (N_human, 3)

        # Robot trajectory after blackout (OUTPUT / ground truth)
        "robot_eef_poses_post": np.array(post_blackout_eef_poses),   # (N_robot, 3)
        "robot_actions_post": np.array(post_blackout_actions),       # (N_robot, action_dim)

        # Robot trajectory before blackout (context, not used for training directly)
        "robot_eef_poses_pre": np.array(pre_blackout_eef_poses),     # (blackout_step, 3)
    }

    return data


# ============================================================================
# SAVE / LOAD UTILITIES
# ============================================================================

def save_trial(data, output_dir, trial_idx):
    """Save a single trial to an HDF5 file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"trial_{trial_idx:04d}.hdf5")

    with h5py.File(filepath, "w") as f:
        # Metadata as attributes
        f.attrs["trial_idx"] = data["trial_idx"]
        f.attrs["blackout_step"] = data["blackout_step"]
        f.attrs["blackout_frac"] = data["blackout_frac"]
        f.attrs["task_success"] = data["task_success"]
        f.attrs["timestamp"] = data["timestamp"]

        # Numpy arrays as datasets
        f.create_dataset("blackout_eef_pos", data=data["blackout_eef_pos"])
        f.create_dataset("cube_pos_at_blackout", data=data["cube_pos_at_blackout"])
        f.create_dataset("frozen_frame", data=data["frozen_frame"],
                         compression="gzip", compression_opts=4)

        # Human trajectory group
        human = f.create_group("human")
        human.create_dataset("tvecs", data=data["human_tvecs"])
        human.create_dataset("rvecs", data=data["human_rvecs"])
        human.create_dataset("timestamps", data=data["human_timestamps"])
        human.create_dataset("delta_tvecs", data=data["human_delta_tvecs"])

        # Robot trajectory group
        robot = f.create_group("robot")
        robot.create_dataset("eef_poses_post", data=data["robot_eef_poses_post"])
        robot.create_dataset("actions_post", data=data["robot_actions_post"])
        robot.create_dataset("eef_poses_pre", data=data["robot_eef_poses_pre"])

    print(f"  Saved: {filepath}")
    return filepath


def save_collection_summary(output_dir, all_trials_metadata):
    """Save a JSON summary of all collected trials."""
    summary_path = os.path.join(output_dir, "collection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_trials_metadata, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect paired human-robot trajectories for mapping policy training"
    )
    parser.add_argument(
        "--n_trajectories", type=int, default=50,
        help="Number of successful paired trajectories to collect (default: 50)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: paired_trajectories/ in current directory)"
    )
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH,
        help="Path to trained BC model checkpoint (.pth) — required"
    )
    parser.add_argument(
        "--calibration_path", type=str, default=CALIBRATION_PATH,
        help="Path to camera calibration .npz (default: calibration.npz in current directory)"
    )
    parser.add_argument(
        "--camera_id", type=int, default=CAMERA_ID,
        help=f"Webcam device index (default: {CAMERA_ID})"
    )
    parser.add_argument(
        "--skip_failed", action="store_true",
        help="Only save trials where the vision policy succeeded after blackout"
    )
    args = parser.parse_args()

    # Validate paths
    if not args.model_path:
        print("ERROR: --model_path is required. Pass the path to your .pth checkpoint.")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.calibration_path):
        print(f"ERROR: Calibration file not found at {args.calibration_path}")
        print("Run camera_calibration.py first to generate calibration.npz.")
        sys.exit(1)

    # Create output directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"session_{timestamp_str}")
    os.makedirs(session_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("PAIRED TRAJECTORY DATA COLLECTION")
    print("=" * 60)
    print(f"  Model:        {args.model_path}")
    print(f"  Calibration:  {args.calibration_path}")
    print(f"  Output:       {session_dir}")
    print(f"  Target:       {args.n_trajectories} trajectories")
    print(f"  Blackout:     {BLACKOUT_MIN_FRAC*100:.0f}-{BLACKOUT_MAX_FRAC*100:.0f}% of trajectory")
    print(f"  Camera ID:    {args.camera_id}")
    print("=" * 60)

    # Load policy and environment
    print("\nLoading policy and environment...")
    env, policy = load_policy_and_env(args.model_path)
    print("Policy and environment loaded.")

    # Initialize ArUco capture
    print("\nInitializing ArUco capture...")
    aruco_capture = ArucoPoseCapture(
        camera_id=args.camera_id,
        calibration_path=args.calibration_path,
        marker_size_m=MARKER_SIZE_M,
    )
    print("ArUco capture ready.")

    # Collection loop
    collected = 0
    attempted = 0
    all_metadata = []

    print(f"\nStarting collection. Press Ctrl+C to stop early.\n")

    try:
        while collected < args.n_trajectories:
            attempted += 1
            print(f"\n--- Attempt {attempted} (collected {collected}/{args.n_trajectories}) ---")

            data = run_single_collection(env, policy, aruco_capture, attempted)

            if data is None:
                print("  Trial skipped (aborted, failed, or insufficient data).")
                continue

            if args.skip_failed and not data["task_success"]:
                print("  Trial skipped (--skip_failed and task did not succeed).")
                continue

            # Save
            save_trial(data, session_dir, collected)

            # Track metadata
            all_metadata.append({
                "trial_idx": collected,
                "attempt": attempted,
                "blackout_step": int(data["blackout_step"]),
                "blackout_frac": float(data["blackout_frac"]),
                "task_success": bool(data["task_success"]),
                "n_human_samples": len(data["human_tvecs"]),
                "n_robot_steps_post": len(data["robot_eef_poses_post"]),
                "timestamp": data["timestamp"],
            })

            collected += 1
            print(f"  >>> Collected {collected}/{args.n_trajectories}")

    except KeyboardInterrupt:
        print(f"\n\nCollection interrupted. {collected} trajectories saved.")

    # Save summary
    save_collection_summary(session_dir, {
        "total_collected": collected,
        "total_attempted": attempted,
        "model_path": args.model_path,
        "blackout_range": [BLACKOUT_MIN_FRAC, BLACKOUT_MAX_FRAC],
        "trials": all_metadata,
    })

    # Cleanup
    aruco_capture.release()
    cv2.destroyAllWindows()

    print(f"\nDone. {collected} paired trajectories saved to {session_dir}")


if __name__ == "__main__":
    main()
