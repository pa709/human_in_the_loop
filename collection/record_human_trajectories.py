"

import argparse
import h5py
import numpy as np
import cv2
import subprocess
import threading
import os
import time
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file, no display window needed
import matplotlib.pyplot as plt


# --- ArUco setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
MARKER_ID = 0
MARKER_SIZE_M = 0.05  # 5cm marker


def load_calibration(calib_path):
    """Load camera calibration from npz file if available."""
    if calib_path and os.path.exists(calib_path):
        data = np.load(calib_path)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        print("Loaded calibration from: {}".format(calib_path))
        return camera_matrix, dist_coeffs
    else:
        # fallback: rough calibration for 640x480
        print("WARNING: No calibration file found. Using approximate calibration.")
        camera_matrix = np.array([
            [600,   0, 320],
            [  0, 600, 240],
            [  0,   0,   1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1))
        return camera_matrix, dist_coeffs


def plot_demo(demo_idx, tvecs, rvecs, timestamps, remaining_eef_poses, output_dir):
    """
    Plot human trajectory (tvec + rvec) against robot ground truth (remaining_eef_poses).
    Saves plot as jpg and opens in Preview.

    human:  tvecs (N,3), rvecs (N,3), timestamps (N,)
    robot:  remaining_eef_poses (T,7) — columns 0:3 are xyz, 3:7 are quaternion
    """
    # normalize human timestamps to 0-1
    if len(timestamps) > 1:
        t_human = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
    else:
        t_human = timestamps

    # normalize robot steps to 0-1
    T = remaining_eef_poses.shape[0]
    t_robot = np.linspace(0, 1, T)

    robot_xyz = remaining_eef_poses[:, 0:3]   # (T, 3)
    robot_quat = remaining_eef_poses[:, 3:7]  # (T, 4)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Demo {} — Human vs Robot Trajectory (normalized time)".format(demo_idx), fontsize=13)

    labels_t = ["x", "y", "z"]
    labels_r = ["rx", "ry", "rz"]

    for col, label in enumerate(labels_t):
        ax = axes[0, col]
        ax.plot(t_human, tvecs[:, col], label="human tvec {}".format(label), color="blue")
        ax.plot(t_robot, robot_xyz[:, col], label="robot eef {}".format(label), color="orange", linestyle="--")
        ax.set_title("Translation — {}".format(label))
        ax.set_xlabel("normalized time")
        ax.set_ylabel("meters" if col < 3 else "")
        ax.legend(fontsize=8)
        ax.grid(True)

    for col, label in enumerate(labels_r):
        ax = axes[1, col]
        ax.plot(t_human, rvecs[:, col], label="human rvec {}".format(label), color="green")
        # robot quaternion — plot the corresponding component (x,y,z of quat)
        ax.plot(t_robot, robot_quat[:, col], label="robot quat {}".format(label), color="red", linestyle="--")
        ax.set_title("Rotation — {} (rvec) vs quat {}".format(label, label))
        ax.set_xlabel("normalized time")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "demo_{}_trajectory_plot.jpg".format(demo_idx))
    plt.savefig(plot_path, dpi=100)
    plt.close()

    subprocess.run(["open", plot_path])
    print("Plot saved and opened: {}".format(plot_path))


def open_stream(camera_name, width=640, height=480):
    """
    Open a continuous ffmpeg stream from the webcam.
    Returns a subprocess with raw BGR frames available on stdout.
    """
    proc = subprocess.Popen(
        [
            "ffmpeg", "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", "{}x{}".format(width, height),
            "-i", "{}:none".format(camera_name),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-loglevel", "error",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    return proc


def read_frame(proc, width=640, height=480):
    """Read one frame from an open ffmpeg stream."""
    expected = width * height * 3
    raw = proc.stdout.read(expected)
    if len(raw) < expected:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


def detect_marker_pose(frame, camera_matrix, dist_coeffs):
    """
    Detect ArUco marker ID 0 and return tvec, rvec.
    Returns (tvec, rvec) as 1D arrays (3,) each, or (None, None) if not detected.
    """
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)
    if ids is None:
        return None, None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == MARKER_ID:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], MARKER_SIZE_M, camera_matrix, dist_coeffs
            )
            tvec = tvecs[0][0]  # (3,)
            rvec = rvecs[0][0]  # (3,)
            return tvec, rvec

    return None, None


def save_frozen_frame(frozen_rgb, output_dir, demo_idx):
    """Save frozen RGB frame as jpg so the human can open it in Preview."""
    os.makedirs(output_dir, exist_ok=True)
    # frozen_rgb from robomimic is RGB, convert to BGR for cv2 , cv2 expects BGR format.
    bgr = cv2.cvtColor(frozen_rgb, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, "demo_{}_frozen_frame.jpg".format(demo_idx))
    cv2.imwrite(path, bgr)
    return path


def record_human_trajectory(camera_name, camera_matrix, dist_coeffs, duration=120):
    """
    Record ArUco marker trajectory for a fixed duration using a continuous ffmpeg stream.
    Frame reading runs in a background thread. Main thread sleeps for duration seconds
    then kills the ffmpeg process — this causes read_frame to return None and the
    capture thread exits cleanly.

    Returns:
        tvecs (np.ndarray): shape (N, 3)
        rvecs (np.ndarray): shape (N, 3)
        timestamps (np.ndarray): shape (N,) seconds since start
    """
    tvecs = []
    rvecs = []
    timestamps = []
    total_frames = [0]
    detected_frames = [0]

    print("  Opening camera stream...")
    proc = open_stream(camera_name)
    start_time = time.time()

    def capture_thread():
        while True:
            frame = read_frame(proc)
            if frame is None:
                break
            total_frames[0] += 1
            tvec, rvec = detect_marker_pose(frame, camera_matrix, dist_coeffs)
            ts = time.time() - start_time
            if tvec is not None:
                tvecs.append(tvec)
                rvecs.append(rvec)
                timestamps.append(ts)
                detected_frames[0] += 1

    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    print("  Stream open. Recording...")

    # main thread handles countdown and kills stream after duration
    interval = 30
    elapsed = 0
    while elapsed < duration:
        sleep_time = min(interval, duration - elapsed)
        time.sleep(sleep_time)
        elapsed += sleep_time
        remaining = duration - elapsed
        if remaining > 0:
            print("  {} seconds remaining...".format(int(remaining)))

    # kill ffmpeg — causes read_frame to return None, capture_thread exits
    proc.terminate()
    proc.wait()
    t.join(timeout=2.0)

    print("  Total frames read: {} | Marker detected in: {}".format(
        total_frames[0], detected_frames[0]
    ))

    if len(tvecs) == 0:
        return None, None, None

    return np.array(tvecs), np.array(rvecs), np.array(timestamps)


def record_all_demos(args):
    camera_matrix, dist_coeffs = load_calibration(args.calib)

    # open ground truth file (read-only)
    gt_file = h5py.File(args.hdf5, "r")
    gt_grp = gt_file["data"]
    n_demos = gt_grp.attrs["n_demos"]
    print("Found {} demos in ground truth HDF5.".format(n_demos))

    # open or create separate human trajectory file
    out_path = os.path.join(os.path.dirname(args.hdf5), "recorded_vision_blackout_trajectories.hdf5")
    out_file = h5py.File(out_path, "a")
    if "data" not in out_file:
        out_grp = out_file.create_group("data")
        out_grp.attrs["n_demos"] = n_demos
    else:
        out_grp = out_file["data"]
    print("Human trajectories will be saved to: {}".format(out_path))

    recorded = 0
    skipped = 0

    for i in range(n_demos):
        demo_key = "demo_{}".format(i)
        gt_ep = gt_grp[demo_key]

        # skip demos already recorded in output file
        if demo_key in out_grp and out_grp[demo_key].attrs.get("human_recorded", False):
            skipped += 1
            continue

        print("\n========================================")
        print("Demo {}/{} | blackout_idx={} | remaining_steps={}".format(
            i, n_demos - 1,
            gt_ep.attrs["blackout_idx"],
            gt_ep.attrs["remaining_len"]
        ))

        # save and open frozen frame from ground truth
        frozen_rgb = gt_ep["frozen_rgb"][:]
        frame_path = save_frozen_frame(frozen_rgb, args.output_dir, i)
        subprocess.run(["open", frame_path])
        print("\nFrozen frame opened in Preview: {}".format(frame_path))
        print("This is what the robot saw just before blackout.")
        print("Imagine your arm is the robot arm. Reach toward the object in the image.")

        print("\nPress ENTER when ready to start recording your gesture...")
        input()

        print("Recording for {} seconds... perform your gesture now.".format(args.duration))
        print("  - Keep hand stationary for 5-7 seconds to mark start")
        print("  - Perform reaching gesture")
        print("  - Rotate wrist and move hand out of frame to mark end")

        tvecs, rvecs, timestamps = record_human_trajectory(
            args.camera_name, camera_matrix, dist_coeffs, duration=args.duration
        )

        if tvecs is None or len(tvecs) < 3:
            print("Not enough marker detections (got {}). Retrying this demo.".format(
                0 if tvecs is None else len(tvecs)
            ))
            print("Make sure the ArUco marker is clearly visible to the EMEET camera.")
            print("Retrying automatically...")
            tvecs, rvecs, timestamps = record_human_trajectory(
                args.camera_name, camera_matrix, dist_coeffs, duration=args.duration
            )

        if tvecs is None or len(tvecs) < 3:
            print("Still not enough detections. Skipping demo {}.".format(i))
            continue

        duration_actual = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        print("Recorded {} poses over {:.2f} seconds.".format(len(tvecs), duration_actual))

        # Added for trajectory testing — skip saving if dry_run is set
        if args.dry_run:
            print("DRY RUN — not saving to HDF5.")
        else:
            # save human trajectory into output file under matching demo key
            if demo_key in out_grp:
                del out_grp[demo_key]
            ep_out = out_grp.create_group(demo_key)
            ep_out.create_dataset("human_tvecs", data=tvecs)            # (N, 3)
            ep_out.create_dataset("human_rvecs", data=rvecs)            # (N, 3)
            ep_out.create_dataset("human_timestamps", data=timestamps)  # (N,)
            # store matching metadata so demo_i here maps to demo_i in ground truth
            ep_out.attrs["demo_idx"] = i
            ep_out.attrs["human_recorded"] = True
            ep_out.attrs["n_human_poses"] = len(tvecs)
            ep_out.attrs["blackout_idx"] = gt_ep.attrs["blackout_idx"]
            ep_out.attrs["remaining_len"] = gt_ep.attrs["remaining_len"]

            out_file.flush()
            recorded += 1
            print("Saved to recorded_vision_blackout_trajectories.hdf5 as demo_{}.".format(i))
            print("({} recorded so far)".format(recorded))

            # ask user if they want to see the plot for this demo
            user_input = input("Plot this trajectory against ground truth? (y/n): ").strip().lower()
            if user_input == "y":
                remaining_eef_poses = gt_ep["remaining_eef_poses"][:]
                plot_demo(i, tvecs, rvecs, timestamps, remaining_eef_poses, args.output_dir)

        # ask user if they want to continue to the next demo
        cont = input("\nContinue recording next trajectory? (y/n): ").strip().lower()
        if cont != "y":
            print("Stopping. Progress saved. Re-run script to continue from demo {}.".format(i + 1))
            break

    gt_file.close()
    out_file.close()
    print("\n========================================")
    print("Done. {} demos recorded, {} already existed.".format(recorded, skipped))
    print("Human trajectories saved to: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hdf5",
        type=str,
        required=True,
        help="path to HDF5 file from Script 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/blackout_frames",
        help="directory to save frozen RGB frames for viewing in Preview",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="EMEET SmartCam C960 4K",
        help="avfoundation camera device name as listed by: ffmpeg -f avfoundation -list_devices true -i ''",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="recording duration in seconds per demo (default: 120)",
    )

    # Added for trajectory testing — runs full pipeline without saving to HDF5
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="test full recording pipeline without saving anything to HDF5",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="(optional) path to calibration.npz file for accurate pose estimation",
    )

    args = parser.parse_args()
    record_all_demos(args)