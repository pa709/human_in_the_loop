import cv2
import numpy as np
import subprocess
import os
import argparse


# --- Board definition — must match exactly what was used to generate the board ---
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((7, 5), 0.03, 0.015, DICTIONARY)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, DETECTOR_PARAMS)
CHARUCO_DETECTOR = cv2.aruco.CharucoDetector(BOARD)


def capture_frame(camera_index, width=1280, height=720):
    """Capture one frame from webcam via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", "{}x{}".format(width, height),
            "-i", "{}:none".format(camera_index),
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-loglevel", "error",
            "-"
        ],
        capture_output=True
    )
    raw = result.stdout
    expected = width * height * 3
    if len(raw) < expected:
        print("Frame capture failed. Check camera index.")
        return None
    frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((height, width, 3))
    return frame


def save_preview(frame, path):
    """Save frame as jpg and open in Preview."""
    cv2.imwrite(path, frame)
    """subprocess.run(["open", path])"""


def detect_charuco(frame):
    """
    Detect ChArUco corners in frame using new API.
    Returns (charuco_corners, charuco_ids) or (None, None) if not detected.
    """
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        CHARUCO_DETECTOR.detectBoard(frame)

    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None

    return charuco_corners, charuco_ids


def calibrate(args):
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    count = 0
    preview_path = "/tmp/calib_preview.jpg"

    print("Camera Calibration")
    print("==================")
    print("Hold the ChArUco board in view of the EMEET camera.")
    print("Vary angle and distance across captures.")
    print("Aim for at least 15 captures.")
    print("")
    print("Commands:")
    print("  ENTER — capture frame")
    print("  Q + ENTER — quit and run calibration")
    print("")

    try:
      while True:
        user_input = input("Press ENTER to capture (or Q+ENTER to calibrate, Ctrl+C to emergency-save): ").strip().lower()

        if user_input == "q":
            break

        print("Capturing frame...")
        frame = capture_frame(args.camera_index)
        if frame is None:
            print("Failed to capture. Try again.")
            continue

        image_size = (frame.shape[1], frame.shape[0])  # (width, height)

        # detect ChArUco
        charuco_corners, charuco_ids = detect_charuco(frame)

        if charuco_corners is None:
            print("Board not detected in this frame. Reposition and try again.")
            # save preview anyway so you can see what the camera sees
            save_preview(frame, preview_path)
            print("Preview saved to: {}".format(preview_path))
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        count += 1
        print("Captured {} — detected {} ChArUco corners.".format(
            count, len(charuco_ids)
        ))

        # draw detections on frame and save preview
        preview = frame.copy()
        cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
        save_preview(preview, preview_path)
        print("Preview saved to: {}".format(preview_path))

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received — running calibration on {} captures so far.".format(count))

    if count < 5:
        print("Not enough captures (got {}, need at least 5). Exiting.".format(count))
        return

    MIN_CORNERS = 15
    filtered_corners = []
    filtered_ids = []
    for c, i in zip(all_charuco_corners, all_charuco_ids):
        if len(c) >= MIN_CORNERS:
            filtered_corners.append(c)
            filtered_ids.append(i)
    print("Filtered {}/{} captures (kept views with >= {} corners).".format(len(filtered_corners), count, MIN_CORNERS))
    if len(filtered_corners) < 5:
        print("Not enough good captures after filtering. Exiting.")
        return
    all_charuco_corners = filtered_corners
    all_charuco_ids = filtered_ids
    count = len(filtered_corners)

    print("\nCalibrating from {} captures...".format(count))

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        BOARD,
        image_size,
        None,
        None
    )

    print("Calibration done. RMS reprojection error: {:.4f}".format(ret))
    if ret > 1.0:
        print("WARNING: RMS error > 1.0 — consider recapturing with more varied angles.")
    else:
        print("RMS error looks good.")

    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:\n", dist_coeffs)

    # save as npz — Script 2 expects keys: camera_matrix, dist_coeffs
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(output_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("\nCalibration saved to: {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_index",
        type=int,
        default=1,
        help="avfoundation camera index for EMEET webcam (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration.npz",
        help="path to save calibration.npz (default: calibration.npz in current directory)",
    )
    args = parser.parse_args()
    calibrate(args)
