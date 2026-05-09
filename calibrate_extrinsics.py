"""
Extrinsic calibration: compute the rigid transform from camera frame to robot base frame.

Procedure:
  1. Move the arm to a pose where the ArUco marker is clearly visible.
  2. Press Enter — the script records the marker position (ArUco) and the
     end-effector position (FK) simultaneously.
  3. Repeat for 8–12 poses spread across the workspace.
  4. Type 'done' — the transform is computed and saved to camera_to_robot.npz.

Result: R, t  such that  p_robot = R @ p_cam + t
        (p_cam is in raw camera frame: X-right, Y-down, Z-forward)
"""

import time
import numpy as np
import cv2
from cv2 import aruco
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

from camera import RealSenseCamera
from robot_config import URDF_PATH, PORT, JOINT_NAMES, apply_offsets

MARKER_SIZE   = 0.04
ARUCO_DICT    = aruco.DICT_4X4_250
N_AVG_FRAMES  = 15   # ArUco frames to average per pose for stability

# Offset from gripper_frame_link origin to ArUco marker centre, in gripper_frame_link frame.
MARKER_OFFSET = np.array([0.0180, -0.0548, -0.0518])

_h = MARKER_SIZE / 2
_OBJ_PTS = np.array([
    [-_h,  _h, 0],
    [ _h,  _h, 0],
    [ _h, -_h, 0],
    [-_h, -_h, 0],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector():
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 23
    params.adaptiveThreshWinSizeStep = 2
    params.minMarkerPerimeterRate    = 0.02
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(ARUCO_DICT), params)


def detect_marker(cam, detector, K, dist, n=N_AVG_FRAMES):
    """Return averaged ArUco tvec in raw camera frame, or None if not found."""
    samples, attempts = [], 0
    while len(samples) < n and attempts < n * 20:
        attempts += 1
        color, _, _ = cam.get_frames()
        if color is None:
            continue
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or 0 not in ids.flatten():
            continue
        idx    = list(ids.flatten()).index(0)
        pts_2d = corners[idx][0].astype(np.float32)
        _, rvec, tvec = cv2.solvePnP(
            _OBJ_PTS, pts_2d, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        samples.append(tvec.flatten())
    return np.mean(samples, axis=0) if samples else None


def solve_rigid_transform(p_src, p_dst):
    """
    Solve p_dst = R @ p_src + t using SVD (Umeyama / Procrustes).
    p_src, p_dst: (N, 3) arrays of corresponding points.
    """
    p_src = np.array(p_src, dtype=float)
    p_dst = np.array(p_dst, dtype=float)

    c_src = p_src.mean(axis=0)
    c_dst = p_dst.mean(axis=0)

    H        = (p_src - c_src).T @ (p_dst - c_dst)
    U, _, Vt = np.linalg.svd(H)
    R        = Vt.T @ U.T

    if np.linalg.det(R) < 0:   # fix reflection
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = c_dst - R @ c_src
    return R, t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Connecting to camera and robot...")
    cam  = RealSenseCamera()
    intr = cam.get_intrinsics()
    K    = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1       ],
    ], dtype=np.float64)
    dist     = np.zeros(5)
    detector = _make_detector()

    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES[:-1],
    )
    robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
    robot.connect()

    print("\n--- Extrinsic Calibration ---")
    print("Torque will be OFF so you can pose the arm freely.")
    print("Aim for 12+ poses filling a fat 3D blob across the workspace.")
    print("  - Critical: span the full Y range (near + far from camera).")
    print("  - Vary wrist orientation, not just position.")
    print("  - Reach the corners of where the cup will actually sit.")
    print("Type 'done' when finished.\n")

    p_cam_list   = []
    p_robot_list = []

    robot.bus.disable_torque()
    print("Torque disabled — arm is free to move.\n")

    while True:
        cmd = input(f"[{len(p_cam_list)} poses] Pose arm, then Enter to record / 'done' to finish: ").strip().lower()
        if cmd == "done":
            break

        # Briefly re-enable torque so the arm holds still during measurement
        robot.bus.enable_torque()
        time.sleep(0.5)  # let motors settle before reading

        # --- Camera measurement ---
        print("  Detecting ArUco marker...", end=" ", flush=True)
        p_cam = detect_marker(cam, detector, K, dist)

        # --- Robot measurement ---
        obs    = apply_offsets(robot.get_observation())

        robot.bus.disable_torque()  # free to move again immediately after reading

        if p_cam is None:
            print("FAILED — marker not visible, reposition and try again.")
            continue

        q_deg  = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES[:-1]], dtype=float)
        T      = kin.forward_kinematics(q_deg)
        p_robot = T[:3, 3] + T[:3, :3] @ MARKER_OFFSET

        print(f"OK")
        print(f"    cam   = ({p_cam[0]:+.4f}, {p_cam[1]:+.4f}, {p_cam[2]:+.4f}) m")
        print(f"    robot = ({p_robot[0]:+.4f}, {p_robot[1]:+.4f}, {p_robot[2]:+.4f}) m")

        p_cam_list.append(p_cam)
        p_robot_list.append(p_robot)

    cam.stop()
    robot.disconnect()

    if len(p_cam_list) < 4:
        print("\nNeed at least 4 poses — exiting without saving.")
        return

    # --- Pose distribution diagnostic ---
    P = np.array(p_robot_list)
    spread = P.max(axis=0) - P.min(axis=0)
    P_centered = P - P.mean(axis=0)
    sigma = np.linalg.svd(P_centered, compute_uv=False)
    cond = sigma[0] / sigma[2] if sigma[2] > 1e-9 else float("inf")
    print(f"\n--- Pose distribution (robot frame) ---")
    print(f"Spread:      X={spread[0]*100:.1f} cm  Y={spread[1]*100:.1f} cm  Z={spread[2]*100:.1f} cm")
    print(f"SVD sigmas:  {sigma[0]:.3f}, {sigma[1]:.3f}, {sigma[2]:.3f}")
    print(f"Condition #: {cond:.1f}  (want < 10; > 10 means an axis is under-sampled)")
    if cond > 10:
        weakest = ["X", "Y", "Z"][int(np.argmin(spread))]
        print(f"  WARNING: poses are pancake-like; weakest axis ≈ {weakest}.")
        print(f"  Rotation around that axis is under-constrained → expect lever-arm errors at runtime.")

    R, t = solve_rigid_transform(p_cam_list, p_robot_list)

    # Per-pose errors
    errors = np.array([np.linalg.norm(R @ pc + t - pr)
                       for pc, pr in zip(p_cam_list, p_robot_list)])
    print(f"\n--- Per-pose errors ---")
    for i, e in enumerate(errors):
        flag = "  <-- OUTLIER" if e > 3 * np.median(errors) else ""
        print(f"  Pose {i+1:2d}: {e*100:.1f} cm{flag}")

    # Refit without outliers (> 3x median)
    threshold = 3 * np.median(errors)
    keep = errors <= threshold
    if keep.sum() < len(errors):
        print(f"\nRemoving {(~keep).sum()} outlier(s) and refitting...")
        p_cam_clean   = [p for p, k in zip(p_cam_list,   keep) if k]
        p_robot_clean = [p for p, k in zip(p_robot_list, keep) if k]
        R, t = solve_rigid_transform(p_cam_clean, p_robot_clean)
        final_errors = np.array([np.linalg.norm(R @ pc + t - pr)
                                 for pc, pr in zip(p_cam_clean, p_robot_clean)])
    else:
        print(f"\nNo outliers detected.")
        final_errors = errors

    print(f"\n--- Final Results ({len(final_errors)} poses) ---")
    print(f"Mean error: {np.mean(final_errors)*100:.1f} cm")
    print(f"Max  error: {np.max(final_errors)*100:.1f} cm")
    print(f"\nR =\n{R}")
    print(f"t = {t}")

    np.savez("camera_to_robot.npz", R=R, t=t)
    print("\nSaved to camera_to_robot.npz")
    print("Usage:  p_robot = R @ p_cam + t   (p_cam in raw camera frame, Y-down)")


if __name__ == "__main__":
    main()
