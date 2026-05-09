"""
Measures the offset from gripper_frame_link to the ArUco marker centre,
expressed in the gripper_frame_link local frame.

Run with the arm held in any pose where the marker is clearly visible.
Prints d_marker (3 values) to paste into calibrate_extrinsics.py.
"""

import numpy as np
import cv2
from cv2 import aruco
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

from camera import RealSenseCamera
from robot_config import URDF_PATH, PORT, JOINT_NAMES, apply_offsets

MARKER_SIZE  = 0.04
ARUCO_DICT   = aruco.DICT_4X4_250
N_AVG_FRAMES = 30

_h = MARKER_SIZE / 2
_OBJ_PTS = np.array([[-_h,_h,0],[_h,_h,0],[_h,-_h,0],[-_h,-_h,0]], dtype=np.float32)

cam  = RealSenseCamera()
intr = cam.get_intrinsics()
K    = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]], dtype=np.float64)
dist = np.zeros(5)

params = aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin  = 3
params.adaptiveThreshWinSizeMax  = 23
params.adaptiveThreshWinSizeStep = 2
params.minMarkerPerimeterRate    = 0.02
detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(ARUCO_DICT), params)

kin = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link",
                      joint_names=JOINT_NAMES[:-1])

robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
robot.connect()

# Load the existing camera→robot transform
cal = np.load("camera_to_robot.npz")
R_cr, t_cr = cal["R"], cal["t"]

input("Hold the arm still with the marker clearly visible, then press Enter...")

# --- Average ArUco detection ---
samples, attempts = [], 0
while len(samples) < N_AVG_FRAMES and attempts < N_AVG_FRAMES * 20:
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
    _, rvec, tvec = cv2.solvePnP(_OBJ_PTS, pts_2d, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    samples.append(tvec.flatten())

cam.stop()

if not samples:
    print("Marker not detected.")
    robot.disconnect()
    exit()

p_cam   = np.mean(samples, axis=0)
p_robot_marker = R_cr @ p_cam + t_cr   # marker centre in robot base frame

# --- FK: gripper_frame_link position and orientation in robot base frame ---
obs   = apply_offsets(robot.get_observation())
q_deg = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES[:-1]], dtype=float)
T     = kin.forward_kinematics(q_deg)
p_robot_fk = T[:3, 3]    # gripper_frame_link origin in robot base frame
R_gripper  = T[:3, :3]   # rotation of gripper_frame_link in robot base frame

robot.disconnect()

# Offset in robot base frame, then rotate into gripper_frame_link frame
offset_world   = p_robot_marker - p_robot_fk
offset_gripper = R_gripper.T @ offset_world

print(f"\nMarker in robot frame : {p_robot_marker}")
print(f"FK (gripper_frame_link): {p_robot_fk}")
print(f"Offset in world frame  : {offset_world}  (magnitude: {np.linalg.norm(offset_world)*100:.1f} cm)")
print(f"\nd_marker in gripper_frame_link frame:")
print(f"  X: {offset_gripper[0]:+.4f} m")
print(f"  Y: {offset_gripper[1]:+.4f} m")
print(f"  Z: {offset_gripper[2]:+.4f} m")
print(f"\nPaste this into calibrate_extrinsics.py:")
print(f"  MARKER_OFFSET = np.array([{offset_gripper[0]:.4f}, {offset_gripper[1]:.4f}, {offset_gripper[2]:.4f}])")
