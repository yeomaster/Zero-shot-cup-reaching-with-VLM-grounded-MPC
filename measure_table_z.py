"""Park the gripper just touching the table, then run this script to read the
robot-frame FK z value at that pose. Use the printed z (minus a small safety
margin like 1cm) as FLOOR_Z in executor.py.

Steps:
  1. Open torque (so you can backdrive the arm by hand)
  2. Position the gripper tip flush with the table surface
  3. Hold it there
  4. Run: uv run python measure_table_z.py
  5. Press Enter when the gripper is in place
"""
import numpy as np
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

from robot_config import URDF_PATH, PORT, JOINT_NAMES
from executor import current_joints

robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
robot.connect()
robot.bus.disable_torque()   # let user backdrive the arm by hand
kin = RobotKinematics(urdf_path=URDF_PATH,
                      target_frame_name="gripper_frame_link",
                      joint_names=JOINT_NAMES[:-1])

try:
    print("\nTorque DISABLED — you can move the arm by hand.")
    print("Position the gripper tip flush with the table.")
    print("Support the arm so it doesn't drop. Press Enter to sample.")
    samples = []
    while True:
        cmd = input("[Enter to sample, 's' to stop]: ").strip().lower()
        if cmd == "s":
            break
        q = current_joints(robot)[:5]
        T = kin.forward_kinematics(q)
        pos = T[:3, 3]
        samples.append(pos)
        print(f"  sample {len(samples)}: x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}")

    if samples:
        arr = np.asarray(samples)
        print("\n--- Summary ---")
        print(f"  z mean: {arr[:,2].mean():+.4f} m")
        print(f"  z min : {arr[:,2].min():+.4f} m")
        print(f"  z max : {arr[:,2].max():+.4f} m")
        print(f"\nSuggested FLOOR_Z = {arr[:,2].mean() + 0.005:+.3f}  "
              f"(table z + 5 mm safety margin)")
finally:
    robot.disconnect()
