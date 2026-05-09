import os

HERE      = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(HERE, "so101.urdf")
PORT      = "/dev/ttyACM0"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# Degrees to add to each motor reading before passing to FK or pinocchio.
# wrist_roll is -90 due to a calibration mismatch between the URDF and physical motors.
JOINT_OFFSETS = {
    "wrist_roll": -90,
}


def apply_offsets(obs: dict) -> dict:
    """Motor readings → URDF-frame values (for FK/IK input)."""
    corrected = dict(obs)
    for name, offset in JOINT_OFFSETS.items():
        key = f"{name}.pos"
        if key in corrected:
            corrected[key] = float(corrected[key]) + offset
    return corrected


def remove_offsets(urdf_angles: dict) -> dict:
    """URDF-frame values → motor commands (for send_action)."""
    motor = dict(urdf_angles)
    for name, offset in JOINT_OFFSETS.items():
        key = f"{name}.pos"
        if key in motor:
            motor[key] = float(motor[key]) - offset
    return motor
