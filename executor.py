"""
Executes robot motion to a goal position on the SO-ARM101.

Frame chain:
  voxel-map (Y-up)  --flip Y-->  camera (Y-down)  --R,t-->  robot base frame
  robot base frame  --IK-->      joint angles      --remove_offsets-->  motor cmd
"""

import time
import numpy as np
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

from robot_config import URDF_PATH, PORT, JOINT_NAMES, apply_offsets, remove_offsets

# Lift applied directly in robot-frame Z (up axis) after the frame conversion.
LIFT_OFFSET_M  = 0.13
# Distance from gripper_frame_link origin to the actual finger tip, along the
# gripper's local +Z axis (URDF puts gripper_frame_link 9.8cm short of the tip,
# rotated 180° so the local Z points back toward the wrist — we negate below).
# Tune by eye: larger value → goal/floor checks treat the tip as further out.
TIP_OFFSET_M   = 0.06
# Hard floor: robot-frame Z will never be commanded below this (table protection).
# Table measured at robot-frame z ≈ -0.045; FLOOR_Z = -0.030 sits 1.5 cm above
# the table. Gripper rest pose typically lives at z ≈ -0.01, so the absolute
# stop won't fire on normal configurations but the table stays protected.
FLOOR_Z        = -0.030
# Soft buffer above the floor: we start aborting when the gripper would dip
# this close to FLOOR_Z, not just below it.
FLOOR_BUFFER   = 0.001
# When the floor guard triggers, we don't snap back to FLOOR_Z in one step
# (that's a big upward jump that causes IK to flip joints → severe jitter).
# Instead we nudge the gripper up by this amount each step until it clears.
FLOOR_RECOVERY_STEP = 0.003
# Max joint change allowed per IK step (degrees). The FK floor check below
# is the real safety net for slam protection, so this only needs to catch
# truly insane IK solutions (full joint flips), not normal initial alignment.
MAX_JOINT_DELTA_DEG = 50.0
# Interpolation steps for the move (higher = slower/smoother).
INTERP_STEPS   = 70
# Seconds between each interpolation step.
INTERP_PAUSE   = 0.05
# Gripper motor value: positive = open, zero = closed (SO-ARM101 calibrated convention).
GRIPPER_CLOSED    = 0.0
GRIPPER_HALF_OPEN = 17.5
GRIPPER_OPEN      = 35.0

# Default starting pose in robot base frame. Centre of the SO-101 sweet
# spot (x ∈ [0.20, 0.40], y ∈ [-0.20, 0.20], z ∈ [0.05, 0.30]) so every
# run begins from a known reachable configuration before YOLO/A*/MPC.
HOME_POSE_ROBOT = np.array([0.22, 0.0, 0.18])


def load_calibration(path: str = "camera_to_robot.npz"):
    cal = np.load(path)
    return cal["R"], cal["t"]


def voxmap_to_robot(p_vox: np.ndarray, R: np.ndarray, t: np.ndarray,
                    lift: float = 0.0) -> np.ndarray:
    """Voxel-map (Y-up) → robot base frame. lift adds directly to robot Y after transform."""
    p_cam   = np.array([p_vox[0], -p_vox[1], p_vox[2]])  # Y-up → Y-down
    p_robot = R @ p_cam + t
    p_robot[2] += lift
    return p_robot


def current_joints(robot) -> np.ndarray:
    """Current joint positions in URDF-frame degrees, all 6 joints."""
    obs = apply_offsets(robot.get_observation())
    return np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=float)


def tip_from_T(T: np.ndarray) -> np.ndarray:
    """Project gripper_frame_link FK pose forward by TIP_OFFSET_M to the
    physical finger tip. The URDF rotates gripper_frame_link 180° relative to
    the wrist, so the local +Z axis points back toward the wrist — we negate
    to reach the tip."""
    return T[:3, 3] + TIP_OFFSET_M * (-T[:3, 2])


def tip_pose(T: np.ndarray) -> np.ndarray:
    """Same as `T` but with translation shifted to the tip — use as IK target
    so IK plans around where the tip should be, not the wrist link."""
    T_tip = T.copy()
    T_tip[:3, 3] = tip_from_T(T)
    return T_tip


def wrist_target_from_tip_target(T_tip_goal: np.ndarray) -> np.ndarray:
    """Inverse of `tip_pose`: given a desired tip pose, return the wrist-link
    pose IK should hit so the tip lands on T_tip_goal."""
    T_wrist = T_tip_goal.copy()
    T_wrist[:3, 3] = T_tip_goal[:3, 3] - TIP_OFFSET_M * (-T_tip_goal[:3, 2])
    return T_wrist


def _offset(name: str) -> float:
    from robot_config import JOINT_OFFSETS
    return JOINT_OFFSETS.get(name, 0.0)


def send_joints(robot, q_urdf_5: np.ndarray, gripper_val: float):
    obs_raw = robot.get_observation()
    action  = dict(obs_raw)
    for i, name in enumerate(JOINT_NAMES[:-1]):
        action[f"{name}.pos"] = q_urdf_5[i] - _offset(name)
    action["gripper.pos"] = gripper_val
    robot.send_action(action)


def move_to(robot, kin, goal_pos_robot: np.ndarray, gripper_val: float = GRIPPER_OPEN,
            step_size: float = 0.02):
    """
    Move to goal_pos_robot (robot base frame) via Cartesian stepping.
    Takes small steps of step_size metres, calling IK for each increment.
    Returns True on success, False if a step fails.

    Floor protection works in three places:
      1. The goal itself is clamped to FLOOR_Z up front.
      2. Each Cartesian sub-target is clamped to FLOOR_Z.
      3. After IK, FK is recomputed on the proposed joints — if the gripper
         would actually end up at/below FLOOR_Z + FLOOR_BUFFER, the step is
         rejected. This catches cases where IK satisfies position but flips
         the gripper orientation downward.
    """
    goal_pos_robot = goal_pos_robot.copy()
    if goal_pos_robot[2] < FLOOR_Z:
        print(f"  Goal Z={goal_pos_robot[2]:.3f} below FLOOR_Z={FLOOR_Z} — clamping.")
        goal_pos_robot[2] = FLOOR_Z

    print(f"  Moving to {np.round(goal_pos_robot, 3)}")

    for _ in range(200):   # max 200 steps (~4m of travel at 2cm/step)
        q_now   = current_joints(robot)[:5]
        T_now   = kin.forward_kinematics(q_now)
        tip_now = tip_from_T(T_now)

        error = goal_pos_robot - tip_now
        dist  = np.linalg.norm(error)

        if dist < 0.01:   # within 1cm — close enough
            print(f"  Reached target (error {dist*100:.1f} cm)")
            return True

        # Take a step of step_size toward the goal.
        step         = error * min(step_size / dist, 1.0)
        next_tip     = tip_now + step

        # Directional floor guard: only intervene when the tip is *descending*
        # below FLOOR_Z. If the tip is already below the floor and trying to
        # climb out, let the upward step through.
        if next_tip[2] < FLOOR_Z and next_tip[2] < tip_now[2] - 0.001:
            print("  z axis limit reached! going back up")
            next_tip[2] = tip_now[2] + FLOOR_RECOVERY_STEP

        # IK target is the wrist pose that puts the *tip* at next_tip.
        T_tip_goal        = T_now.copy()
        T_tip_goal[:3, 3] = next_tip
        T_step            = wrist_target_from_tip_target(T_tip_goal)

        # Stronger orientation weight keeps the gripper from flipping downward
        # mid-trajectory just to satisfy the position target.
        q_step = kin.inverse_kinematics(
            q_now, T_step, position_weight=1.0, orientation_weight=0.5
        )

        # Post-IK FK floor guard: catch cases where IK hit the position target
        # but flipped the gripper orientation downward.
        tip_after = tip_from_T(kin.forward_kinematics(q_step))
        if tip_after[2] < FLOOR_Z + FLOOR_BUFFER and tip_after[2] < tip_now[2] - 0.001:
            print(f"  z axis limit reached! going back up (post-IK tip_z={tip_after[2]:.3f})")
            return False

        delta = np.abs(q_step - q_now)
        if delta.max() > MAX_JOINT_DELTA_DEG:
            print(f"  IK step rejected (max delta {delta.max():.1f} deg "
                  f"> {MAX_JOINT_DELTA_DEG}) — stopping.")
            return False

        send_joints(robot, q_step, gripper_val)
        time.sleep(INTERP_PAUSE)

    print("  Max steps reached.")
    return False


def track_target(goal_provider, gripper_val: float = GRIPPER_CLOSED,
                 step_size: float = 0.01, pause: float = 0.10,
                 max_misses: int = 25,
                 detect_every_n: int = 5,
                 goal_smoothing: float = 0.3,
                 calibration_path: str = "camera_to_robot.npz"):
    """
    Closed-loop visual servo: keep aiming the gripper at whatever
    `goal_provider()` returns, forever, until the user hits Ctrl-C or the
    target is lost for max_misses consecutive detection attempts.

    goal_provider() must return either:
      - np.ndarray shape (3,) in robot frame (the new goal), or
      - None if the target wasn't detected this attempt.

    The provider is called every `detect_every_n` IK iterations to keep the
    loop responsive without running YOLO at full step rate.

    No "reached target" exit — once at the goal, the arm just stops moving
    (Cartesian step is zero) and waits for the goal to shift.
    """
    R, t = load_calibration(calibration_path)
    kin  = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link",
                           joint_names=JOINT_NAMES[:-1])
    robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
    robot.connect()

    last_goal = None
    misses = 0
    iteration = 0
    rejections = {"ik_delta": 0, "floor_guard": 0, "moves_sent": 0}
    print("  Tracking target — Ctrl+C to stop.")

    try:
        while True:
            iteration += 1

            if iteration == 1 or iteration % detect_every_n == 0:
                new_goal = goal_provider()
                if new_goal is None:
                    misses += 1
                    if misses >= max_misses:
                        print(f"  Lost target for {max_misses} attempts — holding last position.")
                        break
                else:
                    new_goal = np.asarray(new_goal, dtype=float)
                    if last_goal is None:
                        last_goal = new_goal
                    else:
                        # EMA: damp bbox-center jitter so a stationary cup doesn't
                        # produce a moving goal. alpha=0.3 → ~3 detections to fully
                        # respond to a real cup motion.
                        last_goal = goal_smoothing * new_goal + (1 - goal_smoothing) * last_goal
                    if last_goal[2] < FLOOR_Z:
                        last_goal[2] = FLOOR_Z
                    misses = 0
                    q_dbg = current_joints(robot)[:5]
                    tip_dbg = tip_from_T(kin.forward_kinematics(q_dbg))
                    err = last_goal - tip_dbg
                    print(f"  goal: x={last_goal[0]:+.3f} y={last_goal[1]:+.3f} z={last_goal[2]:+.3f}  "
                          f"tip: x={tip_dbg[0]:+.3f} y={tip_dbg[1]:+.3f} z={tip_dbg[2]:+.3f}  "
                          f"err={np.linalg.norm(err)*100:.1f}cm  "
                          f"sent/skip-ik/skip-floor: {rejections['moves_sent']}/{rejections['ik_delta']}/{rejections['floor_guard']}")

            if last_goal is None:
                time.sleep(pause)
                continue

            goal_pos = last_goal.copy()

            q_now   = current_joints(robot)[:5]
            T_now   = kin.forward_kinematics(q_now)
            tip_now = tip_from_T(T_now)

            error = goal_pos - tip_now
            dist  = np.linalg.norm(error)

            if dist < 0.005:   # already on target — hold, don't exit
                time.sleep(pause)
                continue

            step          = error * min(step_size / dist, 1.0)
            next_tip      = tip_now + step

            # Directional floor guard: only intervene when actually descending
            # below FLOOR_Z. Snap-up to FLOOR_Z in one step caused IK to flip
            # joints and jitter — instead nudge up by FLOOR_RECOVERY_STEP only.
            if next_tip[2] < FLOOR_Z and next_tip[2] < tip_now[2] - 0.001:
                print("  z axis limit reached! going back up")
                next_tip[2] = tip_now[2] + FLOOR_RECOVERY_STEP
                rejections["floor_guard"] += 1

            T_tip_goal        = T_now.copy()
            T_tip_goal[:3, 3] = next_tip
            T_step            = wrist_target_from_tip_target(T_tip_goal)

            q_step = kin.inverse_kinematics(
                q_now, T_step, position_weight=1.0, orientation_weight=0.2
            )

            # Slam guards: skip the step instead of bailing — the cup might
            # move into a reachable spot before the next detection.
            if np.abs(q_step - q_now).max() > MAX_JOINT_DELTA_DEG:
                rejections["ik_delta"] += 1
                time.sleep(pause)
                continue

            # Post-IK FK floor guard: skip if IK twisted the gripper below floor.
            tip_after = tip_from_T(kin.forward_kinematics(q_step))
            if tip_after[2] < FLOOR_Z + FLOOR_BUFFER and tip_after[2] < tip_now[2] - 0.001:
                print(f"  z axis limit reached! going back up (post-IK tip_z={tip_after[2]:.3f})")
                rejections["floor_guard"] += 1
                time.sleep(pause)
                continue

            send_joints(robot, q_step, gripper_val)
            rejections["moves_sent"] += 1
            time.sleep(pause)
    except KeyboardInterrupt:
        print("\n  Tracking stopped by user.")
    finally:
        try:
            _hold_until_user_releases(robot)
        finally:
            robot.disconnect()


def open_and_home(home_pose: np.ndarray = HOME_POSE_ROBOT,
                  gripper_val: float = GRIPPER_CLOSED,
                  step_size: float = 0.02,
                  position_weight: float = 1.0,
                  orientation_weight: float = 0.3):
    """Open the robot, drive the gripper to a known starting pose, return
    the still-connected (robot, kin). The caller is responsible for the
    eventual `_hold_until_user_releases` + `robot.disconnect()`.

    We start every run from this pose so the rest of the pipeline (MPC
    tracking in particular) begins from a configuration where small
    Cartesian steps map cleanly to small joint motions — i.e. away from
    the singularities and folded-arm basins that left the gripper stuck
    in earlier runs."""
    kin = RobotKinematics(urdf_path=URDF_PATH,
                          target_frame_name="gripper_frame_link",
                          joint_names=JOINT_NAMES[:-1])
    robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
    robot.connect()
    print(f"  Moving to home pose {np.round(home_pose, 3)}...")
    ok = move_to(robot, kin, np.asarray(home_pose, dtype=float),
                 gripper_val=gripper_val, step_size=step_size)
    if not ok:
        print("  WARNING: did not reach home cleanly — continuing anyway.")
    return robot, kin


def mpc_track(scene_provider,
              gripper_val: float = GRIPPER_CLOSED,
              step_size: float = 0.02,
              horizon: int = 6,
              n_candidates: int = 64,
              smoothness_weight: float = 0.1,
              pause: float = 0.10,
              detect_every_n: int = 3,
              max_misses: int = 25,
              goal_dead_band: float = 0.02,
              workspace_lo=(0.10, -0.25, FLOOR_Z),
              workspace_hi=(0.45,  0.25, 0.35),
              calibration_path: str = "camera_to_robot.npz",
              robot=None,
              kin=None):
    """
    VoxPoser-style closed-loop controller.

    `scene_provider()` returns either:
        {"target":          np.array([x, y, z]),  # robot frame, hover-lifted
         "obstacles":       [np.array([x, y, z]), ...],
         "obstacle_radii":  [float, ...]}         # metres, optional
    or None if perception failed this attempt.

    Each iteration:
      1. (Every detect_every_n iters) refresh the scene.
      2. Build a fresh ValueField — attractor at the target, Gaussian
         repulsor at every other detection, floor barrier, workspace box.
      3. MPCPlanner.step() returns the next Cartesian waypoint.
      4. Apply IK + slam guard + directional floor guard, then send to motors.
    """
    from value_field import ValueField
    from mpc import MPCPlanner

    R, t = load_calibration(calibration_path)
    own_robot = robot is None
    if kin is None:
        kin = RobotKinematics(urdf_path=URDF_PATH,
                              target_frame_name="gripper_frame_link",
                              joint_names=JOINT_NAMES[:-1])
    if robot is None:
        robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
        robot.connect()

    planner = MPCPlanner(step_size=step_size,
                         horizon=horizon,
                         n_candidates=n_candidates,
                         smoothness_weight=smoothness_weight)

    scene = None
    misses = 0
    iteration = 0
    rejections = {"ik_delta": 0, "floor_guard": 0, "moves_sent": 0}

    # Rescue-mode: when the gripper FK reaches the floor, abandon the current
    # target and slowly drive joints back to the home pose in JOINT space
    # (no IK). This sidesteps both the wrist-flip and joint-limit failure
    # modes that kept IK-based rescue stuck.
    RESCUE_TRIGGER_Z    = FLOOR_Z + 0.005   # enter rescue when within 5 mm of floor
    RESCUE_LIFT         = 0.10              # climb this much above FLOOR_Z before resuming
    RESCUE_PER_STEP_DEG = 5.0               # max joint delta per rescue step
    in_rescue           = False

    # Posture bias: when the goal's radial distance from the base is below
    # POSTURE_BIAS_RADIUS, blend lift joints toward Q_HOME by up to
    # POSTURE_BIAS_MAX. This stops the arm from staying fully extended after
    # chasing a far cup when a closer cup appears — full-extension lift
    # joints have very low IK gradient toward folded-back configs, so without
    # this nudge the arm just hovers at maximum reach.
    POSTURE_BIAS_RADIUS = 0.30   # metres; below this the bias starts ramping in
    POSTURE_BIAS_MAX    = 0.05   # max blend weight toward Q_HOME (0..1)

    # Capture the home joint configuration at startup. open_and_home() ran
    # before mpc_track, so the current joints are the known-good home pose.
    Q_HOME = current_joints(robot)[:5].copy()
    print(f"  Captured rescue home joints: {np.round(Q_HOME, 1)}°")

    # Sent-vs-actual diagnostic: remember the last command we sent so we can
    # compare it against the observation we read in the next iteration. If
    # the motor isn't responding, sent_q != actual_q for that joint.
    last_sent_q = None

    print("  MPC tracking — Ctrl+C to stop.")

    try:
        while True:
            iteration += 1

            if iteration == 1 or iteration % detect_every_n == 0:
                got = scene_provider()
                if got is None:
                    misses += 1
                    if misses >= max_misses:
                        print(f"  Lost target for {max_misses} attempts — holding.")
                        break
                else:
                    scene = got
                    misses = 0

            if scene is None:
                time.sleep(pause)
                continue

            field = ValueField(floor_z=FLOOR_Z, floor_buffer=FLOOR_BUFFER)
            field.set_workspace(workspace_lo, workspace_hi)
            field.add_attractor(scene["target"], weight=1.0)
            obs_list = scene.get("obstacles", []) or []
            radii    = scene.get("obstacle_radii", []) or []
            for i, obs in enumerate(obs_list):
                sigma = radii[i] if i < len(radii) else 0.05
                field.add_repulsor(obs, sigma=sigma, weight=0.3)

            q_now   = current_joints(robot)[:5]
            T_now   = kin.forward_kinematics(q_now)
            tip_now = tip_from_T(T_now)

            err = scene["target"] - tip_now
            err_norm = float(np.linalg.norm(err))

            if iteration == 1 or iteration % detect_every_n == 0:
                print(f"  goal: x={scene['target'][0]:+.3f} y={scene['target'][1]:+.3f} z={scene['target'][2]:+.3f}  "
                      f"tip: x={tip_now[0]:+.3f} y={tip_now[1]:+.3f} z={tip_now[2]:+.3f}  "
                      f"err={err_norm*100:.1f}cm  obs={len(obs_list)}  "
                      f"sent/skip-ik/skip-floor: {rejections['moves_sent']}/{rejections['ik_delta']}/{rejections['floor_guard']}")

            # Sent-vs-actual diagnostic: compare the q we commanded last
            # iteration against the observed q this iteration. Big deltas
            # between sent-target and actual-now mean the motor isn't tracking.
            if last_sent_q is not None and iteration % 30 == 0:
                tracking_err = q_now - last_sent_q
                print(f"    [motor] sent={np.round(last_sent_q, 1)}  "
                      f"actual={np.round(q_now, 1)}  "
                      f"err={np.round(tracking_err, 2)}°")

            # Rescue mode: tip at/below floor — climb +Z slowly, ignoring
            # the target until clear. Bypasses the strict floor stop because
            # we're explicitly trying to lift, and a slightly-below-floor IK
            # output is acceptable as long as the step is upward.
            if not in_rescue and tip_now[2] <= RESCUE_TRIGGER_Z:
                in_rescue = True
                print(f"  [rescue] Tip at z={tip_now[2]:+.3f} (floor={FLOOR_Z:+.3f}). "
                      f"Climbing to z={FLOOR_Z + RESCUE_LIFT:+.3f}...")
            if in_rescue:
                if tip_now[2] >= FLOOR_Z + RESCUE_LIFT:
                    in_rescue = False
                    print(f"  [rescue] Cleared at tip_z={tip_now[2]:+.3f}. Resuming target tracking.")
                else:
                    # Joint-space rescue: nudge ONLY the lift joints
                    # (shoulder_lift idx 1, elbow_flex idx 2) toward Q_HOME by
                    # up to RESCUE_PER_STEP_DEG per step. shoulder_pan (yaw)
                    # and wrist joints stay frozen so the gripper rises in
                    # place rather than warping back to a specific xyz home.
                    LIFT_JOINTS = (1, 2)
                    delta = Q_HOME - q_now
                    delta_clipped = np.clip(delta, -RESCUE_PER_STEP_DEG, RESCUE_PER_STEP_DEG)
                    q_step_smooth = q_now.copy()
                    for j in LIFT_JOINTS:
                        q_step_smooth[j] = q_now[j] + delta_clipped[j]
                    send_joints(robot, q_step_smooth, gripper_val)
                    last_sent_q = q_step_smooth.copy()
                    rejections["moves_sent"] += 1
                    if iteration % 30 == 0:
                        print(f"    [rescue] tip_z={tip_now[2]:+.3f}  "
                              f"Δq[lift]={np.round(delta_clipped[list(LIFT_JOINTS)], 1)}°  "
                              f"(joints 0,3,4 frozen)")
                    time.sleep(pause)
                    continue

            # Dead band: once within goal_dead_band of the target, hold
            # position. Below this radius the attractor gradient is flat
            # and YOLO+depth noise dominates, which causes jitter.
            if err_norm < goal_dead_band:
                time.sleep(pause)
                continue

            # Inject the raw goal direction as a guaranteed candidate so
            # the random shooting cannot miss it by a few degrees.
            next_tip, _best_score = planner.step(
                tip_now, field, goal_hint=(scene["target"] - tip_now)
            )

            # Directional floor guard: only intervene when the MPC step is
            # *descending* below FLOOR_Z. Snap-up to FLOOR_Z in one step
            # caused jitter — instead nudge up by FLOOR_RECOVERY_STEP.
            if next_tip[2] < FLOOR_Z and next_tip[2] < tip_now[2] - 0.001:
                print("  z axis limit reached! going back up")
                next_tip[2] = tip_now[2] + FLOOR_RECOVERY_STEP
                rejections["floor_guard"] += 1

            # IK target is the wrist pose that puts the *tip* at next_tip.
            T_tip_goal        = T_now.copy()
            T_tip_goal[:3, 3] = next_tip
            T_step            = wrist_target_from_tip_target(T_tip_goal)

            q_step = kin.inverse_kinematics(
                q_now, T_step, position_weight=50.0, orientation_weight=0.01
            )

            # Posture bias: when the goal is close to the base (small radial
            # reach), pull joints toward Q_HOME so the arm folds back instead
            # of staying extended. Bias strength ramps from 0 (goal far) to
            # POSTURE_BIAS_MAX (goal at base). Only the lift joints are
            # blended — wrist joints stay where IK put them so the gripper
            # keeps pointing at the cup.
            goal_radial = float(np.linalg.norm(scene["target"][:2]))
            posture_alpha = float(np.clip(
                (POSTURE_BIAS_RADIUS - goal_radial) / POSTURE_BIAS_RADIUS,
                0.0, 1.0,
            )) * POSTURE_BIAS_MAX
            if posture_alpha > 0.0:
                LIFT_JOINTS = (1, 2)
                for j in LIFT_JOINTS:
                    q_step[j] = (1.0 - posture_alpha) * q_step[j] + posture_alpha * Q_HOME[j]

            if np.abs(q_step - q_now).max() > MAX_JOINT_DELTA_DEG:
                rejections["ik_delta"] += 1
                time.sleep(pause)
                continue

            # Absolute floor stop: if the IK output puts the tip below the
            # table, reject regardless of current position. Without this, the
            # tip can tunnel down 1mm at a time when IK can't satisfy +Z
            # requests cleanly.
            tip_after = tip_from_T(kin.forward_kinematics(q_step))
            if tip_after[2] < FLOOR_Z:
                rejections["floor_guard"] += 1
                time.sleep(pause)
                continue
            if tip_after[2] < FLOOR_Z + FLOOR_BUFFER and tip_after[2] < tip_now[2] - 0.001:
                print(f"  z axis limit reached! going back up (post-IK tip_z={tip_after[2]:.3f})")
                rejections["floor_guard"] += 1
                time.sleep(pause)
                continue

            # Diagnostic: every 30 iters, print what IK is actually producing.
            # If max|delta| is ~0, IK is returning q_now (stuck — unreachable
            # target from this configuration). If delta is non-trivial but the
            # robot doesn't move, the motors aren't executing the writes.
            if iteration % 30 == 0:
                delta = q_step - q_now
                step_vec = next_tip - tip_now
                print(f"    [diag] step_vec={np.round(step_vec, 4)}  "
                      f"|step|={np.linalg.norm(step_vec)*100:.2f}cm  "
                      f"max|Δq|={np.abs(delta).max():.3f}°  "
                      f"posture_α={posture_alpha:.2f}  "
                      f"q_now={np.round(q_now, 1)}  "
                      f"Δq={np.round(delta, 2)}")

            send_joints(robot, q_step, gripper_val)
            last_sent_q = q_step.copy()
            rejections["moves_sent"] += 1
            time.sleep(pause)
    except KeyboardInterrupt:
        print("\n  MPC tracking stopped by user.")
    finally:
        if own_robot:
            try:
                _hold_until_user_releases(robot)
            finally:
                robot.disconnect()


def set_gripper(robot, value: float):
    """Send a gripper-only command, leaving every other joint at its current target."""
    obs = robot.get_observation()
    action = dict(obs)
    action["gripper.pos"] = float(value)
    robot.send_action(action)


def _hold_until_user_releases(robot):
    """Keep torque on so the arm stays where it is, then wait for the user to
    physically support it before disabling torque. Without this, robot.disconnect()
    cuts torque and the arm collapses under gravity ("the slam")."""
    try:
        print("\n  Arm is holding position under torque.")
        print("  >>> SUPPORT THE ARM with your hand, then press Enter to release torque <<<")
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def hover(goal_voxmap: np.ndarray, calibration_path: str = "camera_to_robot.npz"):
    """
    Move to LIFT_OFFSET_M above the goal position and hold.
    goal_voxmap: np.ndarray shape (3,) in voxel-map frame (Y-up, metres).
    """
    R, t = load_calibration(calibration_path)
    kin  = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link",
                           joint_names=JOINT_NAMES[:-1])
    robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
    robot.connect()

    try:
        hover_pos = voxmap_to_robot(goal_voxmap, R, t, lift=LIFT_OFFSET_M)
        print(f"Moving to hover  robot={np.round(hover_pos,3)} (gripper closed during travel)")
        move_to(robot, kin, hover_pos, gripper_val=GRIPPER_CLOSED)
        print("Hover complete.")

        print(f"Opening gripper (value={GRIPPER_OPEN}).")
        set_gripper(robot, GRIPPER_OPEN)
        time.sleep(0.5)

        _hold_until_user_releases(robot)
    finally:
        robot.disconnect()


def execute_path(waypoints: list, gripper_close_at_goal: bool = False,
                 calibration_path: str = "camera_to_robot.npz"):
    """
    Move directly to hover above the final waypoint (goal).
    Intermediate A* waypoints are ignored — IK per-waypoint is too unstable.
    If gripper_close_at_goal, descend and grasp after hovering.
    """
    goal_voxmap = np.asarray(waypoints[-1])

    if not gripper_close_at_goal:
        hover(goal_voxmap, calibration_path)
        return

    R, t = load_calibration(calibration_path)
    kin  = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link",
                           joint_names=JOINT_NAMES[:-1])
    robot = make_robot_from_config(SOFollowerRobotConfig(port=PORT))
    robot.connect()

    try:
        hover_pos = voxmap_to_robot(goal_voxmap, R, t, lift=LIFT_OFFSET_M)
        grasp_pos = voxmap_to_robot(goal_voxmap, R, t, lift=0.0)

        print("Moving to hover...")
        q_hover = move_to(robot, kin, hover_pos)
        if q_hover is None:
            return

        print("Descending to grasp...")
        move_to(robot, kin, grasp_pos, step_size=0.01)

        print("Closing gripper...")
        obs_raw = robot.get_observation()
        action  = dict(obs_raw)
        action["gripper.pos"] = GRIPPER_CLOSED
        robot.send_action(action)
        time.sleep(1.0)
        print("Done.")
        _hold_until_user_releases(robot)
    finally:
        robot.disconnect()
