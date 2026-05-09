"""
End-to-end VoxPoser pipeline:
  capture → voxel map → YOLO → VLM → value map → A* path → visualise
"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from cv2 import aruco

from camera import RealSenseCamera
from voxel_map import VoxelMap
from vlm_query import query_vlm, position_to_value_map
from path_planner import plan_path
from executor import (
    execute_path,
    track_target,
    mpc_track,
    open_and_home,
    voxmap_to_robot,
    load_calibration,
    LIFT_OFFSET_M,
    HOME_POSE_ROBOT,
    _hold_until_user_releases,
)

ARUCO_MARKER_ID   = 0
ARUCO_MARKER_SIZE = 0.04   # metres
ARUCO_DICT        = aruco.DICT_4X4_250

_h = ARUCO_MARKER_SIZE / 2
_ARUCO_OBJ_PTS = np.array([
    [-_h,  _h, 0],
    [ _h,  _h, 0],
    [ _h, -_h, 0],
    [-_h, -_h, 0],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Gripper localisation via ArUco
# ---------------------------------------------------------------------------

def detect_gripper_position(cam, n_samples: int = 10):
    """
    Detect the ArUco marker on the gripper using an already-open RealSense.
    Returns the marker's 3D position in voxel-map frame (X-right, Y-up,
    Z-forward), metres, or None if not found.
    """
    intr = cam.get_intrinsics()
    K    = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1       ],
    ], dtype=np.float64)
    dist = np.zeros(5)

    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 23
    params.adaptiveThreshWinSizeStep = 2
    params.minMarkerPerimeterRate    = 0.02
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(ARUCO_DICT), params)

    samples = []
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 10:
        attempts += 1
        color, _, _ = cam.get_frames()
        if color is None:
            continue
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or ARUCO_MARKER_ID not in ids.flatten():
            continue
        idx    = list(ids.flatten()).index(ARUCO_MARKER_ID)
        pts_2d = corners[idx][0].astype(np.float32)
        _, rvec, tvec = cv2.solvePnP(
            _ARUCO_OBJ_PTS, pts_2d, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        samples.append(tvec.flatten())

    if not samples:
        return None

    pos = np.mean(samples, axis=0)   # camera frame: X-right, Y-down, Z-forward
    pos[1] *= -1                     # → voxel-map frame: Y-up
    return pos


# ---------------------------------------------------------------------------
# Scene capture
# ---------------------------------------------------------------------------

def capture_scene(cam):
    """Capture one aligned frame + full coloured point cloud using an
    already-open RealSense."""
    pc_gen = rs.pointcloud()

    print("  warming up camera...")
    for _ in range(15):
        cam.pipeline.wait_for_frames()

    color_frame, depth_frame = cam.get_aligned_frames()
    intrinsics = cam.get_intrinsics()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    pc_gen.map_to(color_frame)
    pts  = pc_gen.calculate(depth_frame)
    vtx  = np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)
    tex  = np.asanyarray(pts.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

    h, w = color_image.shape[:2]
    u    = np.clip((tex[:, 0] * w).astype(int), 0, w - 1)
    v    = np.clip((tex[:, 1] * h).astype(int), 0, h - 1)
    rgb  = color_image[v, u][:, ::-1] / 255.0   # BGR → RGB float

    valid = vtx[:, 2] > 0
    vtx, rgb = vtx[valid], rgb[valid]
    vtx[:, 1] *= -1   # camera Y-down → Y-up (VoxelMap convention)

    return color_image, depth_frame, depth_image, intrinsics, vtx, rgb


# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------

def detect_objects(color_image, depth_frame, intrinsics):
    """Return list of {label, bbox, position_3d} for every detected object."""
    from ultralytics import YOLO

    model = YOLO("yolov8s-world.pt")
    model.set_classes(["cup", "bottle", "box", "bowl", "mug"])
    results = model.predict(color_image, verbose=False)

    detected = []
    for box in results[0].boxes:
        u1, v1, u2, v2 = map(int, box.xyxy[0])
        cx, cy = (u1 + u2) // 2, (v1 + v2) // 2
        depth  = depth_frame.get_distance(cx, cy)
        if depth <= 0:
            continue
        xyz   = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
        label = model.names[int(box.cls[0])]
        detected.append({"label": label, "bbox": [u1, v1, u2, v2], "position_3d": tuple(xyz)})

    return detected


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def build_vis(voxel_map, waypoints, start_world, goal_world):
    """Assemble Open3D geometries: value-map voxels + A* path + markers."""
    geometries = []

    # Voxel map coloured by real scene RGB
    geometries.append(voxel_map.to_open3d(color_by="rgb"))

    # A* path as a red line set
    if waypoints and len(waypoints) > 1:
        pts   = np.array(waypoints)
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        ls    = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([1, 0, 0])
        geometries.append(ls)

    # Start (blue) and goal (green) markers
    for pos, colour in [(start_world, [0, 0, 1]), (goal_world, [0, 1, 0])]:
        marker = o3d.geometry.PointCloud()
        marker.points = o3d.utility.Vector3dVector([pos])
        marker.paint_uniform_color(colour)
        geometries.append(marker)

    return geometries


def visualize_plan(color_image, detected, target_label, vmap, waypoints,
                   start_world, goal_world):
    """
    Show the planned path before execution. Two windows:
      1) RGB camera with YOLO bboxes (target green, others gray)
      2) Open3D voxel map + A* waypoints + start (blue) / goal (green)
    Each blocks until closed; close both to start the arm.
    """
    annotated = color_image.copy()
    for obj in detected:
        x1, y1, x2, y2 = obj["bbox"]
        is_target = (obj["label"] == target_label)
        colour = (0, 255, 0) if is_target else (160, 160, 160)
        thick  = 3 if is_target else 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thick)
        cv2.putText(annotated, obj["label"], (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick)
    cv2.putText(annotated, "press any key to continue",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("plan: RGB + detections", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    geoms = build_vis(vmap, waypoints, start_world, goal_world)
    print("      Close the 3D viewer window to start the arm.")
    o3d.visualization.draw_geometries(
        geoms, window_name="plan: voxel map + A* path",
    )


def make_scene_provider(cam, yolo_model, target_label, R_cal, t_cal,
                        initial_camera_xyz,
                        target_smoothing: float = 0.3,
                        obstacle_smoothing: float = 0.5,
                        repulsor_min_dist: float = 0.08,
                        conf_threshold: float = 0.4):
    """
    Build a closure that, on every call, returns the current scene as
    VoxPoser-style structured perception:

        {"target":         robot-frame xyz of the chosen target detection
                           (LIFT_OFFSET_M added — we hover, not grasp),
         "obstacles":      [robot-frame xyz of every other detection],
         "obstacle_radii": [approximate metres per obstacle, from
                            bbox_width_px * depth / fx]}

    Returns None on a missed frame so the controller can hold position.

    Smoothing & filtering applied in-line (matches what VoxPoser does in
    its perception stack so the value field is stable frame-to-frame):
      • Confidence threshold drops low-quality boxes.
      • EMA on the target position (alpha=target_smoothing).
      • EMA on obstacle positions matched to last frame by nearest-neighbour
        within 10 cm (alpha=obstacle_smoothing).
      • Repulsors closer than `repulsor_min_dist` to the chosen target are
        dropped — these are typically the gripper itself or duplicate
        detections of the target's edge.
      • Target identity is locked by nearest-to-last-camera-xyz so a second
        cup walking into frame doesn't swap targets.
    """
    last_camera_xyz   = np.array(initial_camera_xyz, dtype=float)
    target_smoothed: np.ndarray | None = None
    last_obstacles: list[tuple[np.ndarray, float]] = []

    def scene_provider():
        nonlocal last_camera_xyz, target_smoothed, last_obstacles
        color_image, _, depth_frame = cam.get_frames()
        if color_image is None or depth_frame is None:
            return None

        results = yolo_model.predict(color_image, verbose=False)
        intr = cam.get_intrinsics()

        target_candidates = []   # list of (xyz_cam, xyz_robot)
        raw_obstacles     = []
        raw_radii         = []

        for box in results[0].boxes:
            conf = float(box.conf[0]) if box.conf is not None else 1.0
            if conf < conf_threshold:
                continue
            label = yolo_model.names[int(box.cls[0])]
            u1, v1, u2, v2 = map(int, box.xyxy[0])
            cx, cy = (u1 + u2) // 2, (v1 + v2) // 2
            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue

            xyz_cam = np.array(
                rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth),
                dtype=float,
            )
            p_vox = xyz_cam.copy()
            p_vox[1] *= -1   # camera Y-down → voxel-map Y-up
            p_robot = voxmap_to_robot(p_vox, R_cal, t_cal, lift=0.0)

            half_width_px = max(1.0, (u2 - u1) / 2.0)
            radius_m      = half_width_px * depth / max(intr.fx, 1.0)

            if label == target_label:
                target_candidates.append((xyz_cam, p_robot))
            else:
                raw_obstacles.append(p_robot)
                raw_radii.append(radius_m)

        if not target_candidates:
            return None

        chosen_cam, chosen_robot = min(
            target_candidates,
            key=lambda pair: np.linalg.norm(pair[0] - last_camera_xyz),
        )
        last_camera_xyz = chosen_cam

        target_lifted = chosen_robot.copy()
        target_lifted[2] += LIFT_OFFSET_M

        if target_smoothed is None:
            target_smoothed = target_lifted
        else:
            target_smoothed = (
                target_smoothing * target_lifted
                + (1.0 - target_smoothing) * target_smoothed
            )

        # Drop repulsors that overlap the target (gripper, duplicate edges).
        kept_obs   = []
        kept_radii = []
        for obs, r in zip(raw_obstacles, raw_radii):
            if np.linalg.norm(obs - chosen_robot) >= repulsor_min_dist:
                kept_obs.append(obs)
                kept_radii.append(r)

        # Obstacle EMA via greedy nearest-neighbour matching to last frame.
        smoothed_obs   = []
        smoothed_radii = []
        prev_used = [False] * len(last_obstacles)
        for obs, r in zip(kept_obs, kept_radii):
            best_i = -1
            best_d = float("inf")
            for j, (prev_obs, _) in enumerate(last_obstacles):
                if prev_used[j]:
                    continue
                d = float(np.linalg.norm(obs - prev_obs))
                if d < best_d:
                    best_d, best_i = d, j
            if best_i >= 0 and best_d < 0.10:
                prev_obs, _ = last_obstacles[best_i]
                obs = (
                    obstacle_smoothing * obs
                    + (1.0 - obstacle_smoothing) * prev_obs
                )
                prev_used[best_i] = True
            smoothed_obs.append(obs)
            smoothed_radii.append(r)
        last_obstacles = list(zip(smoothed_obs, smoothed_radii))

        return {
            "target":         target_smoothed.copy(),
            "obstacles":      smoothed_obs,
            "obstacle_radii": smoothed_radii,
        }

    return scene_provider


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(command: str):
    cam = RealSenseCamera()
    try:
        return _run_with_camera(command, cam)
    finally:
        cam.stop()


def _run_with_camera(command: str, cam):
    print(f"\n[0/7] Moving to home pose {np.round(HOME_POSE_ROBOT, 3)}...")
    robot, kin = open_and_home()

    try:
        print("\n[1/7] Localising gripper via ArUco...")
        gripper_pos = detect_gripper_position(cam)
        if gripper_pos is not None:
            print(f"      Gripper @ X={gripper_pos[0]:+.3f}  Y={gripper_pos[1]:+.3f}  Z={gripper_pos[2]:+.3f} m")
        else:
            print("      Marker not found — using fallback HOME_POSITION.")
            gripper_pos = np.array([0.0, -0.2, 0.4])

        print("\n[2/7] Capturing scene...")
        color_image, depth_frame, depth_image, intrinsics, xyz, rgb = capture_scene(cam)

        print("[3/7] Building voxel map...")
        vmap = VoxelMap(resolution=0.01)
        vmap.update(xyz, rgb)
        vmap.finalize()
        print(f"      {vmap.summary()}")

        print("[4/7] Detecting objects (YOLO)...")
        detected = detect_objects(color_image, depth_frame, intrinsics)
        if not detected:
            print("      No objects found — try adjusting YOLO classes or camera position.")
            return
        for obj in detected:
            x, y, z = obj["position_3d"]
            print(f"      {obj['label']:10s} → X={x:.2f}m  Y={y:.2f}m  Z={z:.2f}m")

        print(f"[5/7] Querying VLM: '{command}'")
        vlm_result   = query_vlm(command, detected)
        target_label = vlm_result["target_label"]
        print(f"      Target: {target_label}  |  {vlm_result.get('reasoning', '')}")

        if target_label is None:
            print("      VLM couldn't match the command to any detected object.")
            return

        target_obj = next((o for o in detected if o["label"] == target_label), None)
        if target_obj is None:
            print(f"      Label '{target_label}' not found in detections.")
            return

        # Convert camera frame (Y-down) → voxel map frame (Y-up)
        goal_3d    = np.array(target_obj["position_3d"])
        goal_3d[1] *= -1

        print("[6/7] Building value map & planning path (A*)...")
        value_grid = position_to_value_map(goal_3d, vmap)
        vmap.set_value_map(value_grid)

        waypoints = plan_path(vmap, gripper_pos, goal_3d)
        if waypoints:
            print(f"      Path found: {len(waypoints)} waypoints")
        else:
            print("      No free path found — falling back to straight line.")
            waypoints = [gripper_pos, goal_3d]

        print(f"      Goal → X={goal_3d[0]:.2f}m  Y={goal_3d[1]:.2f}m  Z={goal_3d[2]:.2f}m")

        print(f"[7/7] MPC-tracking '{target_label}' (Ctrl+C to stop)...")
        visualize_plan(color_image, detected, target_label, vmap, waypoints,
                       gripper_pos, goal_3d)

        from ultralytics import YOLO
        yolo_model = YOLO("yolov8s-world.pt")
        yolo_model.set_classes(["cup", "bottle", "box", "bowl", "mug"])

        R_cal, t_cal = load_calibration("camera_to_robot.npz")

        initial_camera_xyz = np.array(target_obj["position_3d"], dtype=float)

        # Closed-loop VoxPoser-style controller: scene_provider feeds target +
        # obstacles every few frames, mpc_track rebuilds the value field and
        # picks one Cartesian step per iteration via random shooting.
        scene_provider = make_scene_provider(cam, yolo_model, target_label,
                                             R_cal, t_cal, initial_camera_xyz)
        mpc_track(scene_provider, robot=robot, kin=kin)

        return waypoints
    finally:
        # Always release the arm cleanly, whether MPC ran, errored, or we
        # bailed early on a detection miss. Keeps the SO-101 from slamming.
        try:
            _hold_until_user_releases(robot)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Command: ").strip()
    run(cmd)
