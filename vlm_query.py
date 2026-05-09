import os
import json
import re
import time
import numpy as np
import cv2
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = """You are a robot manipulation assistant.
You will be given a list of objects detected in the scene with their 3D positions (in metres),
and a manipulation command. Select the correct target object.

Respond ONLY with valid JSON in this exact format — no markdown, no explanation:
{
  "target_label": "<label of the target object>",
  "reasoning": "<one sentence>"
}

If no detected object matches the command, return "target_label": null."""


def query_vlm(command: str, detected_objects: list[dict]) -> dict:
    """
    Ask the VLM to select a target object from a list of YOLO detections.

    detected_objects: list of dicts with keys:
        "label"       : str   — YOLO class name
        "position_3d" : tuple — (x, y, z) in metres, camera frame

    Returns dict: {target_label, reasoning}
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Export it before running:\n"
            "  export GROQ_API_KEY=your_key_here"
        )

    if not detected_objects:
        return {"target_label": None, "reasoning": "No objects detected in scene."}

    scene_lines = [
        f"- {obj['label']} at X={obj['position_3d'][0]:.2f}m "
        f"Y={obj['position_3d'][1]:.2f}m Z={obj['position_3d'][2]:.2f}m"
        for obj in detected_objects
    ]
    user_message = f"Command: {command}\n\nDetected objects:\n" + "\n".join(scene_lines)

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )

    text = response.choices[0].message.content.strip()
    text = re.sub(r"```(?:json)?\n?", "", text).strip("` \n")
    return json.loads(text)


def position_to_value_map(
    position_3d: np.ndarray,
    voxel_map,
    sigma: float = 0.05,
) -> np.ndarray:
    """
    Build a value map over all occupied voxels centred on a 3D target position.

    position_3d : (3,) array, metres
    sigma       : controls the spread — larger = broader high-value region (default 5 cm)
    Returns float array of shape voxel_map.shape, values in [0, 1].

    ---------------------------------------------------------------------------
    SWAP POINT — Scoring function
    ---------------------------------------------------------------------------
    Currently: Gaussian falloff  score = exp(-0.5 * (dist / sigma)^2)
      - Voxels right on the target score ~1.0, drops smoothly to ~0 further away.
      - sigma controls the width: 0.05 = tight 5 cm blob, 0.15 = wide 15 cm cloud.

    To replace with your own scoring:
      - scores = 1.0 - (dists / dists.max())          # linear falloff
      - scores = (dists < threshold).astype(float)     # hard cutoff
      - scores = vlm_generated_array                   # VLM assigns per-voxel scores directly
      - scores = some_other_function(centers, goal)    # anything that returns shape (N,)
    The only requirement: scores must be the same length as `centers` (one per occupied voxel).
    ---------------------------------------------------------------------------
    """
    value_grid = np.zeros(voxel_map.shape, dtype=float)
    centers, idx = voxel_map.occupied_centers()
    if len(centers) == 0:
        return value_grid

    # Distance from every occupied voxel centre to the target position
    dists = np.linalg.norm(centers - np.asarray(position_3d), axis=1)

    # Gaussian score: 1.0 at the target, falls off with distance
    scores = np.exp(-0.5 * (dists / sigma) ** 2)

    value_grid[idx[:, 0], idx[:, 1], idx[:, 2]] = scores
    return value_grid


# ---------------------------------------------------------------------------
# Standalone test — detect objects with YOLO, query VLM, show result
# ---------------------------------------------------------------------------

def _detect_objects(color_image, depth_frame, intrinsics):
    """Run YOLO-World on the frame and return detected objects with 3D positions."""
    import pyrealsense2 as rs
    from ultralytics import YOLO

    model = YOLO("yolov8s-world.pt")
    model.set_classes(["cup", "bottle", "box", "bowl", "mug"])

    results = model.predict(color_image, verbose=False)
    detected = []

    for box in results[0].boxes:
        u1, v1, u2, v2 = map(int, box.xyxy[0])
        cx, cy = (u1 + u2) // 2, (v1 + v2) // 2
        depth = depth_frame.get_distance(cx, cy)
        if depth <= 0:
            continue

        xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
        label = model.names[int(box.cls[0])]
        detected.append({
            "label": label,
            "bbox": [u1, v1, u2, v2],
            "position_3d": tuple(xyz),
        })

        cv2.rectangle(color_image, (u1, v1), (u2, v2), (200, 200, 200), 1)
        cv2.putText(color_image, label, (u1, v1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return detected, color_image


def _capture_frame():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)

    align = rs.align(rs.stream.color)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = depth_profile.get_intrinsics()

    time.sleep(1.0)
    frames = None
    for _ in range(30):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            break
        except RuntimeError:
            time.sleep(0.5)

    if frames is None:
        pipeline.stop()
        raise RuntimeError("Camera didn't produce frames after 15s.")

    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    pipeline.stop()

    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_frame, intrinsics


if __name__ == "__main__":
    command = input("Enter command (e.g. 'pick up the cup'): ").strip()

    print("Capturing frame...")
    color_image, depth_frame, intrinsics = _capture_frame()

    print("Running YOLO detection...")
    detected, annotated = _detect_objects(color_image, depth_frame, intrinsics)

    if not detected:
        print("No objects detected. Try adjusting YOLO classes or camera position.")
    else:
        print(f"Detected {len(detected)} object(s):")
        for obj in detected:
            x, y, z = obj["position_3d"]
            print(f"  {obj['label']:10s} → X={x:.2f}m Y={y:.2f}m Z={z:.2f}m")

        print(f"\nQuerying VLM: '{command}'")
        result = query_vlm(command, detected)

        print("\n--- VLM response ---")
        print(f"  Target   : {result['target_label']}")
        print(f"  Reasoning: {result.get('reasoning', '')}")

        # Highlight the chosen object in green
        if result["target_label"]:
            for obj in detected:
                if obj["label"] == result["target_label"]:
                    x1, y1, x2, y2 = obj["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"TARGET: {obj['label']}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    break

    cv2.imshow("VLM result", annotated)
    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
