import heapq
import numpy as np

# 26-connected neighbours
_DELTAS = np.array(
    [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if dx or dy or dz],
    dtype=int,
)
_STEP_COSTS = np.linalg.norm(_DELTAS.astype(float), axis=1)

# Soft-cost weights
VALUE_WEIGHT     = 5.0   # how strongly the VLM value map repels/attracts
OBSTACLE_PENALTY = 50.0  # cost added for stepping into an occupied voxel (not goal)
HEURISTIC_WEIGHT = 1.0   # 1.0 = admissible A*, >1 = greedy/faster but suboptimal


def _line_of_sight(voxel_map, a_idx, b_idx) -> bool:
    """3D Bresenham — true if every voxel on the segment from a to b is passable."""
    a = np.array(a_idx, dtype=int)
    b = np.array(b_idx, dtype=int)
    d = np.abs(b - a)
    s = np.sign(b - a).astype(int)
    cur = a.copy()

    # Driving axis = longest delta
    ax = int(np.argmax(d))
    other = [i for i in range(3) if i != ax]
    err1 = 2 * d[other[0]] - d[ax]
    err2 = 2 * d[other[1]] - d[ax]

    for _ in range(int(d[ax])):
        idx = tuple(cur)
        if idx != tuple(b_idx) and voxel_map.occupied[idx]:
            return False
        if err1 > 0:
            cur[other[0]] += s[other[0]]
            err1 -= 2 * d[ax]
        if err2 > 0:
            cur[other[1]] += s[other[1]]
            err2 -= 2 * d[ax]
        err1 += 2 * d[other[0]]
        err2 += 2 * d[other[1]]
        cur[ax] += s[ax]

    return True


def _shortcut(voxel_map, idx_path: list[tuple]) -> list[tuple]:
    """Greedy line-of-sight shortcutting: skip waypoints whenever a straight
    voxel line between non-adjacent waypoints stays in free space."""
    if len(idx_path) <= 2:
        return idx_path
    out = [idx_path[0]]
    i = 0
    while i < len(idx_path) - 1:
        # Find the farthest j such that out[-1] -> idx_path[j] is clear
        j = len(idx_path) - 1
        while j > i + 1 and not _line_of_sight(voxel_map, out[-1], idx_path[j]):
            j -= 1
        out.append(idx_path[j])
        i = j
    return out


def plan_path(
    voxel_map,
    start_world: np.ndarray,
    goal_world: np.ndarray,
) -> list[np.ndarray] | None:
    """
    Soft-cost A* through the voxel grid using the VLM value map.

    Cost per step = step length
                  + VALUE_WEIGHT * (1 - value[neighbour])    # prefer high-value voxels
                  + OBSTACLE_PENALTY if neighbour is occupied (and not the goal)

    The returned path is shortcut via line-of-sight to remove jagged voxel-grid steps.
    Returns ordered list of (3,) world-coord waypoints in metres, or None if unreachable.
    """
    start_idx = tuple(voxel_map.world_to_index(start_world.reshape(1, 3))[0])
    goal_idx  = tuple(voxel_map.world_to_index(goal_world.reshape(1, 3))[0])
    shape     = tuple(voxel_map.shape)

    def in_bounds(idx):
        return all(0 <= idx[i] < shape[i] for i in range(3))

    if not in_bounds(start_idx) or not in_bounds(goal_idx):
        return None

    values   = voxel_map.values
    occupied = voxel_map.occupied
    v_max    = float(values.max()) if values.size else 0.0
    v_norm   = values / v_max if v_max > 0 else values  # keep in [0,1] for cost shaping

    open_heap = [(0.0, start_idx)]
    came_from: dict = {}
    g: dict = {start_idx: 0.0}

    goal_arr = np.array(goal_idx)

    while open_heap:
        _, cur = heapq.heappop(open_heap)

        if cur == goal_idx:
            # Reconstruct index path
            idx_path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                idx_path.append(cur)
            idx_path.reverse()

            # Smooth via line-of-sight shortcutting
            smoothed = _shortcut(voxel_map, idx_path)
            return [voxel_map.index_to_world(np.array(i)) for i in smoothed]

        cur_arr = np.array(cur)
        for delta, step in zip(_DELTAS, _STEP_COSTS):
            nb = tuple(cur_arr + delta)
            if not in_bounds(nb):
                continue

            # Soft obstacle: occupied voxels are expensive but not impossible
            # (the goal voxel itself is the target object, so it's free).
            obstacle_cost = 0.0
            if occupied[nb] and nb != goal_idx:
                obstacle_cost = OBSTACLE_PENALTY

            value_cost = VALUE_WEIGHT * (1.0 - float(v_norm[nb]))
            tg = g[cur] + step + value_cost + obstacle_cost

            if tg < g.get(nb, float("inf")):
                came_from[nb] = cur
                g[nb] = tg
                h = HEURISTIC_WEIGHT * float(np.linalg.norm(np.array(nb) - goal_arr))
                heapq.heappush(open_heap, (tg + h, nb))

    return None
