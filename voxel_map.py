import numpy as np
import open3d as o3d


class VoxelMap:
    """
    Discretizes a 3D point cloud into a voxel grid for VoxPoser-style reasoning.

    Coordinate convention: Y-up, Z-forward (camera depth), X-right.
    All units in meters.
    """

    def __init__(
        self,
        resolution: float = 0.02,
        x_range: tuple = (-0.5, 0.5),
        y_range: tuple = (-0.5, 0.5),
        z_range: tuple = (0.1, 1.5),
    ):
        self.resolution = resolution
        self.bounds = np.array([x_range, y_range, z_range], dtype=float)  # (3, 2)

        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.shape = np.ceil(ranges / resolution).astype(int)  # (nx, ny, nz)

        self._reset()

    def _reset(self):
        self.occupied = np.zeros(self.shape, dtype=bool)
        self._color_accum = np.zeros((*self.shape, 3), dtype=float)
        self._counts = np.zeros(self.shape, dtype=int)
        self.colors = np.zeros((*self.shape, 3), dtype=float)   # averaged RGB [0,1]
        self.values = np.zeros(self.shape, dtype=float)         # VLM scores [0,1]

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        """(N,3) world coords → (N,3) integer voxel indices."""
        return ((points - self.bounds[:, 0]) / self.resolution).astype(int)

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        """(N,3) or (3,) voxel indices → voxel center in world coords."""
        return idx * self.resolution + self.bounds[:, 0] + self.resolution / 2.0

    def _in_bounds(self, idx: np.ndarray) -> np.ndarray:
        return np.all((idx >= 0) & (idx < self.shape), axis=1)

    # ------------------------------------------------------------------
    # Building the map
    # ------------------------------------------------------------------

    def voxels_from_detection(
        self,
        bbox: tuple,
        depth_image: np.ndarray,
        intrinsics,
        depth_scale: float = 0.001,
    ) -> np.ndarray:
        """
        Project a 2D YOLO bounding box into the voxel grid.

        bbox        : (u1, v1, u2, v2) pixel ints from YOLO
        depth_image : (H, W) uint16 raw depth from RealSense (z16 format)
        intrinsics  : rs2 intrinsics object (.fx .fy .ppx .ppy)
        depth_scale : metres per depth unit (RealSense default 0.001)

        Returns (M, 3) int array of occupied voxel indices that fall inside
        the 3-D projection of the bounding box.
        """
        u1, v1, u2, v2 = bbox
        h, w = depth_image.shape[:2]

        # Build pixel grid for the bounding box, clipped to image bounds
        us = np.arange(max(u1, 0), min(u2 + 1, w))
        vs = np.arange(max(v1, 0), min(v2 + 1, h))
        uu, vv = np.meshgrid(us, vs)
        uu, vv = uu.ravel(), vv.ravel()

        # Depth in metres; discard pixels with no depth reading
        z = depth_image[vv, uu] * depth_scale
        valid = z > 0
        uu, vv, z = uu[valid], vv[valid], z[valid]

        if len(z) == 0:
            return np.empty((0, 3), dtype=int)

        # Vectorised pinhole deprojection
        x = (uu - intrinsics.ppx) * z / intrinsics.fx
        y = (vv - intrinsics.ppy) * z / intrinsics.fy
        y *= -1  # camera Y-down → Y-up (matches VoxelMap convention)

        points_3d = np.stack([x, y, z], axis=1)  # (N, 3)

        # Map to voxel indices, keep only those inside the grid that are occupied
        idx = self.world_to_index(points_3d)
        idx = idx[self._in_bounds(idx)]
        if len(idx) == 0:
            return np.empty((0, 3), dtype=int)

        occupied_mask = self.occupied[idx[:, 0], idx[:, 1], idx[:, 2]]
        return np.unique(idx[occupied_mask], axis=0)

    def update(self, points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None):
        """
        Add a batch of points to the voxel grid.
        points_xyz : (N, 3) float, meters
        colors_rgb : (N, 3) float, [0, 1] — optional
        """
        idx = self.world_to_index(points_xyz)
        mask = self._in_bounds(idx)
        idx = idx[mask]

        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
        self.occupied[ix, iy, iz] = True
        np.add.at(self._counts, (ix, iy, iz), 1)

        if colors_rgb is not None:
            c = colors_rgb[mask]
            for ch in range(3):
                np.add.at(self._color_accum[..., ch], (ix, iy, iz), c[:, ch])

    def finalize(self):
        """Average accumulated colors. Call once after all update() calls."""
        filled = self._counts > 0
        self.colors[filled] = (
            self._color_accum[filled] / self._counts[filled, np.newaxis]
        )

    def set_value_map(self, value_grid: np.ndarray):
        """
        Assign VLM-generated scores to voxels.
        value_grid : array of shape self.shape, float [0, 1]
        """
        assert value_grid.shape == tuple(self.shape), (
            f"Expected shape {tuple(self.shape)}, got {value_grid.shape}"
        )
        self.values = value_grid.astype(float)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def occupied_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (world_centers, voxel_indices) for all occupied voxels."""
        idx = np.argwhere(self.occupied)          # (M, 3)
        return self.index_to_world(idx), idx

    def highest_value_center(self) -> np.ndarray | None:
        """World position of the voxel with the highest VLM value score."""
        masked = np.where(self.occupied, self.values, -np.inf)
        flat = np.argmax(masked)
        if masked.flat[flat] == -np.inf:
            return None
        idx = np.array(np.unravel_index(flat, self.shape))
        return self.index_to_world(idx)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_open3d(self, color_by: str = "rgb") -> o3d.geometry.PointCloud:
        """
        Build an Open3D PointCloud of voxel centers for visualization.
        Each dot = one occupied voxel.
        color_by : 'rgb'   — real scene colors
                   'value' — blue (low) → red (high) VLM heatmap
        """
        centers, idx = self.occupied_centers()
        if len(centers) == 0:
            return o3d.geometry.PointCloud()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)

        if color_by == "rgb":
            c = self.colors[idx[:, 0], idx[:, 1], idx[:, 2]]
        elif color_by == "value":
            v = self.values[idx[:, 0], idx[:, 1], idx[:, 2]]
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            c = np.stack([v, np.zeros_like(v), 1 - v], axis=1)  # red=high, blue=low
        else:
            raise ValueError(f"Unknown color_by: {color_by!r}")

        pcd.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1))
        return pcd

    def summary(self) -> str:
        n = int(self.occupied.sum())
        total = int(np.prod(self.shape))
        return (
            f"VoxelMap  resolution={self.resolution*100:.0f}cm  "
            f"grid={tuple(self.shape)}  "
            f"occupied={n}/{total} ({100*n/total:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Standalone demo — capture one frame and show the voxel map
# ---------------------------------------------------------------------------

def _capture_frame():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)

    align = rs.align(rs.stream.color)
    pc_gen = rs.pointcloud()

    # Warm-up: discard first few frames
    for _ in range(10):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    pipeline.stop()

    pc_gen.map_to(color_frame)
    points = pc_gen.calculate(depth_frame)

    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

    color_image = np.asanyarray(color_frame.get_data())
    h, w = color_image.shape[:2]
    u = np.clip((tex[:, 0] * w).astype(int), 0, w - 1)
    v = np.clip((tex[:, 1] * h).astype(int), 0, h - 1)
    colors_rgb = color_image[v, u][:, ::-1] / 255.0  # BGR→RGB, float

    valid = vtx[:, 2] > 0
    vtx, colors_rgb = vtx[valid], colors_rgb[valid]

    vtx[:, 1] *= -1  # Y-flip: camera Y-down → Y-up
    return vtx, colors_rgb


if __name__ == "__main__":
    print("Capturing frame from RealSense...")
    xyz, rgb = _capture_frame()
    print(f"  {len(xyz):,} valid points captured.")

    vmap = VoxelMap(resolution=0.02)
    vmap.update(xyz, rgb)
    vmap.finalize()
    print(vmap.summary())

    pcd = vmap.to_open3d(color_by="rgb")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Voxel Map — each dot = one 2cm voxel",
        width=1280,
        height=720,
        point_show_normal=False,
    )
