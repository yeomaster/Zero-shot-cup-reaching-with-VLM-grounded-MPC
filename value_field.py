"""
VoxPoser-style 3D value field.

VoxPoser composes a scalar reward field over the workspace from
language-generated voxel maps (affordance, avoidance, velocity, rotation,
gripper). For our SO-ARM101 + RealSense pipeline we use a simpler analytic
form of the same idea:

    V(p) = -sum( w_i * |p - target_i| )                # attractors (linear pull)
           - sum( w_j * exp(-|p - obs_j|^2 / 2 sigma^2) )  # repulsors (Gaussian bump)
           - floor_barrier(p_z)                        # smooth ramp near table
           - workspace_box(p)                          # hard penalty outside reach

Higher V = better. The MPC planner picks rollouts that maximise the
integral of V along the trajectory.

We trade VoxPoser's discretised voxel grid for analytic sums of attractors
and repulsors so we can score thousands of candidate waypoints per second
on CPU. The composability story is identical — each detected obstacle adds
one term, each goal adds one term — so when we wire the VLM up later it
can author the same kind of structured scene description VoxPoser uses.
"""

from __future__ import annotations

import numpy as np


class ValueField:
    def __init__(self, floor_z: float = 0.05, floor_buffer: float = 0.02):
        self._attractors: list[tuple[np.ndarray, float]] = []
        self._repulsors:  list[tuple[np.ndarray, float, float]] = []
        self._floor_z      = float(floor_z)
        self._floor_buffer = float(floor_buffer)
        self._workspace: tuple[np.ndarray, np.ndarray] | None = None

    def add_attractor(self, center, weight: float = 1.0) -> None:
        self._attractors.append(
            (np.asarray(center, dtype=float), float(weight))
        )

    def add_repulsor(self, center, sigma: float = 0.05, weight: float = 1.0) -> None:
        # sigma is roughly the obstacle's radius — repulsion fades to ~0
        # at distances >> sigma.
        sigma = float(np.clip(sigma, 0.04, 0.12))
        self._repulsors.append(
            (np.asarray(center, dtype=float), sigma, float(weight))
        )

    def set_workspace(self, lo, hi) -> None:
        self._workspace = (
            np.asarray(lo, dtype=float),
            np.asarray(hi, dtype=float),
        )

    def evaluate(self, pts: np.ndarray) -> np.ndarray:
        """pts: (..., 3) → (...,) reward. Higher = better."""
        pts = np.asarray(pts, dtype=float)
        flat = pts.reshape(-1, 3)
        v = np.zeros(len(flat))

        for c, w in self._attractors:
            v -= w * np.linalg.norm(flat - c, axis=1)

        for c, sigma, w in self._repulsors:
            d2 = np.sum((flat - c) ** 2, axis=1)
            v -= w * np.exp(-d2 / (2.0 * sigma * sigma))

        # Floor barrier — penalty (positive number subtracted from V) that
        # ramps up as Z approaches FLOOR_Z and explodes below it. The ramp
        # weight has to stay comparable to attractor strength: attractor
        # gradient is ~1 per metre, so a 5/buffer = 250/m floor gradient
        # would steamroll the goal pull and lock the MPC into pure +Z near
        # the floor. We keep the hard sub-floor penalty huge so the planner
        # still refuses to dive below the table.
        floor_dist = flat[:, 2] - self._floor_z
        ramp = 0.3 * np.clip(
            (self._floor_buffer - floor_dist) / self._floor_buffer, 0.0, 1.0
        )
        below = np.maximum(0.0, -floor_dist)
        v -= ramp + 200.0 * below

        if self._workspace is not None:
            lo, hi = self._workspace
            outside = np.any((flat < lo) | (flat > hi), axis=1)
            v -= np.where(outside, 100.0, 0.0)

        return v.reshape(pts.shape[:-1])
