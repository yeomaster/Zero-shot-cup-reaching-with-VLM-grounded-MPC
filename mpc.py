"""
Random-shooting model-predictive controller over a ValueField.

This is the controller VoxPoser uses on top of its language-generated voxel
maps: at every timestep, sample a batch of candidate trajectories, score
each one against the current value field, and execute the first step of the
best candidate. The field is rebuilt from the latest perception every step,
so the controller is closed-loop on both the goal *and* the obstacles.

Differences from the paper:
  - VoxPoser uses model-predictive path integral (MPPI) on a discretised
    voxel grid. We use uniform random shooting on a continuous analytic
    field (`value_field.py`). Cheaper and adequate for SO-ARM101's small
    workspace; can swap in MPPI later without changing the interface.
  - We bias toward continuing the previous direction with a cosine-similarity
    bonus. VoxPoser gets temporal smoothness implicitly from MPPI's softmax
    update; we get it explicitly here. Both serve the same purpose:
    suppress jitter from frame-to-frame perception noise.
"""

from __future__ import annotations

import numpy as np


class MPCPlanner:
    def __init__(
        self,
        step_size: float = 0.01,
        horizon: int = 6,
        n_candidates: int = 64,
        smoothness_weight: float = 0.1,
        rng_seed: int = 0,
    ):
        self.step_size = float(step_size)
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.smoothness_weight = float(smoothness_weight)
        self._rng = np.random.default_rng(rng_seed)
        self._prev_dir: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_dir = None

    def _sample_directions(self, goal_hint: np.ndarray | None = None) -> np.ndarray:
        u = self._rng.standard_normal((self.n_candidates, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        extras = [np.zeros((1, 3))]
        if self._prev_dir is not None:
            extras.append(self._prev_dir.reshape(1, 3))
        if goal_hint is not None:
            g = np.asarray(goal_hint, dtype=float)
            n = float(np.linalg.norm(g))
            if n > 1e-9:
                extras.append((g / n).reshape(1, 3))
        return np.vstack([u] + extras)

    def step(self, pos_now: np.ndarray, value_field,
             goal_hint: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Return (next_pos, best_score). `next_pos` is one `step_size`
        Cartesian move along the highest-scoring sampled trajectory.

        `goal_hint` is an optional vector pointing roughly toward the goal
        (e.g. `target - pos_now`). When provided, the unit vector along it
        is included as a guaranteed candidate so random-shooting cannot
        miss the gradient direction by a few degrees and let smoothness
        lock in a wrong heading."""
        pos_now = np.asarray(pos_now, dtype=float)
        directions = self._sample_directions(goal_hint)        # (K, 3)
        K = len(directions)
        H = self.horizon

        steps = np.arange(1, H + 1, dtype=float)[None, :, None]  # (1, H, 1)
        offsets = steps * self.step_size * directions[:, None, :]  # (K, H, 3)
        rollouts = pos_now[None, None, :] + offsets               # (K, H, 3)

        v = value_field.evaluate(rollouts.reshape(-1, 3)).reshape(K, H)
        scores = v.sum(axis=1)

        if self._prev_dir is not None:
            scores += self.smoothness_weight * (directions @ self._prev_dir)

        best = int(np.argmax(scores))
        best_dir = directions[best]
        next_pos = pos_now + self.step_size * best_dir

        if np.linalg.norm(best_dir) > 1e-9:
            self._prev_dir = best_dir
        return next_pos, float(scores[best])
