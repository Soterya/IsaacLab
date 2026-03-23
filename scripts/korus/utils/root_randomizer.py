#!/usr/bin/env python3
"""
Root randomizer for Korus humanoid reset.

Requirements:
- Translation noise only in X,Y
- Z is constant (default 4.0)
- Rotation noise only in yaw (about +Z), no roll/pitch
- Tunable std + clip
- Deterministic with seed

Outputs:
- pos_env: (B,3) float32  [x,y,z] in ENV frame (caller adds scene.env_origins)
- quat_wxyz: (B,4) float32 quaternion in (w,x,y,z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np


# ------------------------
# Quaternion utilities
# ------------------------

def yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    """Yaw-only quaternion in wxyz."""
    half = 0.5 * float(yaw_rad)
    w = math.cos(half)
    z = math.sin(half)
    return np.array([w, 0.0, 0.0, z], dtype=np.float32)


@dataclass
class RootRandomizerCfg:
    # Base/root mean in ENV frame (caller adds scene.env_origins)
    base_xyz: Tuple[float, float, float] = (0.0, 0.0, 4.0)  # Z fixed at 4 by default

    # XY translation noise (meters)
    x_std: float = 0.02
    y_std: float = 0.02
    x_clip: float = 0.06
    y_clip: float = 0.06

    # Z is constant (keep std/clip = 0)
    z_value: float = 4.0

    # yaw noise (radians)
    yaw_base: float = 0.0
    yaw_std: float = math.radians(3.0)
    yaw_clip: float = math.radians(10.0)

    # Determinism
    seed: Optional[int] = 0


class RootRandomizer:
    def __init__(self, cfg: RootRandomizerCfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def reseed(self, seed: Optional[int]):
        self.cfg.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, B: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            pos_env:   (B,3) float32
            quat_wxyz: (B,4) float32
        """
        c = self.cfg
        B = int(B)
        if B <= 0:
            raise ValueError("B must be > 0")

        # Base
        bx, by, _bz = float(c.base_xyz[0]), float(c.base_xyz[1]), float(c.base_xyz[2])
        z = float(c.z_value)  # force constant z

        # Sample XY
        dx = self.rng.normal(0.0, float(c.x_std), size=B)
        dy = self.rng.normal(0.0, float(c.y_std), size=B)
        dx = np.clip(dx, -float(c.x_clip), float(c.x_clip))
        dy = np.clip(dy, -float(c.y_clip), float(c.y_clip))

        pos_env = np.zeros((B, 3), dtype=np.float32)
        pos_env[:, 0] = bx + dx.astype(np.float32)
        pos_env[:, 1] = by + dy.astype(np.float32)
        pos_env[:, 2] = np.float32(z)

        # Sample yaw only
        dyaw = self.rng.normal(0.0, float(c.yaw_std), size=B)
        dyaw = np.clip(dyaw, -float(c.yaw_clip), float(c.yaw_clip))
        yaw = float(c.yaw_base) + dyaw

        quat = np.zeros((B, 4), dtype=np.float32)
        for i in range(B):
            quat[i] = yaw_to_quat_wxyz(float(yaw[i]))

        return pos_env, quat
