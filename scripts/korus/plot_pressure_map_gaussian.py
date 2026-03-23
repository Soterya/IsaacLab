#!/usr/bin/env python3
"""
Live viewer for ALL foam-bed pressure grids with optional Gaussian smoothing.

Subscribes to:
  /foam_bed/top_surface/pressure_grid_{row}_{col}

and shows a rows×cols panel with a shared colorbar.

Examples:
  python3 view_all_pressure_grids.py --rows 2 --cols 3
  python3 view_all_pressure_grids.py --rows 4 --cols 8 --fps 20 --vmin 0 --vmax 5
  python3 view_all_pressure_grids.py --rows 2 --cols 2 --prefix /foam_bed/top_surface/pressure_grid_
  python3 view_all_pressure_grids.py --rows 2 --cols 2 --transpose
  python3 view_all_pressure_grids.py --rows 2 --cols 3 --gauss-sigma 2.0
"""

import argparse
import threading
import time
from typing import Optional, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def parse_grid(msg: Float32MultiArray) -> np.ndarray:
    """Convert Float32MultiArray -> (H,W) grid using layout if present."""
    arr = np.array(msg.data, dtype=np.float32)
    dims = msg.layout.dim
    if len(dims) >= 2 and dims[0].size > 0 and dims[1].size > 0:
        rows = int(dims[0].size)
        cols = int(dims[1].size)
        if rows * cols == arr.size and rows > 0 and cols > 0:
            return arr.reshape(rows, cols)
    # Fallbacks
    if arr.size == 0:
        return np.zeros((1, 1), np.float32)
    side = int(np.sqrt(arr.size))
    if side * side == arr.size and side > 0:
        return arr.reshape(side, side)
    return arr.reshape(1, -1)


# --------------------------
# Gaussian (NaN-aware) utils
# --------------------------

def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Return a 1D Gaussian kernel normalized to 1. If sigma<=0, returns [1.]."""
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def _convolve1d_same(x: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """1D convolution 'same' along an axis using np.apply_along_axis."""
    def conv(v):
        return np.convolve(v, k, mode="same")
    return np.apply_along_axis(conv, axis, x)


def gaussian_blur2d_nan(a: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """
    NaN-aware separable Gaussian blur.
    We blur (a * mask) and the mask, then divide to ignore NaNs/infs.
    """
    if k1d.size == 1:
        return a
    mask = np.isfinite(a).astype(np.float32)
    a0 = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Blur data
    tmp = _convolve1d_same(a0,  k1d, axis=1)
    out = _convolve1d_same(tmp, k1d, axis=0)

    # Blur mask
    tmpm = _convolve1d_same(mask, k1d, axis=1)
    den = _convolve1d_same(tmpm, k1d, axis=0)
    den = np.maximum(den, 1e-6)
    return out / den


class AllPressureViewer(Node):
    def __init__(self, rows: int, cols: int, prefix: str, transpose: bool, flipx: bool, flipy: bool,
                 gauss_sigma: float, gauss_truncate: float):
        super().__init__('all_pressure_grid_viewer')
        self.R = rows
        self.C = cols
        self.prefix = prefix
        self.transpose = transpose
        self.flipx = flipx
        self.flipy = flipy
        self._gkernel = _gaussian_kernel1d(gauss_sigma, gauss_truncate)

        # Latest grids + lock
        self._latest: List[List[Optional[np.ndarray]]] = [[None for _ in range(cols)] for _ in range(rows)]
        self._lock = threading.Lock()

        # Create subscribers
        self._subs = []
        for r in range(rows):
            for c in range(cols):
                topic = f"{prefix}{r}_{c}"
                self.get_logger().info(f"Subscribing to {topic}")
                self._subs.append(self.create_subscription(
                    Float32MultiArray, topic, self._mk_cb(r, c), 10
                ))

    def _mk_cb(self, r: int, c: int):
        def _cb(msg: Float32MultiArray):
            grid = parse_grid(msg)
            # Optional orientation fixes
            if self.transpose:
                grid = grid.T
            if self.flipx:
                grid = np.flip(grid, axis=1)
            if self.flipy:
                grid = np.flip(grid, axis=0)
            # Gaussian smoothing (before storing/displaying)
            grid = gaussian_blur2d_nan(grid, self._gkernel)
            with self._lock:
                self._latest[r][c] = grid
        return _cb

    def snapshot(self) -> List[List[Optional[np.ndarray]]]:
        with self._lock:
            return [[None if self._latest[r][c] is None else self._latest[r][c].copy()
                     for c in range(self.C)] for r in range(self.R)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, required=True, help='Number of rows.')
    ap.add_argument('--cols', type=int, required=True, help='Number of cols.')
    ap.add_argument('--prefix', type=str, default='/foam_bed/top_surface/pressure_grid_',
                    help='Topic prefix before {row}_{col}.')
    ap.add_argument('--fps', type=float, default=20.0, help='UI refresh rate.')
    ap.add_argument('--vmin', type=float, default=None, help='Fixed vmin for colormap.')
    ap.add_argument('--vmax', type=float, default=None, help='Fixed vmax for colormap.')
    ap.add_argument('--interpolation', default='nearest',
                    choices=['nearest', 'bilinear', 'bicubic', 'none'],
                    help='imshow interpolation.')
    ap.add_argument('--transpose', action='store_true', help='Transpose each tile (swap X/Y).')
    ap.add_argument('--flipx', action='store_true', help='Flip each tile horizontally.')
    ap.add_argument('--flipy', action='store_true', help='Flip each tile vertically.')
    ap.add_argument('--tight', action='store_true', help='Tight layout.')
    ap.add_argument('--gauss-sigma', type=float, default=1.0,
                    help='Gaussian sigma in pixels (0 disables).')
    ap.add_argument('--gauss-truncate', type=float, default=3.0,
                    help='Truncate radius = truncate*sigma.')
    args = ap.parse_args()

    rclpy.init()
    node = AllPressureViewer(args.rows, args.cols, args.prefix, args.transpose, args.flipx, args.flipy,
                             gauss_sigma=args.gauss_sigma, gauss_truncate=args.gauss_truncate)

    # Run ROS in a background thread so callbacks are always serviced
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # Matplotlib: rows×cols subplots
    plt.ion()
    fig, axes = plt.subplots(args.rows, args.cols, squeeze=False)
    try:
        fig.canvas.manager.set_window_title("All Pressure Grids")
    except Exception:
        pass
    fig.suptitle(f"Pressure (area-weighted, compression positive)  •  Gaussian σ={args.gauss_sigma}")

    # Shared normalization
    if args.vmin is not None or args.vmax is not None:
        vmin = args.vmin if args.vmin is not None else 0.0
        vmax = args.vmax if args.vmax is not None else vmin + 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)
        autoscale = False
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
        autoscale = True

    # Create axes images
    ims = [[None for _ in range(args.cols)] for _ in range(args.rows)]
    for r in range(args.rows):
        for c in range(args.cols):
            ax = axes[r][c]
            im = ax.imshow(np.zeros((2, 2), dtype=np.float32), origin='lower',
                           interpolation=args.interpolation, norm=norm, aspect='auto')
            ax.set_title(f"r={r}, c={c}")
            ax.set_xticks([]); ax.set_yticks([])
            ims[r][c] = im

    # Shared colorbar
    cbar = fig.colorbar(ims[0][0], ax=axes, shrink=0.9)

    if args.tight:
        plt.tight_layout()

    dt = 1.0 / max(1e-3, args.fps)

    try:
        while plt.fignum_exists(fig.number):
            snap = node.snapshot()

            # Autoscale across all received tiles (robust to NaN/inf)
            if autoscale:
                mins, maxs = [], []
                for r in range(args.rows):
                    for c in range(args.cols):
                        g = snap[r][c]
                        if g is None:
                            continue
                        finite = np.isfinite(g)
                        if finite.any():
                            mins.append(np.nanpercentile(g[finite], 1.0))
                            maxs.append(np.nanpercentile(g[finite], 99.0))
                if mins and maxs:
                    vmin = float(np.min(mins))
                    vmax = float(np.max(maxs))
                    if not np.isfinite(vmin): vmin = 0.0
                    if not np.isfinite(vmax): vmax = vmin + 1.0
                    if vmax <= vmin: vmax = vmin + 1e-6
                    norm.vmin = vmin
                    norm.vmax = vmax
                    cbar.update_normal(ims[0][0])

            # Update images
            for r in range(args.rows):
                for c in range(args.cols):
                    g = snap[r][c]
                    im = ims[r][c]
                    if g is None:
                        continue
                    # Always update data
                    im.set_data(g)
                    # Update extent to match current shape so it fills the axes
                    H, W = g.shape
                    im.set_extent((0, W, 0, H))
                    # Keep axes limits in sync so image isn't stuck in a corner
                    ax = axes[r][c]
                    ax.set_xlim(0, W)
                    ax.set_ylim(0, H)

            plt.pause(0.001)
            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
