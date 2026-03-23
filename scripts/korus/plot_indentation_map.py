#!/usr/bin/env python3
"""
Live viewer for ALL foam-bed indentation grids.

Subscribes to:
  /foam_bed/top_surface/indentation_grid_{idx:02d}

and shows a rows×cols panel with a shared colorbar.

This version can downsample each incoming tile (e.g., 12x12 -> 6x6) via pooling
and supports nonlinear colormap scaling (log / power / symlog).

Examples:
  python3 view_all_indentation_grids.py --rows 4 --cols 8
  python3 view_all_indentation_grids.py --rows 4 --cols 8 --fps 20
  python3 view_all_indentation_grids.py --rows 4 --cols 8 --vmin 0 --vmax 0.05
  python3 view_all_indentation_grids.py --rows 4 --cols 8 --transpose

  # Force 12x12 -> 6x6 (2x2 average pooling)
  python3 view_all_indentation_grids.py --rows 4 --cols 8 --out_rows 6 --out_cols 6 --pool avg

  # Nonlinear visualization
  python3 view_all_indentation_grids.py --rows 4 --cols 8 --norm power --gamma 0.4
  python3 view_all_indentation_grids.py --rows 4 --cols 8 --norm log --eps 1e-6
  python3 view_all_indentation_grids.py --rows 4 --cols 8 --norm symlog --linthresh 1e-4
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
from matplotlib.colors import Normalize, LogNorm, PowerNorm, SymLogNorm

def parse_grid(msg: Float32MultiArray) -> np.ndarray:
    """Convert Float32MultiArray -> (H,W) grid using layout if present."""
    arr = np.array(msg.data, dtype=np.float32)
    dims = msg.layout.dim
    if len(dims) >= 2 and dims[0].size > 0 and dims[1].size > 0:  # type: ignore
        rows = int(dims[0].size)  # type: ignore
        cols = int(dims[1].size)  # type: ignore
        if rows * cols == arr.size and rows > 0 and cols > 0:
            return arr.reshape(rows, cols)

    # Fallbacks
    if arr.size == 0:
        return np.zeros((1, 1), np.float32)
    side = int(np.sqrt(arr.size))
    if side * side == arr.size and side > 0:
        return arr.reshape(side, side)
    return arr.reshape(1, -1)


def pool_downsample(grid: np.ndarray, out_rows: int, out_cols: int, mode: str = "avg") -> np.ndarray:
    """
    Downsample grid to (out_rows, out_cols).
    Fast-path when divisible: uses block pooling (avg or max).
    Otherwise falls back to nearest-neighbor indexing.
    """
    H, W = grid.shape
    if out_rows <= 0 or out_cols <= 0:
        return grid
    if H == out_rows and W == out_cols:
        return grid

    # Divisible block pooling (best quality for 12->6)
    if (H % out_rows == 0) and (W % out_cols == 0):
        bh = H // out_rows
        bw = W // out_cols
        g = grid[:out_rows * bh, :out_cols * bw]
        g = g.reshape(out_rows, bh, out_cols, bw)

        if mode == "max":
            return np.nanmax(g, axis=(1, 3))
        # default avg
        return np.nanmean(g, axis=(1, 3))

    # Fallback: nearest-neighbor resampling
    rr = (np.linspace(0, H - 1, out_rows)).astype(np.int32)
    cc = (np.linspace(0, W - 1, out_cols)).astype(np.int32)
    return grid[np.ix_(rr, cc)]


def make_norm(kind: str, vmin: float, vmax: float, *, gamma: float, linthresh: float, eps: float):
    """
    Build a Matplotlib normalization object.
    - linear: Normalize
    - log: LogNorm (requires vmin>0)
    - power: PowerNorm (gamma<1 boosts low values)
    - symlog: SymLogNorm (for data crossing 0)
    """
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = vmin + 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    if kind == "linear":
        return Normalize(vmin=vmin, vmax=vmax)

    if kind == "log":
        # LogNorm requires strictly positive vmin
        vmin_pos = max(vmin, float(eps))
        vmax_pos = max(vmax, vmin_pos + float(eps))
        return LogNorm(vmin=vmin_pos, vmax=vmax_pos)

    if kind == "power":
        return PowerNorm(gamma=float(gamma), vmin=vmin, vmax=vmax)

    if kind == "symlog":
        # Allows negative/positive. Uses linear region around 0 sized by linthresh.
        lt = max(float(linthresh), 1e-12)
        return SymLogNorm(linthresh=lt, vmin=vmin, vmax=vmax)

    raise ValueError(f"Unknown norm kind: {kind}")


class AllIndentationViewer(Node):
    def __init__(
        self,
        rows: int,
        cols: int,
        prefix: str,
        transpose: bool,
        flipx: bool,
        flipy: bool,
        out_rows: Optional[int],
        out_cols: Optional[int],
        pool: str,
    ):
        super().__init__("all_indentation_grid_viewer")
        self.R = rows
        self.C = cols
        self.prefix = prefix
        self.transpose = transpose
        self.flipx = flipx
        self.flipy = flipy

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.pool = pool

        # Latest grids + lock
        self._latest: List[List[Optional[np.ndarray]]] = [[None for _ in range(cols)] for _ in range(rows)]
        self._lock = threading.Lock()

        # Create subscribers
        self._subs = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                topic = f"{prefix}{idx:02d}"
                self.get_logger().info(f"Subscribing to {topic}")
                self._subs.append(
                    self.create_subscription(
                        Float32MultiArray, topic, self._mk_cb(r, c), 10
                    )
                )

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

            # Downsample (e.g., 12x12 -> 6x6)
            if self.out_rows is not None and self.out_cols is not None:
                grid = pool_downsample(grid, self.out_rows, self.out_cols, mode=self.pool)

            with self._lock:
                self._latest[r][c] = grid

        return _cb

    def snapshot(self) -> List[List[Optional[np.ndarray]]]:
        with self._lock:
            return [
                [
                    None if self._latest[r][c] is None else self._latest[r][c].copy()  # type: ignore
                    for c in range(self.C)
                ]
                for r in range(self.R)
            ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True, help="Number of rows.")
    ap.add_argument("--cols", type=int, required=True, help="Number of cols.")
    ap.add_argument(
        "--prefix",
        type=str,
        default="/foam_bed/top_surface/indentation_grid_",
        help="Topic prefix before {idx:02d}.",
    )
    ap.add_argument("--fps", type=float, default=20.0, help="UI refresh rate.")
    ap.add_argument("--vmin", type=float, default=None, help="Fixed vmin for colormap.")
    ap.add_argument("--vmax", type=float, default=None, help="Fixed vmax for colormap.")
    ap.add_argument(
        "--interpolation",
        default="nearest",
        choices=["nearest", "bilinear", "bicubic", "none"],
        help="imshow interpolation.",
    )
    ap.add_argument("--transpose", action="store_true", help="Transpose each tile (swap X/Y).")
    ap.add_argument("--flipx", action="store_true", help="Flip each tile horizontally.")
    ap.add_argument("--flipy", action="store_true", help="Flip each tile vertically.")
    ap.add_argument("--tight", action="store_true", help="Tight layout.")

    # Output resolution controls (12x12 -> 6x6)
    ap.add_argument("--out_rows", type=int, default=6, help="Downsample each tile to this many rows (default: 6).")
    ap.add_argument("--out_cols", type=int, default=6, help="Downsample each tile to this many cols (default: 6).")
    ap.add_argument(
        "--pool",
        type=str,
        default="avg",
        choices=["avg", "max", "nearest"],
        help="Downsample mode: avg/max pooling if divisible, else nearest. 'nearest' forces nearest behavior.",
    )
    ap.add_argument("--no_downsample", action="store_true", help="Disable downsampling and show native tile size.")

    # NEW: nonlinear visualization
    ap.add_argument("--norm", default="linear", choices=["linear", "log", "power", "symlog"],
                    help="Colormap scaling. 'log' is great for wide dynamic range (requires positive values).")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="Gamma for --norm power (gamma<1 boosts low values).")
    ap.add_argument("--linthresh", type=float, default=1e-4,
                    help="Linear threshold for --norm symlog (around 0).")
    ap.add_argument("--eps", type=float, default=1e-9,
                    help="Small epsilon for --norm log to avoid log(0).")

    args = ap.parse_args()

    # Resolve downsample settings
    if args.no_downsample:
        out_rows = None
        out_cols = None
        pool = "avg"
    else:
        out_rows = args.out_rows
        out_cols = args.out_cols
        pool = args.pool  # keep avg/max as requested; nearest means force nearest fallback

    rclpy.init()
    node = AllIndentationViewer(
        args.rows, args.cols, args.prefix, args.transpose, args.flipx, args.flipy,
        out_rows, out_cols, pool
    )

    # Run ROS in a background thread so callbacks are always serviced
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # Matplotlib: rows×cols subplots
    plt.ion()
    fig, axes = plt.subplots(args.rows, args.cols, squeeze=False)
    fig.canvas.manager.set_window_title("All Indentation Grids")  # type: ignore
    fig.suptitle("Indentation (positive = compressed/down)")

    # Shared normalization
    autoscale = (args.vmin is None and args.vmax is None)

    if autoscale:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = args.vmin if args.vmin is not None else 0.0
        vmax = args.vmax if args.vmax is not None else (vmin + 1.0)

    norm = make_norm(args.norm, vmin, vmax, gamma=args.gamma, linthresh=args.linthresh, eps=args.eps)

    # Create axes images
    ims = [[None for _ in range(args.cols)] for _ in range(args.rows)]
    for r in range(args.rows):
        for c in range(args.cols):
            ax = axes[r][c]
            im = ax.imshow(
                np.zeros((2, 2), dtype=np.float32),
                origin="lower",
                interpolation=args.interpolation,
                norm=norm,
                aspect="auto",
            )
            ax.set_title(f"idx={r*args.cols+c:02d}")
            ax.set_xticks([])
            ax.set_yticks([])
            ims[r][c] = im

    # Shared colorbar
    cbar = fig.colorbar(ims[0][0], ax=axes, shrink=0.9)  # type: ignore

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
                        if not finite.any():
                            continue

                        gg = g[finite]

                        # For log scale, ignore non-positive values (they can't be log-scaled)
                        if args.norm == "log":
                            gg = gg[gg > args.eps]
                            if gg.size == 0:
                                continue

                        mins.append(np.nanpercentile(gg, 1.0))
                        maxs.append(np.nanpercentile(gg, 99.0))

                if mins and maxs:
                    vmin = float(np.min(mins))
                    vmax = float(np.max(maxs))

                    # Rebuild norm (safer than mutating attributes for non-linear norms)
                    norm = make_norm(args.norm, vmin, vmax, gamma=args.gamma, linthresh=args.linthresh, eps=args.eps)

                    # Update norm for all images + colorbar
                    for r in range(args.rows):
                        for c in range(args.cols):
                            ims[r][c].set_norm(norm)
                    cbar.update_normal(ims[0][0])

            # Update images
            for r in range(args.rows):
                for c in range(args.cols):
                    g = snap[r][c]
                    im = ims[r][c]
                    if g is None:
                        continue
                    im.set_data(g)  # type: ignore
                    H, W = g.shape
                    im.set_extent((0, W, 0, H))  # type: ignore
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
