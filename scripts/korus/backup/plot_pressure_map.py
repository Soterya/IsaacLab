#!/usr/bin/env python3
"""
Live viewer for foam-bed pressure grid published as Float32MultiArray.

Usage:
  # default topic from your sim code
  python3 view_pressure_grid.py

  # custom topic & fixed color limits
  python3 view_pressure_grid.py --topic /foam_bed/top_surface/pressure_grid_0 --fps 30 --vmin 0 --vmax 5.0
"""
import argparse
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you prefer
import matplotlib.pyplot as plt


class PressureGridViewer(Node):
    def __init__(self, topic: str):
        super().__init__('pressure_grid_viewer')
        self._sub = self.create_subscription(
            Float32MultiArray, topic, self._cb, 10
        )
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def _cb(self, msg: Float32MultiArray):
        try:
            # Expecting 2 dims: rows, cols
            dims = msg.layout.dim
            if len(dims) >= 2 and dims[0].size > 0 and dims[1].size > 0:
                rows = int(dims[0].size)
                cols = int(dims[1].size)
                arr = np.array(msg.data, dtype=np.float32)
                if arr.size != rows * cols:
                    # Fallback: try to infer square grid
                    side = int(np.sqrt(arr.size))
                    if side * side == arr.size:
                        rows = cols = side
                    else:
                        # give up; keep flat
                        with self._lock:
                            self._latest = arr
                        return
                grid = arr.reshape(rows, cols)
            else:
                # No layout info; try square
                arr = np.array(msg.data, dtype=np.float32)
                side = int(np.sqrt(arr.size))
                grid = arr.reshape(side, side) if side * side == arr.size else arr

            with self._lock:
                self._latest = grid
        except Exception as e:
            self.get_logger().warn(f"Failed to parse pressure grid: {e}")

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="/foam_bed/top_surface/pressure_grid_0",
        help="Pressure grid topic (Float32MultiArray).",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0, help="UI refresh rate."
    )
    parser.add_argument(
        "--vmin", type=float, default=0, help="Fixed colormap min (default: auto)."
    )
    parser.add_argument(
        "--vmax", type=float, default=10000, help="Fixed colormap max (default: auto)."
    )
    parser.add_argument(
        "--extent", type=float, nargs=4, default=None,
        metavar=("XMIN","XMAX","YMIN","YMAX"),
        help="Axes extent if you want physical units (e.g., 0 1 0 1)."
    )
    parser.add_argument(
        "--interpolation", default="nearest",
        choices=["nearest","bilinear","bicubic","none"],
        help="imshow interpolation."
    )
    args = parser.parse_args()

    rclpy.init()
    node = PressureGridViewer(args.topic)

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(f"Pressure Grid Viewer — {args.topic}")
    im = None
    cbar = None

    dt = 1.0 / max(1e-3, args.fps)
    last_shape: Optional[Tuple[int, int]] = None

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            grid = node.get_latest()
            if grid is None:
                time.sleep(dt)
                continue

            # If shape changed (first frame or resolution change), rebuild artist
            if (im is None) or (grid.shape != last_shape):
                ax.clear()
                im = ax.imshow(
                    grid,
                    origin="lower",
                    interpolation=args.interpolation,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    extent=args.extent if args.extent else None,
                )
                if cbar is not None:
                    cbar.remove()
                cbar = fig.colorbar(im, ax=ax, shrink=0.85)
                ax.set_title("Pressure (area-weighted, compression positive)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                last_shape = grid.shape
            else:
                im.set_data(grid)
                # Auto-scale if limits are not fixed
                if args.vmin is None or args.vmax is None:
                    # avoid NaNs/inf
                    finite = np.isfinite(grid)
                    if finite.any():
                        vmin = np.nanmin(grid[finite]) if args.vmin is None else args.vmin
                        vmax = np.nanmax(grid[finite]) if args.vmax is None else args.vmax
                        # keep vmax > vmin
                        if vmax <= vmin:
                            vmax = vmin + 1e-6
                        im.set_clim(vmin=vmin, vmax=vmax)

            plt.pause(0.001)
            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
