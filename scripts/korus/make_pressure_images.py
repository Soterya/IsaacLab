#!/usr/bin/env python3
"""
Batch plotter: reads per-frame Korus GT npz files and saves pressure-map mosaics as images.

Input NPZ format (from your converter):
  - timestamp     : scalar
  - pose_name     : string
  - pressure_maps : [Ncells, H, W]   (Ncells=32)

Output:
  - One image per npz file:
      posename_{pose_name}_timestamp_{timestamp}.png   (or based on npz filename)

Layout:
  - tiles a (rows x cols) grid (default 4x8 = 32 cells)
  - shared colorbar
  - supports downsample per tile (avg/max/nearest) and nonlinear scaling (linear/log/power/symlog)

Examples:
  python scripts/korus/make_pressure_images.py --rows 8 --cols 4 --norm power --gamma 0.6 --in_dir scripts/korus/korus_pressure_pose_dataset_final/general_supine/female_train_roll0_f_lay_set10to13_8000/ --out_dir scripts/korus/korus_pressure_pose_dataset_images/general_supine/female_train_roll0_f_lay_set10to13_8000/  --out_rows 6 --out_cols 6
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless batch rendering
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, PowerNorm, SymLogNorm


# ------------------------
# helpers (from your viewer)
# ------------------------

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

    if mode == "nearest":
        rr = (np.linspace(0, H - 1, out_rows)).astype(np.int32)
        cc = (np.linspace(0, W - 1, out_cols)).astype(np.int32)
        return grid[np.ix_(rr, cc)]

    # Divisible block pooling
    if (H % out_rows == 0) and (W % out_cols == 0):
        bh = H // out_rows
        bw = W // out_cols
        g = grid[:out_rows * bh, :out_cols * bw]
        g = g.reshape(out_rows, bh, out_cols, bw)
        if mode == "max":
            return np.nanmax(g, axis=(1, 3))
        return np.nanmean(g, axis=(1, 3))  # avg

    # Fallback nearest
    rr = (np.linspace(0, H - 1, out_rows)).astype(np.int32)
    cc = (np.linspace(0, W - 1, out_cols)).astype(np.int32)
    return grid[np.ix_(rr, cc)]


def make_norm(kind: str, vmin: float, vmax: float, *, gamma: float, linthresh: float, eps: float):
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
        vmin_pos = max(vmin, float(eps))
        vmax_pos = max(vmax, vmin_pos + float(eps))
        return LogNorm(vmin=vmin_pos, vmax=vmax_pos)

    if kind == "power":
        return PowerNorm(gamma=float(gamma), vmin=vmin, vmax=vmax)

    if kind == "symlog":
        lt = max(float(linthresh), 1e-12)
        return SymLogNorm(linthresh=lt, vmin=vmin, vmax=vmax)

    raise ValueError(f"Unknown norm kind: {kind}")


def safe_slug(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s if s else "unknown"


def format_timestamp_for_name(ts: float, decimals: int) -> str:
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(float(ts))
    # match your earlier naming convention (optional)
    s = s.replace(".", "p").replace("-", "m")
    return s


def load_npz_pressure(path: Path) -> Tuple[str, float, np.ndarray]:
    """
    Returns (pose_name, timestamp, pressure_maps[Ncells,H,W])
    """
    d = np.load(path, allow_pickle=True)
    if "pressure_maps" not in d:
        raise RuntimeError(f"{path} missing key 'pressure_maps'")
    p = d["pressure_maps"].astype(np.float32)
    if p.ndim != 3:
        raise RuntimeError(f"{path}: pressure_maps must be [Ncells,H,W], got {p.shape}")
    pose_name = str(d["pose_name"]) if "pose_name" in d else path.stem
    timestamp = float(d["timestamp"]) if "timestamp" in d else np.nan
    return pose_name, timestamp, p


def compute_autoscale(pmaps: np.ndarray, norm_kind: str, eps: float, pmin: float, pmax: float) -> Tuple[float, float]:
    """
    Robust autoscale across all cells/pixels using percentiles.
    For log: ignore non-positive values.
    """
    flat = pmaps.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0

    if norm_kind == "log":
        flat = flat[flat > float(eps)]
        if flat.size == 0:
            return float(eps), float(eps) * 10.0

    vmin = float(np.nanpercentile(flat, pmin))
    vmax = float(np.nanpercentile(flat, pmax))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    # Avoid log vmin <= 0
    if norm_kind == "log":
        vmin = max(vmin, float(eps))
        vmax = max(vmax, vmin + float(eps))
    return vmin, vmax


# ------------------------
# plotting
# ------------------------

def plot_one(
    pmaps: np.ndarray,
    out_path: Path,
    *,
    rows: int,
    cols: int,
    transpose: bool,
    flipx: bool,
    flipy: bool,
    out_rows: Optional[int],
    out_cols: Optional[int],
    pool: str,
    interpolation: str,
    norm_kind: str,
    vmin: Optional[float],
    vmax: Optional[float],
    gamma: float,
    linthresh: float,
    eps: float,
    pmin: float,
    pmax: float,
    title: Optional[str],
    dpi: int,
):
    Ncells, H, W = pmaps.shape
    if rows * cols != Ncells:
        raise RuntimeError(f"rows*cols must equal Ncells. Got rows*cols={rows*cols}, Ncells={Ncells}")

    # Per-tile transforms + optional downsample
    tiles = []
    for i in range(Ncells):
        g = pmaps[i]
        if transpose:
            g = g.T
        if flipx:
            g = np.flip(g, axis=1)
        if flipy:
            g = np.flip(g, axis=0)
        if out_rows is not None and out_cols is not None:
            g = pool_downsample(g, out_rows, out_cols, mode=pool)
        tiles.append(g)

    # Autoscale if needed
    autoscale = (vmin is None and vmax is None)
    if autoscale:
        vmin2, vmax2 = compute_autoscale(np.stack(tiles, axis=0), norm_kind, eps, pmin, pmax)
    else:
        vmin2 = float(vmin) if vmin is not None else 0.0
        vmax2 = float(vmax) if vmax is not None else (vmin2 + 1.0)

    norm = make_norm(norm_kind, vmin2, vmax2, gamma=gamma, linthresh=linthresh, eps=eps)

    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(8,16))
    if title:
        fig.suptitle(title)

    # Draw tiles
    im0 = None
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            ax = axes[r][c]
            im = ax.imshow(
                tiles[idx],
                origin="lower",
                interpolation=interpolation if interpolation != "none" else "nearest",
                norm=norm,
                aspect="auto",
            )
            ax.set_title(f"{idx:02d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if im0 is None:
                im0 = im

    # Shared colorbar
    if im0 is not None:
        fig.colorbar(im0, ax=axes, shrink=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing per-frame NPZs (posename_*_timestamp_*.npz)")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write images")

    ap.add_argument("--rows", type=int, default=4, help="Mosaic rows (default 4)")
    ap.add_argument("--cols", type=int, default=8, help="Mosaic cols (default 8)")

    ap.add_argument("--transpose", action="store_true", help="Transpose each tile")
    ap.add_argument("--flipx", action="store_true", help="Flip each tile horizontally")
    ap.add_argument("--flipy", action="store_true", help="Flip each tile vertically")

    ap.add_argument("--interpolation", default="nearest",
                    choices=["nearest", "bilinear", "bicubic", "none"],
                    help="imshow interpolation")

    # Downsample per tile
    ap.add_argument("--out_rows", type=int, default=None, help="Downsample each tile to rows")
    ap.add_argument("--out_cols", type=int, default=None, help="Downsample each tile to cols")
    ap.add_argument("--pool", type=str, default="avg", choices=["avg", "max", "nearest"], help="Downsample mode")

    # Scaling
    ap.add_argument("--norm", default="linear", choices=["linear", "log", "power", "symlog"], help="Colormap scaling")
    ap.add_argument("--vmin", type=float, default=None, help="Fixed vmin (else autoscale)")
    ap.add_argument("--vmax", type=float, default=None, help="Fixed vmax (else autoscale)")
    ap.add_argument("--gamma", type=float, default=0.5, help="Gamma for power norm")
    ap.add_argument("--linthresh", type=float, default=1e-4, help="Linthresh for symlog")
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for log norm & filtering")
    ap.add_argument("--pmin", type=float, default=1.0, help="Autoscale lower percentile (default 1)")
    ap.add_argument("--pmax", type=float, default=99.0, help="Autoscale upper percentile (default 99)")

    # Output naming
    ap.add_argument("--use_npz_stem", action="store_true",
                    help="Name output png as <npz_stem>.png instead of pose+timestamp from file contents")
    ap.add_argument("--ts_decimals", type=int, default=6, help="Timestamp decimals in output filename (if not using stem)")

    ap.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    ap.add_argument("--ext", type=str, default="png", choices=["png", "jpg", "jpeg"], help="Image extension")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    ap.add_argument("--dry_run", action="store_true", help="Print outputs without saving")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted([p for p in in_dir.iterdir() if p.suffix == ".npz"])
    if args.limit and args.limit > 0:
        npz_files = npz_files[: args.limit]

    if not npz_files:
        raise SystemExit(f"[ERROR] No npz files found in {in_dir}")

    print(f"[INFO] Found {len(npz_files)} npz files in {in_dir}")
    print(f"[INFO] Writing images to {out_dir}")

    for i, p in enumerate(npz_files):
        pose_name, timestamp, pmaps = load_npz_pressure(p)

        if args.use_npz_stem:
            out_name = f"{p.stem}.{args.ext}"
            title = p.stem
        else:
            pose_token = safe_slug(pose_name)
            ts_token = format_timestamp_for_name(timestamp, args.ts_decimals)
            out_name = f"posename_{pose_token}_timestamp_{ts_token}.{args.ext}"
            title = f"{pose_name}  t={timestamp:.6f}"

        out_path = out_dir / out_name

        if args.dry_run:
            print(f"[DRY] {p.name} -> {out_path.name}  pmaps={pmaps.shape}")
            continue

        plot_one(
            pmaps,
            out_path,
            rows=args.rows,
            cols=args.cols,
            transpose=args.transpose,
            flipx=args.flipx,
            flipy=args.flipy,
            out_rows=args.out_rows,
            out_cols=args.out_cols,
            pool=args.pool,
            interpolation=args.interpolation,
            norm_kind=args.norm,
            vmin=args.vmin,
            vmax=args.vmax,
            gamma=args.gamma,
            linthresh=args.linthresh,
            eps=args.eps,
            pmin=args.pmin,
            pmax=args.pmax,
            title=title,
            dpi=args.dpi,
        )

        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i+1}/{len(npz_files)}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
