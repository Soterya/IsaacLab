#!/usr/bin/env python3
"""
Visualize recorded Korus dataset (.npz): MoCap + pressure.

Usage:
  python viz_recording_npz.py /path/to/episode_0000.npz
  python viz_recording_npz.py /path/to/episode_0000.npz --fps 30
  python viz_recording_npz.py /path/to/episode_0000.npz --no-pressure
  python viz_recording_npz.py /path/to/episode_0000.npz --save_mp4 out.mp4

Controls:
  Space: play/pause
  Left/Right: step frame
  Home/End: first/last frame
  r: reset view
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# ---------- Helpers ----------

def _as_str_list(x) -> list[str]:
    """Convert npz string-ish arrays to python list[str]."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("U", "S", "O"):
            return [str(v) for v in x.tolist()]
    return [str(x)]

def rotvec_to_rotmat(rv: np.ndarray) -> np.ndarray:
    """Rodrigues: rotvec (3,) -> R (3,3)."""
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]], dtype=np.float64)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def find_first_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None

def guess_rows_cols(npz, n_cells: int):
    # Prefer explicit meta
    for k_rows in ("bed_rows", "rows"):
        for k_cols in ("bed_cols", "cols"):
            if k_rows in npz and k_cols in npz:
                try:
                    r = int(npz[k_rows])
                    c = int(npz[k_cols])
                    if r * c == n_cells:
                        return r, c
                except Exception:
                    pass
    # Fallback: square-ish
    r = int(np.floor(np.sqrt(n_cells)))
    c = int(np.ceil(n_cells / max(r, 1)))
    if r * c != n_cells:
        r = 1
        c = n_cells
    return r, c

def stitch_cells_to_bed(cells: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    cells: (T, N, H, W) where N = rows*cols.vered yo
    returns: (T, rows*H, cols*W)
    """
    T, N, H, W = cells.shape
    assert N == rows * cols, f"cells N={N} does not match rows*cols={rows*cols}"
    bed = np.zeros((T, rows * H, cols * W), dtype=cells.dtype)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            bed[:, r*H:(r+1)*H, c*W:(c+1)*W] = cells[:, idx]
    return bed

def build_skeleton_edges(body_names: list[str]) -> list[tuple[str, str]]:
    """
    A reasonable chain graph for your 24-link naming.
    Uses *body link* names, not joint names.
    """
    edges = []

    def add(a, b):
        if a in body_names and b in body_names:
            edges.append((a, b))

    # Core
    add("Pelvis", "Spine")
    add("Spine", "Torso")
    add("Torso", "Chest")
    add("Chest", "Neck")
    add("Neck", "Head")

    # Left leg
    add("Pelvis", "L_Hip")
    add("L_Hip", "L_Knee")
    add("L_Knee", "L_Ankle")
    add("L_Ankle", "L_Toe")

    # Right leg
    add("Pelvis", "R_Hip")
    add("R_Hip", "R_Knee")
    add("R_Knee", "R_Ankle")
    add("R_Ankle", "R_Toe")

    # Left arm / shoulder chain
    add("Chest", "L_Thorax")
    add("L_Thorax", "L_Shoulder")
    add("L_Shoulder", "L_Elbow")
    add("L_Elbow", "L_Wrist")
    add("L_Wrist", "L_Hand")

    # Right arm / shoulder chain
    add("Chest", "R_Thorax")
    add("R_Thorax", "R_Shoulder")
    add("R_Shoulder", "R_Elbow")
    add("R_Elbow", "R_Wrist")
    add("R_Wrist", "R_Hand")

    # If Pelvis isn't present (your asset root is Spine), still show torso chain
    if "Pelvis" not in body_names and "Spine" in body_names:
        # no extra edges needed; torso chain already works
        pass

    return edges

# ---------- Main visualizer ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", type=str)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-pressure", action="store_true")
    ap.add_argument("--no-rot-axes", action="store_true", help="Don't draw local axes from rotvec")
    ap.add_argument("--axis-scale", type=float, default=0.15, help="Length of drawn orientation axes in meters")
    ap.add_argument("--save_mp4", type=str, default=None, help="Path to save MP4 (requires ffmpeg)")
    args = ap.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    keys = list(data.keys())
    print("[INFO] Keys in npz:\n  " + "\n  ".join(keys))

    # --- MoCap: body names + body positions ---
    k_body_names = find_first_key(data, ["body_names", "humanoid_body_names", "link_names"])
    if k_body_names is None:
        raise RuntimeError("Could not find body_names in NPZ. Expected key like 'body_names'.")

    body_names = _as_str_list(data[k_body_names])
    body_index = {n: i for i, n in enumerate(body_names)}
    print(f"[INFO] Found {len(body_names)} body names. First few: {body_names[:8]}")

    k_body_pos = find_first_key(data, ["body_pos_w", "body_link_pos_w", "mocap_body_pos_w", "link_pos_w"])
    if k_body_pos is None:
        raise RuntimeError("Could not find body positions in NPZ. Expected key like 'body_pos_w' or 'body_link_pos_w'.")
    body_pos = np.asarray(data[k_body_pos], dtype=np.float32)  # (T, B, 3)

    k_body_rotvec = find_first_key(data, ["body_rotvec_w", "body_link_rotvec_w", "mocap_body_rotvec_w", "link_rotvec_w"])
    body_rotvec = None
    if k_body_rotvec is not None:
        body_rotvec = np.asarray(data[k_body_rotvec], dtype=np.float32)  # (T, B, 3)

    T = body_pos.shape[0]
    B = body_pos.shape[1]
    assert body_pos.shape[-1] == 3, "body positions must be (...,3)"
    assert B == len(body_names), f"body_pos second dim {B} != len(body_names) {len(body_names)}"

    # --- Pressure detection / stitching ---
    pressure_bed = None  # (T, H, W)
    pressure_title = "Pressure"
    if not args.no_pressure:
        k_pressure = find_first_key(data, [
            "pressure_bed", "pressure_map", "pressure_full", "pressure_grid_full",
            "pressure_grid", "pressure"
        ])
        if k_pressure is not None:
            P = np.asarray(data[k_pressure])
            # Accept a variety of shapes:
            # (T, H, W) full-bed
            # (T, N, H, W) per-cell -> stitch
            if P.ndim == 3:
                pressure_bed = P.astype(np.float32)
                pressure_title = f"Pressure ({k_pressure})"
            elif P.ndim == 4:
                n_cells = P.shape[1]
                rows, cols = guess_rows_cols(data, n_cells)
                pressure_bed = stitch_cells_to_bed(P.astype(np.float32), rows, cols)
                pressure_title = f"Pressure stitched ({k_pressure})  [{rows}x{cols} cells]"
            else:
                print(f"[WARN] Pressure key '{k_pressure}' has unsupported shape: {P.shape}")
        else:
            print("[WARN] No obvious pressure key found; run with --no-pressure to hide this warning.")

    # --- Plot setup ---
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[18, 2], width_ratios=[12, 10])

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    axP = fig.add_subplot(gs[0, 1])
    axS = fig.add_subplot(gs[1, :])

    # Set initial views
    ax3d.set_title("MoCap: body link positions (world)")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # Use a consistent bounds based on all frames (can be large; ok for verification)
    xyz_min = body_pos.reshape(-1, 3).min(axis=0)
    xyz_max = body_pos.reshape(-1, 3).max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    span = float(np.max(xyz_max - xyz_min) + 1e-6)
    half = 0.55 * span

    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)

    edges = build_skeleton_edges(body_names)

    # Artists
    scat = ax3d.scatter([], [], [], s=25) # type: ignore
    line_artists = []
    for _ in edges:
        (ln,) = ax3d.plot([], [], [], linewidth=2)
        line_artists.append(ln)

    # Orientation axes (optional): draw at Pelvis if exists else Spine
    anchor_name = "Pelvis" if "Pelvis" in body_index else ("Spine" if "Spine" in body_index else body_names[0])
    anchor_id = body_index[anchor_name]
    quiv = None  # will be a tuple of quiver artists

    # Pressure image
    im = None
    if pressure_bed is not None:
        axP.set_title(pressure_title)
        im = axP.imshow(pressure_bed[0], origin="lower", aspect="auto")
        fig.colorbar(im, ax=axP, fraction=0.046, pad=0.04)
    else:
        axP.set_title("Pressure (not found / disabled)")
        axP.text(0.5, 0.5, "No pressure data", ha="center", va="center")
        axP.set_axis_off()

    # Slider + playback state
    slider = Slider(axS, "frame", 0, T - 1, valinit=0, valstep=1)
    playing = {"on": False}
    frame_state = {"i": 0}

    def set_frame(i: int):
        i = int(np.clip(i, 0, T - 1))
        frame_state["i"] = i

        pts = body_pos[i]  # (B,3)
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2]) # type: ignore

        # label a few key points
        ax3d.set_title(f"MoCap (world)  frame={i}/{T-1}  anchor={anchor_name}")

        # edges
        for (a, b), ln in zip(edges, line_artists):
            ia = body_index[a]
            ib = body_index[b]
            xs = [pts[ia, 0], pts[ib, 0]]
            ys = [pts[ia, 1], pts[ib, 1]]
            zs = [pts[ia, 2], pts[ib, 2]]
            ln.set_data(xs, ys)
            ln.set_3d_properties(zs)

        # pressure
        if im is not None:
            im.set_data(pressure_bed[i]) # type: ignore
            # optional: auto-scale color based on robust range
            vmin = float(np.nanpercentile(pressure_bed[i], 2)) # type: ignore
            vmax = float(np.nanpercentile(pressure_bed[i], 98)) + 1e-6 # type: ignore
            im.set_clim(vmin, vmax)

        # orientation axes from rotvec (if present)
        nonlocal quiv
        if (not args.no_rot_axes) and (body_rotvec is not None):
            p = pts[anchor_id].astype(np.float64)
            Rm = rotvec_to_rotmat(body_rotvec[i, anchor_id].astype(np.float64))
            # local x,y,z axes in world
            ax_len = float(args.axis_scale)
            ux = Rm @ np.array([ax_len, 0, 0], dtype=np.float64)
            uy = Rm @ np.array([0, ax_len, 0], dtype=np.float64)
            uz = Rm @ np.array([0, 0, ax_len], dtype=np.float64)

            # Remove previous quivers
            if quiv is not None:
                for q in quiv:
                    q.remove()

            qx = ax3d.quiver(p[0], p[1], p[2], ux[0], ux[1], ux[2])
            qy = ax3d.quiver(p[0], p[1], p[2], uy[0], uy[1], uy[2])
            qz = ax3d.quiver(p[0], p[1], p[2], uz[0], uz[1], uz[2])
            quiv = (qx, qy, qz)

        fig.canvas.draw_idle()

    def on_slider(val):
        set_frame(int(val))

    slider.on_changed(on_slider)

    def on_key(event):
        i = frame_state["i"]
        if event.key == " ":
            playing["on"] = not playing["on"]
        elif event.key in ("right", "d"):
            set_frame(i + 1)
            slider.set_val(frame_state["i"])
        elif event.key in ("left", "a"):
            set_frame(i - 1)
            slider.set_val(frame_state["i"])
        elif event.key == "home":
            set_frame(0)
            slider.set_val(0)
        elif event.key == "end":
            set_frame(T - 1)
            slider.set_val(T - 1)
        elif event.key == "r":
            ax3d.view_init(elev=20, azim=-60)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def tick(_):
        if playing["on"]:
            i = frame_state["i"] + 1
            if i >= T:
                i = 0
            set_frame(i)
            slider.set_val(i)

    # Initialize
    set_frame(0)

    interval_ms = int(max(1, 1000.0 / float(args.fps)))
    ani = FuncAnimation(fig, tick, interval=interval_ms) # type: ignore

    if args.save_mp4:
        print(f"[INFO] Saving MP4 to: {args.save_mp4}")
        ani.save(args.save_mp4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
