"""
Usage:
    $ cd ~/IsaacLab
    $ preload ./isaaclab.sh -p scripts/korus/run_korusbed_keyboard_ros.py
"""
# --------------------------------
# Imports for the App Creation ---
# --------------------------------
import argparse
import torch
from isaaclab.app import AppLauncher
import numpy as np
from collections import defaultdict
# -----------------------------------------------
# --- Launch the IsaacSim App (with argparse) ---
# -----------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--num_rows", type=int, default=1)
parser.add_argument("--num_cols", type=int, default=1)
parser.add_argument("--every", type=int, default=1, help="Log every N frames to reduce file size")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experience = "isaacsim.exp.full.kit"  # keyboard + ROS need full kit
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# ------------------------------
# Imports after app creation ---
# ------------------------------
import yaml
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import (
    DeformableObject, DeformableObjectCfg,
    RigidObject, RigidObjectCfg,
    Articulation, ArticulationCfg
)
from isaaclab.actuators import ImplicitActuatorCfg
import omni.usd
import isaacsim.core.utils.prims as prim_utils
from keycontrol import KeyControl

# ----------------
# ROS2 Imports ---
# ----------------
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState

# --------------------
# Paths / Globals  ---
# --------------------
ENV_USD       = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
KORUSBED_USD  = "file:///home/rutwik/IsaacLab/scripts/korus/assets/inflatable_cell_all.usd" 

# --------------------
# Global Variables ---
# --------------------
GROUND_TO_BASE_BOTTOM = 0.8
BASE_SIZE   = (1., 1., 0.1)
FOAM_SIZE   = (1., 1., 0.2)
SPHERE_RADIUS = 0.2

BASE_ORIGIN   = (-1.65, 3.85, GROUND_TO_BASE_BOTTOM + round(BASE_SIZE[2]/2,2))
FOAM_ORIGIN   = (-1.65, 3.85, BASE_ORIGIN[2] + round(BASE_SIZE[2]/2,2) + round(FOAM_SIZE[2]/2,2))
SPHERE_ORIGIN = (-1.65, 3.85, 3.0)
DECIMALS      = 1

NODE_SPACING = 1.1  # spacing between Node centers

# -----------------------
# --- Scene creation  ---
# -----------------------
def design_scene():
    """
    Open environment stage and place a single KorusBed at /World/KorusBed.
    Adopt it as ONE articulation that contains many prismatic joints
    (PrismaticJoint0, PrismaticJoint1, ...). Then create per-cell base/foam/sphere
    laid out on an R x C grid.
    """
    # open the usd scene
    scene_context = omni.usd.get_context()
    scene_context.open_stage(ENV_USD)
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    dome.func("/World/DomeLight", dome)
    
    # import the korus bed asset containing cells with prismatic joints
    bed_asset = sim_utils.UsdFileCfg(usd_path=KORUSBED_USD)
    bed_asset.func("/World/KorusBed", bed_asset)

    bed_cfg = ArticulationCfg(
        class_type=Articulation,
        prim_path="/World/KorusBed",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.1), rot=(1, 0, 0, 0),
            joint_pos={f"PrismaticJoint{i}": 0.7 for i in range(32)},
            joint_vel={f"PrismaticJoint{i}": 0.0 for i in range(32)},
        ),
        actuators={
            "prismatic_pd": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint.*"],
                effort_limit_sim=4000.0,
                stiffness=30000.0,
                damping=1200.0,
            ),
        },
    )
    bed = Articulation(cfg=bed_cfg)

    # --- Create R x C “cells” aligned with joints spatially ---
    rows = max(1, int(args_cli.num_rows))
    cols = max(1, int(args_cli.num_cols))
    sx = float(NODE_SPACING)
    sy = float(NODE_SPACING)

    cells = []
    for r in range(rows):
        for c in range(cols):
            idx   = r * cols + c
            x_off = c * sx
            y_off = -r * sy

            prim_utils.create_prim(f"/World/KorusBed/Node{idx}", "Xform")
            
            # Base (rigid)
            base_cfg = RigidObjectCfg(
                prim_path=f"/World/KorusBed/Node{idx}/BasePlate{idx}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=BASE_SIZE,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(BASE_ORIGIN[0] + x_off, BASE_ORIGIN[1] + y_off, BASE_ORIGIN[2])
                ),
            )
            base_obj = RigidObject(cfg=base_cfg)

            # Foam (deformable)
            deform_cfg = DeformableObjectCfg(
                prim_path=f"/World/KorusBed/Node{idx}/DeformableCuboid{idx}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=FOAM_SIZE,
                    deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                        rest_offset=0.0,
                        contact_offset=0.001,
                        simulation_hexahedral_resolution=10, 
                    ),
                    physics_material=sim_utils.DeformableBodyMaterialCfg(
                        poissons_ratio=0.2,
                        youngs_modulus=3e4,
                        dynamic_friction=1.,
                        elasticity_damping=0.06, 
                        damping_scale=1.,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                ),
                init_state=DeformableObjectCfg.InitialStateCfg(
                    pos=(FOAM_ORIGIN[0] + x_off, FOAM_ORIGIN[1] + y_off, FOAM_ORIGIN[2])
                ),
                debug_vis=False,
            )
            cube_obj = DeformableObject(cfg=deform_cfg)

            # Sphere (rigid)
            sphere_cfg = RigidObjectCfg(
                prim_path=f"/World/KorusBed/Node{idx}/RigidSphere{idx}",
                spawn=sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(SPHERE_ORIGIN[0] + x_off, SPHERE_ORIGIN[1] + y_off, SPHERE_ORIGIN[2])
                ),
            )
            sphere_obj = RigidObject(cfg=sphere_cfg)

            cells.append({
                "cube": cube_obj,
                "sphere": sphere_obj,
                "base": base_obj,
                "node_index": idx,
                "row": r,
                "col": c,
            }) # a list of dictionaries that holds the cube,sphere and base objects a row,col index pair. 


    return {"bed": bed, "cells": cells}

# ---------------------------------------------
# Utilities: indices / packing for ROS msgs ---
# ---------------------------------------------
def make_multiarray_2d(arr2d: np.ndarray) -> Float32MultiArray:
    rows, cols = arr2d.shape
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="rows", size=int(rows), stride=int(rows * cols)),
        MultiArrayDimension(label="cols", size=int(cols), stride=int(cols)),
    ]
    msg.data = arr2d.astype(np.float32).ravel().tolist()
    return msg

def make_multiarray_xyz(xyz: np.ndarray) -> Float32MultiArray:
    K = xyz.shape[0]
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="points", size=int(K), stride=int(K * 3)),
        MultiArrayDimension(label="channels", size=3, stride=3),
    ]
    msg.data = xyz.astype(np.float32).ravel().tolist()
    return msg

def get_pin_targets_on_rigid_body(root_pos_w_rigid, root_quat_w_rigid):
    """
    this function returns the positions of corners of rigid base plates, upon which the bottom corners of deformable objects will be pinned
    """
    half_x, half_y, half_z = round(BASE_SIZE[0]/2.0,2), round(BASE_SIZE[1]/2.0,2), round(BASE_SIZE[2]/2.0,2)
    corners_dict = {}
    for idx, center in enumerate(root_pos_w_rigid):
        cx, cy, cz = center
        z_top = cz + half_z
        corners = np.array([
            (cx - half_x, cy - half_y, z_top),  # BL
            (cx - half_x, cy + half_y, z_top),  # TL
            (cx + half_x, cy - half_y, z_top),  # BR
            (cx + half_x, cy + half_y, z_top),  # TR
        ], dtype=float)
        corners_dict[f"cuboid_{idx}"] = np.round(corners, 2).tolist()
    return corners_dict

# --------------------------------
# --- SIM-mesh grid & pressure ---
# --------------------------------
def _round_to_dec(x: np.ndarray, decimals: int):
    s = 10.0 ** decimals
    return np.round(x * s) / s

# def build_top_surface_index_grid_from_sim(cube, decimals=DECIMALS):
#     """Return (index_grid, sim_pos0) where index_grid is HxW of SIM node indices on the *top* surface."""
#     sim_pos = cube.root_physx_view.get_sim_nodal_positions()[0].detach().cpu().numpy()  # (Nv,3)
#     z = sim_pos[:, 2]
#     z_top = _round_to_dec(np.max(z), decimals)
#     z_r = _round_to_dec(z, decimals)
#     top_mask = (z_r == z_top)
#     top_idx = np.nonzero(top_mask)[0]
#     if top_idx.size == 0:
#         raise RuntimeError("No top SIM nodes found; adjust DECIMALS or check mesh resolution.")
#     top_pos = sim_pos[top_idx]

#     x_r = _round_to_dec(top_pos[:, 0], decimals)
#     y_r = _round_to_dec(top_pos[:, 1], decimals)
#     xs = np.unique(np.sort(x_r))
#     ys = np.unique(np.sort(y_r))
#     H, W = len(ys), len(xs)
#     x_to_c = {float(v): i for i, v in enumerate(xs)}
#     y_to_r = {float(v): i for i, v in enumerate(ys)}
#     index_grid = -np.ones((H, W), dtype=np.int64)
#     for xr, yr, vi in zip(x_r, y_r, top_idx):
#         r = y_to_r[float(yr)]
#         c = x_to_c[float(xr)]
#         index_grid[r, c] = int(vi)
#     if (index_grid < 0).any():
#         raise RuntimeError("Top SIM index grid has holes; adjust DECIMALS.")
#     return index_grid, sim_pos  # sim_pos is sim_pos0
def build_top_surface_index_grid_from_sim(cube, decimals=DECIMALS):
    """
    Return (index_grid, sim_pos0) where index_grid is HxW of SIM node indices on the *top* surface.
    Robust version: uses anchored integer bins instead of rounded floats to avoid phantom bins.
    """
    # 1) Pull SIM positions once (host numpy)
    sim_pos_any = cube.root_physx_view.get_sim_nodal_positions()[0]
    if hasattr(sim_pos_any, "detach"):
        sim_pos = sim_pos_any.detach().cpu().numpy()
    else:
        sim_pos = np.asarray(sim_pos_any, dtype=np.float32)
    sim_pos = sim_pos.astype(np.float64, copy=False)  # stable math

    # 2) Find top-plane nodes using anchored z-bins
    s = float(10 ** decimals)
    z = sim_pos[:, 2]
    z_bin = np.rint((z - z.min()) * s).astype(np.int64)
    top_bin = int(z_bin.max())
    top_idx = np.nonzero(z_bin == top_bin)[0]
    if top_idx.size == 0:
        raise RuntimeError("No top SIM nodes found; adjust DECIMALS or check mesh resolution.")

    top_pos = sim_pos[top_idx]
    x = top_pos[:, 0]
    y = top_pos[:, 1]

    # 3) Anchored integer bins for x/y (relative to each top-plane min)
    x0 = float(x.min())
    y0 = float(y.min())
    x_bin = np.rint((x - x0) * s).astype(np.int64)
    y_bin = np.rint((y - y0) * s).astype(np.int64)

    xs_idx = np.unique(x_bin)         # columns (W)
    ys_idx = np.unique(y_bin)         # rows (H)
    W = int(xs_idx.size)
    H = int(ys_idx.size)

    # 4) Maps from integer bin -> grid index
    x_to_c = {int(v): i for i, v in enumerate(xs_idx.tolist())}
    y_to_r = {int(v): i for i, v in enumerate(ys_idx.tolist())}

    # 5) Build HxW grid (tie-break by closeness to bin center to resolve duplicates)
    index_grid = -np.ones((H, W), dtype=np.int64)
    used = np.zeros((H, W), dtype=bool)

    # cell centers for tie-break
    x_center_for_pt = x0 + np.array([x_to_c[int(b)] for b in x_bin], dtype=np.float64) / s
    y_center_for_pt = y0 + np.array([y_to_r[int(b)] for b in y_bin], dtype=np.float64) / s
    tie_score = np.abs(x - x_center_for_pt) + np.abs(y - y_center_for_pt)
    order = np.argsort(tie_score)

    for k in order:
        r = y_to_r[int(y_bin[k])]
        c = x_to_c[int(x_bin[k])]
        if not used[r, c]:
            index_grid[r, c] = int(top_idx[k])
            used[r, c] = True

    # 6) Sanity
    holes = int((index_grid < 0).sum())
    if holes:
        raise RuntimeError(
            f"Top SIM index grid has {holes} unfilled cells "
            f"(H={H}, W={W}, points={top_idx.size}). "
            f"Try changing DECIMALS (e.g., 1 or 3)."
        )

    return index_grid, sim_pos  # sim_pos is sim_pos0


# -------------------------------------------
# Helpers for default planes (collision mesh)
# -------------------------------------------
def get_z_planes_from_default(cube, decimals=DECIMALS):
    default_pos = cube.data.default_nodal_state_w[..., :3][0]
    z = default_pos[:, 2]
    s = 10 ** decimals
    zr = torch.round(z * s) / s
    planes = torch.sort(torch.unique(zr)).values
    if planes.numel() < 2:
        return float(torch.min(z)), float(torch.max(z))
    return float(planes[0]), float(planes[-1])

def get_surface_indices_by_known_z(cube, z_target, decimals=DECIMALS, atol=None):
    default_pos = cube.data.default_nodal_state_w[..., :3][0]
    z = default_pos[:, 2]
    if decimals is not None:
        s = 10 ** decimals
        zr = torch.round(z * s) / s
        zt = torch.tensor(round(float(z_target), decimals), device=z.device, dtype=z.dtype)
        mask = (zr == zt)
    else:
        if atol is None:
            uniq = torch.unique(torch.round(z * 1e5) / 1e5)
            uniq = torch.sort(uniq).values
            dz = float(uniq[-1] - uniq[-2]) if uniq.numel() >= 2 else 1e-4
            atol = max(1e-5, 0.25 * dz)
        zt = torch.tensor(float(z_target), device=z.device, dtype=z.dtype)
        mask = torch.isclose(z, zt, atol=atol)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if idx.numel() > 0:
        return idx
    # Fallback to nearest rounded plane
    s = 10 ** (decimals if decimals is not None else 3)
    zr = torch.round(z * s) / s
    planes = torch.sort(torch.unique(zr)).values
    diffs = torch.abs(planes - float(z_target))
    nearest = planes[torch.argmin(diffs)]
    mask2 = (zr == nearest)
    idx2 = torch.nonzero(mask2, as_tuple=False).squeeze(1)
    return idx2

def get_surface_corner_indices_from_idx(cube, surface_idx, env_id: int = 0):
    default_pos = cube.data.default_nodal_state_w[env_id, :, :3]
    pts_surface = default_pos.index_select(0, surface_idx)
    xy = pts_surface[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = torch.min(x), torch.max(x)
    ymin, ymax = torch.min(y), torch.max(y)
    targets = torch.stack([
        torch.tensor([xmin, ymin], dtype=xy.dtype, device=xy.device),  # BL
        torch.tensor([xmin, ymax], dtype=xy.dtype, device=xy.device),  # TL
        torch.tensor([xmax, ymin], dtype=xy.dtype, device=xy.device),  # BR
        torch.tensor([xmax, ymax], dtype=xy.dtype, device=xy.device),  # TR
    ], dim=0).round(decimals=2)
    diffs = xy[:, None, :] - targets[None, :, :]
    dists = torch.sum(diffs * diffs, dim=2)
    chosen_local = []
    for j in range(4):
        order = torch.argsort(dists[:, j])
        pick = None
        for cand in order.tolist():
            if cand not in chosen_local:
                pick = cand; break
        if pick is None:
            pick = order[0].item()
        chosen_local.append(pick)
    chosen_local = torch.tensor(chosen_local, dtype=torch.long, device=surface_idx.device)
    return surface_idx.index_select(0, chosen_local)

# def build_top_surface_index_grid(cube, z_target, decimals=DECIMALS):
    
#     # import ipdb; ipdb.set_trace()
    
#     top_idx = get_surface_indices_by_known_z(cube, z_target=z_target, decimals=decimals)
#     default_pos = cube.data.default_nodal_state_w[..., :3][0]
#     top_pos = default_pos[top_idx]
#     s = 10**decimals
#     x_round = torch.round(top_pos[:, 0] * s) / s
#     y_round = torch.round(top_pos[:, 1] * s) / s
#     xs = torch.sort(torch.unique(x_round)).values
#     ys = torch.sort(torch.unique(y_round)).values
#     x_to_col = {float(v): i for i, v in enumerate(xs)}
#     y_to_row = {float(v): i for i, v in enumerate(ys)}
#     H, W = ys.numel(), xs.numel()
#     index_grid = torch.full((H, W), -1, dtype=torch.long, device=top_idx.device)
#     for n in range(top_idx.numel()):
#         r = y_to_row[float(y_round[n])]
#         c = x_to_col[float(x_round[n])]
#         index_grid[r, c] = top_idx[n]
#     assert (index_grid >= 0).all(), "Index grid has holes; adjust DECIMALS."
#     return index_grid
def build_top_surface_index_grid(cube, z_target, decimals=DECIMALS):
    """
    Robust: builds an HxW grid of top-surface node indices by anchoring integer bins
    to the min x/y. Avoids phantom bins caused by float rounding.
    """
    # pick top nodes from DEFAULT (rest) state
    top_idx = get_surface_indices_by_known_z(cube, z_target=z_target, decimals=decimals)
    default_pos = cube.data.default_nodal_state_w[..., :3][0]   # (N,3)
    top_pos = default_pos[top_idx]                               # (K,3)

    # anchored integer bins
    s = 10 ** decimals
    x = top_pos[:, 0]
    y = top_pos[:, 1]
    x0 = torch.min(x)
    y0 = torch.min(y)

    # integer bin IDs relative to minima
    x_bin = torch.round((x - x0) * s).to(torch.long)
    y_bin = torch.round((y - y0) * s).to(torch.long)

    # sorted unique bin IDs (integers -> no FP drift)
    xs_idx = torch.sort(torch.unique(x_bin)).values  # columns
    ys_idx = torch.sort(torch.unique(y_bin)).values  # rows
    W = xs_idx.numel()
    H = ys_idx.numel()

    # map bin id -> col/row (ints, not floats)
    x_to_col = {int(v): i for i, v in enumerate(xs_idx.tolist())}
    y_to_row = {int(v): i for i, v in enumerate(ys_idx.tolist())}

    index_grid = torch.full((H, W), -1, dtype=torch.long, device=top_idx.device)

    # tie-breaker: if multiple points quantize to same (r,c), pick the one
    # closest to the cell center so we fill deterministically.
    x_center_for_pt = x0 + torch.tensor([x_to_col[int(v)] for v in x_bin.tolist()],
                                        device=x.device, dtype=x.dtype) / s
    y_center_for_pt = y0 + torch.tensor([y_to_row[int(v)] for v in y_bin.tolist()],
                                        device=y.device, dtype=y.dtype) / s
    tie_score = (torch.abs(x - x_center_for_pt) + torch.abs(y - y_center_for_pt))
    order = torch.argsort(tie_score)

    used = torch.zeros((H, W), dtype=torch.bool, device=top_idx.device)
    for k in order.tolist():
        r = y_to_row[int(y_bin[k].item())]
        c = x_to_col[int(x_bin[k].item())]
        if not used[r, c]:
            index_grid[r, c] = top_idx[k]
            used[r, c] = True

    holes = (index_grid < 0).sum().item()
    if holes:
        raise AssertionError(
            f"Index grid has {holes} unfilled cells (H={H}, W={W}, points={top_idx.numel()}). "
            f"Try decimals=1 or 3 if your mesh has more/less jitter."
        )
    return index_grid


# --------------------------
# Simulation loop (keyboard)
# --------------------------
def run_simulator(sim: sim_utils.SimulationContext, entities, ros_node: Node, keyctl: KeyControl):
    """
    - Publishes per-cell top-surface z & xyz (collision mesh, legacy topics)
    - Publishes SIM-mesh dz (vertical), |u| displacement magnitude, and area-weighted pressure (per row/col topic)
    - Drives the bed's prismatic joints via keyboard and ROS
    - Optimized to avoid GPU<->CPU thrash and Python loops
    """

    # -------- helpers (vectorized pressure path) --------
    def precompute_top_surface_tris_arrays(T_valid: np.ndarray, index_grid: np.ndarray):
        """
        Build flat arrays for ALL top-surface triangles once (on reset):
          returns (tri_nodes[K,3], tri_elem[K], cell_ids[K], Hm, Wm)
        where Hm=H-1, Wm=W-1, and cell_ids in [0, Hm*Wm).
        """
        H, W = index_grid.shape
        Hm, Wm = H - 1, W - 1
        node_to_rc = {int(index_grid[r, c]): (r, c) for r in range(H) for c in range(W)}
        top_nodes = set(int(v) for v in index_grid.ravel().tolist())
        local_faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int64)

        tri_nodes = []
        tri_elem  = []
        cell_ids  = []

        for ei, tet in enumerate(T_valid):
            for f in local_faces:
                tri = tet[f]
                a, b, d = int(tri[0]), int(tri[1]), int(tri[2])
                if (a in top_nodes) and (b in top_nodes) and (d in top_nodes):
                    rs, cs = [], []
                    for v in (a, b, d):
                        r, c = node_to_rc[v]
                        rs.append(r); cs.append(c)
                    r0 = min(rs); c0 = min(cs)
                    if r0 < Hm and c0 < Wm:
                        tri_nodes.append([a, b, d])
                        tri_elem.append(ei)               # elem idx inside T_valid
                        cell_ids.append(r0 * Wm + c0)

        if not tri_nodes:
            return (np.zeros((0,3), np.int64),
                    np.zeros((0,),  np.int64),
                    np.zeros((0,),  np.int64),
                    Hm, Wm)

        return (np.asarray(tri_nodes, dtype=np.int64),
                np.asarray(tri_elem,  dtype=np.int64),
                np.asarray(cell_ids,  dtype=np.int64),
                Hm, Wm)

    def compute_pressure_grid_torch(
        tri_nodes_t: torch.Tensor,   # (K,3) long on device
        tri_elem_t:  torch.Tensor,   # (K,)  long on device (index into S_valid)
        cell_ids_t:  torch.Tensor,   # (K,)  long on device in [0, Hm*Wm)
        S_valid_t:   torch.Tensor,   # (Ne_valid, 3, 3) float on device
        sim_pos_t:   torch.Tensor,   # (Nv, 3) float on device
        Hm: int, Wm: int
    ) -> torch.Tensor:
        """Area-weighted traction p = max(0, -nᵀ σ n) with n oriented up. Returns (Hm,Wm)."""
        if tri_nodes_t.numel() == 0:
            return torch.zeros((Hm, Wm), dtype=sim_pos_t.dtype, device=sim_pos_t.device)

        v0 = sim_pos_t.index_select(0, tri_nodes_t[:, 0])
        v1 = sim_pos_t.index_select(0, tri_nodes_t[:, 1])
        v2 = sim_pos_t.index_select(0, tri_nodes_t[:, 2])

        n = torch.cross(v1 - v0, v2 - v0, dim=1)       # (K,3)
        area = 0.5 * torch.linalg.norm(n, dim=1)       # (K,)
        safe = area > 1e-12
        if not torch.any(safe):
            return torch.zeros((Hm, Wm), dtype=sim_pos_t.dtype, device=sim_pos_t.device)

        n = n[safe]
        area = area[safe]
        tri_elem = tri_elem_t[safe]
        cell_ids = cell_ids_t[safe]

        n_unit = n / (2.0 * area).unsqueeze(1)         # |cross| = 2A
        flip = n_unit[:, 2] < 0
        n_unit[flip] = -n_unit[flip]

        sigma = S_valid_t.index_select(0, tri_elem)    # (K,3,3)
        p = torch.einsum('bi,bij,bj->b', n_unit, sigma, n_unit)
        p = torch.clamp(-p, min=0.0)                   # compression positive

        num = torch.zeros(Hm * Wm, dtype=sim_pos_t.dtype, device=sim_pos_t.device)
        den = torch.zeros(Hm * Wm, dtype=sim_pos_t.dtype, device=sim_pos_t.device)
        num.scatter_add_(0, cell_ids, p * area)
        den.scatter_add_(0, cell_ids, area)
        den = torch.where(den > 0, den, torch.ones_like(den))
        return (num / den).reshape(Hm, Wm)

    # -------- scene handles --------
    bed   = entities["bed"]
    cells = entities["cells"]
    rows = max(1, int(args_cli.num_rows))
    cols = max(1, int(args_cli.num_cols))

    # -------- per-cell pubs & caches --------
    pubs_z_xyz = []          # [(pub_z, pub_xyz)]
    pubs_disp  = []          # [(pub_dz_rc, pub_umag_rc)]
    pubs_press = []          # [pub_pressure_rc]

    bottom_corner_idx = []
    kin_buffers       = []   # reuse kinematic target tensors per cell

    # Collision mesh (legacy z/xyz)
    index_grids_colmesh = []
    HW_colmesh = []

    # SIM mesh (displacement + pressure)
    index_grids_sim  = []    # numpy (Hs, Ws) of SIM node indices
    HW_sim           = []    # (Hs, Ws)
    sim_pos0_list    = []    # torch (Nv,3) on device
    valid_mask_list  = []    # torch bool (Ne,)
    tri_maps_list    = []    # (tri_nodes_t, tri_elem_t, cell_ids_t)
    HW_press_list    = []    # (Hm, Wm)

    for ent in cells:
        i = ent["node_index"]; r = ent["row"]; c = ent["col"]
        pub_z   = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/z_grid_{i}", 10)
        pub_xyz = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/xyz_{i}", 10)
        pubs_z_xyz.append((pub_z, pub_xyz))

        pub_dz_rc   = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/dz_grid_{r}_{c}", 10)
        pub_umag_rc = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/disp_mag_grid_{r}_{c}", 10)
        pubs_disp.append((pub_dz_rc, pub_umag_rc))

        pub_pressure_rc = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/pressure_grid_{r}_{c}", 10)
        pubs_press.append(pub_pressure_rc)

        bottom_corner_idx.append(None)
        kin_buffers.append(None)

        index_grids_colmesh.append(None); HW_colmesh.append((0,0))
        index_grids_sim.append(None);     HW_sim.append((0,0))
        sim_pos0_list.append(None);       valid_mask_list.append(None)
        tri_maps_list.append(None);       HW_press_list.append((0,0))

    # -------- bed joint state & ROS control --------
    pris_ids = []
    name_to_id = {}
    q_cmd = q_min = q_max = None

    class _RosCmd:
        def __init__(self): self.val = None; self.mask = None
    ros_cmd = _RosCmd()

    def _cb_joint_pos_target(msg: JointState):
        if ros_cmd.val is None or ros_cmd.mask is None: return
        for nm, pos in zip(msg.name, msg.position):
            j = name_to_id.get(nm)
            if j is not None and j < ros_cmd.val.shape[0]:
                ros_cmd.val[j]  = float(pos)
                ros_cmd.mask[j] = True

    def _cb_cell_grid_pos_target(msg: Float32MultiArray):
        if ros_cmd.val is None or ros_cmd.mask is None: return
        data = np.array(msg.data, dtype=float)
        if data.size == 0: return
        try:
            dims = msg.layout.dim
            if len(dims) >= 2:
                R = int(dims[0].size); C = int(dims[1].size)
            else:
                R, C = rows, cols
        except Exception:
            R, C = rows, cols
        try:
            grid = data.reshape(R, C)
        except Exception:
            rc = rows * cols
            flat = np.zeros(rc, dtype=float)
            flat[: min(rc, data.size)] = data[: min(rc, data.size)]
            grid = flat.reshape(rows, cols)
        for rr in range(min(rows, grid.shape[0])):
            for cc in range(min(cols, grid.shape[1])):
                idx_cell = rr * cols + cc
                jname = f"PrismaticJoint{idx_cell}"
                j = name_to_id.get(jname)
                if j is not None and j < ros_cmd.val.shape[0]:
                    ros_cmd.val[j]  = float(grid[rr, cc])
                    ros_cmd.mask[j] = True

    ros_node.create_subscription(JointState,        "/korusbed/joint_pos_target",     _cb_joint_pos_target,    10)
    ros_node.create_subscription(Float32MultiArray, "/korusbed/cell_grid/pos_target", _cb_cell_grid_pos_target,10)

    # -------- timing --------
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # -------- main loop --------
    while simulation_app.is_running():

        # ------- periodic reset (also runs at start) -------
        if count % 400 == 0:
            sim_time = 0.0
            count    = 0

            # reset bed
            root_state_bed = bed.data.default_root_state.clone()
            bed.write_root_pose_to_sim(root_state_bed[:, :7])
            bed.write_root_velocity_to_sim(root_state_bed[:, 7:])
            jpos = bed.data.default_joint_pos.clone()
            jvel = bed.data.default_joint_vel.clone()
            bed.write_joint_state_to_sim(jpos, jvel)
            bed.reset()

            # reset cells
            for ent in cells:
                base   = ent["base"]
                sphere = ent["sphere"]
                cube   = ent["cube"]

                base_pose = base.data.default_root_state.clone()
                base.write_root_pose_to_sim(base_pose[:, :7])
                base.write_root_velocity_to_sim(base_pose[:, 7:])

                cube_state = cube.data.default_nodal_state_w.clone()
                cube.write_nodal_state_to_sim(cube_state)

                sphere_pose = sphere.data.default_root_state.clone()
                sphere.write_root_pose_to_sim(sphere_pose[:, :7])
                sphere.write_root_velocity_to_sim(sphere_pose[:, 7:])

                base.reset(); cube.reset(); sphere.reset()

            # discover joints & limits
            try:
                jnames = getattr(bed.data, "joint_names", None) or []
                name_to_id = {nm: i for i, nm in enumerate(jnames) if isinstance(nm, str)}
                pris_ids = [i for i, nm in enumerate(jnames) if isinstance(nm, str) and nm.startswith("PrismaticJoint")]
            except Exception:
                name_to_id = {}; pris_ids = []

            B_bed, DoF = bed.data.joint_pos.shape
            lo = hi = None
            try:
                if hasattr(bed.data, "joint_pos_limits"):
                    lims = bed.data.joint_pos_limits
                    if lims.ndim == 3 and lims.shape[-1] == 2: lo, hi = lims[..., 0], lims[..., 1]
                elif hasattr(bed.data, "joint_limits"):
                    lims = bed.data.joint_limits
                    if lims.ndim == 3 and lims.shape[-1] == 2: lo, hi = lims[..., 0], lims[..., 1]
            except Exception:
                lo = hi = None
            if lo is None or hi is None:
                lo = torch.full((B_bed, DoF), -0.05, device=bed.data.joint_pos.device)
                hi = torch.full((B_bed, DoF),  +0.05, device=bed.data.joint_pos.device)

            q_now = bed.data.joint_pos.clone()
            q_cmd = q_now.clone()
            q_min = lo.clone()
            q_max = hi.clone()
            if pris_ids:
                pid_t = torch.tensor(pris_ids, device=q_now.device, dtype=torch.long)
                q_cmd[:, pid_t] = torch.clamp(q_now[:, pid_t], q_min[:, pid_t], q_max[:, pid_t])

            # ROS buffers
            ros_cmd.val  = np.array(q_cmd[0].detach().cpu().numpy(), dtype=float)
            ros_cmd.mask = np.zeros(DoF, dtype=bool)

            # per-cell mappings & pins
            for n, ent in enumerate(cells):
                r, c = ent["row"], ent["col"]
                cube = ent["cube"]
                dev  = cube.data.nodal_pos_w.device

                # collision-mesh grid (legacy z/xyz)
                z_bot_default, z_top_default = get_z_planes_from_default(cube, decimals=DECIMALS)
                top_idx = get_surface_indices_by_known_z(cube, z_target=z_top_default, decimals=DECIMALS)
                bot_idx = get_surface_indices_by_known_z(cube, z_target=z_bot_default, decimals=DECIMALS)
                if top_idx.numel() == 0 or bot_idx.numel() == 0:
                    raise RuntimeError(f"[Cell {n}] Could not find top/bottom collision-surface nodes.")
                
                grid_col = build_top_surface_index_grid(cube, z_target=z_top_default, decimals=1)
                Hc, Wc = grid_col.shape
                index_grids_colmesh[n] = grid_col
                HW_colmesh[n] = (Hc, Wc)

                # bottom-corner pins
                bot_corners = get_surface_corner_indices_from_idx(cube, bot_idx, env_id=0)
                bottom_corner_idx[n] = bot_corners
                # persistent kin buffer
                kin = cube.data.nodal_kinematic_target.clone()
                kin[..., :3] = cube.data.nodal_pos_w
                kin[..., 3]  = 1.0
                kin[:, bot_corners, 3] = 0.0
                cube.write_nodal_kinematic_target_to_sim(kin)
                kin_buffers[n] = kin

                # SIM-mesh grid (top nodes laid out as Hs×Ws)
                grid_sim, sim_pos0_np = build_top_surface_index_grid_from_sim(cube, decimals=1)
                Hs, Ws = grid_sim.shape
                index_grids_sim[n] = grid_sim
                HW_sim[n] = (Hs, Ws)

                # cache sim_pos0 on device
                sim_pos0_t = torch.as_tensor(sim_pos0_np, device=dev, dtype=torch.float32)
                sim_pos0_list[n] = sim_pos0_t

                # valid tets
                T_raw_any = cube.root_physx_view.get_sim_element_indices()[0]
                if isinstance(T_raw_any, np.ndarray):
                    T_raw_np = T_raw_any
                else:
                    T_raw_np = T_raw_any.detach().cpu().numpy()
                valid_mask_np = (T_raw_np >= 0).all(axis=1)
                valid_mask_t  = torch.as_tensor(valid_mask_np, device=dev, dtype=torch.bool)
                valid_mask_list[n] = valid_mask_t
                T_valid = T_raw_np[valid_mask_np]

                # precompute triangle mapping for pressure
                tri_nodes_np, tri_elem_np, cell_ids_np, Hm, Wm = precompute_top_surface_tris_arrays(T_valid, grid_sim)
                tri_nodes_t = torch.as_tensor(tri_nodes_np, device=dev, dtype=torch.long)
                tri_elem_t  = torch.as_tensor(tri_elem_np,  device=dev, dtype=torch.long)
                cell_ids_t  = torch.as_tensor(cell_ids_np,  device=dev, dtype=torch.long)
                tri_maps_list[n] = (tri_nodes_t, tri_elem_t, cell_ids_t)
                HW_press_list[n] = (Hm, Wm)

            print(f"[INFO] Reset bed + {len(cells)} cells; prismatic joints = {len(pris_ids)}")

        # ------- step & update -------
        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        bed.update(sim_dt)
        for ent in cells:
            ent["base"].update(sim_dt)
            ent["sphere"].update(sim_dt)
            ent["cube"].update(sim_dt)

        # ------- apply ROS targets (absolute) -------
        if q_cmd is not None and ros_cmd.val is not None:
            idxs = np.nonzero(ros_cmd.mask)[0].tolist()
            if idxs:
                idxt = torch.tensor(idxs, device=q_cmd.device, dtype=torch.long)
                vals = torch.tensor(ros_cmd.val[idxs], device=q_cmd.device, dtype=q_cmd.dtype)
                q_cmd[:, idxt] = torch.clamp(vals, q_min[:, idxt], q_max[:, idxt])
                ros_cmd.mask[idxs] = False

        # ------- keyboard global delta -------
        if q_cmd is not None and pris_ids:
            pid_t = torch.tensor(pris_ids, device=q_cmd.device, dtype=torch.long)
            delta = (keyctl.step if keyctl.inc else 0.0) - (keyctl.step if keyctl.dec else 0.0)
            if delta != 0.0:
                q_cmd[:, pid_t] = torch.clamp(q_cmd[:, pid_t] + delta, q_min[:, pid_t], q_max[:, pid_t])
        if q_cmd is not None:
            bed.set_joint_position_target(q_cmd)

        # ------- per-cell pins + publishes -------
        for n, ent in enumerate(cells):
            r, c   = ent["row"], ent["col"]
            cube   = ent["cube"]
            base   = ent["base"]
            dev    = cube.data.nodal_pos_w.device

            # keep bottom corners pinned (reuse kin buffer)
            root_pos  = np.round(base.data.root_pos_w.detach().cpu().numpy(), 3)
            root_quat = np.round(base.data.root_quat_w.detach().cpu().numpy(), 3)
            pins_dict = get_pin_targets_on_rigid_body(root_pos, root_quat)
            B_here, _, _ = cube.data.nodal_pos_w.shape
            targets_b = torch.tensor(
                [pins_dict[f"cuboid_{b}"] for b in range(B_here)],
                dtype=cube.data.nodal_kinematic_target.dtype,
                device=cube.data.nodal_kinematic_target.device,
            )
            kin = kin_buffers[n]
            kin[:, bottom_corner_idx[n], :3] = targets_b
            cube.write_nodal_kinematic_target_to_sim(kin)

            # push writes
            cube.write_data_to_sim()

            # ---- legacy collision-mesh z/xyz ----
            grid_col = index_grids_colmesh[n]
            if grid_col is not None and (count % args_cli.every == 0):
                Hc, Wc = HW_colmesh[n]
                pos = cube.data.nodal_pos_w[0]
                z_grid = pos[grid_col.reshape(-1), 2].reshape(Hc, Wc).detach().cpu().numpy()
                xyz    = pos[grid_col.reshape(-1)].detach().cpu().numpy()
                pub_z, pub_xyz = pubs_z_xyz[n]
                pub_z.publish(make_multiarray_2d(z_grid))
                pub_xyz.publish(make_multiarray_xyz(xyz))

            # ---- SIM-mesh displacement + pressure (vectorized on device) ----
            grid_sim = index_grids_sim[n]
            if grid_sim is not None and (count % args_cli.every == 0):
                Hs, Ws = HW_sim[n]
                idx_flat_t = torch.as_tensor(grid_sim.reshape(-1), device=dev, dtype=torch.long)

                # positions & stresses (keep on device)
                sim_pos_any = cube.root_physx_view.get_sim_nodal_positions()[0]
                S_any       = cube.root_physx_view.get_sim_element_stresses()[0]

                sim_pos_t = sim_pos_any.detach().to(dev) if hasattr(sim_pos_any, "detach") else \
                            torch.as_tensor(sim_pos_any, device=dev, dtype=torch.float32)
                S_t = S_any.detach().to(dev) if hasattr(S_any, "detach") else \
                      torch.as_tensor(S_any, device=dev, dtype=torch.float32)

                # valid elements and reshape to (Ne_valid,3,3)
                valid_mask_t = valid_mask_list[n]
                S_valid_t = S_t[valid_mask_t]
                if S_valid_t.ndim == 2 and S_valid_t.size(-1) == 9:
                    S_valid_t = S_valid_t.view(-1, 3, 3)

                # dz and |u|
                sim_pos0_t = sim_pos0_list[n]
                z_now  = sim_pos_t.index_select(0, idx_flat_t)[:, 2].reshape(Hs, Ws)
                z_ref  = sim_pos0_t.index_select(0, idx_flat_t)[:, 2].reshape(Hs, Ws)
                dz_grid = (z_now - z_ref).detach().cpu().numpy()

                u = (sim_pos_t.index_select(0, idx_flat_t) - sim_pos0_t.index_select(0, idx_flat_t))
                u_mag_grid = torch.linalg.norm(u, dim=1).reshape(Hs, Ws).detach().cpu().numpy()

                pub_dz_rc, pub_umag_rc = pubs_disp[n]
                pub_dz_rc.publish(make_multiarray_2d(dz_grid))
                pub_umag_rc.publish(make_multiarray_2d(u_mag_grid))

                # pressure (Hm x Wm)
                tri_nodes_t, tri_elem_t, cell_ids_t = tri_maps_list[n]
                Hm, Wm = HW_press_list[n]
                P_t = compute_pressure_grid_torch(tri_nodes_t, tri_elem_t, cell_ids_t, S_valid_t, sim_pos_t, Hm, Wm)
                pubs_press[n].publish(make_multiarray_2d(P_t.detach().cpu().numpy()))

        # write articulation after updating q_cmd
        bed.write_data_to_sim()

        # spin ROS (non-blocking)
        if count % args_cli.every == 0:
            rclpy.spin_once(ros_node, timeout_sec=0.0)

# --------
# Main ---
# --------
def main():
    # --- ROS2 ---
    rclpy.init(args=None)
    ros_node = rclpy.create_node("foam_bed_publisher")

    # --- Keyboard ---
    keyctl = KeyControl(step=0.003)
    keyctl.subscribe()

    # --- Sim ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.1, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=(3.0, -3.0, 2.0), target=(0.0, 0.0, 0.5))
    entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete... (Up/W=inflate, Down/S=deflate, Space=stop)")

    try:
        run_simulator(sim, entities, ros_node, keyctl)
    finally:
        keyctl.unsubscribe()
        ros_node.destroy_node()
        rclpy.shutdown()

# --------------
# --- Runner ---
# --------------
if __name__ == "__main__":
    main()
    simulation_app.close()
