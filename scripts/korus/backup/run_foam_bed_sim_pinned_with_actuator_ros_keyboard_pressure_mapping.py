"""
Usage:
    $ cd ~/IsaacLab
    $ preload ./isaaclab.sh -p scripts/korus/run_foam_bed_sim_pinned_with_actuator_ros.py
"""

# --------------------------------
# Imports for the App Creation ---
# --------------------------------
import argparse
import torch
from isaaclab.app import AppLauncher
import numpy as np
from collections import Counter, defaultdict

# -------------------------------------------
# Launch the IsaacSim App (with argparse) ---
# -------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--log_path", type=str, default="nodal_positions.txt")
parser.add_argument("--every", type=int, default=1, help="Log every N frames to reduce file size")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experience = "isaacsim.exp.full.kit"  # keyboard + ROS need full kit
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------
# Imports after app creation ---
# ------------------------------
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

# -------- Keyboard ----------
import omni.appwindow
import carb
from carb.input import KeyboardEventType, KeyboardInput

# ----------------
# ROS2 Imports ---
# ----------------
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

# --------------------
# Global Variables ---
# --------------------
ENV_USD             = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
INFLATABLE_CELL_USD = "file:///home/rutwik/IsaacLab/scripts/korus/assets/inflatable_cell.usd"

GROUND_TO_BASE_BOTTOM = 0.9
BASE_SIZE   = (1.0, 1.0, 0.1)
FOAM_SIZE   = (1.0, 1.0, 0.2)
SPHERE_RADIUS = 0.2

BASE_ORIGIN   = (0.0, 0.0, GROUND_TO_BASE_BOTTOM + round(BASE_SIZE[2]/2,2))
FOAM_ORIGIN   = (0.0, 0.0, BASE_ORIGIN[2] + round(BASE_SIZE[2]/2,2) + round(FOAM_SIZE[2]/2,2))
SPHERE_ORIGIN = (.0, .0, 5.0)
DECIMALS      = 1

NODE_NUM = 0

# =========================
# ---- Key Controller -----
# =========================
class KeyControl:
    """Hold/press to move the prismatic joint:
       Up/W = inflate (+), Down/S = deflate (-), Space = stop.
    """
    def __init__(self, step: float = 0.002):
        self.step = float(step)
        self.inc  = False
        self.dec  = False
        self._sub_id = None
        self._input = None
        self._keyboard = None

    def subscribe(self):
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            carb.log_warn("No app window; keyboard control disabled.")
            return
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub_id = self._input.subscribe_to_keyboard_events(self._keyboard, self.on_keyboard_input)

    def unsubscribe(self):
        try:
            if self._input and self._keyboard and self._sub_id:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_id)
        finally:
            self._sub_id = None
            self._keyboard = None
            self._input = None

    def on_keyboard_input(self, e):
        if e.input in (KeyboardInput.W, KeyboardInput.UP):
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.inc = True
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.inc = False
        elif e.input in (KeyboardInput.S, KeyboardInput.DOWN):
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.dec = True
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.dec = False
        elif e.input == KeyboardInput.SPACE:
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.inc = False
                self.dec = False

# ============================================================
# --------- Scene (USD + deformable + rigid sphere) ----------
# ============================================================
def design_scene():
    scene_context = omni.usd.get_context()
    scene_context.open_stage(ENV_USD)
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    dome.func("/World/DomeLight", dome)
    prim_utils.create_prim(f"/World/Node{NODE_NUM}", "Xform")
    
    # Inflatable articulation
    inflatable_asset = sim_utils.UsdFileCfg(usd_path=INFLATABLE_CELL_USD)
    inflatable_asset.func(f"/World/Node{NODE_NUM}/InflatableCell{NODE_NUM}", inflatable_asset)
    inflatable_cell_cfg = ArticulationCfg(
        class_type=Articulation,
        prim_path=f"/World/Node{NODE_NUM}/InflatableCell{NODE_NUM}",
        spawn=None,  # adopt existing prims
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0,0,0), rot=(1,0,0,0),
            joint_pos={f"PrismaticJoint": 0.65}, joint_vel={".*": 0.0},
        ),
        actuators={
            "linear_pd": ImplicitActuatorCfg(
                joint_names_expr=[f"PrismaticJoint"],
                effort_limit_sim=4000.0,
                stiffness=30000.0, damping=1200.0,
            ),
        },
    )
    inflatable_cell_object = Articulation(cfg=inflatable_cell_cfg)

    # Base (rigid)
    base_cfg = RigidObjectCfg(
        prim_path=f"/World/Node{NODE_NUM}/BasePlate{NODE_NUM}",
        spawn=sim_utils.MeshCuboidCfg(
            size=BASE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BASE_ORIGIN),
    )
    base_object = RigidObject(cfg=base_cfg)

    # Foam (deformable)
    deform_cfg = DeformableObjectCfg(
        prim_path=f"/World/Node{NODE_NUM}/DeformableCuboid{NODE_NUM}",
        spawn=sim_utils.MeshCuboidCfg(
            size=FOAM_SIZE,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=10
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
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=deform_cfg)

    # Sphere (rigid)
    sphere_cfg = RigidObjectCfg(
        prim_path=f"/World/Node{NODE_NUM}/RigidSphere{NODE_NUM}",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SPHERE_ORIGIN),
    )
    sphere_object = RigidObject(cfg=sphere_cfg)

    return {
        "cube_object": cube_object,
        "sphere_object": sphere_object,
        "base_object": base_object,
        "inflatable_cell_object": inflatable_cell_object,
    }

# ======================================================
# ------- ROS packing (Float32MultiArray helpers) ------
# ======================================================
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

# ======================================================
# --------- Pin target generation for base plate -------
# ======================================================
def get_pin_targets_on_rigid_body(root_pos_w_rigid, root_quat_w_rigid):
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

# ======================================================
# ------ New: grid & pressure mapping utilities --------
# ======================================================
def _round_to_dec(x: np.ndarray, decimals: int):
    s = 10.0 ** decimals
    return np.round(x * s) / s

def build_top_surface_index_grid_from_sim(cube, decimals=DECIMALS):
    """Return HxW grid of SIM-MESH node indices for the *top* surface."""
    sim_pos = cube.root_physx_view.get_sim_nodal_positions()[0]
    sim_pos = sim_pos.detach().cpu().numpy()  # (Nv,3)

    z = sim_pos[:, 2]
    z_top = _round_to_dec(np.max(z), decimals)
    z_r = _round_to_dec(z, decimals)
    top_mask = (z_r == z_top)
    top_idx = np.nonzero(top_mask)[0]
    top_pos = sim_pos[top_idx]

    x_r = _round_to_dec(top_pos[:, 0], decimals)
    y_r = _round_to_dec(top_pos[:, 1], decimals)
    xs = np.unique(np.sort(x_r))
    ys = np.unique(np.sort(y_r))
    H, W = len(ys), len(xs)

    # map rounded x/y -> column/row
    x_to_c = {float(v): i for i, v in enumerate(xs)}
    y_to_r = {float(v): i for i, v in enumerate(ys)}

    index_grid = -np.ones((H, W), dtype=np.int64)
    for i, (xr, yr, vi) in enumerate(zip(x_r, y_r, top_idx)):
        r = y_to_r[float(yr)]
        c = x_to_c[float(xr)]
        index_grid[r, c] = int(vi)

    if (index_grid < 0).any():
        raise RuntimeError("Top surface index grid has holes; adjust DECIMALS or resolution.")

    # also return default sim positions for later displacement
    return index_grid, sim_pos

def precompute_top_surface_cell_tris(T_valid: np.ndarray, index_grid: np.ndarray):
    """Map each top-surface square cell (r,c) -> list of triangles (tet_id, tri_indices[3])."""
    H, W = index_grid.shape
    node_to_rc = {int(index_grid[r, c]): (r, c) for r in range(H) for c in range(W)}
    top_nodes = set(int(v) for v in index_grid.ravel().tolist())

    local_faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int64)

    cell_tris = defaultdict(list)  # (r,c) -> [(ti, tri[3]), ...]
    # Build boundary faces by filtering faces whose three verts are top nodes.
    for ti, tet in enumerate(T_valid):
        for f in local_faces:
            tri = tet[f]
            if int(tri[0]) in top_nodes and int(tri[1]) in top_nodes and int(tri[2]) in top_nodes:
                # Identify the owning square cell by min row/col of the triangle's nodes
                rs = [node_to_rc[int(v)][0] for v in tri]
                cs = [node_to_rc[int(v)][1] for v in tri]
                r0, c0 = min(rs), min(cs)
                if r0 < H-1 and c0 < W-1:  # inside cell grid
                    cell_tris[(r0, c0)].append((ti, tri.copy()))
    return cell_tris  # typically two tris per cell

def compute_pressure_grid(cell_tris, S_valid, sim_pos, H, W):
    """Area-weighted face pressure per cell from Cauchy stress (compression positive)."""
    P = np.zeros((H-1, W-1), dtype=np.float32)
    A = np.zeros((H-1, W-1), dtype=np.float32)
    for (r0, c0), tris in cell_tris.items():
        for (ti, tri) in tris:
            x0, x1, x2 = sim_pos[tri[0]], sim_pos[tri[1]], sim_pos[tri[2]]
            n = np.cross(x1 - x0, x2 - x0)
            area = 0.5 * np.linalg.norm(n)
            if area <= 1e-12:
                continue
            n_unit = n / (2.0 * area)  # |cross| = 2A
            # orient upwards for top surface
            if n_unit[2] < 0:
                n_unit = -n_unit
            sigma = S_valid[ti].reshape(3, 3)
            p = float(n_unit @ (sigma @ n_unit))  # traction normal
            p = max(0.0, -p)  # compression positive
            P[r0, c0] += p * area
            A[r0, c0] += area
    mask = A > 0
    P[mask] /= A[mask]
    return P

# ======================================================
# -------------- Simulation main loop ------------------
# ======================================================
def run_simulator(sim: sim_utils.SimulationContext, entities, ros_node: Node, keyctl: KeyControl):
    """
    - Pins the deformable cube bottom-corner vertices to the rigid base.
    - Publishes:
        * z_grid (H×W)
        * dz_grid (H×W)  [vertical nodal displacement]
        * disp_mag_grid (H×W)  [|u| per node]
        * pressure_grid ((H-1)×(W-1))  [per-square]
    - Drives the inflatable cell prismatic joint with keyboard.
    """
    cube            = entities["cube_object"]
    sphere          = entities["sphere_object"]
    base            = entities["base_object"]
    inflatable_cell = entities["inflatable_cell_object"]

    # --- ROS2 publishers ---
    pub_z        = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/z_grid_{NODE_NUM}", 10)
    pub_xyz      = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/xyz_{NODE_NUM}", 10)
    pub_dz       = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/dz_grid_{NODE_NUM}", 10)
    pub_umag     = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/disp_mag_grid_{NODE_NUM}", 10)
    pub_pressure = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/pressure_grid_{NODE_NUM}", 10)

    # --- Timing ---
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # --- Deformable buffers ---
    B, N, _ = cube.data.nodal_pos_w.shape
    nodal_kinematic_target = cube.data.nodal_kinematic_target.clone()

    # --- Vars set on reset ---
    bottom_corner_idx_torch = None
    prismatic_jid = 0

    # New mapping state
    index_grid = None         # (H,W) int64 of SIM nodes on top
    sim_pos0   = None         # default SIM nodal positions (Nv,3)
    H = W = 0
    # For pressure:
    valid_mask = None         # mask for valid tets
    T_valid    = None         # (Ne_valid, 4) sim tet indices
    cell_tris  = None         # mapping (r,c)->list of (ti, tri[3])

    # --- Joint command state ---
    q_cmd = None
    q_min = None
    q_max = None

    while simulation_app.is_running():
        # --------------------------- Reset block --------------------------
        if count % 1000 == 0:
            sim_time = 0.0
            count    = 0

            # Reset base
            root_state_base = base.data.default_root_state.clone()
            base.write_root_pose_to_sim(root_state_base[:, :7])
            base.write_root_velocity_to_sim(root_state_base[:, 7:])

            # Reset foam (collision + sim)
            nodal_state_cube = cube.data.default_nodal_state_w.clone()
            cube.write_nodal_state_to_sim(nodal_state_cube)

            # Reset sphere
            root_state_sphere = sphere.data.default_root_state.clone()
            sphere.write_root_pose_to_sim(root_state_sphere[:, :7])
            sphere.write_root_velocity_to_sim(root_state_sphere[:, 7:])

            # Reset inflatable articulation (root + joints)
            root_state_cell = inflatable_cell.data.default_root_state.clone()
            inflatable_cell.write_root_pose_to_sim(root_state_cell[:, :7])
            inflatable_cell.write_root_velocity_to_sim(root_state_cell[:, 7:])
            joint_pos = inflatable_cell.data.default_joint_pos.clone()
            joint_vel = inflatable_cell.data.default_joint_vel.clone()
            inflatable_cell.write_joint_state_to_sim(joint_pos, joint_vel)

            cube.reset(); sphere.reset(); base.reset(); inflatable_cell.reset()
            print("[INFO]: Resetting deformable and rigid object states")

            # ---------- Build top-surface grid from SIM mesh ----------
            index_grid, sim_pos0 = build_top_surface_index_grid_from_sim(cube, decimals=DECIMALS)
            H, W = index_grid.shape
            print(f"[INFO] Top surface (SIM) grid built: rows={H}, cols={W}, total={H*W}")

            # ---------- Precompute top-surface cell triangles ----------
            T_raw = cube.root_physx_view.get_sim_element_indices()[0].detach().cpu().numpy()  # (Ne_max,4)
            valid_mask = (T_raw >= 0).all(axis=1)
            T_valid = T_raw[valid_mask]
            cell_tris = precompute_top_surface_cell_tris(T_valid, index_grid)
            print(f"[INFO] Top surface cells with tris: {len(cell_tris)} (expect {(H-1)*(W-1)})")

            # ---------- Pin bottom corners ----------
            # Bottom surface/corners from *collision* mesh (stable & simple for pinning)
            z_bot_default = round(FOAM_ORIGIN[2] - FOAM_SIZE[2] / 2, 2)
            # find bottom surface on collision mesh
            default_pos = cube.data.default_nodal_state_w[0, :, :3]
            z = default_pos[:, 2]
            z_r = torch.round(z * (10**DECIMALS)) / (10**DECIMALS)
            mask_bot = (z_r == z_bot_default)
            bottom_surface_idx_torch = torch.nonzero(mask_bot, as_tuple=False).squeeze(1)
            # corners on bottom
            # (reuse robust corner-picker)
            def _corners_from(cube, surface_idx, env_id=0):
                default_pos = cube.data.default_nodal_state_w[env_id, :, :3]
                pts_surface = default_pos.index_select(0, surface_idx)
                xy = pts_surface[:, :2]
                x, y = xy[:, 0], xy[:, 1]
                xmin, xmax = torch.min(x), torch.max(x)
                ymin, ymax = torch.min(y), torch.max(y)
                targets = torch.stack([
                    torch.tensor([xmin, ymin], dtype=xy.dtype, device=xy.device),
                    torch.tensor([xmin, ymax], dtype=xy.dtype, device=xy.device),
                    torch.tensor([xmax, ymin], dtype=xy.dtype, device=xy.device),
                    torch.tensor([xmax, ymax], dtype=xy.dtype, device=xy.device),
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

            bottom_corner_idx_torch = _corners_from(cube, bottom_surface_idx_torch, env_id=0)

            # Start with all FREE; pin 4 bottom corners
            nodal_kinematic_target[..., :3] = cube.data.nodal_pos_w
            nodal_kinematic_target[..., 3]  = 1.0
            nodal_kinematic_target[:, bottom_corner_idx_torch, 3] = 0.0
            cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # ---------- Identify prismatic joint & limits ----------
            try:
                joint_names = getattr(inflatable_cell.data, "joint_names", None)
                if isinstance(joint_names, (list, tuple)) and f"PrismaticJoint" in joint_names:
                    prismatic_jid = joint_names.index(f"PrismaticJoint")
                else:
                    prismatic_jid = 0
            except Exception:
                prismatic_jid = 0

            device = inflatable_cell.data.joint_pos.device
            B_cell, DoF = inflatable_cell.data.joint_pos.shape
            lower_limits = upper_limits = None
            try:
                if hasattr(inflatable_cell.data, "joint_pos_limits"):
                    lims = inflatable_cell.data.joint_pos_limits
                    if lims.ndim == 3 and lims.shape[-1] == 2:
                        lower_limits = lims[..., 0]; upper_limits = lims[..., 1]
                elif hasattr(inflatable_cell.data, "joint_limits"):
                    lims = inflatable_cell.data.joint_limits
                    if lims.ndim == 3 and lims.shape[-1] == 2:
                        lower_limits = lims[..., 0]; upper_limits = lims[..., 1]
            except Exception:
                lower_limits = upper_limits = None
            if lower_limits is None or upper_limits is None:
                lower_limits = torch.full((B_cell, DoF), -0.05, device=device)
                upper_limits = torch.full((B_cell, DoF),  +0.05, device=device)

            q_now = inflatable_cell.data.joint_pos[:, prismatic_jid].clone()
            q_min = lower_limits[:, prismatic_jid]
            q_max = upper_limits[:, prismatic_jid]
            q_cmd = torch.clamp(q_now, q_min, q_max)

        # --------------------------- Step & update --------------------------
        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        cube.update(sim_dt); sphere.update(sim_dt); base.update(sim_dt); inflatable_cell.update(sim_dt)

        # --- Keep deformable bottom-corner nodes pinned to base top corners ---
        root_pos_w_rigid  = np.round(base.data.root_pos_w.detach().cpu().numpy(), 3)
        root_quat_w_rigid = np.round(base.data.root_quat_w.detach().cpu().numpy(), 3)
        pin_targets_on_rigid_body_dict = get_pin_targets_on_rigid_body(root_pos_w_rigid, root_quat_w_rigid)
        targets_b = torch.tensor(
            [pin_targets_on_rigid_body_dict[f"cuboid_{b}"] for b in range(B)],
            dtype=nodal_kinematic_target.dtype,
            device=nodal_kinematic_target.device,
        )
        nodal_kinematic_target[:, bottom_corner_idx_torch, :3] = targets_b
        cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # ---------------- Keyboard-driven inflate/deflate ----------------
        if q_cmd is not None:
            delta = 0.0
            if keyctl.inc: delta += keyctl.step
            if keyctl.dec: delta -= keyctl.step
            if delta != 0.0:
                q_cmd = torch.clamp(q_cmd + delta, q_min, q_max)
            q_des = inflatable_cell.data.joint_pos.clone()
            q_des[:, prismatic_jid] = q_cmd
            inflatable_cell.set_joint_position_target(q_des)

        # --- Push writes ---
        cube.write_data_to_sim()
        inflatable_cell.write_data_to_sim()

        # -------------------- Publish top-surface data --------------------
        if (index_grid is not None) and (count % args_cli.every == 0):
            # SIM mesh current positions
            sim_pos = cube.root_physx_view.get_sim_nodal_positions()[0].detach().cpu().numpy()  # (Nv,3)

            # --- Nodal height / displacement grids (H×W) ---
            idx_flat = index_grid.reshape(-1)
            z_grid = sim_pos[idx_flat, 2].reshape(H, W)
            dz_grid = (sim_pos[idx_flat, 2] - sim_pos0[idx_flat, 2]).reshape(H, W)
            u_mag_grid = np.linalg.norm(sim_pos[idx_flat] - sim_pos0[idx_flat], axis=1).reshape(H, W)

            pub_z.publish(make_multiarray_2d(z_grid))
            pub_dz.publish(make_multiarray_2d(dz_grid))
            pub_umag.publish(make_multiarray_2d(u_mag_grid))

            # XYZ list in grid order
            xyz = sim_pos[idx_flat].reshape(H*W, 3)
            pub_xyz.publish(make_multiarray_xyz(xyz))

            # --- Pressure per square ((H-1)×(W-1)) ---
            S_full = cube.root_physx_view.get_sim_element_stresses()[0].detach().cpu().numpy().reshape(-1, 3, 3)
            S_valid = S_full[valid_mask]  # align with T_valid indexing
            pressure_grid = compute_pressure_grid(cell_tris, S_valid, sim_pos, H, W)
            pub_pressure.publish(make_multiarray_2d(pressure_grid))

            rclpy.spin_once(ros_node, timeout_sec=0.0)

# --------
# Main ---
# --------
def main():
    # ROS2 init
    rclpy.init(args=None)
    ros_node = rclpy.create_node("foam_bed_publisher")

    # Keyboard (subscribe using omni.appwindow)
    keyctl = KeyControl(step=0.003)
    keyctl.subscribe()

    # Isaac
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, 0.0, 1.0), target=(0.0, 0.0, 0.5))

    entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete... (Up/W=inflate, Down/S=deflate, Space=stop)")

    try:
        run_simulator(sim, entities, ros_node, keyctl)
    finally:
        keyctl.unsubscribe()
        ros_node.destroy_node()
        rclpy.shutdown()

# --------------------
# Runner -------------
# --------------------
if __name__ == "__main__":
    main()
    simulation_app.close()
