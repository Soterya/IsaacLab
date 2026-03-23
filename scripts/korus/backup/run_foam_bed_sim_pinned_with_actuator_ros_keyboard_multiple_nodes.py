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

# -------------------------------------------
# Launch the IsaacSim App (with argparse) ---
# -------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--num_rows", type=int, default=1)
parser.add_argument("--num_cols", type=int, default=1)
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
BASE_SIZE   = (1., 1., 0.1)
FOAM_SIZE   = (1., 1., 0.2)
SPHERE_RADIUS = 0.2

BASE_ORIGIN   = (0.0, 0.0, GROUND_TO_BASE_BOTTOM + round(BASE_SIZE[2]/2,2))
FOAM_ORIGIN   = (0.0, 0.0, BASE_ORIGIN[2] + round(BASE_SIZE[2]/2,2) + round(FOAM_SIZE[2]/2,2))
SPHERE_ORIGIN = (.0, .0, 5.0)
DECIMALS      = 1

NODE_SPACING = 1.1 # spacing between Node Centers

class KeyControl:
    """Hold/press to move the prismatic joint:
       Up/W = inflate (+), Down/S = deflate (-), Space = stop.
    """
    def __init__(self, step: float = 0.002):
        self.step = float(step)
        self.inc  = False
        self.dec  = False
        self._sub_id   = None
        self._input    = None
        self._keyboard = None

    # --- omni.appwindow + carb.input subscription ---
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

    # --- event callback ---
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
                # quick stop
                self.inc = False
                self.dec = False


# ----------------------------------------------------------
# Manual Scene Builder (USD + deformable + rigid sphere) ---
# ----------------------------------------------------------
def design_scene():
    """
    Build an R x C grid of nodes.
      - Columns advance along +X by env_spacing
      - Rows advance along -Y by env_spacing
      - Each node gets unique prim paths and a flat node_index = r*cols + c
    """
    scene_context = omni.usd.get_context()
    scene_context.open_stage(ENV_USD)

    # spawn distant light
    cfg_light = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light.func("/World/Domelight", cfg_light)

    entities_list = []

    rows = max(1, int(args_cli.num_rows))
    cols = max(1, int(args_cli.num_cols))
    sx = float(NODE_SPACING)
    sy = float(NODE_SPACING)  # use same spacing for rows; change if you want different

    for r in range(rows):
        for c in range(cols):
            idx   = r * cols + c
            x_off = c * sx
            y_off = -r * sy  # rows go "down" along -Y

            # parent Xform for each node
            prim_utils.create_prim(f"/World/Node{idx}", "Xform")

            # --- Inflatable articulation (adopt USD under node) ---
            inflatable_asset = sim_utils.UsdFileCfg(usd_path=INFLATABLE_CELL_USD)
            inflatable_asset.func(f"/World/Node{idx}/InflatableCell{idx}", inflatable_asset)
            inflatable_cell_cfg = ArticulationCfg(
                class_type=Articulation,
                prim_path=f"/World/Node{idx}/InflatableCell{idx}",
                spawn=None,  # adopt existing prims
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(x_off, y_off, 0.0), rot=(1, 0, 0, 0),
                    joint_pos={"PrismaticJoint": 0.65}, joint_vel={".*": 0.0},
                ),
                actuators={
                    "linear_pd": ImplicitActuatorCfg(
                        joint_names_expr=["PrismaticJoint"],
                        effort_limit_sim=4000.0,
                        stiffness=30000.0,
                        damping=1200.0,
                    ),
                },
            )
            inflatable_cell_obj = Articulation(cfg=inflatable_cell_cfg)


            # --- Base (rigid) ---
            base_cfg = RigidObjectCfg(
                prim_path=f"/World/Node{idx}/BasePlate{idx}",
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

            # --- Foam (deformable) ---
            deform_cfg = DeformableObjectCfg(
                prim_path=f"/World/Node{idx}/DeformableCuboid{idx}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=FOAM_SIZE,
                    deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                        rest_offset=0.0,
                        contact_offset=0.001,
                        simulation_hexahedral_resolution=3
                    ),
                    physics_material=sim_utils.DeformableBodyMaterialCfg(
                        poissons_ratio=0.4, youngs_modulus=1e5
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

            # --- Sphere (rigid) ---
            sphere_cfg = RigidObjectCfg(
                prim_path=f"/World/Node{idx}/RigidSphere{idx}",
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

            entities_list.append({
                "cube": cube_obj,
                "sphere": sphere_obj,
                "base": base_obj,
                "cell": inflatable_cell_obj,
                "node_index": idx,  # used by run_simulator for ROS topic suffix
                "row": r,
                "col": c,
            })

    return entities_list

# ---------------------------------------------
# Utilities: indices / packing for ROS msgs ---
# ---------------------------------------------
def get_surface_indices_by_known_z(cube, z_target, decimals=DECIMALS, atol=None):
    """
    Return indices of nodes whose DEFAULT world-z is on the plane at z_target.
    If a direct match (rounded or atol) finds none, we fall back to the nearest
    rounded plane present in the mesh so multi-node rounding won't break.
    """
    default_pos = cube.data.default_nodal_state_w[..., :3][0]
    z = default_pos[:, 2]

    # First attempt: exact-by-rounding OR atol
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

    # Fallback: snap to the nearest existing rounded plane
    s = 10 ** (decimals if decimals is not None else 3)
    zr = torch.round(z * s) / s
    planes = torch.sort(torch.unique(zr)).values
    if planes.numel() == 0:
        raise RuntimeError("No nodes found in default state for deformable.")
    # pick plane with minimum absolute distance to target
    diffs = torch.abs(planes - float(z_target))
    nearest = planes[torch.argmin(diffs)]
    mask2 = (zr == nearest)
    idx2 = torch.nonzero(mask2, as_tuple=False).squeeze(1)
    if idx2.numel() == 0:
        # ultimate fallback: choose bottom/top by raw min/max
        zmin, zmax = torch.min(z), torch.max(z)
        pick = zmin if abs(float(z_target) - float(zmin)) < abs(float(z_target) - float(zmax)) else zmax
        mask3 = torch.isclose(z, torch.tensor(pick, device=z.device, dtype=z.dtype), atol=1e-6)
        idx3 = torch.nonzero(mask3, as_tuple=False).squeeze(1)
        if idx3.numel() == 0:
            raise RuntimeError("Could not find any nodes on a surface plane (even after fallback).")
        return idx3
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

def build_top_surface_index_grid(cube, z_target, decimals=DECIMALS):
    top_idx = get_surface_indices_by_known_z(cube, z_target=z_target, decimals=decimals)
    default_pos = cube.data.default_nodal_state_w[..., :3][0]
    top_pos = default_pos[top_idx]
    s = 10**decimals
    x_round = torch.round(top_pos[:, 0] * s) / s
    y_round = torch.round(top_pos[:, 1] * s) / s
    xs = torch.sort(torch.unique(x_round)).values
    ys = torch.sort(torch.unique(y_round)).values
    x_to_col = {float(v): i for i, v in enumerate(xs)}
    y_to_row = {float(v): i for i, v in enumerate(ys)}
    H, W = ys.numel(), xs.numel()
    index_grid = torch.full((H, W), -1, dtype=torch.long, device=top_idx.device)
    for n in range(top_idx.numel()):
        r = y_to_row[float(y_round[n])]
        c = x_to_col[float(x_round[n])]
        index_grid[r, c] = top_idx[n]
    assert (index_grid >= 0).all(), "Index grid has holes; adjust DECIMALS."
    return index_grid

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

# -----------------------------------------------
# Utility Functions : For Obtaining Positions --- 
# -----------------------------------------------    
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

# -------------------------------------------
# Simulation loop: keyboard-driven inflate --
# -------------------------------------------
def get_z_planes_from_default(cube, decimals=DECIMALS):
    """
    Inspect the deformable's default world positions and return (z_bot, z_top)
    using rounded planes at the given decimals.
    """
    default_pos = cube.data.default_nodal_state_w[..., :3][0]  # (N,3)
    z = default_pos[:, 2]
    s = 10 ** decimals
    zr = torch.round(z * s) / s
    planes = torch.sort(torch.unique(zr)).values
    if planes.numel() < 2:
        # fallback without rounding if mesh is degenerate
        return float(torch.min(z)), float(torch.max(z))
    return float(planes[0]), float(planes[-1])


def run_simulator(sim: sim_utils.SimulationContext, entities_list, ros_node: Node, keyctl: KeyControl):
    """
    Multi-node loop (NODE_COUNT in a row).
    Each node i has its own:
      - deformable, base, sphere, inflatable cell
      - ROS pubs: /foam_bed/top_surface/{z_grid,xyz}_i
      - kinematic pins & top-surface index grid
      - prismatic joint command/state
    Keyboard drives ALL nodes' prismatic joints together:
      Up/W -> inflate, Down/S -> deflate, Space -> stop
    """
    # --- Per-node state containers ---
    pubs = []             # [(pub_z, pub_xyz)]
    bottom_corner_idx = []
    index_grids = []      # per-node torch.LongTensor (H,W)
    HW = []               # per-node (H, W)
    pris_id = []          # prismatic joint index per node
    q_cmd = []            # commanded joint pos per node (torch)
    q_min = []            # lower limits per node (torch)
    q_max = []            # upper limits per node (torch)

    # Create ROS pubs per node
    for ent in entities_list:
        i = ent["node_index"]
        pub_z   = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/z_grid_{i}", 10)
        pub_xyz = ros_node.create_publisher(Float32MultiArray, f"/foam_bed/top_surface/xyz_{i}", 10)
        pubs.append((pub_z, pub_xyz))

        # placeholders (filled after first reset)
        bottom_corner_idx.append(None)
        index_grids.append(None)
        HW.append((0, 0))
        pris_id.append(0)
        q_cmd.append(None)
        q_min.append(None)
        q_max.append(None)

    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    while simulation_app.is_running():
        # ------------------ Reset all nodes together (interval) ------------------
        if count % 10000 == 0:
            sim_time = 0.0
            count    = 0

            for n, ent in enumerate(entities_list):
                cube   = ent["cube"]
                base   = ent["base"]
                sphere = ent["sphere"]
                cell   = ent["cell"]

                # --- reset states ---
                base_pose = base.data.default_root_state.clone()
                base.write_root_pose_to_sim(base_pose[:, :7])
                base.write_root_velocity_to_sim(base_pose[:, 7:])

                cube_state = cube.data.default_nodal_state_w.clone()
                cube.write_nodal_state_to_sim(cube_state)

                sphere_pose = sphere.data.default_root_state.clone()
                sphere.write_root_pose_to_sim(sphere_pose[:, :7])
                sphere.write_root_velocity_to_sim(sphere_pose[:, 7:])

                cell_pose = cell.data.default_root_state.clone()
                cell.write_root_pose_to_sim(cell_pose[:, :7])
                cell.write_root_velocity_to_sim(cell_pose[:, 7:])
                jpos = cell.data.default_joint_pos.clone()
                jvel = cell.data.default_joint_vel.clone()
                cell.write_joint_state_to_sim(jpos, jvel)

                cube.reset(); base.reset(); sphere.reset(); cell.reset()

                # --- discover the actual Z planes from this cube's default state ---
                z_bot_default, z_top_default = get_z_planes_from_default(cube, decimals=DECIMALS)

                # --- surfaces / corners / grid (robust to rounding) ---
                top_idx    = get_surface_indices_by_known_z(cube, z_target=z_top_default, decimals=DECIMALS)
                bot_idx    = get_surface_indices_by_known_z(cube, z_target=z_bot_default, decimals=DECIMALS)

                # Guard against unexpected empties (shouldn't happen with robust finder)
                if top_idx.numel() == 0 or bot_idx.numel() == 0:
                    raise RuntimeError(f"[Node {n}] Could not find top/bottom surface nodes.")

                bot_corners = get_surface_corner_indices_from_idx(cube, bot_idx, env_id=0)

                grid = build_top_surface_index_grid(cube, z_target=z_top_default, decimals=DECIMALS)
                H, W = grid.shape

                # --- kinematic target buffer (pin only bottom corners) ---
                kin = cube.data.nodal_kinematic_target.clone()
                kin[..., :3] = cube.data.nodal_pos_w
                kin[..., 3]  = 1.0
                kin[:, bot_corners, 3] = 0.0
                cube.write_nodal_kinematic_target_to_sim(kin)

                # --- prismatic joint id + limits ---
                try:
                    jnames = getattr(cell.data, "joint_names", None)
                    jid = jnames.index("PrismaticJoint") if (isinstance(jnames, (list, tuple)) and "PrismaticJoint" in jnames) else 0
                except Exception:
                    jid = 0

                device = cell.data.joint_pos.device
                B_cell, DoF = cell.data.joint_pos.shape
                lo = hi = None
                try:
                    if hasattr(cell.data, "joint_pos_limits"):
                        lims = cell.data.joint_pos_limits
                        if lims.ndim == 3 and lims.shape[-1] == 2:
                            lo, hi = lims[..., 0], lims[..., 1]
                    elif hasattr(cell.data, "joint_limits"):
                        lims = cell.data.joint_limits
                        if lims.ndim == 3 and lims.shape[-1] == 2:
                            lo, hi = lims[..., 0], lims[..., 1]
                except Exception:
                    lo = hi = None
                if lo is None or hi is None:
                    lo = torch.full((B_cell, DoF), -0.05, device=device)
                    hi = torch.full((B_cell, DoF),  +0.05, device=device)

                q_now = cell.data.joint_pos[:, jid].clone()
                q_cmd[n] = torch.clamp(q_now, lo[:, jid], hi[:, jid])
                q_min[n] = lo[:, jid]
                q_max[n] = hi[:, jid]
                pris_id[n] = jid
                bottom_corner_idx[n] = bot_corners
                index_grids[n] = grid
                HW[n] = (H, W)

            print(f"[INFO] Reset {len(entities_list)} nodes; built grids and pins.")

        # ---------------------- step & update all nodes ----------------------
        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        for ent in entities_list:
            ent["cube"].update(sim_dt)
            ent["base"].update(sim_dt)
            ent["sphere"].update(sim_dt)
            ent["cell"].update(sim_dt)

        # --------------- per-node pins, keyboard, writes, publish ---------------
        for n, ent in enumerate(entities_list):
            cube   = ent["cube"]
            base   = ent["base"]
            cell   = ent["cell"]

            # keep bottom-corner pins on base corners
            root_pos  = np.round(base.data.root_pos_w.detach().cpu().numpy(), 3)
            root_quat = np.round(base.data.root_quat_w.detach().cpu().numpy(), 3)
            pins_dict = get_pin_targets_on_rigid_body(root_pos, root_quat)

            B_here, _, _ = cube.data.nodal_pos_w.shape  # usually 1
            targets_b = torch.tensor(
                [pins_dict[f"cuboid_{b}"] for b in range(B_here)],
                dtype=cube.data.nodal_kinematic_target.dtype,
                device=cube.data.nodal_kinematic_target.device,
            )

            kin = cube.data.nodal_kinematic_target.clone()
            kin[:, bottom_corner_idx[n], :3] = targets_b
            cube.write_nodal_kinematic_target_to_sim(kin)

            # keyboard-driven inflate/deflate
            if q_cmd[n] is not None:
                delta = (keyctl.step if keyctl.inc else 0.0) - (keyctl.step if keyctl.dec else 0.0)
                if delta != 0.0:
                    q_cmd[n] = torch.clamp(q_cmd[n] + delta, q_min[n], q_max[n])
                q_des = cell.data.joint_pos.clone()
                q_des[:, pris_id[n]] = q_cmd[n]
                cell.set_joint_position_target(q_des)

            # push writes
            cube.write_data_to_sim()
            cell.write_data_to_sim()

            # publish top-surface
            grid = index_grids[n]
            if grid is not None and (count % args_cli.every == 0):
                H, W = HW[n]
                pos = cube.data.nodal_pos_w[0]
                z_grid = pos[grid.reshape(-1), 2].reshape(H, W).detach().cpu().numpy()
                xyz = pos[grid.reshape(-1)].detach().cpu().numpy()
                pub_z, pub_xyz = pubs[n]
                pub_z.publish(make_multiarray_2d(z_grid))
                pub_xyz.publish(make_multiarray_xyz(xyz))

        # keep ROS spinning (non-blocking)
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

    # --- Sim (force torch backend to avoid numpy->torch swap) ---
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,      # "cuda:0"
    )
    sim = SimulationContext(sim_cfg)

    # try:    # (Optional) print to verify the effective caps PhysX sees
    #     print("[PhysX GPU Caps] gpu_found_lost_aggregate_pairs_capacity=", sim.cfg.physx.gpu_found_lost_aggregate_pairs_capacity)
    #     print("[PhysX GPU Caps] gpu_collision_stack_size=", sim.cfg.physx.gpu_collision_stack_size)
    #     print("[PhysX GPU Caps] gpu_total_aggregate_pairs_capacity=", sim.cfg.physx.gpu_total_aggregate_pairs_capacity)
    #     print("[PhysX GPU Caps] gpu_temp_buffer_capacity=", sim.cfg.physx.gpu_temp_buffer_capacity)
    #     print("[PhysX GPU Caps] gpu_collision_stack_size=", sim.cfg.physx.gpu_collision_stack_size)
    #     print("[PhysX GPU Caps] gpu_found_lost_pairs_capacity=", sim.cfg.physx.gpu_found_lost_pairs_capacity)
    #     print("[PhysX GPU Caps] gpu_max_soft_body_contacts=", sim.cfg.physx.gpu_max_soft_body_contacts)
    #     print("[PhysX GPU Caps] gpu_heap_capacity=", sim.cfg.physx.gpu_heap_capacity)
    #     print("[PhysX GPU Caps] gpu_temp_buffer_capacity=", sim.cfg.physx.gpu_temp_buffer_capacity)
    #     print("[PhysX GPU Caps] gpu_max_rigid_contact_count=", sim.cfg.physx.gpu_max_rigid_contact_count)
        
    # except Exception as e:
    #     print(f"[WARN] Could not set GPU broadphase caps programmatically: {e}")

    # --- Build scene *after* caps are set, then reset ---
    sim.set_camera_view(eye=(3.0, -3.0, 1.0), target=(0.0, 0.0, 0.5))
    entities_list = design_scene()

    # The first reset() creates the PhysX scene with your caps already applied
    sim.reset()
    print("[INFO]: Setup complete... (Up/W=inflate, Down/S=deflate, Space=stop)")

    try:
        run_simulator(sim, entities_list, ros_node, keyctl)
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
