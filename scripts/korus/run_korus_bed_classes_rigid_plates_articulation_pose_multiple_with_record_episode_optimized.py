# --------------------------------
#  0) IMPORTS FOR APP CREATION ---
# --------------------------------
import argparse
from isaaclab.app import AppLauncher

# ----------------------------------------------
# 1) LAUNCH THE ISAACSIM APP (with argparse) ---
# ----------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab + YAML config")
parser.add_argument("--num_envs"    , type=int  , default=1                                     , help="")
parser.add_argument("--env_spacing" , type=float, default=2.0                                   , help="")
parser.add_argument("--num_rows"    , type=int  , default=None                                  , help="override YAML rows")
parser.add_argument("--num_cols"    , type=int  , default=None                                  , help="override YAML cols")
parser.add_argument("--every"       , type=int  , default=10                                    , help="ROS publish + record every N frames")
parser.add_argument("--config"      , type=str  , default="scripts/korus/config/korus_bed.yaml" , help="YAML file with globals")
parser.add_argument("--npz_dir"     , type=str  , default="scripts/korus/assets/npz"            , help="Directory containing pose npz files (e.g., 0000.npz, 0010.npz, ...)")
parser.add_argument("--npz_glob"    , type=str  , default="*.npz"                               , help="Glob pattern inside npz_dir (default: *.npz)")
parser.add_argument("--npz_start"   , type=int  , default=None                                  , help="Optional: only use poses with index >= this (e.g., 0)")
parser.add_argument("--npz_end"     , type=int  , default=None                                  , help="Optional: only use poses with index <= this (e.g., 10)")

grp = parser.add_mutually_exclusive_group()
grp.add_argument("--npz_loop", action="store_true"  , dest="npz_loop", help="Loop through poses (default)")
grp.add_argument("--npz_once", action="store_false" , dest="npz_loop", help="Run poses once then stop")
parser.set_defaults(npz_loop=True)

parser.add_argument("--record_dir"  , type=str  , default="outputs/korus_mocap"                 ,help="Output folder for MoCap + pressure recordings (.npz)")

# --- speed knobs ---
rend_grp = parser.add_mutually_exclusive_group()
rend_grp.add_argument("--render"    , action="store_true"    , dest="render",    help="Render each sim step (default)")
rend_grp.add_argument("--no_render" , action="store_false"   , dest="render",    help="Disable rendering for speed")
parser.set_defaults(render=True)

parser.add_argument("--ros_spin_every"          , type=int  , default=1     , help="Call rclpy.spin_once every N sim steps (1 = every step).")
parser.add_argument("--anchor_every"            , type=int  , default=1     , help="Update foam anchor kinematic targets every N sim steps (1 = every step).")
parser.add_argument("--plate_pos_eps"           , type=float, default=1e-6  , help="Only rewrite foam anchors if TopPlate moved by > eps (L_inf).")
parser.add_argument("--humanoid_reassert_every" , type=int  , default=30    , help="Reassert humanoid joint targets every N steps (0 disables periodic reassert).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experience = "isaacsim.exp.full.kit" 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------
# 2) IMPORTS AFTER APP CREATION ---
# ---------------------------------
# --- isaaclab / isaacsim / omni imports ---
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import (
    DeformableObject,
    DeformableObjectCfg,
    Articulation,
    ArticulationCfg
)
from isaaclab.actuators import ImplicitActuatorCfg
import omni
import omni.usd
from pxr import Usd
import isaacsim.core.utils.prims as prim_utils

# --- math imports --- 
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

# --- system imports ---
import os
from typing import Optional, List, Dict, TypedDict
import ipdb as pdb
from dataclasses import dataclass

# --- custom utility class imports ---
# from utils.keycontrol         import KeyControl       # NOTE: UNCOMMENT IF YOU WANT KEYBOARD CONTROL
# from utils.ros_bed_controller import ROSBedController # NOTE: UNCOMMENT IF YOU WANT KEYBOARD CONTROL
from utils.pressure_mapper    import PressureMapper
from utils.grid_builder       import GridBuilder
from utils.cfg_utils import (
    AppCfg,
    MaterialCfg,
    load_cfg
)
from utils.stiffness_and_damping import (
    STIFFNESS_PER_JOINT,
    DAMPING_PER_JOINT,
    EFFORT_PER_JOINT,
    ALL_HUMANOID_JOINTS,
    DEFAULT_STIFFNESS, DEFAULT_EFFORT, DEFAULT_DAMPING
)
from utils.joint_mappings import (
    SMPL_BODY_JOINT_ORDER,
    SMPL_TO_ISAAC
)
from utils.rotation_utils import (
    rotvec_to_euler_xyz,
    quat_wxyz_to_rotvec,
    euler_xyz_to_rotvec
)
from utils.npz_utils import (
    list_npz_files
)

# --- ROS2 imports ---
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState

# --- some house keeping ---
class CellEntry(TypedDict):
    cube: DeformableObject
    node_index: int
    row: int
    col: int

def _req(t: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if t is None:
        raise RuntimeError(f"{name} is None (cache not initialized; did reset-precompute run?)")
    return t

# ---------------------------------------------
# --- Constants --- NOTE: Later Shift to YAML
# ---------------------------------------------
TOPPLATE_SIZE   = (1.0, 1.0, 0.1)  # NOTE: This comes from the Bed Asset. 
RESET_AFTER     = 6e2

# ---------------------
# 1) Custom Classes ---
# ---------------------
class PrismaticJointTargetSubscriber:
    def __init__(self, ros_node: Node, topic="/korusbed/joint_pos_target"):
        self.ros_node = ros_node
        self.topic = topic
        self._subscriber = ros_node.create_subscription(JointState, topic, self._callback, 10)

        self._name_to_id: dict[str, int] = {}
        self._prismatic_ids: list[int] = []

        self._joint_cmd: torch.Tensor | None = None   # [B, DoF]
        self._joint_min: torch.Tensor | None = None   # [B, DoF]
        self._joint_max: torch.Tensor | None = None   # [B, DoF]

        self._latest_named: dict[str, float] = {}
        self._latest_prismatic_vector: np.ndarray | None = None
        self._has_update: bool = False

    def attach_bed(self, joint_names: List[str], joint_angles: torch.Tensor, joint_limits: torch.Tensor):
        """
        Call after you reset the bed.
        joint_angles: [B, DoF]
        joint_limits: ideally [B, DoF, 2] (lo/hi). If [DoF, 2], we broadcast to B.
        """
        self._name_to_id = {nm: i for i, nm in enumerate(joint_names) if isinstance(nm, str)}
        self._prismatic_ids = [
            i for i, nm in enumerate(joint_names)
            if isinstance(nm, str) and nm.startswith("PrismaticJoint")
        ]

        B, DoF = joint_angles.shape

        # Normalize joint_limits shape
        if joint_limits.ndim == 2 and joint_limits.shape == (DoF, 2):
            joint_limits = joint_limits.unsqueeze(0).expand(B, DoF, 2)
        elif joint_limits.ndim != 3 or joint_limits.shape[-1] != 2:
            raise ValueError(f"joint_limits must be [B,DoF,2] or [DoF,2], got {tuple(joint_limits.shape)}")

        low_limits = joint_limits[..., 0].to(device=joint_angles.device, dtype=joint_angles.dtype)
        high_limits = joint_limits[..., 1].to(device=joint_angles.device, dtype=joint_angles.dtype)

        self._joint_cmd = joint_angles.clone()
        self._joint_min = low_limits.clone()
        self._joint_max = high_limits.clone()

        self._latest_named.clear()
        self._latest_prismatic_vector = None
        self._has_update = False

        self.ros_node.get_logger().info(
            f"[PrismaticJointTargetSubscriber] Attached bed. DoF={DoF}, prismatic={len(self._prismatic_ids)}"
        )

    def _callback(self, msg: JointState):
        """Executed when a message is received."""
        if msg.name:
            for name, pos in zip(msg.name, msg.position):
                self._latest_named[name] = float(pos)
            self._latest_prismatic_vector = None
            self._has_update = True
            return

        if len(msg.position) > 0:
            self._latest_prismatic_vector = np.asarray(msg.position, dtype=np.float32)
            self._latest_named.clear()
            self._has_update = True

    def apply_latest(self):
        """
        Call each sim step (or at your --every rate).
        Updates self._joint_cmd in-place.
        """
        if (not self._has_update) or (self._joint_cmd is None) or (self._joint_min is None) or (self._joint_max is None):
            return

        joint_angles = self._joint_cmd

        # Named update
        if self._latest_named:
            idxs: list[int] = []
            vals: list[float] = []

            for name, val in self._latest_named.items():
                j = self._name_to_id.get(name)
                if j is not None:
                    idxs.append(j)
                    vals.append(val)

            if idxs:
                idxt = torch.tensor(idxs, device=joint_angles.device, dtype=torch.long)
                valt = torch.tensor(vals, device=joint_angles.device, dtype=joint_angles.dtype).view(1, -1)
                valt = valt.expand(joint_angles.shape[0], -1)  # broadcast to all envs

                joint_angles[:, idxt] = torch.clamp(
                    valt,
                    self._joint_min[:, idxt],
                    self._joint_max[:, idxt],
                )

        # Vector update (names omitted) -> apply in prismatic order
        elif self._latest_prismatic_vector is not None and self._prismatic_ids:
            v = self._latest_prismatic_vector
            n = min(len(v), len(self._prismatic_ids))
            pid = torch.tensor(self._prismatic_ids[:n], device=joint_angles.device, dtype=torch.long)
            valt = torch.tensor(v[:n], device=joint_angles.device, dtype=joint_angles.dtype).view(1, -1)
            valt = valt.expand(joint_angles.shape[0], -1)

            joint_angles[:, pid] = torch.clamp(
                valt,
                self._joint_min[:, pid],
                self._joint_max[:, pid],
            )

        self._has_update = False

    @property
    def joint_cmd(self) -> torch.Tensor | None:
        return self._joint_cmd
    
# -----------------------
# 2) HELPER FUNCTIONS ---
# -----------------------
def build_joint_config(path: str) -> dict[str, float]:
    """
    Builds Joint Configuration (for Implicit Actuator ) given a .npz smpl pose file.
    Produces Euler XYZ (radians) per Isaac hinge triplet. 
    """
    data = np.load(path, allow_pickle=True)
    body = data["pose_body"][0].astype(np.float32).reshape(-1, 3)

    joint_cfg: dict[str, float] = {}
    for smpl_name, isaac_triplet in SMPL_TO_ISAAC.items():
        smpl_idx = SMPL_BODY_JOINT_ORDER.index(smpl_name)  # 0..22
        rotvec = body[smpl_idx]  # (3,)
        rx, ry, rz = rotvec_to_euler_xyz(rotvec, degrees=False)  # radians
        jx, jy, jz = isaac_triplet
        joint_cfg[jx] = float(rx)
        joint_cfg[jy] = float(ry)
        joint_cfg[jz] = float(rz)
    return joint_cfg

def pad_to_shape(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pad a 2D array to (H,W) with zeros (bottom/right)."""
    h, w = arr.shape
    if h == H and w == W:
        return arr
    out = np.zeros((H, W), dtype=arr.dtype)
    out[:h, :w] = arr
    return out

def save_episode_npz(
    out_dir: str,
    pose_basename: str,
    episode_idx: int,
    body_names: List[str],
    times_s: List[float],
    root_pos_list: List[np.ndarray],
    root_rotvec_list: List[np.ndarray],
    spine_pos_list: List[np.ndarray],
    spine_rotvec_list: List[np.ndarray],
    body_pos_list: List[np.ndarray],
    body_rotvec_list: List[np.ndarray],
    pose_body_rotvec_list: List[np.ndarray],
    pressure_list: List[np.ndarray],
):
    """
    Save one episode (one pose between resets) to a single .npz file.
    NOTE: Verify all the data entries before they are saved. 
    """
    if not root_pos_list:
        print(f"[WARN] Episode {episode_idx} ({pose_basename}): no frames recorded, skipping save.")
        return
     
    t                   = np.asarray(times_s, dtype=np.float32)                         # [T] - timestamp
    root_pos            = np.stack(root_pos_list, axis=0).astype(np.float32)            # [T, 3] - pelvic pos
    root_rotvec         = np.stack(root_rotvec_list, axis=0).astype(np.float32)         # [T, 3] - pelvic rotvec 
    spine_pos           = np.stack(spine_pos_list, axis=0).astype(np.float32)            # [T, 3] - spine pos 
    spine_rotvec        = np.stack(spine_rotvec_list, axis=0).astype(np.float32)         # [T, 3] - spine rotvec 
    body_pos            = np.stack(body_pos_list, axis=0).astype(np.float32)            # [T, Nb, 3] - translation w.r.t World for 23 body links
    body_rotvec         = np.stack(body_rotvec_list, axis=0).astype(np.float32)         # [T, Nb, 3] - rotation (rotvec) w.r.t World for 23 Body Links 
    pose_body_rotvec    = np.stack(pose_body_rotvec_list, axis=0).astype(np.float32)    # [T, J, 3] - Joint Angles w.r.t to previous frame consistent with SMPL Kinematic Tree

    # pressure: [T, Ncells, H, W] (we try to stack; if shapes differ, pad to max H/W)
    # NOTE: H and W here is the dimension of the pressure map. If the input does'nt map the dimension, then we do paddin. g
    # TODO: padding should never be the case because therefore verify i
    if pressure_list:
        # pressure_list entries are [Ncells, H, W]
        maxH = max(p.shape[-2] for p in pressure_list)
        maxW = max(p.shape[-1] for p in pressure_list)
        padded = []
        for p in pressure_list:
            # p: [Ncells, H, W]
            pcells = []
            for c in range(p.shape[0]):
                pcells.append(pad_to_shape(p[c], maxH, maxW))
            padded.append(np.stack(pcells, axis=0))
        pressure = np.stack(padded, axis=0).astype(np.float32)
    else:
        pressure = np.zeros((0, 0, 0, 0), dtype=np.float32)

    # define directory and output path and filename.
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{episode_idx:04d}_{pose_basename}_mocap_press.npz")
    
    # save the numpy file
    np.savez_compressed(
        out_path,
        time_s              =   t,                      # timestamp
        root_pos_w          =   root_pos,               # root pose w.r.t World Frame
        root_rotvec_w       =   root_rotvec,            # root orientation (rotvec) w.r.t World Frame
        spine_pos_w         =   spine_pos,              # spine pose w.r.t World Frame
        spine_rotvec_w      =   spine_rotvec,           # spine orientation (rotvec) w.r.t World Frame
        body_pos_w          =   body_pos,               # joint positions w.r.t World Frame 
        body_rotvec_w       =   body_rotvec,            # joint orientations w.r.t World Frame
        pose_body_rotvec    =   pose_body_rotvec,       # joint angles (SMPL style in rotvec) 
        pressure            =   pressure,               # pressure grid [T, Ncells, H, W]
        body_names          =   np.array(body_names),   # mc 
        smpl_joint_order    =   np.array(SMPL_BODY_JOINT_ORDER),
        pose_name           =   pose_basename,  
    )
    print(f"[INFO] Saved episode {episode_idx} → {out_path}")

# ------------------------
# 4) PUBLISHER HELPERS ---
# ------------------------
def make_multiarray_2d(arr2d: np.ndarray) -> Float32MultiArray:
    rows, cols = arr2d.shape
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="rows", size=int(rows), stride=int(rows * cols)),
        MultiArrayDimension(label="cols", size=int(cols), stride=int(cols)),
    ]
    msg.data = arr2d.astype(np.float32).ravel().tolist()
    return msg

# ----------------------
# 4.5) SPEED HELPERS ---
# ----------------------
def _torch_cmd_changed(last: Optional[torch.Tensor], cur: torch.Tensor, eps: float = 1e-6) -> bool:
    if last is None:
        return True
    if last.shape != cur.shape:
        return True
    return bool(torch.any(torch.abs(cur - last) > eps).item())

def _fill_topplate_corners_inplace(
    plate_pos_xyz: torch.Tensor,   # (3,)
    half_x: torch.Tensor,
    half_y: torch.Tensor,
    half_z: torch.Tensor,
    out_corners: torch.Tensor,     # (1,4,3)
) -> None:
    cx, cy, cz = plate_pos_xyz[0], plate_pos_xyz[1], plate_pos_xyz[2]
    z_top = cz + half_z
    # BL
    out_corners[0, 0, 0] = cx - half_x
    out_corners[0, 0, 1] = cy - half_y
    out_corners[0, 0, 2] = z_top
    # TL
    out_corners[0, 1, 0] = cx - half_x
    out_corners[0, 1, 1] = cy + half_y
    out_corners[0, 1, 2] = z_top
    # BR
    out_corners[0, 2, 0] = cx + half_x
    out_corners[0, 2, 1] = cy - half_y
    out_corners[0, 2, 2] = z_top
    # TR
    out_corners[0, 3, 0] = cx + half_x
    out_corners[0, 3, 1] = cy + half_y
    out_corners[0, 3, 2] = z_top

def _plate_moved_enough(last: Optional[torch.Tensor], cur: torch.Tensor, eps: float) -> bool:
    if last is None:
        return True
    # L_inf threshold
    return bool(torch.max(torch.abs(cur - last)).item() > eps)

@dataclass
class _CellCache:
    pub: Publisher
    cube: DeformableObject
    node_index: int
    row: int
    col: int

    # computed on reset:
    bottom_corners: Optional[torch.Tensor] = None     # (4,) long
    kin: Optional[torch.Tensor] = None                # (B,N,4)
    valid_mask: Optional[torch.Tensor] = None         # (Ne,) bool
    tri_nodes: Optional[torch.Tensor] = None          # (Nt,3) long
    tri_elem: Optional[torch.Tensor] = None           # (Nt,) long
    cell_ids: Optional[torch.Tensor] = None           # (Nt,) long
    Hm: int = 0
    Wm: int = 0

    # per-step buffers:
    corners_buf: Optional[torch.Tensor] = None        # (1,4,3) same device/dtype as kin
    last_plate_pos: Optional[torch.Tensor] = None     # (3,)

# --------------------
# 5) SCENE BUILDER ---
# --------------------
def design_scene(cfg: AppCfg, initial_joint_config: dict[str, float]):
    """
    Build world + a single KorusBed articulation + (rows x cols) deformable cells.
    Returns dict with 'bed' and 'cells' (each cell has cube row/col/index) + humanoid.
    """
    # opening a scene and getting stage
    scene_context = omni.usd.get_context() 
    scene_context.open_stage(cfg.env_usd)
    stage = scene_context.get_stage() 

    # uniform lighting
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    dome.func("/World/DomeLight", dome)

    # defining the humanoid asset
    humanoid_asset = sim_utils.UsdFileCfg(usd_path=cfg.humanoid_usd, scale=(4.4, 4.4, 4.4))
    humanoid_asset.func("/World/Humanoid", humanoid_asset)

    # scene-only override: deactivate /World/Humanoid/worldBody
    worldbody_path = "/World/Humanoid/worldBody"
    prim_composed = stage.GetPrimAtPath(worldbody_path)
    if prim_composed.IsValid():
        print(f"[INFO] Deactivating {worldbody_path} via session-layer override.")
        session_layer = stage.GetSessionLayer()
        with Usd.EditContext(stage, session_layer):  # type: ignore
            prim_override = stage.OverridePrim(worldbody_path)
            prim_override.SetActive(False)
    else:
        print(f"[INFO] {worldbody_path} not found in composed stage; nothing to deactivate.")

    # import the bed asset
    bed_asset = sim_utils.UsdFileCfg(usd_path=cfg.korusbed_usd)
    bed_asset.func("/World/KorusBed", bed_asset)

    # bed articulation
    bed_cfg = ArticulationCfg(
        class_type=Articulation,
        prim_path="/World/KorusBed",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.1), rot=(1, 0, 0, 0),
            joint_pos={f"PrismaticJoint{i}": 0.0 for i in range(cfg.rows * cfg.cols)},
            joint_vel={f"PrismaticJoint{i}": 0.0 for i in range(cfg.rows * cfg.cols)},
        ),
        actuators={
            "prismatic_pd": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint.*"],
                effort_limit_sim=10_000_000.0,
                stiffness=6_000_000.0,
                damping=800.0,
            ),
        },
    )
    bed = Articulation(cfg=bed_cfg)

    def make_joint_pd_cfg(joint_name: str) -> ImplicitActuatorCfg:
        return ImplicitActuatorCfg(
            joint_names_expr=[joint_name],
            effort_limit_sim=EFFORT_PER_JOINT.get(joint_name, DEFAULT_EFFORT),
            stiffness=STIFFNESS_PER_JOINT.get(joint_name, DEFAULT_STIFFNESS),
            damping=DAMPING_PER_JOINT.get(joint_name, DEFAULT_DAMPING),
        )

    humanoid_actuators = {f"{jn}_pd": make_joint_pd_cfg(jn) for jn in ALL_HUMANOID_JOINTS}

    humanoid_cfg = ArticulationCfg(
        class_type=Articulation,
        prim_path="/World/Humanoid",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 1.5, 2.5),
            rot=(1, 0, 0, 0),
            joint_pos=initial_joint_config,
            joint_vel={".*": 0.0},
        ),
        actuators=humanoid_actuators,  # type: ignore
    )
    humanoid = Articulation(cfg=humanoid_cfg)

    # deformable foam cells
    sx = float(cfg.node_spacing)
    sy = float(cfg.node_spacing)
    cells = []

    for r in range(cfg.rows):
        for c in range(cfg.cols):
            idx = r * cfg.cols + c
            x_off = c * sx
            y_off = -r * sy

            prim_utils.create_prim(f"/World/KorusBed/Node{idx}", "Xform")

            deform_cfg = DeformableObjectCfg(
                prim_path=f"/World/KorusBed/Node{idx}/DeformableCuboid{idx}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=cfg.foam_size,
                    deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                        rest_offset=0.0,
                        contact_offset=0.001,
                        simulation_hexahedral_resolution=cfg.material.hexa_res,
                    ),
                    physics_material=sim_utils.DeformableBodyMaterialCfg(
                        poissons_ratio=cfg.material.poissons_ratio,
                        youngs_modulus=cfg.material.youngs_modulus,
                        dynamic_friction=cfg.material.dynamic_friction,
                        elasticity_damping=cfg.material.elasticity_damping,
                        damping_scale=cfg.material.damping_scale,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                ),
                init_state=DeformableObjectCfg.InitialStateCfg(
                    pos=(cfg.foam_origin[0] + x_off, cfg.foam_origin[1] + y_off, cfg.foam_origin[2])
                ),
                debug_vis=False,
            )
            cube_obj = DeformableObject(cfg=deform_cfg)

            cells.append({"cube": cube_obj, "node_index": idx, "row": r, "col": c})

    return {"bed": bed, "cells": cells, "humanoid": humanoid}

# ---------------------
# 6) Run simulator ----
# ---------------------
def run_simulator(
    sim: sim_utils.SimulationContext,
    entities,
    ros_node: Node,
    cfg: AppCfg,
    npz_files: list[str],
    npz_loop: bool,
    record_dir: str,
):
    # --- extracting entities (isaaclab objects) ---
    bed: Articulation       = entities["bed"]
    cells: List[CellEntry]  = entities["cells"]
    humanoid: Articulation  = entities["humanoid"]


    # --- humanoid joint helpers ---
    joint_names     = humanoid.data.joint_names
    joint_index_map = {name: i for i, name in enumerate(joint_names)}
    body_names      = humanoid.data.body_names  # NOTE: Spine is the root for this.
    pelvis_idx = body_names.index("Pelvis")

    # --- pose iteration ---
    humanoid_pose_idx               = 0
    target_humanoid_joint_angles    = None
    current_npz                     = None

    # --- episode recording buffers ---
    episode_idx                                 = 0
    episode_pose_name: Optional[str]            = None
    episode_timestamp: List[float]              = []
    episode_root_pos: List[np.ndarray]          = []
    episode_root_rotvec: List[np.ndarray]       = []
    episode_spine_pos: List[np.ndarray]         = []
    episode_spine_rotvec: List[np.ndarray]      = []
    episode_body_pos: List[np.ndarray]          = []
    episode_body_rotvec: List[np.ndarray]       = []
    episode_pose_body_rotvec: List[np.ndarray]  = []
    episode_pressure: List[np.ndarray]          = []

    # --- ros2 prismatic controller ---
    ros_ctl = PrismaticJointTargetSubscriber(ros_node, topic="/korusbed/joint_pos_target")
    last_bed_cmd: Optional[torch.Tensor] = None

    # --- per-cell caches ---
    cell_caches: List[_CellCache] = []
    for ent in cells:
        row = ent["row"]; col = ent["col"]
        pub = ros_node.create_publisher(
            Float32MultiArray,
            f"/foam_bed/top_surface/pressure_grid_{row}_{col}",
            10
        )
        cell_caches.append(_CellCache(
            pub=pub,
            cube=ent["cube"],
            node_index=ent["node_index"],
            row=row, col=col,
        ))

    # --- timing ---
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # --- top plate body ids (rebuilt on reset) ---
    top_plate_body_ids: List[int] = []
    for idx in range(cfg.rows * cfg.cols):
        ids, _ = bed.find_bodies(f"TopPlate{idx}")
        if not ids:
            raise RuntimeError(f"TopPlate{idx} not found in bed bodies.")
        top_plate_body_ids.append(ids[0])

    # -----------------------------
    # 8) SIMULATION LOOP STARTS ---
    # -----------------------------
    while simulation_app.is_running():
        # ------------------
        # --- RESET LOOP ---
        # ------------------
        if (count % RESET_AFTER) == 0:
            # --- save previous episode if exists ---
            if (episode_pose_name is not None) and episode_root_pos:
                save_episode_npz(
                    record_dir,
                    episode_pose_name,
                    episode_idx,
                    body_names,
                    episode_timestamp,
                    episode_root_pos,
                    episode_root_rotvec,
                    episode_spine_pos,
                    episode_spine_rotvec,
                    episode_body_pos,
                    episode_body_rotvec,
                    episode_pose_body_rotvec,
                    episode_pressure,
                )
                episode_idx += 1
                episode_timestamp = []
                episode_root_pos = []
                episode_root_rotvec = []
                episode_spine_pos = []
                episode_spine_rotvec = []
                episode_body_pos = []
                episode_body_rotvec = []
                episode_pose_body_rotvec = []
                episode_pressure = []

            sim_time = 0.0
            count = 0

            # --- reset bed ---
            root_state_bed = bed.data.default_root_state.clone()
            bed.write_root_pose_to_sim(root_state_bed[:, :7])
            bed.write_root_velocity_to_sim(root_state_bed[:, 7:])

            joint_pos_bed = bed.data.default_joint_pos.clone()
            joint_vel_bed = bed.data.default_joint_vel.clone()
            bed.write_joint_state_to_sim(joint_pos_bed, joint_vel_bed)
            bed.reset()

            # IMPORTANT: re-attach controller after reset (fresh tensors/limits)
            ros_ctl.attach_bed(
                joint_names=bed.data.joint_names,
                joint_angles=bed.data.joint_pos,
                joint_limits=bed.data.joint_pos_limits,
            )
            last_bed_cmd = None

            # --- pick next humanoid pose ---
            if not npz_files:
                raise RuntimeError("No NPZ files provided (npz_files list is empty).")

            current_npz         = npz_files[humanoid_pose_idx]
            joint_cfg           = build_joint_config(current_npz)
            pose_name           = os.path.splitext(os.path.basename(current_npz))[0]
            episode_pose_name   = pose_name

            humanoid_pose_idx += 1
            if humanoid_pose_idx >= len(npz_files):
                if npz_loop:
                    humanoid_pose_idx = 0
                else:
                    print("[INFO] Finished all NPZ poses (--npz_once). Exiting simulation loop.")
                    break

            # --- build target humanoid joint angles ---
            target_humanoid_joint_angles = humanoid.data.default_joint_pos.clone()
            for joint_name, angle in joint_cfg.items():
                j = joint_index_map.get(joint_name)
                if j is not None:
                    target_humanoid_joint_angles[:, j] = float(angle)  

            print(f"[INFO] Using pose: {os.path.basename(current_npz)}")

            # --- reset humanoid ---
            root_state_humanoid = humanoid.data.default_root_state.clone()
            humanoid.write_root_pose_to_sim(root_state_humanoid[:, :7])
            humanoid.write_root_velocity_to_sim(root_state_humanoid[:, 7:])

            joint_pos_humanoid = target_humanoid_joint_angles.clone()  
            joint_vel_humanoid = torch.zeros_like(joint_pos_humanoid)
            humanoid.write_joint_state_to_sim(joint_pos_humanoid, joint_vel_humanoid)
            humanoid.reset()
            humanoid.set_joint_position_target(joint_pos_humanoid)

            # --- reset deformables ---
            for cc in cell_caches:
                cube_state = cc.cube.data.default_nodal_state_w.clone()
                cc.cube.write_nodal_state_to_sim(cube_state)
                cc.cube.reset()

            # initialize half sizes on bed device/dtype
            bed_dev = bed.data.body_link_pos_w.device
            bed_dtype = bed.data.body_link_pos_w.dtype
            half_x_t = torch.tensor(TOPPLATE_SIZE[0] * 0.5, device=bed_dev, dtype=bed_dtype)
            half_y_t = torch.tensor(TOPPLATE_SIZE[1] * 0.5, device=bed_dev, dtype=bed_dtype)
            half_z_t = torch.tensor(TOPPLATE_SIZE[2] * 0.5, device=bed_dev, dtype=bed_dtype)

            # per-cell mappings & pins + pressure precompute
            for n, cc in enumerate(cell_caches):

                cube = cc.cube
                dev = cube.data.nodal_pos_w.device

                # bottom-corner pins (DEFAULT mesh)
                default_pos = cube.data.default_nodal_state_w[..., :3][0] # default nodal positions: [847, 3]
                z = default_pos[:, 2] # [847]     
                s = 10 ** cfg.decimals # 10                
                zr = torch.round(z * s) / s # [847]
 
                planes = torch.sort(torch.unique(zr)).values # [7] unique values nodal heights (NOTE: 847 / 7 = 121) hence, every surface has 121 nodes
                z_bot_default = float(planes[0]) # height of the bottom surface of deformable cube 

                bot_idx = GridBuilder._get_surface_indices_by_z(cube, z_bot_default, cfg.decimals) # [121]  # type: ignore
                if bot_idx.numel() == 0:
                    raise RuntimeError(f"[Cell {n}] collision bottom nodes not found.")

                bot_corners = GridBuilder.corners_from_surface_idx(cube, bot_idx, env_id=0)
                cc.bottom_corners = bot_corners

                # kinematic target buffer
                kin = cube.data.nodal_kinematic_target.clone()
                kin[..., :3] = cube.data.nodal_pos_w
                kin[..., 3] = 1.0
                kin[:, bot_corners, 3] = 0.0
                cube.write_nodal_kinematic_target_to_sim(kin)
                cc.kin = kin

                # prealloc corners buffer (same device/dtype as kin)
                cc.corners_buf = torch.zeros((1, 4, 3), device=kin.device, dtype=kin.dtype)
                cc.last_plate_pos = None

                # pressure precompute
                grid_sim, _ = GridBuilder.build_sim_grid(cube, cfg.decimals)

                T_raw_any = cube.root_physx_view.get_sim_element_indices()[0]
                T_raw_np = T_raw_any.detach().cpu().numpy() if hasattr(T_raw_any, "detach") else np.asarray(T_raw_any)
                valid_mask_np = (T_raw_np >= 0).all(axis=1)
                cc.valid_mask = torch.as_tensor(valid_mask_np, device=dev, dtype=torch.bool)
                T_valid = T_raw_np[valid_mask_np]

                tri_nodes_np, tri_elem_np, cell_ids_np, Hm, Wm = PressureMapper.precompute_top_surface_tris_arrays(
                    T_valid, grid_sim
                )
                cc.tri_nodes = torch.as_tensor(tri_nodes_np, device=dev, dtype=torch.long)
                cc.tri_elem  = torch.as_tensor(tri_elem_np,  device=dev, dtype=torch.long)
                cc.cell_ids  = torch.as_tensor(cell_ids_np,  device=dev, dtype=torch.long)
                cc.Hm, cc.Wm = int(Hm), int(Wm)

            pris_count = sum(isinstance(nm, str) and nm.startswith("PrismaticJoint") for nm in bed.data.joint_names)
            print(f"[INFO] Reset bed + humanoid + {len(cell_caches)} deformables; prismatic joints = {pris_count}")

        # --------------------------
        # --- STEP & UPDATE LOOP ---
        # --------------------------
        # ---step through the simulation
        sim.step(render=bool(args_cli.render)) # tunable for speed
        sim_time += sim_dt 
        count += 1

        # --- update all buffers --- 
        bed.update(sim_dt)
        humanoid.update(sim_dt)
        for cc in cell_caches:
            cc.cube.update(sim_dt)

        # ROS callbacks (decimated)
        if args_cli.ros_spin_every <= 1 or (count % args_cli.ros_spin_every == 0):
            rclpy.spin_once(ros_node, timeout_sec=0.0)

        # Apply latest ROS joint targets to bed (skip if unchanged)
        ros_ctl.apply_latest()
        if ros_ctl.joint_cmd is not None:
            cmd = ros_ctl.joint_cmd
            if _torch_cmd_changed(last_bed_cmd, cmd, eps=1e-6):
                bed.set_joint_position_target(cmd)
                last_bed_cmd = cmd.clone()

        # Reassert humanoid target occasionally (instead of every step)
        if target_humanoid_joint_angles is not None and args_cli.humanoid_reassert_every > 0:
            if (count % args_cli.humanoid_reassert_every) == 0:
                humanoid.set_joint_position_target(target_humanoid_joint_angles)

        # ---- Record MoCap every `every` frames ----
        if target_humanoid_joint_angles is not None and (count % args_cli.every == 0):
            episode_timestamp.append(float(sim_time))

            assert body_names[0] == "Spine"
            assert body_names[pelvis_idx] == "Pelvis"

            # Record "root" as Pelvis (SMPL-style)
            pelvis_pose_numpy   = humanoid.data.body_link_pose_w[0][pelvis_idx].detach().cpu().numpy().astype(np.float32)  
            pelvis_pos          = pelvis_pose_numpy[:3]
            pelvis_quat_wxyz    = pelvis_pose_numpy[3:]
            pelvis_rotvec       = quat_wxyz_to_rotvec(pelvis_quat_wxyz)
            episode_root_pos.append(pelvis_pos.astype(np.float32))
            episode_root_rotvec.append(pelvis_rotvec.astype(np.float32))

            spine_pose_w_numpy  = humanoid.data.root_pose_w[0].detach().cpu().numpy()
            spine_pos           = spine_pose_w_numpy[:3]
            spine_quat_wxyz     = spine_pose_w_numpy[3:]
            spine_rotvec        = quat_wxyz_to_rotvec(spine_quat_wxyz)
            episode_spine_pos.append(spine_pos.astype(np.float32))
            episode_spine_rotvec.append(spine_rotvec.astype(np.float32))
            body_pos            = humanoid.data.body_link_pos_w[0].detach().cpu().numpy().astype(np.float32)
            body_quat           = humanoid.data.body_link_quat_w[0].detach().cpu().numpy().astype(np.float32)  # wxyz
            Nb                  = body_pos.shape[0]
            body_rotvec         = np.zeros((Nb, 3), dtype=np.float32)
            for i in range(Nb):
                body_rotvec[i]  = quat_wxyz_to_rotvec(body_quat[i])
            episode_body_pos.append(body_pos)
            episode_body_rotvec.append(body_rotvec)

            joint_pos_np = humanoid.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            pose_body_rotvec = np.zeros((len(SMPL_BODY_JOINT_ORDER), 3), dtype=np.float32)
            for j_idx, smpl_name in enumerate(SMPL_BODY_JOINT_ORDER):
                jx, jy, jz = SMPL_TO_ISAAC[smpl_name]
                ix = joint_index_map.get(jx)
                iy = joint_index_map.get(jy)
                iz = joint_index_map.get(jz)
                if ix is None or iy is None or iz is None:
                    continue
                e_xyz                   = np.array([joint_pos_np[ix], joint_pos_np[iy], joint_pos_np[iz]], dtype=np.float32)
                pose_body_rotvec[j_idx] = euler_xyz_to_rotvec(e_xyz)
            episode_pose_body_rotvec.append(pose_body_rotvec)

        # Collect + publish pressure (aligned with `--every`)
        frame_pressure_cells: List[np.ndarray] = []
        do_anchor_write = (args_cli.anchor_every <= 1) or (count % args_cli.anchor_every == 0)

        for cc in cell_caches:
            cube = cc.cube
            idx  = cc.node_index

            # --- Update kinematic anchors (torch-only) ---
            if do_anchor_write:
                body_id   = top_plate_body_ids[idx]
                plate_pos = bed.data.body_link_pos_w[0, body_id, :3]  # torch (3,)

                if _plate_moved_enough(cc.last_plate_pos, plate_pos, eps=float(args_cli.plate_pos_eps)):
                    corners_buf    = _req(cc.corners_buf, "cc.corners_buf")
                    kin            = _req(cc.kin, "cc.kin")
                    bottom_corners = _req(cc.bottom_corners, "cc.bottom_corners")

                    _fill_topplate_corners_inplace(
                        plate_pos,
                        half_x_t, half_y_t, half_z_t,
                        corners_buf,   # Tensor (1,4,3)
                    )

                    # make RHS match kin device/dtype and batch
                    rhs = corners_buf
                    if rhs.device != kin.device or rhs.dtype != kin.dtype:
                        rhs = rhs.to(device=kin.device, dtype=kin.dtype)
                    if rhs.shape[0] != kin.shape[0]:
                        rhs = rhs.expand(kin.shape[0], -1, -1)

                    kin[:, bottom_corners, :3] = rhs
                    cube.write_nodal_kinematic_target_to_sim(kin)
                    cube.write_data_to_sim()
                    cc.last_plate_pos = plate_pos.detach().clone()

            # --- Pressure only every `--every` ---
            if count % args_cli.every != 0:
                continue

            valid_mask = _req(cc.valid_mask, "cc.valid_mask")
            tri_nodes  = _req(cc.tri_nodes,  "cc.tri_nodes")
            tri_elem   = _req(cc.tri_elem,   "cc.tri_elem")
            cell_ids   = _req(cc.cell_ids,   "cc.cell_ids")

            dev = cube.data.nodal_pos_w.device
            sim_pos_any = cube.root_physx_view.get_sim_nodal_positions()[0]
            S_any       = cube.root_physx_view.get_sim_element_stresses()[0]

            sim_pos_t = sim_pos_any.detach().to(dev) if hasattr(sim_pos_any, "detach") else \
                        torch.as_tensor(sim_pos_any, device=dev, dtype=torch.float32)
            S_t = S_any.detach().to(dev) if hasattr(S_any, "detach") else \
                torch.as_tensor(S_any, device=dev, dtype=torch.float32)

            S_valid_t = S_t[valid_mask]
            if S_valid_t.ndim == 2 and S_valid_t.size(-1) == 9:
                S_valid_t = S_valid_t.view(-1, 3, 3)

            P_t = PressureMapper.compute_pressure_grid_torch(
                tri_nodes, tri_elem, cell_ids,
                S_valid_t, sim_pos_t, cc.Hm, cc.Wm
            )

            # single CPU conversion per cell
            P_np = P_t.detach().cpu().numpy()
            frame_pressure_cells.append(P_np.astype(np.float32))
            cc.pub.publish(make_multiarray_2d(P_np))

        if frame_pressure_cells and (count % args_cli.every == 0):
            episode_pressure.append(np.stack(frame_pressure_cells, axis=0))

        # Flush command data (kept for correctness)
        bed.write_data_to_sim()
        humanoid.write_data_to_sim()

    # ---------------------------------------------------------
    # Save final episode if loop ends
    # ---------------------------------------------------------
    if episode_pose_name is not None and episode_root_pos:
        save_episode_npz(
            record_dir,
            episode_pose_name,
            episode_idx,
            body_names,
            episode_timestamp,
            episode_root_pos,
            episode_root_rotvec,
            episode_spine_pos,
            episode_spine_rotvec,
            episode_body_pos,
            episode_body_rotvec,
            episode_pose_body_rotvec,
            episode_pressure,
        )

# ------------
# --- Main ---
# ------------
def main():
    # --- YAML config ---
    cfg = load_cfg(args_cli.config, args_cli.num_rows, args_cli.num_cols)

    # --- ROS2 ---
    rclpy.init(args=None)
    ros_node = rclpy.create_node("foam_bed_publisher")

    # --- Sim ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, -3.0, 2.0), target=(0.0, 0.0, 0.5))

    # --- NPZ pose list ---
    npz_files = list_npz_files(
        args_cli.npz_dir,
        args_cli.npz_glob,
        start=args_cli.npz_start,
        end=args_cli.npz_end,
    )
    if not npz_files:
        raise RuntimeError(f"No npz files found in {args_cli.npz_dir} matching {args_cli.npz_glob}")

    print(f"[INFO] Found {len(npz_files)} pose files. First={os.path.basename(npz_files[0])}, Last={os.path.basename(npz_files[-1])}")

    # --- Record dir ---
    os.makedirs(args_cli.record_dir, exist_ok=True)

    # --- Scene ---
    initial_joint_config = build_joint_config(npz_files[0])
    entities = design_scene(cfg, initial_joint_config)
    sim.reset()

    print(f"[INFO]: Setup complete (rows={cfg.rows}, cols={cfg.cols}).")
    print(f"[INFO]: Recording to: {args_cli.record_dir}")
    print(f"[INFO]: render={args_cli.render}, ros_spin_every={args_cli.ros_spin_every}, anchor_every={args_cli.anchor_every}, plate_pos_eps={args_cli.plate_pos_eps}")

    try:
        run_simulator(
            sim,
            entities,
            ros_node,
            cfg,
            npz_files,
            args_cli.npz_loop,
            args_cli.record_dir,
        )
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
