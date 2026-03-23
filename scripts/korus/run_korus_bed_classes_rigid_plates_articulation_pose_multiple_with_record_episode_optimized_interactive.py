"""
Solution to convert the design scene function into an InteractiveSceneCfg that works: 

Steps: 
    
    
    (1) use the .replace(prim_path=..., ArticulationCfg.InitialStateCfg(pos=..., rot=...)) to make 32 different 
    DeformableObjectCfg's that have difference init poses. maybe you'll need a global variable that has all (x,y,z) of
    all the foams. 
    
    (2) (1) can be done in 2 ways: 
        - Either make new meshes using spawn = sim_utils.MeshCuboidCfg(...). 
        - Or keep the deformable part of the Bed Asset and then put spawn=None, and prim_path=<path-of-the-deformable-prim>  
        Former method will help with tuning; latter gives more access to the tuning parameters. 

    (3) After the spawning has occured successfully, make the appropriate changes to run_simulator() and main() functions. 
    
    (4) Change all the utility functions to work for batches (B), since we are gonna be tackling multiple environments.  

"""
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
    ArticulationCfg,
    AssetBase, 
    AssetBaseCfg, 
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
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
from utils.pressure_mapper import PressureMapper
from utils.grid_builder import GridBuilder
from utils.prismatic_joint_target_subscriber import PrismaticJointTargetSubscriber
from utils.cfg_utils_backup import (
    AppCfg,
    MaterialCfg,
    CFG,
    KORUSBED_CFG,
    HUMANOID_CFG,
    KORUSFOAM_CFG, 
    FOAM_ORIGIN_LIST,
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

def compute_foam_xyz(app_cfg: AppCfg) -> list[tuple[float, float, float]]:
    """
    Returns a list of (x,y,z) for each foam cell (len = rows*cols),
    ordered by idx = r*cols + c.
    """
    ox, oy, oz = map(float, app_cfg.foam_origin)
    sx = float(app_cfg.node_spacing)
    sy = float(app_cfg.node_spacing)

    out: list[tuple[float, float, float]] = []
    for r in range(app_cfg.rows):
        for c in range(app_cfg.cols):
            x = ox + c * sx
            y = oy - r * sy
            z = oz
            out.append((x, y, z))
    return out
    
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

# ----------------------------------
# 4.5) INTERACTIVE SCENE BUILDER ---
# ----------------------------------
@configclass
class KorusInteractiveSceneCfg(InteractiveSceneCfg):
    
    # --- non interactive assets (global)
    # ground plane cfg (global)
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg()
    # )
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
        ),
    )

    # dome light cfg (global)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # --- interactive assets (need to be cloned)
    # korus bed cfg
    bed: ArticulationCfg = KORUSBED_CFG.replace(prim_path="{ENV_REGEX_NS}/KorusBed") # type: ignore

    # humanoid cfg
    humanoid: ArticulationCfg = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Humanoid") # type: ignore

    # deformable foam cfgs (32) TODO: There has got to be a better way to do this. 
    deformable_0: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid0",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[0]),
    )

    deformable_1: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid1",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[1]),
    )

    deformable_2: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid2",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[2]),
    )

    deformable_3: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid3",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[3]),
    )

    deformable_4: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid4",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[4]),
    )

    deformable_5: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid5",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[5]),
    )

    deformable_6: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid6",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[6]),
    )

    deformable_7: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid7",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[7]),
    )

    deformable_8: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid8",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[8]),
    )

    deformable_9: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid9",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[9]),
    )

    deformable_10: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid10",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[10]),
    )

    deformable_11: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid11",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[11]),
    )

    deformable_12: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid12",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[12]),
    )

    deformable_13: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid13",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[13]),
    )

    deformable_14: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid14",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[14]),
    )

    deformable_15: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid15",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[15]),
    )

    deformable_16: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid16",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[16]),
    )

    deformable_17: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid17",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[17]),
    )

    deformable_18: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid18",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[18]),
    )

    deformable_19: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid19",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[19]),
    )

    deformable_20: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid20",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[20]),
    )

    deformable_21: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid21",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[21]),
    )

    deformable_22: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid22",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[22]),
    )

    deformable_23: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid23",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[23]),
    )

    deformable_24: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid24",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[24]),
    )

    deformable_25: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid25",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[25]),
    )

    deformable_26: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid26",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[26]),
    )

    deformable_27: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid27",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[27]),
    )

    deformable_28: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid28",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[28]),
    )

    deformable_29: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid29",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[29]),
    )

    deformable_30: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid30",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[30]),
    )

    deformable_31: DeformableObjectCfg = KORUSFOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid31",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[31]),
    )



# --------------------
# 5) SCENE BUILDER ---
# --------------------
# def design_scene():
#     """
#     Build world + a single KorusBed articulation + (rows x cols) deformable cells.
#     Returns dict with 'bed' and 'cells' (each cell has cube row/col/index) + humanoid.
#     """
#     # # --- opening a scene and getting stage ---
#     scene_context = omni.usd.get_context()
#     scene_context.open_stage(CFG.env_usd)
#     # stage = scene_context.get_stage() 
#     stage = sim_utils.get_current_stage()

#     # --- uniform lighting ---
#     dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
#     dome.func("/World/DomeLight", dome)

#     # -----------
#     # --- BED ---
#     # -----------
#     bed = Articulation(cfg=KORUSBED_CFG) # imported from CFG UTILS TODO: use replace() for Interactive Scene

#     # ----------------
#     # --- HUMANOID ---
#     # ---------------- 
#     humanoid = Articulation(cfg=HUMANOID_CFG) # imported from CFG Utils TODO: use replace() for Interactive Scene
 
#     # -----------------------
#     # --- DEFORMABLE FOAM ---
#     # -----------------------
#     cells = []

#     for r in range(CFG.rows):
#         for c in range(CFG.cols):
            
#             idx = r * CFG.cols + c # TODO: Remove double for loop for the Interactive Scene

#             deform_cfg = KORUSFOAM_CFG.replace( # type: ignore
#                 prim_path=f"/World/KorusBed/DeformableCuboid{idx}",
#                 init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[idx]),
#             ) 
#             cube_obj = DeformableObject(cfg=deform_cfg)


#             cells.append({"cube": cube_obj, "node_index": idx, "row": r, "col": c})
        

#     return {"bed": bed, "cells": cells, "humanoid": humanoid}

# ---------------------
# 6) Run simulator ----
# ---------------------
def run_simulator(
    sim: sim_utils.SimulationContext,
    # entities,
    scene: InteractiveScene, 
    ros_node: Node,
    cfg: AppCfg,
    npz_files: list[str],
    npz_loop: bool,
    record_dir: str,
):
    # # --- extracting entities (isaaclab objects) ---
    # bed: Articulation       = entities["bed"]
    # cells: List[CellEntry]  = entities["cells"]
    # humanoid: Articulation  = entities["humanoid"]

    # --- extracting entities (isaaclab objects) ---
    bed: Articulation       = scene["bed"]
    # cells: List[CellEntry]  = scene["cells"]
    humanoid: Articulation  = scene["humanoid"]

    def _cells_from_scene(scene: InteractiveScene, rows: int, cols: int) -> List[CellEntry]:
        cells: List[CellEntry] = []
        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            cube: DeformableObject = scene[f"deformable_{idx}"]  # <-- field name in config class
            cells.append({"cube": cube, "node_index": idx, "row": r, "col": c})
        return cells

    cells: List[CellEntry] = _cells_from_scene(scene, cfg.rows, cfg.cols)

    # pdb.set_trace()

    # --- humanoid joint helpers ---
    joint_names     = humanoid.data.joint_names
    joint_index_map = {name: i for i, name in enumerate(joint_names)}
    body_names      = humanoid.data.body_names  # NOTE: Spine is the root for this.
    pelvis_idx      = body_names.index("Pelvis")

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
                grid_sim = GridBuilder.build_sim_grid(cube, cfg.decimals)

                T_raw_any = cube.root_physx_view.get_sim_element_indices()[0]
                T_raw_np = T_raw_any.detach().cpu().numpy() if hasattr(T_raw_any, "detach") else np.asarray(T_raw_any)
                valid_mask_np = (T_raw_np >= 0).all(axis=1)
                cc.valid_mask = torch.as_tensor(valid_mask_np, device=dev, dtype=torch.bool)
                T_valid = T_raw_np[valid_mask_np]

                tri_nodes_np, tri_elem_np, cell_ids_np, Hm, Wm = PressureMapper.precompute_top_surface_triangle_arrays(
                    T_valid, grid_sim # NOTE: this error exists because of change in grid_builder.py [returns vector now instead of numpy]
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
    # initial_joint_config = build_joint_config(npz_files[0])
    # entities = design_scene()
    scene_cfg = KorusInteractiveSceneCfg(num_envs=args_cli.num_envs,env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()

    print(f"[INFO]: Setup complete (rows={CFG.rows}, cols={CFG.cols}).")
    print(f"[INFO]: Recording to: {args_cli.record_dir}")
    print(f"[INFO]: render={args_cli.render}, ros_spin_every={args_cli.ros_spin_every}, anchor_every={args_cli.anchor_every}, plate_pos_eps={args_cli.plate_pos_eps}")

    try:
        run_simulator(
            sim,
            # entities,
            scene,
            ros_node,
            CFG,
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
