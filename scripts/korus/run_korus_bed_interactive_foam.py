# --------------------------------
# 0) IMPORTS FOR APP CREATION ---
# --------------------------------
import argparse
from isaaclab.app import AppLauncher

# ----------------------------------------------
# 1) LAUNCH THE ISAACSIM APP (with argparse) ---
# ----------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab + YAML config")
parser.add_argument("--num_envs"        , type=int  , default = 1                                               , help="Number of Instances in the Interactive Scene")
parser.add_argument("--env_spacing"     , type=float, default = 2.0                                             , help="Physical Space between the Environments (in meters)")
parser.add_argument("--num_rows"        , type=int  , default = None                                            , help="Override YAML rows")
parser.add_argument("--num_cols"        , type=int  , default = None                                            , help="Override YAML cols")
parser.add_argument("--pressure_every"  , type=int  , default = 1                                               , help="Compute pressure every n frames; it will compute mocap data for every pressure frame as well")
parser.add_argument("--anchor_every"    , type=int  , default = 1                                               , help="Update foam anchor kinematic targets every N sim steps (1 = every step).")
parser.add_argument("--settle_steps"    , type=int  , default = 20                                              , help="Number of Iteration that you want the sim to run without any actions so that the anchors can settle")
parser.add_argument("--config"          , type=str  , default = "scripts/korus/config/korus_bed.yaml"           , help="YAML file with globals")
parser.add_argument("--reset_after"     , type=int  , default = 600                                             , help="Reset the whole scene after n steps")
parser.add_argument("--record_dir"      , type=str  , default = "scripts/korus/korus_pressure_pose_dataset"     , help="Where to save episode npz files.")
parser.add_argument("--in_pose_dir"     , type=str  , default = "scripts/korus/assets/npz"                      , help="Input Directory where there exists pose data in form of npz files")

parser.add_argument("--ros"             , action = "store_true" , help = "Enable ROS2; Note that this feature is helpful for real-time debugging")  
parser.add_argument("--calibration"     , action = "store_true" , help = "This will drop a sphere on the on the first cell for debugging")  
parser.add_argument("--save_data"       , action = "store_true" , help = "This will invoke the save data functionality")
parser.add_argument("--override"        , action = "store_true" , help = "Start from scratch (delete existing episode_*.npz in the output dir). Disables resume.")

# --- speed knobs ---
rend_grp = parser.add_mutually_exclusive_group()
rend_grp.add_argument("--render"    , action="store_true"    , dest="render",   help="Render each sim step (default)")
rend_grp.add_argument("--no_render" , action="store_false"   , dest="render",   help="Disable rendering for speed")
parser.set_defaults(render=True)

parser.add_argument("--plate_pos_eps"           , type=float, default=1e-6  , help="Only rewrite foam anchors if TopPlate moved by > eps (L_inf).")
parser.add_argument("--humanoid_reassert_every" , type=int  , default=30    , help="Reassert humanoid joint targets every N steps (0 disables periodic reassert).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.ros: 
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
    RigidObjectCfg, 
    RigidObject,
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
import math

# --- system imports ---
import os
import ipdb as pdb
from dataclasses import dataclass
import re
from pathlib import Path

# --- custom utility class imports ---
from utils.pressure_mapper import PressureMapper
from utils.grid_builder import GridBuilder
from utils.prismatic_joint_target_subscriber import PrismaticJointTargetSubscriber
from utils.cfg_utils import (
    AppCfg,
    MaterialCfg,
    CFG,
    KORUS_BED_CFG,
    KORUS_HUMANOID_CFG,
    KORUS_FOAM_CFG, 
    FOAM_ORIGIN_LIST,
)
from utils.joint_mappings import (
    SMPL_BODY_JOINT_ORDER,
    SMPL_TO_ISAAC
)
from utils.rotation_utils import (
    rotvec_to_euler_xyz,
    quat_wxyz_to_rotvec,
    euler_xyz_to_rotvec,
    rotvec_to_quat_wxyz
)
from utils.npz_utils import (
    list_npz_files, 
    build_joint_config_from_npz
)
from utils.anchor_utils import (
    AnchorCache, 
    plate_has_moved_enough, 
    compute_plate_corners
)
from utils.root_randomizer import (
    RootRandomizer,
    RootRandomizerCfg
)
from utils.episode_manager_utils import (
    EpisodeResumeManager
)

if args_cli.ros:
    # --- ros2 imports ---
    import rclpy
    from rclpy.node import Node
    from rclpy.publisher import Publisher
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    from sensor_msgs.msg import JointState

# --- data structure for computing pressure map --- 
from typing import Any
@dataclass
class PressureCache:
    valid_mask: torch.Tensor
    top_triangle_nodes: torch.Tensor
    top_triangle_elements: torch.Tensor
    cell_ids: torch.Tensor
    Hm: int
    Wm: int
    indent_cache: dict[str, Any]   # NOTE: make keys within this dict part of the class. 

# --- episode recorder --- 
from typing import Optional
@dataclass
class EpisodeBuffers:
    episode_idx: int
    pose_names: list[str]                 # [B] pose basename per env for this episode ("0000", "0001", ...)
    timestamps_list: list[float]               # [T]
    root_pos_w_list: list[np.ndarray]          # list of [B,3]
    root_rotvec_w_list: list[np.ndarray]       # list of [B,3]
    spine_pos_w_list: list[np.ndarray]         # list of [B,3]
    spine_rotvec_w_list: list[np.ndarray]      # list of [B,3]
    body_pos_w_list: list[np.ndarray]          # list of [B,Nb,3]
    body_rotvec_w_list: list[np.ndarray]       # list of [B,Nb,3]
    pose_body_rotvec_list: list[np.ndarray]    # list of [B,J,3]
    pressure_map_list: list[np.ndarray]        # list of [B,Ncells,H,W]   

    H: Optional[int]            = None
    W: Optional[int]            = None
    num_cells: Optional[int]    = 32

    def reset(self, episode_idx: int, pose_names: list[str]):
        self.episode_idx = int(episode_idx)
        self.pose_names = list(pose_names)
        self.timestamps_list.clear()
        self.root_pos_w_list.clear()
        self.root_rotvec_w_list.clear()
        self.spine_pos_w_list.clear()
        self.spine_rotvec_w_list.clear()
        self.body_pos_w_list.clear()
        self.body_rotvec_w_list.clear()
        self.pose_body_rotvec_list.clear()
        self.pressure_map_list.clear()
        self.H = None
        self.W = None

    def append_frame(
        self, 
        timestamp: float, 
        root_pos_w: np.ndarray,
        root_rotvec_w: np.ndarray,
        spine_pos_w: np.ndarray,
        spine_rotvec_w: np.ndarray,
        body_pos_w: np.ndarray,
        body_rotvec_w: np.ndarray,
        pose_body_rotvec: np.ndarray,
        pressure_map: np.ndarray,
    ):
        # checks to see if data shapes are valid
        if pressure_map.ndim != 4:
            raise RuntimeError(f"pressure must be [B,Ncells,H,W], got {pressure_map.shape}")
        B, num_cells, H, W = pressure_map.shape
        if num_cells != self.num_cells:
            raise RuntimeError(f"Expected Ncells={self.num_cells}, got {num_cells}")        
        if self.H is None:
            self.H, self.W = int(H), int(W)
        # appending the frame. 
        self.timestamps_list.append(float(timestamp))
        self.root_pos_w_list.append(root_pos_w.astype(np.float32))
        self.root_rotvec_w_list.append(root_rotvec_w.astype(np.float32))
        self.spine_pos_w_list.append(spine_pos_w.astype(np.float32))
        self.spine_rotvec_w_list.append(spine_rotvec_w.astype(np.float32))
        self.body_pos_w_list.append(body_pos_w.astype(np.float32))
        self.body_rotvec_w_list.append(body_rotvec_w.astype(np.float32))
        self.pose_body_rotvec_list.append(pose_body_rotvec.astype(np.float32))
        self.pressure_map_list.append(pressure_map.astype(np.float32))

    def save(
        self, 
        out_dir: str, 
        body_names: list[str], 
        smpl_joint_order: list[str]
    ):
        
        if len(self.timestamps_list) == 0: 
            raise RuntimeError(f"Episode {self.episode_idx}: is empty... computed data.")
        
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"episode_{self.episode_idx:04d}_B{len(self.pose_names)}_gt.npz") # TODO: write another script that reads this type of saving and then converts it to 0001.npz, 0002.npz, ... 

        timestamps_np       = np.asarray(self.timestamps_list, dtype=np.float32)                # [T]
        root_pos_w_np       = np.stack(self.root_pos_w_list, axis=0)                            # [T,B,3]
        root_rotvec_w_np    = np.stack(self.root_rotvec_w_list, axis=0)                         # [T,B,3]
        spine_pos_w_np      = np.stack(self.spine_pos_w_list, axis=0)                           # [T,B,3]
        spine_rotvec_w_np   = np.stack(self.spine_rotvec_w_list, axis=0)                        # [T,B,3]
        body_pos_w_np       = np.stack(self.body_pos_w_list, axis=0)                            # [T,B,Nb,3]
        body_rotvec_w_np    = np.stack(self.body_rotvec_w_list, axis=0)                         # [T,B,Nb,3]
        pose_body_rotvec_np = np.stack(self.pose_body_rotvec_list, axis=0)                      # [T,B,J,3]
        pressure_maps_np    = np.stack(self.pressure_map_list, axis=0)                          # [T,B,Ncells,H,W]

        np.savez_compressed(
            out_path,
            timestamps          = timestamps_np,
            pose_names          = np.asarray(self.pose_names),          # [B]
            body_names          = np.asarray(body_names),
            smpl_joint_order    = np.asarray(smpl_joint_order),
            root_pos_w          = root_pos_w_np,
            root_rotvec_w       = root_rotvec_w_np,
            spine_pos_w         = spine_pos_w_np,
            spine_rotvec_w      = spine_rotvec_w_np,
            body_pos_w          = body_pos_w_np,
            body_rotvec_w       = body_rotvec_w_np,
            pose_body_rotvec    = pose_body_rotvec_np,
            pressure_maps       = pressure_maps_np,
        )
        print(f"[INFO] Saved GT episode {self.episode_idx} -> {out_path}")


# ------------------------------------
# 3) INTERACTIVE SCENE BUILDER CFG ---
# ------------------------------------
@configclass
class KorusInteractiveSceneCfg(InteractiveSceneCfg):
    """configuration for a korus bed scene"""

    # --- ground (global) --- 
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
        ),
    )

    # --- dome lights (global) --- 
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # --- bed articulation ---  
    bed: ArticulationCfg = KORUS_BED_CFG.replace(prim_path="{ENV_REGEX_NS}/KorusBed") # type: ignore

    # --- humanoid articulation --- 
    humanoid: ArticulationCfg = KORUS_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Humanoid") # type: ignore

    # --- deformable foam(s) ---
    deformable_0: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid0",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[0]),
    )

    deformable_1: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid1",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[1]),
    )

    deformable_2: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid2",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[2]),
    )

    deformable_3: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid3",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[3]),
    )

    deformable_4: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid4",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[4]),
    )

    deformable_5: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid5",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[5]),
    )

    deformable_6: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid6",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[6]),
    )

    deformable_7: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid7",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[7]),
    )

    deformable_8: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid8",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[8]),
    )

    deformable_9: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid9",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[9]),
    )

    deformable_10: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid10",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[10]),
    )

    deformable_11: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid11",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[11]),
    )

    deformable_12: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid12",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[12]),
    )

    deformable_13: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid13",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[13]),
    )

    deformable_14: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid14",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[14]),
    )

    deformable_15: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid15",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[15]),
    )

    deformable_16: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid16",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[16]),
    )

    deformable_17: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid17",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[17]),
    )

    deformable_18: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid18",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[18]),
    )

    deformable_19: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid19",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[19]),
    )

    deformable_20: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid20",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[20]),
    )

    deformable_21: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid21",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[21]),
    )

    deformable_22: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid22",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[22]),
    )

    deformable_23: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid23",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[23]),
    )

    deformable_24: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid24",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[24]),
    )

    deformable_25: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid25",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[25]),
    )

    deformable_26: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid26",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[26]),
    )

    deformable_27: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid27",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[27]),
    )

    deformable_28: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid28",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[28]),
    )

    deformable_29: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid29",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[29]),
    )

    deformable_30: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid30",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[30]),
    )

    deformable_31: DeformableObjectCfg = KORUS_FOAM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/KorusBed/DeformableCuboid31",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN_LIST[31]),
    )

    # Rigid sphere (falling on the deformable cuboid)
    if args_cli.calibration: 
        sphere: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/RigidSphere",
            spawn=sim_utils.SphereCfg(
                radius=0.4,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    metallic=0.2,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(FOAM_ORIGIN_LIST[0][0], FOAM_ORIGIN_LIST[0][1], 3.0)
            ),
        )

# --------------------------------------------------------------------------
# 4) Some helpers --- NOTE: shift them to a helper script and import them 
# --------------------------------------------------------------------------
def list_pose_npzs(npz_dir: str, glob_pat: str = "*.npz") -> list[str]:
    """ makes a list of all the file paths in the npz pose directory"""
    files = sorted([os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")])
    return files

def make_round_robin_batches(npz_files: list[str], num_envs: int) -> list[list[Optional[str]]]:
    """
    Returns episodes as a list of length E.
    Example: files [0..7], B=4 => episodes:
      ep0: [0,1,2,3]
      ep1: [4,5,6,7]
    """
    B = int(num_envs)
    N = len(npz_files)
    E = (N + B - 1) // B
    episodes: list[list[Optional[str]]] = []
    for k in range(E):
        ep: list[Optional[str]] = []
        for e in range(B):
            idx = k * B + e
            ep.append(npz_files[idx] if idx < N else None)
        episodes.append(ep)
    return episodes

def load_root_from_pose_npz(path: str):
    """
    Returns:
      trans:  (3,) float32
      rotvec: (3,) float32
    """
    pose_npz = np.load(path, allow_pickle=True)
    if "trans" in pose_npz:
        trans = pose_npz["trans"][0]   
    else:
        raise RuntimeError(f"Global translation vector not found in the pose file")
    if "poses" in pose_npz:
        rotvec = pose_npz["poses"][0]
    else: 
        raise RuntimeError(f"Global rotation vector not found in the pose file")
    return trans, rotvec

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """runs the simulation loop"""
    # --- num envs ---
    B = int(scene.num_envs)

    # --- create a episode manager so that it only iterates over newer files --- 
    if args_cli.save_data:
        manager = EpisodeResumeManager(
            record_root=args_cli.record_dir,
            in_pose_dir=args_cli.in_pose_dir,
            override=args_cli.override,
            expected_B=B,
            mirror_after="data_npz",
            verbose=True,
        )
        record_dir         = manager.record_dir
        resume_episode_idx = manager.resume_episode_idx
    else:
        manager = None
        record_dir         = args_cli.record_dir
        resume_episode_idx = 0

    # --- root randomizer for humanoid --- 
    root_rand = RootRandomizer(
        RootRandomizerCfg(
            base_xyz = (0.0, 1., 3.0),              # mean in ENV frame
            z_value     = 3.0,                      # constant Z
            x_std       = 0.5   , y_std     = 1.0,  # meters
            x_clip      = 0.6   , y_clip    = 1.6,  # meters
            yaw_base    = 0.0,                      # radians
            yaw_std     = math.radians(20.0),       # radians
            yaw_clip    = math.radians(20.0),       # radians
            seed        = 0,
        )
    )

    # --- extracting entities --- 
    bed: Articulation               = scene["bed"]
    humanoid: Articulation          = scene["humanoid"]
    cubes: list[DeformableObject]   = [scene[f"deformable_{ii}"] for ii in range(32)]
    sphere: Optional[RigidObject]   = scene["sphere"] if args_cli.calibration else None    

    # --- pose file scheduling (round robin by env)---
    pose_dir    = args_cli.in_pose_dir
    pose_files  = list_pose_npzs(pose_dir)
    if len(pose_files) == 0:
        raise RuntimeError(f"No .npz pose files found in {pose_dir}.")
    if (len(pose_files) % B) != 0:
        raise RuntimeError(f"N pose files ({len(pose_files)}) must be divisible by num_envs ({B}) for clean episodes.")
    # episodes: list of lists, each length B (no None)
    episodes: list[list[str]] = []
    for k in range(len(pose_files) // B):
        episodes.append([pose_files[k * B + env_num] for env_num in range(B)])
    print(f"[INFO] Found {len(pose_files)} pose files -> {len(episodes)} episodes for B={B} envs")

    # --- humanoid joint mapping helpers --- 
    joint_names     = humanoid.data.joint_names
    joint_index_map = {name: i for i, name in enumerate(joint_names)}    
    body_names_humanoid = humanoid.data.body_names
    pelvis_idx = body_names_humanoid.index("Pelvis")

    # --- mass scaling for humanoid --- 
    KNEE_ANKLE_NAMES = ["L_Knee", "R_Knee", "L_Ankle", "R_Ankle"]
    knee_ankle_ids = [body_names_humanoid.index(n) for n in KNEE_ANKLE_NAMES]
    baseline_masses = None  

    # --- timing ---
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # --- top-plate-idx -> bed-asset-id mapping ---   
    top_plate_body_ids: list[int] = []
    for idx in range(32):
        ids, _ = bed.find_bodies(f"TopPlate{idx}")
        if not ids:
            raise RuntimeError(f"TopPlate{idx} not found in bed bodies")
        top_plate_body_ids.append(ids[0])

    # --- initializing caches (re-built on reset) --- 
    anchor_caches: list[AnchorCache]        = []
    pressure_caches: list[PressureCache]    = []

    # --- episode recorder --- 
    if args_cli.save_data: 
        episode_buffer = EpisodeBuffers(
            episode_idx             = 0,
            pose_names              = [""] * B,
            timestamps_list         = [],
            root_pos_w_list         = [],
            root_rotvec_w_list      = [],
            spine_pos_w_list        = [],
            spine_rotvec_w_list     = [],
            body_pos_w_list         = [],
            body_rotvec_w_list      = [],
            pose_body_rotvec_list   = [],
            pressure_map_list       = [],
            num_cells               = 32
        )

    # --- ros publisher block --- 
    indent_pubs: list = []
    if args_cli.ros: 
        if not rclpy.ok():
            rclpy.init(args=None)
        ros_node = Node("korus_ros_api")
        
        for idx in range(len(cubes)):
            topic = f"/foam_bed/top_surface/indentation_grid_{idx:02d}"
            indent_pubs.append(ros_node.create_publisher(Float32MultiArray, topic, 10))
        
        def publish_indent(idx: int, grid_t: torch.Tensor):
            """grid_t: (Hm,Wm) torch tensor"""
            msg = Float32MultiArray()
            Hm = int(grid_t.shape[0])
            Wm = int(grid_t.shape[1])
            msg.layout.dim = [
                MultiArrayDimension(label="rows", size=Hm, stride=Hm * Wm),
                MultiArrayDimension(label="cols", size=Wm, stride=Wm),
            ]
            msg.data = grid_t.detach().float().cpu().reshape(-1).tolist()
            indent_pubs[idx].publish(msg)

    # --- episode indices ---
    episode_idx = resume_episode_idx
    current_episode = resume_episode_idx
    target_humanoid_joint_angles = None

    # --- save last episode safely ---
    def save_last_episode():
        if args_cli.save_data and episode_buffer is not None:
            if len(episode_buffer.timestamps_list) > 0:
                episode_buffer.save(
                    record_dir,
                    body_names=body_names_humanoid,
                    smpl_joint_order=SMPL_BODY_JOINT_ORDER,
                )

    # --- simulation loop starts --- 
    while simulation_app.is_running():

        # --- check through record directory if previous data exists --- 
        if args_cli.save_data and manager is not None and manager.is_done(total_episodes=len(episodes)):
            print(f"[INFO] All episodes already completed in {record_dir}. Exiting.")
            return
    
        # ---------------
        # RESET BLOCK ---
        # ---------------
        if (count % args_cli.reset_after) == 0:
            
            if count != 0:
                save_last_episode()

            # --- check if out of episodes --- 
            if current_episode >= len(episodes):
                print("[INFO] Completed all episodes. Exiting.")
                break

            # --- pick pose paths for the current episode --- 
            pose_paths = episodes[current_episode]
            pose_names = []
            for env_num in range(B):
                if pose_paths[env_num] is None: 
                    raise RuntimeError(f"Episode {current_episode}, missing pose for Env {env_num}")
                pose_names.append(os.path.splitext(os.path.basename(pose_paths[env_num]))[0]) # type:ignore
            current_episode += 1  # always advance
            
            # --- reset epidode buffers --- 
            if args_cli.save_data and episode_buffer is not None:
                episode_buffer.reset(episode_idx = episode_idx, pose_names = pose_names)
                print(f"[INFO] Episode {episode_idx:04d} uses poses: {pose_names}")

            # --- reset counters --- 
            sim_time = 0.0
            count = 0

            # --- reset sphere (if present)---
            if args_cli.calibration: 
                if sphere is not None:
                    root_state_sphere = sphere.data.default_root_state.clone()
                    root_state_sphere[:, :3] += scene.env_origins
                    sphere.write_root_pose_to_sim(root_state_sphere[:, :7])
                    sphere.write_root_velocity_to_sim(root_state_sphere[:, 7:])

            # --- reset bed ---    
            root_state_bed = bed.data.default_root_state.clone()
            root_state_bed[:,:3] += scene.env_origins 
            bed.write_root_pose_to_sim(root_state_bed[:, :7])
            bed.write_root_velocity_to_sim(root_state_bed[:, 7:])
            joint_pos_bed = bed.data.default_joint_pos.clone()
            joint_vel_bed = bed.data.default_joint_vel.clone()
            bed.write_joint_state_to_sim(joint_pos_bed, joint_vel_bed) 
            # # inflate cell number 13 # NOTE: uncomment this block if you want cell inflation, and comment the above line.  
            # cell_13_idx = 11  # joint index for PrismaticJoint13
            # inflate_min = 0.0
            # inflate_max = 0.9
            # inflate_amount:float = 0.01  # start value
            # inflate_step:float = 0.01    # +0.01 per sim step (tune)
            # joint_pos_bed[:, cell_13_idx] = inflate_amount
            bed.write_joint_state_to_sim(joint_pos_bed, joint_vel_bed)
            bed.set_joint_position_target(joint_pos_bed)
            # bed_joint_target = joint_pos_bed.clone()

            # --- reset humanoid --- 
            # build target humanoid joint angles pre env
            target_humanoid_joint_angles = humanoid.data.default_joint_pos.clone()
            for env_num in range(B):
                joint_cfg = build_joint_config_from_npz(
                    path = str(pose_paths[env_num]), 
                    SMPL_TO_ISAAC = SMPL_TO_ISAAC, 
                    SMPL_BODY_JOINT_ORDER = SMPL_BODY_JOINT_ORDER
                )
                for joint_name, angle in joint_cfg.items():
                    j = joint_index_map.get(joint_name)
                    if j is not None: 
                        target_humanoid_joint_angles[env_num, j] = float(angle)
                    else:
                        RuntimeError(f"Joint not found")            

            # --- clamp joint targets --- 
            with torch.no_grad():
                lim = humanoid.data.joint_pos_limits
                if lim.shape[0] == 1:
                    lo = lim[0, :, 0]  # (Nj,)
                    hi = lim[0, :, 1]
                else:
                    lo = lim[:, :, 0]  # (B, Nj)
                    hi = lim[:, :, 1]
                # wrap angles to [-pi, pi] to avoid +180/-180 artifacts
                target_humanoid_joint_angles = (target_humanoid_joint_angles + math.pi) % (2.0 * math.pi) - math.pi
                # for "almost locked" DOFs (tiny range), force them to the midpoint (usually ~0)
                tiny_range = math.radians(10.0)  # tune: 5-15 deg works well
                if lo.ndim == 1:
                    mid = 0.5 * (lo + hi)
                    locked = (hi - lo) < tiny_range           # (Nj,)
                    target_humanoid_joint_angles[:, locked] = mid[locked]
                    # 3) Clamp to limits
                    target_humanoid_joint_angles = torch.clamp(
                        target_humanoid_joint_angles,
                        lo.unsqueeze(0),
                        hi.unsqueeze(0),
                    )
                else:
                    mid = 0.5 * (lo + hi)
                    locked = (hi - lo) < tiny_range           # (B, Nj)
                    target_humanoid_joint_angles[locked] = mid[locked]
                    target_humanoid_joint_angles = torch.max(torch.min(target_humanoid_joint_angles, hi), lo)
                # knee flexion sign: if your knee limits are like [-135, +30], but you want +135 flexion, flip just the knee hinge axis.
                for nm in ["L_Knee_x", "R_Knee_x"]:
                    if nm in joint_index_map:
                        j = joint_index_map[nm]
                        # try flipping; keep it if it reduces violation
                        if lo.ndim == 1:
                            t = target_humanoid_joint_angles[:, j]
                            t_flip = -t
                            ok_flip = (t_flip >= lo[j]) & (t_flip <= hi[j])
                            target_humanoid_joint_angles[:, j] = torch.where(ok_flip, t_flip, t)
                        else:
                            t = target_humanoid_joint_angles[:, j]
                            t_flip = -t
                            ok_flip = (t_flip >= lo[:, j]) & (t_flip <= hi[:, j])
                            target_humanoid_joint_angles[:, j] = torch.where(ok_flip, t_flip, t)
            
            # reset humanoid
            root_state_humanoid = humanoid.data.default_root_state.clone()
            root_pos_env_np, root_quat_env_np = root_rand.sample(B)  # (B,3), (B,4)
            root_pos_env_torch  = torch.from_numpy(root_pos_env_np).to(humanoid.device, dtype=root_state_humanoid.dtype)
            root_quat_env_torch = torch.from_numpy(root_quat_env_np).to(humanoid.device, dtype=root_state_humanoid.dtype)
            # root_state_humanoid[:,:3] += scene.env_origins
            root_state_humanoid[:, :3] = scene.env_origins + root_pos_env_torch
            root_state_humanoid[:, 3:7] = root_quat_env_torch
            humanoid.write_root_pose_to_sim(root_state_humanoid[:, :7])
            humanoid.write_root_velocity_to_sim(root_state_humanoid[:, 7:])
            joint_pos_humanoid = target_humanoid_joint_angles.clone()  
            joint_vel_humanoid = torch.zeros_like(joint_pos_humanoid)
            humanoid.write_joint_state_to_sim(joint_pos_humanoid, joint_vel_humanoid)
            # set joint position
            humanoid.set_joint_position_target(joint_pos_humanoid)

            # --- mass override --- 
            # NOTE: these are tuning parameters for good pressure distribution
            # NOTE: All body names: ['Spine', 'Torso', 'Chest', 'Pelvis', 'Neck', 'L_Thorax', 'R_Thorax', 'L_Hip', 'R_Hip', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Knee', 'R_Knee', 'L_Elbow', 'R_Elbow', 'L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist', 'L_Toe', 'R_Toe', 'L_Hand', 'R_Hand']
            # mass factors (tune as per need)
            KNEE_ANKLE_MASS_FACTOR      = 4.0
            CHEST_PELVIS_MASS_FACTOR    = 0.5
            ELBOW_WRIST_MASS_FACTOR     = 4.0
            HEAD_MASS_FACTOR            = 3.0
            # get current masses
            masses = humanoid.root_physx_view.get_masses()
            if baseline_masses is None:
                baseline_masses = masses.detach().clone()
            # formulate new masses based on defined factors
            new_masses = baseline_masses.detach().clone()
            name_to_id = {n: i for i, n in enumerate(body_names_humanoid)}
            knee_ankle_toe_ids = torch.tensor(
                [
                    name_to_id["L_Knee"]    ,   name_to_id["R_Knee"],
                    name_to_id["L_Ankle"]   ,   name_to_id["R_Ankle"],
                    name_to_id['L_Toe']     ,   name_to_id['R_Toe'],
                ], device = new_masses.device, dtype = torch.long,
            )
            elbow_wrist_ids = torch.tensor(
                [
                    name_to_id['L_Elbow']   ,   name_to_id['R_Elbow'],
                    name_to_id['L_Wrist']   ,   name_to_id['R_Wrist'],
                ], device = new_masses.device, dtype = torch.long,
            )
            head_ids = torch.tensor(
                [
                    name_to_id['Head']
                ], device = new_masses.device, dtype = torch.long,
            )
            chest_pelvis_ids = torch.tensor(
                [
                    name_to_id["Chest"],
                    name_to_id["Pelvis"],
                ], device = new_masses.device, dtype = torch.long,
            )
            new_masses[:, knee_ankle_toe_ids]   = baseline_masses[:, knee_ankle_toe_ids] * KNEE_ANKLE_MASS_FACTOR
            new_masses[:, chest_pelvis_ids]     = baseline_masses[:, chest_pelvis_ids]   * CHEST_PELVIS_MASS_FACTOR
            new_masses[:, elbow_wrist_ids]      = baseline_masses[:, elbow_wrist_ids]    * ELBOW_WRIST_MASS_FACTOR
            new_masses[:, head_ids]             = baseline_masses[:, head_ids]           * HEAD_MASS_FACTOR
            # apply new masses
            indices = torch.arange(new_masses.shape[0], device=new_masses.device, dtype=torch.int32)
            humanoid.root_physx_view.set_masses(new_masses.to(torch.float32).contiguous(), indices)

            # --- reset deformables (all 32) --- 
            env_origins = scene.env_origins
            for ii, cube in enumerate(cubes):
                nodal_state = cube.data.default_nodal_state_w.clone()
                base = torch.as_tensor(FOAM_ORIGIN_LIST[ii], dtype=nodal_state.dtype, device=cube.device)
                desired_mean = env_origins + base
                cur_mean = nodal_state[..., :3].mean(dim=1)
                delta = desired_mean - cur_mean  
                # NOTE: ignore z position in this                
                nodal_state[..., 0] += delta[:, 0].unsqueeze(1)
                nodal_state[..., 1] += delta[:, 1].unsqueeze(1)
                nodal_state[..., 3:] = 0.0
                cube.write_nodal_state_to_sim(nodal_state)

            # --- clean internal buffers --- 
            scene.reset()
            
            # --- build anchor caches --- TODO: all of this can be wrapped in one function and can be made part of GridBuilder ---
            anchor_caches.clear()
            for idx, cube in enumerate(cubes):
                # find bottom surface plane (robust to mesh)
                default_nodal_positions = cube.data.default_nodal_state_w[..., :3][0]
                default_nodal_positions_z = default_nodal_positions[:, 2]
                s = 10 ** int(CFG.decimals)
                default_nodal_positions_z_rounded = torch.round(default_nodal_positions_z * s) / s
                planes = torch.sort(torch.unique(default_nodal_positions_z_rounded)).values
                z_bottom = planes[0]
                bottom_surface_idxs = GridBuilder.get_surface_indices_by_z(cube, z_target=z_bottom, decimals=CFG.decimals)
                if bottom_surface_idxs.numel() == 0:
                    raise RuntimeError(f"[Cell {idx}] bottom nodes not found.")
                # pick 4 corner node indices on that bottom surface
                bottom_corners = GridBuilder.corners_from_surface_idx(cube, surface_idx=bottom_surface_idxs, env_id=0)  # (4,)
                # build kin target: (B,N,4) = [x,y,z,flag]
                kinematic_targets = cube.data.nodal_kinematic_target.clone()
                kinematic_targets[..., :3] = cube.data.nodal_pos_w     # current sim nodal pos
                kinematic_targets[..., 3]  = 1.0                       # free all nodes
                kinematic_targets[:, bottom_corners, 3] = 0.0          # constrain only the 4 corners
                # initialize corner targets from current plate position
                body_id = top_plate_body_ids[idx]
                plate_position = bed.data.body_link_pos_w[:, body_id, :3]  # (B,3)
                plate_corners = compute_plate_corners(plate_position)  # (B,4,3)
                kinematic_targets[:, bottom_corners, :3] = plate_corners
                cube.write_nodal_kinematic_target_to_sim(kinematic_targets)
                anchor_caches.append(
                    AnchorCache(
                        bottom_corners_idxs = bottom_corners,
                        kinematic_targets   = kinematic_targets,
                        corners_buffer      = plate_corners.clone(),
                        last_plate_position = plate_position.detach().clone(),
                    )
                )

            # --- let constraints settle before building indentation baseline ---
            for _ in range(args_cli.settle_steps):
                scene.write_data_to_sim()
                sim.step(render=True)          
                scene.update(sim_dt)

            # --- build pressure caches --- 
            pressure_caches.clear()
            for idx, cube in enumerate(cubes):                 
                # find all top surface indices, and arrange them into a pressure grid like 2d array
                simulation_grid_top_surface_idxs_np = GridBuilder.build_sim_grid(cube, decimals=CFG.decimals)
                # plate position for this cube
                body_id = top_plate_body_ids[idx]
                sim_baseline_pos = cube.data.nodal_pos_w.detach().clone()        # (B,Nv,3)
                plate_position0  = bed.data.body_link_pos_w[:, body_id, :3].detach().clone()  # (B,3)
                # compute indent cache                
                indent_cache = PressureMapper.precompute_top_surface_indentation_cache(
                    simulation_grid_top_surface_idxs_np = simulation_grid_top_surface_idxs_np,
                    sim_default_nodal_positions_w       = sim_baseline_pos,
                    plate_pos_w                         = plate_position0,
                )
                # TODO: REMOVE THIS LATER AND CHANGE THE PRESSURE CACHE STRUCTURE TO INCLUDE INDENTATION 
                # find all valid tetrahedrals in the deformable cube                
                tetrahedrals_all_torch: torch.Tensor = cube.root_physx_view.get_sim_element_indices()[0] 
                valid_mask_torch = (tetrahedrals_all_torch >= 0).all(dim=1)
                valid_tetrahedrals_all_torch = tetrahedrals_all_torch[valid_mask_torch]
                valid_tetrahedrals_all_np = valid_tetrahedrals_all_torch.detach().cpu().numpy()
                top_triangle_nodes_np, top_triangle_elements_np, cell_ids_np, Hm, Wm = PressureMapper.precompute_top_surface_boundary_triangle_arrays(
                    valid_tetrahedrals_all_np,
                    simulation_grid_top_surface_idxs_np,
                )
                # append pressure cache to collection
                pressure_caches.append(
                    PressureCache(
                        valid_mask              = valid_mask_torch, 
                        top_triangle_nodes      = torch.as_tensor(top_triangle_nodes_np, device=cube.device, dtype=torch.long), 
                        top_triangle_elements   = torch.as_tensor(top_triangle_elements_np, device=cube.device, dtype=torch.long),  
                        cell_ids                = torch.as_tensor(cell_ids_np, device=cube.device, dtype=torch.long), 
                        Hm                      = Hm, 
                        Wm                      = Wm,
                        indent_cache            = indent_cache,
                    )
                )

            # --- print debug --- 
            print(f"[INFO]: Resetting Bed + Humanoid + Re-anchoring Deformable Foams + Built pressure caches for {len(pressure_caches)} cubes.")
            episode_idx += 1

        # -------------------------
        # STEP AND UPDATE BLOCK ---
        # -------------------------
        
        # --- anchoring deformables to the top plate (if the plate has moved enough) --- 
        do_anchor_write = (args_cli.anchor_every <= 1) or (count % args_cli.anchor_every == 0)
        if do_anchor_write and anchor_caches:
            for idx, cube in enumerate(cubes):
                anchor_cache = anchor_caches[idx]
                body_id = top_plate_body_ids[idx]
                plate_position = bed.data.body_link_pos_w[:, body_id, :3]  # (B,3)
                if plate_has_moved_enough(last=anchor_cache.last_plate_position, current=plate_position, epsilon=float(args_cli.plate_pos_eps)):
                    plate_corners = compute_plate_corners(plate_position)  # (B,4,3)
                    anchor_cache.kinematic_targets[:, anchor_cache.bottom_corners_idxs, :3] = plate_corners
                    cube.write_nodal_kinematic_target_to_sim(anchor_cache.kinematic_targets)
                    anchor_cache.last_plate_position = plate_position.detach().clone()

        # --- compute motion capture data ---
        compute_pressure    = (count % args_cli.pressure_every) == 0
        save_this_frame = compute_pressure and (
            (count != 0) or (args_cli.pressure_every == args_cli.reset_after)
        )
        if compute_pressure:
            if not pressure_caches: 
                raise RuntimeError(f"Pressure caches not built yet. Did reset/caching fail?")

            # --- compute mocap ---
            # compute pelvis pose
            pelvis_pose = humanoid.data.body_link_pose_w[:, pelvis_idx, :]
            pelvis_pose_np = pelvis_pose.detach().cpu().numpy().astype(np.float32)
            root_pos_w = pelvis_pose_np[:, :3]
            root_rotvec_w = np.stack([quat_wxyz_to_rotvec(pelvis_pose_np[env_num, 3:]) for env_num in range(B)], axis=0).astype(np.float32) 
            # spine/root pose:
            spine_pose = humanoid.data.root_pose_w[:, :]  # [B,7]
            spine_pose_np = spine_pose.detach().cpu().numpy().astype(np.float32)
            spine_pos_w = spine_pose_np[:, :3]
            spine_rotvec_w = np.stack([quat_wxyz_to_rotvec(spine_pose_np[e, 3:]) for e in range(B)], axis=0).astype(np.float32)
            # body link poses:
            body_link_pos_w    = humanoid.data.body_link_pos_w[:, :, :].detach().cpu().numpy().astype(np.float32)     # [B,Nb,3]
            body_link_quat_w   = humanoid.data.body_link_quat_w[:, :, :].detach().cpu().numpy().astype(np.float32)  # [B,Nb,4] wxyz
            num_bodies = body_link_pos_w.shape[1]
            body_link_rotvec_w = np.zeros((B, num_bodies, 3), dtype=np.float32)
            for env_num in range(B):
                for i in range(num_bodies):
                    body_link_rotvec_w[env_num, i] = quat_wxyz_to_rotvec(body_link_quat_w[env_num, i])
            # pose_body_rotvec (SMPL style) from joint_pos euler triplets
            joint_pos_humanoid_np = humanoid.data.joint_pos[:, :].detach().cpu().numpy().astype(np.float32)  # [B,Nj]
            J = len(SMPL_BODY_JOINT_ORDER)
            pose_body_rotvec = np.zeros((B, J, 3), dtype=np.float32)
            for env_num in range(B):
                for joint_idx, smpl_name in enumerate(SMPL_BODY_JOINT_ORDER):
                    jx, jy, jz = SMPL_TO_ISAAC[smpl_name]
                    ix = joint_index_map.get(jx); iy = joint_index_map.get(jy); iz = joint_index_map.get(jz)
                    if ix is None or iy is None or iz is None:
                        raise RuntimeError(f"Joint triplet not found for SMPL joint '{smpl_name}")
                    e_xyz = np.array([joint_pos_humanoid_np[env_num, ix], joint_pos_humanoid_np[env_num, iy], joint_pos_humanoid_np[env_num, iz]], dtype=np.float32)
                    pose_body_rotvec[env_num, joint_idx] = euler_xyz_to_rotvec(e_xyz)

            # --- computing pressure map ---
            pressure_maps_pa_bed = []       
            for idx, cube in enumerate(cubes): 
                # extract pressure cache    
                pressure_cache = pressure_caches[idx]
                # compute indentation map
                sim_pos_w   = cube.data.nodal_pos_w                         
                body_id = top_plate_body_ids[idx]
                plate_pos_w = bed.data.body_link_pos_w[:, body_id, :3]      
                indentation_map = PressureMapper.compute_indentation_grid(
                    indentation_cache = pressure_cache.indent_cache,
                    sim_nodal_pos_w   = sim_pos_w,
                    plate_pos_w       = plate_pos_w,
                    out               = "cell",
                    reduce            = "mean",
                ) 
                # compute pressure map based on indentation map using non-linear modelling
                k_area  = 2.0e6              # Pa/m. TODO: calibrate based on pressure range. 
                n       = 1.0                # TODO: currently linear tune later or change the model later for true sensor behaviour. 
                pressure_maps_pa_cell = PressureMapper.indentation_to_pressure_grid(
                    indent_cell  = indentation_map,   # batched
                    k_area       = k_area,
                    n            = n,
                    clamp_max_pa = None
                )  
                # publish pressure map as a ros2 topic (if enabled)
                if args_cli.ros:
                    # _publish_indent(idx, indentation_map) # publish indentation map
                    publish_indent(idx, pressure_maps_pa_cell[0]) # publish pressure map for the cell NOTE: this is just for debuggin   
                pressure_maps_pa_bed.append(pressure_maps_pa_cell.detach().cpu().numpy().astype(np.float32))
            pressure_maps_bed = np.stack(pressure_maps_pa_bed, axis=0).transpose(1, 0, 2, 3)
            
            # --- record data ---
            if (save_this_frame) and (args_cli.save_data) and (episode_buffer is not None):
                episode_buffer.append_frame(
                    timestamp           = float(sim_time),
                    root_pos_w          = root_pos_w,           # pelvis position wrt World Frame. (SMPL Parameter)
                    root_rotvec_w       = root_rotvec_w,        # pelvis rotation (in rotation vector form) wrt World Frame (SMPL Parameter)
                    spine_pos_w         = spine_pos_w,           
                    spine_rotvec_w      = spine_rotvec_w,       
                    body_pos_w          = body_link_pos_w,       
                    body_rotvec_w       = body_link_rotvec_w,
                    pose_body_rotvec    = pose_body_rotvec,
                    pressure_map        = pressure_maps_bed,
                )

            # spin ros
            if args_cli.ros:
                rclpy.spin_once(ros_node, timeout_sec=0.0) 

        # # --- incremental cell inflation --- (NOTE: this is temp code, uncomment if you want incremental cell inflation)            
        # inflate_every = 10  # 1 = every step
        # if (count % inflate_every) == 0:
        #     inflate_amount = min(inflate_amount + inflate_step, inflate_max) # type: ignore
        #     bed_joint_target[:, cell_13_idx] = inflate_amount
        #     bed.write_joint_state_to_sim(bed_joint_target, torch.zeros_like(bed_joint_target))
        #     bed.set_joint_position_target(bed_joint_target)

        # --- step and update --- 
        # write data to sim
        scene.write_data_to_sim()
        # step through the simulation
        sim.step(render=bool(args_cli.render)) 
        # increment counters
        sim_time += sim_dt 
        count += 1
        # update all buffers
        scene.update(sim_dt)

    # --- save the final episode if we exit without reset --- 
    save_last_episode()

    # --- ros shutdown --- 
    if args_cli.ros and ros_node is not None:
        ros_node.destroy_node()
        rclpy.shutdown()

def main():
    """main function"""

    # --- load kit helper ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # --- set main camera ---  
    sim.set_camera_view(eye=(3.0, -3.0, 2.0), target=(0.0, 0.0, 0.5))

    # --- design scene ---     
    scene_cfg = KorusInteractiveSceneCfg(
        num_envs            = args_cli.num_envs,
        env_spacing         = args_cli.env_spacing, 
        replicate_physics   = False, 
        filter_collisions   = True
    )
    scene = InteractiveScene(scene_cfg)
    scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

    # --- play the simulator --- 
    sim.reset()
    scene.reset()

    print(f"[INFO]: Setup Complete...")

    # --- run simulator --- 
    run_simulator(sim, scene)

if __name__ == "__main__":
    # --- run the main function --- 
    main()
    # --- close the simulation app --- 
    simulation_app.close()
    




