# --------------------------------
# 0) IMPORTS FOR APP CREATION ---
# --------------------------------
import argparse
from isaaclab.app import AppLauncher

# ----------------------------------------------
# 1) LAUNCH THE ISAACSIM APP (with argparse) ---
# ----------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab + YAML config")
parser.add_argument("--num_envs"       , type=int  , default=1                                      , help="Number of Instances in the Interactive Scene")
parser.add_argument("--env_spacing"    , type=float, default=2.0                                    , help="Physical Space between the Environments (in meters)")
parser.add_argument("--num_rows"       , type=int  , default=None                                   , help="override YAML rows")
parser.add_argument("--num_cols"       , type=int  , default=None                                   , help="override YAML cols")
parser.add_argument("--pressure_every" , type=int  , default=1                                      , help="ROS publish + record every N frames")
parser.add_argument("--config"         , type=str  , default="scripts/korus/config/korus_bed.yaml"  , help="YAML file with globals")

parser.add_argument("--ros"            , action="store_true"                                        , help="Enable ROS2; Note that this feature is helpful for real-time debugging")  
parser.add_argument("--calibration"    , action="store_true"                                        , help="This will drop a sphere on the on the first cell for debugging")  


grp = parser.add_mutually_exclusive_group()
grp.add_argument("--npz_loop", action="store_true"  , dest="npz_loop", help="Loop through poses (default)")
grp.add_argument("--npz_once", action="store_false" , dest="npz_loop", help="Run poses once then stop")

parser.set_defaults(npz_loop=True)

# --- speed knobs ---
rend_grp = parser.add_mutually_exclusive_group()
rend_grp.add_argument("--render"    , action="store_true"    , dest="render",    help="Render each sim step (default)")
rend_grp.add_argument("--no_render" , action="store_false"   , dest="render",    help="Disable rendering for speed")
parser.set_defaults(render=True)

parser.add_argument("--anchor_every"            , type=int  , default=1     , help="Update foam anchor kinematic targets every N sim steps (1 = every step).")
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

# --- system imports ---
import os
from typing import Any
import ipdb as pdb
from dataclasses import dataclass

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
    euler_xyz_to_rotvec
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

if args_cli.ros:
    # --- ros2 imports ---
    import rclpy
    from rclpy.node import Node
    from rclpy.publisher import Publisher
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    from sensor_msgs.msg import JointState

# --- data structure for computing pressure map --- 
@dataclass
class PressureCache:
    valid_mask: torch.Tensor
    top_triangle_nodes: torch.Tensor
    top_triangle_elements: torch.Tensor
    cell_ids: torch.Tensor
    Hm: int
    Wm: int
    indent_cache: dict[str, Any]   # NOTE: make keys within this dict part of the class. 

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

    # NOTE: this is only for pressure map debugging. 
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

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """runs the simulation loop"""

    # --- extracting entities --- 
    bed: Articulation               = scene["bed"]
    humanoid: Articulation          = scene["humanoid"]
    cubes: list[DeformableObject]   = [scene[f"deformable_{ii}"] for ii in range(32)]
    if args_cli.calibration: 
        sphere: RigidObject         = scene["sphere"]  # NOTE: for debugging pressure computation. 

    # --- extracting joint positions for articulations --- 
    
    # bed
    # TODO: cell heights will be read from npz files (same as humanoid)
    
    # humanoid TODO: currently only one npz file is being read; this should be looped over a folder. 
    joint_names     = humanoid.data.joint_names
    joint_index_map = {name: i for i, name in enumerate(joint_names)}    
    current_npz     = f"scripts/korus/assets/npz/0000.npz" 
    joint_cfg       = build_joint_config_from_npz(current_npz, SMPL_TO_ISAAC, SMPL_BODY_JOINT_ORDER)
    
    target_humanoid_joint_angles = humanoid.data.default_joint_pos.clone()
    for joint_name, angle in joint_cfg.items():
        j = joint_index_map.get(joint_name)
        if j is not None:
            target_humanoid_joint_angles[:, j] = float(angle) 

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

    # --- anchor caches (re-built on reset) --- 
    anchor_caches: list[AnchorCache] = []

    # --- anchor caches (re-built on reset) --- 
    pressure_caches: list[PressureCache] = []

    # --- ros publisher block --- 
    ros_node = None
    indent_pubs: list = []
    if args_cli.ros: 
        if not rclpy.ok():
            rclpy.init(args=None)
        ros_node = Node("korus_ros_api")
        for idx in range(len(cubes)):
            topic = f"/foam_bed/top_surface/indentation_grid_{idx:02d}"
            indent_pubs.append(ros_node.create_publisher(Float32MultiArray, topic, 10))
        def _publish_indent(idx: int, grid_t: torch.Tensor):
            """grid_t: (Hm,Wm) torch tensor"""
            msg = Float32MultiArray()
            Hm = int(grid_t.shape[0])
            Wm = int(grid_t.shape[1])
            # layout (row-major)
            msg.layout.dim = [
                MultiArrayDimension(label="rows", size=Hm, stride=Hm * Wm),
                MultiArrayDimension(label="cols", size=Wm, stride=Wm),
            ]
            # torch -> CPU list
            msg.data = grid_t.detach().float().cpu().reshape(-1).tolist()
            indent_pubs[idx].publish(msg)


    # --- simulation loop starts --- 
    while simulation_app.is_running():
        
        # ---------------
        # RESET BLOCK ---
        # ---------------
        if (count % 600) == 0:
            
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
            
            # --- reset humanoid --- 
            root_state_humanoid = humanoid.data.default_root_state.clone()
            root_state_humanoid[:,:3] += scene.env_origins
            humanoid.write_root_pose_to_sim(root_state_humanoid[:, :7])
            humanoid.write_root_velocity_to_sim(root_state_humanoid[:, 7:])
            joint_pos_humanoid = target_humanoid_joint_angles.clone()  
            joint_vel_humanoid = torch.zeros_like(joint_pos_humanoid)
            humanoid.write_joint_state_to_sim(joint_pos_humanoid, joint_vel_humanoid)
            # set joint position
            humanoid.set_joint_position_target(joint_pos_humanoid)

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
            SETTLE_STEPS = 20  
            for _ in range(SETTLE_STEPS):
                scene.write_data_to_sim()
                sim.step(render=True)          # keep it fast
                scene.update(sim_dt)

            # --- build pressure caches --- 
            pressure_caches.clear()
            for idx, cube in enumerate(cubes): 
                
                # find all top surface indices, and arrange them into a pressure grid like 2d array
                simulation_grid_top_surface_idxs_np = GridBuilder.build_sim_grid(cube, decimals=CFG.decimals)
                
                # plate position for this cube
                body_id = top_plate_body_ids[idx]
                plate_position = bed.data.body_link_pos_w[:, body_id, :3]
                # sim_baseline_pos = cube.data.nodal_pos_w[0].detach().clone() # NOTE: worked for num_env = 1
                sim_baseline_pos = cube.data.nodal_pos_w.detach().clone()        # (B,Nv,3)
                plate_position0  = bed.data.body_link_pos_w[:, body_id, :3].detach().clone()  # (B,3)
                
                # build indent cache NOTE: worked for num_env = 1 
                # indent_cache = PressureMapper.precompute_top_surface_indentation_cache(
                #     simulation_grid_top_surface_idxs_np = simulation_grid_top_surface_idxs_np, 
                #     sim_default_nodal_positions         = sim_baseline_pos, 
                #     plate_pos_w                         = plate_position, 
                #     env_id                              = 0
                # )
                indent_cache = PressureMapper.precompute_top_surface_indentation_cache(
                    simulation_grid_top_surface_idxs_np = simulation_grid_top_surface_idxs_np,
                    sim_default_nodal_positions_w       = sim_baseline_pos,
                    plate_pos_w                         = plate_position0,
                )
               
                # import ipdb; ipdb.set_trace()

                # TODO: for previous method of computing pressure, this can be updated later for efficiency. 
                # find all valid tetrahedrals in the deformable cube                
                tetrahedrals_all_torch: torch.Tensor = cube.root_physx_view.get_sim_element_indices()[0] 
                valid_mask_torch = (tetrahedrals_all_torch >= 0).all(dim=1)
                valid_tetrahedrals_all_torch = tetrahedrals_all_torch[valid_mask_torch]
                valid_tetrahedrals_all_np = valid_tetrahedrals_all_torch.detach().cpu().numpy()
                top_triangle_nodes_np, top_triangle_elements_np, cell_ids_np, Hm, Wm = PressureMapper.precompute_top_surface_boundary_triangle_arrays(
                    valid_tetrahedrals_all_np,
                    simulation_grid_top_surface_idxs_np,
                )

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
                    # cube.write_data_to_sim() # TODO: might not need this since we are doing scene.write_data_to_sim()
                    anchor_cache.last_plate_position = plate_position.detach().clone()

        # --- computing pressure map NOTE: currently only implemented for 1 environment --- 
        compute_pressure = (count % args_cli.pressure_every) == 0
        if compute_pressure: 
            for idx, cube in enumerate(cubes): 
                # extract nodal positions and stresses
                pressure_cache = pressure_caches[idx]
                
                # NOTE: worked for num_env = 1
                # # current sim nodal positions
                # simulation_nodal_positions = cube.data.nodal_pos_w[0] 

                # # current plate position
                # body_id = top_plate_body_ids[idx]
                # plate_position = bed.data.body_link_pos_w[:, body_id, :3]  # (B,3)

                # # compute indentation map
                # indentation_map = PressureMapper.compute_indentation_grid(
                #     indentation_cache = pressure_cache.indent_cache,
                #     sim_nodal_pos_w   = simulation_nodal_positions,
                #     plate_pos_w       = plate_position,
                #     env_id            = 0,
                #     out               = "cell",     # returns (H-1, W-1)
                #     reduce            = "mean",     # or "max"
                # )

                sim_pos_w   = cube.data.nodal_pos_w                         # (B,Nv,3)
                body_id = top_plate_body_ids[idx]
                plate_pos_w = bed.data.body_link_pos_w[:, body_id, :3]      # (B,3)

                indentation_map = PressureMapper.compute_indentation_grid(
                    indentation_cache = pressure_cache.indent_cache,
                    sim_nodal_pos_w   = sim_pos_w,
                    plate_pos_w       = plate_pos_w,
                    out               = "cell",
                    reduce            = "mean",
                )  # (B,Hm,Wm)

                

                # compute pressure map based on indentation map using non-linear modelling
                k_area  = 2.0e6             # Pa/m  (STARTING GUESS; you will calibrate)
                n       = 1.0                # linear foundation; try 1.5 later

                # pressure_map_pa = PressureMapper.indentation_to_pressure_grid(
                #     indent_cell     = indentation_map,
                #     k_area          = k_area,
                #     n               = n,
                #     clamp_max_pa    = None
                # )
                
                pressure_map_pa = PressureMapper.indentation_to_pressure_grid(
                    indent_cell  = indentation_map,   # batched
                    k_area       = k_area,
                    n            = n,
                    clamp_max_pa = None
                )  

                # NOTE: does not work as expected. you should put calibration code here, run once and then use that calibration parameters in the main script. 
                # if args_cli.calibration:
                #     if sphere:  
                #         # sphere params
                #         sphere_mass = float(sphere.data.default_mass)
                #         g = 9.81
                #         target_force = sphere_mass * g
                #         # estimate cell area
                #         Lx = CFG.base_size[0]   
                #         Ly = CFG.base_size[0]   
                #         Hm, Wm = pressure_map_pa.shape
                #         cell_area = (Lx / Wm) * (Ly / Hm)

                #         pressure_map_pa = PressureMapper.normalize_pressure_to_total_force(
                #             pressure_pa     = pressure_map_pa,
                #             cell_area_m2    = cell_area,
                #             target_force_n  = target_force
                #         )

                # if count == 100: 
                #     pdb.set_trace()

                # # TODO: Refactor the code below so that it works for the cloned environments, instead of only 1 deformable cube.  
                # simulation_nodal_positions = cube.data.nodal_pos_w[0] 
                # simulation_tetrahedral_stresses = cube.data.sim_element_stress_w[0]
                # # filter valid stresses
                # valid_simulation_tetrahedral_stresses = simulation_tetrahedral_stresses[pressure_cache.valid_mask]                                
                # # compute pressure grid 
                # pressure_map = PressureMapper.compute_pressure_grid( 
                #     top_triangle_nodes                  = pressure_cache.top_triangle_nodes, 
                #     top_triangle_elements               = pressure_cache.top_triangle_elements, 
                #     cell_ids                            = pressure_cache.cell_ids, 
                #     valid_simulation_element_stresses   = valid_simulation_tetrahedral_stresses, 
                #     simulation_nodal_positions          = simulation_nodal_positions, 
                #     Hm                                  = pressure_cache.Hm, 
                #     Wm                                  = pressure_cache.Wm 
                # )

                # publish if ROS enabled
                if args_cli.ros:
                    # _publish_indent(idx, indentation_map) # publish indentation map
                    _publish_indent(idx, pressure_map_pa[1]) # publish pressure map

            if args_cli.ros:
                rclpy.spin_once(ros_node, timeout_sec=0.0) # type: ignore

                    
        # --- step and update --- 
          
        # write data to sim
        scene.write_data_to_sim()
        # step through the simulation
        sim.step(render=bool(args_cli.render)) # tunable for speed
        # increment counters
        sim_time += sim_dt 
        count += 1
        # update all buffers
        scene.update(sim_dt)

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
    




