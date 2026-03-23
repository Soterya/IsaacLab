#!/usr/bin/env python3
"""
Usage:
    $ cd ~/IsaacLab
    $ preload ./isaaclab.sh -p scripts/korus/run_korusbed_keyboard_ros_yaml_refactor.py \
        --config scripts/korus/configs/korus.yaml

NOTE: Check the YAML file for tweaking the material properties.
"""

# ------------------------------------
# --- Imports for the App Creation ---
# ------------------------------------
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from isaaclab.app import AppLauncher

# -----------------------------------------------
# --- Launch the IsaacSim App (with argparse) ---
# -----------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab + YAML config")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--num_rows", type=int, default=None,                                  help="override YAML rows")
parser.add_argument("--num_cols", type=int, default=None,                                  help="override YAML cols")
parser.add_argument("--every",    type=int, default=1,                                     help="ROS publish every N frames")
parser.add_argument("--config",   type=str, default="scripts/korus/config/korus_bed.yaml", help="YAML file with globals")

# --- Parser for the NPZ Directory (contains different poses)
parser.add_argument("--npz_dir",  type=str,  default="scripts/korus/assets/npz", help="Directory containing pose npz files (e.g., 0000.npz, 0010.npz, ...)")
parser.add_argument("--npz_glob", type=str,  default="*.npz",                    help="Glob pattern inside npz_dir (default: *.npz)")
parser.add_argument("--npz_start", type=int, default=None,                       help="Optional: only use poses with index >= this (e.g., 0)")
parser.add_argument("--npz_end", type=int,   default=None,                       help="Optional: only use poses with index <= this (e.g., 10)")

grp = parser.add_mutually_exclusive_group()
grp.add_argument("--npz_loop", action="store_true",  dest="npz_loop", help="Loop through poses (default)")
grp.add_argument("--npz_once", action="store_false", dest="npz_loop", help="Run poses once then stop")
parser.set_defaults(npz_loop=True)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experience = "isaacsim.exp.full.kit"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------
# --- Imports after app creation ---
# ----------------------------------
import yaml
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import (
    DeformableObject, DeformableObjectCfg,
    Articulation, ArticulationCfg
)
from isaaclab.actuators import ImplicitActuatorCfg
import omni.usd
from pxr import Usd
import isaacsim.core.utils.prims as prim_utils

# ----------------------
# --- Custom Classes ---
# ----------------------
from utils.keycontrol import KeyControl
from utils.cfg_utils import AppCfg, MaterialCfg
from utils.ros_bed_controller import ROSBedController
from utils.pressure_mapper import PressureMapper
from utils.grid_builder import GridBuilder
from utils.stiffness_and_damping import (
    STIFFNESS_PER_JOINT,
    DAMPING_PER_JOINT,
    EFFORT_PER_JOINT,
    ALL_HUMANOID_JOINTS, 
    DEFAULT_STIFFNESS, DEFAULT_EFFORT, DEFAULT_DAMPING 
)

# --------------------
# --- ROS2 Imports ---
# --------------------
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R

# ---------------------------
# --- Pose Change Imports ---
# ---------------------------
import os
import glob
import re

# ---------------------------------------
# 0) SMPL joint order for pose_body  
# ---------------------------------------
SMPL_BODY_JOINT_ORDER = [
    "L_Hip"     ,
    "R_Hip"     ,
    "Torso"     ,
    "L_Knee"    ,
    "R_Knee"    ,
    "Spine"     ,
    "L_Ankle"   ,
    "R_Ankle"   ,
    "Chest"     ,
    "L_Toe"     ,
    "R_Toe"     ,
    "Neck"      ,
    "L_Thorax"  ,
    "R_Thorax"  ,
    "Head"      ,
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow"   ,
    "R_Elbow"   ,
    "L_Wrist"   ,
    "R_Wrist"   ,
    "L_Hand"    ,
    "R_Hand"    ,
]

# ----------------------------
# 1) SMPL to ISAAC Mapping ---
# ----------------------------
SMPL_TO_ISAAC = {
    "L_Hip"     :("L_Hip_x"     , "L_Hip_y"     , "L_Hip_z"     ),
    "R_Hip"     :("R_Hip_x"     , "R_Hip_y"     , "R_Hip_z"     ),
    "Torso"     :("Torso_x"     , "Torso_y"     , "Torso_z"     ),
    "L_Knee"    :("L_Knee_x"    , "L_Knee_y"    , "L_Knee_z"    ),
    "R_Knee"    :("R_Knee_x"    , "R_Knee_y"    , "R_Knee_z"    ),
    "Spine"     :("Spine_x"     , "Spine_y"     , "Spine_z"     ),
    "L_Ankle"   :("L_Ankle_x"   , "L_Ankle_y"   , "L_Ankle_z"   ),
    "R_Ankle"   :("R_Ankle_x"   , "R_Ankle_y"   , "R_Ankle_z"   ),
    "Chest"     :("Chest_x"     , "Chest_y"     , "Chest_z"     ),
    "L_Toe"     :("L_Toe_x"     , "L_Toe_y"     , "L_Toe_z"     ),
    "R_Toe"     :("R_Toe_x"     , "R_Toe_y"     , "R_Toe_z"     ),
    "Neck"      :("Neck_x"      , "Neck_y"      , "Neck_z"      ),
    "L_Thorax"  :("L_Thorax_x"  , "L_Thorax_y"  , "L_Thorax_z"  ),
    "R_Thorax"  :("R_Thorax_x"  , "R_Thorax_y"  , "R_Thorax_z"  ),
    "Head"      :("Head_x"      , "Head_y"      , "Head_z"      ),
    "L_Shoulder":("L_Shoulder_x", "L_Shoulder_y", "L_Shoulder_z"),
    "R_Shoulder":("R_Shoulder_x", "R_Shoulder_y", "R_Shoulder_z"),
    "L_Elbow"   :("L_Elbow_x"   , "L_Elbow_y"   , "L_Elbow_z"   ),
    "R_Elbow"   :("R_Elbow_x"   , "R_Elbow_y"   , "R_Elbow_z"   ),
    "L_Wrist"   :("L_Wrist_x"   , "L_Wrist_y"   , "L_Wrist_z"   ),
    "R_Wrist"   :("R_Wrist_x"   , "R_Wrist_y"   , "R_Wrist_z"   ),
    "L_Hand"    :("L_Hand_x"    , "L_Hand_y"    , "L_Hand_z"    ),
    "R_Hand"    :("R_Hand_x"    , "R_Hand_y"    , "R_Hand_z"    ),
}

# -----------------------
# 2) Helper Functions ---
# -----------------------
def rotvec_to_euler_xyz(rotvec: np.ndarray, degrees: bool=False) -> np.ndarray:
    """
    Converts from axis-angle form to euler xyz (either in degrees or radians)
    """
    r = R.from_rotvec(np.asarray(rotvec, dtype=np.float64))
    e = r.as_euler("XYZ", degrees=degrees)
    return e.astype(np.float32)

def build_joint_config_from_npz(path: str) -> dict[str, float]:
    """
    Builds Joint Configuration given a .npz smpl pose file
    """   
    data = np.load(path, allow_pickle=True)
    body = data["pose_body"][0].astype(np.float32).reshape(-1, 3)
    joint_cfg: dict[str, float] = {}
    for smpl_name, isaac_triplet in SMPL_TO_ISAAC.items():
        smpl_idx = SMPL_BODY_JOINT_ORDER.index(smpl_name)  # 0..22, matches body.shape[0]
        rotvec = body[smpl_idx]  # (3,)
        rx, ry, rz = rotvec_to_euler_xyz(rotvec, degrees=False) # Kept Radians for now
        jx, jy, jz = isaac_triplet
        joint_cfg[jx] = float(rx)
        joint_cfg[jy] = float(ry)
        joint_cfg[jz] = float(rz)
    return joint_cfg

def _npz_index(path: str) -> int:
    """
    Extract trailing numeric index from filenames like 0000.npz, 12.npz, pose_0010.npz.
    Returns -1 if no number is found.
    """
    base = os.path.basename(path)
    m = re.search(r"(\d+)(?=\.npz$)", base)
    return int(m.group(1)) if m else -1

def list_npz_files(npz_dir: str, pattern: str = "*.npz",
                   start: Optional[int] = None, end: Optional[int] = None) -> list[str]:
    paths = glob.glob(os.path.join(npz_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort(key=_npz_index)

    if start is not None:
        paths = [p for p in paths if _npz_index(p) >= start]
    if end is not None:
        paths = [p for p in paths if _npz_index(p) <= end]

    return paths

# --------------------
# --- Constants    ---
# --------------------
TOPPLATE_SIZE = (1.0, 1.0, 0.1)  # (x, y, z) for TopPlate{idx}
RESET_AFTER   = 6e2

# ------------------------------
# --- Loads Config from YAML ---
# ------------------------------
def load_cfg(path: Optional[str]) -> AppCfg:
    if path is None:
        raise RuntimeError("Please provide --config path to a YAML file.")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    # material sub-dict
    mat_raw = raw.get("material", {}) or {}
    mat = MaterialCfg(
        poissons_ratio     = float(mat_raw.get("poissons_ratio", 0.2)),
        youngs_modulus     = float(mat_raw.get("youngs_modulus", 3.0e4)),
        dynamic_friction   = float(mat_raw.get("dynamic_friction", 1.0)),
        elasticity_damping = float(mat_raw.get("elasticity_damping", 0.06)),
        damping_scale      = float(mat_raw.get("damping_scale", 1.0)),
        hexa_res           =   int(mat_raw.get("hexa_res", 10)),
    )
    cfg = AppCfg(
        env_usd               =   str(raw.get("env_usd", "")),
        korusbed_usd          =   str(raw.get("korusbed_usd", "")),
        humanoid_usd          =   str(raw.get("humanoid_usd", "")),
        ground_to_base_bottom = float(raw.get("ground_to_base_bottom", 0.9)),
        base_size             = tuple(raw.get("base_size", [1.0, 1.0, 0.1])),  # unused now
        foam_size             = tuple(raw.get("foam_size", [1.0, 1.0, 0.2])),
        sphere_radius         = float(raw.get("sphere_radius", 0.2)),
        base_origin_xy        = tuple(raw.get("base_origin_xy", [-1.65, 3.85])),  # unused now
        sphere_origin_z       = float(raw.get("sphere_origin_z", 3.0)),
        node_spacing          = float(raw.get("node_spacing", 1.1)),
        decimals              =   int(raw.get("decimals", 1)),
        rows                  =   int(raw.get("rows", 1)),
        cols                  =   int(raw.get("cols", 1)),
        material              =   mat,
    )

    # allow CLI to override rows/cols
    if args_cli.num_rows is not None:
        cfg.rows = int(args_cli.num_rows)
    if args_cli.num_cols is not None:
        cfg.cols = int(args_cli.num_cols)
    cfg.compute_derivatives()
    return cfg

# ---------------
# --- helpers ---
# ---------------
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

# ---------------------
# --- Scene builder ---
# ---------------------
def design_scene(cfg: AppCfg, initial_joint_config: dict[str, float]):
    """
    Build world + a single KorusBed articulation + (rows x cols) deformable cells.
    Returns dict with 'bed' and 'cells' (each cell has cube row/col/index).
    """
    # load the environment
    scene_context = omni.usd.get_context()
    scene_context.open_stage(cfg.env_usd)

    stage = scene_context.get_stage()


    # light
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    dome.func("/World/DomeLight", dome)

    # import the humanoid (rigid or soft)
    humanoid_asset = sim_utils.UsdFileCfg(usd_path=cfg.humanoid_usd, scale=(4., 4., 4.))
    humanoid_asset.func("/World/Humanoid", humanoid_asset)

    # ---------------------------------------------------------
    # SCENE-ONLY OVERRIDE: deactivate /World/Humanoid/worldBody
    # (does NOT touch humanoid_0005.usd on disk)
    # ---------------------------------------------------------
    worldbody_path = "/World/Humanoid/worldBody"

    # Check if the composed prim exists (it will, from the reference)
    prim_composed = stage.GetPrimAtPath(worldbody_path)
    if prim_composed.IsValid():
        print(f"[INFO] Deactivating {worldbody_path} via session-layer override.")

        # Use the SESSION layer so this is 100% non-destructive
        session_layer = stage.GetSessionLayer()

        # Any specs authored in this context live only in the session layer
        with Usd.EditContext(stage, session_layer): # type: ignore
            prim_override = stage.OverridePrim(worldbody_path)
            prim_override.SetActive(False)
    else:
        print(f"[INFO] {worldbody_path} not found in composed stage; nothing to deactivate.")

    # import the bed asset
    bed_asset = sim_utils.UsdFileCfg(usd_path=cfg.korusbed_usd)
    bed_asset.func("/World/KorusBed", bed_asset)

    # define articulation for the bed (must have Articulation Root)
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
                damping=8_00.0,
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

    humanoid_actuators = {
        f"{joint_name}_pd": make_joint_pd_cfg(joint_name)
        for joint_name in ALL_HUMANOID_JOINTS
    }

    # NOTE: Stiffness, Damping and Effort Parameters should be tuned here. 
    # TODO: Incorporate these params into the scripts/korus/utils/convert_xml1_to_xml2_joint_limits.py. This will be better for batch scripting. 
    humanoid_cfg = ArticulationCfg(
        class_type=Articulation,
        prim_path="/World/Humanoid",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 1.5, 4.0),
            rot=(1, 0, 0, 0),
            joint_pos=initial_joint_config, # zero all hinge joints by default
            joint_vel={".*": 0.0},
        ),
        actuators=humanoid_actuators, # type: ignore
    )
    humanoid = Articulation(cfg=humanoid_cfg)

    sx = float(cfg.node_spacing)
    sy = float(cfg.node_spacing)
    cells = []

    for r in range(cfg.rows):
        for c in range(cfg.cols):
            idx   = r * cfg.cols + c
            x_off = c * sx
            y_off = -r * sy

            prim_utils.create_prim(f"/World/KorusBed/Node{idx}", "Xform")

            # deformable foam ONLY (no rigid base)
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
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                ),
                init_state=DeformableObjectCfg.InitialStateCfg(
                    pos=(cfg.foam_origin[0] + x_off, cfg.foam_origin[1] + y_off, cfg.foam_origin[2])
                ),
                debug_vis=False,
            )
            cube_obj = DeformableObject(cfg=deform_cfg)

            cells.append({
                "cube": cube_obj,
                "node_index": idx,
                "row": r,
                "col": c,
            })

    return {"bed": bed, "cells": cells, "humanoid":humanoid}

# ---------------------
# --- Run simulator ---
# ---------------------
def run_simulator(sim: sim_utils.SimulationContext, entities, ros_node: Node, keyctl: KeyControl, cfg: AppCfg, npz_files: list[str], npz_loop: bool):
    """
    - Publishes per-cell legacy z/xyz (collision mesh)
    - Publishes SIM-mesh dz, |u|, and area-weighted pressure (per row/col topic)
    - Drives the bed’s prismatic joints via keyboard and ROS (ROSBedController)
    """
    bed      = entities["bed"]
    cells    = entities["cells"]
    humanoid = entities["humanoid"]

    joint_names = humanoid.data.joint_names
    joint_index_map = {name: i for i, name in enumerate(joint_names)}

    pose_idx = 0
    q_target_hum = None  # will be set on reset
    current_npz = None

    # publishers & per-cell caches
    pubs_z_xyz = []          # [(pub_z, pub_xyz)]
    pubs_disp  = []          # [(pub_dz_rc, pub_umag_rc)]
    pubs_press = []          # [pub_pressure_rc]

    bottom_corner_idx = []
    kin_buffers       = []

    index_grids_colmesh = []; HW_colmesh = []
    index_grids_sim  = [];    HW_sim = []
    sim_pos0_list    = []
    valid_mask_list  = []
    tri_maps_list    = []
    HW_press_list    = []

    # ROS controller
    ros_ctl = ROSBedController(ros_node, cfg.rows, cfg.cols)

    # make per-cell pubs
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

        bottom_corner_idx.append(None); kin_buffers.append(None)
        index_grids_colmesh.append(None); HW_colmesh.append((0, 0))
        index_grids_sim.append(None);     HW_sim.append((0, 0))
        sim_pos0_list.append(None);       valid_mask_list.append(None)
        tri_maps_list.append(None);       HW_press_list.append((0, 0))

    # timing
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # initialize before loop
    topplate_body_ids: List[int] = []

    while simulation_app.is_running():

        # --- Reset block (also on start) ---
        if count % RESET_AFTER == 0:
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

            # ---- Pose swap (one pose per reset) ----
            if not npz_files:
                raise RuntimeError("No NPZ files provided (npz_files list is empty).")

            current_npz = npz_files[pose_idx]
            joint_cfg = build_joint_config_from_npz(current_npz)

            pose_idx += 1
            if pose_idx >= len(npz_files):
                if npz_loop:
                    pose_idx = 0
                else:
                    print("[INFO] Finished all NPZ poses (--npz_once). Exiting simulation loop.")
                    break

            # Build q_target_hum from this joint_cfg
            # q_target_hum = humanoid.data.joint_pos.clone() # NOTE: Might need to change back
            q_target_hum = humanoid.data.default_joint_pos.clone()
            for j_name, angle in joint_cfg.items():
                j = joint_index_map.get(j_name)
                if j is not None:
                    q_target_hum[:, j] = float(angle)

            print(f"[INFO] Using pose: {os.path.basename(current_npz)}")

            # reset humanoid (root + joints) using NPZ pose
            root_state_hum = humanoid.data.default_root_state.clone()
            humanoid.write_root_pose_to_sim(root_state_hum[:, :7])
            humanoid.write_root_velocity_to_sim(root_state_hum[:, 7:])

            # joints: pose as both state and PD target
            jpos_h = q_target_hum.clone()
            jvel_h = torch.zeros_like(jpos_h)
            humanoid.write_joint_state_to_sim(jpos_h, jvel_h)
            humanoid.reset()
            humanoid.set_joint_position_target(jpos_h)

            # reset deformables
            for ent in cells:
                cube = ent["cube"]
                cube_state = cube.data.default_nodal_state_w.clone()
                cube.write_nodal_state_to_sim(cube_state)
                cube.reset()

            # prime ROS joint control
            ros_ctl.attach_bed(bed)

            # build TopPlate id map once per reset
            topplate_body_ids = []
            for idx in range(cfg.rows * cfg.cols):
                ids, _ = bed.find_bodies(f"TopPlate{idx}")
                if not ids:
                    raise RuntimeError(f"TopPlate{idx} not found in bed bodies.")
                topplate_body_ids.append(ids[0])

            # per-cell mappings & pins
            for n, ent in enumerate(cells):
                cube = ent["cube"]
                dev  = cube.data.nodal_pos_w.device

                # collision-mesh grid (legacy z/xyz)
                default_pos = cube.data.default_nodal_state_w[..., :3][0]
                z = default_pos[:, 2]
                s = 10 ** cfg.decimals
                zr = torch.round(z * s) / s
                planes = torch.sort(torch.unique(zr)).values
                z_bot_default = float(planes[0]); z_top_default = float(planes[-1])

                top_idx = GridBuilder._get_surface_indices_by_z(cube, z_top_default, cfg.decimals)  # type: ignore
                bot_idx = GridBuilder._get_surface_indices_by_z(cube, z_bot_default, cfg.decimals)  # type: ignore
                if top_idx.numel() == 0 or bot_idx.numel() == 0:
                    raise RuntimeError(f"[Cell {n}] collision top/bottom nodes not found.")

                grid_col = GridBuilder.build_colmesh_grid(cube, z_top_default, cfg.decimals)
                Hc, Wc = grid_col.shape
                index_grids_colmesh[n] = grid_col; HW_colmesh[n] = (Hc, Wc)

                # bottom-corner pins: initialize kin buffer and mark bottom corners kinematic=0
                bot_corners = GridBuilder.corners_from_surface_idx(cube, bot_idx, env_id=0)
                bottom_corner_idx[n] = bot_corners

                kin = cube.data.nodal_kinematic_target.clone()
                kin[..., :3] = cube.data.nodal_pos_w
                kin[..., 3]  = 1.0
                kin[:, bot_corners, 3] = 0.0
                cube.write_nodal_kinematic_target_to_sim(kin)
                kin_buffers[n] = kin

                # SIM grid
                grid_sim, sim_pos0_np = GridBuilder.build_sim_grid(cube, cfg.decimals)
                Hs, Ws = grid_sim.shape
                index_grids_sim[n] = grid_sim; HW_sim[n] = (Hs, Ws)
                sim_pos0_list[n] = torch.as_tensor(sim_pos0_np, device=dev, dtype=torch.float32)

                # valid tets
                T_raw_any = cube.root_physx_view.get_sim_element_indices()[0]
                T_raw_np = T_raw_any.detach().cpu().numpy() if hasattr(T_raw_any, "detach") else np.asarray(T_raw_any)
                valid_mask_np = (T_raw_np >= 0).all(axis=1)
                valid_mask_list[n] = torch.as_tensor(valid_mask_np, device=dev, dtype=torch.bool)
                T_valid = T_raw_np[valid_mask_np]

                # precompute triangle mapping
                tri_nodes_np, tri_elem_np, cell_ids_np, Hm, Wm = PressureMapper.precompute_top_surface_tris_arrays(T_valid, grid_sim)
                tri_nodes_t = torch.as_tensor(tri_nodes_np, device=dev, dtype=torch.long)
                tri_elem_t  = torch.as_tensor(tri_elem_np,  device=dev, dtype=torch.long)
                cell_ids_t  = torch.as_tensor(cell_ids_np,  device=dev, dtype=torch.long)
                tri_maps_list[n] = (tri_nodes_t, tri_elem_t, cell_ids_t)
                HW_press_list[n] = (Hm, Wm)

            print(f"[INFO] Reset bed + Humanoid + {len(cells)} deformables; prismatic joints = {len(ros_ctl.pris_ids)}")

        # --- Step & update ---
        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        bed.update(sim_dt)
        humanoid.update(sim_dt)

        for ent in cells:
            ent["cube"].update(sim_dt)

        # apply ROS targets
        ros_ctl.apply_ros_targets()

        # keyboard delta
        delta = (keyctl.step if keyctl.inc else 0.0) - (keyctl.step if keyctl.dec else 0.0)
        ros_ctl.apply_keyboard_delta(delta)

        # send joint targets
        if ros_ctl.q_cmd is not None:
            bed.set_joint_position_target(ros_ctl.q_cmd)

        # per-cell pins + publishes
        for n, ent in enumerate(cells):
            idx    = ent["node_index"]
            cube   = ent["cube"]
            dev    = cube.data.nodal_pos_w.device

            # --- use TopPlate{idx} world pose for kinematic pins ---
            body_id = topplate_body_ids[idx]
            pos_np  = bed.data.body_link_pos_w[0, body_id].detach().cpu().numpy()  # (3,)
            cx, cy, cz = pos_np.tolist()

            half_x = TOPPLATE_SIZE[0] * 0.5
            half_y = TOPPLATE_SIZE[1] * 0.5
            half_z = TOPPLATE_SIZE[2] * 0.5
            z_top  = cz + half_z

            corners = [
                (cx - half_x, cy - half_y, z_top),
                (cx - half_x, cy + half_y, z_top),
                (cx + half_x, cy - half_y, z_top),
                (cx + half_x, cy + half_y, z_top),
            ]
            targets_b = torch.tensor(
                [corners],  # (B=1, 4, 3)
                dtype=cube.data.nodal_kinematic_target.dtype,
                device=cube.data.nodal_kinematic_target.device,
            )

            kin = kin_buffers[n]
            kin[:, bottom_corner_idx[n], :3] = targets_b
            cube.write_nodal_kinematic_target_to_sim(kin)
            cube.write_data_to_sim()

            # legacy z/xyz (collision mesh)
            if (count % args_cli.every == 0) and (index_grids_colmesh[n] is not None):
                Hc, Wc = HW_colmesh[n]
                grid_col = index_grids_colmesh[n]
                pos = cube.data.nodal_pos_w[0]
                z_grid = pos[grid_col.reshape(-1), 2].reshape(Hc, Wc).detach().cpu().numpy()
                xyz    = pos[grid_col.reshape(-1)].detach().cpu().numpy()
                pub_z, pub_xyz = pubs_z_xyz[n]
                pub_z.publish(make_multiarray_2d(z_grid))
                pub_xyz.publish(make_multiarray_xyz(xyz))

            # SIM dz / |u| / pressure
            if (count % args_cli.every == 0) and (index_grids_sim[n] is not None):
                Hs, Ws = HW_sim[n]
                idx_flat_t = torch.as_tensor(index_grids_sim[n].reshape(-1), device=dev, dtype=torch.long)

                sim_pos_any = cube.root_physx_view.get_sim_nodal_positions()[0]
                S_any       = cube.root_physx_view.get_sim_element_stresses()[0]

                sim_pos_t = sim_pos_any.detach().to(dev) if hasattr(sim_pos_any, "detach") else \
                            torch.as_tensor(sim_pos_any, device=dev, dtype=torch.float32)
                S_t = S_any.detach().to(dev) if hasattr(S_any, "detach") else \
                      torch.as_tensor(S_any, device=dev, dtype=torch.float32)

                valid_mask_t = valid_mask_list[n]
                S_valid_t = S_t[valid_mask_t]
                if S_valid_t.ndim == 2 and S_valid_t.size(-1) == 9:
                    S_valid_t = S_valid_t.view(-1, 3, 3)

                sim_pos0_t = sim_pos0_list[n]
                z_now  = sim_pos_t.index_select(0, idx_flat_t)[:, 2].reshape(Hs, Ws)
                z_ref  = sim_pos0_t.index_select(0, idx_flat_t)[:, 2].reshape(Hs, Ws)
                dz_grid = (z_now - z_ref).detach().cpu().numpy()

                u = (sim_pos_t.index_select(0, idx_flat_t) - sim_pos0_t.index_select(0, idx_flat_t))
                u_mag_grid = torch.linalg.norm(u, dim=1).reshape(Hs, Ws).detach().cpu().numpy()

                pub_dz_rc, pub_umag_rc = pubs_disp[n]
                pub_dz_rc.publish(make_multiarray_2d(dz_grid))
                pub_umag_rc.publish(make_multiarray_2d(u_mag_grid))

                tri_nodes_t, tri_elem_t, cell_ids_t = tri_maps_list[n]
                Hm, Wm = HW_press_list[n]
                P_t = PressureMapper.compute_pressure_grid_torch(tri_nodes_t, tri_elem_t, cell_ids_t,
                                                                 S_valid_t, sim_pos_t, Hm, Wm)
                pubs_press[n].publish(make_multiarray_2d(P_t.detach().cpu().numpy()))

        # push articulation after q_cmd updates
        bed.write_data_to_sim()
        humanoid.write_data_to_sim()

        # humanoid.set_joint_position_target(q_target_hum)
        if q_target_hum is not None:
            humanoid.set_joint_position_target(q_target_hum)

        # ROS spin
        if count % args_cli.every == 0:
            rclpy.spin_once(ros_node, timeout_sec=0.0)

# ------------
# --- Main ---
# ------------
def main():
    # --- YAML config ---
    cfg = load_cfg(args_cli.config)

    # --- ROS2 ---
    rclpy.init(args=None)
    ros_node = rclpy.create_node("foam_bed_publisher")

    # --- Keyboard ---
    keyctl = KeyControl(step=0.003)
    keyctl.subscribe()

    # --- Sim ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, -3.0, 2.0), target=(0.0, 0.0, 0.5))

    npz_files = list_npz_files(
    args_cli.npz_dir,
    args_cli.npz_glob,
    start=args_cli.npz_start,
    end=args_cli.npz_end,
    )
    if not npz_files:
        raise RuntimeError(f"No npz files found in {args_cli.npz_dir} matching {args_cli.npz_glob}")

    print(f"[INFO] Found {len(npz_files)} pose files. First={os.path.basename(npz_files[0])}, Last={os.path.basename(npz_files[-1])}")

    initial_joint_config = build_joint_config_from_npz(npz_files[0])

    entities = design_scene(cfg, initial_joint_config)
    sim.reset()
    print(f"[INFO]: Setup complete (rows={cfg.rows}, cols={cfg.cols})... (Up/W=inflate, Down/S=deflate, Space=stop)")

    try:
        run_simulator(sim, entities, ros_node, keyctl, cfg, npz_files, args_cli.npz_loop)

    finally:
        keyctl.unsubscribe()
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()
