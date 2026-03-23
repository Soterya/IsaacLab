#!/usr/bin/env python3
"""
Usage:
    $ cd ~/IsaacLab
    $ preload ./isaaclab.sh -p scripts/korus/run_humanoid_tuning.py \
        --config scripts/korus/config/korus_bed.yaml

NOTE: Check the YAML file for tweaking the asset paths (env_usd, humanoid_usd).
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
parser = argparse.ArgumentParser(description="Humanoid tuning in IsaacLab + YAML config")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--num_rows", type=int, default=None, help="(unused) kept for compatibility")
parser.add_argument("--num_cols", type=int, default=None, help="(unused) kept for compatibility")
parser.add_argument("--every",    type=int, default=1, help="(unused) kept for compatibility")
parser.add_argument("--config",   type=str, default="scripts/korus/config/korus_bed.yaml",help="YAML file with globals")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------
# --- Imports after app creation ---
# ----------------------------------
import yaml
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import omni.usd
from pxr import Usd
from scipy.spatial.transform import Rotation as R

# ----------------------
# --- Custom Classes ---
# ----------------------
from utils.cfg_utils import AppCfg, MaterialCfg

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
        rx, ry, rz = rotvec_to_euler_xyz(rotvec, degrees=False)  # radians

        jx, jy, jz = isaac_triplet
        joint_cfg[jx] = float(rx)
        joint_cfg[jy] = float(ry)
        joint_cfg[jz] = float(rz)

    return joint_cfg

NPZ_FILE = "scripts/korus/assets/npz/0000.npz"
JOINT_CONFIG = build_joint_config_from_npz(NPZ_FILE)
HUMANOID_ACTUATORS = None

# ------------------------------
# --- Loads Config from YAML ---
# ------------------------------
def load_cfg(path: Optional[str]) -> AppCfg:
    if path is None:
        raise RuntimeError("Please provide --config path to a YAML file.")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # material sub-dict (still parsed but unused in this script)
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
        korusbed_usd          =   str(raw.get("korusbed_usd", "")),   # parsed but unused
        humanoid_usd          =   str(raw.get("humanoid_usd", "")),
        ground_to_base_bottom = float(raw.get("ground_to_base_bottom", 0.9)),
        base_size             = tuple(raw.get("base_size", [1.0, 1.0, 0.1])),
        foam_size             = tuple(raw.get("foam_size", [1.0, 1.0, 0.2])),
        sphere_radius         = float(raw.get("sphere_radius", 0.2)),
        base_origin_xy        = tuple(raw.get("base_origin_xy", [-1.65, 3.85])),
        sphere_origin_z       = float(raw.get("sphere_origin_z", 3.0)),
        node_spacing          = float(raw.get("node_spacing", 1.1)),
        decimals              =   int(raw.get("decimals", 1)),
        rows                  =   int(raw.get("rows", 1)),
        cols                  =   int(raw.get("cols", 1)),
        material              =   mat,
    )

    # if your AppCfg has a compute_derivatives() helper, keep it
    if hasattr(cfg, "compute_derivatives"):
        cfg.compute_derivatives()
    return cfg

# ---------------------
# --- Scene builder ---
# ---------------------
def design_scene(cfg: AppCfg) -> Articulation:
    """
    Build world + a single humanoid articulation.
    No bed, no deformables, no ROS / keyboard control.
    """
    # load the environment
    scene_context = omni.usd.get_context()
    scene_context.open_stage(cfg.env_usd)

    stage = scene_context.get_stage()

    # simple dome light
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    dome.func("/World/DomeLight", dome)

    # import the humanoid (rigid USD with joints)
    humanoid_asset = sim_utils.UsdFileCfg(usd_path=cfg.humanoid_usd, scale=(4.4, 4.4, 4.4))
    humanoid_asset.func("/World/Humanoid", humanoid_asset)

    # ---------------------------------------------------------
    # SCENE-ONLY OVERRIDE: deactivate /World/Humanoid/worldBody
    # (does NOT touch humanoid_XXXX.usd on disk)
    # ---------------------------------------------------------
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

    # define articulation for the humanoid (reads existing joints from USD)
    humanoid_cfg = ArticulationCfg(
        class_type=Articulation,
        prim_path="/World/Humanoid",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            # drop a bit above the ground so it can fall
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=JOINT_CONFIG,   # NPZ pose as initial joint configuration
            joint_vel={".*": 0.0},
        ),
        actuators={
            # PD over all hinge joints: Spine_x, Chest_y, L_Hip_z, ...
            "humanoid_pd": ImplicitActuatorCfg(
                joint_names_expr=[
                    "Spine_.*", "Chest_.*", "Torso_.*", "Neck_.*",
                    "L_Thorax_.*", "R_Thorax_.*",
                    "L_Hip_.*", "R_Hip_.*",
                    "Head_.*",
                    "L_Shoulder_.*", "R_Shoulder_.*",
                    "L_Knee_.*", "R_Knee_.*",
                    "L_Elbow_.*", "R_Elbow_.*",
                    "L_Ankle_.*", "R_Ankle_.*",
                    "L_Wrist_.*", "R_Wrist_.*",
                    "L_Toe_.*", "R_Toe_.*",
                    "L_Hand_.*", "R_Hand_.*",
                ],
                # Tune these for your stiffness / damping experiments
                effort_limit_sim = 60_000.0,
                stiffness        = 30_000.0,
                damping          =  4_000.0,
            ),
        },
    )
    humanoid = Articulation(cfg=humanoid_cfg)

    return humanoid

# ---------------------
# --- Run simulator ---
# ---------------------
def run_simulator(sim: SimulationContext, humanoid: Articulation):
    """
    Very simple loop:
      - sets NPZ pose as PD target once
      - lets the humanoid fall under gravity
    No ROS, no keyboard, no bed, no deformables.
    """
    # physics timing
    sim_dt = sim.get_physics_dt()

    # make sure default state is applied once
    sim.reset()
    humanoid.reset()

    # Build humanoid joint target from JOINT_CONFIG (NPZ pose)
    joint_names = humanoid.data.joint_names
    joint_index_map = {name: i for i, name in enumerate(joint_names)}

    # Start from current joint_pos shape
    num_envs, num_joints = humanoid.data.joint_pos.shape
    q_target_hum = humanoid.data.joint_pos.clone()

    for j_name, angle in JOINT_CONFIG.items():
        idx = joint_index_map.get(j_name)
        if idx is None:
            # Uncomment for debugging if names don't line up:
            # print(f"[WARN] Joint {j_name} not found in humanoid.data.joint_names")
            continue
        q_target_hum[:, idx] = angle

    # Apply NPZ pose as current state and PD target
    jvel_h = torch.zeros_like(q_target_hum)
    humanoid.write_joint_state_to_sim(q_target_hum, jvel_h)
    humanoid.set_joint_position_target(q_target_hum)

    print("[INFO] Starting humanoid tuning sim (no ROS, no bed, no deformables).")

    # simple loop while app is running
    while simulation_app.is_running():
        sim.step(render=True)
        humanoid.update(sim_dt)

        # keep PD target applied (comment out if you want pure ragdoll after initial pose)
        humanoid.set_joint_position_target(q_target_hum)

        # push data to sim
        humanoid.write_data_to_sim()

# ------------
# --- Main ---
# ------------
def main():
    # --- YAML config ---
    cfg = load_cfg(args_cli.config)

    # --- Sim ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, -3.0, 2.0), target=(0.0, 0.0, 1.0))

    humanoid = design_scene(cfg)

    try:
        run_simulator(sim, humanoid)
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
