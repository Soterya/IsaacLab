"""
One setting that seems to work: 
 preload ./isaaclab.sh -p ./scripts/korus/particle_cloth_scene.py --num_envs 1 --size_x 6 --size_y 10 --u_res 60 --v_res 100 --z 2. --humanoid  --stretch 15000 --bend 60000 --shear 1500 --friction 1.0  --damping 0.7 --mass 10 --solver_iters 64 --rest_offset 0.5 --contact_scale 2.0 --self_collision
"""

# --------------------------------
# 0) Imports for the App Creation ---
# --------------------------------
import argparse
from isaaclab.app import AppLauncher

# ------------------------------------------------
# 1) Launch the Simulation App (with argparse)
# ------------------------------------------------
parser = argparse.ArgumentParser(description="Particle Cloth Topper (Isaac Sim 4.5 / deprecated API)")

parser.add_argument("--save", type=str, default="", help="Optional: save USD to this path")

# Topper geometry
parser.add_argument("--path", type=str, default="/World/Topper", help="Desired prim path for the cloth mesh")
parser.add_argument("--size_x", type=float, default=10.0)
parser.add_argument("--size_y", type=float, default=10.0)
parser.add_argument("--u_res", type=int, default=120, help="# patches along X (u_patches)")
parser.add_argument("--v_res", type=int, default=60,  help="# patches along Y (v_patches)")

# Topper pose (world)
parser.add_argument("--x", type=float, default=0.0)
parser.add_argument("--y", type=float, default=0.0)
parser.add_argument("--z", type=float, default=3.0)
parser.add_argument("--humanoid_pos_x", type=float, default = 0.0)
parser.add_argument("--humanoid_pos_y", type=float, default = 1.3)
parser.add_argument("--humanoid_pos_z", type=float, default = 4.0)

# Cloth tuning (deprecated particle cloth)
parser.add_argument("--stretch", type=float, default=15000.)
parser.add_argument("--bend", type=float, default=60000)
parser.add_argument("--shear", type=float, default=100.0)
parser.add_argument("--damping", type=float, default=0.2)
parser.add_argument("--mass", type=float, default=10., help="TOTAL mass of the cloth mesh (UsdPhysics.MassAPI.mass)")

parser.add_argument("--self_collision", action="store_true")
parser.add_argument("--self_collision_filter", action="store_true")

# Particle system tuning
parser.add_argument("--solver_iters", type=int, default=16)
parser.add_argument("--contact_scale", type=float, default=1.5, help="contactOffset = restOffset * contact_scale")
parser.add_argument("--rest_offset", type=float, default=0.1, help="contactOffset = restOffset * contact_scale")

# Material params (PBD particle material)
parser.add_argument("--particle_material_path", type=str, default="/World/TopperParticleMaterial")
parser.add_argument("--friction", type=float, default=0.6)
parser.add_argument("--drag", type=float, default=0.00)
parser.add_argument("--lift", type=float, default=0.00)

# Particle system prim path
parser.add_argument("--particle_system_path", type=str, default="/World/TopperParticleSystem")

# Optional ground + run
parser.add_argument("--add_ground", action="store_true")
parser.add_argument("--ground_z", type=float, default=0.0)

parser.add_argument("--humanoid", action="store_true")
parser.add_argument("--deformables", action="store_true")

parser.add_argument("--run", action="store_true")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--dt", type=float, default=1.0 / 60.0)

parser.add_argument("--num_envs"        , type=int  , default = 1                                               , help="Number of Instances in the Interactive Scene")
parser.add_argument("--env_spacing"     , type=float, default = 2.0                                             , help="Physical Space between the Environments (in meters)")
parser.add_argument("--num_rows"        , type=int  , default = None                                            , help="Override YAML rows")
parser.add_argument("--num_cols"        , type=int  , default = None                                            , help="Override YAML cols")
parser.add_argument("--anchor_every"    , type=int  , default = 1                                               , help="Update foam anchor kinematic targets every N sim steps (1 = every step).")
parser.add_argument("--settle_steps"    , type=int  , default = 20                                              , help="Number of Iteration that you want the sim to run without any actions so that the anchors can settle")
parser.add_argument("--reset_after"     , type=int  , default = 60000                                           , help="Reset the whole scene after n steps")
parser.add_argument("--plate_pos_eps"   , type=float, default=1e-6                                              , help="Only rewrite foam anchors if TopPlate moved by > eps (L_inf).")

# -------------------------------
# PIN CUBES + ATTACHMENTS (NEW)
# -------------------------------
parser.add_argument("--pin_dx", type=float, default=0.5, help="Pin cuboid size in X (meters)")
parser.add_argument("--pin_dy", type=float, default=0.5, help="Pin cuboid size in Y (meters)")
parser.add_argument("--pin_dz", type=float, default=0.5, help="Pin cuboid size in Z (meters)")
parser.add_argument("--pin_offset_x", type=float, default=0.1, help="Outward X offset from corner plate center (meters)")
parser.add_argument("--pin_offset_y", type=float, default=0.1, help="Outward Y offset from corner plate center (meters)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------
# 2) Imports after App Creation ---
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
# --- isaaclab and isaacsim imports --- 

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import isaacsim.core.utils.prims as prim_utils

# --- omni and pxr imports --- 
import omni
import omni.usd
from omni.physx.scripts import physicsUtils, particleUtils
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, Vt

# --- math imports --- 
import torch
import numpy as np
import ipdb as pdb

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

# --------------
# 3) Helpers ---
# --------------
# --- returning the stage --- 
def get_stage():
    return sim_utils.get_current_stage()

# --- returning the physics scene --- 
def get_physics_scene(stage):
    return sim_utils.UsdPhysics.Scene.Get(stage, "/physicsScene")  # type: ignore

def delete_prim_tree(stage, prim_path: str):
    """Delete prim (and its subtree) if it exists."""
    prim = stage.GetPrimAtPath(sim_utils.Sdf.Path(prim_path)) # type: ignore
    if prim.IsValid():
        stage.RemovePrim(sim_utils.Sdf.Path(prim_path)) # type: ignore

# --- function that creates triangulated mesh --- 
def create_triangulated_grid_mesh(
    stage,
    path: sim_utils.Sdf.Path,  # type: ignore
    size_x: float, size_y: float,
    u_patches: int, v_patches: int,
):
    """Create a triangulated plane mesh (grid) centered at origin."""
    mesh = sim_utils.UsdGeom.Mesh.Define(stage, path)  # type: ignore

    u_patches = int(max(1, u_patches))
    v_patches = int(max(1, v_patches))

    pts = []
    for j in range(v_patches + 1):
        v = j / v_patches
        y = (v - 0.5) * size_y
        for i in range(u_patches + 1):
            u = i / u_patches
            x = (u - 0.5) * size_x
            pts.append(prim_utils.Gf.Vec3f(x, y, 0.0))

    def idx(i, j):
        return j * (u_patches + 1) + i

    face_counts = []
    face_indices = []
    for j in range(v_patches):
        for i in range(u_patches):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            face_counts.extend([3, 3])
            face_indices.extend([a, b, c, a, c, d])

    mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts))  # type: ignore
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    return mesh


def create_particle_cloth_topper(stage) -> str:
    """Creates the particle cloth at args_cli.path and returns that path as string."""
    physics_scene = get_physics_scene(stage)

    u_patches = int(max(1, args_cli.u_res))
    v_patches = int(max(1, args_cli.v_res))

    cloth_mesh_path = prim_utils.Sdf.Path(args_cli.path)
    cloth_mesh = create_triangulated_grid_mesh(
        stage=stage,
        path=cloth_mesh_path,
        size_x=float(args_cli.size_x),
        size_y=float(args_cli.size_y),
        u_patches=u_patches,
        v_patches=v_patches,
    )

    # place cloth in world
    physicsUtils.setup_transform_as_scale_orient_translate(cloth_mesh)
    physicsUtils.set_or_add_scale_op(cloth_mesh, prim_utils.Gf.Vec3f(1.0, 1.0, 1.0))
    physicsUtils.set_or_add_translate_op(
        cloth_mesh,
        prim_utils.Gf.Vec3f(float(args_cli.x), float(args_cli.y), float(args_cli.z)),
    )
    physicsUtils.set_or_add_orient_op(cloth_mesh, prim_utils.Gf.Quatf(1.0, prim_utils.Gf.Vec3f(0.0, 0.0, 0.0)))

    # particle spacing (approx)
    dx = float(args_cli.size_x) / float(u_patches)
    dy = float(args_cli.size_y) / float(v_patches)
    spacing = min(dx, dy)

    restOffset = float(args_cli.rest_offset) * spacing
    contactOffset = restOffset * float(args_cli.contact_scale)

    # particle system
    particle_system_path = prim_utils.Sdf.Path(args_cli.particle_system_path)
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=particle_system_path,
        contact_offset=float(contactOffset),
        rest_offset=float(restOffset),
        particle_contact_offset=float(contactOffset),
        solid_rest_offset=float(restOffset),
        fluid_rest_offset=0.0,
        solver_position_iterations=int(args_cli.solver_iters),
        simulation_owner=physics_scene.GetPrim().GetPath(),  # must be Sdf.Path
    )

    # particle material
    particle_material_path = prim_utils.Sdf.Path(args_cli.particle_material_path)
    particleUtils.add_pbd_particle_material(
        stage,
        particle_material_path,
        friction=float(args_cli.friction),
        drag=float(args_cli.drag),
        lift=float(args_cli.lift),
    )
    physicsUtils.add_physics_material_to_prim(
        stage,
        stage.GetPrimAtPath(particle_system_path),
        particle_material_path,
    )

    # particle cloth
    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=cloth_mesh_path,
        dynamic_mesh_path=None,
        particle_system_path=particle_system_path,
        spring_stretch_stiffness=float(args_cli.stretch),
        spring_bend_stiffness=float(args_cli.bend),
        spring_shear_stiffness=float(args_cli.shear),
        spring_damping=float(args_cli.damping),
        self_collision=bool(args_cli.self_collision),
        self_collision_filter=bool(args_cli.self_collision_filter),
        pressure=0.0,
    )

    # visuals
    cloth_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([prim_utils.Gf.Vec3f(1.0, 0.6, 0.0)]))  # type: ignore

    # mass
    mass_api = sim_utils.UsdPhysics.MassAPI.Apply(cloth_mesh.GetPrim())  # type: ignore
    mass_api.GetMassAttr().Set(float(args_cli.mass))

    print(
        f"[INFO] Cloth created at: {cloth_mesh_path}\n"
        f"       u_patches={u_patches} v_patches={v_patches}\n"
        f"       spacing≈{spacing:.4f} restOffset={restOffset:.5f} contactOffset={contactOffset:.5f}\n"
        f"       particleSystem={particle_system_path}\n"
        f"       particleMaterial={particle_material_path}"
    )
    return str(cloth_mesh_path)

# --- helpers for pinning cubes for mattress topper --- 
CORNER_TOPPLATES = ["TopPlate0", "TopPlate3", "TopPlate28", "TopPlate31"]
TOPPLATE_SIZE       = (1.0, 1.0, 0.1)  # (x,y,z) meters, as you noted

def make_attachment(stage, actor0_path: sim_utils.Sdf.Path, actor1_path: sim_utils.Sdf.Path, name: str): # type: ignore
    att_path = actor0_path.AppendElementString(name)
    att = sim_utils.PhysxSchema.PhysxPhysicsAttachment.Define(stage, att_path)
    att.GetActor0Rel().SetTargets([actor0_path])
    att.GetActor1Rel().SetTargets([actor1_path])
    sim_utils.PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    return att

def create_corner_pins_from_topplates_and_attach_cloth(
    stage,
    bed: Articulation,
    cloth_path: str,
    env_ns: str = "/World/envs/env_0",
    pin_root_name: str = "TopperPins",
):
    """
    Simple version:
      - Read centers of TopPlate0/3/28/31
      - Offset outward in X/Y by (pin_offset_x, pin_offset_y)
      - Set Z = z_cloth (args_cli.z)
      - Spawn 4 rigid boxes and auto-attach cloth <-> box
    """
    # ---- get body ids for the 4 corner top plates
    corner_ids = []
    for name in CORNER_TOPPLATES:
        ids, _ = bed.find_bodies(name)
        if not ids:
            raise RuntimeError(f"[PIN] {name} not found in bed bodies")
        corner_ids.append(ids[0])

    # ---- centers: (B,4,3) ; you said assume num_env=1 => B=1
    centers = torch.stack([bed.data.body_link_pos_w[:, bid, :] for bid in corner_ids], dim=1)  # (1,4,3)

    # ---- figure outward direction from the mean of the 4 corners
    mean_xy = centers[..., :2].mean(dim=1, keepdim=True)     # (1,1,2)
    sign_xy = torch.sign(centers[..., :2] - mean_xy)         # (1,4,2)
    sign_xy[sign_xy == 0] = 1.0                              # avoid zeros

    # ---- apply ONLY the user offsets in outward direction
    pin_pos = centers.clone()
    pin_pos[..., 0] += sign_xy[..., 0] * float(args_cli.pin_offset_x)
    pin_pos[..., 1] += sign_xy[..., 1] * float(args_cli.pin_offset_y)

    # ---- Z = cloth Z (no penetration logic)
    cloth_z = float(args_cli.z)
    pin_pos[..., 2] = cloth_z

    # ---- create root prim under env_0
    pin_root = f"{env_ns}/{pin_root_name}"
    prim_utils.create_prim(pin_root, "Xform")

    cloth_sdf = sim_utils.Sdf.Path(cloth_path)  # type: ignore

    created_paths = []
    b = 0  # env_0 only

    for i, name in enumerate(CORNER_TOPPLATES):
        p = pin_pos[b, i]
        cube_path = f"{pin_root}/Pin_{name}"

        cube_prim = physicsUtils.add_rigid_box(
            stage,
            cube_path,
            size=prim_utils.Gf.Vec3f(float(args_cli.pin_dx), float(args_cli.pin_dy), float(args_cli.pin_dz)),
            position=prim_utils.Gf.Vec3f(float(p[0].item()), float(p[1].item()), float(p[2].item())),
            density=0.0,
            color=prim_utils.Gf.Vec3f(0.85, 0.85, 0.85),
        )

        make_attachment(stage, cloth_sdf, cube_prim.GetPath(), f"attachment_{name}")
        created_paths.append(str(cube_prim.GetPath()))

    print("[INFO] Pin cubes + cloth attachments created:")
    for p in created_paths:
        print("       ", p)

    return created_paths


# --------------------
# 4) Scene Builder ---
# --------------------
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
        spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75))
    )

    # --- bed articulation --- 
    bed: ArticulationCfg = KORUS_BED_CFG.replace(prim_path="{ENV_REGEX_NS}/KorusBed") # type: ignore

    # --- humanoid articulation ---
    if args_cli.humanoid:  
        humanoid: ArticulationCfg = KORUS_HUMANOID_CFG.replace( # type: ignore
            prim_path = "{ENV_REGEX_NS}/Humanoid",
            init_state = ArticulationCfg.InitialStateCfg(pos=(args_cli.humanoid_pos_x, args_cli.humanoid_pos_y, args_cli.humanoid_pos_z)) 
        ) 

    if args_cli.deformables:
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


def design_scene():
    """populates the scene with assets which cannot be accessed a isaaclab standard object types"""
    stage = get_stage()
    cloth_path = create_particle_cloth_topper(stage)
    return cloth_path

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, cloth_path: str):
    """runs the simulation loop"""
    # --- num envs ---
    # B = int(scene.num_envs)
    
    # --- extracting entities --- 
    bed: Articulation               = scene["bed"]
    if args_cli.humanoid: 
        humanoid: Articulation      = scene["humanoid"]
    
    if args_cli.deformables:
        cubes: list[DeformableObject]   = [scene[f"deformable_{ii}"] for ii in range(32)]
    
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

    # --- attachment creation flag --- 
    attachments_created = False
    stage = get_stage()
    env_ns = "/World/envs/env_0"
    pin_root_path = f"{env_ns}/TopperPins"

    # --- cell inflation indices ---
    cell_0_idx      = 16; cell_1_idx    = 1 ; cell_2_idx    = 14; cell_3_idx    = 24;   
    cell_4_idx      = 23; cell_5_idx    = 2 ; cell_6_idx    =  0; cell_7_idx    = 15;   
    cell_8_idx      =  8; cell_9_idx    = 20; cell_10_idx   =  9; cell_11_idx   =  3; 
    cell_12_idx     = 10; cell_13_idx   = 11; cell_14_idx   = 21; cell_15_idx   = 12;  
    cell_16_idx     =  4; cell_17_idx   =  7; cell_18_idx   = 22; cell_19_idx   =  5;  
    cell_20_idx     = 13; cell_21_idx   =  6; cell_22_idx   = 18; cell_23_idx   = 19;   
    cell_24_idx     = 17; cell_25_idx   = 25; cell_26_idx   = 26; cell_27_idx   = 27;   
    cell_28_idx     = 28; cell_29_idx   = 29; cell_30_idx   = 30; cell_31_idx   = 31;   

    inflate_min = 0.0
    inflate_max = 0.9
    inflate_amount:float    = 0.001  # start value # NOTE: modify this if you are not doing the incremental approach
    inflate_step:float      = 0.001  # +0.01 per sim step (tune)

    # --- simulation loop starts --- 
    while simulation_app.is_running():
    
        # ---------------
        # RESET BLOCK ---
        # ---------------
        if (count % args_cli.reset_after) == 0:
                        
            # --- reset counters --- 
            sim_time = 0.0
            count = 0

            # --- reset bed ---    
            root_state_bed = bed.data.default_root_state.clone()
            root_state_bed[:,:3] += scene.env_origins 
            bed.write_root_pose_to_sim(root_state_bed[:, :7])
            bed.write_root_velocity_to_sim(root_state_bed[:, 7:])
            joint_pos_bed = bed.data.default_joint_pos.clone()
            joint_vel_bed = bed.data.default_joint_vel.clone()
            # bed.write_joint_state_to_sim(joint_pos_bed, joint_vel_bed) # NOTE: uncomment this a comment the later part if you want no cell inflation
            
            # NOTE: change cell inflation here
            joint_pos_bed[:, cell_4_idx]    = inflate_amount 
            joint_pos_bed[:, cell_12_idx]   = inflate_amount
            joint_pos_bed[:, cell_7_idx]    = inflate_amount 
            joint_pos_bed[:, cell_15_idx]   = inflate_amount 
            bed.write_joint_state_to_sim(joint_pos_bed, joint_vel_bed)
            bed.set_joint_position_target(joint_pos_bed)
            bed_joint_target = joint_pos_bed.clone()

            # reset humanoid
            if args_cli.humanoid: 
                root_state_humanoid = humanoid.data.default_root_state.clone()
                root_state_humanoid[:,:3] += scene.env_origins
                humanoid.write_root_pose_to_sim(root_state_humanoid[:, :7])
                humanoid.write_root_velocity_to_sim(root_state_humanoid[:, 7:])
                target_humanoid_joint_angles = humanoid.data.default_joint_pos.clone()
                joint_pos_humanoid = target_humanoid_joint_angles.clone()  
                joint_vel_humanoid = torch.zeros_like(joint_pos_humanoid)
                humanoid.write_joint_state_to_sim(joint_pos_humanoid, joint_vel_humanoid)
                # set joint position
                humanoid.set_joint_position_target(joint_pos_humanoid)

            if args_cli.deformables:
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
            if args_cli.deformables: 
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

            if not attachments_created:
                create_corner_pins_from_topplates_and_attach_cloth(
                    stage=stage,
                    bed=bed,
                    cloth_path=cloth_path,
                    env_ns="/World/envs/env_0",
                    pin_root_name="TopperPins",
                )
                attachments_created = True
            
            # --- let constraints settle before building indentation baseline ---
            for _ in range(args_cli.settle_steps):
                scene.write_data_to_sim()
                sim.step(render=True)          
                scene.update(sim_dt)

            print(f"[INFO]: Resetting Bed + Humanoid + Re-anchoring Deformable Foams")

        # -------------------------
        # STEP AND UPDATE BLOCK ---
        # -------------------------
        # --- anchoring deformables to the top plate (if the plate has moved enough) --- 
        if args_cli.deformables: 
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

        # --- incremental cell inflation --- (NOTE: this is temp code)            
        inflate_every = 1  # 1 = every step
        if (count % inflate_every) == 0:
            inflate_amount = min(inflate_amount + inflate_step, inflate_max) # type: ignore
            bed_joint_target[:, cell_4_idx]     = inflate_amount
            bed_joint_target[:, cell_12_idx]    = inflate_amount 
            bed_joint_target[:, cell_7_idx]     = inflate_amount
            bed_joint_target[:, cell_15_idx]    = inflate_amount 
            bed.write_joint_state_to_sim(bed_joint_target, torch.zeros_like(bed_joint_target))
            bed.set_joint_position_target(bed_joint_target)

        # --- step and update --- 
        # write data to sim
        scene.write_data_to_sim()
        # step through the simulation
        sim.step(render=bool(True)) 
        # increment counters
        sim_time += sim_dt 
        count += 1
        # update all buffers
        scene.update(sim_dt)


def main():
    """ main function"""
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim     = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view((2.0, 0.0, 2.5), (-0.5, 0.0, 0.5))

    scene_cfg = KorusInteractiveSceneCfg(num_envs=args_cli.num_envs,env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)
    
    cloth_path = design_scene()

    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(
        sim     = sim, 
        scene   = scene, 
        cloth_path = cloth_path
    )

if __name__ == "__main__":
    main()
    simulation_app.close()
