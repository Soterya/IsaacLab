"""
Usage:
    $ cd ~/IsaacLab
    $ ./isaaclab.sh -p scripts/korus/run_foam_bed_sim.py
    
    NOTE: 
        1. Make sure you run this from a `env_isaaclab` conda environment.   
        2. If you want to run it with a ros topic then checkout "run_foam_bed_sim_ros.py"
"""

# ------------------------------------------
# IMPORTS FOR THE ISAACSIM APP 
# ------------------------------------------
import argparse
import torch
from isaaclab.app import AppLauncher
import numpy as np

# ------------------------------------------
# LAUNCH THE ISAACSIM APP (with argparse)
# ------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab")
parser.add_argument("--num_envs", type=int, default=1)        # TODO: important later making multiple instances of the simulation                                               
parser.add_argument("--env_spacing", type=float, default=2.0) # TODO: important later making multiple instances of the simulation
parser.add_argument("--log_path", type=str, default="nodal_positions.txt") 
parser.add_argument("--every", type=int, default=1, help="Log every N frames to reduce file size")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------
# Imports after app creation
# ------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.utils import configclass

# ---------------------------
# GLOBAL PATHS
# ---------------------------
ENV_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
)

# --------------------------------------------------------
# Manual scene builder (USD + deformable + rigid sphere)
# --------------------------------------------------------
@configclass
class KorusBedSceneCfg(InteractiveSceneCfg):
    """
    Configuration for the Korus Bed Scene. NOTE: ONLY DEFINE CONFIGS HERE, NOT OBJECTS 
    """
    env_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/World", 
        spawn=sim_utils.UsdFileCfg(usd_path=ENV_USD),
    )

    # Deformable cuboid
    cube_cfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/World/Environment/DeformableCuboid",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, .2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=10,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(.0, .0, 1.5)),
        debug_vis=True,
    )

    # Rigid sphere (falling on the deformable cuboid)
    sphere_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/World/Environment/RigidSphere",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )


# ----------------------
# UTILITY FUNCTIONS
# ----------------------
def get_top_idx_by_known_z(cube, z_target=1.6, decimals=1, atol=None):
    """
    Return (mask, idx) for nodes whose DEFAULT world-z equals z_target
    (within tolerance). Use either rounding (decimals) or atol.
    """
    default_pos = cube.data.default_nodal_state_w[..., :3][0]  # (N,3) world-frame
    z = default_pos[:, 2]

    if decimals is not None:
        z_rounded = torch.round(z * (10**decimals)) / (10**decimals)
        z_target = round(float(z_target), decimals)
        mask = (z_rounded == z_target)
    else:
        # choose a tolerance if not provided
        if atol is None:
            # auto-pick something smaller than the plane spacing
            uniq = torch.unique(torch.round(z * 1e5) / 1e5)
            uniq = torch.sort(uniq).values
            dz = float(uniq[-1] - uniq[-2]) if uniq.numel() >= 2 else 1e-4
            atol = max(1e-5, 0.25 * dz)
        mask = torch.isclose(z, torch.tensor(z_target, device=z.device, dtype=z.dtype), atol=atol)

    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    return mask, idx

# ------------------------------------------
# Simulation loop: print nodal positions
# ------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, scene:InteractiveScene):
    """
    running the simulation loop
    """
    cube   = scene["cube_cfg"]
    sphere = scene["sphere_cfg"]

    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0 

    # simulate the physics
    while simulation_app.is_running():
        # this block resets the simulation
        if count % 400 == 0:
            # reset counters 
            sim_time = 0.0 
            count    = 0

            # reset deforamble cube state
            nodal_state_cube = cube.data.default_nodal_state_w.clone() # state where the cube started
            cube.write_nodal_state_to_sim(nodal_state_cube) 
            
            # reset the rigid sphere state
            root_state = sphere.data.default_root_state.clone() # pose where the sphere started
            sphere.write_root_pose_to_sim(root_state[:, :7])
            sphere.write_root_velocity_to_sim(root_state[:, 7:])

            # reset the internal state
            cube.reset()
            sphere.reset()
            print(f"[INFO]: Resetting deformable and rigid object states")

        # perform step
        sim.step(render=True)
        # update sim time
        sim_time += sim_dt
        count += 1
        # update buffers
        cube.update(sim_dt)
        sphere.update(sim_dt)

        # ---------------------------------
        # Extracting all Deformable Data
        # ---------------------------------
        # root position of the deformable body wrt world frame 
        root_pose_w = np.round(cube.data.root_pos_w.detach().cpu().numpy(), 3)
        # state of the deformable body when it spawned in the simulation
        default_nodal_state = np.round(cube.data.default_nodal_state_w.detach().cpu().numpy(), 3)
        # position of all vertices in the body. Shape: (num of instances, num vertices in the body, 3)
        nodal_pos_w = np.round(cube.data.nodal_pos_w.detach().cpu().numpy(), 3)
        # velocity of all vertices in the body  (num of instances, num vertices in the body, 3)
        nodal_vel_w = np.round(cube.data.nodal_vel_w.detach().cpu().numpy(), 3)


        # extracting indices of all top surface vertices and then pushing to cpu
        top_surface_mask, top_surface_idx = get_top_idx_by_known_z(cube, z_target=1.6, decimals=1)
        top_surface_mask = top_surface_mask.detach().cpu().numpy()
        top_surface_idx  = top_surface_idx.detach().cpu().numpy()
        
        top_surface_nodal_pos_w = nodal_pos_w[0, top_surface_idx, :]
        top_surface_nodal_vel_w = nodal_vel_w[0, top_surface_idx, :]

        # Printing all Deformable Data of the "x" th deformable body every "y" th timestep for "z" th vertex in the deformable body NOTE: x,y,z are arbitrary numbers   
        x = 0    # index of deformable body
        y = 10   # interval for printing data
        z = 100  # index of of the vertix in the deformable body `x`
        if count % y == 0:
            print(f"[INFO]: Deformable Body Data for Body Number: {x} (w.r.t World Frame)")
            print(f"        Root Position of Body: {root_pose_w}, Shape: {root_pose_w.shape}")
            print(f"        Default Nodal State  : {default_nodal_state[x, z, :]}, Shape: {default_nodal_state.shape}")
            print(f"        Nodal Positions      : {nodal_pos_w[x, z, :]}, Shape: {nodal_pos_w.shape}")
            print(f"        Nodal Velocities     : {nodal_vel_w[x, z, :]}, Shape: {nodal_vel_w.shape}")

            # print(f"        Nodal Velocities (w.r.t World): {nodal_vel_w[x, z, :]}, Shape: {nodal_vel_w.shape}")
            
            # print(f"[INFO]: Top Surface Nodal Pos:\n{top_nodal_pos_w}")
            

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    """
    main function
    """
    # initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # set main camera
    sim.set_camera_view(eye=(3.0, 0.0, 1.0), target=(0.0, 0.0, 0.5))

    # build scene and run
    scene_cfg = KorusBedSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    # play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    
    # run the simulator
    run_simulator(sim, scene) # args_cli.log_path, args_cli.every)

if __name__ == "__main__":
    # run the main function
    main()
    # for smooth closing of the sim app after ctrl+c
    simulation_app.close()
