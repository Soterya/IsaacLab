"""
Usage:
    $ cd ~/IsaacLab
    $ ./isaaclab.sh -p scripts/korus/run_foam_bed_sim_pinned.py
    
    TODO: Imports are not successfull because it needs the full kit version. Need to make changes to the script. 

    NOTE: 
        1. Make sure you run this from a `env_isaaclab` conda environment.   
        2. If you want to run it with a ros topic then checkout "run_foam_bed_sim_pinned_ros.py"
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
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg 

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
# Paths
ENV_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
INFLATABLE_CELL_USD = ""


# Object Placement with Scene 
# NOTE: The code only works when:
#       1. DECIMALS = 1; and FOAM_SIZE[2] and BASE_SIZE[2] are incremented by 0.1 eg. 0.1, 0.2, 0.3 and so on...   
GROUND_TO_BASE_BOTTOM = 1.0

BASE_SIZE = (1.0, 1.0, .1)
FOAM_SIZE = (1.0, 1.0, .2)
SPHERE_RADIUS = 0.2

BASE_ORIGIN = (0.0, 0.0, GROUND_TO_BASE_BOTTOM + round(BASE_SIZE[2]/2,2))
FOAM_ORIGIN = (0.0, 0.0, BASE_ORIGIN[2] + round(BASE_SIZE[2]/2,2) + round(FOAM_SIZE[2]/2,2)) 
SPHERE_ORIGIN = (0.0, 0.0, 5.0)

DECIMALS = 1

# ------------------------------------------
# Manual scene builder (USD + deformable + rigid sphere)
# ------------------------------------------
def design_scene():
    """
    populate the scene here
    """
    # Import default environment under /World
    env_cfg = sim_utils.UsdFileCfg(usd_path=ENV_USD)
    env_cfg.func("/World", env_cfg)

    base_cfg = RigidObjectCfg(
        prim_path="/World/Environment/BasePlate",
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

    # Deformable cuboid
    deform_cfg = DeformableObjectCfg(
        prim_path="/World/Environment/DeformableCuboid",
        spawn=sim_utils.MeshCuboidCfg(
            size=FOAM_SIZE,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=10,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=deform_cfg)

    # Rigid sphere (falling on the deformable cuboid)
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/Environment/RigidSphere",
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

    return {"cube_object": cube_object, "sphere_object": sphere_object, "base_object": base_object}

# ---------------------------------------------
# UTILITY FUNCTIONS : FOR OBTAINING INDICES
# ---------------------------------------------
def get_surface_indices_by_known_z(cube, z_target, decimals=DECIMALS, atol=None):
    """
    Return (mask, idx) for nodes whose DEFAULT world-z equals z_target
    (within tolerance). Use either rounding (decimals) or atol.
    NOTE: This function can be used for bottom as well as top surface for deformable objects, 
    NOTE: Needs to be Done Once per Trial
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

    # return mask, idx
    return idx

def get_surface_corner_indices_from_idx(cube, surface_idx, env_id: int = 0):
    """
    Given a set of surface node indices (w.r.t. ALL nodes), find the 4 corner node indices
    on that surface: [BL, TL, BR, TR] in global indexing.

    Args:
        cube: DeformableObject
        surface_idx: 1D torch.LongTensor of shape (K,) with global node indices on the surface
        env_id: which batch/env to use (default 0)

    Returns:
        corners_idx_global: torch.LongTensor of shape (4,)
            Global node indices in order: [BL, TL, BR, TR]
    """
    # default node positions (N, 3) for this env
    default_pos = cube.data.default_nodal_state_w[env_id, :, :3]  # (N,3) -> positions of all vertices
    pts_surface = default_pos.index_select(0, surface_idx)        # (K,3) -> positions of all bottom surface vertices 
    xy = pts_surface[:, :2]                                       # (K,2)
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = torch.min(x), torch.max(x)
    ymin, ymax = torch.min(y), torch.max(y)
    # targets: BL, TL, BR, TR
    targets = torch.stack([
        torch.tensor([xmin, ymin], dtype=xy.dtype, device=xy.device),  # BL
        torch.tensor([xmin, ymax], dtype=xy.dtype, device=xy.device),  # TL
        torch.tensor([xmax, ymin], dtype=xy.dtype, device=xy.device),  # BR
        torch.tensor([xmax, ymax], dtype=xy.dtype, device=xy.device),  # TR
    ], dim=0).round(decimals=2)  # (4,2)

    # squared distances from each surface node to each target
    # (K,4) = ((K,1,2) - (1,4,2))^2 -> sum over last dim
    diffs = xy[:, None, :] - targets[None, :, :]
    dists = torch.sum(diffs * diffs, dim=2)

    # choose nearest UNIQUE node for each corner
    chosen_local = []
    for j in range(4):
        order = torch.argsort(dists[:, j])  # ascending
        pick = None
        for cand in order.tolist():
            if cand not in chosen_local:
                pick = cand
                break
        if pick is None:
            pick = order[0].item()
        chosen_local.append(pick)

    chosen_local = torch.tensor(chosen_local, dtype=torch.long, device=surface_idx.device)  # (4,)
    corners_idx_global = surface_idx.index_select(0, chosen_local)  # (4,)
    
    return corners_idx_global

# ---------------------------------------------
# UTILITY FUNCTIONS : FOR OBTAINING POSITIONS 
# ---------------------------------------------    
def get_pin_targets_on_rigid_body(root_pos_w_rigid, root_quat_w_rigid):
    """
    This will give POSITION targets on the rigid base plate, where the the bottom surface corners will have kinematic targets. 
    NOTE: 
        1. These targets will keep updating as the cube moves
        2. Needs to be done in every iteration of the simulation
    TODO:
        1. This function is not currently Rotation aware, meaning change in rotation is not incorporated, which is okay 
           currently because base plate if fixed angularly.  
    """
    current_center_pos   = root_pos_w_rigid
    current_center_angle = root_quat_w_rigid # TODO: To be incorporated later    
    half_x, half_y, half_z = round(BASE_SIZE[0] / 2.0,2), round(BASE_SIZE[1] / 2.0,2), round(BASE_SIZE[2] / 2.0,2)

    corners_dict = {}

    for idx, center in enumerate(current_center_pos):
        
        cx, cy, cz = center
        z_top = cz + half_z
        corners = np.array([
            (cx - half_x, cy - half_y, z_top),  # bottom-left  (xmin, ymin) 
            (cx - half_x, cy + half_y, z_top),  # top-left     (xmin, ymax)
            (cx + half_x, cy - half_y, z_top),  # bottom-right (xmax, ymin)
            (cx + half_x, cy + half_y, z_top),  # top-right    (xmax, ymax)
        ], dtype=float)
        corners = np.round(corners,2).tolist()

        corners_dict[f"cuboid_{idx}"] = corners

    return corners_dict

# ------------------------------------------
# Simulation loop: print nodal positions
# ------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, entities):
    """
    running the simulation loop
    """
    cube   = entities["cube_object"]
    sphere = entities["sphere_object"]
    base   = entities["base_object"]

    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0 

    # prepare kinematic target buffer once
    B,N,_ = cube.data.nodal_pos_w.shape # B: num of deform objects, N: Num of Vertices in a Deform Body, _ : x,y,z
    nodal_kinematic_target = cube.data.nodal_kinematic_target.clone()  # (B, N, 4) # [x, y, z, flag] 
    # NOTE:
    #       1. 0.0 = constrained (the solver locks this node to the specified [x,y,z]).
    #       2. 1.0 = free (the node is left to normal dynamics; [x,y,z] is ignored). 

    # simulate the physics
    while simulation_app.is_running():
        # -----------------------------------------------------------------------------------------------
        # NOTE: this block resets the simulation. never touch it. NEEDS TO EXECUTE ONLY ONCE PER TRIAL
        # -----------------------------------------------------------------------------------------------
        
        if count % 400 == 0:
            # reset counters 
            sim_time = 0.0 
            count    = 0
            # reset base plate state
            root_state_base = base.data.default_root_state.clone()
            base.write_root_pose_to_sim(root_state_base[:, :7])
            base.write_root_velocity_to_sim(root_state_base[:, 7:])
            # reset deforamble cube state
            nodal_state_cube = cube.data.default_nodal_state_w.clone() # state where the cube started
            cube.write_nodal_state_to_sim(nodal_state_cube) 
            # reset the rigid sphere state
            root_state_sphere = sphere.data.default_root_state.clone() # pose where the sphere started
            sphere.write_root_pose_to_sim(root_state_sphere[:, :7])
            sphere.write_root_velocity_to_sim(root_state_sphere[:, 7:])
            # reset the internal state
            cube.reset(); sphere.reset(); base.reset()
            print(f"[INFO]: Resetting deformable and rigid object states")

            # ----------------------------------------------
            # Extracting Default data and pushing to CPU
            # ----------------------------------------------
            default_nodal_state_deformable       = np.round(cube.data.default_nodal_state_w.detach().cpu().numpy(), 3)  
            default_root_state_rigid             = np.round(base.data.default_root_state.detach().cpu().numpy(), 3)

            # --- Processing the Default Data to get TOP and BOTTOM Surface Indices of Deformable Body --- 
            # extracting indices of all top and bottom surface vertices of deformable body and then pushing to cpu
            top_surface_idx_torch = get_surface_indices_by_known_z(cube=cube, z_target=round(FOAM_ORIGIN[2] + FOAM_SIZE[2]/2, 2), decimals=DECIMALS)
            top_surface_idx  = top_surface_idx_torch.detach().cpu().numpy()
            bottom_surface_idx_torch = get_surface_indices_by_known_z(cube=cube, z_target=round(FOAM_ORIGIN[2] - FOAM_SIZE[2]/2, 2), decimals=DECIMALS)
            bottom_surface_idx  = bottom_surface_idx_torch.detach().cpu().numpy()
            # extracting indices of all bottom corner vertices of deformable body and then pushing to cpu
            bottom_corner_idx_torch = get_surface_corner_indices_from_idx(cube, bottom_surface_idx_torch, env_id = 0)
            bottom_corner_idx = bottom_corner_idx_torch.detach().cpu().numpy()

            # --- Attaching the Bottom Corners of the Deformable Body to Top Corners of Rigid Body using Kinematic Targetting. 
            # start with all vertices FREE (flag = 1.0)
            nodal_kinematic_target[..., :3] = cube.data.nodal_pos_w    # current positions
            nodal_kinematic_target[..., 3]  = 1.0                      # 1.0 = free
            # pin just the 4 bottom-corner nodes (flag = 0.0)
            nodal_kinematic_target[:, bottom_corner_idx_torch, 3] = 0.0
            cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            print(f"[INFO]: Reset done. Corner indices = {bottom_corner_idx_torch.tolist()}")

        # ------------------------------------------
        # this performs a step and updates buffers
        # ------------------------------------------
        # perform step
        sim.step(render=True)
        # update sim time
        sim_time += sim_dt
        count += 1
        # update buffers
        cube.update(sim_dt); sphere.update(sim_dt); base.update(sim_dt)

        # ------------------------------------------
        # Extracting all Deformable Data for Foam
        # ------------------------------------------
        # root position of the deformable body wrt world frame 
        root_pose_w = np.round(cube.data.root_pos_w.detach().cpu().numpy(), 3)
        # position of all vertices in the body. Shape: (num of instances, num vertices in the body, 3)
        nodal_pos_w = np.round(cube.data.nodal_pos_w.detach().cpu().numpy(), 3)

        # indexing positions of all top surface vertices  
        top_surface_nodal_pos_w = nodal_pos_w[:, top_surface_idx, :]
        # indexing positions of all bottom surface vertices 
        bottom_surface_nodal_pos_w = nodal_pos_w[:, bottom_surface_idx, :]

        # indexing positions of 4 corner indices of bottom surface (all cuboids)
        bottom_corners_nodal_pos_w_all = nodal_pos_w[:, bottom_corner_idx, :]
        bottom_corners_nodal_pos_w = bottom_corners_nodal_pos_w_all[0]
        # ------------------------------------------
        # Extracting all Rigid Data for Base Plate
        # ------------------------------------------
        # root position and orientation of the base plate
        root_pos_w_rigid          = np.round(base.data.root_pos_w.detach().cpu().numpy(), 3)
        root_quat_w_rigid         = np.round(base.data.root_quat_w.detach().cpu().numpy(), 3) 
        
        # getting kinematic target (top corner positions) from base plate for pinning
        pin_targets_on_rigid_body_dict = get_pin_targets_on_rigid_body(root_pos_w_rigid, root_quat_w_rigid)
        pin_targets_on_rigid_body = np.array(pin_targets_on_rigid_body_dict["cuboid_0"], dtype=float)

        targets_b = torch.tensor(
            [pin_targets_on_rigid_body_dict[f"cuboid_{b}"] for b in range(B)],    # list of 4x3
            dtype=nodal_kinematic_target.dtype,
            device=nodal_kinematic_target.device,
        )  # (B, 4, 3), order: [BL, TL, BR, TR]

        # Place those into the nodal_kinematic_target at the bottom-corner indices
        nodal_kinematic_target[:, bottom_corner_idx_torch, :3] = targets_b
        cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
        cube.write_data_to_sim()  # keep if you're mirroring the tutorial pattern


        # Printing all Deformable Data of the "x" th deformable body every "y" th timestep for "z" th vertex in the deformable body NOTE: x,y,z are arbitrary numbers   
        x = 0    # index of deformable body
        y = 10   # interval for printing data
        z = 100  # index of of the vertex in the deformable body `x`
        if count % y == 0:
            # # FOR DEFORMABLE CUBOID
            # print(f"[INFO]: Deformable Body Data for Body Number: {x} (w.r.t World Frame)")
            # print(f"        Root Position of Body  : {root_pose_w}, Shape: {root_pose_w.shape}")
            # print(f"        Default Nodal State    : {default_nodal_state_deformable[x, z, :]}, Shape: {default_nodal_state_deformable.shape}")
            # print(f"        Nodal Positions        : {nodal_pos_w[x, z, :]}, Shape: {nodal_pos_w.shape}")
            # print(f"        TOP    Nodal Positions : {top_surface_nodal_pos_w[x, z, :]}, Shape: {top_surface_nodal_pos_w.shape}")
            # print(f"        BOTTOM Nodal Positions : {bottom_surface_nodal_pos_w[x, z, :]}, Shape: {bottom_surface_nodal_pos_w.shape}")
            
            # # FOR RIGID BASE PLATE
            # print(f"[INFO]: Rigid Body Data for Body Number: {x} (w.r.t World Frame)")
            # print(f"        Root Position of Body: {root_pos_w_rigid}, Shape: {root_pos_w_rigid.shape}")
            # print(f"        Default Root State   : {default_root_state_rigid[:, :3]}, Shape: {default_root_state_rigid.shape}")

            # DEBUG FOR PINNING
            print(f"[INFO]: Debug for Pinning: {x} (w.r.t World Frame)")
            print(f"        Bottom Corners Pos of Defomable Body:\n{bottom_corners_nodal_pos_w_all[x]}, Shape: {bottom_corners_nodal_pos_w_all.shape}")
            key = f"cuboid_{x}"
            print(f"        Pin Targets on the Rigid Body       :\n{np.array(pin_targets_on_rigid_body_dict[key], dtype=float)}, Shape: {np.array(pin_targets_on_rigid_body_dict[key]).shape}")



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
    entities = design_scene()
    
    # play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    
    # run the simulator
    run_simulator(sim, entities,) # args_cli.log_path, args_cli.every)

if __name__ == "__main__":
    # run the main function
    main()
    # for smooth closing of the sim app after ctrl+c
    simulation_app.close()
