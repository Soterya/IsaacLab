# ------------------------------------
# --- Imports for the App Creation ---
# ------------------------------------
import argparse
import torch
from isaaclab.app import AppLauncher
import numpy as np

# -----------------------------------------------
# --- Launch the IsaacSim App (with argparse) ---
# -----------------------------------------------
parser = argparse.ArgumentParser(description="Korus Digital Twin using IsaacLab")
parser.add_argument("--num_envs", type=int, default=1)        # TODO: important later making multiple instances of the simulation                                               
parser.add_argument("--env_spacing", type=float, default=2.0) # TODO: important later making multiple instances of the simulation
parser.add_argument("--log_path", type=str, default="nodal_positions.txt") 
parser.add_argument("--every", type=int, default=1, help="Log every N frames to reduce file size")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------
# --- Imports after app creation ---
# ----------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg, Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import omni.usd 
# ----------------------------------
# --- Imports for Keyboard Input ---
# ----------------------------------
import omni.appwindow
import carb
from carb.input import KeyboardEventType, KeyboardInput
import ipdb as pdb

# --------- plotting (save-to-file) ----------
import os
import matplotlib
matplotlib.use("Agg")  # no GUI needed; save PNGs
import matplotlib.pyplot as plt

# ------------------------ 
# --- Global Variables ---
# ------------------------
# Paths
ENV_USD =             "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
# Object Placement with Scene # NOTE: The code only works when: 1. DECIMALS = 1; and FOAM_SIZE[2] and BASE_SIZE[2] are incremented by 0.1 eg. 0.1, 0.2, 0.3 and so on...   
GROUND_TO_BASE_BOTTOM = 0.0
# Sizes of Geometries
BASE_SIZE = (1.0, 1.0, .1)
FOAM_SIZE = (1.0, 1.0, .2)
SPHERE_RADIUS = 0.2
# Origin of Geometries
BASE_ORIGIN = (0.0, 0.0, GROUND_TO_BASE_BOTTOM + round(BASE_SIZE[2]/2,2))
FOAM_ORIGIN = (0.0, 0.0, BASE_ORIGIN[2] + round(BASE_SIZE[2]/2,2) + round(FOAM_SIZE[2]/2,2)) 
SPHERE_ORIGIN = (2.0, 2.0, 1.0)
# Tolerance Val
DECIMALS = 1

MAP_DIR = "top_surface_maps"  # where we save maps (optional)
os.makedirs(MAP_DIR, exist_ok=True)

# --------------------------------------------------------------
# --- Manual Scene Builder (USD + deformable + rigid sphere) ---
# --------------------------------------------------------------
def design_scene():
    """
    populate the scene here
    """
    # opening an existing scene # adding lights for illumination
    scene_context = omni.usd.get_context()
    scene_context.open_stage(ENV_USD)
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    dome.func("/World/DomeLight", dome)
    
    # define rigid cuboid (base plate) -> has rigid body data
    base_cfg = RigidObjectCfg(
        prim_path="/World/BasePlate",
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

    # define deformable cuboid (foam) -> has FEM Data
    deform_cfg = DeformableObjectCfg(
        prim_path="/World/DeformableCuboid",
        spawn=sim_utils.MeshCuboidCfg(
            size=FOAM_SIZE,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=5,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=0.2,
                youngs_modulus=3e4,
                dynamic_friction=1., 
                elasticity_damping=0.06,
                damping_scale=1.,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=FOAM_ORIGIN),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=deform_cfg)

    # define rigid sphere (falling on deformable foam)
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/RigidSphere",
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


# ----------------------------------------------------------------------------------------
# --- Utility Function : Converting Cauchy's Stress Tensor to Von Misses Stress Scalar ---
# ----------------------------------------------------------------------------------------
def von_mises_from_cauchy(stress: torch.Tensor) -> torch.Tensor:
    """
    stress: (..., 3, 3) Cauchy stress tensor (Pa)
    returns: (...) von Mises equivalent stress (Pa)
    NOTE: This function might not be necessary since, Von Misses is not indicative of contace sensing
    """
    # Make sure we use the symmetric part (numerics can introduce asymmetry)
    S = 0.5 * (stress + stress.transpose(-1, -2))

    sxx = S[..., 0, 0]
    syy = S[..., 1, 1]
    szz = S[..., 2, 2]
    sxy = S[..., 0, 1]
    syz = S[..., 1, 2]
    szx = S[..., 2, 0]

    vm2 = 0.5 * (
        (sxx - syy) ** 2 +
        (syy - szz) ** 2 +
        (szz - sxx) ** 2 +
        6.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    )
    return torch.sqrt(torch.clamp(vm2, min=0.0))


# ----------------------------------------------------------------
# --- Utility Function : Finds Indices of Sim Surface Vertices ---
# ----------------------------------------------------------------
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

    return idx # return mask, idx

# ----------------------------------------------------------------
# --- Utility Function : Finds Indices of Sim Surface Vertices ---
# ----------------------------------------------------------------
def get_top_sim_surface_tets_from_vertex_set(cube, top_vertex_idx: torch.Tensor, env_id: int = 0,
                                             require_upward_normal: bool = True):
    """
    NOTE: This is for getting the Ids of Tetrahedral Sim Elements
    Fast way to find top-surface sim tets when you already know the top-surface vertex indices.

    Returns:
      - top_tet_ids: (Kt,) unique tet ids that have a top face
      - faces_top:  (Kf,3) the actual top faces (triangle vertex indices)
      - parent_top:(Kf,) parent tet id for each face in faces_top
    """
    # pull sim tet connectivity
    view = cube.root_physx_view
    # view.get_sim_element_indices() -> shape (count, max_sim_elements_per_body, 4)
    tets = torch.as_tensor(view.get_sim_element_indices()).long()[env_id]   # (Ne,4)
    # filter padded rows if present
    if (tets < 0).any():
        tets = tets[(tets >= 0).all(dim=1)]

    device = tets.device
    Ns = cube.data.nodal_pos_w.shape[1]

    # build vertex-in-top-set mask
    top_mask = torch.zeros(Ns, dtype=torch.bool, device=device)
    top_mask[top_vertex_idx.to(device)] = True

    # count how many of each tet's 4 vertices lie on the top set
    cnt = top_mask[tets].sum(dim=1)                 # (Ne,)
    top_tet_sel = cnt >= 3
    top_tet_ids = torch.nonzero(top_tet_sel, as_tuple=False).squeeze(1)   # (Kt,)

    # extract the actual top faces from those tets
    faces_top = tets.new_empty((0, 3))
    parent_top = tets.new_empty((0,), dtype=torch.long)
    if top_tet_ids.numel() > 0:
        tt = tets[top_tet_ids]                      # (Kt,4)
        f0 = tt[:, [1,2,3]]
        f1 = tt[:, [0,2,3]]
        f2 = tt[:, [0,1,3]]
        f3 = tt[:, [0,1,2]]
        all_faces = torch.stack([f0, f1, f2, f3], dim=1)   # (Kt,4,3)
        is_top_face = top_mask[all_faces].all(dim=2)       # (Kt,4)

        k_idx, f_idx = torch.nonzero(is_top_face, as_tuple=True)
        faces_top  = all_faces[k_idx, f_idx, :]            # (Kf,3)
        parent_top = top_tet_ids[k_idx]                    # (Kf,)

        if require_upward_normal and faces_top.numel() > 0:
            # Orient normals using current sim nodal positions
            P = cube.data.nodal_pos_w[env_id][faces_top]   # (Kf,3,3)
            v1 = P[:,1,:] - P[:,0,:]
            v2 = P[:,2,:] - P[:,0,:]
            n  = torch.cross(v1, v2, dim=1)
            keep = n[:, 2] > 0.0
            faces_top  = faces_top[keep]
            parent_top = parent_top[keep]

    return top_tet_ids, faces_top, parent_top


# ----------------------------------------------
# --- Simulation loop: print nodal positions ---
# ----------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, entities):
    """
    """
    # ---------- unpack entities ----------
    cube            = entities["cube_object"]
    sphere          = entities["sphere_object"]
    base            = entities["base_object"]

    # ---------- sim params ----------
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # ---------- main loop ----------
    while simulation_app.is_running():
        # ------------------------------
        # --- Reset every 1000 steps ---
        # ------------------------------
        if count % 1000 == 0:
            
            sim_time = 0.0
            count    = 0

            # reset base
            root_state_base = base.data.default_root_state.clone()
            base.write_root_pose_to_sim(root_state_base[:, :7])
            base.write_root_velocity_to_sim(root_state_base[:, 7:])

            # reset deformable cube
            nodal_state_cube = cube.data.default_nodal_state_w.clone()
            cube.write_nodal_state_to_sim(nodal_state_cube)

            # reset sphere
            root_state_sphere = sphere.data.default_root_state.clone()
            sphere.write_root_pose_to_sim(root_state_sphere[:, :7])
            sphere.write_root_velocity_to_sim(root_state_sphere[:, 7:])

            # reset internal state trackers
            cube.reset(); sphere.reset(); base.reset()
            print("[INFO]: Resetting deformable and rigid object states")

            deformable_object_id = 0
            # Getting the Default Nodal Positions of all Nodes (Sim Vertices Positions)   
            default_nodal_pos_w = cube.data.default_nodal_state_w[deformable_object_id, :, :3]
            # Getting Indices of all Top Surface Sim Vertices of Deformable Body
            top_surface_idx_torch = get_surface_indices_by_known_z(cube=cube, z_target=round(FOAM_ORIGIN[2] + FOAM_SIZE[2]/2, 2), decimals=DECIMALS)
            # Soft Body View that exposes the API 
            soft_body_view = cube.root_physx_view
            # Getting Max Sim Elements per Body - Number of Tetrahedral Elements
            max_sim_elements_per_body = soft_body_view.max_sim_elements_per_body
            # Getting Max Sim Vertices per Body - Number of Nodes (Sim Vertices)
            max_sim_vertices_per_body = soft_body_view.max_sim_vertices_per_body
            # Getting Indices Associated with Sim Elements (Nodes making up the Tetrahedral Elements)
            sim_element_indices = soft_body_view.get_sim_element_indices().cpu().numpy().reshape(soft_body_view.max_sim_elements_per_body, 4) # Get the simulation mesh element indices for all soft bodies in the view
            # Getting the Cauchy's Stress Tensor for All Sim Elements
            sim_element_stress_w_cauchys = cube.data.sim_element_stress_w[deformable_object_id]

            print(f"All Default Nodal Positions   : {default_nodal_pos_w.shape}") # torch.Size([1, 72, 3]) - xyz of all vertices
            print(f"Top Surface Nodal Indices     : {top_surface_idx_torch.shape}") # torch.Size([36])    - idx of all top surface vertices
            print(f"Max Sim Elements (Tets)       : {max_sim_elements_per_body}") # 125 - number of tetrahedrals (sim elements in the body)
            print(f"Max Sim Vertices (nodes)      : {max_sim_vertices_per_body}") # 72 
            print(f"Sim Element Indices           : {sim_element_indices.shape}") # [125, 4] 
            print(f"Sim Element Stresses          : {sim_element_stress_w_cauchys.shape}") #  
     
            # -------- NEW: get top-surface sim tets (and faces) from the top vertex set --------
            top_tet_ids, faces_top, parent_top = get_top_sim_surface_tets_from_vertex_set(
                cube, top_surface_idx_torch, env_id=deformable_object_id, require_upward_normal=True
            )
            print(f"Top-surface sim tet count     : {top_tet_ids.numel()} / {max_sim_elements_per_body}") # 50 
            print(f"Top-surface face array shape  : {tuple(faces_top.shape)}  (triangles on the top plane)")
            print(f"Top-face→parent tet shape     : {tuple(parent_top.shape)}")

            raster = build_top_surface_raster(cube, top_surface_idx_torch, env_id=deformable_object_id, decimals=3)
            ny, nx = raster["grid_shape"]
            print(f"[INFO] Top-surface raster grid built: shape = (ny={ny}, nx={nx}), total={ny*nx}")
            print(f"       First row global node ids: {raster['global_order_1d'][:nx].tolist()}")


            pdb.set_trace()

        # ---------------- one physics step ----------------
        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        # refresh buffers
        cube.update(sim_dt); sphere.update(sim_dt); base.update(sim_dt)

        # ---------------- pin to current rigid top corners ----------------
        root_pos_w_rigid  = np.round(base.data.root_pos_w.detach().cpu().numpy(), 3)
        root_quat_w_rigid = np.round(base.data.root_quat_w.detach().cpu().numpy(), 3)

        # push writes
        cube.write_data_to_sim()

        # sim_element_stress_w_deformable_cauchys = cube.data.sim_element_stress_w 
        # sim_element_stress_w_deformable_von_misses = von_mises_from_cauchy(sim_element_stress_w_deformable_cauchys)                   

        # # ---------------- debug prints & plots every y steps ----------------
        # x = 0    # deformable body index
        # y = 100   # print interval
        # if count % y == 0:
        #     # quick looks
        #     print(f"[INFO]: Deformable Body Data for Body #{x}")

        #     sim_element_rest_poses = cube.root_physx_view.get_sim_element_rest_poses()
            
        #     for ii in range(125):
        #         print(f"REST POSES: {sim_element_rest_poses.cpu().numpy().reshape(cube.root_physx_view.count, cube.root_physx_view.max_sim_elements_per_body, 9)[0, ii, :]}")

        #     print(f"        Simulation Element Stress Shape (Cauchys): {sim_element_stress_w_deformable_cauchys.shape}")
        #     print(f"        Simulation Element Stress Shape (Von Misses): {sim_element_stress_w_deformable_von_misses}")

# ------------
# --- Main ---
# ------------
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
    run_simulator(sim, entities)
    
# --------------------
# Running the Main ---
# -------------------- 
if __name__ == "__main__":
    """
    run the main function
    """
    main()
    simulation_app.close()