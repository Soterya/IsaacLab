"""
Usage:
    $ cd ~/IsaacLab
    $ ./isaaclab.sh -p scripts/korus/run_foam_bed_sim_all_cells.py
    
    NOTE: 
        1. Make sure you run this from a `env_isaaclab` conda environment.   
        2. TODO: If you want to run it with a ros topic then checkout "run_foam_bed_sim_all_cells_ros.py" 
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
import isaacsim.core.utils.prims as prim_utils

# ---------------------------
# GLOBAL PATHS
# ---------------------------
ENV_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
)

# ------------------------------------------
# Utility Functions for Scene Design
# ------------------------------------------

# Defining Centers for all the foams TODO: 1st Cell starts from the bottom, get them in proper order later
from isaacsim.core.utils import prims as prim_utils
def make_bed_origins(
    rows=4, cols=8,                 # bed configuration
    gap_xy=(0.0, 0.0),              # gap between foams (x,y)
    foam_xy=(0.4, 0.4),             # foam length/width (x,y)
    bed_center=(0.0, 0.0, 0.1),     # center of the whole bed (z is the *center* of tiles)
    prefix="/World/Environment/Bed"
):
    """Create /World/Environment/Bed/r{r}c{c} Xforms laid out as a centered grid."""
    foam_x, foam_y = foam_xy
    gap_x, gap_y   = gap_xy

    bed_len_x = cols * foam_x + (cols - 1) * gap_x
    bed_len_y = rows * foam_y + (rows - 1) * gap_y

    x0 = bed_center[0] - 0.5 * bed_len_x + 0.5 * foam_x
    y0 = bed_center[1] + 0.5 * bed_len_y - 0.5 * foam_y
    z  = bed_center[2]

    origins = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * (foam_x + gap_x)
            y = y0 - r * (foam_y + gap_y)   # <-- row steps downward
            path = f"{prefix}/r{r}c{c}"
            prim_utils.create_prim(path, "Xform", translation=(x, y, z))
            origins.append((round(x, 3), round(y, 3), round(z, 3)))
    return origins



# ------------------------------------------
# Scene Design Function
# ------------------------------------------
def design_scene():
    """
    Populate the scene:
      - default environment
      - 32 deformable foam tiles (wildcard prim_path)
      - rigid base slab
      - original single deformable cuboid + sphere (kept for current run_simulator)
    """
    # Importing an Environment TODO: Later make your own environment since managing the asset tree is better that way
    env_cfg = sim_utils.UsdFileCfg(usd_path=ENV_USD)
    env_cfg.func("/World", env_cfg)
    
    # --- Bed layout parameters (keep together so base/tile sizes stay in sync) ---
    rows, cols = 4, 8
    foam_xy    = (0.4, 0.4)      # tile X,Y
    foam_z     = 0.2             # tile thickness
    gap_xy     = (0.0, 0.0)
    bed_center = (0.0, 0.0, 0.15) # tile centers at z=0.1 → top surface at 0.2

    # Create 32 Xform origins (r{r}c{c})
    origins = make_bed_origins(
        rows=rows, cols=cols,
        gap_xy=gap_xy, foam_xy=foam_xy,
        bed_center=bed_center,
        prefix="/World/Environment/Bed"
    )

    # --- 32 deformable tiles (spawned under each origin Xform) ---
    bed_cfg = DeformableObjectCfg(
        prim_path="/World/Environment/Bed/r.*/Foam",   # matches r0c0/Tile, r0c1/Tile, ...
        spawn=sim_utils.MeshCuboidCfg(
            size=(foam_xy[0], foam_xy[1], foam_z),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=20,    # adjust fidelity as needed
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=0.4,
                youngs_modulus=1e5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),                            # per tile (tune)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        ),
        # Local to each origin Xform
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        debug_vis=True,
    )
    bed_tiles = DeformableObject(cfg=bed_cfg)

    # --- Rigid base under the bed (optional but useful) ---
    gap_x, gap_y = gap_xy
    bed_len_x = cols * foam_xy[0] + (cols - 1) * gap_x
    bed_len_y = rows * foam_xy[1] + (rows - 1) * gap_y
    base_thick = 0.05
    base_z = bed_center[2] - (foam_z * 0.5 + base_thick * 0.5)

    base_cfg = RigidObjectCfg(
        prim_path="/World/Environment/Bed/Base",
        spawn=sim_utils.MeshCuboidCfg(
            size=(bed_len_x, bed_len_y, base_thick),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(bed_center[0], bed_center[1], base_z)),
    )
    base = RigidObject(cfg=base_cfg)

    # rigid sphere config
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/Environment/RigidSphere",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )
    sphere_object = RigidObject(cfg=sphere_cfg)

    # Return everything; your current run_simulator uses cube_object/sphere_object,
    # but now you also have bed_tiles and base available.
    scene_entities = {
        "bed_tiles": bed_tiles,
        "base": base,
        "sphere_object": sphere_object,
    }
    return scene_entities, origins


# ----------------------
# UTILITY FUNCTIONS
# ----------------------
# ----------------------
# UTILITY FUNCTIONS (per-deformable)
# ----------------------
def get_top_idx_by_max_z_single(deform, b: int, atol=1e-4):
    """
    For one deformable instance b inside a batched DeformableObject:
      return (mask_b: (N,), idx_b: (n_top,), zmax_b: scalar tensor)
    Top nodes are those whose DEFAULT world-Z is within `atol` of the max Z
    for that instance.
    """
    # default_nodal_state_w: (B, N, 6) -> take xyz for instance b
    default_pos_b = deform.data.default_nodal_state_w[b, :, :3]  # (N, 3)
    z_b = default_pos_b[:, 2]                                    # (N,)
    zmax_b = z_b.max()                                           # scalar
    mask_b = torch.isclose(z_b, zmax_b, atol=atol)               # (N,)
    idx_b = torch.nonzero(mask_b, as_tuple=False).squeeze(1)     # (n_top,)
    return mask_b, idx_b, zmax_b


# ------------------------------------------
# Simulation loop: print nodal positions (per-deformable)
# ------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, entities):
    """
    Run loop for 32 tiles + base.
    - Resets every 400 frames
    - Computes & stores top-surface node indices *per tile* (list of tensors)
    """
    bed_tiles = entities["bed_tiles"]  # DeformableObject (with B instances internally)
    base      = entities["base"]       # RigidObject
    sphere    = entities.get("sphere_object", None)

    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # Per-tile caches
    top_idx_per = []   # list[LongTensor], len = B
    top_z_per   = []   # list[Tensor scalar], len = B

    def _reset_all():
        nonlocal sim_time, count, top_idx_per, top_z_per
        sim_time = 0.0
        count    = 0

        # Reset tiles to default
        default_nodal = bed_tiles.data.default_nodal_state_w.clone()   # (B, N, 6)
        bed_tiles.write_nodal_state_to_sim(default_nodal)

        # Reset base
        root_state = base.data.default_root_state.clone()
        base.write_root_pose_to_sim(root_state[:, :7])
        base.write_root_velocity_to_sim(root_state[:, 7:])

        # Optional sphere
        if sphere is not None:
            sroot = sphere.data.default_root_state.clone()
            sphere.write_root_pose_to_sim(sroot[:, :7])
            sphere.write_root_velocity_to_sim(sroot[:, 7:])
            sphere.reset()

        bed_tiles.reset()
        base.reset()

        # --- Per-tile top-surface indices from DEFAULT geometry (no batching) ---
        B = bed_tiles.data.root_pos_w.shape[0]
        top_idx_per = []
        top_z_per   = []
        for b in range(B):
            _, idx_b, zmax_b = get_top_idx_by_max_z_single(bed_tiles, b, atol=1e-4)
            top_idx_per.append(idx_b)
            top_z_per.append(zmax_b)

        print(f"[INFO] Reset complete: computed per-tile top indices for B={len(top_idx_per)} tiles.")
        return top_idx_per, top_z_per

    # Initial reset
    top_idx_per, top_z_per = _reset_all()

    # Main loop
    while simulation_app.is_running():
        if count != 0 and count % 400 == 0:
            top_idx_per, top_z_per = _reset_all()

        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        # Update buffers
        bed_tiles.update(sim_dt)
        base.update(sim_dt)
        if sphere is not None:
            sphere.update(sim_dt)

        # Current states
        nodal_pos_w = bed_tiles.data.nodal_pos_w.detach().cpu().numpy()  # (B, N, 3)
        nodal_vel_w = bed_tiles.data.nodal_vel_w.detach().cpu().numpy()  # (B, N, 3)
        root_pos_w  = bed_tiles.data.root_pos_w.detach().cpu().numpy()   # (B, 3)

        # Example: print info for the first tile every 10 steps
        if count % 10 == 0:
            b = 0
            zidx = int(top_idx_per[b][0].item()) if top_idx_per[b].numel() > 0 else 0
            top_ids_b = top_idx_per[b].detach().cpu().numpy()
            top_z_b   = float(top_z_per[b].detach().cpu().numpy())

            top_pos_b_z = nodal_pos_w[b, top_ids_b, 2]  # Z of top nodes
            mean_defl   = float(top_pos_b_z.mean() - top_z_b)

            print(f"[INFO] (t={sim_time:.2f}s, step={count}) Tile {b}")
            print(f"       Root pos (w): {np.round(root_pos_w[b], 3)}")
            print(f"       Node[{zidx}] pos (w): {np.round(nodal_pos_w[b, zidx, :], 3)}")
            print(f"       Node[{zidx}] vel (w): {np.round(nodal_vel_w[b, zidx, :], 3)}")
            print(f"       Top nodes: {len(top_ids_b)} | mean deflection Z: {mean_defl:.6f} m")

       

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Camera aimed at the bed (tiles centered near z≈0.1; top ≈ 0.2)
    sim.set_camera_view(eye=(3.0, 0.0, 1.0), target=(0.0, 0.0, 0.2))

    # Build scene
    entities, _origins = design_scene()

    # Start sim
    sim.reset()
    print("[INFO]: Setup complete...")

    # Run
    run_simulator(sim, entities)

if __name__ == "__main__":
    main()
    simulation_app.close()


