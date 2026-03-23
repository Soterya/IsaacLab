"""
Usage:

    $ preload ./isaaclab.sh -p scripts/korus/run_foam_bed_sim_ros.py

    NOTE: 
        1.  Make sure you run this from a `env_isaaclab` conda environment.  
        2. `preload` is an alias that stands for 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' 
"""
# ------------------------------------------
# LAUNCH THE OMNIVERSE APP (with argparse)
# ------------------------------------------
import argparse
import torch
from isaaclab.app import AppLauncher
import numpy as np

parser = argparse.ArgumentParser(description="Deformable object: print nodal positions each frame.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--log_path", type=str, default="nodal_positions.txt")
parser.add_argument("--every", type=int, default=5, help="Publish every N frames")  # << throttle ROS pubs
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experience = "isaacsim.exp.full.kit"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------
# Imports after app creation
# ------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
import omni.usd

# -------------------
# ROS2
# -------------------
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension  # MultiArrayLayout not needed explicitly

# ---------------------------
# GLOBAL PATHS
# ---------------------------
ENV_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
)

# ------------------------------------------
# Scene
# ------------------------------------------
def design_scene():
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/Environment"):
        stage.RemovePrim("/Environment")

    env_cfg = sim_utils.UsdFileCfg(usd_path=ENV_USD)
    env_cfg.func("/World/MyWorld", env_cfg)

    deform_cfg = DeformableObjectCfg(
        prim_path="/World/MyWorld/Environment/DeformableCuboid",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=10,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=deform_cfg)

    sphere_cfg = RigidObjectCfg(
        prim_path="/World/MyWorld/Environment/RigidSphere",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    sphere_object = RigidObject(cfg=sphere_cfg)

    return {"cube_object": cube_object, "sphere_object": sphere_object}

# ----------------------
# UTILITIES
# ----------------------
def get_top_idx_by_known_z(cube, z_target=0.2, decimals=1, atol=None):
    """Pick nodes whose DEFAULT world-z ≈ z_target (either rounding or atol)."""
    default_pos = cube.data.default_nodal_state_w[..., :3][0]  # (N,3)
    z = default_pos[:, 2]
    if decimals is not None:
        zr = torch.round(z * (10**decimals)) / (10**decimals)
        zt = round(float(z_target), decimals)
        mask = (zr == zt)
    else:
        if atol is None:
            uniq = torch.unique(torch.round(z * 1e5) / 1e5)
            uniq = torch.sort(uniq).values
            dz = float(uniq[-1] - uniq[-2]) if uniq.numel() >= 2 else 1e-4
            atol = max(1e-5, 0.25 * dz)
        mask = torch.isclose(z, torch.tensor(z_target, device=z.device, dtype=z.dtype), atol=atol)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    return mask, idx

def build_top_surface_index_grid(cube, z_target=0.2, decimals=1):
    """
    Build a stable (rows, cols) grid of node indices for the top layer.
    rows = increasing Y, cols = increasing X (both rounded to 'decimals').
    """
    _, top_idx = get_top_idx_by_known_z(cube, z_target=z_target, decimals=decimals)
    default_pos = cube.data.default_nodal_state_w[..., :3][0]  # (N,3)
    top_pos = default_pos[top_idx]  # (K,3)

    s = 10**decimals
    x_round = torch.round(top_pos[:, 0] * s) / s
    y_round = torch.round(top_pos[:, 1] * s) / s

    xs = torch.sort(torch.unique(x_round)).values
    ys = torch.sort(torch.unique(y_round)).values

    x_to_col = {float(v): i for i, v in enumerate(xs)}
    y_to_row = {float(v): i for i, v in enumerate(ys)}

    H, W = ys.numel(), xs.numel()
    index_grid = torch.full((H, W), -1, dtype=torch.long, device=top_idx.device)

    for n in range(top_idx.numel()):
        r = y_to_row[float(y_round[n])]
        c = x_to_col[float(x_round[n])]
        index_grid[r, c] = top_idx[n]

    assert (index_grid >= 0).all(), "Index grid has holes; check rounding 'decimals'."

    return index_grid  # (H,W) of global node indices

def make_multiarray_2d(arr2d: np.ndarray) -> Float32MultiArray:
    """Pack a 2D numpy array (rows, cols) into Float32MultiArray with proper layout."""
    rows, cols = arr2d.shape
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="rows", size=int(rows), stride=int(rows * cols)),
        MultiArrayDimension(label="cols", size=int(cols), stride=int(cols)),
    ]
    msg.data = arr2d.astype(np.float32).ravel().tolist()
    return msg

def make_multiarray_xyz(xyz: np.ndarray) -> Float32MultiArray:
    """Pack (K,3) into Float32MultiArray with dims [points, channels]."""
    K = xyz.shape[0]
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="points", size=int(K), stride=int(K * 3)),
        MultiArrayDimension(label="channels", size=3, stride=3),
    ]
    msg.data = xyz.astype(np.float32).ravel().tolist()
    return msg

# ------------------------------------------
# Simulation loop: publish top-surface grids
# ------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, entities, ros_node: Node):
    cube   = entities["cube_object"]
    sphere = entities["sphere_object"]

    # Build stable top-surface grid ONCE from default state
    index_grid = build_top_surface_index_grid(cube, z_target=0.2, decimals=1)  # tweak decimals if needed
    H, W = index_grid.shape
    print(f"[INFO] Top surface grid: rows={H}, cols={W}, total={H*W} nodes")

    # ROS publishers
    pub_z   = ros_node.create_publisher(Float32MultiArray, "/foam_bed/top_surface/z_grid", 10)
    pub_xyz = ros_node.create_publisher(Float32MultiArray, "/foam_bed/top_surface/xyz", 10)

    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0 

    # Main loop
    while simulation_app.is_running():
        if count % 1000 == 0:
            sim_time = 0.0
            count    = 0
            # reset states
            nodal_state_cube = cube.data.default_nodal_state_w.clone()
            cube.write_nodal_state_to_sim(nodal_state_cube)
            root_state = sphere.data.default_root_state.clone()
            sphere.write_root_pose_to_sim(root_state[:, :7])
            sphere.write_root_velocity_to_sim(root_state[:, 7:])
            cube.reset()
            sphere.reset()
            print(f"[INFO]: Resetting deformable and rigid object states")

        sim.step(render=True)
        sim_time += sim_dt
        count += 1

        cube.update(sim_dt)
        sphere.update(sim_dt)

        # Live positions (torch, on-device)
        pos = cube.data.nodal_pos_w[0]  # (N,3)

        print(f"[DEBUG]: POS SHAPE: {pos.shape}")

        if count % args_cli.every == 0:
            # Build heightfield (rows, cols) from current Z
            z_grid = pos[index_grid.reshape(-1), 2].reshape(H, W).detach().cpu().numpy()
            pub_z.publish(make_multiarray_2d(z_grid))

            # Optional full XYZ for those nodes (K,3)
            xyz = pos[index_grid.reshape(-1)].detach().cpu().numpy()
            pub_xyz.publish(make_multiarray_xyz(xyz))

            # Keep ROS event loop alive (not strictly required for publishers)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    # ROS2 init
    rclpy.init(args=None)
    ros_node = rclpy.create_node("foam_bed_publisher")

    # Isaac
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, 0.0, 1.0), target=(0.0, 0.0, 0.5))

    entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")

    try:
        run_simulator(sim, entities, ros_node)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()
