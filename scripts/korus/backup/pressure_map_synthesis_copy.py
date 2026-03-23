"""
NOTE: Currently in this script I am using the `simulation tetrahedrals (vertices and elements)` for all pressure and stress computation.
TODO: I can also use the `collision tetrahedrals (vertices and elements obtained from the below link)` for all the computations and those will definitely finer.   
Link -> https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.DeformableBodyView.get_collision_nodal_positions
"""

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
# Object Placement with Scene
GROUND_TO_BASE_BOTTOM = 0.0
# Sizes of Geometries
BASE_SIZE = (1.0, 1.0, .1)
FOAM_SIZE = (1.0, 1.0, .2)
SPHERE_RADIUS = 0.2
# Origin of Geometries
BASE_ORIGIN = (0.0, 0.0, GROUND_TO_BASE_BOTTOM + round(BASE_SIZE[2]/2,2))
FOAM_ORIGIN = (0.0, 0.0, BASE_ORIGIN[2] + round(BASE_SIZE[2]/2,2) + round(FOAM_SIZE[2]/2,2))
SPHERE_ORIGIN = (.3, .0, 1.0)
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
                simulation_hexahedral_resolution=10,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=0.2,
                youngs_modulus=3e4,
                dynamic_friction=1.,
                elasticity_damping=0.06,
                damping_scale=1.,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
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


# -----------------------------------------------------------------
# --- SIMPLE top-surface picking by Z (ROS-style, rounded bins) ---
# -----------------------------------------------------------------
def get_top_idx_by_known_z(cube, z_target, decimals=DECIMALS, atol=None):
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

def build_top_surface_index_grid(cube, z_target, decimals=DECIMALS, y_descending=True):
    """
    Build a stable (rows, cols) grid of node indices for the top layer.
    rows = Y bins, cols = X bins (rounded to 'decimals').
    If y_descending=True, first row is the visually 'top' row (max Y).
    """
    _, top_idx = get_top_idx_by_known_z(cube, z_target=z_target, decimals=decimals)
    default_pos = cube.data.default_nodal_state_w[..., :3][0]  # (N,3)
    top_pos = default_pos[top_idx]  # (K,3)

    s = 10**decimals
    x_round = torch.round(top_pos[:, 0] * s) / s
    y_round = torch.round(top_pos[:, 1] * s) / s

    xs = torch.sort(torch.unique(x_round)).values                      # left -> right
    ys_sorted = torch.sort(torch.unique(y_round)).values               # bottom -> top
    ys = torch.flip(ys_sorted, dims=[0]) if y_descending else ys_sorted  # top -> bottom if requested

    x_to_col = {float(v): i for i, v in enumerate(xs)}
    y_to_row = {float(v): i for i, v in enumerate(ys)}

    H, W = ys.numel(), xs.numel()
    index_grid = torch.full((H, W), -1, dtype=torch.long, device=top_idx.device)

    for n in range(top_idx.numel()):
        c = x_to_col[float(x_round[n])]
        r = y_to_row[float(y_round[n])]
        index_grid[r, c] = top_idx[n]

    assert (index_grid >= 0).all(), "Index grid has holes; bump 'decimals' if needed."
    return index_grid  # (H,W) of global node indices


# ----------------------------------------------------------------
# --- Fast: find top-surface sim tets using top vertex set (optional) ---
# ----------------------------------------------------------------
def get_top_sim_surface_tets_from_vertex_set(cube, top_vertex_idx: torch.Tensor, env_id: int = 0,
                                             require_upward_normal: bool = True):
    """
    NOTE: This is for getting the Ids of Tetrahedral Sim Elements -> this function works
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
 
# NOTE: This function is just for pressure mapping projected onto the nodes. 
# ----------------------------------------------
# --- Simulation loop: print nodal positions ---
# ----------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, entities):
    """
    Builds a stable (rows, cols) index grid for the TOP surface from the default pose.
    During the sim:
      - computes z-displacement maps (node-based)
      - computes pressure maps (node-based) by projecting Cauchy stress onto the current
        face normals of top triangles and area-averaging to nodes.
      - saves heatmaps with node indices overlaid.
    """
    # ---------- unpack entities ----------
    cube   = entities["cube_object"]
    sphere = entities["sphere_object"]
    base   = entities["base_object"]

    # ---------- sim params ----------
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # Built on reset
    index_grid = None     # (H, W) LongTensor of global node indices on the top surface
    z0_grid    = None     # (H, W) baseline Z for displacement
    H = W = 0

    # Top-surface faces/parents for pressure projection
    faces_top  = None     # (Kf, 3) triangles on top surface
    parent_top = None     # (Kf,)    parent tet id for each face
    deformable_object_id = 0

    # ---- helper: compute node pressure map from faces_top/parent_top ----
    def pressure_map_from_top_faces():
        """
        Returns (p_map, p_node) where:
          - p_map  : (H, W) node pressures (Pa) in the same raster order as index_grid
          - p_node : (N,)   node pressures for ALL nodes (zeros where no top face contributes)
        Uses p = -(n^T sigma n), n = unit face normal in CURRENT pose.
        Face pressures are area-averaged equally to the 3 nodes and then normalized by area.
        """
        # if top elements not detected, return none
        if faces_top is None or parent_top is None or faces_top.numel() == 0:
            return None, None
        # 
        device = cube.data.nodal_pos_w.device                            # cuda:0 or cpu
        pos = cube.data.nodal_pos_w[deformable_object_id]                # (Num Nodes,3)
        sigma_all = cube.data.sim_element_stress_w[deformable_object_id] # (Num Tetrahedrals, 3,3) 

        # Geometry of faces in CURRENT pose
        P  = pos[faces_top]                      # (Kf, 3, 3) -> For each top_face [1,3 node indices] find positions of those 3 indices. row 0 corresponds to index 0, and so on...  
        v1 = P[:, 1, :] - P[:, 0, :]   # computing vectors from triangle edges 
        v2 = P[:, 2, :] - P[:, 0, :]   # computing vectors from triangle edges
        n_raw = torch.cross(v1, v2, dim=1)      # (Kf,3), |n_raw| = 2*area # taking cross product of those vectors so that they give us normals pointing outwards
        A_face = 0.5 * torch.linalg.norm(n_raw, dim=1) + 1e-20 # computing area of the top surface triangles based on the normals -> this actually check out to be correct
        n_hat  = n_raw / (2.0 * A_face).unsqueeze(-1)   # unit normal # converting to unit normals

        # Ensure normals point upward (+Z): if not, flip triangle winding and recompute
        neg = n_hat[:, 2] < 0
        if neg.any():
            ft = faces_top.clone()
            ft[neg] = ft[neg][:, [0, 2, 1]]     # swap winding
            # recompute with flipped winding
            P  = pos[ft]
            v1 = P[:, 1, :] - P[:, 0, :]
            v2 = P[:, 2, :] - P[:, 0, :]
            n_raw = torch.cross(v1, v2, dim=1)
            A_face = 0.5 * torch.linalg.norm(n_raw, dim=1) + 1e-20
            n_hat  = n_raw / (2.0 * A_face).unsqueeze(-1)
            faces_used = ft
        else:
            faces_used = faces_top

        # Pull parent tet stress for each face, use symmetric part
        sigma_face = sigma_all[parent_top]                     # (Kf,3,3)
        sigma_face = 0.5 * (sigma_face + sigma_face.transpose(-1, -2))

        # Traction t = sigma * n ; pressure p = - n · t
        t = torch.bmm(sigma_face, n_hat.unsqueeze(-1)).squeeze(-1)  # (Kf,3)
        p_face = -(t * n_hat).sum(-1)                                # (Kf,), Pa (+ in compression)

        # Scatter to nodes area-weighted (equal share per corner), then normalize by accumulated area
        N = pos.shape[0]
        p_sum = torch.zeros(N, device=device)
        a_sum = torch.zeros(N, device=device)
        share = A_face / 3.0
        for corner in range(3):
            idx = faces_used[:, corner]
            p_sum.index_add_(0, idx, p_face * share)
            a_sum.index_add_(0, idx, share)
        eps = 1e-20
        p_node = p_sum / (a_sum + eps)    # (N,)

        # Pack into raster order
        order = index_grid.reshape(-1)
        p_map = p_node.index_select(0, order).reshape(H, W)
        return p_map, p_node

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

            # -------- build TOP-surface (rows, cols) index grid from DEFAULT state --------
            pos0   = cube.data.default_nodal_state_w[deformable_object_id, :, :3]  # (N,3) -> default positions of all vertices (nodes)
            device = pos0.device 
            s      = 10 ** DECIMALS

            # 1) pick TOP layer by rounded Z == top plane
            z0_rounded = torch.round(pos0[:, 2] * s) / s
            z_top = round(float(FOAM_ORIGIN[2] + FOAM_SIZE[2] / 2.0), DECIMALS)
            top_mask = (z0_rounded == z_top)
            top_idx  = torch.nonzero(top_mask, as_tuple=False).squeeze(1)  # (K,) -> Indices of all the top vertices (nodes)
            assert top_idx.numel() > 0, "[ERROR] No top-surface nodes found — check DECIMALS / geometry."

            top_pos0 = pos0.index_select(0, top_idx)  # (K,3) -> Positions of all the Top Vertices

            # 2) bin by rounded X/Y to get a rectangular grid
            x_round = torch.round(top_pos0[:, 0] * s) / s  # all x axis positions values   
            y_round = torch.round(top_pos0[:, 1] * s) / s  # all y axis positions values

            xs = torch.sort(torch.unique(x_round)).values              # left -> right
            ys = torch.sort(torch.unique(y_round), descending=True).values  # TOP row first

            x_to_col = {float(v): i for i, v in enumerate(xs)}
            y_to_row = {float(v): i for i, v in enumerate(ys)}

            H, W = ys.numel(), xs.numel()
            assert H * W == top_idx.numel(), (
                f"[ERROR] Top grid not rectangular: H*W={H*W}, K={top_idx.numel()} "
                f"(try adjusting DECIMALS={DECIMALS})"
            ) 

            index_grid = torch.full((H, W), -1, dtype=torch.long, device=device) # this block stores indices into the grid according to x,y positions
            for n in range(top_idx.numel()):
                r = y_to_row[float(y_round[n])]
                c = x_to_col[float(x_round[n])]
                index_grid[r, c] = top_idx[n]
            assert (index_grid >= 0).all(), "[ERROR] Index grid has holes; check rounding."
             

            # 3) store baseline Z for displacement
            z0_grid = pos0.index_select(0, index_grid.reshape(-1))[:, 2].reshape(H, W) # This will store the default z position values according to index_grid indices

            print(f"[INFO] Top surface grid built: rows={H}, cols={W}, total={H*W}")
            first_row_ids = index_grid[0, :].detach().cpu().tolist() # this is just for the printing
            print(f"       First row (global node ids, left->right): {first_row_ids}") 

            # 4) get top-surface faces & parents (for pressure)
            _, faces_top, parent_top = get_top_sim_surface_tets_from_vertex_set(         
                cube, top_idx, env_id=deformable_object_id, require_upward_normal=False
            ) # extracts top surface triangles # faces_top -> gives node indices triplets that make up tetrahedral `parent top`
            print(f"[INFO] Found {faces_top.shape[0]} top triangles for pressure mapping.")


        # ---------------- one physics step ----------------
        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        # refresh buffers
        cube.update(sim_dt); sphere.update(sim_dt); base.update(sim_dt)

        # push any writes (none here)
        cube.write_data_to_sim()

        # --------------- every N steps, record & plot maps ---------------
        if index_grid is not None and count % args_cli.every == 0:
            # live positions
            pos = cube.data.nodal_pos_w[deformable_object_id]  # (N,3) -> extract all current vertex (node) positions
            z_now = pos.index_select(0, index_grid.reshape(-1))[:, 2].reshape(H, W) # -> this arranges all current z positions into the shape of an index grid 
            disp_z = (z_now - z0_grid).detach().cpu().numpy()  # (H,W) in meters -> different of the current z and baseline z
            
            # quick stats
            print(f"[Step {count}] Top-surface z-displacement (m): "
                  f"min={disp_z.min():.6f}, mean={disp_z.mean():.6f}, max={disp_z.max():.6f}")

            # save numpy
            # np.save(os.path.join(MAP_DIR, f"dispZ_step{count:06d}.npy"), disp_z)

            # heatmap with node indices overlaid (displacement)
            fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
            im = ax.imshow(disp_z, origin="upper")
            plt.colorbar(im, ax=ax, label="z-displacement (m)")
            index_grid_cpu = index_grid.detach().cpu()
            for r in range(H):
                for c in range(W):
                    node_id = int(index_grid_cpu[r, c].item())
                    val = disp_z[r, c]
                    text_color = "white" if im.norm(val) > 0.5 else "black"
                    ax.text(c, r, str(node_id), ha="center", va="center",
                            fontsize=8, color=text_color)
            ax.set_title(f"Top-surface z-displacement @ step {count}")
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            ax.set_xticks(range(W)); ax.set_yticks(range(H))
            ax.set_xticklabels(range(W)); ax.set_yticklabels(range(H))
            plt.tight_layout()
            out_png = os.path.join(MAP_DIR, f"dispZ_step{count:06d}.png")
            # NOTE: Uncomment to save Nodal Displacement Map
            # fig.savefig(out_png)
            # plt.close(fig)
            # print(f"    Saved: {out_png}")

            # above was for the plotting displacement map, the below is for pressure

            # -------- pressure map (Pa, + = compression) --------
            if faces_top is not None and faces_top.numel() > 0:
                p_map, _ = pressure_map_from_top_faces()
                if p_map is not None:
                    p = p_map.detach().cpu().numpy()
                    print(f"[Step {count}] Top-surface pressure (Pa): "
                          f"min={p.min():.3f}, mean={p.mean():.3f}, max={p.max():.3f}")
                    np.save(os.path.join(MAP_DIR, f"pressure_step{count:06d}.npy"), p)

                    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
                    im = ax.imshow(p, origin="upper")
                    plt.colorbar(im, ax=ax, label="pressure (Pa, + = compression)")
                    # overlay node ids
                    for r in range(H):
                        for c in range(W):
                            node_id = int(index_grid_cpu[r, c].item())
                            val = p[r, c]
                            text_color = "white" if im.norm(val) > 0.5 else "black"
                            ax.text(c, r, str(node_id), ha="center", va="center",
                                    fontsize=8, color=text_color)
                    ax.set_title(f"Top-surface pressure @ step {count}")
                    ax.set_xlabel("col")
                    ax.set_ylabel("row")
                    ax.set_xticks(range(W)); ax.set_yticks(range(H))
                    ax.set_xticklabels(range(W)); ax.set_yticklabels(range(H))
                    plt.tight_layout()
                    out_png = os.path.join(MAP_DIR, f"pressure_step{count:06d}.png")
                    fig.savefig(out_png)
                    plt.close(fig)
                    print(f"    Saved: {out_png}")
                else:
                    print("[WARN] pressure_map_from_top_faces() returned None.")
            else:
                print("[WARN] No top faces available for pressure mapping.")

# def run_simulator(sim: sim_utils.SimulationContext, entities):
#     """
#     Builds a stable (rows, cols) index grid for the TOP surface from the default pose,
#     then during the sim:
#       1) computes z-displacement maps (node-based) and saves a heatmap with node ids
#       2) computes pressure maps (cell-based squares) from top-face Cauchy stresses
#          and saves a heatmap with cell indices.
#     """
#     # ---------- unpack entities ----------
#     cube   = entities["cube_object"]
#     sphere = entities["sphere_object"]
#     base   = entities["base_object"]

#     # ---------- sim params ----------
#     sim_dt   = sim.get_physics_dt()
#     sim_time = 0.0
#     count    = 0

#     # Built on reset
#     index_grid   = None   # (H, W) LongTensor of global node indices on the top surface
#     z0_grid      = None   # (H, W) baseline Z (default pose) for displacement
#     faces_top    = None   # (Kf,3) triangle node ids on top surface (current mesh triangulation)
#     parent_top   = None   # (Kf,)  parent tet id for each face
#     H = W = 0
#     deformable_object_id = 0

#     # ---- helper: compute face pressures & areas from current pose ----
#     def compute_top_face_pressures():
#         """
#         Returns faces_used, p_face, A_face where:
#           - faces_used : (Kf,3) triangle indices with upward normals enforced
#           - p_face     : (Kf,)  pressure on each face (Pa), positive in compression
#           - A_face     : (Kf,)  face areas
#         """
#         if faces_top is None or parent_top is None or faces_top.numel() == 0:
#             return None, None, None

#         device = cube.data.nodal_pos_w.device
#         pos = cube.data.nodal_pos_w[deformable_object_id]                # (N,3)
#         sigma_all = cube.data.sim_element_stress_w[deformable_object_id] # (Ne,3,3)

#         # Geometry of faces in CURRENT pose
#         P  = pos[faces_top]                      # (Kf, 3, 3)
#         v1 = P[:, 1, :] - P[:, 0, :]
#         v2 = P[:, 2, :] - P[:, 0, :]
#         n_raw = torch.cross(v1, v2, dim=1)      # (Kf,3), |n_raw| = 2*area
#         A_face = 0.5 * torch.linalg.norm(n_raw, dim=1) + 1e-20
#         n_hat  = n_raw / (2.0 * A_face).unsqueeze(-1)   # unit normal

#         # Ensure normals point upward (+Z): if not, flip triangle winding and recompute
#         neg = n_hat[:, 2] < 0
#         if neg.any():
#             ft = faces_top.clone()
#             ft[neg] = ft[neg][:, [0, 2, 1]]     # swap winding
#             P  = pos[ft]
#             v1 = P[:, 1, :] - P[:, 0, :]
#             v2 = P[:, 2, :] - P[:, 0, :]
#             n_raw = torch.cross(v1, v2, dim=1)
#             A_face = 0.5 * torch.linalg.norm(n_raw, dim=1) + 1e-20
#             n_hat  = n_raw / (2.0 * A_face).unsqueeze(-1)
#             faces_used = ft
#         else:
#             faces_used = faces_top

#         # Pull parent tet stress for each face & symmetrize
#         sigma_face = sigma_all[parent_top]                     # (Kf,3,3)
#         sigma_face = 0.5 * (sigma_face + sigma_face.transpose(-1, -2))

#         # Traction t = sigma * n ; pressure p = - n · t  (compression positive)
#         t = torch.bmm(sigma_face, n_hat.unsqueeze(-1)).squeeze(-1)  # (Kf,3)
#         p_face = -(t * n_hat).sum(-1)                                # (Kf,), Pa

#         return faces_used, p_face, A_face

#     # ---- helper: build a lookup from (sorted tri nodes) -> (p, A) for quick cell pairing ----
#     def build_face_lookup(faces_used, p_face, A_face):
#         """
#         Returns dict: key = tuple(sorted(i,j,k)) ; val = (p, A)
#         """
#         key = torch.sort(faces_used, dim=1).values.detach().cpu().numpy()
#         p   = p_face.detach().cpu().numpy()
#         A   = A_face.detach().cpu().numpy()
#         lut = {}
#         for i in range(key.shape[0]):
#             lut[tuple(key[i].tolist())] = (p[i], A[i])
#         return lut

#     # ---- helper: compute cell (square) pressure map from face pressures ----
#     def compute_cell_pressure_map(faces_used, p_face, A_face):
#         """
#         Builds a (H-1, W-1) grid of square pressures.
#         Each square is two top triangles sharing a diagonal; we detect which diagonal exists
#         and take the area-weighted average of the two face pressures.
#         """
#         if index_grid is None or faces_used is None:
#             return None

#         # face lookup by (sorted tri)
#         lut = build_face_lookup(faces_used, p_face, A_face)

#         cell_H, cell_W = H - 1, W - 1
#         cell_map = np.full((cell_H, cell_W), np.nan, dtype=np.float64)
#         missing = 0

#         # Walk cells row-major
#         ig_cpu = index_grid.detach().cpu().numpy()
#         for r in range(cell_H):
#             for c in range(cell_W):
#                 ul = int(ig_cpu[r,   c  ])
#                 ur = int(ig_cpu[r,   c+1])
#                 ll = int(ig_cpu[r+1, c  ])
#                 lr = int(ig_cpu[r+1, c+1])

#                 # Two possible triangle pairs for this square:
#                 # Option A (diag = UL-LR): (UL,UR,LR) and (UL,LR,LL)
#                 triA1 = tuple(sorted((ul, ur, lr)))
#                 triA2 = tuple(sorted((ul, lr, ll)))

#                 # Option B (diag = UR-LL): (UL,UR,LL) and (UR,LR,LL)
#                 triB1 = tuple(sorted((ul, ur, ll)))
#                 triB2 = tuple(sorted((ur, lr, ll)))

#                 if triA1 in lut and triA2 in lut:
#                     (p1,A1) = lut[triA1]
#                     (p2,A2) = lut[triA2]
#                 elif triB1 in lut and triB2 in lut:
#                     (p1,A1) = lut[triB1]
#                     (p2,A2) = lut[triB2]
#                 else:
#                     # no matching triangle pair for this cell (should be rare)
#                     missing += 1
#                     continue

#                 # area-weighted average
#                 cell_map[r, c] = (p1*A1 + p2*A2) / (A1 + A2 + 1e-20)

#         if missing > 0:
#             print(f"[WARN] cell_pressure_map: {missing} cells had no matching triangle pair")

#         return cell_map  # (H-1, W-1) in Pascals

#     # ---------- main loop ----------
#     while simulation_app.is_running():
#         # ------------------------------
#         # --- Reset every 1000 steps ---
#         # ------------------------------
#         if count % 1000 == 0:
#             sim_time = 0.0
#             count    = 0

#             # reset base
#             root_state_base = base.data.default_root_state.clone()
#             base.write_root_pose_to_sim(root_state_base[:, :7])
#             base.write_root_velocity_to_sim(root_state_base[:, 7:])

#             # reset deformable cube
#             nodal_state_cube = cube.data.default_nodal_state_w.clone()
#             cube.write_nodal_state_to_sim(nodal_state_cube)

#             # reset sphere
#             root_state_sphere = sphere.data.default_root_state.clone()
#             sphere.write_root_pose_to_sim(root_state_sphere[:, :7])
#             sphere.write_root_velocity_to_sim(root_state_sphere[:, 7:])

#             # reset internal state trackers
#             cube.reset(); sphere.reset(); base.reset()
#             print("[INFO]: Resetting deformable and rigid object states")

#             # -------- build TOP-surface (rows, cols) index grid from DEFAULT state --------
#             pos0 = cube.data.default_nodal_state_w[deformable_object_id, :, :3]  # (N,3) default world positions
#             device = pos0.device

#             # 1) pick TOP layer by rounded Z == top plane
#             s = 10 ** DECIMALS
#             z0_rounded = torch.round(pos0[:, 2] * s) / s
#             z_top = round(float(FOAM_ORIGIN[2] + FOAM_SIZE[2] / 2.0), DECIMALS)
#             top_mask = (z0_rounded == z_top)
#             top_idx  = torch.nonzero(top_mask, as_tuple=False).squeeze(1)  # (K,)
#             assert top_idx.numel() > 0, "[ERROR] No top-surface nodes found — check DECIMALS / geometry."

#             top_pos0 = pos0.index_select(0, top_idx)  # (K,3)

#             # 2) bin by rounded X/Y to get a rectangular grid (top row first)
#             x_round = torch.round(top_pos0[:, 0] * s) / s
#             y_round = torch.round(top_pos0[:, 1] * s) / s
#             xs = torch.sort(torch.unique(x_round)).values
#             ys = torch.sort(torch.unique(y_round), descending=True).values

#             x_to_col = {float(v): i for i, v in enumerate(xs)}
#             y_to_row = {float(v): i for i, v in enumerate(ys)}

#             H, W = ys.numel(), xs.numel()
#             assert H * W == top_idx.numel(), (
#                 f"[ERROR] Top grid not rectangular: H*W={H*W}, K={top_idx.numel()} "
#                 f"(try adjusting DECIMALS={DECIMALS})"
#             )

#             index_grid = torch.full((H, W), -1, dtype=torch.long, device=device)
#             for n in range(top_idx.numel()):
#                 r = y_to_row[float(y_round[n])]
#                 c = x_to_col[float(x_round[n])]
#                 index_grid[r, c] = top_idx[n]
#             assert (index_grid >= 0).all(), "[ERROR] Index grid has holes; check rounding."

#             # 3) store baseline Z for displacement
#             z0_grid = pos0.index_select(0, index_grid.reshape(-1))[:, 2].reshape(H, W)

#             print(f"[INFO] Top surface grid built: rows={H}, cols={W}, total={H*W}")
#             first_row_ids = index_grid[0, :].detach().cpu().tolist()
#             print(f"       First row (global node ids, left->right): {first_row_ids}")

#             # 4) find top faces & parent tets ONCE (from top vertex set)
#             top_tet_ids, faces_top, parent_top = get_top_sim_surface_tets_from_vertex_set(
#                 cube, top_idx, env_id=deformable_object_id, require_upward_normal=False
#             )
#             print(f"[INFO] Top faces: {faces_top.shape[0]} triangles from {top_tet_ids.numel()} tets.")

#         # ---------------- one physics step ----------------
#         sim.step(render=True)
#         sim_time += sim_dt
#         count    += 1

#         # refresh buffers
#         cube.update(sim_dt)
#         sphere.update(sim_dt)
#         base.update(sim_dt)

#         # push any writes (none here)
#         cube.write_data_to_sim()

#         # --------------- every N steps, record & plot maps ---------------
#         if index_grid is not None and count % args_cli.every == 0:
#             # ---- (A) node z-displacement map ----
#             pos = cube.data.nodal_pos_w[0]  # (N,3) on `device`
#             z_now = pos.index_select(0, index_grid.reshape(-1))[:, 2].reshape(H, W)
#             disp_z = (z_now - z0_grid).detach().cpu().numpy()  # (H,W) in meters

#             print(f"[Step {count}] Top z-displacement (m): "
#                   f"min={disp_z.min():.6f}, mean={disp_z.mean():.6f}, max={disp_z.max():.6f}")

#             # np.save(os.path.join(MAP_DIR, f"dispZ_step{count:06d}.npy"), disp_z) # NOTE: uncomment to save numpy files

#             fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
#             im = ax.imshow(disp_z, origin="upper")
#             plt.colorbar(im, ax=ax, label="z-displacement (m)")
#             index_grid_cpu = index_grid.detach().cpu()
#             for r in range(H):
#                 for c in range(W):
#                     node_id = int(index_grid_cpu[r, c].item())
#                     val = disp_z[r, c]
#                     text_color = "white" if im.norm(val) > 0.5 else "black"
#                     ax.text(c, r, str(node_id), ha="center", va="center",
#                             fontsize=8, color=text_color)
#             ax.set_title(f"Top-surface z-displacement @ step {count}")
#             ax.set_xlabel("col"); ax.set_ylabel("row")
#             ax.set_xticks(range(W)); ax.set_yticks(range(H))
#             ax.set_xticklabels(range(W)); ax.set_yticklabels(range(H))
#             plt.tight_layout()
#             out_png = os.path.join(MAP_DIR, f"dispZ_step{count:06d}.png")
#             # NOTE: Uncomment to save nodal displacement map
#             # fig.savefig(out_png)
#             # plt.close(fig)
#             # print(f"    Saved: {out_png}")

#             # ---- (B) cell (square) pressure map from top faces ----
#             faces_used, p_face, A_face = compute_top_face_pressures()
#             if faces_used is not None:
#                 cell_map = compute_cell_pressure_map(faces_used, p_face, A_face)  # (H-1, W-1)
#                 if cell_map is not None:
#                     # stats (ignore NaNs)
#                     finite = np.isfinite(cell_map)
#                     if finite.any():
#                         cvals = cell_map[finite]
#                         print(f"[Step {count}] Top cell pressure (Pa): "
#                               f"min={cvals.min():.2f}, mean={cvals.mean():.2f}, max={cvals.max():.2f}")
#                     else:
#                         print(f"[Step {count}] Top cell pressure: all NaN (no matching triangle pairs?)")

#                     # save & plot with cell indices overlaid (row-major)
#                     np.save(os.path.join(MAP_DIR, f"cellP_step{count:06d}.npy"), cell_map)

#                     Ch, Cw = cell_map.shape
#                     fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
#                     im = ax.imshow(cell_map, origin="upper")
#                     plt.colorbar(im, ax=ax, label="pressure (Pa)")

#                     cell_id = 0
#                     for r in range(Ch):
#                         for c in range(Cw):
#                             val = cell_map[r, c]
#                             if np.isfinite(val):
#                                 text_color = "white" if im.norm(val) > 0.5 else "black"
#                             else:
#                                 text_color = "black"
#                             ax.text(c, r, str(cell_id), ha="center", va="center",
#                                     fontsize=8, color=text_color)
#                             cell_id += 1

#                     ax.set_title(f"Top-surface cell pressure @ step {count}")
#                     ax.set_xlabel("cell col"); ax.set_ylabel("cell row")
#                     ax.set_xticks(range(Cw)); ax.set_yticks(range(Ch))
#                     ax.set_xticklabels(range(Cw)); ax.set_yticklabels(range(Ch))
#                     plt.tight_layout()
#                     out_png = os.path.join(MAP_DIR, f"cellP_step{count:06d}.png")
#                     fig.savefig(out_png)
#                     plt.close(fig)
#                     print(f"    Saved: {out_png}")
#             else:
#                 print(f"[Step {count}] No faces on top; skipped cell pressure map.")




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
