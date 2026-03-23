"""
Usage:
    $ cd ~/IsaacLab
    $ ./isaaclab.sh -p scripts/korus/run_foam_bed_sim_pinned_with_actuator.py
    
    TODO: Imports are not successfull because it needs the full kit version. Need to make changes to the script. 

    NOTE: 
        1. Make sure you run this from a `env_isaaclab` conda environment.   
        2. If you want to run it with a ros topic then checkout "scripts/korus/run_foam_bed_sim_pinned_with_actuator_ros.py"
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
INFLATABLE_CELL_USD = "file:///home/rutwik/IsaacLab/scripts/korus/assets/inflatable_cell.usd"
# Object Placement with Scene # NOTE: The code only works when: 1. DECIMALS = 1; and FOAM_SIZE[2] and BASE_SIZE[2] are incremented by 0.1 eg. 0.1, 0.2, 0.3 and so on...   
GROUND_TO_BASE_BOTTOM = 0.9
# Sizes of Geometries
BASE_SIZE = (1.0, 1.0, .1)
FOAM_SIZE = (1.0, 1.0, .2)
SPHERE_RADIUS = 0.2
# Origin of Geometries
BASE_ORIGIN = (0.0, 0.0, GROUND_TO_BASE_BOTTOM + round(BASE_SIZE[2]/2,2))
FOAM_ORIGIN = (0.0, 0.0, BASE_ORIGIN[2] + round(BASE_SIZE[2]/2,2) + round(FOAM_SIZE[2]/2,2)) 
SPHERE_ORIGIN = (0.0, 0.0, 5.0)
# Tolerance Val
DECIMALS = 1
# Plot output dir
PLOT_DIR = "pressure_histograms"

# ------------------------ -----
# --- KeyBoard Control Class ---
# ------------------------------
class KeyControl:
    """Hold/press to move the prismatic joint:
       Up/W = inflate (+), Down/S = deflate (-), Space = stop.
    """
    # --- initialization
    def __init__(self, step: float = 0.002):
        self.step = float(step)
        self.inc  = False
        self.dec  = False
        self._sub_id = None
        self._input = None
        self._keyboard = None
    # --- omni.appwindow + carb.input subscription ---
    def subscribe(self):
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            carb.log_warn("No app window; keyboard control disabled.")
            return
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub_id = self._input.subscribe_to_keyboard_events(self._keyboard, self.on_keyboard_input)
    # --- unsubscribe ---
    def unsubscribe(self):
        try:
            if self._input and self._keyboard and self._sub_id:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_id)
        finally:
            self._sub_id = None
            self._keyboard = None
            self._input = None
    # --- event callback ---
    def on_keyboard_input(self, e):
        if e.input in (KeyboardInput.W, KeyboardInput.UP):
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.inc = True
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.inc = False
        elif e.input in (KeyboardInput.S, KeyboardInput.DOWN):
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.dec = True
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.dec = False
        elif e.input == KeyboardInput.SPACE:
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                # quick stop
                self.inc = False
                self.dec = False

# --------------------------------------------------------------
# --- Manual Scene Builder (USD + deformable + rigid sphere) ---
# --------------------------------------------------------------
def design_scene():
    """
    populate the scene here
    """
    # opening an existing scene
    scene_context = omni.usd.get_context()
    scene_context.open_stage(ENV_USD)
    # adding lights for illumination
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    dome.func("/World/DomeLight", dome)
    # Import the Inflatable Cell under /InflatableCell
    inflatable_asset = sim_utils.UsdFileCfg(usd_path=INFLATABLE_CELL_USD)
    inflatable_asset.func("/World/InflatableCell", inflatable_asset)

    # define inflatable cell object (has actuation) 
    inflatable_cell_cfg = ArticulationCfg(
        class_type=Articulation,
        prim_path="/World/InflatableCell",
        spawn=None,  # <- adopt existing prims (don't spawn again)
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0,0,0), rot=(1,0,0,0),
            joint_pos={"PrismaticJoint": 0.72}, joint_vel={".*": 0.0},
        ),
        actuators={
            "linear_pd": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                effort_limit_sim=4000.0,
                stiffness=30000.0, damping=1200.0,
            ),
        },
    )
    inflatable_cell_object = Articulation(cfg=inflatable_cell_cfg)

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
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
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

    return {"cube_object": cube_object, "sphere_object": sphere_object, "base_object": base_object, "inflatable_cell_object":inflatable_cell_object}

# -------------------------------------------------
# --- Utility Functions : For Obtaining Indices ---
# -------------------------------------------------
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

def get_surface_corner_indices_from_idx(cube, surface_idx, env_id: int = 0):
    """
    Given a set of surface node indices (w.r.t. ALL nodes), find the 4 corner node indices
    on that surface: [BL, TL, BR, TR] in global indexing.

    Args:
        cube: DeformableObject
        surface_idx: 1D torch.LongTensor of shape (K,) with global node indices on the surface NOTE: get from `get_surface_indices_by_known_z()`
        env_id: which batch/env to use (default 0)

    Returns:
        corners_idx_global: torch.LongTensor of shape (4,)
        Global node indices in order: [BL, TL, BR, TR]
    """
    # default node positions (N, 3) for this env
    default_pos = cube.data.default_nodal_state_w[env_id, :, :3]  # (N,3) -> positions of all vertices
    pts_surface = default_pos.index_select(0, surface_idx)        # (K,3) -> positions of all bottom surface vertices 
    xy = pts_surface[:, :2]                                       # (K,2) -> xy postions of all bottom surface vertices
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

# -----------------------------------------------
# Utility Functions : For Obtaining Positions --- 
# -----------------------------------------------    
def get_pin_targets_on_rigid_body(root_pos_w_rigid, root_quat_w_rigid):
    """
    This will give POSITION targets on the rigid base plate, where the the bottom surface corners will have kinematic targets. 
    NOTE: 
        1. These targets will keep updating as the cube moves
        2. Needs to be done in every iteration of the simulation
    TODO:
        1. This function is NOT currently Rotation aware, meaning change in rotation is not incorporated, which is okay 
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

# -----------------------------------------------------------------------------------
# --- Utility Functions: Top-surface extraction for collision & simulation meshes ---
# -----------------------------------------------------------------------------------
def _top_plane_z_from_default_sim_nodes(cube, env_id=0, decimals=1):
    pos0 = cube.data.default_nodal_state_w[env_id, :, :3]  # (N,3) sim nodes at rest
    z0   = pos0[:, 2]
    s    = 10 ** decimals
    zr   = torch.round(z0 * s) / s
    return zr.max(), zr, pos0  # top_z, rounded z, positions

def _build_surface_faces_from_tets(tets: torch.Tensor):
    """
    tets: (Ne, 4) long -> return boundary faces (Nb, 3) and parent tet ids (Nb,)
    """
    Ne = tets.shape[0]
    if Ne == 0:
        return tets.new_empty((0,3), dtype=torch.long), tets.new_empty((0,), dtype=torch.long)
    f0 = tets[:, [1,2,3]]
    f1 = tets[:, [0,2,3]]
    f2 = tets[:, [0,1,3]]
    f3 = tets[:, [0,1,2]]
    faces  = torch.cat([f0, f1, f2, f3], dim=0)  # (4*Ne, 3)
    parent = torch.arange(Ne, device=tets.device, dtype=torch.long).repeat_interleave(4)
    # boundary = faces that occur once (unique w.r.t sorted vertex ids)
    faces_sorted, _ = torch.sort(faces, dim=1)
    uniq, inv, counts = torch.unique(faces_sorted, dim=0, return_inverse=True, return_counts=True)
    is_boundary = counts[inv] == 1
    return faces[is_boundary], parent[is_boundary]

# ---------- top-surface (collision mesh) ----------
def get_top_surface_collision_faces(cube, env_id=0, decimals=1, require_upward_normal=False):
    """
    Uses SoftBodyView.get_element_indices() + get_nodal_positions()
    to extract boundary faces of the **collision** (element) mesh on the top Z plane.
    Returns: faces_top (K,3), parent_elem_ids (K,), top_z (float)
    """
    view = cube.root_physx_view
    elem_idx = torch.as_tensor(view.get_element_indices())[env_id].long()     # (Ne_coll, 4)
    coll_pos = torch.as_tensor(view.get_nodal_positions())[env_id]            # (Nc, 3)

    # filter padded rows if backend uses -1 padding
    valid = (elem_idx >= 0).all(dim=1)
    elem_idx = elem_idx[valid]

    faces_bnd, parent_elem = _build_surface_faces_from_tets(elem_idx)

    top_z, _, _ = _top_plane_z_from_default_sim_nodes(cube, env_id, decimals)
    s  = 10 ** decimals
    zr = torch.round(coll_pos[:, 2] * s) / s
    z3 = zr[faces_bnd]                              # (Nb,3)
    on_top = (z3 == top_z).all(dim=1)

    faces_top  = faces_bnd[on_top]
    parent_top = parent_elem[on_top]

    if require_upward_normal and faces_top.numel() > 0:
        P = coll_pos[faces_top]                     # (K,3,3)
        v1 = P[:,1,:] - P[:,0,:]
        v2 = P[:,2,:] - P[:,0,:]
        n  = torch.cross(v1, v2, dim=1)
        keep = n[:,2] > 0.0
        faces_top  = faces_top[keep]
        parent_top = parent_top[keep]

    return faces_top, parent_top, float(top_z)

# ---------- top-surface (simulation mesh) ----------
def get_top_surface_sim_faces_and_parent_tets(cube, env_id=0, decimals=1, require_upward_normal=False):
    """
    Uses SoftBodyView.get_sim_element_indices() to extract boundary faces
    of the **simulation** tet mesh on the top Z plane.
    Returns: faces_top (K,3), parent_tet_ids (K,), top_z (float)
    """
    view = cube.root_physx_view
    tets = torch.as_tensor(view.get_sim_element_indices())[env_id].long()     # (Ne_sim, 4)

    valid = (tets >= 0).all(dim=1)
    tets = tets[valid]

    faces_bnd, parent = _build_surface_faces_from_tets(tets)

    top_z, zr_default, _ = _top_plane_z_from_default_sim_nodes(cube, env_id, decimals)
    z3 = zr_default[faces_bnd]                        # (Nb,3)
    on_top = (z3 == top_z).all(dim=1)

    faces_top  = faces_bnd[on_top]
    parent_top = parent[on_top]

    if require_upward_normal and faces_top.numel() > 0:
        # you can also use view.get_sim_nodal_positions(); using cube.data.nodal_pos_w is fine too
        pos_now = cube.data.nodal_pos_w[env_id]       # (Ns,3)
        P = pos_now[faces_top]
        v1 = P[:,1,:] - P[:,0,:]
        v2 = P[:,2,:] - P[:,0,:]
        n  = torch.cross(v1, v2, dim=1)
        keep = n[:,2] > 0.0
        faces_top  = faces_top[keep]
        parent_top = parent_top[keep]

    return faces_top, parent_top, float(top_z)

# ---------- pressure from SIM element stress on top triangles ----------
# def compute_sim_top_pressures(cube, faces_top: torch.Tensor, parent_top: torch.Tensor, env_id=0):
#     """
#     For each top triangle, compute pressure p = max(0, - (sigma n̂)·n̂ ).
#     Returns:
#         p_pa  : (K,) pressures in Pascals (compressive positive)
#         area  : (K,) triangle areas (m^2) from current sim geometry
#     """
#     # current sim node positions
#     pos = cube.data.nodal_pos_w[env_id]         # (Ns,3)
#     P   = pos[faces_top]                        # (K,3,3): triangle vertices
#     v1  = P[:,1,:] - P[:,0,:]
#     v2  = P[:,2,:] - P[:,0,:]
#     n   = torch.cross(v1, v2, dim=1)            # (K,3)
#     area = 0.5 * torch.linalg.norm(n, dim=1)    # (K,)
#     n_norm = torch.linalg.norm(n, dim=1, keepdim=True) + 1e-12
#     n_hat  = n / n_norm                         # (K,3)

#     # sim element stress in world frame: (B, Ne_sim, 3, 3)
#     sigma = cube.data.sim_element_stress_w[env_id, parent_top, :, :]  # (K,3,3)

#     # traction t = sigma @ n_hat
#     t = torch.einsum('kij,kj->ki', sigma, n_hat)  # (K,3)
#     # pressure (compressive positive): p = max(0, -t·n_hat)
#     p = -(t * n_hat).sum(dim=1)
#     p = torch.clamp(p, min=0.0)
#     return p, area


def compute_sim_top_pressures(cube, faces_top, parent_top, env_id=0, clamp_tension=True):
    """
    Inputs:
      cube: DeformableObject
      faces_top: (K,3) Long tensor of sim-mesh surface faces (vertex ids)
      parent_top: (K,) Long tensor, parent tet id for each face
      env_id: batch index
      clamp_tension: if True, negatives after p=-n^T S n are clamped to 0 (ignore tension)

    Returns:
      p_face: (K,) tensor of face pressures (Pa, compression >= 0)
      centers: (K,3) face centroids in world frame
      areas: (K,) face areas
    """
    pos = cube.data.nodal_pos_w[env_id]                  # (Ns,3)
    S   = cube.data.sim_element_stress_w[env_id]         # (Ne,3,3)

    P0 = pos[faces_top[:, 0]]
    P1 = pos[faces_top[:, 1]]
    P2 = pos[faces_top[:, 2]]
    v1 = P1 - P0
    v2 = P2 - P0
    n  = torch.cross(v1, v2, dim=1)
    areas = 0.5 * torch.linalg.norm(n, dim=1)           # (K,)
    # unit normals (avoid divide-by-zero)
    n = n / (areas[:, None] * 2.0 + 1e-12)

    S_parent = S[parent_top]                             # (K,3,3)
    # p = - n^T S n (positive for compression)
    # (K,) via batch matmul
    Sn = torch.bmm(S_parent, n.unsqueeze(2)).squeeze(2)  # (K,3)
    p  = -(n * Sn).sum(dim=1)                            # (K,)

    if clamp_tension:
        p = torch.clamp(p, min=0.0)

    centers = (P0 + P1 + P2) / 3.0
    return p, centers, areas

# ----------------------------------------------
# --- Simulation loop: print nodal positions ---
# ----------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, entities, keyctl: KeyControl):
    """
    Run the simulation loop, pin cube corners, and:
      1) plot a vertical bar chart of σ_zz for ALL valid sim elements
      2) (also) compute & plot a top-surface pressure map from stresses

    Notes:
    - Saves figures into ./plots/
    - Uses sim mesh stresses (cube.data.sim_element_stress_w)
    """
    # ---------- small helper: pressure on top-surface faces from stresses ----------
    def _compute_sim_top_pressures(cube, faces_top, parent_top, env_id=0, clamp_tension=True):
        pos = cube.data.nodal_pos_w[env_id]                  # (Ns,3)
        S   = cube.data.sim_element_stress_w[env_id]         # (Ne,3,3)
        P0 = pos[faces_top[:, 0]]
        P1 = pos[faces_top[:, 1]]
        P2 = pos[faces_top[:, 2]]
        v1 = P1 - P0
        v2 = P2 - P0
        n  = torch.cross(v1, v2, dim=1)
        areas = 0.5 * torch.linalg.norm(n, dim=1)            # (K,)
        n = n / (areas[:, None] * 2.0 + 1e-12)               # unit normals

        S_parent = S[parent_top]                             # (K,3,3)
        Sn = torch.bmm(S_parent, n.unsqueeze(2)).squeeze(2)  # (K,3)
        p  = -(n * Sn).sum(dim=1)                            # (K,) Pa, compression >= 0
        if clamp_tension:
            p = torch.clamp(p, min=0.0)
        centers = (P0 + P1 + P2) / 3.0
        return p, centers, areas

    # ---------- unpack entities ----------
    cube            = entities["cube_object"]
    sphere          = entities["sphere_object"]
    base            = entities["base_object"]
    inflatable_cell = entities["inflatable_cell_object"]

    # ---------- sim params ----------
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0

    # kinematic targets buffer (B, N, 4)
    B, N, _ = cube.data.nodal_pos_w.shape
    nodal_kinematic_target = cube.data.nodal_kinematic_target.clone()

    # keyboard command state
    q_cmd = None
    q_min = None
    q_max = None

    # holders for SIM surface faces + parent tets + valid-elements mask for bar chart
    top_sim_faces = None
    top_sim_parent_tets = None
    sim_valid_mask = None

    # plot dir
    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---------- main loop ----------
    while simulation_app.is_running():
        # ---------------- reset every 1000 steps ----------------
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

            # reset inflatable articulation
            root_state_cell = inflatable_cell.data.default_root_state.clone()
            inflatable_cell.write_root_pose_to_sim(root_state_cell[:, :7])
            inflatable_cell.write_root_velocity_to_sim(root_state_cell[:, 7:])
            joint_pos = inflatable_cell.data.default_joint_pos.clone()
            joint_vel = inflatable_cell.data.default_joint_vel.clone()
            inflatable_cell.write_joint_state_to_sim(joint_pos, joint_vel)

            # --- compute SIM top-surface once (needs helpers defined above the function) ---
            top_sim_faces, top_sim_parent_tets, _ = get_top_surface_sim_faces_and_parent_tets(
                cube, env_id=0, decimals=1, require_upward_normal=True
            )

            # --- build valid sim-element mask from indices (to avoid padded rows in stresses) ---
            try:
                view = cube.root_physx_view
                tets = torch.as_tensor(view.get_sim_element_indices())[0].long()  # (Ne,4)
                sim_valid_mask = (tets >= 0).all(dim=1)
            except Exception:
                sim_valid_mask = None  # fallback: use all rows

            # reset internal state trackers
            cube.reset(); sphere.reset(); base.reset()
            print("[INFO]: Resetting deformable and rigid object states")

            # --------------------------------------------------------------------------------
            # --- pin bottom 4 corners of deformable to the rigid base using kinematic flags
            # --------------------------------------------------------------------------------
            # find top/bottom sets once from default state (helpers assumed defined above)
            top_surface_idx_torch    = get_surface_indices_by_known_z(
                cube=cube, z_target=round(FOAM_ORIGIN[2] + FOAM_SIZE[2] / 2, 2), decimals=DECIMALS
            )
            bottom_surface_idx_torch = get_surface_indices_by_known_z(
                cube=cube, z_target=round(FOAM_ORIGIN[2] - FOAM_SIZE[2] / 2, 2), decimals=DECIMALS
            )
            bottom_corner_idx_torch = get_surface_corner_indices_from_idx(
                cube, bottom_surface_idx_torch, env_id=0
            )

            # start all FREE, then pin the four bottom corners
            nodal_kinematic_target[..., :3] = cube.data.nodal_pos_w
            nodal_kinematic_target[..., 3]  = 1.0
            nodal_kinematic_target[:, bottom_corner_idx_torch, 3] = 0.0
            cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # --------------------------------------------
            # prismatic joint index + soft stroke limits
            # --------------------------------------------
            try:
                joint_names = inflatable_cell.data.joint_names
                if isinstance(joint_names, (list, tuple)) and "PrismaticJoint" in joint_names:
                    prismatic_jid = joint_names.index("PrismaticJoint")
                else:
                    prismatic_jid = 0
            except Exception:
                prismatic_jid = 0

            device = inflatable_cell.data.joint_pos.device
            B_cell, DoF = inflatable_cell.data.joint_pos.shape
            lower_limits = None
            upper_limits = None
            try:
                if hasattr(inflatable_cell.data, "joint_pos_limits"):
                    lims = inflatable_cell.data.joint_pos_limits
                    if lims.ndim == 3 and lims.shape[-1] == 2:
                        lower_limits = lims[..., 0]; upper_limits = lims[..., 1]
                elif hasattr(inflatable_cell.data, "joint_limits"):
                    lims = inflatable_cell.data.joint_limits
                    if lims.ndim == 3 and lims.shape[-1] == 2:
                        lower_limits = lims[..., 0]; upper_limits = lims[..., 1]
            except Exception:
                lower_limits = None; upper_limits = None

            if lower_limits is None or upper_limits is None:
                lower_limits = torch.full((B_cell, DoF), -0.05, device=device)
                upper_limits = torch.full((B_cell, DoF),  +0.05, device=device)

            q_now = inflatable_cell.data.joint_pos[:, prismatic_jid].clone()
            q_min = lower_limits[:, prismatic_jid]
            q_max = upper_limits[:, prismatic_jid]
            q_cmd = torch.clamp(q_now, q_min, q_max)

        # ---------------- one physics step ----------------
        sim.step(render=True)
        sim_time += sim_dt
        count    += 1

        # refresh buffers
        cube.update(sim_dt); sphere.update(sim_dt); base.update(sim_dt); inflatable_cell.update(sim_dt)

        # ---------------- pin to current rigid top corners ----------------
        root_pos_w_rigid  = np.round(base.data.root_pos_w.detach().cpu().numpy(), 3)
        root_quat_w_rigid = np.round(base.data.root_quat_w.detach().cpu().numpy(), 3)
        pin_targets_on_rigid_body_dict = get_pin_targets_on_rigid_body(root_pos_w_rigid, root_quat_w_rigid)
        targets_b = torch.tensor(
            [pin_targets_on_rigid_body_dict[f"cuboid_{b}"] for b in range(B)],
            dtype=nodal_kinematic_target.dtype, device=nodal_kinematic_target.device
        )  # (B,4,3)
        nodal_kinematic_target[:, bottom_corner_idx_torch, :3] = targets_b
        cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # ---------------- keyboard: inflate/deflate ----------------
        if q_cmd is not None:
            delta = 0.0
            if keyctl.inc: delta += keyctl.step
            if keyctl.dec: delta -= keyctl.step
            if delta != 0.0:
                q_cmd = torch.clamp(q_cmd + delta, q_min, q_max)
            q_des = inflatable_cell.data.joint_pos.clone()
            q_des[:, prismatic_jid] = q_cmd
            inflatable_cell.set_joint_position_target(q_des)

        # push writes
        cube.write_data_to_sim()
        inflatable_cell.write_data_to_sim()

        # ---------------- debug prints & plots every y steps ----------------
        x = 0    # deformable body index
        y = 10   # print interval
        if count % y == 0:
            # quick looks
            S_sim = cube.data.sim_element_stress_w[x]  # (Ne,3,3)
            print(f"[INFO]: Deformable Body Data for Body #{x}")
            print(f"        Simulation Element Stress shape: {tuple(S_sim.shape)}")

            # ====== (A) Vertical bar chart of σ_zz over ALL VALID sim elements ======
            if sim_valid_mask is not None and sim_valid_mask.any():
                S_valid = S_sim[sim_valid_mask]
            else:
                S_valid = S_sim  # fallback

            sigma_zz = S_valid[:, 2, 2].detach().cpu().numpy()  # (Ne_valid,)
            idxs = np.arange(sigma_zz.shape[0])

            fig = plt.figure(figsize=(8, 4.5), dpi=120)
            plt.bar(idxs, sigma_zz, width=1.0)  # vertical bars; sign shows tension(+) / compression(-) per σ_zz
            plt.xlabel("Simulation element index")
            plt.ylabel("σ_zz (Pa)")
            plt.title(f"σ_zz per sim element — step {count}")
            plt.tight_layout()
            out_bar = os.path.join(PLOT_DIR, f"sim_sigma_zz_bars_step{count:06d}.png")
            fig.savefig(out_bar); plt.close(fig)
            print(f"        [Bars σ_zz] saved -> {out_bar}")

            # ====== (B) Top-surface pressure map from stresses (optional, but helpful) ======
            if top_sim_faces is not None and top_sim_faces.numel() > 0:
                p_face, centers, areas = _compute_sim_top_pressures(cube, top_sim_faces, top_sim_parent_tets, env_id=0, clamp_tension=True)
                p_np = p_face.detach().cpu().numpy()
                C    = centers.detach().cpu().numpy()
                A    = areas.detach().cpu().numpy()
                print(f"        [Top-surface p] Pa stats: min={p_np.min():.3f}, mean={p_np.mean():.3f}, max={p_np.max():.3f}, count={p_np.size}")

                # pressure map scatter
                fig = plt.figure(figsize=(6, 5), dpi=120)
                sc = plt.scatter(C[:, 0], C[:, 1], c=p_np, s=5 + 2000 * A / (np.max(A) + 1e-12))
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xlabel("x (m)"); plt.ylabel("y (m)")
                plt.title(f"Top-surface pressure (Pa) — step {count}")
                cb = plt.colorbar(sc); cb.set_label("Pressure (Pa)")
                plt.tight_layout()
                out_map = os.path.join(PLOT_DIR, f"sim_top_pressure_map_step{count:06d}.png")
                fig.savefig(out_map); plt.close(fig)
                print(f"        [Top-surface p] map saved -> {out_map}")

                # histogram of top-surface pressures
                fig = plt.figure(figsize=(6, 4), dpi=120)
                plt.hist(p_np, bins=40)
                plt.xlabel("Pressure (Pa)"); plt.ylabel("Count")
                plt.title(f"Top-surface pressure histogram — step {count}")
                plt.tight_layout()
                out_hist = os.path.join(PLOT_DIR, f"sim_top_pressure_hist_step{count:06d}.png")
                fig.savefig(out_hist); plt.close(fig)
                print(f"        [Top-surface p] hist saved -> {out_hist}")
            else:
                print("        [Top-surface p] No top-surface faces found on simulation mesh.")




# ------------
# --- Main ---
# ------------
def main():
    """
        main function
    """
    # Keyboard (subscribe using omni.appwindow)
    keyctl = KeyControl(step=0.003)  # tweak step for faster/slower motion
    keyctl.subscribe()
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
    try:
        run_simulator(sim, entities, keyctl)  # pass keyctl through
    finally:
        keyctl.unsubscribe()

# --------------------
# Running the Main ---
# -------------------- 
if __name__ == "__main__":
    # run the main function
    main()
    # for smooth closing of the sim app after ctrl+c
    simulation_app.close()
