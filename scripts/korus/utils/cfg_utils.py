import yaml
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# ----------------------
# --- Configurations ---
# ----------------------
@dataclass
class MaterialCfg:
    poissons_ratio: float       = 0.2
    youngs_modulus: float       = 3.0e4
    dynamic_friction: float     = 1.0
    elasticity_damping: float   = 0.06
    damping_scale: float        = 1.0
    hexa_res: int               = 10

@dataclass
class AppCfg:
    # assets
    env_usd: str        = ""
    korusbed_usd: str   = ""
    humanoid_usd: str   = ""

    # geometry/layout
    ground_to_base_bottom: float            = 0.8
    base_size: Tuple[float, float, float]   = (1.0, 1.0, 0.1)
    foam_size: Tuple[float, float, float]   = (1.0, 1.0, 0.2)
    sphere_radius: float                    = 0.2
    base_origin_xy: Tuple[float, float]     = (-1.65, 3.85)
    sphere_origin_z: float                  = 3.0
    node_spacing: float                     = 1.1
    decimals: int                           = 1
    foam_mass: float                        = 10
    humanoid_scale: tuple                   = (3.0, 3.0, 3.0)

    # grid
    rows: int = 1
    cols: int = 1

    # material
    material: MaterialCfg = MaterialCfg()

    # derived (computed later)
    base_origin  : Tuple[float, float, float]   = None # type: ignore
    foam_origin  : Tuple[float, float, float]   = None # type: ignore
    sphere_origin: Tuple[float, float, float]   = None # type: ignore

    def compute_derivatives(self):
        bx, by = self.base_origin_xy
        bz = self.ground_to_base_bottom + round(self.base_size[2]/2, 2)
        self.base_origin = (bx, by, bz)

        fz = bz + round(self.base_size[2]/2, 2) + round(self.foam_size[2]/2, 2)
        self.foam_origin = (bx, by, fz)

        self.sphere_origin = (bx, by, self.sphere_origin_z)


def load_cfg(path: Optional[str], num_rows: int = 8, num_cols: int = 4, ) -> AppCfg:
    """
    loads environment config from the yaml file. Also does some preprocessing on the data. 
    """
    if path is None:
        raise RuntimeError("Please provide --config path to a YAML file.")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    mat_raw = raw.get("material", {}) or {}
    mat = MaterialCfg(
        poissons_ratio      = float(mat_raw.get("poissons_ratio", 0.2)),
        youngs_modulus      = float(mat_raw.get("youngs_modulus", 3.0e4)),
        dynamic_friction    = float(mat_raw.get("dynamic_friction", 1.0)),
        elasticity_damping  = float(mat_raw.get("elasticity_damping", 0.06)),
        damping_scale       = float(mat_raw.get("damping_scale", 1.0)),
        hexa_res            = int(mat_raw.get("hexa_res", 10)), 
    )
    cfg = AppCfg(
        env_usd                 = str(raw.get("env_usd", "")),
        korusbed_usd            = str(raw.get("korusbed_usd", "")),
        humanoid_usd            = str(raw.get("humanoid_usd", "")),
        ground_to_base_bottom   = float(raw.get("ground_to_base_bottom", 0.9)),
        base_size               = tuple(raw.get("base_size", [1.0, 1.0, 0.1])),  # unused now
        foam_size               = tuple(raw.get("foam_size", [1.0, 1.0, 0.2])),
        sphere_radius           = float(raw.get("sphere_radius", 0.2)),
        base_origin_xy          = tuple(raw.get("base_origin_xy", [-1.65, 3.85])),  # unused now
        sphere_origin_z         = float(raw.get("sphere_origin_z", 3.0)),
        node_spacing            = float(raw.get("node_spacing", 1.1)),
        decimals                = int(raw.get("decimals", 1)),
        rows                    = int(raw.get("rows", 1)),
        cols                    = int(raw.get("cols", 1)),
        material                = mat,
        foam_mass               = float(raw.get("foam_mass", 10)),
        humanoid_scale          = tuple(raw.get("humanoid_scale", [3.0, 3.0, 3.0]))
    )
    
    if num_rows is not None:
        cfg.rows = int(num_rows)
    if num_cols is not None:
        cfg.cols = int(num_cols)
    cfg.compute_derivatives()
    
    return cfg


# -----------------------------
# --- KORUSBED CFG TEMPLATE ---
# -----------------------------
# imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Loading the Config from Path
CFG: AppCfg = load_cfg(path= f"scripts/korus/config/korus_bed.yaml")

# define bed articulation config
KORUS_BED_CFG = ArticulationCfg(
    class_type  = Articulation,
    prim_path   = "/World/KorusBed",
    spawn       = sim_utils.UsdFileCfg(usd_path=CFG.korusbed_usd),
    init_state  = ArticulationCfg.InitialStateCfg(
        pos         = (0, 0, 0.1),
        rot         = (1, 0, 0, 0),
        joint_pos   = {f"PrismaticJoint{i}": 0.0 for i in range(CFG.rows * CFG.cols)},
        joint_vel   = {f"PrismaticJoint{i}": 0.0 for i in range(CFG.rows * CFG.cols)},
    ),
    actuators   = {
        "prismatic_pd": ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint.*"],
            effort_limit_sim    = 10_000_000.0,
            stiffness           = 6_000_000.0,
            damping             = 800.0,
        ),
    },
)

# -----------------------------
# --- HUMANOID CFG TEMPLATE ---  
# -----------------------------
# imports
from utils.stiffness_and_damping import (
    STIFFNESS_PER_JOINT,
    DAMPING_PER_JOINT,
    EFFORT_PER_JOINT,
    ALL_HUMANOID_JOINTS,
    DEFAULT_STIFFNESS, DEFAULT_EFFORT, DEFAULT_DAMPING
)

# actuator cfg helper
def make_joint_pd_cfg(joint_name: str) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr    = [joint_name],
        effort_limit_sim    = EFFORT_PER_JOINT.get(joint_name, DEFAULT_EFFORT),
        stiffness           = STIFFNESS_PER_JOINT.get(joint_name, DEFAULT_STIFFNESS),
        damping             = DAMPING_PER_JOINT.get(joint_name, DEFAULT_DAMPING),
    )
humanoid_actuators = {f"{jn}_pd": make_joint_pd_cfg(jn) for jn in ALL_HUMANOID_JOINTS}

# define humanoid cfg
KORUS_HUMANOID_CFG = ArticulationCfg(
    class_type                  = Articulation,
    prim_path                   = "/World/Humanoid",
    articulation_root_prim_path = "/Pelvis",
    spawn                       = sim_utils.UsdFileCfg(usd_path=CFG.humanoid_usd, scale=tuple(CFG.humanoid_scale)),
    init_state                  = ArticulationCfg.InitialStateCfg(
        pos         = (0, 1.4, 3.0),
        rot         = (1, 0, 0, 0),
        joint_pos   = {".*": 0.0},
        joint_vel   = {".*": 0.0},
    ),
    actuators=humanoid_actuators,  # type: ignore
)

# --------------------------------
# --- DEFORMABLE FOAM TEMPLATE ---
# --------------------------------
# imports
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg

# helper to precompute all 32 foam origins
def compute_foam_xyz(app_cfg: AppCfg) -> list[tuple[float, float, float]]:
    """
    Returns a list of (x,y,z) for each foam cell (len = rows*cols),
    ordered by idx = r*cols + c.
    """
    ox, oy, oz = map(float, app_cfg.foam_origin)
    sx = float(app_cfg.node_spacing)
    sy = float(app_cfg.node_spacing)

    out: list[tuple[float, float, float]] = []
    for r in range(app_cfg.rows):
        for c in range(app_cfg.cols):
            x = ox + c * sx
            y = oy - r * sy
            z = oz
            out.append((x, y, z))
    return out

FOAM_ORIGIN_LIST: list[tuple[float, float, float]] = compute_foam_xyz(CFG)

# define deformable foam config
KORUS_FOAM_CFG: DeformableObjectCfg = DeformableObjectCfg(
    prim_path="/World/KorusBed/Node0/DeformableCuboid0", # NOTE: placeholder... will be invoked by using replace()
    spawn=sim_utils.MeshCuboidCfg(
        size=CFG.foam_size,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0,
            contact_offset=0.001,
            simulation_hexahedral_resolution=CFG.material.hexa_res,
        ),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            poissons_ratio      = CFG.material.poissons_ratio,
            youngs_modulus      = CFG.material.youngs_modulus,
            dynamic_friction    = CFG.material.dynamic_friction,
            elasticity_damping  = CFG.material.elasticity_damping,
            damping_scale       = CFG.material.damping_scale,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=CFG.foam_mass),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
    ), 
    init_state=DeformableObjectCfg.InitialStateCfg(pos=CFG.foam_origin),
    debug_vis=False,
)