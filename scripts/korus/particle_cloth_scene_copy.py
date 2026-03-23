#!/usr/bin/env python3
import argparse
from isaaclab.app import AppLauncher

# ----------------------------
# Args + App launch
# ----------------------------
parser = argparse.ArgumentParser(description="Particle cloth corner attachments (Isaac Sim 4.5 / PhysX PBD)")
parser.add_argument("--run", action="store_true")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--dt", type=float, default=1.0 / 60.0)

# Cloth
parser.add_argument("--cloth_path", type=str, default="/World/Cloth")
parser.add_argument("--size_x", type=float, default=10.0)
parser.add_argument("--size_y", type=float, default=10.0)
parser.add_argument("--u_res", type=int, default=40)
parser.add_argument("--v_res", type=int, default=40)
parser.add_argument("--cloth_z", type=float, default=0.45)  # chosen so corners lie inside cubes (cube half height=0.5)

# Cloth springs
parser.add_argument("--stretch", type=float, default=1500.0)
parser.add_argument("--bend", type=float, default=8000.0)
parser.add_argument("--shear", type=float, default=300.0)
parser.add_argument("--damping", type=float, default=0.2)
parser.add_argument("--mass", type=float, default=0.5)

parser.add_argument("--self_collision", action="store_true")
parser.add_argument("--self_collision_filter", action="store_true")

# Particle system
parser.add_argument("--particle_system_path", type=str, default="/World/ParticleSystem")
parser.add_argument("--particle_material_path", type=str, default="/World/ParticleMaterial")
parser.add_argument("--solver_iters", type=int, default=16)
parser.add_argument("--rest_offset_factor", type=float, default=0.15)  # restOffset = factor * spacing
parser.add_argument("--contact_scale", type=float, default=1.5)

# Cubes
parser.add_argument("--cube_size", type=float, default=1.0)  # 1m cube => half-extent 0.5
parser.add_argument("--cube_z", type=float, default=0.0)
parser.add_argument("--cube_prefix", type=str, default="/World/PinCube")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------------------
# Imports after app creation
# ----------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import isaacsim.core.utils.prims as prim_utils

from omni.physx.scripts import physicsUtils, particleUtils
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, Vt, PhysxSchema


def get_stage():
    return sim_utils.get_current_stage()


def ensure_physics_scene(stage, path="/physicsScene"):
    scene = UsdPhysics.Scene.Get(stage, path)
    if not scene:
        scene = UsdPhysics.Scene.Define(stage, path)
        PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    return scene


def create_triangulated_grid_mesh(stage, path: Sdf.Path, size_x, size_y, u_patches, v_patches):
    mesh = UsdGeom.Mesh.Define(stage, path)

    u_patches = int(max(1, u_patches))
    v_patches = int(max(1, v_patches))

    pts = []
    for j in range(v_patches + 1):
        v = j / v_patches
        y = (v - 0.5) * size_y
        for i in range(u_patches + 1):
            u = i / u_patches
            x = (u - 0.5) * size_x
            pts.append(Gf.Vec3f(x, y, 0.0))

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

    mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts))
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    return mesh


def add_static_cube(stage, path: str, size: float, pos_xyz):
    # density=0 => static rigid
    prim = physicsUtils.add_rigid_box(
        stage,
        path,
        size=Gf.Vec3f(size, size, size),
        position=Gf.Vec3f(*pos_xyz),
        density=0.0,
        color=Gf.Vec3f(0.8, 0.8, 0.8),
    )
    return prim


def create_cloth_and_system(stage):
    physics_scene = ensure_physics_scene(stage)

    cloth_path = Sdf.Path(args.cloth_path)
    cloth_mesh = create_triangulated_grid_mesh(
        stage, cloth_path, args.size_x, args.size_y, args.u_res, args.v_res
    )

    # place cloth so the corner particles start inside the cubes
    physicsUtils.setup_transform_as_scale_orient_translate(cloth_mesh)
    physicsUtils.set_or_add_translate_op(cloth_mesh, Gf.Vec3f(0.0, 0.0, float(args.cloth_z)))
    physicsUtils.set_or_add_orient_op(cloth_mesh, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    physicsUtils.set_or_add_scale_op(cloth_mesh, Gf.Vec3f(1.0, 1.0, 1.0))

    # spacing from grid
    dx = float(args.size_x) / float(args.u_res)
    dy = float(args.size_y) / float(args.v_res)
    spacing = min(dx, dy)

    restOffset = float(args.rest_offset_factor) * spacing
    contactOffset = restOffset * float(args.contact_scale)

    # particle system
    ps_path = Sdf.Path(args.particle_system_path)
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=ps_path,
        contact_offset=float(contactOffset),
        rest_offset=float(restOffset),
        particle_contact_offset=float(contactOffset),
        solid_rest_offset=float(restOffset),
        fluid_rest_offset=0.0,
        solver_position_iterations=int(args.solver_iters),
        simulation_owner=physics_scene.GetPrim().GetPath(),
    )

    # particle material
    pm_path = Sdf.Path(args.particle_material_path)
    particleUtils.add_pbd_particle_material(stage, pm_path, friction=0.6, drag=0.0, lift=0.0)
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(ps_path), pm_path)

    # particle cloth
    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=cloth_path,
        dynamic_mesh_path=None,
        particle_system_path=ps_path,
        spring_stretch_stiffness=float(args.stretch),
        spring_bend_stiffness=float(args.bend),
        spring_shear_stiffness=float(args.shear),
        spring_damping=float(args.damping),
        self_collision=bool(args.self_collision),
        self_collision_filter=bool(args.self_collision_filter),
        pressure=0.0,
    )

    # mass
    mass_api = UsdPhysics.MassAPI.Apply(cloth_mesh.GetPrim())
    mass_api.GetMassAttr().Set(float(args.mass))

    # color
    cloth_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.2, 1.0)])

    print(f"[OK] Cloth: {cloth_path}")
    print(f"[OK] restOffset={restOffset:.4f} contactOffset={contactOffset:.4f} spacing={spacing:.4f}")
    return cloth_path


def make_attachment(stage, actor0_path: Sdf.Path, actor1_path: Sdf.Path, name: str):
    att_path = actor0_path.AppendElementString(name)
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, att_path)
    att.GetActor0Rel().SetTargets([actor0_path])
    att.GetActor1Rel().SetTargets([actor1_path])
    PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    return att


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=args.dt, device=args.device)
    sim = SimulationContext(sim_cfg)

    stage = get_stage()
    ensure_physics_scene(stage)

    # Create 4 cubes at cloth corners (cloth centered at origin => corners at ±size/2)
    hx = args.size_x * 0.5
    hy = args.size_y * 0.5
    corners = [
        (+hx, +hy, args.cube_z),
        (-hx, +hy, args.cube_z),
        (-hx, -hy, args.cube_z),
        (+hx, -hy, args.cube_z),
    ]

    cube_prims = []
    for i, p in enumerate(corners):
        cube_prims.append(add_static_cube(stage, f"{args.cube_prefix}_{i}", args.cube_size, p))

    cloth_path = create_cloth_and_system(stage)

    # Attach cloth to each cube (auto-attachment uses initial overlap/penetration)
    for i, cube_prim in enumerate(cube_prims):
        make_attachment(stage, cloth_path, cube_prim.GetPath(), f"attachment_{i}")

    sim.reset()
    sim.set_camera_view((12.0, 0.0, 7.0), (0.0, 0.0, 0.0))

    if args.run:
        for _ in range(int(args.steps)):
            sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
