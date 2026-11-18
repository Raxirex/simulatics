"""Parametric CAD parts for the humanoid: hollow-shell links with joint bosses, real mass properties.

Every robot body of the compiled MuJoCo model becomes one solid part, generated from the body's
primitive collision geoms, so the parts follow `HumanoidParams` (and any other robot built from
capsules, spheres, cylinders and boxes) without hard-coded dimensions:

* shell    = union(outer primitives) - union(inner primitives offset by the wall thickness)
             (capsule -> hollow capsule, sphere -> spherical shell, box -> hollow box with rounded
             edges, i.e. the cheap fillet: the convex hull of 8 corner spheres)
* bosses   = a bearing housing (cylinder along the hinge axis) at every hinge joint of the body
             (proximal) and of its children (distal), skipped when the axis runs along the host
             capsule, minus a bore for the bearing
* motors   = one actuator mass per actuated hinge, lumped at the joint on the parent body
             (stator on the proximal link), and an optional payload (battery, computer) in the root body

Geometry is built as a small CSG tree evaluated with manifold3d (guaranteed-manifold booleans,
Lalish et al.) for meshes, or with build123d / OpenCASCADE for STEP B-rep export. Mass properties
of the closed triangle meshes come from trimesh (divergence theorem, Eberly "Polyhedral mass
properties"), combined with the point masses by the parallel-axis theorem.

Outputs: STL/OBJ/GLB meshes, STEP (optional, when OCP loads), an MJCF variant that keeps the
primitive collision geoms but adds the meshes as visual geoms (group 3, no contacts) and explicit
<inertial> elements from the CAD, and a URDF (see urdf.py) for ROS and other tools.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import mujoco
import numpy as np

from ..models import HumanoidParams, build_mjcf

# ---------------------------------------------------------------------------------------------
# materials and options
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    name: str
    density: float          # kg/m^3
    E: float                # Young's modulus, Pa
    nu: float               # Poisson ratio
    yield_strength: float   # Pa


# Typical datasheet values (room temperature). FDM-printed parts are anisotropic and weaker than the
# bulk values listed here; treat stresses from them as optimistic.
MATERIALS: dict[str, Material] = {
    "pla": Material("PLA (printed)", 1240.0, 3.5e9, 0.36, 50e6),
    "petg": Material("PETG (printed)", 1270.0, 2.1e9, 0.38, 50e6),
    "pa12": Material("PA12 nylon (SLS)", 1010.0, 1.7e9, 0.40, 48e6),
    "pa12_cf": Material("PA12 carbon-filled", 1150.0, 6.0e9, 0.35, 80e6),
    "aluminium": Material("Aluminium 6061-T6", 2700.0, 68.9e9, 0.33, 276e6),
    "titanium": Material("Ti-6Al-4V", 4430.0, 113.8e9, 0.34, 880e6),
    "steel": Material("Steel S355", 7850.0, 200e9, 0.29, 355e6),
    "cfrp": Material("CFRP quasi-isotropic", 1550.0, 50e9, 0.30, 500e6),
}


@dataclass
class CadOptions:
    wall: float = 0.003            # shell wall thickness (m)
    material: str = "pla"          # key of MATERIALS
    density: float | None = None   # overrides the material density (kg/m^3)
    motor_mass: float = 0.35       # kg per actuated hinge, lumped at the joint on the parent body
    payload_mass: float = 4.0      # kg in the root body (battery, computer)
    bosses: bool = True            # bearing housings at the joints
    boss_scale: float = 0.8        # boss radius / smaller of the two link radii
    bore_scale: float = 0.55       # bore radius / boss radius
    fillet: float = 0.006          # edge radius of box-shaped parts (m)
    segments: int = 40             # facets per full circle

    @property
    def mat(self) -> Material:
        base = MATERIALS.get(self.material, MATERIALS["pla"])
        if self.density is not None and self.density > 0:
            return Material(base.name, float(self.density), base.E, base.nu, base.yield_strength)
        return base

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CadOptions":
        d = dict(d or {})
        if "wall_mm" in d:
            d["wall"] = float(d.pop("wall_mm")) / 1000.0
        for k in ("wall", "density", "motor_mass", "payload_mass", "boss_scale", "bore_scale", "fillet"):
            if d.get(k) is not None:
                d[k] = float(d[k])  # form / query values arrive as strings
        if d.get("segments") is not None:
            d["segments"] = int(float(d["segments"]))
        names = {f.name for f in fields(cls)}
        o = cls(**{k: v for k, v in d.items() if k in names})
        o.wall = float(np.clip(o.wall, 0.0003, 0.05))
        o.segments = int(np.clip(o.segments, 12, 128))
        return o


# ---------------------------------------------------------------------------------------------
# a tiny CSG tree, evaluated by manifold3d (meshes) or build123d (STEP)
# ---------------------------------------------------------------------------------------------
@dataclass
class Prim:
    kind: str                 # "cylinder" (r, h; axis z, centred) | "sphere" (r,) | "rbox" (hx, hy, hz, rc)
    dims: tuple
    T: np.ndarray             # 4x4 pose in the body frame


@dataclass
class Op:
    op: str                   # "union" | "difference" (first minus the union of the rest)
    children: list


def _pose(pos=(0, 0, 0), R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    if R is not None:
        T[:3, :3] = R
    return T


def _quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _z_to(axis) -> np.ndarray:
    """Rotation taking +z onto the unit vector `axis`."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, a), float(z @ a)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _manifold(node, segments: int):
    import manifold3d as mf
    M = mf.Manifold
    if isinstance(node, Op):
        kids = [_manifold(c, segments) for c in node.children]
        if node.op == "union":
            return M.batch_boolean(kids, mf.OpType.Add) if len(kids) > 1 else kids[0]
        rest = M.batch_boolean(kids[1:], mf.OpType.Add) if len(kids) > 2 else kids[1]
        return kids[0] - rest
    if node.kind == "cylinder":
        r, h = node.dims
        m = M.cylinder(h, r, r, circular_segments=segments, center=True)
    elif node.kind == "sphere":
        m = M.sphere(node.dims[0], circular_segments=segments)
    elif node.kind == "rbox":
        hx, hy, hz, rc = node.dims
        if rc <= 1e-6:
            m = M.cube((2 * hx, 2 * hy, 2 * hz), center=True)
        else:
            seg = max(8, segments // 2)
            m = M.batch_hull([M.sphere(rc, circular_segments=seg).translate((sx * (hx - rc), sy * (hy - rc), sz * (hz - rc)))
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    else:
        raise ValueError(node.kind)
    return m.transform(node.T[:3, :4])


def csg_contains(node, pts: np.ndarray) -> np.ndarray:
    """Exact inside test of points (n, 3) against a CSG tree (used to voxelise parts for FE checks)."""
    if isinstance(node, Op):
        masks = [csg_contains(c, pts) for c in node.children]
        if node.op == "union":
            return np.logical_or.reduce(masks)
        return masks[0] & ~np.logical_or.reduce(masks[1:])
    q = (pts - node.T[:3, 3]) @ node.T[:3, :3]  # R^T (p - t), row-wise
    if node.kind == "cylinder":
        r, h = node.dims
        return (q[:, 0] ** 2 + q[:, 1] ** 2 <= r * r) & (np.abs(q[:, 2]) <= h / 2)
    if node.kind == "sphere":
        return (q * q).sum(axis=1) <= node.dims[0] ** 2
    hx, hy, hz, rc = node.dims  # rounded box = inner box (h - rc) Minkowski-summed with a sphere rc
    d = np.maximum(np.abs(q) - (np.array([hx, hy, hz]) - rc), 0.0)
    return (d * d).sum(axis=1) <= rc * rc + 1e-15


def _brep(node, scale: float = 1000.0):
    """build123d solid of a CSG node, in millimetres."""
    import build123d as bd
    from OCP.gp import gp_Trsf
    if isinstance(node, Op):
        kids = [_brep(c, scale) for c in node.children]
        out = kids[0]
        for k in kids[1:]:
            out = out + k if node.op == "union" else out - k
        return out
    if node.kind == "cylinder":
        r, h = node.dims
        s = bd.Cylinder(r * scale, h * scale)
    elif node.kind == "sphere":
        s = bd.Sphere(node.dims[0] * scale)
    else:
        hx, hy, hz, rc = node.dims
        s = bd.Box(2 * hx * scale, 2 * hy * scale, 2 * hz * scale)
        if rc > 1e-6:
            s = bd.fillet(s.edges(), rc * scale * 0.999)
    T = node.T
    t = gp_Trsf()
    t.SetValues(*T[0, :3], T[0, 3] * scale, *T[1, :3], T[1, 3] * scale, *T[2, :3], T[2, 3] * scale)
    return s.moved(bd.Location(t))


# ---------------------------------------------------------------------------------------------
# mass properties
# ---------------------------------------------------------------------------------------------
@dataclass
class MassProps:
    mass: float
    com: np.ndarray            # (3,) in the body frame
    inertia: np.ndarray        # (3, 3) about the COM, body axes

    @staticmethod
    def combine(items: list["MassProps"]) -> "MassProps":
        items = [i for i in items if i.mass > 0]
        m = sum(i.mass for i in items)
        if m <= 0:
            return MassProps(0.0, np.zeros(3), np.zeros((3, 3)))
        c = sum(i.mass * i.com for i in items) / m
        I = np.zeros((3, 3))
        for i in items:
            d = i.com - c
            I += i.inertia + i.mass * ((d @ d) * np.eye(3) - np.outer(d, d))
        return MassProps(m, c, (I + I.T) / 2)

    @staticmethod
    def sphere(mass: float, pos, radius: float) -> "MassProps":
        return MassProps(mass, np.asarray(pos, float), 0.4 * mass * radius ** 2 * np.eye(3))

    @staticmethod
    def of_mesh(mesh, density: float) -> "MassProps":
        mesh.density = density
        return MassProps(float(mesh.mass), np.array(mesh.center_mass), np.array(mesh.moment_inertia))

    def fullinertia(self) -> tuple[float, ...]:
        I = self.inertia
        return (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])


def tube_mesh(outer_r: float, inner_r: float, length: float, segments: int = 96):
    """Open-ended tube along z (z in [0, length]) as a trimesh: the analytic-mass test case."""
    import manifold3d as mf
    import trimesh
    M = mf.Manifold
    m = M.cylinder(length, outer_r, outer_r, circular_segments=segments)
    if inner_r > 0:
        m = m - M.cylinder(length * 1.2, inner_r, inner_r, circular_segments=segments).translate((0, 0, -0.1 * length))
    return _to_trimesh(m)


def _to_trimesh(m):
    import trimesh
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3].astype(np.float64),
                           np.asarray(mesh.tri_verts).astype(np.int64), process=False)


# ---------------------------------------------------------------------------------------------
# parts from the compiled model
# ---------------------------------------------------------------------------------------------
_CAPSULE, _SPHERE, _BOX, _CYL = (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                                 mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER)


@dataclass
class CadPart:
    name: str                  # body name
    mesh: object               # trimesh.Trimesh in the body frame (m)
    csg: object                # CSG tree (for STEP export)
    shell: MassProps           # the printed / machined part alone
    extras: list[tuple[str, MassProps]] = field(default_factory=list)  # motors, payload
    props: MassProps | None = None

    @property
    def mass(self) -> float:
        return self.props.mass

    def summary(self) -> dict:
        return {"name": self.name, "mass_kg": self.props.mass, "shell_kg": self.shell.mass,
                "extras_kg": sum(m.mass for _, m in self.extras), "volume_cm3": self.mesh.volume * 1e6,
                "com": self.props.com.tolist(), "inertia_diag": np.diag(self.props.inertia).tolist()}


def robot_bodies(model: mujoco.MjModel) -> list[int]:
    """Body ids of the robot (the tree under the free-floating root, or under body 1 without one)."""
    root = 1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            root = int(model.jnt_bodyid[j])
            break
    return [b for b in range(1, model.nbody) if model.body_rootid[b] == model.body_rootid[root]]


def _geoms(model, b: int) -> list[int]:
    out = []
    for g in np.flatnonzero(model.geom_bodyid == b):
        if model.geom_type[g] not in (_CAPSULE, _SPHERE, _BOX, _CYL):
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual-only
        out.append(int(g))
    return out


def _geom_radius(model, g: int) -> float:
    s = model.geom_size[g]
    if model.geom_type[g] == _BOX:
        return float(min(s))
    return float(s[0])


def _body_radius(model, b: int) -> float:
    gs = _geoms(model, b)
    return max((_geom_radius(model, g) for g in gs), default=0.03)


def _geom_frame(model, g: int) -> np.ndarray:
    return _pose(model.geom_pos[g], _quat_to_mat(model.geom_quat[g]))


def _geom_distance(model, g: int, p: np.ndarray) -> float:
    """Signed distance from point p (body frame) to the surface of geom g."""
    T = _geom_frame(model, g)
    q = T[:3, :3].T @ (p - T[:3, 3])
    s = model.geom_size[g]
    t = model.geom_type[g]
    if t == _SPHERE:
        return float(np.linalg.norm(q) - s[0])
    if t == _CAPSULE:
        z = np.clip(q[2], -s[1], s[1])
        return float(np.linalg.norm(q - [0, 0, z]) - s[0])
    if t == _CYL:
        return float(max(np.hypot(q[0], q[1]) - s[0], abs(q[2]) - s[1]))
    d = np.abs(q) - s[:3]
    return float(np.linalg.norm(np.maximum(d, 0)) + min(d.max(), 0))


def _shell_prims(model, g: int, wall: float, fillet: float) -> tuple[list[Prim], list[Prim]]:
    T = _geom_frame(model, g)
    s = model.geom_size[g]
    t = model.geom_type[g]

    def at(dz):
        return T @ _pose((0, 0, dz))

    outer, inner = [], []
    if t == _CAPSULE:
        r, hl = float(s[0]), float(s[1])
        outer = [Prim("cylinder", (r, 2 * hl), T), Prim("sphere", (r,), at(hl)), Prim("sphere", (r,), at(-hl))]
        ri = r - wall
        if ri > 0.25 * r:
            inner = [Prim("cylinder", (ri, 2 * hl), T), Prim("sphere", (ri,), at(hl)), Prim("sphere", (ri,), at(-hl))]
    elif t == _SPHERE:
        r = float(s[0])
        outer = [Prim("sphere", (r,), T)]
        if r - wall > 0.25 * r:
            inner = [Prim("sphere", (r - wall,), T)]
    elif t == _CYL:
        r, hl = float(s[0]), float(s[1])
        outer = [Prim("cylinder", (r, 2 * hl), T)]
        if r - wall > 0.25 * r and hl - wall > 0.25 * hl:
            inner = [Prim("cylinder", (r - wall, 2 * (hl - wall)), T)]
    else:
        h = np.asarray(s[:3], float)
        rc = float(min(fillet, 0.5 * h.min()))
        outer = [Prim("rbox", (*h, rc), T)]
        hi = h - wall
        if (hi > 0.25 * h).all():
            inner = [Prim("rbox", (*hi, max(0.0, rc - wall)), T)]
    return outer, inner


def _boss(model, host: int, anchor: np.ndarray, axis: np.ndarray, r_boss: float, opts: CadOptions):
    """Bearing housing along `axis` at `anchor`, spanning the host body's nearest geom. None if not useful."""
    gs = _geoms(model, host)
    if not gs:
        return None
    g = min(gs, key=lambda i: _geom_distance(model, i, anchor))
    Tg = _geom_frame(model, g)
    axis = axis / np.linalg.norm(axis)
    t = model.geom_type[g]
    s = model.geom_size[g]
    if t in (_CAPSULE, _CYL) and abs(axis @ Tg[:3, 2]) > 0.9:
        return None  # axis along the tube: the tube itself is the housing
    if t == _BOX:
        half = float(np.abs(Tg[:3, :3].T @ axis) @ s[:3])
    else:
        half = float(s[0])
    if _geom_distance(model, g, anchor) > max(r_boss, half):
        return None  # anchor too far from this body's material
    R = _z_to(axis)
    T = _pose(anchor, R)
    boss = Prim("cylinder", (r_boss, 2 * half), T)
    bore = Prim("cylinder", (r_boss * opts.bore_scale, 2 * half + 4 * opts.wall + 1e-3), T)
    return boss, bore


def _hinges(model, b: int) -> list[int]:
    return [int(j) for j in np.flatnonzero(model.jnt_bodyid == b) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def body_csg(model: mujoco.MjModel, b: int, opts: CadOptions):
    """CSG tree of body b's part in its body frame, or None if the body has no primitive geoms."""
    gs = _geoms(model, b)
    if not gs:
        return None
    outer, inner = [], []
    for g in gs:
        o, i = _shell_prims(model, g, opts.wall, opts.fillet)
        outer += o
        inner += i
    shell = Op("union", outer) if not inner else Op("difference", [Op("union", outer), Op("union", inner)])
    if not opts.bosses:
        return shell
    bosses, bores, seen = [], [], []
    r_self = _body_radius(model, b)
    jobs = []  # (anchor, axis, other body)
    parent = int(model.body_parentid[b])
    for j in _hinges(model, b):
        jobs.append((np.array(model.jnt_pos[j], float), np.array(model.jnt_axis[j], float), parent))
    for c in np.flatnonzero(model.body_parentid == b):
        c = int(c)
        if c == b:
            continue
        Rc = _quat_to_mat(model.body_quat[c])
        for j in _hinges(model, c):
            jobs.append((model.body_pos[c] + Rc @ model.jnt_pos[j], Rc @ model.jnt_axis[j], c))
    for anchor, axis, other in jobs:
        key = (np.round(anchor, 5), np.round(np.abs(axis), 4))
        if any(np.array_equal(key[0], k[0]) and np.array_equal(key[1], k[1]) for k in seen):
            continue
        seen.append(key)
        r_other = _body_radius(model, other) if other > 0 else r_self
        made = _boss(model, b, anchor, axis, opts.boss_scale * min(r_self, r_other), opts)
        if made:
            bosses.append(made[0])
            bores.append(made[1])
    if not bosses:
        return shell
    return Op("difference", [Op("union", [shell, *bosses]), Op("union", bores)])


def _extras(model: mujoco.MjModel, bodies: list[int], opts: CadOptions) -> dict[int, list[tuple[str, MassProps]]]:
    """Motor and payload point masses per body (motors on the parent of the joint's body)."""
    out: dict[int, list] = {b: [] for b in bodies}
    actuated = {int(model.actuator_trnid[a, 0]) for a in range(model.nu)
                if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
    if opts.motor_mass > 0:
        for j in sorted(actuated):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            c = int(model.jnt_bodyid[j])
            p = int(model.body_parentid[c])
            if p not in out:
                p = c  # parent outside the robot: keep the motor on the body itself
                pos = model.jnt_pos[j]
            else:
                pos = model.body_pos[c] + _quat_to_mat(model.body_quat[c]) @ model.jnt_pos[j]
            r = 0.6 * min(_body_radius(model, c), _body_radius(model, p))
            out[p].append((f"motor:{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)}",
                           MassProps.sphere(opts.motor_mass, pos, r)))
    if opts.payload_mass > 0 and bodies:
        root = bodies[0]
        out[root].append(("payload", MassProps.sphere(opts.payload_mass, np.zeros(3), 0.5 * _body_radius(model, root))))
    return out


@dataclass
class RobotCad:
    parts: dict[str, CadPart]
    opts: CadOptions
    primitive_mass: float          # total robot mass of the primitive (density-based) model
    xml: str                       # MJCF the parts were generated from

    @property
    def total_mass(self) -> float:
        return float(sum(p.mass for p in self.parts.values()))

    def summary(self) -> dict:
        mat = self.opts.mat
        return {"total_mass_kg": self.total_mass, "primitive_mass_kg": self.primitive_mass,
                "shell_mass_kg": float(sum(p.shell.mass for p in self.parts.values())),
                "material": mat.name, "density": mat.density, "wall_mm": self.opts.wall * 1000,
                "parts": [p.summary() for p in self.parts.values()]}

    # ---- exports -----------------------------------------------------------------------------
    def export(self, folder: str | Path, formats=("stl",)) -> list[Path]:
        """Write one mesh file per part (<body>.<fmt>, metres) and, for "glb", the posed assembly robot.glb."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            if fmt == "glb":
                out.append(self.export_assembly(folder / "robot.glb"))
                continue
            if fmt in ("step", "stp"):
                out += self.export_step(folder)
                continue
            for name, part in self.parts.items():
                path = folder / f"{name}.{fmt}"
                part.mesh.export(path)
                out.append(path)
        return out

    def export_assembly(self, path: str | Path) -> Path:
        """All parts placed at the model's default pose, as one GLB (or any trimesh scene format)."""
        import trimesh
        model = mujoco.MjModel.from_xml_string(self.xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = trimesh.Scene()
        for name, part in self.parts.items():
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            T = _pose(data.xpos[b], data.xmat[b].reshape(3, 3))
            scene.add_geometry(part.mesh.copy(), node_name=name, geom_name=name, transform=T)
        path = Path(path)
        scene.export(path)
        return path

    def export_step(self, folder: str | Path) -> list[Path]:
        """One STEP file per part (millimetres) through build123d/OCP. Returns [] when OCP is unavailable."""
        try:
            import build123d as bd
        except Exception:  # noqa: BLE001 - optional dependency (OCP wheels can fail to load)
            return []
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for name, part in self.parts.items():
            try:
                solid = _brep(part.csg)
                path = folder / f"{name}.step"
                if bd.export_step(solid, str(path), unit=bd.Unit.MM):
                    out.append(path)
            except Exception:  # noqa: BLE001 - keep the other parts
                continue
        return out

    def mjcf(self, xml: str | None = None, mesh_dir: str | Path | None = None,
             rgba=(0.62, 0.72, 0.86, 1.0)) -> str:
        """MJCF with the CAD meshes as visual geoms and CAD masses as body inertials.

        xml: the model to post-process (default: the one the parts were built from; pass the session's
        XML with obstacles). Primitive geoms stay as the collision model. With mesh_dir the meshes are
        referenced as <mesh file=...> (written there as STL if missing); without it the vertex/face
        data is inlined so the string compiles on its own with MjModel.from_xml_string.
        """
        return mjcf_with_cad(xml or self.xml, self, mesh_dir=mesh_dir, rgba=rgba)


def build_robot_cad(params: HumanoidParams | None = None, opts: CadOptions | None = None,
                    xml: str | None = None) -> RobotCad:
    """Generate CAD parts for every robot body of `xml` (default: build_mjcf(params))."""
    opts = opts or CadOptions()
    if xml is None:
        xml = build_mjcf(params or HumanoidParams())
    model = mujoco.MjModel.from_xml_string(xml)
    bodies = robot_bodies(model)
    extras = _extras(model, bodies, opts)
    rho = opts.mat.density
    parts: dict[str, CadPart] = {}
    for b in bodies:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        csg = body_csg(model, b, opts)
        if csg is None:
            continue
        mesh = _to_trimesh(_manifold(csg, opts.segments))
        shell = MassProps.of_mesh(mesh, rho)
        part = CadPart(name, mesh, csg, shell, extras.get(b, []))
        part.props = MassProps.combine([shell] + [m for _, m in part.extras])
        parts[name] = part
    prim_mass = float(sum(model.body_mass[b] for b in bodies))
    return RobotCad(parts, opts, prim_mass, xml)


# ---------------------------------------------------------------------------------------------
# MJCF post-processing
# ---------------------------------------------------------------------------------------------
def _fmt(v) -> str:
    return " ".join(f"{float(x):.7g}" for x in np.ravel(v))


def mjcf_with_cad(xml: str, cad: RobotCad, mesh_dir: str | Path | None = None, rgba=(0.62, 0.72, 0.86, 1.0),
                  replace_meshes: dict[str, object] | None = None) -> str:
    """Return `xml` with CAD visual meshes (group 3, contype=conaffinity=0) and CAD <inertial>s added.

    replace_meshes: {body name: trimesh} to use another mesh (e.g. a topology-optimised link) as that
    body's visual; its mass properties must already be in cad.parts[name].props.
    """
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(list(root).index(compiler) + 1, asset)
    if mesh_dir is not None:
        mesh_dir = Path(mesh_dir).resolve()
        mesh_dir.mkdir(parents=True, exist_ok=True)
        compiler.set("meshdir", str(mesh_dir).replace("\\", "/"))
    meshes = dict(replace_meshes or {})
    for body in root.iter("body"):
        name = body.get("name")
        part = cad.parts.get(name)
        if part is None:
            continue
        mesh = meshes.get(name, part.mesh)
        mname = f"cad_{name}"
        for old in body.findall("inertial"):
            body.remove(old)
        p = part.props
        body.insert(0, ET.Element("inertial", {"pos": _fmt(p.com), "mass": f"{p.mass:.7g}",
                                               "fullinertia": _fmt(p.fullinertia())}))
        if mesh_dir is not None:
            fname = f"{name}.stl"
            mesh.export(mesh_dir / fname)
            asset.append(ET.Element("mesh", {"name": mname, "file": fname}))
        else:
            asset.append(ET.Element("mesh", {"name": mname, "vertex": _fmt(np.asarray(mesh.vertices)),
                                             "face": " ".join(str(int(i)) for i in np.ravel(mesh.faces))}))
        # visual geom after the body's own geoms (before child bodies)
        kids = list(body)
        idx = max((i for i, k in enumerate(kids) if k.tag in ("geom", "inertial", "joint", "freejoint", "site")),
                  default=0) + 1
        body.insert(idx, ET.Element("geom", {"name": mname, "type": "mesh", "mesh": mname, "group": "3",
                                             "contype": "0", "conaffinity": "0", "density": "0",
                                             "rgba": _fmt(rgba)}))
    return ET.tostring(root, encoding="unicode")


def cad_model_xml(params: HumanoidParams | None = None, opts: CadOptions | None = None,
                  xml: str | None = None, mesh_dir: str | Path | None = None) -> tuple[str, RobotCad]:
    """Convenience: build the parts and return (MJCF with CAD meshes and inertials, RobotCad)."""
    cad = build_robot_cad(params, opts, xml)
    return cad.mjcf(xml, mesh_dir=mesh_dir), cad


def to_gui_mesh(mesh) -> dict:
    """{"vert": base64 float32 xyz, "face": base64 uint32} as the GUI's meshGeometry() expects."""
    import base64
    v = np.ascontiguousarray(np.asarray(mesh.vertices, np.float32))
    f = np.ascontiguousarray(np.asarray(mesh.faces, np.uint32))
    return {"vert": base64.b64encode(v.tobytes()).decode(), "face": base64.b64encode(f.tobytes()).decode()}


def export_urdf(params: HumanoidParams | None = None, opts: CadOptions | None = None, **kw) -> str:
    """URDF of the robot (see urdf.export_urdf); CAD inertials are used when opts is given."""
    from .urdf import export_urdf as _export
    return _export(params, opts, **kw)
