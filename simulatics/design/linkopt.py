"""Topology-optimise one humanoid limb against the loads it sees in simulation.

Pipeline (one pass of the load-then-optimise fixed-point loop):

1. loads.extract_link_loads: simulate walking (or standing with pushes), keep the extreme joint
   reaction wrenches of the link (equivalent static loads, x dynamic factor).
2. Design domain: the link's bounding box from the proximal to the distal joint (length L x 2r x 2r,
   r = capsule radius, read from the compiled model; the cross-section is rounded up to an even voxel
   count). By default only voxels whose centre lies inside the capsule radius may hold material, so the
   part stays inside the capsule collision geometry to within half a voxel; one voxel layer at each end
   is passive solid (the joint flanges).
3. Boundary conditions: the proximal face is clamped; each load case is applied on the distal face as
   the tip wrench that produces the recorded proximal reaction (F_tip = -F_p, M_tip = -M_p + d x F_p
   about the distal joint d), spread over the face nodes RBE3-style (exact resultant). The link's own
   weight and inertia are thereby lumped at the tip: an engineering approximation.
4. topopt.run_topopt (SIMP, density filter, OC) minimises the sum of compliances at the requested
   volume fraction of the envelope (or, with match_baseline, at the CAD tube's own fraction).
5. density_to_mesh -> closed surface in the link frame -> mass, COM and inertia with the CAD material.

For comparison the baseline CAD part (cad.py hollow tube with bosses) is voxelised on the same grid,
with stiffness proportional to the occupied fraction of each voxel (a Voigt upper bound, which
flatters the baseline), and its compliance is evaluated under the same loads.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, fields

import mujoco
import numpy as np

from ..models import HumanoidParams, build_mjcf
from . import cad as cadmod
from .loads import LINKS, LoadCase, LoadConfig, extract_link_loads
from .topopt import (TopOptConfig, TopOptProblem, TopOptResult, density_to_mesh, evaluate_compliance,
                     run_topopt, wrench_to_nodal_forces)


@dataclass
class LinkOptConfig:
    nel: int = 40                  # voxels along the link
    volfrac: float = 0.3           # material fraction of the (cylindrical) envelope
    match_baseline: bool = False   # use the CAD tube's own fraction of the envelope instead (equal material)
    iterations: int = 40
    rmin: float = 1.5              # filter radius (voxels)
    cylinder: bool = True          # only the inscribed cylinder may hold material
    end_layers: int = 1            # passive solid voxel layers at each end
    threads: int = 4
    backend: str = "auto"          # topopt backend: "auto" | "native" | "numpy"
    cg_tol: float = 1e-5
    smooth_iterations: int = 6
    baseline: bool = True          # also evaluate the voxelised CAD part under the same loads
    cad: cadmod.CadOptions = field(default_factory=cadmod.CadOptions)
    loads: LoadConfig = field(default_factory=LoadConfig)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LinkOptConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in d.items() if k in names and k not in ("cad", "loads")})
        cad_d = d.get("cad") if isinstance(d.get("cad"), dict) else {k: d[k] for k in ("wall_mm", "wall", "density",
                                                                                       "material") if k in d}
        cfg.cad = d["cad"] if isinstance(d.get("cad"), cadmod.CadOptions) else cadmod.CadOptions.from_dict(cad_d)
        if isinstance(d.get("loads"), LoadConfig):
            cfg.loads = d["loads"]
        elif isinstance(d.get("loads"), dict):
            cfg.loads = LoadConfig(**{k: v for k, v in d["loads"].items() if k in LoadConfig.__dataclass_fields__})
        cfg.nel = int(np.clip(cfg.nel, 8, 200))
        cfg.volfrac = float(np.clip(cfg.volfrac, 0.05, 0.95))
        cfg.iterations = int(np.clip(cfg.iterations, 1, 1000))
        return cfg


@dataclass
class LinkGeometry:
    body: str
    length: float          # proximal joint -> distal joint (m)
    radius: float          # capsule radius (m)
    axis: np.ndarray       # unit vector proximal -> distal, link frame
    u: np.ndarray          # first cross-section axis (grid y), link frame
    w: np.ndarray          # second cross-section axis (grid z) = axis x u

    def grid_to_link(self, width: float) -> np.ndarray:
        """4x4 transform from grid coordinates (x along the link, y/z in [0, width]) to the link frame."""
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = self.axis, self.u, self.w
        T[:3, 3] = -(width / 2) * (self.u + self.w)
        return T


def link_geometry(model: mujoco.MjModel, link: str, side: str = "left") -> LinkGeometry:
    """Length, radius and axes of `<side>_<link>` from the compiled model (never hard-coded)."""
    body = f"{side}_{link}"
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if b < 0:
        raise ValueError(f"model has no body {body!r}")
    geoms = [g for g in np.flatnonzero(model.geom_bodyid == b)
             if model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER)]
    if not geoms:
        raise ValueError(f"{body} has no capsule/cylinder geom to optimise")
    g = max(geoms, key=lambda i: model.geom_size[i, 1])
    radius = float(model.geom_size[g, 0])
    children = [c for c in np.flatnonzero(model.body_parentid == b) if c != b]
    if children:
        d = np.array(model.body_pos[children[0]], float)
    else:  # terminal link: far end of the capsule's segment
        R = cadmod._quat_to_mat(model.geom_quat[g])
        ends = [model.geom_pos[g] + R @ [0, 0, s * model.geom_size[g, 1]] for s in (-1, 1)]
        d = max(ends, key=np.linalg.norm)
    length = float(np.linalg.norm(d))
    axis = d / length
    y = np.array([0.0, 1.0, 0.0])
    u = y - (y @ axis) * axis
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0]) - axis[0] * axis
    u /= np.linalg.norm(u)
    return LinkGeometry(body, length, radius, axis, u, np.cross(axis, u))


@dataclass
class LinkOptResult:
    link: str
    geometry: LinkGeometry
    mesh: object                       # trimesh.Trimesh, link frame, metres
    props: cadmod.MassProps            # optimised part (material only)
    mass_before: float                 # whole CAD shell part of the link (cad.py, incl. end caps and bosses
                                       # outside the optimisation domain), same material
    mass_after: float
    compliance: float                  # J, SIMP objective (sum over load cases, dynamic factor included)
    compliance_thresholded: float | None  # design thresholded at 0.5, E linear in occupancy
    compliance_baseline: float | None  # voxelised CAD part, E linear in occupancy, same grid and loads
    baseline_volfrac: float | None
    max_displacement_mm: float
    load_cases: list[LoadCase]
    load_source: str
    topopt: TopOptResult
    domain: dict
    seconds: float

    @property
    def mass_saving_pct(self) -> float:
        return 100.0 * (1.0 - self.mass_after / self.mass_before) if self.mass_before > 0 else 0.0

    def summary(self) -> dict:
        return {
            "link": self.link, "mass_before_kg": self.mass_before, "mass_after_kg": self.mass_after,
            "mass_saving_pct": self.mass_saving_pct, "compliance_J": self.compliance,
            "compliance_thresholded_J": self.compliance_thresholded,
            "compliance_baseline_J": self.compliance_baseline,
            "stiffness_vs_baseline": (self.compliance_baseline / self.compliance_thresholded
                                      if self.compliance_baseline and self.compliance_thresholded else None),
            "baseline_volfrac": self.baseline_volfrac, "max_displacement_mm": self.max_displacement_mm,
            "com": self.props.com.tolist(), "inertia_diag": np.diag(self.props.inertia).tolist(),
            "load_source": self.load_source, "load_cases": [c.to_dict() for c in self.load_cases],
            "domain": self.domain, "topopt": self.topopt.summary(), "seconds": self.seconds,
        }

    def mirrored_mesh(self):
        """The right-side part: the left mesh reflected y -> -y (faces re-oriented)."""
        m = self.mesh.copy()
        m.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
        if m.volume < 0:
            m.invert()
        return m

    def apply(self, robot: cadmod.RobotCad) -> cadmod.RobotCad:
        """Replace the left/right part of this link in `robot` (meshes and mass properties) in place."""
        rho = robot.opts.mat.density
        for side, mesh in (("left", self.mesh), ("right", self.mirrored_mesh())):
            name = f"{side}_{self.link}"
            part = robot.parts.get(name)
            if part is None:
                continue
            part.mesh = mesh
            part.shell = cadmod.MassProps.of_mesh(mesh, rho)
            part.props = cadmod.MassProps.combine([part.shell] + [m for _, m in part.extras])
            part.csg = None  # no B-rep for a marching-cubes part
        return robot


def _occupancy(csg, prob: TopOptProblem, T: np.ndarray, sub: int = 3) -> np.ndarray:
    """Fraction of every voxel inside the CSG part (sub^3 samples per voxel), grid -> link transform T.

    Pick sub so the sample spacing h/sub is below the thinnest wall, or thin shells alias (with 3
    samples per 10 mm voxel a 3 mm tube came out 14 % light to 23 % heavy).
    """
    h = prob.voxel
    off = (np.arange(sub) + 0.5) / sub * h
    so = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    corners = prob.element_centres().reshape(-1, 3) - h / 2
    out = np.empty(len(corners))
    chunk = max(1, 200000 // len(so))
    for s in range(0, len(corners), chunk):
        pts = (corners[s:s + chunk, None, :] + so[None]).reshape(-1, 3)
        pl = pts @ T[:3, :3].T + T[:3, 3]
        out[s:s + chunk] = cadmod.csg_contains(csg, pl).reshape(-1, len(so)).mean(axis=1)
    return out.reshape(prob.shape)


def baseline_occupancy(csg, prob: TopOptProblem, T: np.ndarray, wall: float) -> np.ndarray:
    """The CAD part as an occupancy field on the topology-optimisation grid, for the like-for-like check.

    Sampled finely enough to resolve its walls; the passive solid end flanges are set to 1 as in the
    optimised design. Wall material in voxels whose centre lies just outside the design cylinder is kept
    (evaluate_compliance ignores the passive masks), so the baseline keeps all of its material.
    """
    sub = int(np.clip(np.ceil(2 * prob.voxel / max(wall, 1e-6)), 3, 8))
    occ = _occupancy(csg, prob, T, sub=sub)
    occ[prob.passive_solid] = 1.0
    return occ


def build_problem(geo: LinkGeometry, cases: list[LoadCase], cfg: LinkOptConfig, mat: cadmod.Material
                  ) -> tuple[TopOptProblem, float]:
    """Voxel domain, supports and load cases of a link. Returns (problem, domain width).

    The square cross-section is the smallest even number of voxels that covers the capsule diameter
    2r (even, so the grid can be coarsened by multigrid). With cfg.cylinder, voxels whose centre lies
    outside the capsule radius r are passive void, so the part stays inside the capsule collision
    geometry to within half a voxel.
    """
    nelx = cfg.nel
    h = geo.length / nelx
    n_cross = max(4, 2 * int(np.ceil(geo.radius / h - 1e-9)))
    width = n_cross * h
    r_dom = geo.radius if cfg.cylinder else width / 2  # radius of the material cylinder / loaded disc
    prob = TopOptProblem(nelx, n_cross, n_cross, voxel=h, E0=mat.E, nu=mat.nu)
    c = prob.element_centres()
    rr = np.hypot(c[..., 1] - width / 2, c[..., 2] - width / 2)
    void = (rr > r_dom + 1e-12) if cfg.cylinder else np.zeros(prob.shape, bool)
    ends = np.zeros(prob.shape, bool)
    k = max(0, min(cfg.end_layers, nelx // 4))
    if k:
        ends[:k], ends[-k:] = True, True
    prob.passive_void, prob.passive_solid = void, ends & ~void
    prob.fixed[TopOptProblem.dofs(prob.nodes_where(0, 0))] = True
    xyz = prob.node_coords()
    tip = prob.nodes_where(0, -1)
    tip = tip[np.hypot(xyz[tip, 1] - width / 2, xyz[tip, 2] - width / 2) <= r_dom + 1e-9]
    Rg = np.stack([geo.axis, geo.u, geo.w])  # link -> grid rotation
    d = geo.axis * geo.length
    centre = np.array([geo.length, width / 2, width / 2])
    loads = np.zeros((len(cases), prob.ndof))
    for n, lc in enumerate(cases):
        F_tip = -lc.force
        M_tip = -lc.moment + np.cross(d, lc.force)
        f = wrench_to_nodal_forces(xyz[tip], centre, Rg @ F_tip, Rg @ M_tip)
        loads[n, (3 * tip[:, None] + np.arange(3)).ravel()] = f.ravel()
    prob.loads = loads
    prob.weights = np.ones(len(cases))
    return prob, width


def optimize_link(params: HumanoidParams | None = None, link: str = "thigh", cfg: LinkOptConfig | None = None,
                  load_cases: list[LoadCase] | None = None, xml: str | None = None, callback=None,
                  should_stop=None) -> LinkOptResult:
    """Topology-optimise `link` ("thigh", "shin", "upper_arm", "forearm") of the robot.

    load_cases: precomputed cases in the left link's frame (default: simulate with cfg.loads).
    callback(info) gets topopt's per-iteration dict plus "voxel" and "grid_to_link" (4x4), so a viewer
    can mesh info["x"]; should_stop() aborts between iterations.
    """
    t0 = time.perf_counter()
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}")
    cfg = cfg or LinkOptConfig()
    params = params or HumanoidParams()
    xml = xml or build_mjcf(params)
    model = mujoco.MjModel.from_xml_string(xml)
    geo = link_geometry(model, link)
    source = "given"
    if load_cases is None:
        report = extract_link_loads(params, cfg.loads, xml=xml)
        load_cases, source = report.cases.get(link, []), report.source
    if not load_cases:
        raise RuntimeError(f"no load cases for {link}")
    mat = cfg.cad.mat
    prob, width = build_problem(geo, load_cases, cfg, mat)
    T = geo.grid_to_link(width)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, geo.body)
    csg = cadmod.body_csg(model, b, cfg.cad)
    occ = vf_base = None
    if cfg.baseline or cfg.match_baseline:
        occ = baseline_occupancy(csg, prob, T, cfg.cad.wall)
        vf_base = float(min(1.0, occ.sum() / max(1, int((~prob.passive_void).sum()))))  # equal material
    volfrac = vf_base if cfg.match_baseline else cfg.volfrac
    to_cfg = TopOptConfig(volfrac=volfrac, rmin=cfg.rmin, max_iter=cfg.iterations, threads=cfg.threads,
                          backend=cfg.backend, cg_tol=cfg.cg_tol)
    cb = None
    if callback is not None:
        def cb(info):
            callback({**info, "voxel": prob.voxel, "grid_to_link": T})
    res = run_topopt(prob, to_cfg, callback=cb, should_stop=should_stop)
    mesh = density_to_mesh(res.x, prob.voxel, smooth_iterations=cfg.smooth_iterations)
    mesh.apply_transform(T)
    props = cadmod.MassProps.of_mesh(mesh, mat.density)

    base_mesh = cadmod._to_trimesh(cadmod._manifold(csg, cfg.cad.segments))
    mass_before = float(abs(base_mesh.volume) * mat.density)
    c_base = c_thr = None
    if cfg.baseline:
        # like-for-like stiffness check: both designs as occupancy fields with E proportional to the
        # occupied fraction (p = 1), loaded through the same solid end flanges
        c_base, _, _ = evaluate_compliance(prob, occ, to_cfg, penal=1.0)
        c_thr, _, _ = evaluate_compliance(prob, (res.x >= 0.5).astype(float), to_cfg, penal=1.0)
    U = res.displacements
    umax = float(np.sqrt((U.reshape(len(U), -1, 3) ** 2).sum(-1)).max()) * 1000 if U is not None else 0.0
    domain = {"nelx": prob.nelx, "nely": prob.nely, "nelz": prob.nelz, "voxel_mm": prob.voxel * 1000,
              "length_m": geo.length, "radius_m": geo.radius, "material": mat.name, "E_GPa": mat.E / 1e9,
              "volfrac": volfrac, "cylinder": cfg.cylinder}
    return LinkOptResult(link, geo, mesh, props, mass_before, props.mass, res.compliance, c_thr, c_base, vf_base,
                         umax, list(load_cases), source, res, domain, time.perf_counter() - t0)


class TopOptJob:
    """optimize_link in a background thread with a status() for the GUI (Design tab)."""

    def __init__(self, params: HumanoidParams, link: str = "thigh", cfg: LinkOptConfig | None = None,
                 xml: str | None = None, preview_every: int = 5):
        self.params, self.link, self.cfg, self.xml = params, link, cfg or LinkOptConfig(), xml
        self.preview_every = preview_every
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state, self.error = "idle", ""
        self.history: list[dict] = []
        self.result: LinkOptResult | None = None
        self.mesh_payload: dict | None = None
        self.mesh_id = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_safe, name="topopt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def status(self) -> dict:
        with self._lock:
            out = {"state": self.state, "error": self.error, "link": self.link,
                   "iteration": len(self.history), "iterations": self.cfg.iterations,
                   "history": [{k: h[k] for k in ("iteration", "compliance", "volume", "change")}
                               for h in self.history[-400:]],
                   "mesh": self.mesh_payload, "mesh_id": self.mesh_id,
                   "mass_before_kg": None, "mass_after_kg": None, "mass_saving_pct": None}
            if self.result is not None:
                s = self.result.summary()
                out.update(mass_before_kg=s["mass_before_kg"], mass_after_kg=s["mass_after_kg"],
                           mass_saving_pct=s["mass_saving_pct"], summary=s)
            return out

    def _callback(self, info: dict) -> None:
        x = info.pop("x", None)
        voxel, T = info.pop("voxel", None), info.pop("grid_to_link", None)
        with self._lock:
            self.history.append(info)
        if x is not None and self.preview_every > 0 and info["iteration"] % self.preview_every == 0:
            try:  # live preview of the evolving part
                mesh = density_to_mesh(x, voxel, smooth_iterations=0)
                mesh.apply_transform(T)
                payload = cadmod.to_gui_mesh(mesh)
            except ValueError:
                return
            with self._lock:
                self.mesh_payload = payload
                self.mesh_id += 1

    def _run_safe(self) -> None:
        try:
            self.state = "running"
            res = optimize_link(self.params, self.link, self.cfg, xml=self.xml, callback=self._callback,
                                should_stop=self._stop.is_set)
            payload = cadmod.to_gui_mesh(res.mesh)
            with self._lock:
                self.result, self.mesh_payload = res, payload
                self.mesh_id += 1
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as e:  # noqa: BLE001 - surface in the GUI
            import traceback
            traceback.print_exc()
            self.error, self.state = f"{type(e).__name__}: {e}", "error"
