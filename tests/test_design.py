"""Tests for simulatics.design: CAD parts, MJCF/URDF export, SIMP topology optimisation, loads, design search."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from simulatics.design import cad, loads, topopt
from simulatics.design.evaluate import EvalConfig
from simulatics.design.linkopt import (LinkOptConfig, TopOptJob, baseline_occupancy, build_problem, link_geometry,
                                       optimize_link)
from simulatics.design.optimizer import CMAES, BayesOpt, DesignConfig, DesignOptimizer, design_genes
from simulatics.models import HumanoidParams, build_mjcf

PUSH = loads.LoadConfig(scenario="push", push_seconds=2.0)


@pytest.fixture(scope="module")
def robot_cad():
    return cad.build_robot_cad(HumanoidParams(), cad.CadOptions())


# ------------------------------------------------------------------------------------------- CAD
def test_tube_mass_matches_analytic():
    ro, ri, L, rho = 0.05, 0.046, 0.4, 1240.0
    mesh = cad.tube_mesh(ro, ri, L, segments=256)
    mp = cad.MassProps.of_mesh(mesh, rho)
    m = rho * np.pi * (ro ** 2 - ri ** 2) * L
    assert mesh.is_watertight
    assert mp.mass == pytest.approx(m, rel=2e-3)
    assert np.allclose(mp.com, [0, 0, L / 2], atol=1e-6)
    assert mp.inertia[2, 2] == pytest.approx(m * (ro ** 2 + ri ** 2) / 2, rel=3e-3)
    assert mp.inertia[0, 0] == pytest.approx(m * (3 * (ro ** 2 + ri ** 2) + L ** 2) / 12, rel=3e-3)


def test_cad_parts_are_closed_hollow_and_carry_motor_masses(robot_cad):
    model = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    assert set(robot_cad.parts) == {model.body(b).name for b in cad.robot_bodies(model)}
    opts = robot_cad.opts
    for p in robot_cad.parts.values():
        assert p.mesh.is_watertight and p.mesh.volume > 0
        assert p.shell.mass > 0 and np.all(np.linalg.eigvalsh(p.props.inertia) > 0)
    thigh = robot_cad.parts["left_thigh"]
    r, hl = model.geom("left_thigh").size[:2]
    solid = opts.mat.density * (np.pi * r ** 2 * 2 * hl + 4 / 3 * np.pi * r ** 3)
    assert thigh.shell.mass < 0.5 * solid  # hollow
    n_hinges = sum(1 for a in range(model.nu)
                   if model.jnt_type[model.actuator_trnid[a, 0]] == mujoco.mjtJoint.mjJNT_HINGE)
    extras = sum(m.mass for p in robot_cad.parts.values() for _, m in p.extras)
    assert extras == pytest.approx(n_hinges * opts.motor_mass + opts.payload_mass)
    s = robot_cad.summary()
    assert s["total_mass_kg"] == pytest.approx(s["shell_mass_kg"] + extras)
    assert s["primitive_mass_kg"] == pytest.approx(model.body_subtreemass[1])


def test_cad_mjcf_compiles_and_stands(robot_cad):
    p = HumanoidParams()
    xml = robot_cad.mjcf()
    m = mujoco.MjModel.from_xml_string(xml)
    m0 = mujoco.MjModel.from_xml_string(build_mjcf(p))
    assert m.nu == m0.nu and m.nq == m0.nq
    assert m.body_subtreemass[1] == pytest.approx(robot_cad.total_mass, rel=1e-6)
    vis = [g for g in range(m.ngeom) if m.geom(g).name.startswith("cad_")]
    assert len(vis) == len(robot_cad.parts)
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] == 3 for g in vis)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # the visual mesh sits where the part was modelled (MuJoCo re-centres meshes internally)
    g = m.geom("cad_left_thigh").id
    mid = m.geom_dataid[g]
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    world = d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T
    b = m.body("left_thigh").id
    expect = d.xpos[b] + robot_cad.parts["left_thigh"].mesh.vertices @ d.xmat[b].reshape(3, 3).T
    assert np.allclose(world.min(0), expect.min(0), atol=1e-4) and np.allclose(world.max(0), expect.max(0), atol=1e-4)
    mujoco.mj_step(m, d, 1000)
    assert d.qpos[2] > 0.95 * p.stand_height


def test_cad_exports_mesh_files(robot_cad, tmp_path):
    paths = robot_cad.export(tmp_path, formats=("stl", "obj", "glb"))
    assert (tmp_path / "left_thigh.stl").stat().st_size > 1000
    assert (tmp_path / "left_thigh.obj").exists() and (tmp_path / "robot.glb").exists()
    assert len(paths) == 2 * len(robot_cad.parts) + 1
    xml = robot_cad.mjcf(mesh_dir=tmp_path / "meshes")
    assert mujoco.MjModel.from_xml_string(xml).nmesh == len(robot_cad.parts)


def test_step_export_matches_mesh_volume(robot_cad, tmp_path):
    pytest.importorskip("build123d")
    part = robot_cad.parts["left_thigh"]
    solid = cad._brep(part.csg)
    assert solid.volume / 1e9 == pytest.approx(part.mesh.volume, rel=0.015)  # mm^3 vs m^3, tessellation
    one = cad.RobotCad({"left_thigh": part}, robot_cad.opts, robot_cad.primitive_mass, robot_cad.xml)
    paths = one.export_step(tmp_path)
    assert len(paths) == 1 and paths[0].read_text(errors="ignore").startswith("ISO-10303-21")


def test_urdf_has_every_joint_and_matches_mujoco_kinematics():
    p = HumanoidParams()
    m0 = mujoco.MjModel.from_xml_string(build_mjcf(p))
    urdf = cad.export_urdf(p)
    root = ET.fromstring(urdf)
    joints = root.findall("joint")
    n_hinge = int((m0.jnt_type == mujoco.mjtJoint.mjJNT_HINGE).sum())
    n_slide = int((m0.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE).sum())
    assert sum(j.get("type") == "revolute" for j in joints) == n_hinge
    assert sum(j.get("type") == "prismatic" for j in joints) == n_slide
    assert n_hinge >= 20
    names = {j.get("name") for j in joints}
    assert all(m0.actuator(a).name in names for a in range(m0.nu))
    mass = sum(float(e.get("value")) for e in root.iter("mass"))
    assert mass == pytest.approx(m0.body_subtreemass[1], abs=0.05)  # + tiny dummy-link masses
    # MuJoCo reads the URDF; same joint angles give the same link poses relative to the torso
    mu = mujoco.MjModel.from_xml_string(urdf)
    d0, du = mujoco.MjData(m0), mujoco.MjData(mu)
    rng = np.random.default_rng(0)
    for j in range(mu.njnt):
        j0 = m0.joint(mu.joint(j).name).id
        q = rng.uniform(*m0.jnt_range[j0]) if m0.jnt_limited[j0] else 0.2
        d0.qpos[m0.jnt_qposadr[j0]] = du.qpos[mu.jnt_qposadr[j]] = q
    mujoco.mj_kinematics(m0, d0)
    mujoco.mj_kinematics(mu, du)
    t0 = m0.body("torso").id
    tu = max(mujoco.mj_name2id(mu, mujoco.mjtObj.mjOBJ_BODY, "torso"), 0)  # fused into world when static
    for b in range(1, mu.nbody):
        name = mu.body(b).name
        if mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            continue
        b0 = m0.body(name).id
        R0, Ru = d0.xmat[t0].reshape(3, 3), du.xmat[tu].reshape(3, 3)
        assert np.allclose(R0.T @ (d0.xpos[b0] - d0.xpos[t0]), Ru.T @ (du.xpos[b] - du.xpos[tu]), atol=1e-9)
        assert np.allclose(R0.T @ d0.xmat[b0].reshape(3, 3), Ru.T @ du.xmat[b].reshape(3, 3), atol=1e-9)


def test_urdf_with_cad_inertials_and_meshes(robot_cad):
    urdf = cad.export_urdf(HumanoidParams(), robot_cad.opts, cad=robot_cad, mesh_prefix="meshes/")
    root = ET.fromstring(urdf)
    assert len(root.findall(".//visual/geometry/mesh")) == len(robot_cad.parts)
    mass = sum(float(e.get("value")) for e in root.iter("mass"))
    assert mass == pytest.approx(robot_cad.total_mass, abs=0.05)


# --------------------------------------------------------------------------------------- topopt
def test_hex8_stiffness_has_six_rigid_body_modes():
    KE = topopt.hex8_stiffness(0.3)
    w = np.linalg.eigvalsh(KE)
    assert np.allclose(KE, KE.T) and (np.abs(w) < 1e-10).sum() == 6 and w[6] > 1e-3
    assert np.allclose(topopt.hex8_stiffness(0.3, 0.01), 0.01 * KE)


def test_solid_cantilever_matches_beam_theory():
    nx, n, h, E, F = 40, 8, 0.01, 70e9, 1000.0
    prob = topopt.cantilever(nx, n, n, voxel=h, E0=E, nu=0.3, force=F)
    c, _, _ = topopt.evaluate_compliance(prob, np.ones(prob.shape), topopt.TopOptConfig(cg_tol=1e-9, threads=2))
    L, b = nx * h, n * h
    delta = F * L ** 3 / (3 * E * b ** 4 / 12) + F * L / (E / 2.6 * b * b * 5 / 6)  # Timoshenko
    assert c / (F * delta) == pytest.approx(1.0, abs=0.05)


def test_wrench_to_nodal_forces_reproduces_resultant():
    rng = np.random.default_rng(0)
    pts = np.c_[np.zeros(30), rng.uniform(-1, 1, (30, 2))]
    F, M, c = np.array([1.0, -2.0, 3.0]), np.array([0.5, 0.2, -0.7]), np.array([0.0, 0.1, 0.0])
    f = topopt.wrench_to_nodal_forces(pts, c, F, M)
    assert np.allclose(f.sum(0), F) and np.allclose(np.cross(pts - c, f).sum(0), M)


@pytest.mark.skipif(not topopt.native_available(), reason="topopt native library not built")
def test_topopt_native_matches_numpy_fallback():
    prob = topopt.cantilever(12, 4, 4)
    prob.passive_void[5:7, :2, :2] = True
    runs = [topopt.run_topopt(prob, topopt.TopOptConfig(volfrac=0.3, max_iter=8, cg_tol=1e-11, backend=b,
                                                        threads=2)) for b in ("native", "numpy")]
    a, b = runs
    assert (a.backend, b.backend) == ("native", "numpy")
    ca = np.array([h["compliance"] for h in a.history])
    cb = np.array([h["compliance"] for h in b.history])
    assert np.allclose(ca, cb, rtol=1e-7)
    assert np.abs(a.x - b.x).max() < 1e-6


@pytest.mark.skipif(not topopt.native_available(), reason="topopt native library not built")
def test_multigrid_matches_jacobi_with_far_fewer_iterations():
    prob = topopt.cantilever(24, 8, 8)
    r = topopt.run_topopt(prob, topopt.TopOptConfig(volfrac=0.3, max_iter=10, threads=2))
    E = 1e-9 + r.x.ravel() ** 3  # a realistic, filtered SIMP field
    out = {}
    for solver in ("jacobi", "mg"):
        cfg = topopt.TopOptConfig(cg_tol=1e-9, threads=2, solver=solver, cg_maxit=20000)
        be = topopt.make_backend(prob, cfg)
        out[solver] = be.solve_all(E, prob.loads, np.zeros_like(prob.loads))
        assert solver == "jacobi" or be.levels >= 2
    (uj, ij), (um, im) = out["jacobi"], out["mg"]
    assert np.abs(uj - um).max() < 1e-6 * np.abs(uj).max()
    assert 0 < im[0] < ij[0] / 5


def test_topopt_cantilever_benchmark():
    prob = topopt.cantilever(24, 8, 8)
    seen = []
    res = topopt.run_topopt(prob, topopt.TopOptConfig(volfrac=0.25, max_iter=30, threads=4), callback=seen.append)
    c = [h["compliance"] for h in res.history]
    assert len(seen) == res.iterations and set(seen[0]) >= {"compliance", "volume", "change", "x"}
    assert c[-1] < 0.2 * c[0] and c[-1] <= min(c[5:]) * 1.02  # decreases and settles
    assert res.volume == pytest.approx(0.25, abs=0.0025)    # volume constraint within 1 %
    assert all(abs(h["volume"] - 0.25) < 0.0025 for h in res.history[1:])
    assert 0 <= res.x.min() and res.x.max() <= 1
    mesh = topopt.density_to_mesh(res.x, 1.0)
    assert mesh.is_watertight and 0.5 * 0.25 * prob.nel < mesh.volume < 1.5 * 0.25 * prob.nel


def test_topopt_stop_flag():
    prob = topopt.cantilever(12, 4, 4)
    n = []
    res = topopt.run_topopt(prob, topopt.TopOptConfig(max_iter=50, tol_change=0.0), callback=n.append,
                            should_stop=lambda: len(n) >= 3)
    assert res.stopped and res.iterations == 3 and not res.converged


def test_topopt_rejects_degenerate_inputs_with_a_message():
    # these used to end in a bare ZeroDivisionError from the OC bisection (or NaN densities)
    def run(prob=None, x0=None, **kw):
        prob = prob or topopt.cantilever(8, 4, 4)
        return topopt.run_topopt(prob, topopt.TopOptConfig(max_iter=2, threads=2, **kw), x0=x0)

    for kw in ({"volfrac": 0.0}, {"volfrac": 1.2}, {"rmin": 0.0}):
        with pytest.raises(ValueError):
            run(**kw)
    for edit in (lambda p: p.loads.fill(0.0), lambda p: p.fixed.fill(False),
                 lambda p: p.loads.__setitem__((0, -1), np.nan)):
        prob = topopt.cantilever(8, 4, 4)
        edit(prob)
        with pytest.raises(ValueError):
            run(prob)
    with pytest.raises(ValueError):
        run(x0=np.ones(5))


def test_topopt_oc_survives_an_unreachable_volume_target():
    # the move limit keeps the first steps below volfrac (and volfrac = 1 is never exceeded); the
    # bisection used to divide by zero instead of taking the largest allowed step
    prob = topopt.cantilever(8, 4, 4)
    res = topopt.run_topopt(prob, topopt.TopOptConfig(volfrac=0.5, move=0.2, max_iter=3, tol_change=0.0, threads=2),
                            x0=np.full(prob.nel, 0.1))
    vols = [h["volume"] for h in res.history]
    assert np.all(np.diff(vols) > 0) and res.volume == pytest.approx(0.5, abs=0.01)
    full = topopt.run_topopt(prob, topopt.TopOptConfig(volfrac=1.0, max_iter=2, threads=2))
    assert np.isfinite(full.compliance) and full.volume == pytest.approx(1.0, abs=1e-6)


def test_density_to_mesh_keeps_the_largest_of_many_small_islands():
    x = np.zeros((20, 20, 20))
    x[::3, ::3, ::3] = 1.0  # 343 equal islands, each below min_component_fraction
    mesh = topopt.density_to_mesh(x, 1.0)
    assert len(mesh.faces) > 0 and mesh.volume > 0


def test_density_to_mesh_of_a_block():
    x = np.zeros((10, 6, 6))
    x[2:8, 1:5, 1:5] = 1.0
    mesh = topopt.density_to_mesh(x, 0.01, smooth_iterations=0)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(6 * 4 * 4 * 1e-6, rel=0.1)
    assert np.allclose(mesh.bounds, [[0.02, 0.01, 0.01], [0.08, 0.05, 0.05]], atol=0.006)


# ---------------------------------------------------------------------------------------- loads
def test_link_wrench_matches_statics_when_standing():
    m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 400)
    mujoco.mj_forward(m, d)
    mujoco.mj_rnePostConstraint(m, d)
    ids = np.array([m.body("left_thigh").id, m.body("left_forearm").id])
    W = loads.link_wrenches(m, d, ids)
    legs = sum(m.body_subtreemass[m.body(f"{s}_thigh").id] for s in ("left", "right"))
    upper = m.body_subtreemass[m.body("torso").id] - legs
    assert W[0, 2] == pytest.approx(-upper * 9.81 / 2, rel=0.02)   # hip pushes the thigh down
    assert W[1, 2] == pytest.approx(m.body_subtreemass[ids[1]] * 9.81, rel=0.02)  # elbow holds the forearm up


def test_extract_link_loads_push_scenario():
    rep = loads.extract_link_loads(HumanoidParams(), PUSH)
    assert rep.source == "push" and rep.seconds > 1.0
    assert set(rep.cases) == set(loads.LINKS)
    for link, cases in rep.cases.items():
        assert 1 <= len(cases) <= PUSH.max_cases
        assert all(np.isfinite(c.force).all() and np.isfinite(c.moment).all() for c in cases)
    assert rep.peak["thigh"]["force_N"] > rep.peak["forearm"]["force_N"]


# ------------------------------------------------------------------------------- link optimisation
def test_optimize_link_and_apply(robot_cad):
    cfg = LinkOptConfig(nel=16, iterations=8, loads=PUSH, threads=4)
    res = optimize_link(HumanoidParams(), "shin", cfg)
    s = res.summary()
    assert res.mesh.is_watertight and res.mass_after > 0 and res.mass_before > 0
    assert s["topopt"]["volume"] == pytest.approx(cfg.volfrac, abs=0.01)
    lo, hi = res.mesh.bounds
    L, r = res.geometry.length, res.geometry.radius
    assert lo[2] > -L - 0.02 and hi[2] < 0.02 and max(np.abs(lo[:2]).max(), np.abs(hi[:2]).max()) < r + 0.02
    assert s["compliance_baseline_J"] > 0 and s["compliance_thresholded_J"] > 0
    assert s["mass_saving_pct"] == pytest.approx(100 * (1 - res.mass_after / res.mass_before))
    robot = cad.build_robot_cad(HumanoidParams(), robot_cad.opts)
    before = robot.total_mass
    res.apply(robot)
    assert robot.total_mass == pytest.approx(before - 2 * (res.mass_before - res.mass_after), rel=0.02)
    assert robot.parts["right_shin"].mesh.volume > 0  # mirrored part keeps outward normals
    m = mujoco.MjModel.from_xml_string(robot.mjcf())
    assert m.body_subtreemass[1] == pytest.approx(robot.total_mass, rel=1e-6)


def test_link_domain_is_the_capsule_and_the_baseline_keeps_the_whole_tube_wall():
    # The cross-section used to be 2*round(r/h) voxels: narrower than the shin capsule (8 vs 8.5 voxels),
    # so the voxelised baseline lost ~half its tube wall (a spurious "6x stiffer than the tube"), and
    # wider than the forearm capsule, so the optimised part poked out of the collision capsule.
    model = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    cfg = LinkOptConfig(cad=cad.CadOptions(bosses=False))
    case = [loads.LoadCase("x", np.array([0.0, 0.0, -100.0]), np.zeros(3), 0.0, "left", "test", "given")]
    for link in ("shin", "forearm"):
        geo = link_geometry(model, link)
        prob, width = build_problem(geo, case, cfg, cfg.cad.mat)
        assert width >= 2 * geo.radius and prob.nely % 2 == 0
        c = prob.element_centres()
        rr = np.hypot(c[..., 1] - width / 2, c[..., 2] - width / 2)
        assert np.array_equal(prob.passive_void, rr > geo.radius + 1e-12)
        csg = cad.body_csg(model, model.body(geo.body).id, cfg.cad)
        occ = baseline_occupancy(csg, prob, geo.grid_to_link(width), cfg.cad.wall)
        h = prob.voxel
        tube = np.pi * (geo.radius ** 2 - (geo.radius - cfg.cad.wall) ** 2) * (prob.nelx - 2) * h
        assert occ[1:-1].sum() * h ** 3 == pytest.approx(tube, rel=0.05)


def test_topopt_job_lifecycle():
    job = TopOptJob(HumanoidParams(), "forearm", LinkOptConfig(nel=12, iterations=4, loads=PUSH, baseline=False),
                    preview_every=2)
    assert job.status()["state"] == "idle"
    job.start()
    job.join(120)
    st = job.status()
    assert st["state"] == "finished", st["error"]
    assert st["iteration"] == 4 and st["mesh"] and st["mesh_id"] >= 2 and st["mass_saving_pct"] is not None


# ---------------------------------------------------------------------------------- design search
def test_cmaes_and_bo_find_toy_optima():
    es = CMAES(np.full(4, 0.8), 0.3, seed=1)
    for _ in range(60):
        X = es.ask()
        es.tell(X, ((X - 0.3) ** 2).sum(1))
    assert np.allclose(es.mean, 0.3, atol=1e-3)
    bo = BayesOpt(2, np.array([0.5, 0.5]), seed=0)
    for _ in range(6):
        X = bo.ask(4)
        bo.tell(X, -((X - [0.7, 0.2]) ** 2).sum(1))
    assert np.abs(bo.X[np.argmax(bo.y)] - [0.7, 0.2]).max() < 0.08


def test_config_inputs_from_the_gui_are_validated_or_coerced(tmp_path):
    o = cad.CadOptions.from_dict({"wall_mm": "2.5", "density": "1200", "segments": "24"})
    assert o.wall == pytest.approx(0.0025) and o.mat.density == 1200.0 and o.segments == 24
    with pytest.raises(ValueError):  # used to run the stand test silently
        EvalConfig.from_dict({"mode": "wlak"})
    opt = DesignOptimizer(DesignConfig(generations=1, population=2, workers=1, eval={"mode": "bogus"}),
                          runs_dir=tmp_path)
    opt.start()
    opt.join(30)
    assert opt.status()["state"] == "error" and "mode" in opt.status()["error"]


def test_design_genes_listing():
    g = design_genes()
    assert {"name", "label", "default_on"} <= set(g[0]) and any(x["default_on"] for x in g)


@pytest.mark.parametrize("algo,mode", [("cmaes", "auto"), ("bo", "stand")])
def test_design_optimizer_smoke(tmp_path, algo, mode):
    cfg = DesignConfig(algo=algo, generations=2, population=3, workers=2, name="smoke",
                       genes=["thigh_length", "shin_length", "knee_kp"],
                       eval={"mode": mode, "goals": [[0.6, 0.0]], "timeout": 4.0, "stand_seconds": 1.5})
    opt = DesignOptimizer(cfg, HumanoidParams(), runs_dir=tmp_path)
    opt.start()
    opt.join(300)
    st = opt.status()
    assert st["state"] == "finished", st["error"]
    assert st["generation"] == 2 and len(st["history"]) == 2
    assert set(st["history"][0]) >= {"gen", "best", "mean", "center", "sps"}
    assert np.isfinite(st["best"]) and st["best"] >= max(h["best"] for h in st["history"]) - 1e-12
    doc = json.loads((tmp_path / "smoke" / "design.json").read_text())
    assert doc["score"] == pytest.approx(st["best"]) and set(doc["gene_values"]) == set(cfg.genes)
    p = HumanoidParams.from_dict(doc["params"])
    assert mujoco.MjModel.from_xml_string(build_mjcf(p)).nu > 0
