"""URDF export of the parametric humanoid (or any MJCF robot of hinge joints and primitive geoms).

The kinematics come from the compiled MuJoCo model, so the URDF matches the simulation exactly:

* one URDF <link> per robot body, with the same name, plus massless-in-practice "*_link" frames
  between the hinges of bodies that carry several joints (URDF allows one joint per link; MuJoCo
  composes the hinges of one body in order, which a chain of revolute joints reproduces),
* one revolute (prismatic) <joint> per MuJoCo hinge (slide), named like it and like the actuator
  driving it, with limits in rad (m), effort = actuator force limit, damping = joint damping, and
  <mimic> for joints coupled by a MuJoCo joint equality constraint,
* collision geometry: capsules as a cylinder plus two spheres, boxes, spheres, cylinders,
* inertials: from the CAD parts when `opts` (CadOptions) is given, otherwise from the MjModel.

The root link is the free-floating torso; tools such as ros2_control add the floating
base themselves. Mesh visuals are referenced as `<mesh_prefix><body>.stl` when `mesh_prefix` is set.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..models import HumanoidParams, build_mjcf

_DUMMY_MASS = 1e-3      # kg, URDF consumers dislike zero-mass links
_DUMMY_INERTIA = 1e-6   # kg m^2


def _f(v) -> str:
    return " ".join(f"{float(x):.7g}" for x in np.ravel(v))


def _rpy(quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz")  # extrinsic XYZ = URDF roll-pitch-yaw


def _origin(parent: ET.Element, xyz, rpy=(0, 0, 0)) -> None:
    ET.SubElement(parent, "origin", {"xyz": _f(xyz), "rpy": _f(rpy)})


def _inertial(link: ET.Element, mass: float, com, I: np.ndarray) -> None:
    el = ET.SubElement(link, "inertial")
    _origin(el, com)
    ET.SubElement(el, "mass", {"value": f"{mass:.7g}"})
    ET.SubElement(el, "inertia", {"ixx": f"{I[0, 0]:.7g}", "iyy": f"{I[1, 1]:.7g}", "izz": f"{I[2, 2]:.7g}",
                                  "ixy": f"{I[0, 1]:.7g}", "ixz": f"{I[0, 2]:.7g}", "iyz": f"{I[1, 2]:.7g}"})


def _geometry(link: ET.Element, tag: str, model: mujoco.MjModel, g: int, name: str, shift=np.zeros(3)) -> None:
    t, s = model.geom_type[g], model.geom_size[g]
    pos, rpy = model.geom_pos[g] - shift, _rpy(model.geom_quat[g])
    R = Rotation.from_euler("xyz", rpy).as_matrix()

    def add(shape: str, attrs: dict, xyz) -> None:
        el = ET.SubElement(link, tag, {"name": f"{name}_{len(link.findall(tag))}"})
        _origin(el, xyz, rpy)
        ET.SubElement(ET.SubElement(el, "geometry"), shape, attrs)

    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        add("cylinder", {"radius": f"{s[0]:.7g}", "length": f"{2 * s[1]:.7g}"}, pos)
        for sgn in (1, -1):
            add("sphere", {"radius": f"{s[0]:.7g}"}, pos + R @ [0, 0, sgn * s[1]])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        add("cylinder", {"radius": f"{s[0]:.7g}", "length": f"{2 * s[1]:.7g}"}, pos)
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        add("sphere", {"radius": f"{s[0]:.7g}"}, pos)
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        add("box", {"size": _f(2 * s[:3])}, pos)


def export_urdf(params: HumanoidParams | None = None, opts=None, xml: str | None = None,
                mesh_prefix: str | None = None, velocity_limit: float = 15.0, name: str = "simulatics_humanoid",
                cad=None) -> str:
    """URDF string of the robot built from `params` (or from MJCF `xml`).

    opts: CadOptions -> inertials from the CAD parts (cad.build_robot_cad); None -> from MuJoCo.
    cad: an already built RobotCad (skips rebuilding the parts).
    mesh_prefix: e.g. "package://simulatics_humanoid/meshes/" -> visual meshes <prefix><body>.stl.
    """
    from .cad import build_robot_cad, robot_bodies  # local: cad imports this module lazily too
    if xml is None:
        xml = build_mjcf(params or HumanoidParams())
    model = mujoco.MjModel.from_xml_string(xml)
    if cad is None and opts is not None:
        cad = build_robot_cad(params, opts, xml)
    bodies = robot_bodies(model)
    bname = {b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
    effort = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            lo, hi = model.actuator_forcerange[a]
            effort[int(model.actuator_trnid[a, 0])] = (float(max(abs(lo), abs(hi)))
                                                       if model.actuator_forcelimited[a] else 1e3)

    movable = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)

    def hinges_of(b: int) -> list[int]:
        """Hinge and slide joints of body b, in MuJoCo's order."""
        return [int(j) for j in np.flatnonzero(model.jnt_bodyid == b) if model.jnt_type[j] in movable]

    # joint equality constraints q1 = c0 + c1 q2 become URDF <mimic> (e.g. the mirrored gripper finger)
    mimic = {}
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e] >= 0:
            mimic[int(model.eq_obj1id[e])] = (int(model.eq_obj2id[e]), float(model.eq_data[e, 1]),
                                              float(model.eq_data[e, 0]))

    # A body's URDF link frame sits at its last joint anchor (zero for the built-in humanoid).
    shift = {b: (model.jnt_pos[hinges_of(b)[-1]].copy() if hinges_of(b) else np.zeros(3)) for b in bodies}

    robot = ET.Element("robot", {"name": name})
    for b in bodies:
        nm, sh = bname[b], shift[b]
        link = ET.SubElement(robot, "link", {"name": nm})
        if cad is not None and nm in cad.parts:
            p = cad.parts[nm].props
            _inertial(link, p.mass, p.com - sh, p.inertia)
        else:
            R = Rotation.from_quat(np.roll(model.body_iquat[b], -1)).as_matrix()
            _inertial(link, float(model.body_mass[b]), model.body_ipos[b] - sh,
                      R @ np.diag(model.body_inertia[b]) @ R.T)
        for g in np.flatnonzero(model.geom_bodyid == b):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            _geometry(link, "collision", model, int(g), nm, sh)
            if mesh_prefix is None:
                _geometry(link, "visual", model, int(g), nm, sh)
        if mesh_prefix is not None:
            vis = ET.SubElement(link, "visual", {"name": f"{nm}_mesh"})
            _origin(vis, -sh)
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", {"filename": f"{mesh_prefix}{nm}.stl"})

    for b in bodies:
        parent_body = int(model.body_parentid[b])
        if parent_body not in bname:
            continue  # the root link
        hinges = hinges_of(b)
        parent, xyz, rpy = bname[parent_body], model.body_pos[b] - shift[parent_body], _rpy(model.body_quat[b])
        if not hinges:
            j = ET.SubElement(robot, "joint", {"name": f"{bname[b]}_fixed", "type": "fixed"})
            _origin(j, xyz, rpy)
            ET.SubElement(j, "parent", {"link": parent})
            ET.SubElement(j, "child", {"link": bname[b]})
            continue
        for n, jid in enumerate(hinges):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            last = n == len(hinges) - 1
            child = bname[b] if last else f"{bname[b]}_{jname}_link"
            if not last:
                dummy = ET.SubElement(robot, "link", {"name": child})
                _inertial(dummy, _DUMMY_MASS, (0, 0, 0), _DUMMY_INERTIA * np.eye(3))
            # the joint anchor (jnt_pos) is the joint frame origin; the axis is in the body frame
            kind = "revolute" if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
            el = ET.SubElement(robot, "joint", {"name": jname, "type": kind})
            off = model.jnt_pos[jid]
            if n == 0:
                Rb = Rotation.from_euler("xyz", rpy).as_matrix()
                _origin(el, xyz + Rb @ off, rpy)
            else:
                _origin(el, off - model.jnt_pos[hinges[n - 1]])
            ET.SubElement(el, "parent", {"link": parent})
            ET.SubElement(el, "child", {"link": child})
            ET.SubElement(el, "axis", {"xyz": _f(model.jnt_axis[jid])})
            lo, hi = model.jnt_range[jid] if model.jnt_limited[jid] else (-np.pi, np.pi)
            ET.SubElement(el, "limit", {"lower": f"{lo:.7g}", "upper": f"{hi:.7g}",
                                        "effort": f"{effort.get(jid, 1e3):.7g}", "velocity": f"{velocity_limit:.7g}"})
            ET.SubElement(el, "dynamics", {"damping": f"{model.dof_damping[model.jnt_dofadr[jid]]:.7g}"})
            if jid in mimic:
                src, mult, offset = mimic[jid]
                ET.SubElement(el, "mimic", {"joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, src),
                                            "multiplier": f"{mult:.7g}", "offset": f"{offset:.7g}"})
            parent = child
    ET.indent(robot)
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode")
