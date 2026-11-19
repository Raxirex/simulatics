"""Forward 0.35 / 0.4 m/s through RLController from several start headings, per snapshot."""
import sys
import numpy as np
import mujoco
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # repository root
from threadpoolctl import threadpool_limits
threadpool_limits(1)
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.train.ppo import RLPolicy, RLController
from simulatics.control.base import base_pose

m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
for path in sys.argv[1:]:
    pol = RLPolicy.load(path)
    line = []
    for v in (0.35, 0.4):
        falls, vxs = 0, []
        for a0 in np.linspace(-np.pi, np.pi, 5, endpoint=False):
            ctrl = RLController(m, pol)
            d = mujoco.MjData(m)
            d.qpos[3:7] = (np.cos(a0 / 2), 0, 0, np.sin(a0 / 2))
            mujoco.mj_forward(m, d)
            ctrl.reset(m, d)
            ctrl.set_velocity(v, 0.0, 0.0)
            vs = []
            for k in range(1500):
                ctrl.step(m, d)
                mujoco.mj_step(m, d, 2)
                if ctrl.status()["mode"] == "fallen":
                    falls += 1
                    break
                if k > 300:
                    _, _, yaw = base_pose(d)
                    vs.append(np.cos(yaw) * d.qvel[0] + np.sin(yaw) * d.qvel[1])
            if vs:
                vxs.append(np.mean(vs))
        line.append(f"v {v}: falls {falls}/5 mean vx {np.mean(vxs) if vxs else float('nan'):.3f}")
    print(path, " | ".join(line), flush=True)

