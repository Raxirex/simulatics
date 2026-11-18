import numpy as np
import pytest

from simulatics.control import PathFollower


def simulate(follower, x=0.0, y=0.0, yaw=0.0, dt=0.02, steps=3000):
    """Ideal velocity-tracking robot: the follower's command is executed exactly."""
    for _ in range(steps):
        c = follower.command(x, y, yaw)
        if c.done:
            return x, y, yaw, True
        x += (np.cos(yaw) * c.vx - np.sin(yaw) * c.vy) * dt
        y += (np.sin(yaw) * c.vx + np.cos(yaw) * c.vy) * dt
        yaw += c.wz * dt
    return x, y, yaw, False


@pytest.mark.parametrize("goal,final_yaw", [((4.0, 0.0), None), ((-3.0, 2.0), np.pi / 2), ((0.3, -0.2), -2.5)])
def test_reaches_goal_precisely(goal, final_yaw):
    f = PathFollower()
    f.set_path(np.array([[0.0, 0.0], [goal[0] / 2, goal[1] / 2 + 1.0], goal]), final_yaw)
    x, y, yaw, done = simulate(f)
    assert done
    assert np.hypot(x - goal[0], y - goal[1]) < 0.05
    if final_yaw is not None:
        assert abs((yaw - final_yaw + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(5)


def test_no_path_is_done():
    assert PathFollower().command(0, 0, 0).done
