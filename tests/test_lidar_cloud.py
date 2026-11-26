"""The lidar's point-cloud memory (PointCloudMap) and the lidar-only 3D view that draws it.

The map tests need numpy only. The view tests need PySide6 (no OpenGL: the view paints with QPainter).
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from simulatics.sim.cloud import PointCloudMap

V = 0.05


def centres(ix, iy, iz, voxel=V):
    """Points in the middle of the given voxels (never on a voxel boundary)."""
    g = np.stack(np.meshgrid(ix, iy, iz, indexing="ij"), -1).reshape(-1, 3)
    return ((g + 0.5) * voxel).astype(np.float32)


def wall(width=4.0, height=2.0, x=3.0, voxel=V):
    """A wall in the plane x = const, one point per voxel."""
    return centres([int(round(x / voxel))], np.arange(int(round(width / voxel))),
                   np.arange(int(round(height / voxel))), voxel)


# ---------------------------------------------------------------------------------------------- PointCloudMap
def test_empty_map():
    m = PointCloudMap()
    assert len(m) == 0 and m.seq == 0 and not m.full
    assert m.points.shape == (0, 3) and m.points.dtype == np.float32
    assert m.stamps.shape == (0,) and m.trajectory.shape == (0, 3)
    assert m.bounds() is None
    assert m.add(np.zeros((0, 3))) == 0 and m.seq == 0


def test_same_scan_twice_adds_nothing():
    m = PointCloudMap(V)
    scan = wall()
    assert m.add(scan, time=1.0) == len(scan)
    seq = m.seq
    assert m.add(scan, time=2.0) == 0
    assert len(m) == len(scan) and m.seq == seq
    assert np.all(m.stamps == 1.0)                       # first seen, not last seen


def test_points_in_one_voxel_collapse_to_the_first():
    m = PointCloudMap(V)
    rng = np.random.default_rng(0)
    pts = (np.array([0.20, -0.35, 1.00]) + rng.uniform(0.002, V - 0.002, (500, 3))).astype(np.float32)
    assert m.add(pts) == 1
    assert len(m) == 1
    np.testing.assert_array_equal(m.points[0], pts[0])   # the first hit is the one that is kept
    assert m.add(pts[::-1] ) == 0


def test_wall_counts_and_negative_coordinates():
    m = PointCloudMap(V)
    w = wall(4.0, 2.0)
    assert len(w) == 80 * 40
    dense = np.concatenate([w, w + 0.01, w - 0.01])      # three hits in every voxel
    assert m.add(dense) == 80 * 40
    mirrored = -w                                         # negative side: different voxels, same count
    assert m.add(mirrored) == 80 * 40
    assert len(m) == 2 * 80 * 40
    assert m.add(np.array([[np.nan, 0, 0], [np.inf, 0, 0], [1e9, 0, 0]])) == 0


def test_max_points_cap_and_full():
    m = PointCloudMap(V, max_points=1000)
    scan = centres(np.arange(40), np.arange(40), [0])    # 1600 voxels
    assert m.add(scan) == 1000
    assert len(m) == 1000 and m.full
    np.testing.assert_array_equal(m.points, scan[:1000])  # the earliest hits of the scan got in
    seq = m.seq
    assert m.add(centres(np.arange(40), np.arange(40), [5])) == 0
    assert len(m) == 1000 and m.seq == seq
    m.clear()
    assert not m.full and len(m) == 0


def test_trajectory_is_thinned():
    m = PointCloudMap(V)
    for k in range(101):                                  # 1 cm steps along x: 1 m in total
        m.add(np.zeros((0, 3)), origin=[0.01 * k, 0.0, 0.4])
    tr = m.trajectory
    assert tr.dtype == np.float32 and tr.shape[1] == 3
    assert 15 <= len(tr) <= 21
    assert np.all(np.linalg.norm(np.diff(tr, axis=0), axis=1) >= 0.05 - 1e-6)
    np.testing.assert_allclose(tr[0], [0, 0, 0.4], atol=1e-6)
    n = len(tr)
    m.add(np.zeros((0, 3)), origin=tr[-1] + [0.01, 0, 0])
    assert len(m.trajectory) == n


def test_seq_increments_only_on_change():
    m = PointCloudMap(V)
    w = wall()
    m.add(w, origin=[0, 0, 0.4])
    s = m.seq
    assert s > 0
    m.add(w, origin=[0, 0, 0.4])                          # nothing new at all
    m.add(w)
    assert m.seq == s
    m.add(w, origin=[1, 0, 0.4])                          # the sensor moved: the path changed
    assert m.seq == s + 1
    m.add(w + [1.0, 0, 0])                                # new voxels
    assert m.seq == s + 2
    m.clear()
    assert m.seq == s + 3
    m.clear()                                             # already empty
    assert m.seq == s + 3


def test_snapshots_are_frozen_and_never_change():
    m = PointCloudMap(V)
    m.add(wall(1.0, 1.0), origin=[0, 0, 0.4], time=1.0)
    snap, stamps, traj, seq = m.snapshot()
    before = snap.copy()
    with pytest.raises(ValueError):
        snap[0, 0] = 9.0
    for k in range(1, 60):                                # forces several reallocations
        m.add(wall(4.0, 2.0, x=3.0 + k), origin=[k, 0, 0.4], time=float(k))
    m.clear()
    m.add(wall(1.0, 1.0, x=-7.0), time=99.0)
    np.testing.assert_array_equal(snap, before)
    assert len(stamps) == len(snap) and np.all(stamps == 1.0) and len(traj) == 1 and seq == 1


def test_bounds_and_clear():
    m = PointCloudMap(V)
    a = centres([0, 10], [-4, 4], [0, 30])
    m.add(a, origin=[0, 0, 1])
    lo, hi = m.bounds()
    np.testing.assert_allclose(lo, a.min(axis=0))
    np.testing.assert_allclose(hi, a.max(axis=0))
    m.add(centres([100], [0], [0]))
    assert m.bounds()[1][0] == pytest.approx(100.5 * V)
    m.clear()
    assert len(m) == 0 and m.bounds() is None and len(m.trajectory) == 0 and m.points.shape == (0, 3)
    assert m.add(a) == len(a)                             # the voxels are free again


def test_npz_round_trip_is_exact(tmp_path):
    rng = np.random.default_rng(1)
    m = PointCloudMap(0.04, max_points=50_000)
    for k in range(20):
        m.add(rng.uniform(-6, 6, (1500, 3)), origin=rng.uniform(-1, 1, 3), time=0.1 * k)
    path = tmp_path / "cloud.npz"
    m.save(path)
    r = PointCloudMap.load(path)
    assert r.voxel == m.voxel and r.max_points == m.max_points and len(r) == len(m)
    np.testing.assert_array_equal(r.points, m.points)
    np.testing.assert_array_equal(r.stamps, m.stamps)
    np.testing.assert_array_equal(r.trajectory, m.trajectory)
    assert r.points.dtype == np.float32 and r.stamps.dtype == np.float32
    assert r.add(m.points) == 0                           # the loaded map knows its voxels
    assert r.add(m.points + [20.0, 0, 0]) > 0
    d = PointCloudMap.from_dict(m.to_dict())
    np.testing.assert_array_equal(d.points, m.points)


def test_ply_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    m = PointCloudMap(V)
    m.add(rng.uniform(-8, 8, (5000, 3)), origin=[0, 0, 0.5], time=3.0)
    path = tmp_path / "cloud.ply"
    m.save(str(path))
    lines = path.read_text().splitlines()
    assert lines[0] == "ply" and lines[1] == "format ascii 1.0"
    assert f"element vertex {len(m)}" in lines and "end_header" in lines
    r = PointCloudMap.load(path)
    assert len(r) == len(m) and r.voxel == m.voxel
    np.testing.assert_allclose(r.points, m.points, atol=1e-4)
    assert len(r.trajectory) == 0
    # a PLY from elsewhere: extra properties, another column order, a face element after the vertices
    other = tmp_path / "other.ply"
    other.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty uchar red\nproperty float x\n"
                     "property float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_index\n"
                     "end_header\n255 1.5 2.5 3.5\n0 -1 -2 -3\n3 0 1 0\n")
    np.testing.assert_allclose(PointCloudMap.load(other).points, [[1.5, 2.5, 3.5], [-1, -2, -3]])
    empty = tmp_path / "empty.ply"
    PointCloudMap().save(empty)
    assert len(PointCloudMap.load(empty)) == 0
    with pytest.raises(ValueError):
        m.save(tmp_path / "cloud.txt")
    binary = tmp_path / "binary.ply"
    binary.write_text("ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError):
        PointCloudMap.load(binary)


def test_assign_replaces_in_place():
    a, b = PointCloudMap(V), PointCloudMap(0.1)
    a.add(wall(), origin=[0, 0, 1])
    b.add(wall(2.0, 1.0, voxel=0.1), origin=[5, 5, 1], time=7.0)
    old, seq = a.points, a.seq
    a.assign(b)
    assert len(a) == len(b) and a.voxel == 0.1 and a.seq > seq
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.trajectory, b.trajectory)
    assert len(old) == 80 * 40                            # the earlier snapshot is untouched
    assert a.add(b.points) == 0


def filled(n, seed=0, max_points=400_000):
    """A map holding exactly n points spread through a 40 x 40 x 4 m volume."""
    rng = np.random.default_rng(seed)
    m = PointCloudMap(V, max_points=max_points)
    while len(m) < n:
        pts = rng.uniform([-20, -20, 0], [20, 20, 4], (50_000, 3))
        m.add(pts[:max(1, min(len(pts), n - len(m)))] if n - len(m) < 40_000 else pts)
    return m


def test_add_is_fast_on_a_big_map():
    rng = np.random.default_rng(3)
    scans = [rng.uniform([-20, -20, 0], [20, 20, 4], (1500, 3)).astype(np.float32) for _ in range(100)]
    best = 1e9
    for _ in range(3):                                    # best of three: the machine may be busy
        m = filled(200_000)
        n0 = len(m)
        assert n0 >= 200_000
        t = time.perf_counter()
        for k, s in enumerate(scans):
            m.add(s, origin=[0.1 * k, 0, 0.5], time=0.1 * k)
        best = min(best, time.perf_counter() - t)
        assert n0 < len(m) < m.max_points
        if best < 0.5:
            break
    print(f"\n100 scans of 1500 points into a 200k map: {best * 1000:.0f} ms ({best * 10:.2f} ms per add)")
    assert best < 2.0


def test_two_threads():
    m = PointCloudMap(V, max_points=150_000)
    rng = np.random.default_rng(4)
    scans = [rng.uniform([-15, -15, 0], [15, 15, 3], (1500, 3)).astype(np.float32) for _ in range(300)]
    errors, lengths, done = [], [], threading.Event()

    def writer():
        try:
            for k, s in enumerate(scans):
                m.add(s, origin=[0.06 * k, 0, 0.5], time=0.1 * k)
        except Exception as e:                            # pragma: no cover
            errors.append(e)
        finally:
            done.set()

    def reader():
        try:
            while not done.is_set() or not lengths:
                p = m.points
                n = len(p)
                if n:
                    assert np.isfinite(p).all() and p[n - 1].shape == (3,)
                    assert np.abs(p).max() <= 15.0 + 1e-3   # no half-written rows
                pts, stamps, traj, _ = m.snapshot()
                assert len(pts) == len(stamps)
                lengths.append(n)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    assert all(not t.is_alive() for t in threads)
    assert lengths == sorted(lengths)                     # the cloud only ever grows
    assert len(m) > 100_000 and len(m.trajectory) == 300


# ---------------------------------------------------------------------------------------------- the 3D view
@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    existing = QApplication.instance()
    if existing is None:
        try:                                              # the same OpenGL defaults as the Studio tests, in case
            from simulatics.studio.viewport import configure_gl   # they share this process (the view needs none)
            configure_gl()
        except Exception:
            pass
    return existing or QApplication(["test"])


def room(size=8.0, height=2.4, step=0.1):
    """A floor grid plus four walls."""
    a = np.arange(-size / 2, size / 2 + 1e-6, step)
    z = np.arange(0, height + 1e-6, step)
    fx, fy = np.meshgrid(a, a)
    floor = np.stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)], 1)
    wa, wz = np.meshgrid(a, z)
    walls = [np.stack([wa.ravel(), np.full(wa.size, s * size / 2), wz.ravel()], 1) for s in (-1, 1)] + \
            [np.stack([np.full(wa.size, s * size / 2), wa.ravel(), wz.ravel()], 1) for s in (-1, 1)]
    return np.concatenate([floor] + walls).astype(np.float32)


def pixels(img):
    from PySide6.QtGui import QImage
    img = img.convertToFormat(QImage.Format_RGB32)
    return np.frombuffer(img.constBits(), np.uint32).reshape(img.height(), img.width()).copy() & 0xFFFFFF


def shown_room(view):
    pts = room()
    stamps = np.linspace(0, 30, len(pts)).astype(np.float32)
    traj = np.stack([np.linspace(-2, 2, 50), np.zeros(50), np.full(50, 0.4)], 1).astype(np.float32)
    view.set_cloud(pts, stamps, traj, pts[::40] + [0, 0, 0.02], (2.0, 0.0, 0.5))
    return pts


def test_view_draws_the_room(app):
    from simulatics.studio import theme
    from simulatics.studio.panels_lidar import CloudView
    view = CloudView()
    view.resize(640, 480)
    empty = pixels(view.render_image(640, 480))
    shown_room(view)
    view.fit()
    img = view.render_image(640, 480)
    assert (img.width(), img.height()) == (640, 480)
    a = pixels(img)
    bg = int(theme.BG[1:], 16)
    assert np.count_nonzero(a != bg) > 5000               # not blank
    assert len(np.unique(a)) > 12                          # several colours
    assert np.count_nonzero(a != empty) > 5000
    assert view.last_drawn > 0.9 * len(room())             # fitted: (nearly) every point is on screen
    assert int(a[5, 320]) != 0xFF4FD8
    assert np.count_nonzero(a == 0xFF4FD8) > 20            # the latest scan, in its own colour

    view.az += 0.6                                          # orbit
    view.el -= 0.2
    b = pixels(view.render_image(640, 480))
    assert np.count_nonzero(a != b) > 3000
    for mode in ("distance", "age", "plain"):
        view.set_color_mode(mode)
        c = pixels(view.render_image(640, 480))
        assert np.count_nonzero(c != b) > 3000, mode
        b = c
    view.set_point_size(4)
    assert np.count_nonzero(pixels(view.render_image(640, 480)) != b) > 3000
    with pytest.raises(ValueError):
        view.set_color_mode("rainbow")


def test_view_handles_missing_data(app):
    from simulatics.studio.panels_lidar import COLOR_MODES, CloudView
    view = CloudView()
    view.resize(400, 300)
    for mode in COLOR_MODES:                               # no stamps, no path, no robot, no scan
        view.set_color_mode(mode)
        view.set_cloud(room(4.0, 1.0, 0.25), None, None, None, None)
        assert not view.render_image(400, 300).isNull()
        view.set_cloud(np.zeros((0, 3)), None, None, None, None)
        view.fit()
        assert not view.render_image(400, 300).isNull()
    view.set_cloud(None, None, None, None, None)
    view.dist, view.el = 0.3, 0.0                          # camera inside the cloud, level with the grid
    view.set_cloud(room(), None, room()[:5], None, (0.0, 0.0, 0.0))
    assert not view.render_image(400, 300).isNull()
    px, vis = view.project(room())
    assert px.shape == (len(room()), 2) and vis.dtype == bool and not vis.all() and vis.any()


def test_fit_frames_the_whole_cloud(app):
    from simulatics.studio.panels_lidar import CloudView
    view = CloudView()
    rng = np.random.default_rng(5)
    for (w, h), ext, az, el in (((900, 600), (12, 8, 2.4), -2.0, 0.5), ((300, 700), (30, 3, 1), 0.4, 1.2),
                                ((640, 480), (2, 2, 9), 2.5, -0.3), ((500, 500), (40, 40, 3), 1.0, 0.1)):
        view.resize(w, h)
        view.az, view.el = az, el
        pts = (rng.uniform(-0.5, 0.5, (3000, 3)) * ext + [5, -3, 1]).astype(np.float32)
        view.set_cloud(pts, None, None, None, None)
        view.fit()
        lo, hi = view.bounds()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        px, vis = view.project(corners)
        assert vis.all()
        assert np.all(px[:, 0] >= 0) and np.all(px[:, 0] <= view.width())
        assert np.all(px[:, 1] >= 0) and np.all(px[:, 1] <= view.height())
        spread = (px.max(axis=0) - px.min(axis=0)) / [view.width(), view.height()]
        assert spread.max() > 0.3                          # and it is not a dot in the distance
    # the camera turns the way the mouse would expect: +z is up on screen, and from +x looking in, +y is right
    view.resize(600, 400)
    view.target, view.az, view.el, view.dist = np.zeros(3), 0.0, 0.2, 10.0
    px, _ = view.project(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], float))
    assert px[1, 1] < px[0, 1] and px[2, 0] > px[0, 0]
    np.testing.assert_allclose(px[0], [300, 200], atol=1e-6)


def test_mouse_moves_the_camera(app):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from simulatics.studio.panels_lidar import CloudView
    view = CloudView()
    view.resize(600, 400)
    shown_room(view)
    view.fit()
    az, el, dist, target = view.az, view.el, view.dist, view.target.copy()
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, QPoint(300, 200))
    QTest.mouseMove(view, QPoint(360, 230))
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, QPoint(360, 230))
    assert view.az != az and view.el != el and np.allclose(view.target, target)
    QTest.mousePress(view, Qt.RightButton, Qt.NoModifier, QPoint(300, 200))
    QTest.mouseMove(view, QPoint(340, 180))
    QTest.mouseRelease(view, Qt.RightButton, Qt.NoModifier, QPoint(340, 180))
    assert not np.allclose(view.target, target) and view.dist == dist
    app.sendEvent(view, QWheelEvent(QPointF(300, 200), QPointF(300, 200), QPoint(0, 0), QPoint(0, 120),
                                    Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert view.dist < dist
    QTest.mouseDClick(view, Qt.LeftButton, Qt.NoModifier, QPoint(300, 200))
    lo, hi = view.bounds()
    np.testing.assert_allclose(view.target, (lo + hi) / 2)


def test_view_is_fast_enough(app):
    from simulatics.studio.panels_lidar import CloudView
    rng = np.random.default_rng(6)
    view = CloudView()
    view.resize(900, 600)
    pts = rng.uniform([-10, -10, 0], [10, 10, 3], (200_000, 3)).astype(np.float32)
    view.set_cloud(pts, None, pts[:50], pts[:1500], (0.0, 0.0, 0.0))
    view.fit()
    view.render_image(900, 600)                            # warm up: colour bins, layers
    best = 1e9
    for _ in range(5):
        view.az += 0.01                                    # a new camera each time: the cloud really is redrawn
        t = time.perf_counter()
        img = view.render_image(900, 600)
        best = min(best, time.perf_counter() - t)
    assert view.last_drawn > 150_000
    print(f"\nrender_image(900, 600) with 200k points: {best * 1000:.1f} ms")
    assert not img.isNull() and best < 0.25


def test_widget_paints_and_thins_while_dragging(app):
    from simulatics.studio.panels_lidar import DRAG_POINTS, CloudView
    rng = np.random.default_rng(8)
    n = 250_000
    pts = rng.uniform([-10, -10, 0], [10, 10, 3], (n, 3)).astype(np.float32)
    view = CloudView()
    view.resize(900, 600)
    view.show()
    app.processEvents()
    try:
        view.set_cloud(pts, None, pts[:50], pts[:1500], (0.0, 0.0, 0.0))
        view.fit()
        full = pixels(view.grab().toImage())                 # the real paintEvent
        assert view.last_drawn > 0.9 * n
        rendered = pixels(view.render_image(view.width(), view.height()))
        if full.shape == rendered.shape:                     # (a scaled display grabs a bigger image)
            assert np.count_nonzero(full != rendered) < 0.01 * full.size
        times = []
        for _ in range(5):
            view.az += 0.01
            t = time.perf_counter()
            view.grab()
            times.append(time.perf_counter() - t)
        view._dragging = True                                # as while a mouse button is held and moving
        view.az += 0.01
        light = pixels(view.grab().toImage())
        stride = -(-n // DRAG_POINTS)
        assert stride > 1 and 0.9 * n / stride < view.last_drawn <= n / stride + 1
        assert np.count_nonzero(light != light[0, 0]) > 5000
        drag = []
        for _ in range(5):
            view.az += 0.01
            t = time.perf_counter()
            view.grab()
            drag.append(time.perf_counter() - t)
        view._dragging = False
        view.grab()
        assert view.last_drawn > 0.9 * n
        t = time.perf_counter()                              # same camera, new scan only: the cloud layer is reused
        view.set_cloud(pts, None, pts[:50], pts[1500:3000], (0.1, 0.0, 0.0), seq=None)
        view._dirty = False
        view.grab()
        scan_only = time.perf_counter() - t
        print(f"\nwidget repaint with 250k points (screen scale {view.devicePixelRatioF():g}): "
              f"full {min(times) * 1000:.1f} ms, dragging {min(drag) * 1000:.1f} ms, "
              f"new scan only {scan_only * 1000:.1f} ms")
        assert min(drag) < min(times) and min(times) < 0.3
    finally:
        view.close()


# ---------------------------------------------------------------------------------------------- the panel
def fake_studio(n_scans=5):
    cloud = PointCloudMap(V)
    rng = np.random.default_rng(7)
    scan = None
    for k in range(n_scans):
        scan = rng.uniform([-5, -5, 0], [5, 5, 2], (1500, 3)).astype(np.float32)
        cloud.add(scan, origin=[0.1 * k, 0, 0.5], time=0.1 * k)
    qpos = np.zeros(19)
    qpos[:7] = [1.0, 2.0, 0.6, np.cos(0.4), 0, 0, np.sin(0.4)]         # yaw 0.8 rad
    sess = SimpleNamespace(cloud=cloud, cloud_on=True, lidar_points=scan, lock=threading.RLock(),
                           data=SimpleNamespace(qpos=qpos), task=object())
    messages = []
    return SimpleNamespace(sess=sess, log=messages.append, messages=messages)


def test_panel_refresh_clear_record(app):
    from simulatics.studio.panels_lidar import LidarCloudPanel
    studio = fake_studio()
    sess = studio.sess
    panel = LidarCloudPanel(studio)
    panel.resize(700, 500)
    panel.show()
    app.processEvents()
    try:
        assert panel.record.isChecked()
        panel.refresh()
        n = len(sess.cloud)
        assert len(panel.view._pts) == n
        assert f"{n:,} points" in panel.info.text() and "voxel 5 cm" in panel.info.text() and " m x " in panel.info.text()
        assert panel.view._robot == pytest.approx((1.0, 2.0, 0.8))
        panel.view.render_image(300, 200)

        seq = panel.view._seq                               # nothing changed: nothing is pushed
        pushed = panel.view._pts
        panel.refresh()
        assert panel.view._pts is pushed and panel.view._seq == seq
        sess.cloud.add(np.array([[40.0, 40.0, 1.0]]), time=9.0)
        panel._not_before = 0.0                              # (after a slow redraw the panel waits before the next)
        panel.refresh()
        assert len(panel.view._pts) == n + 1

        panel.follow.setChecked(True)                       # follow: the look-at point rides on the robot
        sess.data.qpos[:2] = [3.0, -1.0]
        panel._not_before = 0.0
        panel.refresh()
        np.testing.assert_allclose(panel.view.target[:2], [3.0, -1.0])
        assert panel.view._robot[:2] == pytest.approx((3.0, -1.0))
        panel.follow.setChecked(False)

        scan = sess.lidar_points + np.float32(0.5)           # a new scan alone (robot still, nothing new seen)
        sess.lidar_points = scan
        panel._not_before = 0.0
        panel.refresh()
        np.testing.assert_array_equal(panel.view._scan, scan)
        sess.cloud.add(np.array([[41.0, 40.0, 1.0]]))
        panel._not_before = time.monotonic() + 60            # still busy with the last redraw: not yet
        panel.refresh()
        assert len(panel.view._pts) == n + 1
        panel._not_before = 0.0
        panel.refresh()
        assert len(panel.view._pts) == n + 2

        panel.record.click()                                # Record is bound to sess.cloud_on, both ways
        assert sess.cloud_on is False
        panel.record.click()
        assert sess.cloud_on is True
        sess.cloud_on = False
        panel.refresh()
        assert not panel.record.isChecked()

        panel.color.setCurrentIndex(panel.color.findData("age"))
        assert panel.view.color_mode == "age"
        panel.size.setValue(3)
        assert panel.view.point_size == 3
        panel.fit_btn.click()

        panel.clear_btn.click()
        assert len(sess.cloud) == 0 and len(panel.view._pts) == 0
        assert panel.info.text().startswith("0 points")
        app.processEvents()
    finally:
        panel.close()


def test_panel_tolerates_a_bare_session(app):
    from simulatics.studio.panels_lidar import LidarCloudPanel
    for studio in (SimpleNamespace(), SimpleNamespace(sess=None), SimpleNamespace(sess=SimpleNamespace()),
                   SimpleNamespace(sess=SimpleNamespace(cloud=PointCloudMap(), task=None, lidar_points=None))):
        panel = LidarCloudPanel(studio)
        panel.refresh()                                     # hidden: nothing to do
        panel.refresh(force=True)
        panel.clear_btn.click()
        panel.save_btn.click()                              # nothing to save: no dialog
        panel.record.click()
        panel.follow.setChecked(True)
        panel.refresh(force=True)
        assert not panel.view.render_image(200, 150).isNull()


def test_panel_save_and_load(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from simulatics.studio.panels_lidar import LidarCloudPanel
    studio = fake_studio()
    cloud = studio.sess.cloud
    panel = LidarCloudPanel(studio)
    saved = cloud.points.copy()

    target = tmp_path / "seen.ply"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    panel.save_btn.click()
    assert target.exists()
    np.testing.assert_allclose(PointCloudMap.load(target).points, saved, atol=1e-4)
    assert any("saved" in m for m in studio.messages)

    bare = tmp_path / "no_extension"                         # no extension typed: a .ply is written
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(bare), "")))
    panel.save_btn.click()
    assert (tmp_path / "no_extension.ply").exists()

    exact = tmp_path / "seen.npz"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(exact), "")))
    panel.save_btn.click()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    panel.save_btn.click()                                   # cancelled

    cloud.clear()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(exact), "")))
    panel.load_btn.click()
    assert studio.sess.cloud is cloud                        # loaded in place: the simulation keeps its map
    np.testing.assert_array_equal(cloud.points, saved)
    assert len(cloud.trajectory) > 0 and len(panel.view._pts) == len(saved)

    broken = tmp_path / "broken.ply"
    broken.write_text("not a ply\n")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(broken), "")))
    panel.load_btn.click()
    assert len(cloud) == len(saved) and any("could not load" in m for m in studio.messages)
