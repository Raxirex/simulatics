# PyInstaller spec for Simulatics Studio. Build with:  scripts\build_exe.ps1
# (or: python -m PyInstaller packaging/simulatics_studio.spec --noconfirm)
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
datas, binaries, hidden = [], [], []
for pkg in ("mujoco", "glfw", "manifold3d"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h
hidden += collect_submodules("simulatics")

datas += [(str(p), "simulatics/imitate/motions") for p in (ROOT / "simulatics/imitate/motions").glob("*.npz")]
datas += [(str(p), "simulatics/samples/data") for p in (ROOT / "simulatics/samples/data").glob("*") if p.is_file()]
datas += [(str(p), "simulatics/server/static") for p in (ROOT / "simulatics/server/static").glob("*")]
datas += [(str(p), "simulatics/native/bin") for p in (ROOT / "simulatics/native/bin").glob("*.dll")]
for run in ("pretrained_walker", "rl_walker", "imitate_walk"):          # the policies tracked in git
    datas += [(str(p), f"bundled_runs/{run}") for p in (ROOT / "runs" / run).glob("*") if p.suffix in (".npz", ".json")]
datas += [(str(ROOT / "LICENSE"), "."), (str(ROOT / "THIRD_PARTY_NOTICES.md"), ".")]

a = Analysis(
    [str(ROOT / "packaging" / "studio_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries, datas=datas, hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "notebook", "PyQt5", "PyQt6", "PySide2", "torch"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="SimulaticsStudio",
    console=False,            # errors go to Documents\Simulatics\studio.log
    upx=False,
    runtime_tmpdir=None,
)
