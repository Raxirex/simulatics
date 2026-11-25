"""Start Simulatics from a checkout with one command. Sets everything up the first time.

    python run.py            the desktop app (Simulatics Studio), from this source
    python run.py web        the browser GUI instead
    python run.py exe        download the latest released SimulaticsStudio.exe and run it (no Python packages needed)
    python run.py update     git pull, then start the desktop app

On Windows you can also double-click run.bat.

The first start creates .venv next to this file and installs the dependencies into it (a few minutes, about
1 GB). Later starts go straight to the app.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REPO = "Raxirex/simulatics"
NEEDED = ("mujoco", "numpy", "scipy", "PySide6")


def say(text: str) -> None:
    print(f"[simulatics] {text}", flush=True)


def ensure_env() -> None:
    """A virtual environment with the app's dependencies, created or completed as needed."""
    if sys.version_info < (3, 10):
        raise SystemExit("Simulatics needs Python 3.10 or newer (https://www.python.org/downloads/).")
    if not PY.exists():
        say("first start: creating .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    check = "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(m) is None for m in %r))" % (NEEDED,)
    editable = "import simulatics, pathlib, sys; sys.exit(pathlib.Path(simulatics.__file__).resolve().parent.parent != pathlib.Path(%r))" % str(ROOT)
    missing = subprocess.run([str(PY), "-c", check], cwd=ROOT).returncode != 0
    linked = subprocess.run([str(PY), "-c", editable], cwd=ROOT, capture_output=True).returncode == 0
    if missing or not linked:
        say("installing dependencies (this takes a few minutes the first time) ...")
        subprocess.run([str(PY), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], cwd=ROOT)
        r = subprocess.run([str(PY), "-m", "pip", "install", "--quiet", "-e", ".[studio]"], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit("installing the dependencies failed (see the messages above).")


def start(*args: str) -> int:
    ensure_env()
    say("starting: simulatics " + " ".join(args))
    return subprocess.run([str(PY), "-m", "simulatics", *args], cwd=ROOT).returncode


def run_exe() -> int:
    """Download the latest release's exe next to this file (once per version) and run it."""
    if os.name != "nt":
        raise SystemExit("the packaged exe is for Windows; use  python run.py  on this system.")
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest", headers={"User-Agent": "simulatics-run"})
    with urllib.request.urlopen(req, timeout=30) as r:          # noqa: S310 - fixed GitHub API URL
        rel = json.load(r)
    asset = next((a for a in rel.get("assets", []) if a["name"] == "SimulaticsStudio.exe"), None)
    if asset is None:
        raise SystemExit(f"release {rel.get('tag_name')} has no SimulaticsStudio.exe attached.")
    exe, mark = ROOT / "SimulaticsStudio.exe", ROOT / "SimulaticsStudio.version"
    if not exe.exists() or not mark.exists() or mark.read_text().strip() != rel["tag_name"]:
        say(f"downloading SimulaticsStudio.exe {rel['tag_name']} ({asset['size'] / 1e6:.0f} MB) ...")
        tmp = exe.with_suffix(".part")
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as r, open(tmp, "wb") as f:   # noqa: S310
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r[simulatics] {done / 1e6:5.0f} / {asset['size'] / 1e6:.0f} MB", end="", flush=True)
        print()
        if tmp.stat().st_size != asset["size"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit("the download was incomplete; try again.")
        tmp.replace(exe)
        mark.write_text(rel["tag_name"])
    say(f"starting SimulaticsStudio.exe {rel['tag_name']} (the first seconds are spent unpacking)")
    return subprocess.run([str(exe)], cwd=ROOT).returncode


def main(argv: list[str]) -> int:
    what = argv[0] if argv else "studio"
    if what in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if what == "exe":
        return run_exe()
    if what == "update":
        say("updating the checkout ...")
        if subprocess.run(["git", "pull", "--ff-only"], cwd=ROOT).returncode != 0:
            raise SystemExit("git pull failed (local changes, or no connection).")
        return start("studio")
    if what == "web":
        return start("gui", *argv[1:])
    if what == "studio":
        return start("studio", *argv[1:])
    return start(*argv)                    # anything else goes straight to the simulatics command line


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
