"""Robot library: third-party robot models fetched on demand from MuJoCo Menagerie.

Models are not stored in this repository (Adam Lite alone is 175 MB of meshes). `fetch()` downloads one
model folder, including its own LICENSE file, from a pinned Menagerie commit into ``robots/<name>/``
(git-ignored), using a sparse git checkout.

    python -m simulatics.robots fetch go2 adam_lite
    python -m simulatics.robots list

Licences (from each model's LICENSE file in Menagerie):
  adam_lite  PNDbotics Adam Lite humanoid   MIT           Copyright (c) 2023 pndrobotics
  go2        Unitree Go2 quadruped          BSD-3-Clause  Copyright (c) 2016-2022 Unitree Robotics
Menagerie has no MIT-licensed quadruped; BSD-3-Clause is the most permissive one available.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"  # pinned for reproducibility
ROBOTS_DIR = Path("robots")

LIBRARY: dict[str, dict] = {
    "adam_lite": {"label": "PNDbotics Adam Lite (humanoid)", "folder": "pndbotics_adam_lite", "scene": "scene.xml",
                  "license": "MIT", "kind": "humanoid", "size_mb": 175},
    "go2": {"label": "Unitree Go2 (robot dog)", "folder": "unitree_go2", "scene": "scene.xml",
            "license": "BSD-3-Clause", "kind": "quadruped", "size_mb": 31},
}


def path(name: str) -> Path:
    """Scene file of a downloaded model (may not exist yet)."""
    info = LIBRARY[name]
    return ROBOTS_DIR / name / info["scene"]


def available(name: str) -> bool:
    return name in LIBRARY and path(name).is_file()


def status() -> list[dict]:
    return [{"name": k, **v, "downloaded": available(k), "scene_path": str(path(k).resolve())} for k, v in LIBRARY.items()]


def fetch(name: str, quiet: bool = True) -> Path:
    """Download one model folder from the pinned Menagerie commit. Returns its scene path."""
    if name not in LIBRARY:
        raise ValueError(f"unknown robot {name!r}; choose from {list(LIBRARY)}")
    if available(name):
        return path(name)
    if shutil.which("git") is None:
        raise RuntimeError("git is needed to download robot models")
    folder = LIBRARY[name]["folder"]
    with tempfile.TemporaryDirectory() as tmp:
        q = ["-q"] if quiet else []
        steps = [["git", "init", *q, tmp],
                 ["git", "-C", tmp, "remote", "add", "origin", MENAGERIE_URL],
                 ["git", "-C", tmp, "sparse-checkout", "set", folder],
                 ["git", "-C", tmp, "fetch", *q, "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT],
                 ["git", "-C", tmp, "checkout", *q, "FETCH_HEAD"]]
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"download failed at `{' '.join(cmd[:4])}`: {(r.stderr or r.stdout).strip()[-300:]}")
        dest = ROBOTS_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(Path(tmp) / folder, dest)
    (dest / "SOURCE.txt").write_text(f"{MENAGERIE_URL} @ {MENAGERIE_COMMIT} / {folder}\n"
                                     f"Licence: {LIBRARY[name]['license']} (see LICENSE in this folder)\n")
    return path(name)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m simulatics.robots")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download robot models")
    f.add_argument("names", nargs="+", choices=list(LIBRARY))
    sub.add_parser("list", help="show the library")
    a = ap.parse_args()
    if a.cmd == "fetch":
        for n in a.names:
            print(n, "->", fetch(n, quiet=False))
    else:
        for r in status():
            print(f"{r['name']:10s} {r['license']:13s} {'downloaded' if r['downloaded'] else 'not downloaded':15s} {r['label']}")


if __name__ == "__main__":
    main()
