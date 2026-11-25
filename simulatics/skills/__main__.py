"""python -m simulatics.skills <skill> [--generations N] [--population N] [--variations N] [--name run] [--seed N]"""
from __future__ import annotations

import argparse
import json

from . import SKILLS, SkillTrainer, TrainConfig


def main() -> None:
    p = argparse.ArgumentParser(prog="simulatics.skills", description="Train a skill by parallel search in physics.")
    p.add_argument("skill", choices=sorted(SKILLS))
    p.add_argument("--generations", type=int, default=120)
    p.add_argument("--population", type=int, default=256)
    p.add_argument("--variations", type=int, default=3)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    a = p.parse_args()
    cfg = TrainConfig(population=a.population, variations=a.variations, generations=a.generations, threads=a.threads,
                      seed=a.seed, name=a.name or f"skill_{a.skill}")
    tr = SkillTrainer(SKILLS[a.skill], cfg)
    tape = tr.run(log=print)
    print(json.dumps(tape.meta["report"], indent=1))
    print(f"saved {tr.tape_path}  ({tape.meta['physics_steps'] / 1e6:.1f} million physics steps)")


if __name__ == "__main__":
    main()
