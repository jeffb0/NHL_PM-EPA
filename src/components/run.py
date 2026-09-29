"""Turnover, faceoff and penalty components:  python -m src.components.run [--only STEP ...]

Steps: turnover -> turnover_ra -> faceoff -> penalty -> report. The integrated table is produced by ``python -m src.valuation.run``.
"""

from __future__ import annotations

import argparse
import logging

STEPS = ["turnover", "turnover_ra", "faceoff", "penalty", "report"]
log = logging.getLogger("components")


def run_steps(steps: list[str]) -> None:
    for s in steps:
        log.info("=== components:%s ===", s)
        if s == "turnover":
            from src.components.turnover import run
        elif s == "turnover_ra":
            from src.components.turnover_ra import run
        elif s == "faceoff":
            from src.components.faceoff import run
        elif s == "penalty":
            from src.components.penalty import run
        else:
            from src.components.report import run
        run()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args()
    run_steps(args.only or STEPS)


if __name__ == "__main__":
    main()
