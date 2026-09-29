"""End-to-end PM-EPA pipeline.

    python -m src.pipeline                       # everything after ingestion (uses cached raw data)
    python -m src.pipeline --ingest              # also (re)fetch missing raw files
    python -m src.pipeline --from value          # resume from a step
    python -m src.pipeline --only epa metrics    # run selected steps
    python -m src.pipeline --smoke               # tiny end-to-end run on a few games (CI / quickstart)

Steps (in order): ingest, parse, onice, possessions, xg, value, epa, metrics, validate,
ablations, examples, figures, report, defense, finishing, components, specialteams, goaltending, valuation, calibration, wholemodel. Every step reads/writes Parquet under data/ and is
deterministic given config.yaml (seeded).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from datetime import UTC, datetime

from src.config import ROOT, load_config

log = logging.getLogger("pipeline")

STEPS = ["ingest", "parse", "onice", "possessions", "xg", "value", "epa", "metrics", "validate", "ablations", "examples", "figures", "report",
         "defense", "finishing", "components", "specialteams", "goaltending", "valuation", "calibration", "wholemodel", "projection"]


def step(name: str) -> None:
    t0 = time.time()
    log.info("=== %s ===", name)
    if name == "ingest":
        from src.ingestion.nhl_api import ingest
        ingest(load_config().data.seasons)
    elif name == "parse":
        from src.ingestion.parse import run
        run()
    elif name == "onice":
        from src.possession.onice import run
        run()
    elif name == "possessions":
        from src.possession.build import run
        run()
        run(hits=True)
    elif name == "xg":
        from src.xg.model import run
        run()
    elif name == "value":
        from src.value.fit import run
        run()
    elif name == "epa":
        from src.epa.run import identity_check, run
        run()
        run(lam=1.0, suffix="_lam1")  # risk-adjusted companion: opponent value after turnovers
        (load_config().path(load_config().data.metrics_dir) / "identity_check.json").write_text(json.dumps(identity_check(), indent=2))
    elif name == "metrics":
        from src.metrics.aggregate import run
        run()
    elif name == "validate":
        from src.validation.validate import run
        run()
    elif name == "ablations":
        from src.validation.ablations import run
        run()
    elif name == "examples":
        from src.epa.examples import run
        run()
    elif name == "figures":
        from src.viz.figures import run
        run()
    elif name == "report":
        from src.validation.report import run
        run()
    elif name == "defense":  # Defensive PM-EPA (docs/defense/methodology.md); reads the lambda = 0 credits
        from src.defense.run import STEPS as DEF_STEPS
        from src.defense.run import run_steps
        run_steps(DEF_STEPS)
    elif name == "finishing":  # Finishing talent (docs/finishing/methodology.md); needs the epa/metrics outputs
        from src.finishing.run import STEPS as FIN_STEPS
        from src.finishing.run import run_steps as run_fin
        run_fin(FIN_STEPS)
    elif name == "components":  # turnover cost, faceoffs, penalties at 5v5 (docs/components/methodology.md)
        from src.components.run import STEPS as COMP_STEPS
        from src.components.run import run_steps as run_comp
        run_comp(COMP_STEPS)
    elif name == "specialteams":  # power play, penalty kill, 4v4, 3v3 (docs/specialteams/methodology.md)
        from src.specialteams.run import STEPS as ST_STEPS
        from src.specialteams.run import run_steps as run_st
        run_st(ST_STEPS)
    elif name == "goaltending":  # goalie component (docs/goaltending/methodology.md); needs the finishing step
        from src.goaltending.run import STEPS as GK_STEPS
        from src.goaltending.run import run_steps as run_gk
        run_gk(GK_STEPS)
    elif name == "valuation":  # replacement level, goals per win, combined component table
        from src.valuation.run import STEPS as VAL_STEPS
        from src.valuation.run import run_steps as run_val
        run_val(VAL_STEPS)
    elif name == "calibration":  # joint calibration of component scales (docs/calibration/methodology.md)
        from src.calibration.run import STEPS as CAL_STEPS
        from src.calibration.run import run_steps as run_cal
        run_cal(CAL_STEPS)
    elif name == "wholemodel":  # whole-model validation (docs/wholemodel/methodology.md)
        from src.wholemodel.run import run as run_wm
        run_wm()
    elif name == "projection":  # rolling refits (finishing, faceoffs), then the projection and aging layer (docs/projection/methodology.md)
        from src.wholemodel import rolling
        from src.wholemodel.run import run as run_wm
        rolling.roll_fo(load_config(), rolling.FOLDS)
        rolling.roll_fin(load_config(), rolling.FOLDS)
        rolling.combine(load_config())
        run_wm()
        from src.projection.run import run as run_proj
        run_proj()
    log.info("%s done in %.0fs", name, time.time() - t0)


def run_log(steps: list[str]) -> None:
    cfg = load_config()
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = None
    meta = {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": sha, "python": platform.python_version(),
            "steps": steps, "config": cfg.model_dump()}
    p = cfg.path(cfg.data.metrics_dir) / "run_log.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--from", dest="start", choices=STEPS)
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        from src.smoke import run as smoke
        smoke()
        return
    steps = args.only or STEPS[STEPS.index(args.start):] if (args.only or args.start) else STEPS[1:]
    if args.ingest and "ingest" not in steps:
        steps = ["ingest"] + steps
    run_log(steps)
    for s in steps:
        step(s)


if __name__ == "__main__":
    main()

