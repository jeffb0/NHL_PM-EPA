"""Typed access to config.yaml.

All tunables live in ``config.yaml`` at the repo root. ``load_config`` validates the file
with pydantic so a typo fails loudly instead of silently falling back to a default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataCfg(_Strict):
    seasons: list[int]
    game_type: int = 2
    scoring_regimes: list[list[int]]
    raw_dir: str
    processed_dir: str
    value_dir: str
    output_dir: str
    figures_dir: str
    metrics_dir: str = "reports/metrics"
    ingest_workers: int = 6
    request_timeout_s: int = 30


class SplitCfg(_Strict):
    train: list[int]
    validation: list[int]
    test: list[int]
    cross_fit: bool = True


class PossessionCfg(_Strict):
    rebound_window_s: float = 3
    carry_max_gap_s: float = 12
    max_inferred_start_gap_s: float = 25
    max_segment_gap_s: float = 30
    hits_as_observations: bool = False


class StatesCfg(_Strict):
    grid: Literal["zones", "grid10", "grid20"] = "zones"
    candidate_grids: list[str]
    score_buckets: list[int]
    use_score_in_state: bool = False
    strengths: list[str]


class ValueCfg(_Strict):
    method: Literal["markov", "mc", "gbm", "hazard"] = "markov"
    min_n_per_state: int = 100
    shrinkage_k: float | Literal["auto"] = "auto"
    spatial_smoothing_bandwidth_ft: float = 0
    monotonicity_check: bool = True
    lambda_opponent: float = 1.0
    crossfit_within_regime: bool = True


class EpaCfg(_Strict):
    start_share: float = 0.35
    end_share: float = 0.35
    onice_pool: bool = True
    credit_rebounds: bool = False
    include_turnovers: bool = True
    exclude_empty_net: bool = True


class MetricsCfg(_Strict):
    min_toi_ev: float = 500
    min_toi_pp: float = 200
    min_toi_pk: float = 200
    bootstrap_reps: int = 1000
    ci_level: float = 0.95


class XgCfg(_Strict):
    model: Literal["logit", "gbm"] = "logit"
    calibrate: Literal["isotonic", "none"] = "isotonic"
    rush_window_s: float = 4


class Config(_Strict):
    seed: int
    data: DataCfg
    splits: SplitCfg
    possession: PossessionCfg
    states: StatesCfg
    value: ValueCfg
    epa: EpaCfg
    metrics: MetricsCfg
    xg: XgCfg

    def path(self, rel: str) -> Path:
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True) if not p.suffix else p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml (or the file named by $PM_EPA_CONFIG, used by the smoke test)."""
    path = Path(path) if path else Path(os.environ.get("PM_EPA_CONFIG", ROOT / "config.yaml"))
    with open(path) as fh:
        return Config.model_validate(yaml.safe_load(fh))


def season_id(start_year: int) -> int:
    """2023 -> 20232024 (NHL API season identifier)."""
    return start_year * 10000 + start_year + 1
