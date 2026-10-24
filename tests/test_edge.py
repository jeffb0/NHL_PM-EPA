import gzip
import json

import polars as pl

from src.edge.run import Z, with_edge_features
from src.ingestion.edge import parse


def _write(raw, pid, season, payload):
    path = raw / "edge" / str(season) / f"{pid}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps(payload).encode())


def test_parse_reads_the_fields_and_skips_missing_markers(tmp_path):
    payload = {
        "player": {"gamesPlayed": 70},
        "skatingSpeed": {
            "speedMax": {"imperial": 23.5, "percentile": 0.98},
            "burstsOver20": {"value": 300, "percentile": 0.9},
        },
        "topShotSpeed": {"imperial": 90.0, "percentile": 0.7},
        "totalDistanceSkated": {"imperial": 200.0, "percentile": 0.8},
        "distanceMaxGame": {"imperial": 4.0},
        "zoneTimeDetails": {"offensiveZonePctg": 0.45, "defensiveZonePctg": 0.35, "offensiveZoneEvPctg": 0.44},
    }
    _write(tmp_path, 1, 2022, payload)
    _write(tmp_path, 2, 2022, {"missing": True})
    out = parse(tmp_path, [(1, 2022), (2, 2022), (3, 2022)])
    assert (
        out.height == 1
        and out["max_speed_mph"][0] == 23.5
        and out["bursts_over20"][0] == 300
        and out["edge_gp"][0] == 70
    )


def test_features_are_the_previous_seasons_and_missing_players_get_a_flag():
    d = pl.DataFrame({"player_id": [1], "season": [2022], **{c: [1.5] for c in Z.values()}})
    p = pl.DataFrame({"player_id": [1, 1, 2], "season": [2023, 2024, 2023]})
    x = with_edge_features(p, d).sort("player_id", "season")
    # season 2023 uses the 2022 features; 2024 and player 2 have none
    assert x["edge_missing"].to_list() == [0.0, 1.0, 1.0]
    assert x[Z["max_speed"]].to_list() == [1.5, 0.0, 0.0]
