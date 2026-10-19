import csv
import hashlib

import polars as pl

from src.live import paths
from src.live.picks import PUBLIC_COLS, public_record


def test_fingerprint_appends_hash_size_and_rows(tmp_path):
    logs, out = tmp_path / "logs", tmp_path / "pub" / "fp.csv"
    logs.mkdir()
    (logs / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    paths.fingerprint("daily", logs, out)
    paths.fingerprint("afternoon", logs, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["run"] for r in rows] == ["daily", "afternoon"] and rows[0]["rows"] == "2" and rows[0]["bytes"] == str((logs / "a.csv").stat().st_size)
    assert rows[0]["sha256"] == hashlib.sha256((logs / "a.csv").read_bytes()).hexdigest()


def test_private_and_public_locations():
    assert paths.LOGS.is_relative_to(paths.PRIVATE) and paths.PRIVATE_REPORTS.is_relative_to(paths.PRIVATE) and not paths.PUBLIC.is_relative_to(paths.PRIVATE)


def test_public_record_drops_market_and_sizing_columns():
    row = {c: [None] for c in PUBLIC_COLS} | {"p_market": [0.5], "p_sizing": [0.55], "american": [-125], "edge_vs_breakeven": [0.05], "forecast_made_utc": ["x"]}
    out = public_record(pl.DataFrame(row))
    assert out.columns == PUBLIC_COLS and "p_market" not in out.columns and "american" not in out.columns
