"""Where the live pipeline keeps its files (docs/market/methodology.md §8 item 14).

Logs and reports that contain third-party data (sportsbook prices, the lineup and injury feed) live in data/private, which is its own private git repository and is ignored by this one.
The public record lives in data/live/public and reports/live: the picks ledger in units and decimal odds, the market test's summary, and a fingerprint (SHA-256) of every private log at each run,
so anyone can later check that a private log shown to them is the one that existed when the fingerprint was committed."""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime

from src.config import ROOT

PRIVATE = ROOT / "data/private"
LOGS = PRIVATE / "logs"
PRIVATE_REPORTS = PRIVATE / "reports"
PUBLIC = ROOT / "data/live/public"
PUBLIC_REPORTS = ROOT / "reports/live"
FINGERPRINTS = PUBLIC / "fingerprints.csv"
FP_COLS = ["recorded_utc", "run", "file", "sha256", "bytes", "rows"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(run: str, logs=None, out=None) -> list[dict]:
    """Append one row per private log (hash, size, data rows) to the public fingerprint file."""
    logs, out = logs or LOGS, out or FINGERPRINTS
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for p in sorted(logs.glob("*.csv")):
        with open(p, encoding="utf-8") as fh:
            n = max(sum(1 for _ in fh) - 1, 0)
        rows.append({"recorded_utc": now, "run": run, "file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size, "rows": n})
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FP_COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return rows
