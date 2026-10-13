"""Score logged forecasts and prices against results:  python -m src.live.score   (docs/market/methodology.md §4-§5)

Writes reports/live/market_tracking.md and data/live/logs/scored.csv. Run src.live.update first so results are current."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.config import ROOT
from src.gamemodel import model as gm
from src.live.odds import implied

log = logging.getLogger("live")
LOGS = ROOT / "data/live/logs"
BOOKS = ["FanDuel", "DraftKings", "BetMGM"]
EDGE = 0.03
OUTLIER = 0.08
MIN_GAMES, MIN_BETS = 300, 100


def scored_frame() -> pl.DataFrame:
    """One row per game: the last forecast logged before the start, the prices captured in that run, the market consensus, the paper bet, and the result if played."""
    f = pl.read_csv(LOGS / "forecast_log.csv", infer_schema_length=None).filter(pl.col("made_utc") < pl.col("start_utc").str.replace("Z", "+00:00")).sort("made_utc").unique("game_id", keep="last", maintain_order=True)
    o = pl.read_csv(LOGS / "odds_log.csv").filter(pl.col("market") == "moneyline").join(f.select("game_id", pl.col("made_utc").alias("captured_utc")), on=["game_id", "captured_utc"])
    imp = pl.col("american").map_elements(implied, return_dtype=pl.Float64)
    w = o.with_columns(imp.alias("q")).pivot(on="side", index=["game_id", "book"], values=["american", "q"]).drop_nulls()
    w = w.with_columns((pl.col("q_home") / (pl.col("q_home") + pl.col("q_away"))).alias("m_home"))
    # a book more than OUTLIER from the median of every listed book is a source error (wrong or reversed line), not a price: dropped
    w = w.with_columns((pl.col("m_home") - pl.col("m_home").median().over("game_id")).abs().alias("_dev")).filter((pl.col("_dev") <= OUTLIER) & pl.col("book").is_in(BOOKS))
    mk = w.group_by("game_id").agg(pl.col("m_home").mean().alias("mkt_home"), pl.len().alias("n_books"), pl.col("q_home").min().alias("be_home"), pl.col("q_away").min().alias("be_away"),      # lowest break-even = best price
                                   pl.col("american_home").max().alias("best_home"), pl.col("american_away").max().alias("best_away"))
    d = f.join(mk, on="game_id", how="left")
    eh, ea = pl.col("p_home") - pl.col("be_home"), pl.col("p_away") - pl.col("be_away")
    d = d.with_columns(pl.when(eh >= EDGE).then(pl.lit("home")).when(ea >= EDGE).then(pl.lit("away")).otherwise(None).alias("bet"), pl.max_horizontal(eh, ea).alias("edge"))
    res_path = ROOT / "data/live/processed/games.parquet"
    if res_path.exists():
        r = pl.read_parquet(res_path).select("game_id", (pl.col("home_final") > pl.col("away_final")).cast(pl.Float64).alias("home_win"), "home_final", "away_final", "last_period_type")
        d = d.join(r, on="game_id", how="left")
    else:
        d = d.with_columns(pl.lit(None, pl.Float64).alias("home_win"))
    price = pl.when(pl.col("bet") == "home").then(pl.col("best_home")).otherwise(pl.col("best_away"))
    won = ((pl.col("bet") == "home") & (pl.col("home_win") == 1)) | ((pl.col("bet") == "away") & (pl.col("home_win") == 0))
    pay = pl.when(price > 0).then(price / 100.0).otherwise(100.0 / -price)
    return d.with_columns(pl.when(pl.col("bet").is_null() | pl.col("home_win").is_null()).then(None).when(won).then(pay).otherwise(-1.0).alias("units"))


def summarize(d: pl.DataFrame) -> dict:
    s = d.filter(pl.col("home_win").is_not_null() & pl.col("mkt_home").is_not_null())
    out: dict = {"logged": d.height, "scored": s.height, "pending": d.filter(pl.col("home_win").is_null()).height}
    if s.height == 0:
        return out
    y, pm_, mk = s["home_win"].to_numpy(), s["p_home"].to_numpy(), s["mkt_home"].to_numpy()
    lm, lk = gm.logloss(pm_, y), gm.logloss(mk, y)
    diff = lk - lm
    out.update({"logloss_model": float(lm.mean()), "logloss_market": float(lk.mean()), "diff_market_minus_model": float(diff.mean()), "z": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) > 2 else None,
                "brier_model": float(((pm_ - y) ** 2).mean()), "brier_market": float(((mk - y) ** 2).mean()), "accuracy_model": float(((pm_ > 0.5) == (y > 0.5)).mean()), "accuracy_market": float(((mk > 0.5) == (y > 0.5)).mean())})
    b = s.filter(pl.col("units").is_not_null())
    out["paper_bets"] = b.height
    if b.height:
        u = b["units"].to_numpy()
        rng = np.random.default_rng(0)
        boot = [float(u[rng.integers(0, len(u), len(u))].mean()) for _ in range(2000)] if len(u) > 1 else [float(u.mean())]
        out.update({"units": float(u.sum()), "roi": float(u.mean()), "roi_lo95": float(np.quantile(boot, 0.025)), "roi_hi95": float(np.quantile(boot, 0.975))})
    if s.height >= MIN_GAMES:
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))  # noqa: E731
        X = np.column_stack([lg(mk), lg(pm_)])
        beta = gm.logit_fit(X, y)
        A = np.column_stack([np.ones(len(y)), X])
        p = 1 / (1 + np.exp(-(A @ beta)))
        se = np.sqrt(np.diag(np.linalg.inv(A.T @ (A * (p * (1 - p))[:, None]))))
        out["blend"] = {"coef_market": float(beta[1]), "coef_model": float(beta[2]), "z_model": float(beta[2] / se[2])}
    return out


def _dec(american: int) -> float:
    """Decimal odds for display."""
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def report(d: pl.DataFrame, s: dict) -> str:
    L = ["# Model against the betting market — running log", "", "Protocol and thresholds: [docs/market/methodology.md](../../docs/market/methodology.md). Morning prices (not closing lines) from FanDuel, DraftKings and BetMGM, margin removed; "
         "forecasts logged before each game. **Analysis of a model, not betting advice.**", "", f"Games logged: {s['logged']}. Scored: {s['scored']}. Pending: {s['pending']}.", ""]
    if s["scored"]:
        few = s["scored"] < MIN_GAMES
        L += [f"## Results so far{' (too few games for any claim: ' + str(s['scored']) + ' of ' + str(MIN_GAMES) + ')' if few else ''}", "", "| | model | market |", "|---|---|---|",
              f"| log loss | {s['logloss_model']:.4f} | {s['logloss_market']:.4f} |", f"| Brier | {s['brier_model']:.4f} | {s['brier_market']:.4f} |", f"| picked the winner | {s['accuracy_model']:.1%} | {s['accuracy_market']:.1%} |", "",
              f"Log-loss difference (market − model; positive favors the model): {s['diff_market_minus_model']:+.4f}" + (f", z = {s['z']:.2f}." if s.get("z") is not None else "."), ""]
        if s.get("paper_bets"):
            L += [f"Paper bets (model at least {EDGE:.0%} above break-even at the best of the three books): {s['paper_bets']} bets, {s['units']:+.2f} units, return {s['roi']:+.1%} per unit (95% interval {s['roi_lo95']:+.1%} to {s['roi_hi95']:+.1%})"
                  + ("; too few bets for any claim." if s["paper_bets"] < MIN_BETS else "."), ""]
        if "blend" in s:
            L += [f"Blend: outcome on market logit ({s['blend']['coef_market']:.2f}) and model logit ({s['blend']['coef_model']:.2f}, z = {s['blend']['z_model']:.2f}).", ""]
        done = d.filter(pl.col("home_win").is_not_null()).sort("game_date", "game_id", descending=True).head(40)
        L += ["## Scored games (latest first)", "", "| date | game | result | model home | market home | paper bet | units |", "|---|---|---|---|---|---|---|"]
        for r in done.iter_rows(named=True):
            mk = f"{r['mkt_home']:.1%}" if r["mkt_home"] is not None else "n/a"
            L.append(f"| {r['game_date']} | {r['away']} at {r['home']} | {r['away_final']}–{r['home_final']} {r['last_period_type']} | {r['p_home']:.1%} | {mk} | {r['bet'] or ''} | {'' if r['units'] is None else format(r['units'], '+.2f')} |")
        L.append("")
    pend = d.filter(pl.col("home_win").is_null()).sort("start_utc")
    if pend.height:
        L += ["## Pending games", "", "| date | game | model home | market home | difference | best decimal odds (away / home) | paper bet | goalies assumed (away / home) |", "|---|---|---|---|---|---|---|---|"]
        for r in pend.iter_rows(named=True):
            if r["mkt_home"] is None:
                L.append(f"| {r['game_date']} | {r['away']} at {r['home']} | {r['p_home']:.1%} | n/a | | | | {r['away_goalie']} / {r['home_goalie']} |")
            else:
                L.append(f"| {r['game_date']} | {r['away']} at {r['home']} | {r['p_home']:.1%} | {r['mkt_home']:.1%} | {r['p_home'] - r['mkt_home']:+.1%} | {_dec(r['best_away']):.2f} / {_dec(r['best_home']):.2f} | {r['bet'] or ''} | {r['away_goalie']} / {r['home_goalie']} |")
        L.append("")
    return "\n".join(L)


def run() -> dict:
    d = scored_frame()
    s = summarize(d)
    d.write_csv(LOGS / "scored.csv")
    out = ROOT / "reports/live"
    out.mkdir(parents=True, exist_ok=True)
    (out / "market_tracking.md").write_text(report(d, s), encoding="utf-8")
    log.info("scored %d of %d logged games; %d pending", s["scored"], s["logged"], s["pending"])
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()
