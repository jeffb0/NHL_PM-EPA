"""All report figures (PNG + SVG) and the V(s) monotonicity diagnostic.

Figures -> reports/figures/<group>/ ; monotonicity -> reports/metrics/monotonicity.json
"""

from __future__ import annotations

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from src.config import load_config
from src.viz.style import POS_COLOR, SEQ, SERIES, TEXT2, draw_rink, save

log = logging.getLogger(__name__)
SG_LABEL = {"5v5": "5v5", "PP": "Power play", "PK": "Penalty kill", "4v4": "4v4 / 3v3"}


def grid_frame(strength: str, kind: str = "takeaway", step: float = 2.0) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    xs = np.arange(-99, 100, step)
    ys = np.arange(-42, 42.5, step)
    X, Y = np.meshgrid(xs, ys)
    f = pl.DataFrame({"obs_kind": kind, "x": X.ravel(), "y": Y.ravel(), "strength": strength, "score_diff": 0})
    return f, X, Y


def v_surface_figures(
    model,
    out,
    vmax: float | None = None,
    tag: str = "v_start",
    kind: str = "takeaway",
    label: str = "Value of gaining the puck (V_start)",
    note: str = "Estimated from possession starts (alive observations)",
) -> dict:
    """2D rink heatmaps of V(kind, x, y) per strength + OZ zoom; returns surfaces for reuse."""
    surfaces = {}
    for sg in ["5v5", "PP", "PK", "4v4"]:
        f, X, Y = grid_frame(sg, kind)
        surfaces[sg] = (X, Y, model.predict(f).reshape(X.shape))
    vmax = vmax or float(np.quantile(surfaces["PP"][2], 0.995))
    for sg, (X, Y, V) in surfaces.items():
        for half in [None, "offense"]:
            fig, ax = plt.subplots(figsize=(9 if half is None else 5.2, 4.4))
            im = ax.pcolormesh(X, Y, V, cmap=SEQ, vmin=0, vmax=vmax, shading="auto", zorder=1)
            draw_rink(ax, half)
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label("expected possession xG", color=TEXT2)
            ax.set_title(f"{label} — {SG_LABEL[sg]}" + (" (attacking zone)" if half else ""))
            ax.text(
                0.0,
                -0.04,
                f"Attacking direction → (+x). {note}; colour scale shared across strengths.",
                transform=ax.transAxes,
                color=TEXT2,
                fontsize=8,
            )
            save(fig, out, f"{tag}_{sg.lower()}_{'full' if half is None else 'oz'}")
    # neutral-zone and defensive detail for 5v5 with its own scale (the OZ dominates the shared scale)
    X, Y, V = surfaces["5v5"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cols = X[0] < 30  # defensive + neutral zone columns only (own colour scale)
    Vs = V[:, cols]
    im = ax.pcolormesh(
        X[:, cols],
        Y[:, cols],
        Vs,
        cmap=SEQ,
        vmin=float(Vs.min()),
        vmax=float(np.quantile(Vs, 0.99)),
        shading="auto",
        zorder=1,
    )
    draw_rink(ax)
    ax.set_xlim(-101, 30)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("V(s)", color=TEXT2)
    ax.set_title(f"{label} — 5v5, defensive and neutral zones (own scale)")
    save(fig, out, f"{tag}_5v5_dz_nz")
    return surfaces


def radial_monotonicity(model, kind: str, tol: float = 0.002) -> dict:
    """Primary check: along rays into the net centre (89, 0) at fixed angle, V should not fall by
    more than ``tol`` as distance shrinks (2-ft steps, 2-60 ft, angles -75..75 deg). This is the
    brief's "value increases toward the net" holding shooting angle fixed; lanes at fixed y are
    not used because approaching the goal line along a wide lane narrows the angle (a real drop)."""
    d = np.arange(2, 62, 2.0)
    th = np.radians(np.arange(-75, 76, 5.0))
    D, TH = np.meshgrid(d, th)  # rows: angle, cols: distance (increasing)
    x, y = 89 - D * np.cos(TH), D * np.sin(TH)
    out = {}
    for sg in ["5v5", "PP", "PK", "4v4"]:
        f = pl.DataFrame({"obs_kind": kind, "x": x.ravel(), "y": y.ravel(), "strength": sg, "score_diff": 0})
        V = model.predict(f).reshape(D.shape)
        step = V[:, :-1] - V[:, 1:]  # V(closer) - V(farther): should be >= -tol
        viol = step < -tol
        out[sg] = {
            "ray_steps": int(step.size),
            "violations": int(viol.sum()),
            "violation_rate": float(viol.mean()),
            "worst_drop": float(step.min()),
            "violations_within_10ft": int(viol[:, :4].sum()),
        }
    return out


def monotonicity(surfaces: dict, tol: float = 0.002) -> dict:
    """Along each lane (fixed y) from the blue line (x=25) to the goal line (x=89), V should not
    fall by more than ``tol`` as x increases. Behind the net (x>89) is exempt."""
    out = {}
    for sg, (X, Y, V) in surfaces.items():
        xs = X[0]
        cols = (xs >= 25) & (xs <= 89)
        lanes = [i for i, y in enumerate(Y[:, 0]) if abs(y) <= 36]
        viol, checks, worst = 0, 0, 0.0
        for i in lanes:
            v = V[i, cols]
            d = np.diff(v)
            checks += len(d)
            viol += int((d < -tol).sum())
            worst = min(worst, float(d.min()))
        dist = np.sqrt((89 - X) ** 2 + Y**2)
        oz = (X >= 25) & (X <= 89)
        rho = float(np.corrcoef(-dist[oz], V[oz])[0, 1])
        out[sg] = {
            "lane_steps": checks,
            "violations": viol,
            "violation_rate": viol / max(checks, 1),
            "worst_drop": worst,
            "corr_value_vs_neg_distance_oz": rho,
        }
    return out


def v_distance_lines(
    surfaces: dict, out, stem: str = "v_by_distance", title: str = "V(s) by distance to the net"
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (sg, (X, Y, V)) in enumerate(surfaces.items()):
        dist = np.sqrt((89 - X) ** 2 + Y**2)
        bands = np.arange(0, 190, 5)
        mask_front = X <= 89
        mv = [np.nanmean(V[mask_front & (dist >= a) & (dist < a + 5)]) for a in bands]
        ax.plot(bands + 2.5, mv, color=SERIES[i], label=SG_LABEL[sg])
    ax.axvline(89 - 25, color=TEXT2, lw=0.8, ls=":")
    ax.text(89 - 25 + 1, ax.get_ylim()[1] * 0.92, "blue line", color=TEXT2, fontsize=8)
    ax.set_xlabel("distance from attacking net (ft)")
    ax.set_ylabel("mean V(s) over rink area")
    ax.set_title(title + " (monotonicity check)")
    ax.legend()
    save(fig, out, stem)


def reliability(rows: list[dict], xkey: str, ykey: str, title: str, out, stem: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    x = [r[xkey] for r in rows]
    y = [r[ykey] for r in rows]
    hi = max(max(x), max(y)) * 1.05
    ax.plot([0, hi], [0, hi], color=TEXT2, lw=1, ls="--")
    ax.plot(x, y, color=SERIES[0], marker="o", ms=6)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save(fig, out, stem)


def yoy_scatter(ps: pl.DataFrame, cfg, out) -> None:
    from src.validation.validate import regime, yoy_pairs

    pairs = yoy_pairs(ps, cfg.metrics.min_toi_ev)
    pairs = pairs.filter(
        pl.col("season").map_elements(lambda s: regime(s, cfg) == regime(s + 1, cfg), return_dtype=pl.Boolean)
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
    for ax, pos in zip(axes, ["F", "D"]):
        d = pairs.filter(pl.col("pos_group") == pos)
        a, b = d["pm_epa_per60_ev"].to_numpy(), d["pm_epa_per60_ev_next"].to_numpy()
        ax.scatter(a, b, s=10, alpha=0.45, color=POS_COLOR[pos], edgecolors="none")
        r = np.corrcoef(a, b)[0, 1]
        k = np.polyfit(a, b, 1)
        xx = np.linspace(a.min(), a.max(), 10)
        ax.plot(xx, np.polyval(k, xx), color=TEXT2, lw=1.2)
        ax.set_title(f"{'Forwards' if pos == 'F' else 'Defense'}: r = {r:.2f} (n = {len(a)})")
        ax.set_xlabel("PM-EPA/60 (5v5), season Y")
    axes[0].set_ylabel("PM-EPA/60 (5v5), season Y+1")
    fig.suptitle(
        f"Year-over-year stability, ≥{cfg.metrics.min_toi_ev:.0f} 5v5 min both seasons, within scoring regime",
        x=0.02,
        ha="left",
        fontsize=10,
        color=TEXT2,
    )
    save(fig, out, "yoy_scatter")


def distributions(ps: pl.DataFrame, cfg, out) -> None:
    q = ps.filter(pl.col("qualified_ev"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(q["pm_epa_per60_ev"].quantile(0.005), q["pm_epa_per60_ev"].quantile(0.995), 40)
    for pos in ["F", "D"]:
        v = q.filter(pl.col("pos_group") == pos)["pm_epa_per60_ev"].to_numpy()
        axes[0].hist(
            v,
            bins=bins,
            alpha=0.55,
            color=POS_COLOR[pos],
            label=f"{'Forwards' if pos == 'F' else 'Defense'} (n={len(v)})",
            edgecolor="white",
            linewidth=0.5,
        )
        pc = np.linspace(1, 99, 99)
        axes[1].plot(pc, np.percentile(v, pc), color=POS_COLOR[pos], label="Forwards" if pos == "F" else "Defense")
    axes[0].set_xlabel("PM-EPA/60 (5v5)")
    axes[0].set_ylabel("player-seasons")
    axes[0].set_title("Distribution of PM-EPA/60 by position")
    axes[0].legend()
    axes[1].set_xlabel("percentile within position")
    axes[1].set_ylabel("PM-EPA/60 (5v5)")
    axes[1].set_title("Percentile curves")
    axes[1].axhline(0, color=TEXT2, lw=0.8)
    axes[1].legend()
    save(fig, out, "pm_epa_distributions")


def component_bars(ps: pl.DataFrame, out) -> None:
    comps = [
        "ev_carry",
        "ev_move_start",
        "ev_move_end",
        "ev_move_pool",
        "ev_takeaway",
        "ev_giveaway",
        "ev_shot_attempt",
        "ev_takeaway_against",
        "ev_unobserved_loss",
        "ev_stoppage_loss",
    ]
    q = ps.filter(pl.col("qualified_ev"))
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.38
    for i, pos in enumerate(["F", "D"]):
        d = q.filter(pl.col("pos_group") == pos)
        vals = [float(d[c].sum() * 60 / d["toi_ev"].sum()) for c in comps]
        ax.bar(
            np.arange(len(comps)) + (i - 0.5) * w,
            vals,
            width=w - 0.04,
            color=POS_COLOR[pos],
            label="Forwards" if pos == "F" else "Defense",
        )
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.set_xticks(np.arange(len(comps)))
    ax.set_xticklabels([c.replace("ev_", "").replace("_", "\n") for c in comps], fontsize=8)
    ax.set_ylabel("per 60 min (5v5), qualified players")
    ax.set_title("Where PM-EPA comes from: components by position")
    ax.legend()
    save(fig, out, "pm_epa_components")


def validation_bars(val: dict, out) -> None:
    metrics = [
        "pm_epa_per60_ev",
        "pm_direct_per60_ev",
        "pm_pooled_per60_ev",
        "ixg_per60_ev",
        "p1_per60_ev",
        "oi_xgf_per60_ev",
        "gv_per60_ev",
        "ind_sh_pct",
        "oi_sh_pct_ev",
    ]
    labels = [
        "PM-EPA",
        "PM-EPA direct",
        "PM-EPA pooled",
        "ixG",
        "G+A1",
        "on-ice xGF",
        "giveaways",
        "indiv. SH%",
        "on-ice SH%",
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    for i, pos in enumerate(["F", "D"]):
        row = val["yoy"][f"{pos}/within_regime"]
        vals = [row[m]["r"] for m in metrics]
        ax.bar(
            np.arange(len(metrics)) + (i - 0.5) * w,
            vals,
            width=w - 0.04,
            color=POS_COLOR[pos],
            label="Forwards" if pos == "F" else "Defense",
        )
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20)
    ax.set_ylabel("year-over-year r (per 60, 5v5)")
    ax.set_title("Repeatability: PM-EPA vs familiar rate stats (same sample)")
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.legend()
    save(fig, out, "yoy_vs_baselines")


def example_plays(examples: list[dict], out) -> None:
    """Rink diagrams of real possessions: arrows between observed states, V annotated."""
    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.9 * n))
    axes = np.atleast_1d(axes)
    for ax, ex in zip(axes, examples):
        draw_rink(ax)
        pts = ex["points"]
        for a, b in zip(pts[:-1], pts[1:]):
            if a["x"] is None or b["x"] is None:
                continue
            ax.annotate(
                "",
                xy=(b["x"], b["y"]),
                xytext=(a["x"], a["y"]),
                arrowprops={"arrowstyle": "-|>", "color": SERIES[0], "lw": 2},
                zorder=10,
            )
        for p in pts:
            if p["x"] is None:
                continue
            ax.scatter(
                [p["x"]],
                [p["y"]],
                s=60,
                color=SERIES[1] if p["kind"] in ("shot", "giveaway") else SERIES[0],
                zorder=11,
                edgecolors="white",
                linewidths=1.5,
            )
            ax.annotate(
                f"{p['label']}\nV={p['v']:.3f}",
                (p["x"], p["y"]),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=7.5,
                color="#0b0b0b",
                zorder=12,
            )
        ax.set_title(ex["title"], fontsize=10)
    save(fig, out, "example_plays")


def run() -> None:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    fig_root = cfg.path(cfg.data.figures_dir)
    bundle = joblib.load(cfg.path(cfg.data.value_dir) / "v_models.joblib")
    s_start = v_surface_figures(bundle["models"]["all"], fig_root / "value_surfaces")
    s_dec = v_surface_figures(
        bundle["dec_models"]["all"],
        fig_root / "value_surfaces",
        tag="v_dec",
        kind="shot",
        label="Decision-point value (V_dec)",
        note="Estimated from terminal decisions (shots, giveaways, lost pucks)",
    )
    mono = {
        "v_dec_radial": radial_monotonicity(bundle["dec_models"]["all"], "shot"),
        "v_start_radial": radial_monotonicity(bundle["models"]["all"], "takeaway"),
        "v_dec_lanes": monotonicity(s_dec),
        "v_start_lanes": monotonicity(s_start),
    }
    cfg.path(cfg.data.metrics_dir).joinpath("monotonicity.json").write_text(json.dumps(mono, indent=2))
    v_distance_lines(s_start, fig_root / "value_surfaces", "v_start_by_distance", "V_start by distance to the net")
    v_distance_lines(s_dec, fig_root / "value_surfaces", "v_dec_by_distance", "V_dec by distance to the net")
    vb = json.loads((cfg.path(cfg.data.metrics_dir) / "value_bakeoff.json").read_text())
    chosen = vb["chosen"]
    reliability(
        vb["test"][chosen]["reliability"],
        "mean_v",
        "mean_g",
        f"V(s) calibration, test seasons\n({chosen})",
        fig_root / "calibration",
        "v_reliability_test",
        "predicted V (decile mean)",
        "observed possession xG",
    )
    reliability(
        vb["test"][chosen]["reliability"],
        "mean_v",
        "goal_rate",
        "V(s) vs realized goals, test seasons",
        fig_root / "calibration",
        "v_reliability_goals_test",
        "predicted V (decile mean)",
        "observed goal rate per possession",
    )
    xm = json.loads((cfg.path(cfg.data.metrics_dir) / "xg_metrics.json").read_text())
    reliability(
        xm["crossfit_all"]["reliability"],
        "mean_pred",
        "mean_obs",
        "Shot xG calibration (cross-fitted, all seasons)",
        fig_root / "calibration",
        "xg_reliability_crossfit",
        "predicted xG (decile mean)",
        "observed goal rate",
    )
    reliability(
        xm[f"{xm['chosen']}_test"]["reliability"],
        "mean_pred",
        "mean_obs",
        "Shot xG calibration (temporal test 2023-25)",
        fig_root / "calibration",
        "xg_reliability_temporal_test",
        "predicted xG (decile mean)",
        "observed goal rate",
    )
    ps = pl.read_parquet(proc / "player_season.parquet")
    yoy_scatter(ps, cfg, fig_root / "validation")
    distributions(ps, cfg, fig_root / "validation")
    component_bars(ps, fig_root / "validation")
    vpath = cfg.path(cfg.data.metrics_dir) / "validation.json"
    if vpath.exists():
        validation_bars(json.loads(vpath.read_text()), fig_root / "validation")
    ex_path = cfg.path(cfg.data.metrics_dir) / "example_plays.json"
    if ex_path.exists():
        example_plays(json.loads(ex_path.read_text()), fig_root / "examples")
    _ = Line2D
    log.info(
        "figures written; V_dec radial monotonicity: %s",
        {k: round(v["violation_rate"], 3) for k, v in mono["v_dec_radial"].items()},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
