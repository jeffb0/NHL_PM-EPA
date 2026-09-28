"""Generate the six analysis notebooks (run: python notebooks/_build_notebooks.py).

Notebooks only *read* pipeline outputs, so `make notebooks` executes them reproducibly after
`make pipeline`. Keeping their source here (not hand-edited JSON) keeps diffs reviewable.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = """import os, sys, json
sys.path.insert(0, os.path.abspath('..'))
import polars as pl, numpy as np
import matplotlib.pyplot as plt
from src.config import load_config
cfg = load_config()
proc = cfg.path(cfg.data.processed_dir)
pl.Config.set_tbl_rows(25); pl.Config.set_tbl_width_chars(160)"""

NB = {
    "01_exploration.ipynb": [
        ("md", "# 01 · Data exploration\nEvent coverage, coordinate normalization and the 2024-25 scorekeeping break."),
        ("code", SETUP),
        ("code", "ev = pl.read_parquet(proc / 'events.parquet')\nev.group_by('season').agg(pl.col('game_id').n_unique().alias('games'), pl.len().alias('events')).sort('season')"),
        ("code", "counts = ev.group_by('season', 'type').len().pivot(on='type', index='season', values='len').sort('season')\ncounts.select('season', 'giveaway', 'takeaway', 'hit', 'blocked-shot', 'missed-shot', 'shot-on-goal', 'goal')"),
        ("md", "Giveaways roughly double and takeaways fall from 2024-25: a recording change, handled as a separate **scoring regime**."),
        ("code", "sh = ev.filter(pl.col('type').is_in(['shot-on-goal','missed-shot','goal']))\nfig, ax = plt.subplots(figsize=(8,3.5))\nax.hexbin(sh['x'], sh['y'], gridsize=60, cmap='Blues', mincnt=1)\nax.set_title('Unblocked shot locations after normalization (attacking +x)'); ax.set_aspect('equal')"),
        ("code", "print((proc / 'data_quality.md').read_text())"),
    ],
    "02_possession_chains.ipynb": [
        ("md", "# 02 · Possession chains\nInferred possessions, start/end types, and a hand-audit sample."),
        ("code", SETUP),
        ("code", "poss = pl.read_parquet(proc / 'possessions.parquet')\nprint(poss.height, 'possessions;', round(poss.height / poss['game_id'].n_unique(), 1), 'per game')\nposs.group_by('start_type').len().sort('len', descending=True)"),
        ("code", "poss.group_by('end_type').len().sort('len', descending=True).head(15)"),
        ("code", "poss.with_columns((pl.col('end_t') - pl.col('start_t')).alias('dur')).group_by('start_type').agg(pl.col('dur').median().alias('median_s'), pl.col('dur').quantile(0.9).alias('p90_s')).sort('start_type')"),
        ("code", "obs = pl.read_parquet(proc / 'observations.parquet')\nev = pl.read_parquet(proc / 'events.parquet')\nfor r in poss.sample(3, seed=11).iter_rows(named=True):\n    print('===', r['start_type'], '->', r['end_type'])\n    print(obs.filter(pl.col('poss_id') == r['poss_id']).select('seq','obs_kind','player_id','t','x','y','strength'))\n    print(ev.filter((pl.col('game_id') == r['game_id']) & pl.col('t_game').is_between(r['start_t'] - 2, r['end_t'] + 1)).select('t_game','type','team_id','player1','x','y'))"),
    ],
    "03_value_surface.ipynb": [
        ("md", "# 03 · Value surfaces V(s)\nBake-off results, chosen specification, and rink heatmaps."),
        ("code", SETUP),
        ("code", "vb = json.load(open(cfg.path(cfg.data.metrics_dir) / 'value_bakeoff.json'))\nprint('chosen:', vb['chosen'])\npl.DataFrame([{k: v for k, v in r.items()} for r in vb['bakeoff']['rows']]).select('spec','mse','r2_vs_const','calib_slope','d_mse_vs_best','se_d_mse','tied_with_best').sort('mse')"),
        ("code", "v5 = pl.read_parquet(cfg.path(cfg.data.value_dir) / 'v_state_5v5.parquet')\nv5.filter(pl.col('kind') == 'control').sort('v', descending=True).head(15)"),
        ("code", "from IPython.display import Image\nImage(str(cfg.path(cfg.data.figures_dir) / 'value_surfaces' / 'v_dec_5v5_full.png'))"),
        ("code", "Image(str(cfg.path(cfg.data.figures_dir) / 'value_surfaces' / 'v_dec_by_distance.png'))"),
    ],
    "04_epa_examples.ipynb": [
        ("md", "# 04 · PM-EPA worked examples\nReal possessions, their observed states, V values and credit rows (same as docs/pm_epa/attribution_examples.md Part B)."),
        ("code", SETUP),
        ("code", "tr = pl.read_parquet(proc / 'transitions.parquet')\ntr.group_by('seg_type').agg(pl.len(), pl.col('delta').mean(), pl.col('gap').median()).sort('seg_type')"),
        ("code", "cr = pl.read_parquet(proc / 'credits.parquet')\ncr.group_by('component').agg(pl.len(), pl.col('value').sum()).sort('component')"),
        ("code", "from IPython.display import Image\nImage(str(cfg.path(cfg.data.figures_dir) / 'examples' / 'example_plays.png'))"),
    ],
    "05_validation.ipynb": [
        ("md", "# 05 · Validation\nKey numbers from reports/metrics/validation.json; the full write-up is reports/validation_report.md."),
        ("code", SETUP),
        ("code", "val = json.load(open(cfg.path(cfg.data.metrics_dir) / 'validation.json'))\n{k: {m: round(v[m]['r'], 3) for m in ['pm_epa_per60_ev','pm_direct_per60_ev','pm_pooled_per60_ev','ixg_per60_ev','p1_per60_ev','oi_xgf_per60_ev']} for k, v in val['yoy'].items() if '/' in k}"),
        ("code", "val['split_half']"),
        ("code", "abl = json.load(open(cfg.path(cfg.data.metrics_dir) / 'ablations.json'))\npl.DataFrame([{'variant': k, 'yoy_F': v['F']['yoy_r'], 'yoy_D': v['D']['yoy_r'], 'chg_F': v['F']['yoy_r_changers'], 'conv_F': v['F']['r_p1']} for k, v in abl.items()])"),
        ("code", "from IPython.display import Image\nImage(str(cfg.path(cfg.data.figures_dir) / 'validation' / 'yoy_scatter.png'))"),
    ],
    "06_player_leaderboards.ipynb": [
        ("md", "# 06 · Player leaderboards\nQualified (≥500 5v5 min) PM-EPA/60 with bootstrap CIs. Read the CI before the rank."),
        ("code", SETUP),
        ("code", "lb = pl.read_csv(cfg.path(cfg.data.output_dir) / 'pm_epa_player_leaderboards.csv')\nq = lb.filter(pl.col('qualified_ev'))\nq.filter(pl.col('season') == q['season'].max()).sort('pm_epa_per60_ev', descending=True).select('player_name','team','position','toi_ev','pm_epa_per60_ev','uncertainty_ci_low_per60','uncertainty_ci_high_per60','pm_direct_per60_ev','pm_pooled_per60_ev','percentile_ev').head(25)"),
        ("code", "multi = q.group_by('player_id','player_name','position').agg(pl.col('pm_epa_ev').sum(), pl.col('toi_ev').sum(), pl.len().alias('seasons')).filter(pl.col('seasons') >= 3).with_columns((pl.col('pm_epa_ev') * 60 / pl.col('toi_ev')).alias('per60'))\nmulti.sort('per60', descending=True).head(25)"),
        ("code", "ts = pl.read_csv(cfg.path(cfg.data.output_dir) / 'pm_epa_team_season.csv')\nts.filter(pl.col('season') == ts['season'].max()).sort('pm_epa_attributed', descending=True)"),
    ],
}


def main() -> None:
    for name, cells in NB.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, HERE / name)


if __name__ == "__main__":
    main()
