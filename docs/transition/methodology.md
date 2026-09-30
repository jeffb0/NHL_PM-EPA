# On-ice offense / transition component (mainly for defensemen) — methodology and pre-registration

*Written 2026-09-29 before the component was built or evaluated as a candidate. The gates in §5 are fixed here; changes after results are
seen are logged in §9. Motivation: the Morrissey/Appleton review (docs/calibration §5d).*

## 1. Question

Defensemen's offensive and transition contributions (breakouts, passes to the forwards, controlled exits and entries) are mostly not events in the
play-by-play, so the possession component (PM-EPA), which credits observed actions to the actor, barely varies for defensemen: its standard
deviation is 0.016 goals per 60 against 0.116 for defense. But the same play-by-play supports an **on-ice** measure that needs no individual
credit: a ridge regression (RAPM) of the team's own expected-value production per 60 on who is on the ice, exactly the machinery behind the defensive
component, applied to the offense side. Exploration before this document (independent seasons, no chained prior) showed that the on-ice offense
coefficient has a standard deviation of 0.09 per 60 for defensemen (about six times PM-EPA's) and a year-over-year correlation of 0.35 for defensemen and
0.54 for forwards, and correlates 0.47 with PM-EPA for defensemen. This work asks whether it carries information the current total lacks.

## 2. The measure

* **OPM (on-ice offense).** For each season, ridge RAPM at 5v5 on the stints of the defensive component (`def_stints`), targets `xg` (the sum of the
  ledger components), `start_turnover`, `start_faceoff` and `start_unobserved`, offense block only used, and the **valuation-ready** offense value
  `xg − start_turnover − start_faceoff − start_unobserved`, the same slices that the turnover and faceoff components claim and that the valuation-ready
  DPM removes (docs/components §2); same α as the DPM `xg` target; **ρ = 0** (each season fitted independently, so no persistence
  is built in: docs/chainfree); fringe-count covariates (same regular/fringe status as every component) carry the replacement level, pooled over seasons
  by position. `opm_gar = (offense coefficient − regular × pooled fringe effect of his position) × 5v5 minutes / 60`.
* **TR (the transition component).** The part of OPM not already in PM-EPA: within position, the minute-weighted regression of `opm_gar` on `pm_gar` over
  qualified skaters (≥ 500 minutes) gives `β_pos`, and `tr_gar = opm_gar − β_pos × pm_gar`. TR is by construction uncorrelated with PM-EPA within position,
  so adding it does not count the credited actions twice. It also overlaps finishing only through shot volume, since OPM is xG-based, not goal-based.

## 3. Evaluation

1. **Reliability:** year-over-year correlation of OPM and TR per 60 (season-centred by position, ≥ 500 minutes both seasons).
2. **Incremental validity at the player level:** next-season deployment-adjusted on-ice goal differential per 60 (the calibration extension's outcome, with team-season
   fixed effects, teammate context and state-share controls, ≥ 250 modeled minutes) on the season-t total rate (the position-weighted total of docs/calibration §5d),
   with and without the season-t TR rate; paired z on team-season squared error; all skaters, and defensemen and forwards separately.
3. **Incremental validity at the team level:** the position-specific all-strength identity of §5d with TR added as two more regressors (forwards' and
   defensemen's team sums), held-out team-level error against the §5d model, paired z.
4. **Weights and effects:** the ridge weight on TR (toward 1) with a bootstrap interval; the effect on the named players, the defensemen-forwards parity,
   and the correlation of defensemen's totals with ice time and points before and after (an external-validity look, not a criterion).

## 4. Application (only if adopted)

`total_gar_all_tr = total_gar_all_pos + w_TR × tr_gar` (published beside the others; the default stays the uncalibrated total until the model owner decides),
with `w_TR` from the ridge toward 1, clipped to [0, 3].

## 5. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| TR-1 | reliability: the year-over-year correlation of TR for defensemen is ≥ 0.20 |
| TR-2 | adding TR improves the player-level next-season prediction (paired z > 2), for all skaters and for defensemen |
| TR-3 | adding TR to the position-specific identity improves held-out team-level error (paired z > 2) |
| TR-4 | no harm elsewhere: the total with TR does not worsen the deployment-adjusted player-level check against the position-weighted total (z > −2) |

Adoption (publishing `total_gar_all_tr` as the recommended total) requires TR-1 to TR-4; otherwise the total is not changed and the result is reported.

## 6. What the result can say

A pass means the play-by-play carries offensive/transition signal for defensemen that credited actions miss, available as a separate component. It would
not identify *what* the defenseman does (breakouts versus passing versus positioning); an on-ice measure credits the player with everything that
happens while he is out, net of teammates and opponents as far as the ridge separates them. A failure would mean the on-ice offense signal is already
captured, or too noisy to add.

## 7. Limitations stated in advance

* Teammate confounding: defensive pairs and line combinations are collinear; ridge regularizes but does not solve it, and a defenseman's on-ice offense
  is partly his partner's and his forwards'.
* OPM has no unshrunk analogue, so its team-level slope also carries shrinkage compression.
* Zone entries and exits, passes and puck-carrying are not in the data; this is the nearest available on-ice proxy, not a tracking-based measure.
* Independent-season fits (ρ = 0) are noisier for thin samples than chained ones.

## 8. Deliverables

`src/transition/`, `data/outputs/player_transition.csv`, report, tests, README.

## 9. Deviations log

*(any change to §2–§5 after results are seen is recorded here with its reason)*

1. **Published as a variant although the gates failed (model owner's request, 2026-09-29).** TR-1, TR-2 and TR-3 fail, so by §5 the total is not changed. The owner asked to keep
   `total_gar_all_tr` and `player_transition.csv` because they like how it orders defensemen; they are published as a judgment-based alternative, not as a
   validated improvement, and the default total is unchanged. A follow-up diagnostic (report, "What drives TR") found that TR is not a linemate or team artifact (correlations
   with teammates' and team offense are near zero) but that part of it is a top-end artifact of the linear orthogonalization.
2. **Owner decision: `total_gar_all_tr` is the preferred (headline) skater total (2026-09-29).** After the diagnostic in the report, the model owner chose to
   use the position-weighted total plus TR as the preferred view of skater value, on judgment (it orders elite offensive defensemen more plausibly), knowing that
   TR-1, TR-2 and TR-3 fail. This overrides the §5 adoption rule and is **not** a validated improvement: defensemen's TR has year-over-year reliability 0.13, adds
   nothing out of sample, and part of its effect on the very best defensemen is a top-end artifact of the linear orthogonalization. The weight on TR is the
   fitted ridge weight (forwards 0.03, defensemen 1.04), so in practice the change is a defensemen adjustment. `total_gar_all`, `total_gar_all_pos` and
   `total_gar_all_ridge` remain published; `player_value_all_in.csv` (which stacks skaters and goalies) is not rebuilt, so the preferred skater column lives in
   `data/outputs/player_transition.csv`. Revisit if a tracking-based transition measure becomes available or if TR's reliability improves.
