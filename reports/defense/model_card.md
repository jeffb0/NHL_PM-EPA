# Model Card — Defensive PM-EPA (DPM) v1.0

*Format after Mitchell et al. (2019). Companion to the [PM-EPA model card](../model_card.md).*

## Model details

| | |
|---|---|
| Name | DPM: Defensive PM-EPA |
| Version | 1.0 (2026-09-28) |
| Type | Weighted ridge RAPM on 5v5 stints. The target is the opponent's possession value (xG), split into the seven PM-EPA ledger components. Adjusts for teammates, opponents, zone starts, score and venue. Season fits are chained with a prior (ρ = 1: last season's estimate is the prior mean). |
| Unit | xG against prevented per 60 5v5 minutes vs an average skater (positive = good) |
| Code | `src/defense/`; `python -m src.defense.run` (or the `defense` step of `python -m src.pipeline`) |
| Spec | [docs/defense/methodology.md](../../docs/defense/methodology.md): pre-registered rules and deviations log |

## Intended use

* The **defensive axis** of the planned all-in player valuation, alongside PM-EPA (offense with the puck).
* Comparing skaters' 5v5 on-ice defensive impact **within season and position**, with uncertainty.
* Explaining *how* a skater suppresses chances, through the exact component breakdown: recoveries
  allowed, movement allowed, attempts/losses forced, shot quality allowed, turnover-born and
  faceoff-born opponent possessions.

## Out-of-scope uses

* **Goaltending.** xG is pre-save, so goalies never enter.
* **Penalty kill** (a later component) and playoffs.
* **Fine-grained ranks.** The median 95% CI is ~0.27 xGA/60 wide against a player SD of ~0.15. In
  2025-26, only 37% of qualified skaters have a CI that excludes zero.
* **Individual credit for specific plays.** DPM is an on-ice regression estimate, not an event ledger.

## Training data

5v5 stints from the PM-EPA v1.0 build: 3.12M stints from 10,901 games with on-ice match rate ≥ 0.9,
2017-18 → 2025-26. 99.0% of 5v5 ledger value is assigned to a stint (G5). Seasons from 2021-22
record shift times to the second, which fragments stints (29% under 2 s). The minutes-weighted
likelihood handles this, and totals are unchanged.

## Performance

All numbers are from [validation_report.md](validation_report.md).

| Check | Result |
|---|---|
| Held-out next-season xGA prediction (2023→24, 2024→25) vs league-average defense | ΔMSE −0.048, SE 0.007 (≈ 6.8 SE) |
| Season-centred YOY r (≥ 500 min, independent fits) | F 0.41 · D 0.38 |
| YOY r, team changers | F 0.32 · D 0.31 |
| Split-half r (odd/even games) | 0.19–0.43 (F), 0.30–0.45 (D) by season |
| r with −(raw on-ice xGA/60) | F 0.67 · D 0.61 |
| r with PM-EPA/60 (overlap with the offense component) | F 0.06 · D 0.14 |
| Component additivity (G4) | 3.3e-13 |

**What the pre-registered test found.** The hypothesis was that the ledger's denser components
(movement allowed, forced losses) would give a less noisy defensive estimate than total xGA. It was
**not supported**. The model using only those components predicted next-season xGA worse than total
xGA RAPM (ΔMSE +0.014, SE 0.003 on validation). Movement allowed is the *least* repeatable
component (YOY 0.14–0.16). The headline is therefore total xGA RAPM with a chained prior. The ledger
supplies an exact explanatory breakdown of it.

## Combining with PM-EPA

The two use disjoint ledgers: PM-EPA splits the team's own possession xG, DPM the opponent's.
`dpm_val_per60` removes the opponent value that follows the player's team's **turnovers** and
**lost faceoffs**, which are left to future turnover and faceoff components. Use that column when
adding components into a valuation. The ~0.1 correlation with PM-EPA shows little signal overlap.

## Ethical considerations

* Named athletes are rated. Publish intervals with ranks.
* **Deployment confounding remains.** Ridge regression adjusts for teammates and opponents only
  partly. Players who spend long stretches with one partner share credit.
* **Prior carry-over.** With ρ = 1, earlier seasons keep influencing current values. A real change
  in a player (injury, age) is picked up only gradually.

## Caveats and recommendations

* Read `dpm_lo`/`dpm_hi` before the rank, and prefer the 3-season window file for decisions.
* The per-season penalty was chosen for within-season prediction. Stacking weights above 1 suggest it
  over-shrinks for next-season prediction. A v2 should tune α (or position-specific priors) on the
  next-season target.
* The 2024-25 scorekeeping change affects the turnover and recovery components' *split*, but not
  total xGA.
