# Model Card — Turnover Cost (TO), Faceoffs (FO) and Penalties (PEN) at 5v5, v1.0

*Format after Mitchell et al. (2019). Companion to the PM-EPA, DPM and FIN model cards.*

## Model details

| | |
|---|---|
| Name | TO, FO, PEN: the 5v5 value slices left unclaimed by PM-EPA, DPM and FIN |
| Version | 1.0 (2026-09-29) |
| TO | The opponent's start value after a player's team's turnovers, charged to the committers (λ = 1 minus λ = 0 credit difference for giveaway, unobserved loss and takeaway-against). Arena-adjusted (visitors-only, shrunk), then empirical-Bayes shrunk per 60 above the pooled fringe rate. |
| FO | Hierarchical logistic taker-skill model (penalized Laplace, SQUAREM EM) on 490k 5v5 draws, valued by S = V(dot, own frame) + V(dot, opponent frame) from the ledger's own faceoff start values. |
| PEN | Cost per penalty class = mean net xG for the advantaged team over the man advantage; gamma-Poisson empirical-Bayes rates of minor-equivalent penalties drawn and taken. |
| Unit | Goals (xG scale) at 5v5 above the fringe replacement pool; wins = goals ÷ 5.90 |
| Code | `python -m src.components.run`, then `python -m src.valuation.run` (`make components valuation`) |
| Spec | [docs/components/methodology.md](../../docs/components/methodology.md): pre-registered gates and a deviations log |

## Intended use

* Filling the named gaps in the all-in valuation: turnover cost, faceoff skill and penalties drawn/taken, in the same goals and wins-above-replacement units as the other components.
* Team-level accounting: TO is identity-based (it claims the slice DPM excludes), so it reconciles with team goal differential.

## Out-of-scope uses

* **Player-level turnover value for puck-dominant players** without reading the caveat below (`to_ra_gar` mitigates part of it).
* **Faceoff value beyond the next possession.**
* **Penalty consequences on special teams**, which belong to the special-teams component (PEN uses the league-average consequence).
* **Cross-season turnover comparisons across the 2024-25 recording change** without reading the arena section.

## Performance

All figures from [validation_report.md](validation_report.md). Ten of eleven pre-registered gates pass.

| Component | Result |
|---|---|
| TO accounting | −ΣT2 ÷ (DPM-excluded turnover-born + unobserved starts) = 0.89 |
| TO stability | season-centred YOY 0.60 (F), 0.48 (D); shrunk beats raw next season (z = 14.5) |
| TO-RA (role-adjusted variant) | residual vs involvement abs r 0.02 / 0.01 (gate RA-1); shrunk YOY 0.55 F, 0.45 D; effect modest (rank corr with TO 0.93 / 0.96) |
| TO team-level slope | 1.51 ± 0.66 shrunk (**gate TO-2 missed by 0.01**), 1.18 ± 0.44 unshrunk |
| Arena bias | 15–23% of the mean charge in 2017–2023; ≈ 0 in 2024-25 |
| FO skill | persistent SD 0.24 logit (centers); forward z = 24.5 vs no taker, 12.4 vs raw; slope 0.84 |
| FO value | 0.0245 xG per draw won; best seasons +4 to +5 goals |
| PEN cost | 0.175 xG per minor (95% CI 0.173–0.177) |
| PEN stability | YOY drawn 0.48, taken 0.52; shrunk beats raw (z = 7.7, 7.2) |
| PEN vs special-teams goal differential | slope 1.35 ± 0.51 |

## Ethical considerations

* Named athletes are rated; publish intervals with any ranking.
* **TO is biased in allocation against puck-dominant players.** Giveaways are recorded, successful plays are not. The role-adjusted variant (`to_ra_gar`: charge relative to players of similar observed puck involvement) removes the involvement gradient but only modestly changes rankings, and it also removes some real skill. The most-charged skaters are offensive stars (Draisaitl, Pastrnak, Kucherov, Barzal, MacKinnon) and the least-charged are shutdown defensemen.
* **PEN and FO reflect roles.** Penalty drawing is partly a function of shot volume and puck handling; faceoff value accrues only to centers.
* Penalty calls carry referee and home bias that is not modelled.

## Caveats and recommendations

* Read TO together with PM-EPA: the two are on the same currency, and TO is the charge PM-EPA (at λ = 0) leaves out.
* Team-level slopes above 1 for shrunk components are partly shrinkage compression, not an error in player values.
* Re-validate the arena adjustment whenever NHL event recording changes.
