# Elite-prospect priors and faster learning for inexperienced players — results

Pre-registration: [docs/prospects/elite.md](../../docs/prospects/elite.md). Metrics: `reports/metrics/elite_prospects.json`.

## Gates

| gate | rule | statistic | result |
|---|---|---|---|
| EP-1 | part A better on true rookies (z > 2) | z = 0.29 (230 debut seasons) | FAIL |
| EP-2 | no harm: all players z > −2; game model z > −2 | 0.65; −1.08 (preseason-only), −1.52 (full) | PASS |
| EP-3 | part B improves the full game model (z > 2) | κ = 4 tuned; z = −0.25 | FAIL |

**Part A (draft slot and debut timing): implemented by the pre-registered rule, with the estimated values. Part B (faster in-season learning for inexperienced players): not implemented.**

## What the data say about elite picks in their first season

| debut group (forwards) | debuts in the data | their actual value per 60 | model's projection (90% interval) | generic rookie |
|---|---|---|---|---|
| top-3 pick, debuting at 18 | 14 | 0.11 | 0.16 (0.12–0.19) | 0.13 |
| top-3 pick, debuting a year later | 2 | 0.09 | 0.13 (0.08–0.17) | 0.13 |
| picks 4–10, debuting a year later | 11 | 0.13 | 0.10 (0.06–0.13) | 0.13 |
| picks 11–32, debuting a year later | 9 | 0.18 | 0.12 (0.09–0.14) | 0.13 |
| later pick or undrafted, later debut | 125 | 0.15 | 0.11 (0.10–0.13) | 0.12 |

(Goals above replacement per 60 minutes, all strengths. For scale: a good top-six forward is about 0.35, a star 0.6 or more. Defensemen: every drafted group projects at 0.03–0.04 against 0.06–0.07 for the rest; the bins have one to seven players each.)

All 230 true rookies averaged 0.106; first-round rookies 0.102; top-ten picks 0.103; top-three picks 0.127.

## Interpretation

**In this model's value measure, top picks have not played like stars as rookies.** The fourteen top-three forwards who debuted at 18 since 2018 produced, on average, about what a generic rookie does (0.11 per 60), nowhere near the 0.35–0.60 the author's prior suggested.
The value measure counts defense, turnovers and penalties as well as offense, and 18-year-olds who score often give much of it back; several of these picks also had poor first seasons by any measure.
The fitted coefficients are small and of mixed sign (top-three +0.011, debut at 18 +0.012 for forwards), and the one informative feature remains pre-NHL production (0.18 per NHL-equivalent point per game).

**So the adopted prior is a small bump, not a jump.** Gavin McKenna moves from 0.13 to 0.17, mainly through his college production, and Toronto's win probability against Ottawa from 42.6% to 42.8%. Viggo Björck (eighth overall, debuting at 18) is at 0.06 because his pre-NHL scoring translates low.
The debut-at-18 signal the author expected (teams only keep 18-year-olds who have earned it) is not visible in the data: the coefficient is slightly positive for forwards and negative for defensemen.

**It does not improve forecasts.** True-rookie projections are unchanged (z = 0.29), first-round rookies slightly worse (z = −0.44), and the game model is nil to slightly worse (0.00008 log loss). It was implemented because the rule fixed in advance says a sound method with no harm beyond the threshold is, and because the author asked for draft-aware handling; nobody should expect it to move results.

**Faster learning for inexperienced players did not help.** Giving thin-history players a larger share of each in-season update looked better on the two tuning seasons at every step up to the largest multiplier tried, and did nothing on the five test seasons (z = −0.25). Not adopted.

**The author's alternative.** Setting top-three debut ratings by judgment (0.35–0.60) would be an explicit override of the estimate. For scale: rating McKenna as a star moves Toronto's win probability by about 2 points in a game. The record above is the evidence against it; this season's results for the few such players are the evidence that could support it.

**Limitations.** Fourteen top-three debuts; the value measure may under-credit young scorers (its possession and finishing parts are shrunk); only players who reached the NHL are in the data; no per-game individual value exists to update a rookie's rating from his own shifts.
