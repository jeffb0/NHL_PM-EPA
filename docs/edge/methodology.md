# NHL EDGE tracking features — methodology and pre-registration

*Written 2026-09-29 before any EDGE data was fetched for analysis. The gates in §5 are fixed here; changes after results are seen are
logged in §8.*

## 1. Question

The projection layer models aging with a position-specific age spline and shrunk component histories. Player tracking (NHL EDGE) measures
physical traits directly: skating speed, bursts, distance covered, shot speed. If speed and workload decline before results do, they
are leading indicators of aging; if they explain part of the age effect, the age curve can be made player-specific. This work
ingests EDGE season summaries and tests whether they improve year-over-year projections and sharpen the aging curves.

## 2. Data

* **Source:** `https://api-web.nhle.com/v1/edge/skater-detail/{playerId}/{seasonId}/2` (regular season), one call per skater-season. EDGE
  statistics exist for seasons 2021-22 to 2025-26 only (the API lists them), so **five of our nine seasons (2021–2025)** have tracking data.
  The raw responses are cached under `data/raw/edge/`; ingestion is idempotent.
* **Fields used:** maximum skating speed, bursts over 20 mph, top shot speed, total distance skated, offensive-zone and defensive-zone time
  shares (all strengths), each with the league percentile.
* **Derived features** (season s, per skater): `max_speed` (mph), `bursts_per60` and `distance_per60` (per 60 minutes of all-strength ice time,
  from the shift data: the API totals scale with games played), `shot_speed` (mph), `oz_share` (offensive-zone time share).

## 3. Analyses

1. **Coverage:** share of skater-seasons in 2021–2025 with ≥ 300 modeled minutes that have EDGE data.
2. **Physical aging curves:** each EDGE feature regressed on age (position-specific, with the same hinge basis as the projection) with a
   player-cluster bootstrap; does speed fall with age, and where?
3. **Projection test:** the projection ridge (docs/projection) with the EDGE features of the prior season as extra features (`Spec.extra`),
   against the same model without them, fitted on **the same restricted training window** (target seasons ≥ 2022, the first with an EDGE
   history) so that the only difference is the features. Expanding window, test seasons 2024 and 2025 (the only two with at least two training
   target seasons), players with ≥ 300 target-season minutes, minute-weighted error, player-cluster bootstrap z.
4. **Does speed explain aging?** The age coefficients of the projection ridge with and without the EDGE features, on the EDGE-era data,
   and the size of the age effect between 27 and 34 in each.
5. **Sub-group:** the test of 3 for players aged 30 and over, where aging matters most.

## 4. Variants

EDGE feature sets: speed only (`max_speed`, `bursts_per60`); speed + distance; all five. Reported for each; the pre-registered headline is **all five**.

## 5. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| ED-1 | EDGE coverage of skater-seasons with ≥ 300 minutes in 2021–2025 is at least 90% |
| ED-2 | the maximum-speed age curve declines after age 27 for forwards and defensemen (age-slope interval excludes 0) |
| ED-3 | adding the EDGE features improves the horizon-1 projection on test seasons 2024–2025 (paired z > 2) |
| ED-4 | interval coverage with the EDGE features stays within [70%, 90%] |

Reported, not gated: the improvement for players aged 30+; the attenuation of the age effect; the speed-only and speed + distance variants.

## 6. What the result can and cannot say

Two test seasons and about 1,300 player-seasons: a small test. A null result would mean tracking summaries add nothing detectable to
the value estimates' own history at this sample size, not that physical decline is irrelevant. EDGE summaries are season aggregates (peak
speed, counts), not game-by-game series; an in-season trend cannot be built from them here.

## 7. Limitations stated in advance

* Only five seasons, and the earliest (2021-22) may be partial or lower quality; selection into tracking coverage is not documented.
* Percentiles and league averages are as published; raw values are used for the models.
* Season-level totals depend on games played; per-60 normalization uses modeled all-strength minutes.
* Goalies are excluded.

## 8. Deviations log

*(any change to §2–§5 after results are seen is recorded here with its reason)*

1. **Exploratory leading-indicator analysis added after ED-3 failed** (no gate): whether the change in speed, bursts or distance between t − 1 and t predicts
   next-season value beyond this season's value and the EDGE level (forwards and defensemen separately, three features each). Twelve regressions, so
   about one nominal significant result is expected by chance.
2. **Normalization check before the analysis:** EDGE games played equal games played in every season (ratio 1.00; 2024 averages 1.04), so distance and burst counts
   are normalized per 60 modeled minutes, and the features are z-scored within season and position to absorb any season-level drift in tracking.
