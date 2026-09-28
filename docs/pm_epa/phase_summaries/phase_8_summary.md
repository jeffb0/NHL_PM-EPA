# Phase 8 — Validation (hard gate)

**Done.** `src/validation/` (validate, rapm, ablations, report) → `reports/validation_report.md`: 12 numbered sections, auto-generated from `reports/metrics/*.json`, plus a hand-written interpretation.

**Gates.** 10 of 12 PASS, 2 PARTIAL, 0 FAIL.

| Gate | Result |
|---|---|
| V calibration slope | 1.004 |
| xG calibration-in-the-large | 1.007 |
| YOY, within regime | F 0.56 / D 0.61; season-centred F 0.53 / D 0.52 (targets 0.40 / 0.35) |
| Split-half (≥800 min, half-season r) | F 0.56 / D 0.63 |
| Discriminant (\|r\| with individual SH%) | ≤ 0.25 |
| Accounting identity | 1e-16 |
| **PARTIAL:** convergent r with G+A1/60 | 0.40 (target 0.6–0.8). Assists are passes, and passes are unobserved |
| **PARTIAL:** team PM-EPA vs xGF | 0.69 (target 0.70) |

**Other evidence.**
* 5v5 offensive xG RAPM: r 0.59–0.71 (F) and 0.42–0.62 (D), with incremental R² over ixG/60 + G+A1/60 in all six fits.
* Defense: PM-EPA/60 predicts next-season on-ice xGF/60 better than ixG/60 or G+A1/60.
* YOY survives the 2024-25 scoring break (0.57 / 0.56), while giveaways/60 YOY collapses (0.75 → 0.49).
* Home/road gap of +28% and strong rink scorer effects (index range 0.19–1.84, persistence 0.76) are quantified.

**Bugs found and fixed during validation.**
1. Unsigned-integer underflow in team goal differential (UInt32 counts); it briefly made xG differential predict next-season GD at r = −0.46.
2. A stale validation process from a superseded run was still writing outputs. It was killed before it could overwrite final results.

**Methodological finding.** Pooled YOY r across seasons rewarded a variant whose stability was whole seasons shifting in level (regime-naive cross-fitting: 0.70 pooled, 0.54 centred). Season-centred YOY is now reported and gated.

**Decisions from ablations** (pre-registered rule): λ = 0; shares 0.35/0.35 kept (best alternative +0.010 < 0.02); regime-aware cross-fitting kept; the V-model choice is immaterial (±0.004).

**Next:** polish, docs, exports.
