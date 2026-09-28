# Phase 0 — Scoping & plan

**Done.** `docs/pm_epa/implementation_plan.md`: milestones, architecture, decision register D0–D8, risks, assumptions.

**Key decision (D0).** A probe of the NHL gamecenter feed showed public PBP has **no pass, carry, reception or entry events**. PM-EPA is therefore built as a granularity-agnostic engine, run on PBP at the finest resolution PBP supports. Movement is valued as state change between consecutive observed events; unobserved movement is split between visible endpoint players and on-ice skaters. This is stated in every deliverable.

**Other decisions.** The repo root is the brief's `pm-epa-nhl/` root. Python 3.11 via `uv` (the system Python is 3.9). scikit-learn's `HistGradientBoosting*` replaces LightGBM, because LightGBM needs a system `libomp` that a clean macOS clone lacks.

**Blockers.** None.  **Next:** literature review.
