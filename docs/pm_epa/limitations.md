# Limitations and Roadmap

Ordered by how much each one should change how you read a PM-EPA number.

## 1. Data limitations (the big ones)

### 1.1 Public PBP does not record puck movement
There are no pass, carry, reception, zone-entry, zone-exit or dump-in events. Most puck movement happens *between* recorded events: a possession averages ~1.6 observed states over a median of 13–20 s. Consequences:

* **Individual attribution is partial by construction.** Only three kinds of credit are truly individual: *observed carries* (same player at both ends of a segment), *takeaways* and *giveaways*. Everything else is split between the endpoint players we can see and the skaters on ice. The residual share is a modelling choice (0.35/0.35/0.30), set by ablation evidence in `reports/validation_report.md` §6 rather than known truth.
* **The passer is invisible.** When a player makes the stretch pass that springs a teammate's shot, PBP records only the shot. The passer gets a pool share (1/5 of 30%) unless they also started the segment. This is the largest gap between PM-EPA-on-PBP and the brief's ideal metric (§7.3, "credit the passer").
* **Controlled vs. dump entries cannot be separated** (brief §4.1.5). Their effect appears only through what happens next: a dump that is lost becomes an `unobserved_loss`, and a controlled entry that yields a shot becomes a positive segment.

### 1.2 Scorekeeper-recorded events are noisy and rink-biased
Giveaways, takeaways and hits are judgment calls by arena scorers. Rink indices vary materially and persist across seasons (validation report §9). PM-EPA's turnover components inherit this bias. We report home/away splits, but we do **not** adjust for rink effects in the headline metric; this is roadmap item R3.

### 1.3 Scoring-regime break in 2024-25
Recorded giveaways roughly doubled and takeaways fell ~35% from 2024-25. We cross-fit V within regime and report YOY within regime, but player values are **not directly comparable across the break**. A 2023-24 PM-EPA/60 and a 2024-25 PM-EPA/60 are on slightly different scales, mostly through the turnover components.

### 1.4 Observation bias in V(s)
V is estimated only from *alive* observations: starts and mids. Regions where the puck is rarely *gained*, such as the attacking slot, are informed by few rows and lean on shrinkage toward parent zones. We cannot observe "the puck passed through here and kept going".

## 2. Modelling limitations

* **Own-zone giveaways are nearly free under the production λ = 0.**
  * A giveaway costs the decision-point value V_dec where it happened. In the defensive zone that value is ≈ 0.0003 xG, because possessions almost never *score* from a decision made there.
  * The real cost, handing the opponent a dangerous puck (~0.05 xG), is the λ term. λ = 0 was chosen by the pre-registered rule because at λ = 1 that term dominates and cannot be balanced by the unobserved positives of puck handling.
  * **Mitigation:** the leaderboard ships a risk-adjusted companion column, `pm_epa_per60_ev_lam1` (λ = 1). Use it when turnover risk matters, and read the two together.
  * Brief principle 7 ("moving the puck to a worse state should be ≤ 0") holds at λ = 0, but the penalty is weak in the defensive zone.
* **V_dec ≤ V_att is not enforced.** In theory V_dec = P(attempt)·V_att ≤ V_att, but the two GBMs are fit separately. In 19.5% of shots V_att < V_dec, giving a small negative attempt credit (mean −0.0015 xG; −364 xG in total against +2,029 positive). Planned fix: attempt = max(V_att, V_dec) − V_dec, which keeps the identity.

* **Possession-scoped value.** V covers only the current possession, per the brief's definition. Value that spills into the next possession, such as a dump-in that wins a forecheck, is captured only through the λ term and the opponent's start state. A multi-possession (next-goal) value function is roadmap item R4.
* **No pressure/space context.** Public data has no defender positions. `puck_pressure_proxy` (brief §5.2) was considered: recent opponent hits could serve as a proxy, but hits are themselves scorer-biased, so it was not adopted.
* **xG model drift.** Under a strict temporal holdout, 2023-25 goals are over-predicted by ~13% (isotonic recalibration: ~3%). Cross-fitted per-season calibration ranges 0.97–1.08. PM-EPA levels move with xG levels, while within-season rankings are largely unaffected.
* **Pool credit is not individual.** Pooled components behave like on-ice (with-teammates) stats. They raise stability partly through linemate and system effects. The team-changer YOY test quantifies this (validation report §2).
* **Faceoff winners get zero start share.** This is deliberate (brief §4.2). A center who wins a draw *and* carries the puck is credited only if he is the recorded endpoint.
* **Goalies are excluded** from player credit; goalie puck-handling is invisible anyway.
* **Special-teams surfaces are thinner.** PK and especially 4v4/3v3 V estimates rely on shrinkage. PP/PK per-60 rates need larger TOI minimums (200 min) to be meaningful.

## 3. Scope limitations (by design)

* **Offense-first.** PM-EPA measures what a player does *with* the puck: moving it, winning it, losing it. It does not measure defensive positioning, gap control or entry denial (brief §9.4). A defenseman who never touches the puck but prevents entries can look average here.
* **Not WAR.** No replacement level, no penalties, no finishing (shot quality xG − V_att and G − xG are excluded), no goaltending. Do not add PM-EPA to other components without a joint model; they overlap.
* **Regular season only.** Playoffs, pre-season and international play are out of scope.

## 4. Roadmap

| ID | Upgrade | Needs | Expected effect |
|---|---|---|---|
| R1 | **Tracking-grade events** (Sportlogiq/Stathletes-style passes, carries, receptions with coordinates). The engine already accepts `carry`/`pass` observations: each becomes an observed segment with a known actor, and the unobserved residual shrinks toward zero. | licensed event data | Converts pooled credit into individual credit; passer attribution becomes possible |
| R2 | **NHL EDGE puck/player tracking**: pressure, passing lanes, defender distances as state features (Fernández & Bornn-style space). | tracking feed | V(s) conditional on pressure; true "movement under pressure" value |
| R3 | **Rink-adjusted turnovers**: scale giveaway/takeaway weights by rink index (Schuckers-style), estimated out of sample. | none (public) | Lower scorer bias in turnover components |
| R4 | **Multi-possession value** (next-goal EP, Routley–Schulte style) replacing λ·V_opp. | none | Values forecheck/dump-and-chase strategies properly |
| R5 | **Defensive PM-EPA**: the mirror metric. Opponent PM-EPA allowed while on ice, adjusted RAPM-style for teammates/opponents. **v1.0 built 2026-09-28**: [docs/defense](../defense/methodology.md). | none | The defensive axis requested in brief §9.4 |
| R6 | **Hierarchical player model**: estimate player effects on segment Δ jointly with on-ice teammates (ridge / Bayesian), instead of fixed pool shares. | none | Principled, data-driven credit splitting; honest posterior intervals |
| R7 | Playoff and cross-era validation. | none | Scope |
| R8 | **Finishing talent and value integration**: built 2026-09-29, [docs/finishing](../finishing/methodology.md). Turnover, faceoff and penalty components added the same day ([docs/components](../components/methodology.md)). Open: team-level component slopes are still above 1 (partly shrinkage compression); a joint calibration is the next step. | none | Adds the shooting axis and a common goals/wins-above-replacement scale |
