# PM-EPA Methodology (authoritative specification)

**Puck-Movement Expected Goals Added (PM-EPA)**: the expected goals a skater adds by changing the state of the puck while his team has it (moving it, winning it, or losing it), measured against the value of the state at the moment the team had it. Version 1.0 (2026-09-27), NHL regular seasons 2017-18 → 2025-26.

Companion documents: [formulas](formulas.md) · [definitions](definitions.md) · [attribution examples](attribution_examples.md) · [edge cases](edge_cases.md) · [limitations](limitations.md) · [literature review](literature_review.md) · [validation report](../../reports/validation_report.md) · [model card](../../reports/model_card.md).

---

## 1. Question and design principles

> Given the state of play when a player's team had the puck, how many expected goals did the player add by moving it to a new state (or winning or losing it), independent of finishing?

Following NFL EPA (Carter & Machol 1971; Yurko, Ventura & Horowitz 2019):

$$\text{PM-EPA}_{\text{action}} = V(\text{state after}) - V(\text{state before})$$

The value function V is an expectation of **expected goals** (xG), never of goals, which keeps process separate from results. Credit is **conserved**: every unit of value change is booked exactly once, to a player, to an on-ice pool, or to an explicit *unattributed* bucket. There is no xGChain-style duplication.

## 2. What the data can and cannot see

Public NHL play-by-play (`api-web.nhle.com`) records faceoffs, hits, giveaways, takeaways, shots (on goal, missed, blocked, goals), penalties and stoppages, each with coordinates and players. It records **no passes, carries, receptions or zone entries.** This single fact drives the design (implementation plan §0):

* A possession is observed only at a few events: ≤2 per possession without hits.
* Movement is inferred as the state change between consecutive observed events (a **segment**).
* A segment with the same player at both ends within 12 s is an **observed carry**. Any other segment contains at least one unobserved pass or carry, and its credit is split between the visible endpoint players and the skaters on ice.

The attribution engine uses the brief's action vocabulary. Given tracking-grade events, passes and carries would enter as fully observed segments without code changes (limitations R1).

## 3. Possessions (`src/possession/chains.py`)

The rules are specified in [definitions §3](definitions.md#3-possession-rules-srcpossessionchainspy) and unit-tested in `tests/test_possession.py`. In summary:

* **Starts:** faceoff win, takeaway, or an **inferred loose-puck recovery**. When a team's next recorded event is terminal, its start is inferred from the preceding loose-puck origin (a missed, saved or blocked shot, a giveaway, or an unobserved loss) within 25 s.
* **Terminals:** shot attempt, giveaway, opponent takeaway, any other opponent possession event (unobserved loss), stoppage, penalty, period end.
* **Hits** do not change possession. As an option, a hit is a *mid* observation when the hittee's team is already in possession. This adds a third observation to only 2.3% of possessions, so it is off by default (bake-off §4.4).

9 seasons, 11,052 games → **1,925,919 possessions** (174 per game) and 3.13M observations. Twenty randomly sampled chains were audited by hand against the raw event stream (Phase 3 summary). The audit led to four rule fixes: goals never seed loose starts; tighter inferred-start window; on-ice for losses taken at the ending event; an out-of-order time guard.

## 4. Value function V(s)

### 4.1 Currency
The terminal value of possession p is $G_p$ = xG of its terminal unblocked shot; 0 for blocked shots, turnovers and stoppages. xG comes from our own shot model (§4.5).

The brief is internally inconsistent here: §6.2 lists "Goal → 1.0", while §6.2's own caveat and guardrail §14 forbid results contamination. We resolve it in favor of the guardrail: **a goal is valued at its xG.** Goals are used only as the xG model's label and as a secondary calibration check (V vs realized goal rate, validation report §1).

### 4.2 Which observations define V
$V(s) = \mathbb{E}[G_p \mid \text{puck alive under the possessing team's control in state } s]$ is estimated only from **alive** observations: possession starts and mids. Terminal observations are *outcomes*, not origins. Using them would bias V toward where events get recorded. For example, neutral-zone observations are dominated by recorded turnovers, while clean carries through the neutral zone leave no event.

**Two value functions** (a Phase 6 finding; see the phase summary):

* **V_start(s)** values starts and mids: the value of *having gained* the puck in state s.
* **V_dec(s) = E[G_p | the terminal decision happens at s]** values the *decision point* where the possession ends (shot, giveaway, lost puck). It is fit on terminal observations. It is the analogue of NFL EP measured at the snap: the average outcome of decisions made there.

The first build valued decision points with V_start's takeaway-based control surface. That surface is informed by only ~19k takeaways a season, so it was nearly flat: V ≈ 0.036 at every shot location, out-of-sample R² 0.03. Moving the puck into the slot therefore earned almost nothing, while giveaways were fully charged, and the leaderboard inverted, with stars at the bottom. V_dec has out-of-sample R² ≈ 0.4.

**Attempt value** V_att(s) = E[xG | attempt at s] (blocked = 0) splits a shot's terminal value in two: *attempt* = V_att − V_dec (getting an attempt off here instead of the average decision outcome) and *quality* = xG − V_att (the release). Because V_dec = P(attempt | s)·V_att, attempt credits and giveaway debits are **mean-zero at every spot** (λ = 0). Crediting giveaways without attempts would bias the metric against anyone who handles the puck in dangerous areas; this is the NFL equivalent of charging interceptions but not crediting completions.

### 4.3 State
`strength group × kind × spatial [× score]` (definitions §4):

* **strength group:** 5v5 / PP / PK / 4v4_3v3, with separate surfaces (brief §9.1). Empty-net possessions are excluded.
* **kind:** `control`, `faceoff`, or the loose-puck origin (`loose_opp_miss`, `loose_own_block`, `rebound`, …). Kinds matter: a won faceoff in the offensive zone and a takeaway on the same spot lead to different outcomes. Keeping kinds separate prevents faceoff dots, which carry about a third of all starts, from contaminating the control surface.
* **spatial:** three candidate schemes were compared (Phase 4): 26 named sub-zones, 10 × 8.5 ft cells (200), and 20 × 17 ft cells (50). Loose kinds whose recorded location is not where the puck was recovered (misses, saves, rebounds) use only the coarse zone. `loose_unobserved` has no location.

### 4.4 Estimators and selection (Phases 4–5)

| Approach | Implementation |
|---|---|
| **A. Markov / Monte Carlo** | Tabular fitted value iteration $V=\text{shrink}(\mathbb E[r + V(\text{next})])$. It equals the Monte Carlo mean when there are no mid observations (the default), because possessions are then single-step chains. Tested analytically in `test_markov_*`. |
| **B. Hazard** | Discrete-time competing risks (scoring shot vs other end) over 8 time bins, multinomial logit on one-hot state × bin, times $\mathbb E[\text{xG}\mid\text{shot},s,b]$. |
| **C. GBM** | Poisson-loss histogram GBM on continuous x, y, distance, angle, kind, strength (± score). |

**Shrinkage.** Hierarchical empirical Bayes, state → parent zone → kind → strength group. Each level's prior strength is estimated by method of moments at that level. A bug was found and fixed here: an earlier version used one k for all levels, which over-shrank finer grids (R² 0.141 vs 0.201; see Phase 5 summary). There is a regression test.

**Selection protocol, fixed before results.** Train 2017–21, validate 2022, test 2023–25. The primary criterion is validation MSE of V vs $G_p$. Candidates within 2 paired standard errors of the best are ties, broken by season-to-season stability of V and then by parsimony. The chosen spec is scored once on the test seasons. Results, including the chosen spec, are in [validation report §1](../../reports/validation_report.md).

**Cross-fitting.** The V model that values season Y is trained on other seasons in the same scoring regime. It never sees season Y, which removes in-sample leakage into player metrics (D7).

### 4.5 Shot xG (`src/xg/`)
Features: distance, angle, x, |y|, shot type, previous event (type, team, time since, distance, speed), rebound (same-team attempt ≤ 3 s), rush (previous event outside the attacking zone ≤ 4 s), strength, empty net, own goalie pulled, score (±3), OT, off-wing (roster handedness), home. A logistic spline model and a histogram GBM were trained on 2017–21 and compared on 2022. **GBM chosen** (validation log-loss 0.2269 vs 0.2318; AUC 0.763 vs 0.753). Final xG is cross-fitted by season: all-season AUC 0.773, calibration-in-the-large 1.008. Details are in validation report §1.

## 5. Attribution (`src/epa/engine.py`)

See [formulas §5–6](formulas.md) and the worked examples.

| Component | Credited to | In PM-EPA |
|---|---|---|
| carry | carrier (same player at both ends ≤ 12 s) | direct |
| move_start / move_end | endpoint players, 0.35 each of Δ (start share 0 for faceoff winners and unknown starts) | direct |
| move_pool | remaining share, split equally over on-ice skaters at the segment's end event | pooled |
| takeaway | taker: $V_{\text{start},T}(s)+\lambda V_{\text{dec},O}(\bar s)$ | direct |
| takeaway_against | loser's on-ice skaters: $-(V_{\text{dec},O}(\bar s)+\lambda V_{\text{start},T}(s))$ | pooled |
| giveaway | giver: $-(V_{\text{dec},T}(s)+\lambda V_{\text{start},O}(\text{loose\_opp\_giveaway}))$ | direct |
| unobserved_loss | on-ice skaters when the loss was revealed | pooled |
| stoppage_loss | on-ice skaters at an icing or offside | pooled |
| shot_attempt | shooter: $V_\text{att}(s)-V_\text{dec}(s)$ | direct |
| shot_quality | shooter: $\text{xG}-V_\text{att}(s)$ | **excluded** (reported) |
| unattributed | faceoff and loose-recovery start values, rebound segments, neutral stoppages, pool rows without shift data | excluded |

**Why these choices:**

* **Faceoff winners get no start share.** The draw is a state reset (brief §4.2); winners rarely carry.
* **Shot quality is excluded; the shot attempt is not.** The release (shot type, getting blocked, rebound luck) is the outcome state in brief §4.2, and excluding it keeps PM-EPA from becoming a finishing proxy. Discriminant validity is checked against SH%. The *attempt* term is the success side of the same decision whose failure side is the giveaway, so both are included (see §4.2).
* **λ = 0 (production), decided by evidence.** The project started at λ = 1 (EP symmetry: a giveaway also hands the opponent value). Under the pre-registered rule λ = 0 won decisively:
  - team-changer YOY (mean of F and D): 0.438 vs 0.372 at λ = 1;
  - r with G+A1/60 rose from 0.06/0.14 to 0.40/0.35 (F/D);
  - |r| with individual SH% stayed ≤ 0.25.

  The mechanism: at λ = 1 the opponent-value term dominated each giveaway's cost (~3× the V_dec part). Nothing on the positive side balances it, because the passes and carries that offset a puck-carrier's giveaways are unobserved in PBP, so the metric drifted toward an inverted giveaway rate. λ = 0 is also the brief's own primary turnover spec (§7.3.5, V(turnover) = 0). The opponent's gain is not lost from the system: it appears as the opponent's next possession start value.

  **Cost of this choice, stated plainly:** at λ = 0 a defensive-zone giveaway costs ≈ 0.0003 xG (V_dec there is ≈ 0), against ≈ 0.05 extra at λ = 1. The leaderboards therefore also ship `pm_epa_per60_ev_lam1`, the risk-adjusted λ = 1 rate, as a companion column. See [limitations §2](limitations.md#2-modelling-limitations).
* **Shares 0.35/0.35/0.30.** These encode symmetric ignorance about who moved the puck in an unobserved segment, while keeping a majority of credit individual. Alternatives from 0/0 (all pool) to 0.5/0.5 (no pool) and end-weighted splits are ablated. The default is kept only if no alternative is clearly better on stability and validity (validation report §6).

**Pre-registered share-selection rule** (written 2026-09-27, before any ablation was run). Among the share variants {0.35/0.35, 0.5/0.5, 0.2/0.2, 0/0, 0.15/0.55}, choose the one with the highest **team-changer YOY r** (mean of F and D). A player who changes teams gets new linemates and a new system, so this is the test most resistant to teammate and system effects. The winner must also satisfy two constraints: (a) its r with G+A1/60 is within 0.05 of the default's (convergent validity preserved), and (b) its |r| with individual SH% is ≤ 0.25 (discriminant preserved). If the winner beats the default by less than 0.02, keep the default (parsimony / avoid noise-chasing). The same rule applies to λ ∈ {0, 0.5, 1}.

**Accounting identity.** With λ = 0, every possession's components sum exactly to $G_p$. This is tested on toy data and verified on all 1.9M possessions (max |residual| in `reports/metrics/identity_check.json`).

## 6. Player metrics (`src/metrics/aggregate.py`)

* Per-60 rates use shift-chart TOI of the matching strength over **usable games** (10,901 of 11,052).
* **EV = 5v5**, both goalies in net.
* Leaderboards qualify at ≥500 5v5 minutes (≥200 PP/PK for special-teams columns).
* 95% CIs come from a game-block bootstrap (1,000 resamples).
* Percentiles are within season and position group (F/D) among qualified players.
* Direct and pooled PM-EPA/60 are exported separately so users can see how much of a number is individually observed.

## 7. Validation

The full suite and results are in [reports/validation_report.md](../../reports/validation_report.md): V and xG calibration; YOY stability overall, within scoring regime, across the regime break, and for team-changers; split-half; convergent and discriminant correlations; next-season predictive validity; 5v5 xG RAPM alignment with incremental R²; team-level sums; transition sanity; TOI flattening; rink effects; and ablations.

## 8. Assumptions register

1. Scorer-recorded coordinates and events are unbiased *in expectation within a rink-season* (known to be violated; limitations §1.2).
2. The possessing team is the team of the most recent possession-bearing event (standard PBP inference).
3. Recorded shot, giveaway and takeaway locations are where the puck was.
4. Players on ice at a segment's end event are the plausible contributors to that segment's unobserved movement.
5. The value of a state is stationary within a scoring regime (V cross-fitted across seasons).
6. xG is well calibrated within season: the cross-fitted per-season calibration-in-the-large ranges 0.97–1.08.
