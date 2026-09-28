# Literature Review — Possession Value, Expected Points and Expected Goals

*Phase 1 deliverable. Purpose: decide what PM-EPA borrows, what it rejects, and why.*

> **Citation hygiene.** Every work below was either verified against a primary source during this review or is marked **[unverified]**. The project brief named a few sources we could not locate (see §7). We say so rather than invent a summary.

---

## 1. Expected Points (EP/EPA) in American football — the template

| Work | Core idea | Strengths | Weaknesses for our purpose |
|---|---|---|---|
| Carter & Machol (1971), *Operations Research on Football*, Operations Research 19(2) | First EP table: expected next-score value of a first-and-10 as a function of yard line | Defines value as the expectation of the *next scoring event*, not the drive outcome | Coarse state; no uncertainty |
| Carroll, Palmer & Thorn (1988), *The Hidden Game of Football* | Popularized EP as a linear function of field position; "points added" per play | Simple, interpretable, delta-based player credit | Linear approximation; no context |
| Yurko, Ventura & Horowitz (2019), *nflWAR: A reproducible method for offensive player evaluation in football*, JQAS 15(3) | Multinomial logistic EP model over next-score type (7 classes); EPA = EP_after − EP_before; WPA; credit split between passer/receiver via air yards vs YAC; WAR via multilevel models | Reproducible (open code, nflscrapR); principled split of a single transition into *two causal pieces* (air vs YAC); explicit replacement level | Football has discrete plays with full observation of the ball; hockey does not |
| nflfastR EP/WP models (Baldwin & Carl) | XGBoost EP with calibration plots by down/quarter | Shows GBM + calibration diagnostics is acceptable when explained | Black-box pieces need calibration evidence (we adopt the diagnostics) |

**Adopted.** (i) `EPA = V(after) − V(before)` with V the expectation of a *scoring currency* from the current state; (ii) turnovers valued through the opponent's EP (EP after a turnover is the negative of the opponent's EP) — our λ·V_opp term; (iii) nflWAR's idea of decomposing one transition into components attributable to different actors (air yards vs YAC ↔ our start-share / end-share / on-ice pool); (iv) calibration-by-bin diagnostics.

**Rejected / adapted.** Football EP targets the *next score by either team*; the brief defines hockey V(s) over the *current possession* (goals are rare: ~0.03 per possession). Using next-goal-by-either-team would be dominated by noise. We keep the brief's possession-scoped definition and add the opponent term only at possession changes.

## 2. Hockey shot quality / expected goals

| Work | Contribution | Notes |
|---|---|---|
| Ryder (2004), *Shot Quality* (hockeyanalytics.com) | First public shot-quality model (distance, shot type, rebound) | Established location+type as dominant features |
| Krzywicki (2005–2010), shot-quality regression series | Logistic model incl. rebounds, strength | Early evidence that rebounds and rushes matter |
| Macdonald (2012), *An Expected Goals Model for Evaluating NHL Teams and Players*, MIT SSAC | Team/player xG using shots, missed/blocked shots, faceoffs, hits, giveaways/takeaways as predictors of future goals | Early use of *non-shot* events as value signals — a direct ancestor of possession value in hockey |
| Thomas & Ventura, War-On-Ice (2014) | Scoring-chance definition (home-plate area, rebounds, rushes) | Danger zones we reuse only as a descriptive sub-zone |
| Perry, Corsica (2016) and Younggren & Younggren, Evolving-Hockey (2018 blog: *A New Expected Goals Model for Predicting Goals in the NHL*) | GBM xG with distance, angle, time since last event, prior event type/location, rebound/rush, strength, score | Public benchmark; distance, angle, seconds-since-last are top features |
| Tanner, MoneyPuck xG | GBM with rebound angle change, speed from previous event, off-wing | Well-calibrated public reference |
| McCurdy, HockeyViz ("Magnus") | Spatial shot-rate/threat surfaces with smoothing; isolated impacts | Strong argument for spatial smoothing of rate surfaces |

**Adopted.** A clean in-house xG (logistic baseline + GBM challenger) using exactly the public-standard feature family (distance, angle, shot type, rebound, rush, prior-event context, strength, off-wing, score). Time-based holdouts; isotonic calibration; reliability diagrams. We train our own model because V(s) needs xG for *every* shot in our event table, including seasons and events that public tables do not expose per-event.

**Rejected.** Pulling public xG values into training (leakage/licensing, and incompatible event joins).

## 3. Expected Possession Value (EPV) — basketball and soccer

| Work | Core idea | Strengths | Weaknesses |
|---|---|---|---|
| Cervone, D'Amour, Bornn & Goldsberry (2016), *A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes*, JASA 111(514) | EPV as expectation of possession points; coarse Markov "macro" states + fine-resolution micro model; value added = ΔEPV | The multiresolution idea: coarse, well-estimated Markov transitions for macro-states, fine models only where data supports them | Requires optical tracking |
| Fernández, Bornn & Cervone (2019/2021), *Decomposing the Immeasurable Sport* (SSAC 2019; Machine Learning 110) | Soccer EPV decomposed into pass/drive/shot components, each with its own probability & value surface | Decomposition by action type; separate success-probability models for risk | Tracking-data dependent |
| Fernández & Bornn (2018), *Wide Open Spaces*, SSAC | Pitch control / space value | Pressure/space concept → our `pressure_proxy` roadmap | Needs tracking; out of reach with PBP |
| Rudd (2011), *A framework for tactical analysis … using Markov chains*, NESSIS | Markov chain over pitch zones on event data | **Event-data Markov value is feasible** | Coarse zones |
| Singh (2018), *Introducing Expected Threat (xT)* (blog) | Grid (16×12) Markov value iteration on event data: xT(z) = P(shot|z)·xG(z) + P(move|z)·Σ T(z→z′)xT(z′) | Transparent, event-data-native, linear-algebra solvable; the closest structural analogue to our Approach A | No context; no turnover penalty; grid chosen by taste |
| Decroos, Bransen, Van Haaren & Davis (2019), *Actions Speak Louder than Goals: Valuing Player Actions in Soccer* (VAEP), KDD | Action value = Δ[P(score soon) − P(concede soon)] via classifiers on the last k actions | Offense+defense in one currency; action-level; model-agnostic | Short fixed horizon ("next 10 actions") is arbitrary; results-based labels (goals) |
| Lawrence (2018), StatsBomb *xGChain / xGBuildup* | Give every player in a possession the possession's xG (xGChain) or the same excluding shot/assist (xGBuildup) | Simple, stable | *Duplicates* credit (sum across players ≫ team xG) — explicitly warned against in the brief |

**Adopted.** Singh's xT linear-system form for Approach A (with context and turnovers added); Cervone's multiresolution idea as hierarchical shrinkage (coarse zone → fine cell); Fernández's action-type decomposition as our component reporting (movement vs turnover vs shot attempt vs shot quality); VAEP's two-sided currency *only* at possession changes (λ term).

**Rejected.** xGChain-style duplication (credit must be conserved: Σ player credit = team value change); VAEP's goal-based labels (we use xG, guardrail §14).

## 4. Hockey possession value and Markov models

| Work | Core idea | Strengths | Weaknesses |
|---|---|---|---|
| Thomas, Ventura, Jensen & Ma (2013), *Competing process hazard function models for player ratings in ice hockey*, AoAS 7(3), doi:10.1214/13-AOAS646 | Goal scoring as competing semi-Markov processes; hazards depend on on-ice players | Principled continuous-time treatment; the template for our Approach B (competing risks: shot vs loss) | Player-level hazards need strong regularization |
| Schuckers & Curro (2013), *Total Hockey Rating (THoR)*, MIT SSAC | Every PBP event valued by probability of a goal in the next 20 s given event type and location, adjusted for zone start & teammates | Values non-shot events; location-aware; ridge-regularized player model | Fixed 20 s window mixes possessions; values *events*, not *transitions* |
| Kaplan, Mongeon & Ryan (2014), *A Markov model for hockey: manpower differential and win probability added*, INFOR 52(2) | Markov chain over manpower/score states | Strength states belong in the state | No spatial dimension |
| Routley & Schulte (2015), *A Markov Game Model for Valuing Player Actions in Ice Hockey*, UAI 2015 (pp. 782–791); extended in Schulte et al. (2017), DMKD | Markov game over PBP (2.8M events) with context (score, manpower, period) and action history; value iteration; action impact = Q(s,a) − V(s) | **Closest prior work.** Shows PBP event sequences support a context-aware Markov value and per-action impact | Large sparse state space (history-based); impact is per *event*, not per *movement between events*; reward on goals |
| Liu & Schulte (2018), *Deep Reinforcement Learning in Ice Hockey for Context-Aware Player Evaluation* (Goal Impact Metric, GIM), IJCAI | LSTM Q-function over possession history (Sportlogiq data); GIM = Σ ΔQ | Temporal consistency of GIM reported best among 7 metrics | Proprietary data; black box; goal-based reward |
| Kumagai, Nahabedian, Châtel & Stokes (2021), *Bayesian Space-Time Models for Expected Possession Added Value* (Hockey Graphs, Big Data Cup) | Spatiotemporal (INLA, Matérn) transition models for pass/shot/turnover over OZ sequences; simulate xPV; PAV = Δ xPV | Full action vocabulary incl. passes; spatial smoothing via Gaussian fields; xG as reward | Stathletes data (not NHL-wide), OZ-only |
| Châtel (2020), *Using Sequences for Analysis: Expected Goals Contribution* (Hockey Graphs) | Sequence xG divided among contributors by event importance | Conserves credit | Hand-tracked non-NHL data; weights heuristic |
| Tulsky, Detweiler, Spencer & Sznajder (2013), *Using Zone Entry Data To Separate Offensive, Neutral, and Defensive Zone Performance*, MIT SSAC | Hand-tracked 330 games; controlled entries yield >2× shots/goals vs dump-ins; shot-differential talent lives mainly in the neutral zone | **Transition play is where repeatable skill lives** — strong prior that a movement metric can carry signal | Not reproducible from public PBP |
| Sznajder, *All Three Zones* project; Parnass, PP tracking (2016–17); Stimson, *Passing Project* (2015–17) | Manual tracking of entries/exits/passes/shot assists | Establish convergent targets (shot assists, controlled exits) | Paywalled or discontinued; partial coverage — **not used** as inputs; referenced for construct validity |
| Radke, Radke, Brecht & Pawelczyk (2021), *Passing and Pressure Metrics in Ice Hockey*, AI for Sports Analytics workshop | Pass/pressure metrics from NHL puck & player tracking | Shows exactly what tracking would unlock (pressure, pass options) | NHL tracking not public |
| Macdonald (2011), *A regression-based adjusted plus-minus statistic for NHL players*, JQAS 7(3); Gramacy, Jensen & Taddy (2013), JQAS 9(1) | Ridge / regularized on-ice regression (RAPM) | Our convergent target for "predicts on-ice impact"; our ridge implementation follows these | Not an event-level metric |

**Adopted.**
* Routley & Schulte's premise: PBP supports a context-aware Markov value. We simplify their state to *location × strength (× score if it earns its keep)* and value *movement between events* rather than event types.
* THoR's lesson that non-shot events carry location value — but we value **transitions** (Δ between consecutive in-possession states) rather than events with a fixed 20 s window, so value is not double-counted across overlapping windows.
* Thomas et al.'s competing-risks framing as Approach B.
* Kumagai et al.'s spatial smoothing — implemented as empirical-Bayes cell shrinkage + optional Gaussian kernel (INLA is overkill for a tabular grid).
* Macdonald/Gramacy ridge RAPM as an external validation target.

## 5. Gaps PM-EPA addresses

1. **Public, reproducible, transition-level value on NHL data.** Routley–Schulte and GIM value *events*; xT-style transition value has not been published end-to-end on NHL PBP with transparent attribution and validation.
2. **Credit conservation with honest uncertainty about who moved the puck.** Existing public chain metrics either duplicate credit (xGChain) or require passes. We split each unobserved segment between observed endpoint players and an on-ice pool, and *measure* how individual the result is (YOY stability on team-changers).
3. **Separating movement from finishing.** Shot-decision value (xG − V at shot location) and finishing (G − xG) are split out so movement credit is not a proxy for shooting luck.
4. **Validation as a gate.** Reliability, YOY, split-half, convergent/discriminant, RAPM alignment and ablations — reported with numbers.

## 6. Chosen approach vs rejected alternatives

| Decision | Chosen | Rejected | Justification |
|---|---|---|---|
| Value currency | Possession-scoped expected goals (xG of terminal shot) | Goals; next-goal-either-team | xG removes finishing luck (guardrail); possession scope matches brief and keeps variance manageable |
| V(s) estimator | Markov linear system (A) as primary; MC, hazard (B), GBM (C) as challengers | Deep RL (GIM) | Interpretability, reproducibility, data size; challengers let evidence overturn |
| State | spatial cell × strength (score by ablation) | Full history (Routley–Schulte) | State explosion; history is partly captured by rebound/rush flags in xG |
| Attribution | Observed carry = 100%; unobserved segments split start/end/on-ice pool (shares chosen by evidence) | Equal credit to all involved (xGChain); all credit to shooter | Conservation + honesty about unobserved passes |
| Turnovers | Swing incl. λ·V_opp, λ ablated | Ignore opponent | EP symmetry: own-zone giveaways must cost more than OZ ones |
| Smoothing | EB shrinkage to parent zone + monotonicity diagnostics | Unregularized cell means | Sparse PK/3v3 cells |

## 7. Brief references we could not verify

| As named in brief | Status |
|---|---|
| "Shaw (Shot Quality / possession chains)" | **[unverified]** — no hockey possession-chain work by an author named Shaw located. Not used. |
| "Tulsky et al. (GVT/metrics)" | GVT (Goals Versus Threshold) is Tom Awad's (Hockey Prospectus), not Tulsky's. Tulsky's relevant work is the 2013 zone-entry paper, cited above. |
| "Dellow" | Tyler Dellow's (mc79hockey) shot-attempt/possession-proxy work is well known but predates possession *value*; cited only as context. |
| "Garik Parparyan (pbpstats)" | **[unverified]** — no such author found; pbpstats.com is a *basketball* PBP site (Darryl Blackport). Not used. |
| "Evolving-Hockey's xG Chains / xG Generated" | **[unverified]** as a published Evolving-Hockey method. "xGChain" is StatsBomb's (Lawrence, 2018) soccer metric, reviewed above. |
| "Manny Perry (Corsica)" | Corsica's xG model (2016) is confirmed as a public xG; no zone-entry model by Perry was located. |

## References (links)

* Routley & Schulte (2015) — https://www.auai.org/uai2015/proceedings/papers/130.pdf
* Liu & Schulte (2018) — https://www.ijcai.org/proceedings/2018/478
* Thomas, Ventura, Jensen & Ma (2013) — https://arxiv.org/abs/1208.0799
* Kumagai et al. (2021) — https://hockey-graphs.com/2021/07/06/bayesian-space-time-models-for-expected-possession-added-value-part-1-of-2/
* Châtel (2020) — https://hockey-graphs.com/2020/05/13/using-sequences-for-analysis-expected-goals-contribution-and-more/
* Tulsky et al. (2013) — http://hockeyanalytics.com/Research_files/Using%20Zone%20Entry%20Data%20To%20Separate%20Offensive,%20Neutral,%20And%20Defensive%20Zone%20Performance.pdf
* Radke et al. (2021) — https://cs.uwaterloo.ca/~brecht/papers/hockey-aisa-2021.pdf
* Evolving-Hockey xG (2018) — https://evolving-hockey.com/blog/a-new-expected-goals-model-for-predicting-goals-in-the-nhl/
* Yurko, Ventura & Horowitz (2019) — https://arxiv.org/abs/1802.00998
* Decroos et al. (2019) — https://arxiv.org/abs/1802.07127
* Cervone et al. (2016) — https://arxiv.org/abs/1408.0777
