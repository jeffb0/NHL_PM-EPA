# Pre-NHL priors for young and thin-history players — methodology and pre-registration

*Written 2026-10-01 before any pre-NHL data were fetched or any model fitted. The design, tests and rules in §5–§6 are fixed here; changes after results are seen are logged in §9.
Motivation: the preseason player prior is the weak part of both the standings forecast and the game-level model (docs/gamemodel: the preseason-only model loses to a team rating, z = −3.7), and a player with
little or no NHL history currently gets only a "no history" indicator and his age.*

## 1. Idea (ported from baseball's minor-league equivalencies and basketball draft models)

A player's production in junior, college, European and minor leagues, translated to an NHL-equivalent rate, and his draft position are informative about his NHL value before he has an NHL record, and their
weight should fade as his NHL record grows.

## 2. Data

`https://api-web.nhle.com/v1/player/<id>/landing` for every skater in the player-value panel and on the 2026-27 roster snapshot: `seasonTotals` (league, season, games, goals, assists, points for every league a player has
appeared in) and `draftDetails` (year, overall pick). Stored raw under `data/raw/player_landing/`. The snapshot is taken now, but every feature for season T uses only rows from seasons that started before T
(and the draft, which precedes them); no NHL row from season T or later is used.

## 3. Features for a player entering season T

* **League translation factor** f_L(T): for league L, over players with a season s in L (≥ 20 games) followed by an NHL season s + 1 (≥ 20 games), s + 1 < T, the ratio of mean NHL points per game to mean league points per game,
  shrunk toward the all-league ratio with 10 pseudo-transitions. Estimated only from transitions completed before T (expanding).
* **NHLe:** the player's points per game in his most recent non-NHL regular season among T − 1 and T − 2 (≥ 10 games; the league with the most games if several) times f_L(T). `has_nhle` = 1 if one exists.
* **Draft:** `pick` = log(overall pick), undrafted players set to log(250) with `undrafted` = 1.
* **Fade:** every feature is multiplied by w = M0 / (m_eff + M0), where m_eff is the player's discounted NHL minutes in the projection layer and M0 = 1,000 minutes (the layer's own ballast). A player with no NHL history has w = 1; a three-season regular has w ≈ 0.2.

Four added columns: `w·nhle`, `w·has_nhle`, `w·pick`, `w·undrafted` (within the position-specific ridge; the `no_history` indicator and age terms stay).

## 4. Model

The projection layer's ridge (docs/projection §3, training floor 20 minutes as published) with the four columns added through its covariate hook; everything else unchanged. Called **M+Y**; the current model is **M**.

## 5. Tests

1. **Player level (direct):** expanding-window forecasts for target seasons 2020–2025 as in docs/projection §4. Minute-weighted squared error of the next-season rate, M+Y against M, paired z from a player-cluster bootstrap;
   on **thin-history** player-seasons (m_eff < 1,000 minutes, ≥ 200 target-season minutes) and on all players (≥ 300 target-season minutes).
2. **Game level (the sensitive test):** the game model of docs/gamemodel with the preseason ratings from M+Y instead of M, hyperparameters as already tuned and frozen. Paired z on per-game log loss over the 6,502 test games for
   the preseason-only model (P-pre) and for the full model (P).
3. **Standings (reported):** the published standings model (M1 + C1 + C3) with M+Y rates, RMSE and paired z.
4. **Reported:** the fitted coefficients, the translation factors of the main leagues, the largest changes among 2026-27 roster players.

## 6. Gates and the implementation rule

| gate | rule |
|---|---|
| YP-1 | M+Y beats M on thin-history player-seasons (paired z > 2) |
| YP-2 | M+Y improves the preseason-only game model (paired z > 2) |
| YP-3 | no harm: M+Y is not worse than M on all players (z > −2) and the full game model is not worse (z > −2) |

Implementation follows the author's standing rule (docs/standings/improvements.md §4). The methodology is rated **sound** in advance (equivalency translations and draft priors are standard and the fade is principled),
so: implement if YP-3 holds and neither YP-1 nor YP-2 has z ≤ −1; a "reliable improvement" is claimed only for a gate with z > 2. "Implement" means M+Y becomes the rate model behind the standings forecast and the game model's
preseason ratings; the published `player_projection.csv` is not changed in this step.

## 7. What the result can say

A YP-1 pass means pre-NHL information improves projections for players without an NHL record. A YP-2 pass means that matters for teams. Thin-history players get few minutes, so a player-level gain with
no detectable game-level gain is a plausible outcome and would be reported as such.

## 8. Limitations stated in advance

Points-based equivalencies describe offense, not defense, so the gain should be smaller for defensemen; league factors ignore age, team strength and era; the endpoint's league names are free text (minor spelling variants are separate leagues);
players who never reached the NHL are not in the panel, so the model is conditional on being rostered; the target is the model's own value estimate (player-level test) and game outcomes (game-level test).

## 9. Deviations log

*(any change to §3–§6 after results are seen is recorded here with its reason)*

1. **Correction (not a design change):** §5 first said the game test has 6,493 games; the count is 6,502 (misquoted from the game-model run).
2. **Outcome, applied mechanically (2026-10-01):** YP-1 fails (thin-history z = 1.36), YP-2 fails (preseason-only game model z = −0.25), YP-3 passes (all players z = 2.62; full game model z = −0.54).
   Neither YP-1 nor YP-2 is ≤ −1 and the method was rated sound, so by §6 the priors are **implemented** in the standings forecast (`src/standings/next.py` reads the decision from `reports/metrics/prospects.json`).
   The reported standings check is slightly negative (13.96 → 13.99, z = −1.39); it was a reported item, not part of the rule, and is flagged in the report. The game-model validation report keeps the original rate model
   (its pre-registered results are not re-run); `gamemodel.data.build_priors(..., yp=...)` takes the priors for forward use.
