## Interpretation

**Result: the player-rating model is the best model tested, but the main gate fails by a hair.** On 6,502 test games (2021–2025) the player model P has a log loss of 0.6585 against 0.6613 for the better team rating (G2, a tuned team rating on a blend of goals and expected goals)
and 0.6655 for a team rating on goals alone (G1). P beats G1 clearly (z = 3.40) and G2 by 0.0028 with z = 1.99 (1.93 with date clusters), just under the pre-registered 2, so **GM-2 fails**. It is better than G2 in four of the five seasons (not 2022).
GM-1 (against home ice, z = 9.6), GM-3 (in-season updating, z = 5.7), GM-4 (player attribution, z = 2.4) and GM-5 (calibration slope 0.91) pass. The fair reading: a real but small edge over a well-tuned team rating, at the edge of what five seasons can establish.

**Where the gain comes from.**
* *The preseason player prior alone is not enough.* With no in-season update (P-pre) the model is worse than the team rating (0.6692 against 0.6613, z = −3.7). A team rating that simply carries over from last season knows more on opening night than the summed player projections do, the same finding as the standings backtest, now on 6,500 games.
* *In-season updating is most of the value* (0.6692 → 0.6585).
* *Attributing the updates to players rather than to teams helps* (P against P-team: 0.0013, z = 2.4). Ratings that move with the players handle trades, call-ups and absences better than a team-level state.
* *Lineup news added nothing measurable* (P against the previous game's lineup: 0.0001, z = 0.07). In this model who dresses on a given night barely moves the forecast; the skaters' ratings within a team are similar enough, and the minutes of absent players small enough, that swapping the lineup for the last one changes little. That limits the edge available from reacting to lineup announcements with this model.

**What the tuning chose.** The skater update uses expected goals only (α = 0), with 4% of each game's surprise absorbed per team; the goalie prior is heavy (3,000 shots). So game-to-game goals carry no extra information for skater ratings beyond expected goals, and goalies should be moved slowly. The team benchmarks keep their full rating between seasons (ρ = 1).

**What it means for the larger goal.** The game-level test has the power the standings test lacked: differences of 0.001–0.003 in log loss are now detectable. It confirms that player-level structure adds something beyond a team rating, mostly through updating during the season, and that the preseason player model is the weak part. The two remaining candidates should be judged here: pre-NHL priors for young players (which act on the preseason prior) and the chemistry model (which acts on how lineups combine).

**Market.** No odds are in the repository, so nothing is claimed. The comparison is built: put de-vigged closing probabilities in `data/external/odds.csv` (`game_date, home, away, p_home_close`) and rerun.

**Limitations.** Hyperparameters were tuned on two unusual seasons; lineups are taken from who played (known at puck drop, not earlier); games are treated as independent in the z statistics (date-clustered values are nearly identical); no travel, schedule density beyond back-to-backs, or score-effect adjustment; the priors inherit the player values' full-data hyperparameters.
