# PM-EPA Formula Reference

Notation: team $T$ possesses and $O$ is the opponent. $s$ is a state in $T$'s attacking frame and $\bar s$ is the same location seen from $O$'s frame (mirrored, with strength PP↔PK flipped). $\lambda$ = `value.lambda_opponent`.

## 1. Shot expected goals

$$
\mathrm{xG}(a) = \Pr(\text{goal} \mid a) = f_{\text{GBM}}\big(d_a, \theta_a, \text{type}_a, \text{rebound}_a, \text{rush}_a, \Delta t_a, \dots\big),\qquad \mathrm{xG}(\text{blocked}) \equiv 0
$$

Cross-fitted: $f^{(-y)}$ is trained on seasons $\neq y$ and scores season $y$.

## 2. Possession return and value function

For possession $p$ with terminal event $e_p$:

$$
G_p = \begin{cases}\mathrm{xG}(e_p) & e_p \in \{\text{SOG},\ \text{miss},\ \text{goal}\}\\ 0 & \text{otherwise (block, turnover, stoppage)}\end{cases}
$$

$$
V(s) = \mathbb{E}\left[G_p \mid \text{puck alive under } T\text{'s control in state } s\right]
$$

**Decision-point value** (terminal observations: shot, giveaway, lost puck), and the **attempt value**:

$$
V_{\text{dec}}(s) = \mathbb{E}\left[G_p \mid p\text{'s terminal decision occurs at } s\right],
\qquad
V_{\text{att}}(s) = \mathbb{E}\left[\mathrm{xG} \mid \text{shot attempt at } s\right]\ (\text{blocked} = 0)
$$

V (above, written $V_{\text{start}}$ where ambiguous) values starts and mids. $V_{\text{dec}}$ values the arrival (decision) state of the last segment.

**Markov form** over alive observations $i$ with next alive observation $i^{+}$:

$$
V(s) = \frac{1}{n_s}\sum_{i\,:\,s_i = s}\Big(\mathbb{1}[i^{+}=\varnothing]\,G_{p(i)} + \mathbb{1}[i^{+}\neq\varnothing]\,V(s_{i^{+}})\Big)
\;\;\Longleftrightarrow\;\; (I-P)\,V = r
$$

With no mid observations ($P = 0$) this reduces to the Monte Carlo mean $\bar G_s$.

## 3. Hierarchical empirical-Bayes shrinkage

Levels: state (cell) $\subset$ parent (zone or coarse) $\subset$ kind $\subset$ strength group.

$$
\hat V_{\ell}(s) = \frac{n_s\,\bar G_s + k_\ell\,\hat V_{\ell+1}(\text{parent}(s))}{n_s + k_\ell},
\qquad
k_\ell = \frac{\hat\sigma^2_{\text{within}}}{\hat\tau^2_{\ell}},\quad
\hat\tau^2_{\ell} = \operatorname{Var}_w(\bar G_s) - \hat\sigma^2_{\text{within}}\,\overline{1/n_s}
$$

The between-state variance $\tau^2$ is estimated separately at each level (method of moments).

## 4. Hazard (competing-risks) alternative

$$
V(s) = \sum_{b} S(b-1\mid s)\; h_{\text{shot}}(b\mid s)\; m(s,b),
\qquad S(b\mid s) = \prod_{j\le b}\big(1 - h_{\text{shot}}(j\mid s) - h_{\text{other}}(j\mid s)\big)
$$

$h_c$ comes from a multinomial logit over time bins; $m(s,b) = \mathbb{E}[\mathrm{xG}\mid \text{shot end}, s, b]$.

## 5. Movement segment

For consecutive observations $o_i \to o_{i+1}$ in one possession:

$$
\Delta_i = V\big(s^{\text{arr}}_{i+1}\big) - V\big(s_i\big)
$$

Here $V(s^{\text{arr}}_{i+1}) = V_{\text{dec}}(s_{i+1})$ when $o_{i+1}$ is a terminal decision point (shot, giveaway, lost puck), and $V_{\text{start}}$ otherwise.

**Credit weights** ($\alpha$ = `start_share`, $\beta$ = `end_share`, $g_i$ = time gap):

| segment type | condition | $w_{\text{start}}$ | $w_{\text{end}}$ | $w_{\text{pool}}$ |
|---|---|---|---|---|
| carry | $P_i = P_{i+1}$ known, $g_i \le 12$ s, not a faceoff start | 1 | 0 | 0 |
| split | otherwise, $g_i \le 30$ s | $\alpha\,\mathbb{1}[P_i \text{ known, not faceoff}]$ | $\beta\,\mathbb{1}[P_{i+1}\text{ known}]$ | $1 - w_{\text{start}} - w_{\text{end}}$ |
| long gap | $g_i > 30$ s | 0 | 0 | 1 |
| rebound | $o_i$ is a rebound start | — (unattributed) | — | — |

$$
\text{credit}(P_i) \mathrel{+}= w_{\text{start}}\Delta_i,\quad
\text{credit}(P_{i+1}) \mathrel{+}= w_{\text{end}}\Delta_i,\quad
\text{credit}(q) \mathrel{+}= \frac{w_{\text{pool}}\,\Delta_i}{|\mathcal{O}_{i+1}|}\ \ \forall q\in\mathcal{O}_{i+1}
$$

$\mathcal{O}_{i+1}$ is the set of $T$'s skaters on ice at the end event.

## 6. Possession-change terms

$$
\begin{aligned}
\text{takeaway (taker)} &:\; V_{\text{start},T}(s) + \lambda\,V_{\text{dec},O}(\bar s)\\
\text{takeaway against (loser's on-ice pool)} &:\; -\big(V_{\text{dec},O}(\bar s) + \lambda\,V_{\text{start},T}(s)\big)\\
\text{giveaway (giver)} &:\; -\big(V_{\text{dec},T}(s) + \lambda\,V_{\text{start},O}(\text{loose\_opp\_giveaway at }\bar s)\big)\\
\text{shot attempt (shooter)} &:\; V_{\text{att}}(s) - V_{\text{dec}}(s)\\
\text{unobserved loss (on-ice pool at end event)} &:\; -\big(V_T(s_{\text{last}}) + \lambda\,V_O(\text{loose\_unobserved})\big)\\
\text{icing/offside (on-ice pool at stoppage)} &:\; -V_T(s_{\text{last}})\\
\text{shot quality (shooter; excluded)} &:\; \mathrm{xG}(e_p) - V_{\text{att}}(s_{\text{shot}})
\end{aligned}
$$

## 7. Player metric

$$
\text{PM-EPA}_j = \sum_{c\,\in\,\mathcal C}\ \sum_{\text{rows}} \text{credit}_{j,c},
\qquad \mathcal C = \{\text{carry, move\_start, move\_end, move\_pool, takeaway, takeaway\_against, giveaway, shot\_attempt, unobserved\_loss, stoppage\_loss}\}
$$

$$
\text{PM-EPA/60}_{j}^{\text{EV}} = 60\cdot\frac{\sum_{g\in\mathcal U} \text{PM-EPA}^{5v5}_{j,g}}{\sum_{g\in\mathcal U}\text{TOI}^{5v5}_{j,g}}
$$

$\mathcal U$ = usable games (valid shift data).

**Uncertainty:** game-block bootstrap. Resample player $j$'s games with replacement $B$ = 1000 times; the 95% CI is the 2.5–97.5% quantiles of the resampled ratio.

## 8. Accounting identity (λ = 0)

$$
\underbrace{V(s_0)}_{\text{start}} + \sum_i \Delta_i + \underbrace{\tau_p}_{\text{terminal term}} = G_p
\qquad\text{for every possession } p
$$

The terminal term $\tau_p$ is $(V_{\text{att}}-V_{\text{dec}})+(\mathrm{xG}-V_{\text{att}})$ for shots and $-V$ otherwise.

**Decision terms are mean-zero at a spot (λ = 0).** Here $V_{\text{dec}}(s)=\Pr(\text{attempt}\mid s)\,V_{\text{att}}(s)$, so $\Pr(\text{attempt})(V_{\text{att}}-V_{\text{dec}}) + \Pr(\text{loss})(0-V_{\text{dec}}) = 0$. The start value is $0$ plus the value at the first observation for unobserved starts. Verified on the full dataset in `reports/metrics/identity_check.json`.
