"""Estimators of the possession value function V(s) (brief §6).

V(s) = E[ xG of the current possession's terminal shot | puck alive under the possessing
team's control in state s ]. Terminal value is the xG of an unblocked shot (0 for blocked
shots, turnovers, stoppages). Goals are *not* used as reward (guardrail §14).

Training rows are *alive* observations (role start or mid): the puck is controlled and the
possession continues. Terminal observations are decision outcomes, not states to value;
using them as origins would bias V toward where events get recorded (see methodology §4.3).

Estimators (all expose ``fit(df)`` / ``predict(df)`` on observation-like frames):
  TabularValue(method='mc')      empirical-Bayes-shrunk state means of the return G
  TabularValue(method='markov')  fitted value iteration V = shrink(E[r + V(next)]) — the
                                 Markov/TD form; identical to 'mc' when possessions have
                                 no mid observations (the default PBP case)
  HazardValue                    discrete-time competing risks (shot vs other end) + E[xG|shot]
  GBMValue                       HistGradientBoosting on continuous location + kind + context
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from src.states.mapping import STRENGTH_GROUP, add_state_columns, location_precision, state_kind

KIND_LEVELS = [
    "control", "faceoff", "rebound", "loose_unobserved", "loose_opp_giveaway", "loose_own_giveaway",
    "loose_opp_block", "loose_own_block", "loose_opp_miss", "loose_own_miss", "loose_opp_sog",
    "loose_own_sog", "loose_opp_goal", "loose_own_goal",
]
SG_LEVELS = ["5v5", "PP", "PK", "4v4_3v3"]


def _eb_shrink(n: np.ndarray, mean: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    return (n * mean + k * prior) / (n + k) if k > 0 else np.where(n > 0, mean, prior)


def method_of_moments_k(n: np.ndarray, mean: np.ndarray, var_within: float) -> float:
    """James–Stein / EB prior strength k = sigma^2_within / tau^2_between (weighted)."""
    ok = n > 0
    if ok.sum() < 3:
        return 50.0
    w = n[ok]
    mu = np.average(mean[ok], weights=w)
    between = np.average((mean[ok] - mu) ** 2, weights=w) - var_within * np.mean(1.0 / w)
    tau2 = max(between, 1e-8)
    return float(np.clip(var_within / tau2, 1.0, 1e5))


class TabularValue:
    """Tabular V over state ids with hierarchical EB shrinkage state -> parent -> kind -> strength."""

    def __init__(self, scheme: str = "zones", k: float | str = "auto", method: str = "mc", use_score: bool = False,
                 max_iter: int = 50):
        self.scheme, self.k, self.method, self.use_score, self.max_iter = scheme, k, method, use_score, max_iter
        self.k_used: float | None = None

    def _prep(self, df: pl.DataFrame) -> pl.DataFrame:
        if "state_id" in df.columns and getattr(self, "_prepped_scheme", None) == (self.scheme, self.use_score):
            return df
        return add_state_columns(df, self.scheme, self.use_score)

    def _levels(self, d: pl.DataFrame, target: np.ndarray) -> dict:
        d = d.with_columns(pl.Series("_y", target))
        agg = lambda key: d.group_by(key).agg(pl.len().alias("n"), pl.col("_y").mean().alias("m"))  # noqa: E731
        return {"sg": agg("sgroup"), "kind": agg("kind_id"), "parent": agg("parent_id"), "state": agg("state_id")}

    def _k(self, level: pl.DataFrame, var_within: float) -> float:
        """Prior strength for one level of the hierarchy: fixed, or method of moments *at that level*."""
        if self.k != "auto":
            return float(self.k)
        return method_of_moments_k(level["n"].to_numpy().astype(float), level["m"].to_numpy(), var_within)

    def _solve(self, lv: dict, var_within: float) -> dict:
        sg = dict(zip(lv["sg"]["sgroup"], lv["sg"]["m"]))
        k_kind, k_par, k_state = (self._k(lv[x], var_within) for x in ("kind", "parent", "state"))
        self.k_used = {"kind": k_kind, "parent": k_par, "state": k_state}
        kind_df = lv["kind"].with_columns(pl.col("kind_id").str.split("|").list.first().alias("_sg"))
        kind_prior = np.array([sg.get(s, 0.0) for s in kind_df["_sg"]])
        kind_v = dict(zip(kind_df["kind_id"], _eb_shrink(kind_df["n"].to_numpy(), kind_df["m"].to_numpy(), kind_prior, k_kind)))
        par = lv["parent"].with_columns(pl.col("parent_id").str.split("|").list.slice(0, 2).list.join("|").alias("_k"))
        par_prior = np.array([kind_v.get(s, 0.0) for s in par["_k"]])
        par_v = dict(zip(par["parent_id"], _eb_shrink(par["n"].to_numpy(), par["m"].to_numpy(), par_prior, k_par)))
        return {"sg": sg, "kind": kind_v, "parent": par_v, "k": k_state}

    def fit(self, df: pl.DataFrame, target_col: str = "G") -> TabularValue:
        d = self._prep(df)
        self.parent_of = dict(zip(d["state_id"], d["parent_id"]))
        self.kind_of = dict(zip(d["state_id"], d["kind_id"]))
        g = d[target_col].to_numpy().astype(float)
        var_within = float(np.var(g))
        has_next = "next_idx" in d.columns and self.method == "markov" and d["next_idx"].is_not_null().any()
        target = g.copy()
        if has_next:
            nxt = d["next_idx"].to_numpy()
            nxt_mask = ~np.isnan(nxt.astype(float))
            nxt_i = np.where(nxt_mask, nxt, 0).astype(int)
        for _ in range(self.max_iter if has_next else 1):
            lv = self._levels(d, target)
            pri = self._solve(lv, var_within)
            st = lv["state"]
            prior = np.array([pri["parent"].get(self.parent_of[s], pri["kind"].get(self.kind_of[s], 0.0)) for s in st["state_id"]])
            vals = _eb_shrink(st["n"].to_numpy(), st["m"].to_numpy(), prior, pri["k"])
            new_table = dict(zip(st["state_id"], vals))
            if not has_next:
                break
            # Markov/TD target: reward if terminal next, else V(next state)
            v_obs = np.array([new_table[s] for s in d["state_id"]])
            new_target = np.where(nxt_mask, v_obs[nxt_i], d["R"].to_numpy())
            if np.max(np.abs(new_target - target)) < 1e-9:
                break
            target = new_target
        self.table, self.pri = new_table, pri
        self.n = dict(zip(st["state_id"], st["n"]))
        self.raw = dict(zip(st["state_id"], st["m"]))
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Lookup with fallback state -> parent -> kind -> strength group (vectorized joins)."""
        d = add_state_columns(df, self.scheme, self.use_score).select("state_id", "parent_id", "kind_id", "sgroup").with_row_index("_i")
        lk = lambda m, key: pl.DataFrame({key: list(m.keys()), f"v_{key}": np.array(list(m.values()), dtype=float)}, schema={key: pl.Utf8, f"v_{key}": pl.Float64})  # noqa: E731
        d = (d.join(lk(self.table, "state_id"), on="state_id", how="left")
             .join(lk(self.pri["parent"], "parent_id"), on="parent_id", how="left")
             .join(lk(self.pri["kind"], "kind_id"), on="kind_id", how="left")
             .join(lk(self.pri["sg"], "sgroup"), on="sgroup", how="left")
             .sort("_i"))
        return d.select(pl.coalesce("v_state_id", "v_parent_id", "v_kind_id", "v_sgroup", pl.lit(0.0))).to_series().to_numpy()

    def _predict_loop(self, df: pl.DataFrame) -> np.ndarray:
        """Reference (slow) implementation kept for testing the vectorized path."""
        d = add_state_columns(df, self.scheme, self.use_score)
        out = np.empty(d.height)
        for i, (s, p, k, sg) in enumerate(zip(d["state_id"], d["parent_id"], d["kind_id"], d["sgroup"])):
            v = self.table.get(s)
            if v is None:
                v = self.pri["parent"].get(p)
            if v is None:
                v = self.pri["kind"].get(k)
            if v is None:
                v = self.pri["sg"].get(sg, 0.0)
            out[i] = v
        return out

    def to_frame(self) -> pl.DataFrame:
        ids = list(self.table)
        parts = [s.split("|") for s in ids]
        return pl.DataFrame({
            "state_id": ids,
            "sgroup": [p[0] for p in parts],
            "kind": [p[1] for p in parts],
            "spatial": [p[2] for p in parts],
            "n": [int(self.n[s]) for s in ids],
            "v_raw": [float(self.raw[s]) for s in ids],
            "v_parent": [float(self.pri["parent"].get(self.parent_of[s], np.nan)) for s in ids],
            "v": [float(self.table[s]) for s in ids],
        })


def _features(df: pl.DataFrame, use_score: bool) -> np.ndarray:
    kind = state_kind(df["obs_kind"].to_numpy())
    prec = location_precision(kind)
    x = df["x"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    y = df["y"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    # imprecise / missing locations -> NaN except coarse sign (own/opp end)
    coarse_x = np.where(np.isnan(x), np.nan, np.sign(x) * 60)
    x = np.where(prec == "fine", x, np.where(prec == "coarse", coarse_x, np.nan))
    y = np.where(prec == "fine", y, np.nan)
    dist = np.sqrt((89 - x) ** 2 + y**2)
    ang = np.degrees(np.arctan2(np.abs(y), 89 - x))
    kind_code = np.array([KIND_LEVELS.index(k) if k in KIND_LEVELS else -1 for k in kind], dtype=float)
    sg = df["strength"].replace_strict(STRENGTH_GROUP, default="other").to_numpy()
    sg_code = np.array([SG_LEVELS.index(s) if s in SG_LEVELS else -1 for s in sg], dtype=float)
    cols = [x, y, np.abs(y), dist, ang, kind_code, sg_code]
    if use_score:
        cols.append(np.clip(df["score_diff"].fill_null(0).to_numpy(), -3, 3).astype(float))
    return np.column_stack(cols)


class GBMValue:
    """Approach C: gradient-boosted regression of G on continuous state features."""

    def __init__(self, use_score: bool = False, seed: int = 0, max_iter: int = 300):
        self.use_score = use_score
        self.model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
            l2_regularization=1.0, categorical_features=[5, 6], early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=seed,
        )

    def fit(self, df: pl.DataFrame, target_col: str = "G") -> GBMValue:
        X = _features(df, self.use_score)
        X[:, 5:7] = np.where(X[:, 5:7] < 0, np.nan, X[:, 5:7])
        self.model.fit(X, df[target_col].to_numpy())
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        X = _features(df, self.use_score)
        X[:, 5:7] = np.where(X[:, 5:7] < 0, np.nan, X[:, 5:7])
        return self.model.predict(X)


HAZ_EDGES = np.array([0, 2, 4, 7, 11, 16, 25, 45, np.inf])


class HazardValue:
    """Approach B: discrete-time competing-risks model.

    For an alive observation in state s, the possession ends in time bin b with a *scoring
    shot* (unblocked attempt; carries xG) or an *other end* (block, turnover, stoppage), or
    survives. Cause-specific hazards h_c(b|s) come from a multinomial logit on one-hot
    (parent state, bin) features; E[xG | shot, s, b] from shrunk cell means. Then
        V(s) = sum_b S(b-1|s) * h_shot(b|s) * m(s, b),   S(b|s) = prod_{j<=b} (1 - h_shot - h_other).
    """

    def __init__(self, scheme: str = "zones", max_rows: int = 1_500_000, seed: int = 0):
        self.scheme, self.max_rows, self.seed = scheme, max_rows, seed

    def _expand(self, d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = np.clip(d["tau"].to_numpy(), 0, None)
        last_bin = np.searchsorted(HAZ_EDGES, tau, side="right") - 1
        last_bin = np.clip(last_bin, 0, len(HAZ_EDGES) - 2)
        end_cls = d["end_class"].to_numpy()  # 1 = scoring shot, 2 = other end, 0 = censored
        rows = np.repeat(np.arange(d.height), last_bin + 1)
        bins = np.concatenate([np.arange(b + 1) for b in last_bin])
        is_last = np.concatenate([np.arange(b + 1) == b for b in last_bin])
        y = np.where(is_last, end_cls[rows], 0)
        return rows, bins, y

    def _design(self, pid_codes: np.ndarray, bins: np.ndarray, coarse_codes: np.ndarray) -> sparse.csr_matrix:
        n = len(bins)
        nb = len(HAZ_EDGES) - 1
        a = sparse.csr_matrix((np.ones(n), (np.arange(n), pid_codes)), shape=(n, self.n_pid))
        b = sparse.csr_matrix((np.ones(n), (np.arange(n), bins)), shape=(n, nb))
        c = sparse.csr_matrix((np.ones(n), (np.arange(n), coarse_codes * nb + bins)), shape=(n, self.n_coarse * nb))
        return sparse.hstack([a, b, c]).tocsr()

    def fit(self, df: pl.DataFrame, target_col: str = "G") -> HazardValue:
        d = add_state_columns(df, self.scheme)
        rng = np.random.default_rng(self.seed)
        rows, bins, y = self._expand(d)
        if len(rows) > self.max_rows:
            keep = rng.choice(len(rows), self.max_rows, replace=False)
            rows, bins, y = rows[keep], bins[keep], y[keep]
        self.pid_levels = {p: i for i, p in enumerate(sorted(set(d["parent_id"].to_list()) | set(d["kind_id"].to_list())))}
        self.coarse_levels = {c: i for i, c in enumerate(sorted(set(d["kind_id"].to_list())))}
        self.n_pid, self.n_coarse = len(self.pid_levels), len(self.coarse_levels)
        pid = np.array([self.pid_levels[p] for p in d["parent_id"]])
        kid = np.array([self.coarse_levels[k] for k in d["kind_id"]])
        X = self._design(pid[rows], bins, kid[rows])
        self.clf = LogisticRegression(C=1.0, max_iter=300).fit(X, y)
        # E[xG | scoring-shot end, parent, bin] with shrinkage to parent mean
        shot = d.filter(pl.col("end_class") == 1).with_columns(
            pl.Series("bin", np.clip(np.searchsorted(HAZ_EDGES, d.filter(pl.col("end_class") == 1)["tau"].to_numpy(), side="right") - 1, 0, len(HAZ_EDGES) - 2))
        )
        overall = shot["terminal_xg"].mean()
        par = shot.group_by("parent_id").agg(pl.len().alias("n"), pl.col("terminal_xg").mean().alias("m"))
        self.m_par = dict(zip(par["parent_id"], _eb_shrink(par["n"].to_numpy(), par["m"].to_numpy(), np.full(par.height, overall), 50)))
        pb = shot.group_by("parent_id", "bin").agg(pl.len().alias("n"), pl.col("terminal_xg").mean().alias("m"))
        prior = np.array([self.m_par[p] for p in pb["parent_id"]])
        self.m_pb = {(p, b): v for p, b, v in zip(pb["parent_id"], pb["bin"], _eb_shrink(pb["n"].to_numpy(), pb["m"].to_numpy(), prior, 50))}
        self.overall = overall
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        d = add_state_columns(df, self.scheme)
        nb = len(HAZ_EDGES) - 1
        uniq = d.select("parent_id", "kind_id").unique()
        vals = {}
        for p, k in zip(uniq["parent_id"], uniq["kind_id"]):
            pid = self.pid_levels.get(p, self.pid_levels.get(k, 0))
            kid = self.coarse_levels.get(k, 0)
            X = self._design(np.full(nb, pid), np.arange(nb), np.full(nb, kid))
            pr = self.clf.predict_proba(X)
            cls = list(self.clf.classes_)
            h_shot = pr[:, cls.index(1)] if 1 in cls else np.zeros(nb)
            h_other = pr[:, cls.index(2)] if 2 in cls else np.zeros(nb)
            S = np.concatenate([[1.0], np.cumprod(1 - h_shot - h_other)[:-1]])
            m = np.array([self.m_pb.get((p, b), self.m_par.get(p, self.overall)) for b in range(nb)])
            vals[(p, k)] = float(np.sum(S * h_shot * m))
        return np.array([vals[(p, k)] for p, k in zip(d["parent_id"], d["kind_id"])])
