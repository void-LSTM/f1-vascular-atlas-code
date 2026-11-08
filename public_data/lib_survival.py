"""lib_survival.py - survival analysis and design-based (survey) regression in numpy/scipy only.

Paper F1, Stage 0 analysis library. Written for the HPP `pheno` environment (Python 3.11, numpy 1.26, pandas 2.1,
scipy 1.11): no statsmodels, lifelines or R. Validated only on simulated data with planted answers
(`validate_lib_survival.py`, results in `results/stage0/validation/`).

Contents
--------
Survey design
    SurveyDesign            weights + strata + PSU (PSUs nested in strata), lonely-PSU handling, domain weights
    design_vcov             Taylor-linearisation variance of a sum of influence values
                            (PSU totals, between-PSU variance within strata, factor n_h / (n_h - 1))
    robust_vcov             sandwich "meat" with independent units or clusters (no centring, no small-sample factor)
Cox proportional hazards
    coxph                   Efron or Breslow ties, Newton-Raphson with step halving, case weights, strata,
                            delayed entry (left truncation; enables age as the time scale), model-based, robust
                            (optionally clustered) and design-based (Binder 1992) variance; score residuals
    cox_zph                 scaled Schoenfeld residuals and the Grambsch-Therneau test against g(time)
Non-parametric
    km                      Kaplan-Meier (weights, delayed entry), Greenwood or design-based SE
    cuminc                  1 - KM or Aalen-Johansen cumulative incidence at given times, with design-based SE
    svy_cuminc_by_group     survey-weighted cumulative incidence by group (domain = group, weights zeroed outside)
Survey / regression
    svy_mean                survey-weighted mean or proportion (domain estimation), linearisation SE
    glm_fit                 linear and logistic regression with model, HC0/HC1, cluster (CR1) or design-based SE
Utilities
    rcs_basis               restricted cubic spline basis (Harrell parameterisation)
    wald_test               multi-parameter Wald test
    weighted_sd             (survey-)weighted standard deviation

Conventions
-----------
* Risk set at an event time t: entry < t <= time (subjects entering at t are not at risk at t; subjects censored
  at t are at risk at t). Every record must have time > entry.
* Domain (subpopulation) analysis: keep every record of the design, set the weight to 0 outside the domain. Zero-
  weight records do not enter the point estimate (identical to sub-setting) but their PSUs and strata stay in the
  variance computation (PSU totals of 0), which is the correct design-based variance for a random domain size.
* Influence values passed to `design_vcov` are already multiplied by the case weight (they sum to the linearised
  estimator deviation), e.g. dfbeta_i = w_i * U_i * I^-1 for Cox.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

Z975 = stats.norm.ppf(0.975)


# ================================================================================================================
# Survey design and variance helpers
# ================================================================================================================
def _codes(a: np.ndarray) -> tuple[np.ndarray, int]:
    """Integer codes 0..K-1 for the distinct values of a (any dtype)."""
    _, inv = np.unique(np.asarray(a), return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1 if len(inv) else 0


@dataclass
class SurveyDesign:
    """Stratified multistage design summarised by first-stage units (PSUs), with-replacement approximation.

    weights : sampling weights (0 allowed: records outside a domain).
    strata  : stratum identifiers (None = one stratum).
    psu     : PSU identifiers (None = every record is its own PSU). Nested within strata when `nest=True`
              (NHANES PSU codes 1, 2, 3 repeat across strata).
    lonely_psu : 'adjust' (centre a single-PSU stratum at the grand mean of PSU totals, as R survey 'adjust'),
                 'remove' (contributes 0) or 'fail'.
    """
    weights: np.ndarray
    strata: np.ndarray | None = None
    psu: np.ndarray | None = None
    nest: bool = True
    lonely_psu: str = "adjust"
    psu_code: np.ndarray = field(init=False, repr=False)
    psu_stratum: np.ndarray = field(init=False, repr=False)
    n_psu: int = field(init=False)
    n_strata: int = field(init=False)

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        n = len(self.weights)
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise ValueError("weights must be finite and >= 0")
        st = np.zeros(n, dtype=np.int64) if self.strata is None else _codes(self.strata)[0]
        if self.psu is None:
            ps_key = np.arange(n)
        else:
            p = _codes(self.psu)[0]
            ps_key = st * (int(p.max()) + 1) + p if self.nest else p
        self.psu_code, self.n_psu = _codes(ps_key)
        # stratum of each PSU (a PSU must lie in a single stratum)
        mx = np.full(self.n_psu, -1, dtype=np.int64)
        mn = np.full(self.n_psu, np.iinfo(np.int64).max, dtype=np.int64)
        np.maximum.at(mx, self.psu_code, st)
        np.minimum.at(mn, self.psu_code, st)
        if np.any(mx != mn):
            raise ValueError("a PSU spans several strata; use nest=True or fix the design variables")
        self.psu_stratum = mx
        self.n_strata = int(st.max()) + 1 if n else 0
        if self.lonely_psu not in ("adjust", "remove", "fail"):
            raise ValueError("lonely_psu must be 'adjust', 'remove' or 'fail'")

    @property
    def df(self) -> int:
        """Design degrees of freedom: number of PSUs minus number of strata."""
        return int(self.n_psu - self.n_strata)

    def domain(self, mask) -> "SurveyDesign":
        """Same design with weights set to 0 outside the domain (mask False or NaN)."""
        m = np.asarray(mask)
        m = np.where(pd.isna(m), False, m).astype(bool)
        d = SurveyDesign.__new__(SurveyDesign)
        d.weights = np.where(m, self.weights, 0.0)
        d.strata, d.psu, d.nest, d.lonely_psu = self.strata, self.psu, self.nest, self.lonely_psu
        d.psu_code, d.psu_stratum, d.n_psu, d.n_strata = self.psu_code, self.psu_stratum, self.n_psu, self.n_strata
        return d

    def take(self, idx) -> "SurveyDesign":
        """Design restricted to records idx (rows physically removed; use `domain` for subpopulations)."""
        idx = np.asarray(idx)
        return SurveyDesign(self.weights[idx], None if self.strata is None else np.asarray(self.strata)[idx],
                            None if self.psu is None else np.asarray(self.psu)[idx], self.nest, self.lonely_psu)


def design_vcov(infl: np.ndarray, design: SurveyDesign) -> np.ndarray:
    """Linearisation variance  sum_h n_h/(n_h-1) sum_j (z_hj - zbar_h)(z_hj - zbar_h)^T,  z_hj = PSU totals of infl.

    infl: (n,) or (n, p) influence values already multiplied by the sampling weight.
    """
    a = np.asarray(infl, dtype=float)
    one_d = a.ndim == 1
    if one_d:
        a = a[:, None]
    p = a.shape[1]
    z = np.zeros((design.n_psu, p))
    np.add.at(z, design.psu_code, a)
    nh = np.bincount(design.psu_stratum, minlength=design.n_strata).astype(float)
    sh = np.zeros((design.n_strata, p))
    np.add.at(sh, design.psu_stratum, z)
    with np.errstate(invalid="ignore", divide="ignore"):
        mh = sh / nh[:, None]
    dev = z - mh[design.psu_stratum]
    n_of_psu = nh[design.psu_stratum]
    fac = np.where(n_of_psu > 1, n_of_psu / np.maximum(n_of_psu - 1, 1), 0.0)
    lonely = n_of_psu == 1
    if np.any(lonely):
        if design.lonely_psu == "fail":
            raise ValueError(f"{int(lonely.sum())} stratum/strata with a single PSU")
        if design.lonely_psu == "adjust":
            dev[lonely] = z[lonely] - z.mean(axis=0)
            fac[lonely] = 1.0
    v = (dev * fac[:, None]).T @ dev
    return v[0, 0] if one_d else v


def robust_vcov(infl: np.ndarray, cluster=None) -> np.ndarray:
    """Sandwich meat sum_c (sum_{i in c} infl_i)(...)^T (independent units if cluster is None). No centring."""
    a = np.asarray(infl, dtype=float)
    one_d = a.ndim == 1
    if one_d:
        a = a[:, None]
    if cluster is not None:
        c, k = _codes(cluster)
        z = np.zeros((k, a.shape[1]))
        np.add.at(z, c, a)
        a = z
    v = a.T @ a
    return v[0, 0] if one_d else v


def _ci_table(coef, se, names, alpha=0.05, exp=False, df=None, scale=None) -> pd.DataFrame:
    coef = np.asarray(coef, float)
    se = np.asarray(se, float)
    if scale is not None:
        s = np.broadcast_to(np.asarray(scale, float), coef.shape)
        coef, se = coef * s, se * np.abs(s)
    q = stats.t.ppf(1 - alpha / 2, df) if df is not None else stats.norm.ppf(1 - alpha / 2)
    zval = coef / se
    p = 2 * (stats.t.sf(np.abs(zval), df) if df is not None else stats.norm.sf(np.abs(zval)))
    out = pd.DataFrame({"term": list(names), "coef": coef, "se": se, "lower": coef - q * se,
                        "upper": coef + q * se, "z": zval, "p": p})
    if exp:
        out["exp_coef"] = np.exp(out["coef"])
        out["exp_lower"] = np.exp(out["lower"])
        out["exp_upper"] = np.exp(out["upper"])
    return out


def wald_test(coef, vcov, idx=None, L=None, value=None) -> dict:
    """Wald chi-square test of L beta = value (default: the coefficients idx are all 0)."""
    coef = np.asarray(coef, float)
    vcov = np.asarray(vcov, float)
    if L is None:
        idx = np.atleast_1d(idx)
        L = np.zeros((len(idx), len(coef)))
        L[np.arange(len(idx)), idx] = 1.0
    L = np.atleast_2d(L)
    r = L @ coef - (0 if value is None else np.asarray(value, float))
    m = L @ vcov @ L.T
    stat = float(r @ np.linalg.solve(m, r))
    df = L.shape[0]
    return {"chi2": stat, "df": df, "p": float(stats.chi2.sf(stat, df))}


def weighted_sd(x, w=None) -> float:
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    m = np.isfinite(x) & (w > 0)
    x, w = x[m], w[m]
    mu = np.sum(w * x) / np.sum(w)
    return float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))


def rcs_basis(x, knots) -> np.ndarray:
    """Restricted cubic spline basis (Harrell): columns x, then k-2 non-linear terms (scaled by (t_k - t_1)^2)."""
    x = np.asarray(x, float)
    t = np.sort(np.asarray(knots, float))
    k = len(t)
    if k < 3:
        raise ValueError("need >= 3 knots")
    sc = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        term = (np.clip(x - t[j], 0, None) ** 3
                - np.clip(x - t[k - 2], 0, None) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + np.clip(x - t[k - 1], 0, None) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / sc)
    return np.column_stack(cols)


# ================================================================================================================
# Cox proportional hazards
# ================================================================================================================
@dataclass
class CoxResult:
    coef: np.ndarray
    var_model: np.ndarray
    var_robust: np.ndarray | None
    var_design: np.ndarray | None
    names: list
    loglik: float
    loglik_null: float
    n: int
    n_events: int
    weighted_events: float
    iterations: int
    converged: bool
    ties: str
    score_resid: np.ndarray            # (n_all, p) unweighted per-record score residuals (0 for zero weight)
    info: np.ndarray
    zph_data: dict
    design_df: int | None = None
    warnings: list = field(default_factory=list)

    @property
    def var(self) -> np.ndarray:
        """Default variance: design-based if a design was given, else robust if requested, else model-based."""
        if self.var_design is not None:
            return self.var_design
        if self.var_robust is not None:
            return self.var_robust
        return self.var_model

    def table(self, which: str = "default", alpha: float = 0.05, scale=None, use_t: bool = False) -> pd.DataFrame:
        """Coefficient table. use_t=True: t(design df) intervals/p for design-based variance (default normal,
        as R svycoxph); recommended for small domains (see validation V9)."""
        v = {"default": self.var, "model": self.var_model, "robust": self.var_robust,
             "design": self.var_design}[which]
        df = self.design_df if (use_t and self.design_df) else None
        out = _ci_table(self.coef, np.sqrt(np.diag(v)), self.names, alpha=alpha, exp=True, scale=scale, df=df)
        out.insert(1, "var_type", which if which != "default" else
                   ("design" if self.var_design is not None else "robust" if self.var_robust is not None
                    else "model"))
        return out

    def dfbeta(self, weighted: bool = True) -> np.ndarray:
        """Influence of each record on beta: w_i * U_i * I^-1 (weighted) or U_i * I^-1."""
        return self.zph_data["w_all"][:, None] * self.score_resid @ self.var_model if weighted else \
            self.score_resid @ self.var_model


def _suffix(a: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Suffix sums of a[order] along axis 0, with a zero row appended (index n -> 0)."""
    s = np.cumsum(a[order][::-1], axis=0)[::-1]
    z = np.zeros((1,) + a.shape[1:])
    return np.concatenate([s, z], axis=0)


class _CoxStratum:
    """Pre-computed, beta-independent structures for one stratum."""

    def __init__(self, t, e0, d, w, X, efron: bool):
        self.t, self.e0, self.d, self.w, self.X = t, e0, d.astype(bool), w, X
        self.order_t = np.argsort(t, kind="mergesort")
        self.order_e = np.argsort(e0, kind="mergesort")
        t_sorted, e_sorted = t[self.order_t], e0[self.order_e]
        ev = self.d
        self.ut = np.unique(t[ev])
        K = len(self.ut)
        self.K = K
        self.idx_t = np.searchsorted(t_sorted, self.ut, side="left")
        self.idx_e = np.searchsorted(e_sorted, self.ut, side="left")
        self.ev_idx = np.flatnonzero(ev)
        self.ev_k = np.searchsorted(self.ut, t[ev])
        self.nD = np.bincount(self.ev_k, minlength=K).astype(float)
        self.wD = np.bincount(self.ev_k, weights=w[ev], minlength=K)
        if efron:
            nd = self.nD.astype(int)
            self.rep = np.repeat(np.arange(K), nd)
            starts = np.repeat(np.cumsum(nd) - nd, nd)
            self.f = (np.arange(len(self.rep)) - starts) / self.nD[self.rep]
            self.mw = (self.wD / self.nD)[self.rep]
        else:
            self.rep = np.arange(K)
            self.f = np.zeros(K)
            self.mw = self.wD.copy()
        # per-record event-time window for residuals: event times k with entry < ut_k <= t
        self.kmin = np.searchsorted(self.ut, e0, side="right")
        self.kmax = np.searchsorted(self.ut, t, side="right")
        self.sum_wx_ev = (w[ev][:, None] * X[ev]).sum(axis=0)

    def pieces(self, beta, need_info=True):
        X, w = self.X, self.w
        p = X.shape[1]
        eta = X @ beta
        c = float(eta.max()) if len(eta) else 0.0
        r = np.exp(eta - c)
        v0 = w * r
        v1 = v0[:, None] * X
        S0 = _suffix(v0, self.order_t)[self.idx_t] - _suffix(v0, self.order_e)[self.idx_e]
        S1 = _suffix(v1, self.order_t)[self.idx_t] - _suffix(v1, self.order_e)[self.idx_e]
        ev, k = self.ev_idx, self.ev_k
        S0D = np.bincount(k, weights=v0[ev], minlength=self.K)
        S1D = np.zeros((self.K, p))
        np.add.at(S1D, k, v1[ev])
        rep, f = self.rep, self.f
        denom = S0[rep] - f * S0D[rep]
        denom = np.maximum(denom, 1e-300)
        num1 = S1[rep] - f[:, None] * S1D[rep]
        xbar = num1 / denom[:, None]
        ll = float(np.sum(w[ev] * eta[ev]) - np.sum(self.mw * (np.log(denom) + c)))
        score = self.sum_wx_ev - (self.mw[:, None] * xbar).sum(axis=0)
        info = None
        if need_info:
            v2 = v1[:, :, None] * X[:, None, :]
            S2 = _suffix(v2, self.order_t)[self.idx_t] - _suffix(v2, self.order_e)[self.idx_e]
            S2D = np.zeros((self.K, p, p))
            np.add.at(S2D, k, v2[ev])
            num2 = S2[rep] - f[:, None, None] * S2D[rep]
            info = np.einsum("j,jab->ab", self.mw / denom, num2) - np.einsum("j,ja,jb->ab", self.mw, xbar, xbar)
        return ll, score, info, dict(r=r, denom=denom, xbar=xbar)

    def residuals(self, beta):
        """Score residuals (unweighted, per record) and Schoenfeld residuals (per event record)."""
        _, _, _, aux = self.pieces(beta, need_info=False)
        r, denom, xbar = aux["r"], aux["denom"], aux["xbar"]
        X, K, p = self.X, self.K, self.X.shape[1]
        hz = self.mw / denom
        rep, f = self.rep, self.f
        h = np.bincount(rep, weights=hz, minlength=K)
        hf = np.bincount(rep, weights=f * hz, minlength=K)
        xh = np.zeros((K, p))
        np.add.at(xh, rep, hz[:, None] * xbar)
        xhf = np.zeros((K, p))
        np.add.at(xhf, rep, (f * hz)[:, None] * xbar)
        xbar_avg = np.zeros((K, p))
        np.add.at(xbar_avg, rep, xbar)
        xbar_avg /= np.bincount(rep, minlength=K)[:, None]
        Hc = np.concatenate([[0.0], np.cumsum(h)])
        XHc = np.concatenate([np.zeros((1, p)), np.cumsum(xh, axis=0)])
        Hi = Hc[self.kmax] - Hc[self.kmin]
        XHi = XHc[self.kmax] - XHc[self.kmin]
        res = -r[:, None] * (X * Hi[:, None] - XHi)
        ev, k = self.ev_idx, self.ev_k
        res[ev] += X[ev] - xbar_avg[k] + r[ev][:, None] * (X[ev] * hf[k][:, None] - xhf[k])
        schoen = X[ev] - xbar_avg[k]
        return res, schoen


def coxph(time, event, X, *, entry=None, weights=None, strata=None, ties: str = "efron", robust: bool = False,
          cluster=None, design: SurveyDesign | None = None, names=None, init=None, max_iter: int = 50,
          eps: float = 1e-9, max_halving: int = 30) -> CoxResult:
    """Cox proportional-hazards model.

    time, event : follow-up end and event indicator (0/1).
    X           : (n, p) covariates (no intercept). Rows with NaN must be removed or given weight 0 beforehand
                  (NaN covariates in zero-weight rows are replaced by 0 internally).
    entry       : delayed-entry time (left truncation); default 0. Risk set: entry < t <= time.
    weights     : case weights (frequency or sampling weights). Overridden by design.weights if design is given.
    strata      : baseline-hazard strata.
    ties        : 'efron' or 'breslow'.
    robust      : also compute the robust sandwich variance (clustered if cluster is given).
    design      : SurveyDesign -> design-based (Taylor linearisation) variance; domain = records with weight > 0.
    """
    ties = ties.lower()
    if ties not in ("efron", "breslow"):
        raise ValueError("ties must be 'efron' or 'breslow'")
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n_all, p = X.shape
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]
    if design is not None:
        w_all = design.weights.astype(float)
        if len(w_all) != n_all:
            raise ValueError("design and data lengths differ")
    else:
        w_all = np.ones(n_all) if weights is None else np.asarray(weights, float)
    entry = np.zeros(n_all) if entry is None else np.asarray(entry, float)
    st_all = np.zeros(n_all, dtype=np.int64) if strata is None else _codes(strata)[0]
    pos = w_all > 0
    bad = pos & (~np.isfinite(time) | ~np.isfinite(entry) | ~np.isfinite(event) | ~np.all(np.isfinite(X), axis=1))
    if np.any(bad):
        raise ValueError(f"{int(bad.sum())} records with positive weight have missing time/event/covariates")
    if np.any(pos & (time <= entry)):
        raise ValueError("every record with positive weight needs time > entry")
    if not set(np.unique(event[pos])) <= {0.0, 1.0}:
        raise ValueError("event must be 0/1")
    idx = np.flatnonzero(pos)
    w = w_all[idx]
    Xp = X[idx]
    mu = np.average(Xp, axis=0, weights=w)
    Xc = Xp - mu
    strata_objs = []
    for s in np.unique(st_all[idx]):
        m = st_all[idx] == s
        strata_objs.append((np.flatnonzero(m), _CoxStratum(time[idx][m], entry[idx][m], event[idx][m], w[m],
                                                           Xc[m], ties == "efron")))

    def total(beta, need_info=True):
        ll, sc, inf = 0.0, np.zeros(p), np.zeros((p, p))
        for _, so in strata_objs:
            a, b, c, _ = so.pieces(beta, need_info)
            ll += a
            sc += b
            if need_info:
                inf += c
        return ll, sc, inf

    beta = np.zeros(p) if init is None else np.asarray(init, float).copy()
    ll0, _, _ = total(np.zeros(p), need_info=False)
    ll, sc, inf = total(beta)
    converged, it, warn = False, 0, []
    for it in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(inf, sc)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(inf, sc, rcond=None)[0]
            warn.append("singular information matrix during iterations")
        new = beta + step
        ll_new, sc_new, inf_new = total(new)
        h = 0
        while (not np.isfinite(ll_new) or ll_new < ll - 1e-12 * abs(ll)) and h < max_halving:
            step /= 2
            new = beta + step
            ll_new, sc_new, inf_new = total(new)
            h += 1
        done = abs(ll_new - ll) <= eps * max(1.0, abs(ll_new))
        beta, ll, sc, inf = new, ll_new, sc_new, inf_new
        if done:
            converged = True
            break
    if not converged:
        warn.append(f"did not converge in {max_iter} iterations")
    if np.any(np.abs(beta) > 20):
        warn.append("very large coefficient (possible monotone likelihood / separation)")
    try:
        imat_inv = np.linalg.inv(inf)
    except np.linalg.LinAlgError:
        imat_inv = np.linalg.pinv(inf)
        warn.append("singular final information matrix (pseudo-inverse used)")
    resid = np.zeros((n_all, p))
    zt, zs, zw = [], [], []
    for loc, so in strata_objs:
        rres, schoen = so.residuals(beta)
        resid[idx[loc]] = rres
        zt.append(so.t[so.ev_idx])
        zs.append(schoen)
        zw.append(so.w[so.ev_idx])
    var_robust = var_design = None
    infl = (w_all[:, None] * resid) @ imat_inv
    if robust or cluster is not None:
        var_robust = robust_vcov(infl, cluster)
    if design is not None:
        var_design = design_vcov(infl, design)
    n_ev = int(np.sum(event[idx] == 1))
    zph = dict(time=np.concatenate(zt) if zt else np.zeros(0),
               schoen=np.vstack(zs) if zs else np.zeros((0, p)),
               w=np.concatenate(zw) if zw else np.zeros(0), w_all=w_all,
               fit_time=time[idx], fit_event=event[idx], fit_entry=entry[idx])
    return CoxResult(coef=beta, var_model=imat_inv, var_robust=var_robust, var_design=var_design, names=names,
                     loglik=ll, loglik_null=ll0, n=int(len(idx)), n_events=n_ev,
                     weighted_events=float(np.sum(w[event[idx] == 1])), iterations=it, converged=converged,
                     ties=ties, score_resid=resid, info=inf, zph_data=zph,
                     design_df=None if design is None else design.df, warnings=warn)


def cox_loglik(time, event, X, beta, *, entry=None, weights=None, strata=None, ties="efron") -> float:
    """Partial log-likelihood at a given beta (used for profile/likelihood checks)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(time)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    entry = np.zeros(n) if entry is None else np.asarray(entry, float)
    st = np.zeros(n, dtype=np.int64) if strata is None else _codes(strata)[0]
    ll = 0.0
    for s in np.unique(st):
        m = (st == s) & (w > 0)
        so = _CoxStratum(np.asarray(time, float)[m], entry[m], np.asarray(event)[m], w[m], X[m],
                         ties.lower() == "efron")
        ll += so.pieces(np.asarray(beta, float), need_info=False)[0]
    return ll


def cox_zph(fit: CoxResult, transform: str = "km", use_weights: bool = False) -> pd.DataFrame:
    """Proportional-hazards diagnostics from scaled Schoenfeld residuals (Grambsch & Therneau 1994).

    For each covariate: rho = correlation of the scaled Schoenfeld residual with g(t); chi2 = approximate
    score-test statistic of beta_j(t) = beta_j + gamma_j g(t) (1 df) as in R survival < 3.0; plus a global test
    (p df). transform: 'km' (1 - KM of the event times), 'rank', 'identity' or 'log'.
    use_weights: weight the event-level sums by the case weights (heuristic for weighted fits; default off).
    For weighted fits the model variance is rescaled to weights normalised to mean 1 (diagnostic only).
    Also returns `scaled` residuals and g(t) in the attrs of the returned frame.
    """
    t = fit.zph_data["time"]
    s = fit.zph_data["schoen"]
    d = len(t)
    if d < 3:
        raise ValueError("too few events for PH diagnostics")
    order = np.argsort(t, kind="mergesort")
    t, s = t[order], s[order]
    we = fit.zph_data["w"][order] if use_weights else np.ones(d)
    if transform == "identity":
        g = t.copy()
    elif transform == "log":
        g = np.log(t)
    elif transform == "rank":
        g = stats.rankdata(t)
    elif transform == "km":
        # 1 - left-continuous Kaplan-Meier of the fitted data, evaluated at the event times (R cox.zph 'km')
        kmf = km(fit.zph_data["fit_time"], fit.zph_data["fit_event"], entry=fit.zph_data["fit_entry"])
        k = np.searchsorted(kmf["time"].values, t, side="left")
        s_left = np.concatenate([[1.0], kmf["surv"].values])[k]
        g = 1 - s_left
    else:
        raise ValueError("transform must be km, rank, identity or log")
    # weights normalised to mean 1 (survey weights would otherwise rescale the information matrix)
    wpos = fit.zph_data["w_all"][fit.zph_data["w_all"] > 0]
    V = fit.var_model * (float(np.mean(wpos)) if len(wpos) else 1.0)
    scaled = s @ V * d + fit.coef
    gm = np.sum(we * g) / np.sum(we)
    xx = g - gm
    test = (we * xx) @ s
    sxx = np.sum(we * xx ** 2)
    z = test ** 2 / (np.diag(V) * d * sxx)
    glob = float(test @ V @ test * d / sxx)
    rho = np.array([np.corrcoef(g, scaled[:, j])[0, 1] if np.std(scaled[:, j]) > 0 else np.nan
                    for j in range(s.shape[1])])
    out = pd.DataFrame({"term": fit.names + ["GLOBAL"], "rho": list(rho) + [np.nan],
                        "chi2": list(z) + [glob], "df": [1] * len(z) + [len(z)],
                        "p": list(stats.chi2.sf(z, 1)) + [float(stats.chi2.sf(glob, len(z)))]})
    out.attrs["scaled"] = scaled
    out.attrs["g"] = g
    out.attrs["transform"] = transform
    return out


# ================================================================================================================
# Kaplan-Meier, Aalen-Johansen, cumulative incidence
# ================================================================================================================
def _risk_counts(time, entry, w, ut):
    """Weighted number at risk (entry < t <= time) at each time in ut."""
    ot = np.argsort(time, kind="mergesort")
    oe = np.argsort(entry, kind="mergesort")
    A = _suffix(w, ot)[np.searchsorted(time[ot], ut, side="left")]
    B = _suffix(w, oe)[np.searchsorted(entry[oe], ut, side="left")]
    return A - B


def km(time, event, *, entry=None, weights=None, design: SurveyDesign | None = None) -> pd.DataFrame:
    """Kaplan-Meier survival at every distinct event time.

    SE: Greenwood (weights treated as frequency weights) without a design; Taylor linearisation with a design.
    CI: log(-log) transform.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    n = len(time)
    w = design.weights if design is not None else (np.ones(n) if weights is None else np.asarray(weights, float))
    entry = np.zeros(n) if entry is None else np.asarray(entry, float)
    pos = w > 0
    ut = np.unique(time[pos & (event == 1)])
    Y = _risk_counts(time[pos], entry[pos], w[pos], ut)
    dN = np.bincount(np.searchsorted(ut, time[pos & (event == 1)]), weights=w[pos & (event == 1)],
                     minlength=len(ut))
    haz = np.where(Y > 0, dN / np.maximum(Y, 1e-300), 0)
    surv = np.cumprod(1 - haz)
    if design is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            gw = np.cumsum(np.where(Y - dN > 0, dN / (Y * (Y - dN)), 0))
        se = surv * np.sqrt(gw)
    else:
        infl = cuminc(time, event, ut, entry=entry, design=design, return_influence=True)[1]
        se = np.sqrt(np.diag(design_vcov(infl, design)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.log(-np.log(surv))
        sell = se / (surv * np.abs(np.log(surv)))
        lo = np.exp(-np.exp(ll + Z975 * sell))
        hi = np.exp(-np.exp(ll - Z975 * sell))
    return pd.DataFrame({"time": ut, "n_risk": Y, "n_event": dN, "surv": surv, "se": se,
                         "lower": lo, "upper": hi})


def cuminc(time, event, times, *, cause=None, entry=None, weights=None, design: SurveyDesign | None = None,
           return_influence: bool = False):
    """Cumulative incidence at the requested times with influence-function (linearisation) SE.

    event : 0 = censored; with cause=None any non-zero value is the event and the estimate is 1 - KM.
            With cause = c, event codes != 0 are competing causes and the Aalen-Johansen estimate of cause c
            is returned.
    Without a design the SE uses each record as its own PSU (robust); with a design it is design-based.
    Returns DataFrame(time, cuminc, se, lower, upper) [, influence matrix (n, len(times)) weighted].
    When cause is None and return_influence is True, the influence matrix refers to S(t) = 1 - cuminc.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    n = len(time)
    times = np.atleast_1d(np.asarray(times, float))
    w = design.weights if design is not None else (np.ones(n) if weights is None else np.asarray(weights, float))
    entry = np.zeros(n) if entry is None else np.asarray(entry, float)
    pos = w > 0
    anyev = (event != 0) & pos
    ut = np.unique(time[anyev])
    K = len(ut)
    if K == 0:                                   # no events in the (domain) sample: incidence 0, SE 0
        res = pd.DataFrame({"time": times, "cuminc": 0.0, "se": 0.0, "lower": 0.0, "upper": 0.0})
        return (res, np.zeros((n, len(times)))) if return_influence else res
    Y = _risk_counts(time[pos], entry[pos], w[pos], ut)
    kk = np.searchsorted(ut, time[anyev])
    dN = np.bincount(kk, weights=w[anyev], minlength=K)
    lam = np.where(Y > 0, dN / np.maximum(Y, 1e-300), 0.0)
    surv = np.cumprod(1 - lam)
    surv_left = np.concatenate([[1.0], surv[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where((Y > 0) & (1 - lam > 1e-12), 1.0 / (Y * (1 - lam)), 0.0)
    a = lam * c
    Ac = np.concatenate([[0.0], np.cumsum(a)])
    kmin = np.searchsorted(ut, entry, side="right")
    k_ev = np.where(event != 0, np.searchsorted(ut, time), -1)
    infl = np.zeros((n, len(times)))
    est = np.zeros(len(times))
    if cause is None:
        for j, tau in enumerate(times):
            kt = np.searchsorted(ut, tau, side="right")          # number of event times <= tau
            S_tau = surv[kt - 1] if kt > 0 else 1.0
            est[j] = 1 - S_tau
            kmax = np.minimum(np.searchsorted(ut, time, side="right"), kt)
            risk = Ac[np.maximum(kmax, kmin)] - Ac[kmin]
            evp = np.where((event != 0) & (time <= tau) & (k_ev >= 0), c[np.clip(k_ev, 0, max(K - 1, 0))], 0.0)
            infl[:, j] = -S_tau * (evp - risk)                    # influence on S(tau)
        infl = np.where(pos[:, None], infl * w[:, None], 0.0)
        out_infl = infl
        se_infl = -infl
    else:
        ev1 = (event == cause) & pos
        dN1 = np.bincount(np.searchsorted(ut, time[ev1]), weights=w[ev1], minlength=K)
        lam1 = np.where(Y > 0, dN1 / np.maximum(Y, 1e-300), 0.0)
        F1 = np.cumsum(surv_left * lam1)
        b = np.where(Y > 0, surv_left * lam1 / np.maximum(Y, 1e-300), 0.0)
        Bc = np.concatenate([[0.0], np.cumsum(b)])
        aF = a * F1
        AFc = np.concatenate([[0.0], np.cumsum(aF)])
        kc = np.clip(k_ev, 0, max(K - 1, 0))
        for j, tau in enumerate(times):
            kt = np.searchsorted(ut, tau, side="right")
            F_tau = F1[kt - 1] if kt > 0 else 0.0
            est[j] = F_tau
            kmax = np.maximum(np.minimum(np.searchsorted(ut, time, side="right"), kt), kmin)
            in_tau = (k_ev >= 0) & (time <= tau)
            # first sum: cause-1 counting process martingale
            t1 = np.where(in_tau & (event == cause), surv_left[kc] / np.maximum(Y[kc], 1e-300), 0.0) \
                - (Bc[kmax] - Bc[kmin])
            # second sum: propagation through S(t-)
            t2 = -np.where(in_tau, c[kc] * (F_tau - F1[kc]), 0.0) \
                + F_tau * (Ac[kmax] - Ac[kmin]) - (AFc[kmax] - AFc[kmin])
            infl[:, j] = t1 + t2
        infl = np.where(pos[:, None], infl * w[:, None], 0.0)
        out_infl = infl
        se_infl = infl
    if design is not None:
        V = design_vcov(se_infl, design)
    else:
        V = robust_vcov(se_infl)
    se = np.sqrt(np.clip(np.diag(np.atleast_2d(V)), 0, None))
    res = pd.DataFrame({"time": times, "cuminc": est, "se": se,
                        "lower": np.clip(est - Z975 * se, 0, 1), "upper": np.clip(est + Z975 * se, 0, 1)})
    return (res, out_infl) if return_influence else res


def svy_cuminc_by_group(time, event, group, times, design: SurveyDesign, *, cause=None, entry=None,
                        domain=None) -> pd.DataFrame:
    """Survey-weighted cumulative incidence by group: each group is a domain (weights zeroed outside it)."""
    g = pd.Series(np.asarray(group))
    dom = np.ones(len(g), bool) if domain is None else np.asarray(domain, bool)
    rows = []
    for lev in sorted(g.dropna().unique(), key=lambda v: str(v)):
        d = design.domain(dom & (g.values == lev))
        if not np.any(d.weights > 0):
            continue
        r = cuminc(time, event, times, cause=cause, entry=entry, design=d)
        r.insert(0, "group", lev)
        r.insert(1, "n_unweighted", int(np.sum(d.weights > 0)))
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


# ================================================================================================================
# Survey means and regression
# ================================================================================================================
def svy_mean(y, design: SurveyDesign, domain=None, proportion: bool = False) -> dict:
    """Survey-weighted mean (or proportion) with linearisation SE; domain via zeroed weights.

    Records with missing y are treated as outside the domain (complete-case domain).
    Proportion CI uses the logit transform; mean CI is Wald (normal).
    """
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if domain is not None:
        dm = np.asarray(domain)
        m &= np.where(pd.isna(dm), False, dm).astype(bool)
    d = design.domain(m)
    w = d.weights
    W = w.sum()
    yy = np.where(m, y, 0.0)
    est = float(np.sum(w * yy) / W)
    infl = w * (yy - est) / W
    se = float(np.sqrt(design_vcov(infl, d)))
    out = {"estimate": est, "se": se, "n": int(np.sum(w > 0)), "sum_w": float(W), "df": d.df}
    if proportion and 0 < est < 1:
        lg = np.log(est / (1 - est))
        sl = se / (est * (1 - est))
        out["lower"], out["upper"] = [float(1 / (1 + np.exp(-(lg + s * Z975 * sl)))) for s in (-1, 1)]
    else:
        out["lower"], out["upper"] = est - Z975 * se, est + Z975 * se
    return out


@dataclass
class RegResult:
    coef: np.ndarray
    vcov: np.ndarray
    vcov_type: str
    names: list
    n: int
    df_resid: float
    family: str
    converged: bool
    sigma2: float | None = None
    loglik: float | None = None
    fitted: np.ndarray | None = None
    design_df: int | None = None
    n_clusters: int | None = None

    def table(self, alpha=0.05, exp=None, use_t=False, scale=None) -> pd.DataFrame:
        exp = (self.family == "binomial") if exp is None else exp
        df = None
        if use_t:
            df = self.design_df if self.design_df is not None else (
                self.n_clusters - 1 if self.n_clusters else self.df_resid)
        out = _ci_table(self.coef, np.sqrt(np.diag(self.vcov)), self.names, alpha=alpha, exp=exp, df=df,
                        scale=scale)
        out.insert(1, "vcov", self.vcov_type)
        return out


def glm_fit(y, X, *, family: str = "gaussian", weights=None, vcov: str = "HC1", cluster=None,
            design: SurveyDesign | None = None, names=None, max_iter: int = 100, tol: float = 1e-10) -> RegResult:
    """Linear (family='gaussian') or logistic (family='binomial') regression, weighted.

    X must include an intercept column if wanted. vcov: 'model', 'HC0', 'HC1', 'cluster' (CR1 with
    G/(G-1)(n-1)/(n-p)) or 'design' (requires design; Taylor linearisation of the estimating equations
    sum_i w_i x_i (y_i - mu_i) = 0, domain = records with weight > 0).
    Records with weight 0 are ignored in the fit. Rows with NaN must be removed or given weight 0.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n_all, p = X.shape
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]
    if design is not None:
        w_all = design.weights.astype(float)
        vcov = "design"
    else:
        w_all = np.ones(n_all) if weights is None else np.asarray(weights, float)
    pos = w_all > 0
    if np.any(pos & (~np.isfinite(y) | ~np.all(np.isfinite(X), axis=1))):
        raise ValueError("records with positive weight have missing y or X")
    yi, Xi, wi = y[pos], X[pos], w_all[pos]
    n = int(pos.sum())
    converged = True
    if family == "gaussian":
        A = Xi.T @ (wi[:, None] * Xi)
        beta = np.linalg.solve(A, Xi.T @ (wi * yi))
        mu = Xi @ beta
        vfun = np.ones(n)
    elif family == "binomial":
        if not np.all(np.isin(yi, [0.0, 1.0])):
            raise ValueError("binomial y must be 0/1")
        beta = np.zeros(p)
        dev_old = np.inf
        converged = False
        for _ in range(max_iter):
            eta = Xi @ beta
            mu = 1 / (1 + np.exp(-eta))
            vfun = mu * (1 - mu)
            A = Xi.T @ ((wi * vfun)[:, None] * Xi)
            step = np.linalg.solve(A + 1e-12 * np.eye(p), Xi.T @ (wi * (yi - mu)))
            h = 0
            while True:
                nb = beta + step
                mu_n = 1 / (1 + np.exp(-(Xi @ nb)))
                mu_n = np.clip(mu_n, 1e-15, 1 - 1e-15)
                dev = -2 * np.sum(wi * (yi * np.log(mu_n) + (1 - yi) * np.log(1 - mu_n)))
                if dev <= dev_old + 1e-9 * abs(dev_old if np.isfinite(dev_old) else 1) or h > 30:
                    break
                step /= 2
                h += 1
            beta = nb
            if abs(dev_old - dev) < tol * (abs(dev) + 0.1):
                converged = True
                break
            dev_old = dev
        mu = 1 / (1 + np.exp(-(Xi @ beta)))
        vfun = mu * (1 - mu)
        A = Xi.T @ ((wi * vfun)[:, None] * Xi)
    else:
        raise ValueError("family must be gaussian or binomial")
    Ainv = np.linalg.inv(A)
    e = yi - mu
    sigma2 = None
    ncl = None
    if family == "gaussian":
        sigma2 = float(np.sum(wi * e ** 2) / (np.sum(wi) - p)) if np.sum(wi) > p else np.nan
    u = np.zeros((n_all, p))
    u[pos] = (wi * e)[:, None] * Xi
    if vcov == "model":
        V = Ainv * (sigma2 if family == "gaussian" else 1.0)
    elif vcov in ("HC0", "HC1"):
        V = Ainv @ robust_vcov(u[pos]) @ Ainv
        if vcov == "HC1":
            V *= n / (n - p)
    elif vcov == "cluster":
        if cluster is None:
            raise ValueError("cluster vcov needs cluster")
        cl = np.asarray(cluster)[pos]
        ncl = len(np.unique(cl))
        V = Ainv @ robust_vcov(u[pos], cl) @ Ainv * (ncl / (ncl - 1)) * ((n - 1) / (n - p))
    elif vcov == "design":
        if design is None:
            raise ValueError("design vcov needs design")
        V = Ainv @ design_vcov(u, design) @ Ainv
    else:
        raise ValueError("unknown vcov")
    ll = None
    if family == "binomial":
        mc = np.clip(mu, 1e-15, 1 - 1e-15)
        ll = float(np.sum(wi * (yi * np.log(mc) + (1 - yi) * np.log(1 - mc))))
    fitted = np.full(n_all, np.nan)
    fitted[pos] = mu
    return RegResult(coef=beta, vcov=V, vcov_type=vcov, names=names, n=n, df_resid=n - p, family=family,
                     converged=converged, sigma2=sigma2, loglik=ll, fitted=fitted,
                     design_df=None if design is None else design.df, n_clusters=ncl)
