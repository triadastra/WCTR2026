"""Exploratory tests for paired forecasting losses.

Primary consumers retain the lane-weighted mean over valid observations and
cluster uncertainty by state. Equal-state means answer a different question
and are only a sensitivity analysis. Student-t uses G-1 degrees of freedom;
wild-cluster sign enumeration removes Monte Carlo error, not the assumptions
of independent states and sign symmetry. Eight clusters and shared national
shocks limit inference. Percentile bootstrap intervals and rank comparisons
are descriptive; non-rejection is not evidence of equivalence.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
from scipy import stats as _sps


import numpy as np


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_it, eps, fpmin = 500, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b), without scipy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_sf(t, df):
    """Two-sided tail P(|T| >= |t|) of Student's t with ``df`` degrees of freedom.

    Uses I_x(df/2, 1/2) with x = df / (df + t^2). At df -> inf this is the
    normal tail; at df = 7 it is several times larger than the normal tail for
    the same t, which is the whole point of using it.
    """
    if not math.isfinite(t):
        return 0.0
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return min(1.0, max(0.0, _betainc(df / 2.0, 0.5, x)))


def paired_t(diff_per_unit):
    """Paired t-test on one difference per unit (lane, sweep cell, ...).

    ``diff_per_unit`` is B's mean loss minus A's, one value per unit, so a
    positive mean means B is worse. Returns (t, p, mean_diff). The p-value is
    the two-sided Student-t tail with n-1 degrees of freedom: a normal tail is
    indistinguishable at n=900 but understates p several-fold at n=8, which is
    the size the state-blocked callers were handing this function.
    """
    d = np.asarray(diff_per_unit, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Differences must be finite and one-dimensional")
    n = len(d)
    if n < 3:
        return 0.0, 1.0, float(d.mean()) if n else 0.0
    mean = float(d.mean())
    if d.var(ddof=1) <= 0:
        # Every unit shows the same difference. Zero difference: no evidence
        # (t=0, p=1). Non-zero: the statistic is unbounded and the sample
        # carries no variance estimate, so the p-value is UNDEFINED -- not 0
        # (which would survive any correction) and not 1 (which hides a
        # perfectly consistent effect). Callers treat NaN as "degenerate".
        if mean == 0.0:
            return 0.0, 1.0, mean
        return math.copysign(math.inf, mean), float("nan"), mean
    t = mean / math.sqrt(d.var(ddof=1) / n)
    return float(t), float(_student_t_sf(t, n - 1)), mean


def _cluster_ids(clusters, n):
    c = np.asarray(clusters)
    if len(c) != n:
        raise ValueError(f"clusters has {len(c)} entries for {n} differences")
    labels, inv = np.unique(c, return_inverse=True)
    return inv, len(labels)


def _cluster_robust_t(d, inv, n_clusters):
    """Cluster-robust t of the mean of ``d`` (units grouped by ``inv``).

    Var(mean) = G/(G-1) * sum_g (sum_{i in g} (d_i - mean))^2 / n^2 -- the
    sandwich estimator of a mean with the usual finite-cluster factor.
    Returns (t, mean); t is 0 when the variance is degenerate.
    """
    n = len(d)
    mean = float(d.mean())
    resid = d - mean
    cluster_sums = np.bincount(inv, weights=resid, minlength=n_clusters)
    var = (n_clusters / (n_clusters - 1.0)) * float(np.sum(cluster_sums ** 2)) / (n * n)
    if var <= 0.0:
        # Zero between-cluster variance of the residual sums: no evidence when
        # the mean is zero, an unbounded statistic (undefined p) otherwise.
        return (0.0 if mean == 0.0 else math.copysign(math.inf, mean)), mean
    return mean / math.sqrt(var), mean


def cluster_robust_paired_t(diff_per_unit, clusters):
    """Paired t on unit-level differences with a cluster-robust variance.

    ``clusters`` gives each unit's cluster label (the STATE for a lane). The
    mean differential is the same as in ``paired_t``; its variance allows any
    dependence among units that share a cluster, and the statistic is referred
    to Student t with G-1 degrees of freedom (Cameron & Miller 2015). Returns
    (t, p, mean_diff, n_clusters).
    """
    d = np.asarray(diff_per_unit, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Differences must be finite and one-dimensional")
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0.0, 0
    inv, n_clusters = _cluster_ids(np.asarray(clusters), n)
    if n < 3 or n_clusters < 2:
        return 0.0, 1.0, float(d.mean()), n_clusters
    t, mean = _cluster_robust_t(d, inv, n_clusters)
    if t == 0.0:
        return 0.0, 1.0, mean, n_clusters
    if math.isinf(t):
        return float(t), float("nan"), mean, n_clusters   # degenerate, see _cluster_robust_t
    return float(t), float(_student_t_sf(t, n_clusters - 1)), mean, n_clusters


def wild_cluster_bootstrap(diff_per_unit, clusters, n_boot=4096, seed=0,
                           enumerate_up_to=12):
    """Wild cluster bootstrap p-value for H0: mean differential = 0.

    Cameron, Gelbach & Miller (2008): with few clusters the asymptotic
    cluster-robust t is unreliable, so its reference distribution is built by
    re-signing the (null-restricted) residuals by whole clusters with
    Rademacher weights and recomputing the cluster-robust t. With G <=
    ``enumerate_up_to`` clusters all 2^G sign patterns are enumerated, which
    removes the Monte Carlo error and makes the result deterministic (the
    identity and all-negative patterns reproduce |t_obs|, so the smallest
    attainable p is 2 / 2^G -- 0.0078 at G=8). "Exhaustive" is not
    "finite-sample exact": validity still rests on independence ACROSS
    clusters and sign-symmetry of the cluster residuals under H0 (Cameron &
    Miller 2015); shared national shocks that cross state lines are outside
    what any state-clustered test can absorb. Otherwise ``n_boot`` patterns
    are sampled with ``seed``. A degenerate input (zero cluster-robust
    variance with a non-zero mean) returns p = NaN.

    Returns (p, t_obs, mean_diff, n_clusters, n_replications).
    """
    d = np.asarray(diff_per_unit, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Differences must be finite and one-dimensional")
    n = len(d)
    if n == 0:
        return 1.0, 0.0, 0.0, 0, 0
    inv, n_clusters = _cluster_ids(np.asarray(clusters), n)
    if n < 3 or n_clusters < 2:
        return 1.0, 0.0, float(d.mean()), n_clusters, 0
    t_obs, mean = _cluster_robust_t(d, inv, n_clusters)
    if t_obs == 0.0:
        return 1.0, 0.0, mean, n_clusters, 0
    if math.isinf(t_obs):
        return float("nan"), float(t_obs), mean, n_clusters, 0

    if n_clusters <= enumerate_up_to:
        weights = np.array(list(itertools.product((1.0, -1.0), repeat=n_clusters)))
    else:
        rng = np.random.default_rng(seed)
        weights = rng.choice((1.0, -1.0), size=(n_boot, n_clusters))
    reps = weights.shape[0]

    # Under H0 the restricted residual is d itself. Vectorised over patterns:
    # d_star has one row per sign pattern; cluster sums via a one-hot matrix.
    onehot = np.zeros((n, n_clusters))
    onehot[np.arange(n), inv] = 1.0
    d_star = d[None, :] * weights[:, inv]                       # (reps, n)
    means = d_star.mean(axis=1)
    resid = d_star - means[:, None]
    cluster_sums = resid @ onehot                               # (reps, G)
    var = (n_clusters / (n_clusters - 1.0)) * np.sum(cluster_sums ** 2, axis=1) / (n * n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(var > 0, means / np.sqrt(var), 0.0)
    p = float(np.mean(np.abs(t_star) >= abs(t_obs) - 1e-12))
    return p, float(t_obs), mean, n_clusters, int(reps)


def cluster_bootstrap(diff_per_unit, n_boot=10000, seed=0, clusters=None):
    """Percentile bootstrap of the mean differential.

    Without ``clusters`` it resamples units (lanes) with replacement; with
    ``clusters`` it resamples whole clusters (states) and averages over every
    unit in the drawn clusters, so within-cluster dependence is preserved.
    Returns (p, mean_diff, ci_lo, ci_hi) with a percentile 95% interval. The
    p-value is the two-sided proportion of resamples on the wrong side of
    zero, floored at 1/n_boot -- a bootstrap cannot resolve a p smaller than
    its own resolution, and reporting 0.0 would claim more than the procedure
    knows. With a handful of clusters the interval is honest about its width
    but the wild cluster bootstrap is the better-calibrated p-value.
    """
    d = np.asarray(diff_per_unit, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Differences must be finite and one-dimensional")
    n = len(d)
    if n < 3:
        return 1.0, float(d.mean()) if n else 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        means = d[idx].mean(axis=1)
    else:
        inv, n_clusters = _cluster_ids(np.asarray(clusters), n)
        if n_clusters < 2:
            return 1.0, float(d.mean()), float("nan"), float("nan")
        sums = np.bincount(inv, weights=d, minlength=n_clusters)
        counts = np.bincount(inv, minlength=n_clusters).astype(float)
        idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    p = max(min(p, 1.0), 1.0 / n_boot)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(p), float(d.mean()), float(lo), float(hi)


def holm(pvalues):
    """Holm family-wise adjustment, requiring valid marginal p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues):
    """BH false-discovery adjustment for exploratory screening.

    FDR control requires independent tests or the relevant positive-dependence
    condition. Shared forecasting shocks do not establish that condition.
    Report alongside Holm; do not choose a correction after seeing its result.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.asarray([])
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 1.0
    # Step UP: largest p first, enforcing monotonicity as we descend.
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        adjusted[i] = min(running, 1.0)
    return adjusted


def friedman_nemenyi(loss_matrix, alpha=0.05):
    """Tie-corrected Friedman test over arms, blocked by lane, plus the Nemenyi
    critical difference at ``alpha``.

    ``loss_matrix`` is (n_lanes, n_arms) of mean loss per lane. Every arm is
    ranked within each lane, which discards the magnitude of squared error and
    keeps only the ordering -- so a handful of catastrophic lanes cannot decide
    the leaderboard, as they can for a mean. The statistic carries the standard
    tie correction (scipy.stats.friedmanchisquare agrees), and the Nemenyi
    threshold uses the studentized range at the requested ``alpha`` rather
    than a fixed 5% table. Non-finite entries and out-of-range alphas raise:
    silently dropping a block changes the population under test.

    Both the chi-square approximation and the studentized-range threshold are
    asymptotic; the returned ``asymptotic_reliability_condition_met`` flag
    (n > 10 blocks and k > 6 arms, scipy's documented condition) says whether
    that is defensible. The critical difference also assumes the blocks are
    exchangeable, which lanes are not across states, so treat the CD as
    descriptive and optimistic; the clustered tests above carry the inference.
    """
    X = np.asarray(loss_matrix, dtype=float)
    if X.ndim != 2 or min(X.shape) < 2 or not np.isfinite(X).all():
        raise ValueError(
            "Friedman comparison requires a finite (n_blocks, n_arms) matrix "
            "with at least two blocks and two arms"
        )
    if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    n, k = X.shape
    ranks = _sps.rankdata(X, axis=1, method="average")
    mean_ranks = ranks.mean(axis=0)
    tie_sum = 0
    for row in X:
        counts = np.unique(row, return_counts=True)[1]
        tie_sum += int(np.sum(counts ** 3 - counts))
    correction = 1.0 - tie_sum / (n * k * (k * k - 1))
    if correction <= 0:
        # Every arm equal in every block: no evidence against the null.
        chi2, p = 0.0, 1.0
    else:
        chi2 = max(0.0, float(
            (12.0 * n / (k * (k + 1))) * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4.0)
            / correction
        ))
        p = float(_sps.chi2.sf(chi2, k - 1))
    q_alpha = float(_sps.studentized_range.ppf(1.0 - alpha, k, np.inf) / math.sqrt(2.0))
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))
    reliable = n > 10 and k > 6
    if not reliable:
        warnings.warn(
            "Friedman/Nemenyi inference is asymptotic; with <= 10 blocks or <= 6 "
            "arms report it as exploratory, not as evidence of equivalence.",
            RuntimeWarning, stacklevel=2,
        )
    return {"n": n, "k": k, "chi2": float(chi2), "p": p,
            "mean_ranks": mean_ranks, "cd": float(cd), "alpha": float(alpha),
            "p_method": "chi_square_asymptotic_tie_corrected",
            "asymptotic_reliability_condition_met": bool(reliable)}
