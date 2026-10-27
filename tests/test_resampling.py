"""The significance tests decide which arms count as tied with the best, so
they get pinned against cases with known answers."""

import math

import numpy as np
import pytest
import pandas as pd

from ext.eval.resampling import (
    _student_t_sf,
    cluster_bootstrap,
    cluster_robust_paired_t,
    friedman_nemenyi,
    holm,
    paired_t,
    wild_cluster_bootstrap,
)
from util.make_significance import per_state


def test_paired_t_finds_a_real_difference():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 400)          # clearly positive
    t, p, mean = paired_t(d)
    assert t > 5 and p < 1e-6
    assert mean == pytest.approx(0.5, abs=0.15)


def test_paired_t_does_not_invent_one():
    rng = np.random.default_rng(1)
    t, p, _ = paired_t(rng.normal(0.0, 1.0, 400))
    assert p > 0.05


def test_bootstrap_and_t_agree_on_well_behaved_data():
    """They answer the same question by different routes, so on clean data they
    should land close. A divergence in practice means skew or an outlier is
    driving the mean, which is worth knowing.

    The effect is kept modest on purpose: a large one drives the bootstrap p to
    its 1/n_boot floor, where the two are no longer comparable because one has
    run out of resolution rather than disagreed."""
    rng = np.random.default_rng(2)
    d = rng.normal(0.12, 1.0, 500)
    _, p_t, _ = paired_t(d)
    p_b, mean_b, lo, hi = cluster_bootstrap(d, n_boot=4000, seed=3)
    assert math.log10(max(p_t, 1e-12)) == pytest.approx(
        math.log10(max(p_b, 1e-12)), abs=1.0
    )
    assert lo < mean_b < hi


def test_bootstrap_interval_excludes_zero_when_the_effect_is_real():
    rng = np.random.default_rng(4)
    p, mean, lo, hi = cluster_bootstrap(rng.normal(1.0, 1.0, 300), seed=5)
    assert lo > 0 and p < 0.01


def test_bootstrap_p_is_floored_at_its_own_resolution():
    """With 1000 resamples the procedure cannot resolve p < 1e-3; reporting 0
    would claim more than it knows."""
    rng = np.random.default_rng(6)
    p, *_ = cluster_bootstrap(rng.normal(10.0, 1.0, 200), n_boot=1000, seed=7)
    assert p == pytest.approx(1e-3)


def test_clustering_undoes_the_inflation_from_pooling_points():
    """The whole reason the unit of analysis matters: five correlated months
    per lane look like five independent observations if pooled, which shrinks
    the standard error and inflates significance."""
    rng = np.random.default_rng(8)
    n_combo, n_month = 200, 5
    lane_effect = rng.normal(0.05, 1.0, n_combo)          # the real signal
    points = np.repeat(lane_effect, n_month) + rng.normal(0, 0.05, n_combo * n_month)

    _, p_pooled, _ = paired_t(points)                      # wrong unit
    _, p_clustered, _ = paired_t(lane_effect)              # right unit
    assert p_pooled < p_clustered, "pooling must look more significant"
    assert p_clustered / max(p_pooled, 1e-300) > 3


def test_holm_is_monotone_and_never_shrinks_a_pvalue():
    raw = [0.001, 0.01, 0.04, 0.2, 0.5]
    adj = holm(raw)
    assert all(a >= r for a, r in zip(adj, raw))
    assert list(adj) == sorted(adj)
    assert adj[0] == pytest.approx(0.005)                  # 5 * 0.001


def test_holm_kills_the_borderline_case_in_a_large_family():
    """~40 arms against the best is ~40 tests; about two clear 0.05 by chance."""
    raw = [0.04] + [0.5] * 39
    assert holm(raw)[0] > 0.05


def test_friedman_separates_a_clearly_better_arm():
    rng = np.random.default_rng(9)
    n = 300
    good = rng.normal(0.5, 0.1, n)
    mid = rng.normal(0.7, 0.1, n)
    bad = rng.normal(0.9, 0.1, n)
    r = friedman_nemenyi(np.column_stack([good, mid, bad]))
    assert r["p"] < 1e-10
    assert r["mean_ranks"][0] < r["mean_ranks"][1] < r["mean_ranks"][2]
    assert r["mean_ranks"][2] - r["mean_ranks"][0] > r["cd"]


def test_friedman_calls_identical_arms_a_tie():
    rng = np.random.default_rng(10)
    base = rng.normal(0.5, 0.1, 300)
    r = friedman_nemenyi(np.column_stack([base + rng.normal(0, 1e-9, 300)
                                          for _ in range(3)]))
    assert r["p"] > 0.05
    assert max(r["mean_ranks"]) - min(r["mean_ranks"]) < r["cd"]


def test_friedman_is_immune_to_a_few_catastrophic_lanes():
    """Ranking within each combo is why: one lane blowing up cannot move the
    leaderboard the way it moves a mean."""
    rng = np.random.default_rng(11)
    n = 300
    a = rng.normal(0.5, 0.05, n)
    b = a + 0.1                                   # b is consistently worse
    b[:3] = 0.0                                   # ...except three lanes, hugely better
    a[:3] = 1e4
    assert a.mean() > b.mean(), "the mean is captured by the outliers"
    r = friedman_nemenyi(np.column_stack([a, b]))
    assert r["mean_ranks"][0] < r["mean_ranks"][1], "ranks still favour a"


def test_critical_difference_shrinks_with_more_blocks():
    rng = np.random.default_rng(12)
    small = friedman_nemenyi(rng.normal(0, 1, (50, 5)))
    large = friedman_nemenyi(rng.normal(0, 1, (5000, 5)))
    assert large["cd"] < small["cd"]


def test_state_blocking_collapses_months_and_lanes():
    index = pd.MultiIndex.from_tuples(
        [
            ("A", "corn", "Export", "2025-01"),
            ("A", "corn", "Export", "2025-02"),
            ("A", "soy", "Import", "2025-01"),
            ("A", "soy", "Import", "2025-02"),
            ("B", "corn", "Export", "2025-01"),
            ("B", "corn", "Export", "2025-02"),
        ],
        names=["State", "Commodity", "Import/Export", "Time"],
    )
    loss = pd.Series([1, 3, 5, 7, 10, 14], index=index)
    blocks = per_state(loss)
    assert blocks.to_dict() == {"A": 4.0, "B": 12.0}


# ---------------------------------------------------------------------------
# 2026-09-09: the callers now hand these functions LANE differentials with the
# state as the cluster. The tests below pin the Student-t tail (the old normal
# tail was anti-conservative at n=8) and the clustered machinery.
# ---------------------------------------------------------------------------

def _clustered_panel(seed, cluster_shift_sd, n_clusters=8, per_cluster=100, effect=0.0):
    """Lane differentials with a shared random effect per cluster (state)."""
    rng = np.random.default_rng(seed)
    cluster_effect = rng.normal(effect, cluster_shift_sd, n_clusters)
    d = np.repeat(cluster_effect, per_cluster) + rng.normal(0.0, 1.0, n_clusters * per_cluster)
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    return d, clusters


def test_student_t_tail_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for t, df in [(0.3, 20), (1.0, 3), (2.5, 7), (3.0, 7), (5.0, 399), (12.0, 2)]:
        assert _student_t_sf(t, df) == pytest.approx(2 * stats.t.sf(t, df), rel=1e-7)


def test_paired_t_uses_the_student_tail_at_small_n():
    """Eight values is the size the state-blocked callers used to pass; the
    normal tail gave p=0.0027 where Student t(7) gives 0.020 at t=3."""
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(20)
    d = rng.normal(0.9, 1.0, 8)
    t, p, _ = paired_t(d)
    ref = stats.ttest_1samp(d, 0.0)
    assert t == pytest.approx(ref.statistic, rel=1e-9)
    assert p == pytest.approx(ref.pvalue, rel=1e-7)
    normal_tail = math.erfc(abs(t) / math.sqrt(2))
    assert p > 2 * normal_tail, "Student tail must be heavier than the normal one"


def test_cluster_robust_t_reduces_to_plain_t_with_singleton_clusters():
    rng = np.random.default_rng(21)
    d = rng.normal(0.3, 1.0, 60)
    t_plain, p_plain, m_plain = paired_t(d)
    t_cr, p_cr, m_cr, g = cluster_robust_paired_t(d, np.arange(60))
    assert g == 60
    assert t_cr == pytest.approx(t_plain, rel=1e-9)
    assert p_cr == pytest.approx(p_plain, rel=1e-9)
    assert m_cr == pytest.approx(m_plain)


def test_ignoring_within_state_dependence_inflates_significance():
    """A shared state-level shock makes 100 lanes look like 100 independent
    votes to the plain test; the clustered test sees 8."""
    d, clusters = _clustered_panel(seed=22, cluster_shift_sd=0.5, effect=0.0)
    _, p_plain, _ = paired_t(d)
    _, p_cr, _, g = cluster_robust_paired_t(d, clusters)
    p_wild, *_ = wild_cluster_bootstrap(d, clusters)
    assert g == 8
    assert p_cr > p_plain
    assert p_wild > p_plain


def test_wild_cluster_bootstrap_is_exact_and_detects_a_real_shift():
    d, clusters = _clustered_panel(seed=23, cluster_shift_sd=0.1, effect=1.0)
    p, t_obs, mean, g, reps = wild_cluster_bootstrap(d, clusters)
    assert reps == 2 ** 8, "eight clusters are enumerated, not sampled"
    assert g == 8 and mean > 0.8 and t_obs > 5
    # the identity and all-negative sign patterns reproduce |t_obs| exactly,
    # so the smallest attainable p is 2 / 256
    assert p == pytest.approx(2 / 256)


def test_wild_cluster_bootstrap_does_not_invent_an_effect():
    d, clusters = _clustered_panel(seed=24, cluster_shift_sd=0.5, effect=0.0)
    p, *_ = wild_cluster_bootstrap(d, clusters)
    assert p > 0.05


def test_wild_cluster_bootstrap_samples_when_there_are_many_clusters():
    d, clusters = _clustered_panel(seed=25, cluster_shift_sd=0.1, n_clusters=40,
                                   per_cluster=20, effect=0.5)
    p, _, _, g, reps = wild_cluster_bootstrap(d, clusters, n_boot=999, seed=1)
    assert g == 40 and reps == 999
    assert p < 0.01


def test_cluster_bootstrap_with_clusters_keeps_whole_states_together():
    d, clusters = _clustered_panel(seed=26, cluster_shift_sd=0.5, effect=1.0)
    p, mean, lo, hi = cluster_bootstrap(d, n_boot=2000, seed=2, clusters=clusters)
    assert lo < mean < hi
    assert lo > 0 and p < 0.05
    # resampling whole clusters must give a WIDER interval than resampling lanes
    _, _, lo_lane, hi_lane = cluster_bootstrap(d, n_boot=2000, seed=2)
    assert (hi - lo) > (hi_lane - lo_lane)


def test_clustered_tests_refuse_a_single_cluster():
    d = np.random.default_rng(27).normal(1.0, 1.0, 50)
    assert cluster_robust_paired_t(d, np.zeros(50))[1] == 1.0
    assert wild_cluster_bootstrap(d, np.zeros(50))[0] == 1.0


# --- tie correction, alpha, and fail-closed inputs (ported 2026-09-09) ---

def test_friedman_tie_correction_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    data = np.tile([0.0, 0.0, 1.0], (8, 1))
    with pytest.warns(RuntimeWarning, match="exploratory"):
        result = friedman_nemenyi(data)
    reference = stats.friedmanchisquare(*data.T)
    assert result["chi2"] == pytest.approx(reference.statistic)
    assert result["chi2"] == pytest.approx(16.0)   # uncorrected would be 12
    assert result["p"] == pytest.approx(reference.pvalue)
    assert result["asymptotic_reliability_condition_met"] is False


def test_friedman_all_ties_give_no_evidence():
    with pytest.warns(RuntimeWarning):
        result = friedman_nemenyi(np.ones((8, 48)))
    assert result["chi2"] == 0 and result["p"] == 1
    np.testing.assert_array_equal(result["mean_ranks"], np.full(48, 24.5))


def test_friedman_alpha_moves_the_critical_difference():
    stats = pytest.importorskip("scipy.stats")
    data = np.random.default_rng(26).normal(size=(40, 48))
    strict, loose = friedman_nemenyi(data, alpha=0.01), friedman_nemenyi(data, alpha=0.10)
    assert strict["cd"] > loose["cd"]
    expected = stats.studentized_range.ppf(0.99, 48, np.inf) / np.sqrt(2) * np.sqrt(48 * 49 / (6 * 40))
    assert strict["cd"] == pytest.approx(expected)


def test_friedman_refuses_missing_blocks_and_bad_alpha():
    with pytest.raises(ValueError, match="finite"):
        friedman_nemenyi([[1, 2, 3], [1, np.nan, 3]])
    for alpha in (0, 1, -0.1, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            friedman_nemenyi(np.ones((8, 3)), alpha=alpha)


def test_constant_nonzero_differences_are_degenerate_not_certain():
    """A perfectly consistent non-zero difference has no variance estimate:
    the statistic is unbounded and p is undefined -- never 0 (which would
    survive every correction) and never 1 (which would hide the effect)."""
    t, p, mean = paired_t(np.full(8, 2.0))
    assert math.isinf(t) and t > 0 and math.isnan(p) and mean == 2.0
    assert paired_t(np.zeros(8)) == (0.0, 1.0, 0.0)
    clusters = np.repeat(np.arange(4), 5)
    t, p, mean, g = cluster_robust_paired_t(np.full(20, -1.5), clusters)
    assert math.isinf(t) and t < 0 and math.isnan(p) and g == 4
    p_w, t_w, *_ = wild_cluster_bootstrap(np.full(20, -1.5), clusters)
    assert math.isnan(p_w) and math.isinf(t_w)
    assert cluster_robust_paired_t(np.zeros(20), clusters)[1] == 1.0
