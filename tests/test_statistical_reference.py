"""Numerical reference cases at the actual eight-state inference sample size."""
import math

import numpy as np
import pytest
from scipy import stats

from ext.eval.resampling import paired_t, friedman_nemenyi


def test_eight_state_t_test_does_not_use_normal_pvalue():
    differences = np.arange(8, dtype=float)
    differences = (differences - differences.mean()) / differences.std(ddof=1) + 2 / np.sqrt(8)
    statistic, pvalue, _ = paired_t(differences)
    reference = stats.ttest_1samp(differences, 0)
    assert statistic == pytest.approx(2)
    assert pvalue == pytest.approx(0.08561932856297597)
    assert pvalue == pytest.approx(reference.pvalue)
    assert pvalue > 0.05


@pytest.mark.parametrize("difference,statistic,pvalue", [(0, 0, 1), (2, math.inf, math.nan), (-2, -math.inf, math.nan)])
def test_constant_paired_differences_have_defined_results(difference, statistic, pvalue):
    t, p, mean = paired_t(np.full(8, difference))
    assert t == statistic and mean == difference
    assert math.isnan(p) if math.isnan(pvalue) else p == pvalue


def test_friedman_ties_match_reference_with_explicit_small_sample_warning():
    data = np.tile([0., 0., 1.], (8, 1))
    with pytest.warns(RuntimeWarning, match="exploratory"):
        result = friedman_nemenyi(data)
    reference = stats.friedmanchisquare(*data.T)
    assert result["chi2"] == pytest.approx(reference.statistic)
    assert result["chi2"] == pytest.approx(16)
    assert result["p"] == pytest.approx(reference.pvalue)
    assert not result["asymptotic_reliability_condition_met"]


def test_all_ties_do_not_produce_nan_evidence():
    with pytest.warns(RuntimeWarning):
        result = friedman_nemenyi(np.ones((8, 48)))
    assert result["chi2"] == 0 and result["p"] == 1
    np.testing.assert_array_equal(result["mean_ranks"], np.full(48, 24.5))


def test_alpha_controls_actual_48_arm_critical_difference():
    data = np.random.default_rng(26).normal(size=(40, 48))
    strict = friedman_nemenyi(data, alpha=0.01)
    loose = friedman_nemenyi(data, alpha=0.10)
    assert strict["cd"] > loose["cd"]
    expected = stats.studentized_range.ppf(0.99, 48, np.inf) / np.sqrt(2) * np.sqrt(48 * 49 / (6 * 40))
    assert strict["cd"] == pytest.approx(expected)


def test_missing_blocks_are_rejected_instead_of_dropped():
    with pytest.raises(ValueError, match="finite"):
        paired_t([1, np.nan, 2])
    with pytest.raises(ValueError, match="finite"):
        friedman_nemenyi([[1, 2, 3], [1, np.nan, 3]])


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan")])
def test_invalid_significance_level_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        friedman_nemenyi(np.ones((8, 3)), alpha=alpha)
