import csv
import numpy as np
import pytest

from src.utils.exp1_truth import validate_exp1_truth


@pytest.fixture
def case(tmp_path):
    key = ("ALL", "ALL", "ALL", "2025-01-01")
    truth = {"targets": {key: np.array([0.5, 0.2, 150., 240.])},
             "minimum": np.array([100., 200.]), "scale": np.array([100., 200.]),
             "seasonal_mae": np.ones(2), "seasonal_mse": np.ones(2)}
    row = dict(zip(("State", "Commodity", "Import/Export", "Time"), key))
    row.update(true_value=.5, true_weight=.2, true_value_real=150., true_weight_real=240.,
               pred_value=.6, pred_weight=.3, pred_value_real=160., pred_weight_real=260.)
    return tmp_path, truth, row


def check(case, change=None, duplicate=False):
    tmp_path, truth, row = case
    row.update(change or {})
    with (tmp_path / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        if duplicate:
            writer.writerow(row)
    metrics = {name: .1 for name in (
        "mase", "value_mase", "weight_mase", "rmsse", "value_rmsse", "weight_rmsse",
    )}
    return validate_exp1_truth({"predictions_file": "predictions.csv", "metrics": metrics},
                               tmp_path / "metrics.json", truth)


def test_source_truth_accepts_matching_targets_and_transform(case):
    assert check(case)[0]


@pytest.mark.parametrize("change", [
    {"true_value": .7}, {"true_value_real": 170.},
    {"true_weight": float("nan")}, {"true_weight_real": float("inf")},
])
def test_source_truth_rejects_wrong_targets(case, change):
    assert not check(case, change)[0]


def test_source_truth_rejects_wrong_prediction_units(case):
    ok, reason = check(case, {"pred_value_real": 600.})
    assert not ok and "prediction units" in reason


def test_source_truth_rejects_duplicate_identity(case):
    assert not check(case, duplicate=True)[0]


def test_source_truth_rejects_wrong_month(case):
    assert not check(case, {"Time": "2025-02-01"})[0]


def test_source_truth_rejects_wrong_seasonal_metric_scale(case):
    case[1]["seasonal_mae"] *= 2
    ok, reason = check(case)
    assert not ok and "seasonal scaling" in reason
