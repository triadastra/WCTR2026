import numpy as np
import pytest

from src.inference.metrics import (
    PERCENTAGE_METRIC_NAMES,
    SCALE_DEPENDENT_METRIC_NAMES,
    combined_smape,
    compute_evaluation_metrics,
    compute_metrics,
    symmetric_mape,
)


def test_combined_smape_weights_channels_equally():
    targets = np.array([[1.0, 1000.0], [1.0, 1000.0]])
    predictions = np.array([[0.0, 900.0], [0.0, 900.0]])
    expected = np.mean([
        symmetric_mape(targets[:, 0], predictions[:, 0]),
        symmetric_mape(targets[:, 1], predictions[:, 1]),
    ])
    assert combined_smape(predictions, targets) == pytest.approx(expected)
    assert compute_metrics(predictions, targets)["smape"] == pytest.approx(expected)


def test_combined_smape_rejects_wrong_shape():
    with pytest.raises(ValueError, match="n_samples, 2"):
        combined_smape(np.zeros((3, 1)), np.zeros((3, 1)))


def test_evaluation_rejects_different_real_and_normalized_populations():
    # Both pairs are independently valid, but mixing them would publish MSE
    # and percentages over different target sets under one sample count.
    with pytest.raises(ValueError, match="same.*population"):
        compute_evaluation_metrics(
            np.ones((5, 2)), np.ones((5, 2)),
            np.ones((4, 2)), np.ones((4, 2)),
        )


def test_evaluation_keeps_normalized_loss_but_uses_real_percentage_domain():
    targets_norm = np.array([[0.1, 0.2], [0.2, 0.4], [0.4, 0.8]])
    predictions_norm = targets_norm + np.array([[0.1, -0.1]] * 3)
    # A nonzero min-max translation changes both MAPE and sMAPE.  This mirrors
    # the per-series USD/kg inverse transform used by the real evaluator.
    targets_real = targets_norm * np.array([1000.0, 2000.0]) + np.array([10000.0, 5000.0])
    predictions_real = predictions_norm * np.array([1000.0, 2000.0]) + np.array([10000.0, 5000.0])

    metrics = compute_evaluation_metrics(
        predictions_norm, targets_norm, predictions_real, targets_real
    )
    expected_real = compute_metrics(predictions_real, targets_real)
    expected_normalized = compute_metrics(
        predictions_norm, targets_norm, mape_floor=0.0
    )

    assert metrics["norm_mse_combined"] == pytest.approx(
        np.mean((predictions_norm - targets_norm) ** 2)
    )
    for name in SCALE_DEPENDENT_METRIC_NAMES:
        assert metrics[name] == pytest.approx(expected_normalized[name])
    for name in PERCENTAGE_METRIC_NAMES:
        assert metrics[name] == pytest.approx(expected_real[name])
    assert metrics["value_mape"] != pytest.approx(
        expected_normalized["value_mape"]
    )
    assert metrics["smape"] != pytest.approx(expected_normalized["smape"])
