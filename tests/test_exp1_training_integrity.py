from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from scripts.inference import add_mase_rmsse
from src.data.dataloader import DataConfig, TradeDataPipeline
from src.training.trainer import Trainer


class _ZeroModel(nn.Module):
    variant = "test"
    requires_training = True

    def __init__(self, *, nonfinite=False):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.nonfinite = nonfinite

    def forward(self, x_numeric, state_ids, comm_ids, flow_ids):
        output = self.bias * torch.ones((len(x_numeric), 2), device=x_numeric.device)
        return output * float("nan") if self.nonfinite else output

    def get_num_params(self):
        return 1


class _NativeModel(nn.Module):
    requires_training = False
    saved = False

    def __init__(self, train_mse=4.0):
        super().__init__()
        self.train_mse = train_mse

    def fit(self, train, val, verbose=True):
        self.train = train

    def evaluate_loader(self, loader):
        return {"mse": self.train_mse if loader is self.train else 1.0}

    def save(self, path, **kwargs):
        self.saved = True


def test_native_training_loss_is_measured_not_zero(tmp_path):
    model = _NativeModel()
    result = Trainer(_trainer_config(tmp_path)).fit(model, [], [])
    assert result["train_loss"] == [4.0]
    assert result["val_loss"] == [1.0]
    assert model.saved


def test_nonfinite_native_loss_is_rejected_before_checkpoint(tmp_path):
    model = _NativeModel(float("nan"))
    with pytest.raises(ValueError, match="native-model training MSE"):
        Trainer(_trainer_config(tmp_path)).fit(model, [], [])
    assert not model.saved


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps"])
def test_unavailable_neural_device_cannot_mislabel_cpu_run(tmp_path, monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    cfg = _trainer_config(tmp_path)
    cfg["training"]["device"] = device
    trainer = Trainer(cfg)
    published = []
    with pytest.raises(RuntimeError, match="refusing to silently train on CPU"):
        trainer.fit(_ZeroModel(), [], [], on_ready=lambda: published.append(True))
    assert not published
    assert not (tmp_path / "logs").exists()


def test_native_cpu_estimator_does_not_require_torch_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _trainer_config(tmp_path)
    cfg["training"]["device"] = "cuda"
    model = _NativeModel()
    assert Trainer(cfg).fit(model, [], [])["train_loss"] == [4.0]


def test_explicit_cuda_index_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = _trainer_config(tmp_path)
    cfg["training"]["device"] = "cuda:1"
    assert str(Trainer(cfg).device) == "cuda:1"


def _trainer_config(tmp_path):
    return {
        "training": {
            "device": "cpu",
            "epochs": 1,
            "lr": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": "none",
            "patience": 2,
            "min_delta": 0.0,
            "use_amp": False,
        },
        "checkpointing": {
            "save_dir": str(tmp_path / "checkpoints"),
            "save_best_only": True,
            "save_last": True,
            "monitor": "val_loss",
            "mode": "min",
        },
        "logging": {
            "log_dir": str(tmp_path / "logs"),
            "experiment_name": "integrity-test",
        },
    }


def _batch(values):
    values = torch.tensor(values, dtype=torch.float32)
    n = len(values)
    return {
        "x_numeric": torch.zeros(n, 1, 1),
        "target_value": values,
        "target_weight": torch.zeros(n),
        "state_ids": torch.zeros(n, dtype=torch.long),
        "comm_ids": torch.zeros(n, dtype=torch.long),
        "flow_ids": torch.zeros(n, dtype=torch.long),
    }


def _aggregate_source():
    times = pd.date_range("2009-01-01", "2025-05-01", freq="MS")
    rows = []
    for state, multiplier in (("A", 1.0), ("B", 2.0)):
        for index, time in enumerate(times):
            rows.append({
                "State": state,
                "Commodity": "C",
                "Import/Export": "Import",
                "Country": "X",
                "Time": time,
                "Value": multiplier * (index + 1),
                "Weight": multiplier * (index + 2),
            })
    return pd.DataFrame(rows)


def test_exp1_uses_all_132_targets_in_four_equal_batches():
    config = DataConfig(
        start="2009-01-01",
        end="2025-05-01",
        train_end="2023-12-01",
        val_start="2024-01-01",
        val_end="2024-12-01",
        test_start="2025-01-01",
        test_end="2025-05-01",
        aggregate=True,
        input_len=36,
        lag_count=12,
    )
    pipeline = TradeDataPipeline(config)
    pipeline.df_full = _aggregate_source()
    pipeline.create_splits()
    train, _, _ = pipeline.get_dataloaders(
        batch_size=33, num_workers=0, shuffle_train=False
    )

    assert len(train.dataset) == 132
    assert train.drop_last is False
    assert [len(batch["target_value"]) for batch in train] == [33, 33, 33, 33]
    target_times = [sample["target_time"] for sample in train.dataset.samples]
    assert len(set(target_times)) == 132
    assert min(target_times) == pd.Timestamp("2013-01-01")
    assert max(target_times) == pd.Timestamp("2023-12-01")


def test_train_and_validation_losses_are_target_weighted(tmp_path):
    # Batch-mean averaging would return (0.5 + 50) / 2 = 25.25.  The exact
    # six scalar targets contain total squared error 102, hence MSE=17.
    batches = [_batch([1.0, 1.0]), _batch([10.0])]
    trainer = Trainer(_trainer_config(tmp_path))
    trainer.setup(_ZeroModel(), batches, batches)
    try:
        assert trainer.train_epoch() == pytest.approx(17.0)
        assert trainer.validate() == pytest.approx(17.0)
        assert trainer._last_val_stats["val_targets"] == 3
    finally:
        trainer.logger.close()


def test_nonfinite_loss_fails_before_an_optimizer_step(tmp_path):
    trainer = Trainer(_trainer_config(tmp_path))
    model = _ZeroModel(nonfinite=True)
    batches = [_batch([1.0])]
    trainer.setup(model, batches, batches)
    before = model.bias.detach().clone()
    try:
        with pytest.raises(FloatingPointError, match="before optimizer step"):
            trainer.train_epoch()
        assert torch.equal(model.bias.detach(), before)
    finally:
        trainer.logger.close()


def test_mase_uses_pre_burn_in_training_history():
    times = pd.date_range("2009-01-01", periods=36, freq="MS")
    value = np.arange(36, dtype=float) ** 2
    weight = np.arange(36, dtype=float) ** 3
    full = pd.DataFrame({
        "State": "ALL",
        "Commodity": "ALL",
        "Import/Export": "ALL",
        "Time": times,
        "Value": value,
        "Weight": weight,
    })
    pipeline = SimpleNamespace(
        df_metric_grid=full,
        # Deliberately different/truncated; this was the old, incorrect source.
        df_full_grid=full.iloc[12:].copy(),
        config=SimpleNamespace(train_end=times[-1]),
    )
    prediction = np.array([[12.0, 24.0]])
    target = np.zeros((1, 2))
    metrics = {}
    add_mase_rmsse(
        metrics, pipeline, prediction, target, [("ALL", "ALL", "ALL")]
    )

    value_diff = value[12:] - value[:-12]
    weight_diff = weight[12:] - weight[:-12]
    assert metrics["value_mase"] == pytest.approx(12.0 / np.mean(np.abs(value_diff)))
    assert metrics["weight_mase"] == pytest.approx(24.0 / np.mean(np.abs(weight_diff)))
    assert metrics["value_rmsse"] == pytest.approx(
        np.sqrt(12.0**2 / np.mean(value_diff**2))
    )


def test_trainer_rejects_configuration_that_would_select_worst_loss(tmp_path):
    config = _trainer_config(tmp_path)
    config["checkpointing"]["mode"] = "max"
    with pytest.raises(ValueError, match="mode='min'"):
        Trainer(config)
