#!/usr/bin/env python
"""Manually verify LightGBMModel and the sklearn-style Trainer branch.

Run from the repository root with ``python util/diagnostics/verify_lightgbm.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.models.lightgbm import LightGBMModel
from src.training.trainer import Trainer


def make_synthetic_loader(batch_size=4, num_batches=3):
    samples = []
    for idx in range(batch_size * num_batches):
        samples.append({
            "x_numeric": torch.randn(36, 28),
            "state_ids": torch.tensor(idx % 3),
            "comm_ids": torch.tensor(idx % 4),
            "flow_ids": torch.tensor(idx % 2),
            "target_value": torch.tensor(float(idx) / 10.0),
            "target_weight": torch.tensor(float(idx) / -10.0),
        })
    return DataLoader(samples, batch_size=batch_size)


def main():
    print("=" * 60)
    print("LightGBM Model Verification")
    print("=" * 60)

    model = LightGBMModel(
        variant="embeddings",
        num_numeric_features=28,
        num_states=3,
        num_commodities=4,
        num_flows=2,
        n_estimators=5,
        num_leaves=7,
    )
    print("PASS instantiated LightGBMModel")

    assert isinstance(model, torch.nn.Module)
    assert getattr(model, "requires_training", True) is False
    model.to("cpu")
    print("PASS nn.Module / requires_training contract")

    train_loader = make_synthetic_loader()
    val_loader = make_synthetic_loader()

    cfg = {
        "training": {"batch_size": 4, "epochs": 1, "lr": 1e-3},
        "checkpointing": {"save_dir": "outputs/test_lightgbm"},
        "logging": {
            "log_dir": "outputs/test_lightgbm/logs",
            "experiment_name": "test_lightgbm",
        },
    }
    trainer = Trainer(cfg)
    history = trainer.fit(model, train_loader, val_loader)
    print(f"PASS Trainer.fit sklearn branch, val_loss={history['val_loss'][-1]:.6f}")

    preds, targets = model.predict_loader(val_loader)
    assert preds.shape == targets.shape == (12, 2)
    print(f"PASS predict_loader preds={preds.shape}, targets={targets.shape}")

    print("=" * 60)
    print("All LightGBM verification checks PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
