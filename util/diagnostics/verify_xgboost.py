#!/usr/bin/env python
"""
Manually verify XGBoostModel can be instantiated, registers as nn.Module,
and the sklearn-style Trainer branch works on synthetic data.

Run from the repository root with ``python util/diagnostics/verify_xgboost.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.models.xgboost import XGBoostModel
from src.training.trainer import Trainer


def make_synthetic_loader(batch_size=4, num_batches=3):
    """Create a synthetic DataLoader mimicking the real pipeline."""
    samples = []
    for _ in range(batch_size * num_batches):
        samples.append({
            "x_numeric": torch.randn(36, 28),  # (L, F)
            "state_ids": torch.tensor(0),
            "comm_ids": torch.tensor(0),
            "flow_ids": torch.tensor(0),
            "target_value": torch.tensor(1.0),
            "target_weight": torch.tensor(2.0),
        })
    return DataLoader(samples, batch_size=batch_size)


def main():
    print("=" * 60)
    print("XGBoost Model Verification")
    print("=" * 60)

    # 1. Instantiate
    model = XGBoostModel(
        variant="embeddings",
        num_numeric_features=28,
        num_states=3,
        num_commodities=4,
        num_flows=2,
        n_estimators=10,
        max_depth=3,
    )
    print(f"✅ Instantiated XGBoostModel")

    # 2. Verify nn.Module init worked
    assert isinstance(model, torch.nn.Module), "XGBoostModel must be an nn.Module"
    print(f"✅ Is nn.Module: True")

    # 3. Verify .to() works
    model.to("cpu")
    print(f"✅ .to('cpu') works")

    # 4. Verify .parameters() works
    params = list(model.parameters())
    print(f"✅ .parameters() returns {len(params)} param groups (encoder)")

    # 5. Verify get_num_params() works
    n_params = model.get_num_params()
    print(f"✅ get_num_params() = {n_params:,}")

    # 6. Verify requires_training flag
    assert getattr(model, "requires_training", True) is False
    print(f"✅ requires_training = False")

    # 7. Verify fit() signature accepts DataLoaders
    train_loader = make_synthetic_loader()
    val_loader = make_synthetic_loader()

    cfg = {
        "training": {"batch_size": 4, "epochs": 1, "lr": 1e-3},
        "checkpointing": {"save_dir": "outputs/test_xgb"},
        "logging": {"log_dir": "outputs/test_xgb/logs", "experiment_name": "test_xgb"},
    }
    trainer = Trainer(cfg)
    history = trainer.fit(model, train_loader, val_loader)

    print(f"✅ Trainer.fit() completed via sklearn branch")
    print(f"   Validation loss: {history['val_loss'][-1]:.6f}")

    # 8. Verify predict_loader works
    preds, targets = model.predict_loader(val_loader)
    assert preds.shape == targets.shape
    print(f"✅ predict_loader() returns preds {preds.shape}, targets {targets.shape}")

    print("\n" + "=" * 60)
    print("All XGBoost verification checks PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
