import csv
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.trainer import Trainer
from src.utils.logging import TrainingLogger


def _config(tmp_path, *, epochs=3, patience=1):
    return {
        "training": {
            "device": "cpu",
            "epochs": epochs,
            "lr": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": "none",
            "patience": patience,
            "min_delta": 1.0,
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
            "experiment_name": "resume-test",
        },
    }


class _Dataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = float(index + 1)
        return {
            "x_numeric": torch.tensor([[value]], dtype=torch.float32),
            "target_value": torch.tensor(value, dtype=torch.float32),
            "target_weight": torch.tensor(value * 2, dtype=torch.float32),
            "state_ids": torch.tensor(0),
            "comm_ids": torch.tensor(0),
            "flow_ids": torch.tensor(0),
        }


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 2)

    def forward(self, x_numeric, state_ids, comm_ids, flow_ids):
        return self.proj(x_numeric[:, -1, :])

    def get_num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())


def _loader(seed=7):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(_Dataset(), batch_size=2, shuffle=True, generator=generator)


def test_legacy_last_checkpoint_is_rejected(tmp_path):
    trainer = Trainer(_config(tmp_path))
    path = tmp_path / "legacy-last.pth"
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "score": 1.0,
        },
        path,
    )

    with pytest.raises(ValueError, match="not a resumable checkpoint"):
        trainer._read_resume_checkpoint(path)


def test_rng_state_round_trip_includes_loader_generator(tmp_path):
    trainer = Trainer(_config(tmp_path))
    generator = torch.Generator().manual_seed(41)
    trainer.train_loader = SimpleNamespace(generator=generator)

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state = trainer._capture_rng_state()
    expected = (
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )

    random.random()
    np.random.random()
    torch.rand(1)
    torch.rand(1, generator=generator)
    trainer._restore_rng_state(state)

    actual = (
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_resume_of_early_stopped_run_executes_no_extra_epoch(tmp_path):
    config = _config(tmp_path, epochs=3, patience=1)
    trainer = Trainer(config)
    history = trainer.fit(_Model(), _loader(), _loader(8))
    assert len(history["val_loss"]) == 2

    rows = json.loads((tmp_path / "logs" / "metrics.json").read_text())
    assert [row["early_stop_counter"] for row in rows] == [0.0, 1.0]

    resumed_config = _config(tmp_path, epochs=5, patience=1)
    resumed = Trainer(resumed_config)

    def unexpected_epoch():
        raise AssertionError("an already early-stopped run trained another epoch")

    resumed.train_epoch = unexpected_epoch
    resumed_history = resumed.fit(
        _Model(), _loader(), _loader(8), resume_from=True
    )
    assert len(resumed_history["val_loss"]) == 2

    status = json.loads((tmp_path / "logs" / "status.json").read_text())
    assert status["state"] == "finished"
    assert status["epochs_this_session"] == 0
    assert status["stopped_early"] is True


def test_sklearn_explicit_missing_resume_path_errors(tmp_path):
    trainer = Trainer(_config(tmp_path))
    model = SimpleNamespace(requires_training=False)

    with pytest.raises(FileNotFoundError, match="resume checkpoint not found"):
        trainer.fit(model, [], [], resume_from=tmp_path / "missing.pth")


def test_resumed_logger_preserves_metrics_json(tmp_path):
    first = TrainingLogger(str(tmp_path), "run")
    first.log_epoch(0, {"loss": 1.0})
    first.close()

    resumed = TrainingLogger(str(tmp_path), "run", resumed=True)
    resumed.log_epoch(1, {"loss": 0.5})
    resumed.close()

    rows = json.loads((tmp_path / "metrics.json").read_text())
    assert [row["epoch"] for row in rows] == [0, 1]


def test_resumed_logger_merges_csv_schema_without_second_header(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text("epoch,train_loss,val_loss\n0,1.0,1.1\n")

    logger = TrainingLogger(str(tmp_path), "run", resumed=True)
    logger.log_epoch(1, {"train_loss": 0.9, "val_loss": 1.0})
    logger.close()

    with history.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "0"
    assert rows[1]["epoch"] == "1"
    assert rows[1]["phase"] == "train"
