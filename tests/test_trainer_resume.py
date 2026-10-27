"""Resuming has to restore the run, not just the weights.

The optimizer's moments, where the LR schedule had reached, and how many epochs
early stopping had already waited all decide what the next epochs do. A resume
that reloads only ``model_state_dict`` restarts the schedule and the patience
window, so it trains a different run than the one it claims to continue -- and
on this benchmark that silently changes the published number.
"""

import csv
import json
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.trainer import Trainer


class _Tiny(nn.Module):
    """Smallest model satisfying the per-series contract the trainer expects."""

    requires_training = True
    variant = "test"

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, x_numeric, state_ids=None, comm_ids=None, flow_ids=None):
        return self.net(x_numeric[:, -1, :])

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def _batches(n_batches=3, batch=8):
    torch.manual_seed(0)
    return [
        {
            "x_numeric": torch.randn(batch, 3, 4),
            "target_value": torch.randn(batch),
            "target_weight": torch.randn(batch),
            "state_ids": torch.zeros(batch, dtype=torch.long),
            "comm_ids": torch.zeros(batch, dtype=torch.long),
            "flow_ids": torch.zeros(batch, dtype=torch.long),
        }
        for _ in range(n_batches)
    ]


def _config(out_dir, epochs, patience=100, *, embedded_contract=False):
    config = {
        "training": {
            "device": "cpu", "epochs": epochs, "lr": 1e-2, "optimizer": "adamw",
            "weight_decay": 1e-4, "patience": patience, "min_delta": 1e-9,
            "scheduler": "step", "batch_size": 8,
        },
        "checkpointing": {"save_dir": str(out_dir), "save_last": True,
                          "save_best_only": True, "monitor": "val_loss", "mode": "min"},
        "logging": {"log_dir": str(out_dir / "logs"), "experiment_name": "resume_test"},
    }
    if embedded_contract:
        config["logging"]["run_contract"] = {
            "run_contract_version": 1,
            "training_config": {"epochs": epochs, "lr": 1e-2},
            "source_fingerprint": "a" * 64,
        }
    return config


def _run(out_dir, epochs, resume_from=None, patience=100):
    torch.manual_seed(0)
    trainer = Trainer(_config(out_dir, epochs, patience))
    trainer.fit(_Tiny(), _batches(), _batches(2), resume_from=resume_from)
    return trainer


def _history_rows(out_dir):
    with open(out_dir / "logs" / "history.csv", newline="") as fh:
        return list(csv.DictReader(fh))


# ---- the resume contract ----------------------------------------------------

def test_resume_continues_from_the_next_epoch(tmp_path):
    _run(tmp_path, epochs=3)
    trainer = _run(tmp_path, epochs=6, resume_from=True)

    assert trainer.start_epoch == 3, "should restart after the last completed epoch"
    rows = _history_rows(tmp_path)
    assert [int(r["epoch"]) for r in rows] == [0, 1, 2, 3, 4, 5]


def test_resume_allows_higher_epoch_ceiling_in_embedded_run_contract(tmp_path):
    first = Trainer(_config(tmp_path, 2, embedded_contract=True))
    first.fit(_Tiny(), _batches(), _batches(2))

    resumed = Trainer(_config(tmp_path, 4, embedded_contract=True))
    resumed.fit(_Tiny(), _batches(), _batches(2), resume_from=True)
    assert resumed.start_epoch == 2
    assert len(resumed.history["val_loss"]) == 4


def test_extended_resume_without_new_best_preserves_selected_weights(tmp_path):
    def config(epochs):
        cfg = _config(tmp_path, epochs, embedded_contract=True)
        cfg["training"]["lr"] = 0.0
        cfg["logging"]["run_contract"]["training_config"]["lr"] = 0.0
        return cfg

    first = Trainer(config(2))
    first.fit(_Tiny(), _batches(), _batches(2))
    old = torch.load(tmp_path / "best.pth", weights_only=False)
    resumed = Trainer(config(4))
    resumed.fit(_Tiny(), _batches(), _batches(2), resume_from=True)
    selected = torch.load(tmp_path / "best.pth", weights_only=False)
    assert selected["epoch"] == old["epoch"] == 0
    for key, value in old["model_state_dict"].items():
        assert torch.equal(selected["model_state_dict"][key], value)
    assert selected["run_contract"] == resumed.run_contract
    assert selected["selection_origin_run_contract"] == old["run_contract"]


def test_resume_cannot_reduce_the_saved_epoch_ceiling(tmp_path):
    _run(tmp_path, epochs=3)
    with pytest.raises(ValueError, match="benchmark contract mismatch.*epochs_total"):
        _run(tmp_path, epochs=2, resume_from=True)


def test_completion_failure_never_reports_finished(tmp_path, monkeypatch):
    import src.training.trainer as module
    def fail(*args, **kwargs):
        raise OSError("receipt publication failed")
    monkeypatch.setattr(module, "write_completion", fail)
    trainer = Trainer(_config(tmp_path, 1))
    with pytest.raises(OSError, match="receipt publication failed"):
        trainer.fit(_Tiny(), _batches(), _batches(2))
    status = json.loads((tmp_path / "logs/status.json").read_text())
    assert status["state"] != "finished"
    assert "finished  :" not in (tmp_path / "logs/resume_test.log").read_text()
    trainer.logger.close(state="failed")


def test_invalid_resume_does_not_publish_new_data_artifacts(tmp_path):
    first = Trainer(_config(tmp_path, 2, embedded_contract=True))
    first.fit(_Tiny(), _batches(), _batches(2))
    logs_before = {p.relative_to(tmp_path): p.read_bytes()
                   for p in (tmp_path / "logs").rglob("*") if p.is_file()}
    sidecar = tmp_path / "source_provenance.json"
    sidecar.write_text("original snapshot")
    cfg = _config(tmp_path, 4, embedded_contract=True)
    cfg["logging"]["run_contract"]["source_fingerprint"] = "b" * 64
    trainer = Trainer(cfg)
    with pytest.raises(ValueError, match="run contract mismatch"):
        trainer.fit(
            _Tiny(), _batches(), _batches(2), resume_from=True,
            on_ready=lambda: sidecar.write_text("different snapshot"),
        )
    assert {p.relative_to(tmp_path): p.read_bytes()
            for p in (tmp_path / "logs").rglob("*") if p.is_file()} == logs_before
    assert sidecar.read_text() == "original snapshot"


def test_ready_callback_runs_after_resume_validation(tmp_path):
    _run(tmp_path, epochs=2)
    calls = []
    trainer = Trainer(_config(tmp_path, 3))
    trainer.fit(
        _Tiny(), _batches(), _batches(2), resume_from=True,
        on_ready=lambda: calls.append(len(trainer.history["val_loss"])),
    )
    assert calls == [2]


@pytest.mark.parametrize("field,message", [
    ("optimizer_state_dict", "optimizer state"),
    ("scheduler_state_dict", "scheduler state"),
])
def test_resume_cannot_silently_reset_missing_training_state(tmp_path, field, message):
    _run(tmp_path, epochs=2)
    checkpoint = tmp_path / "last.pth"
    payload = torch.load(checkpoint, weights_only=False)
    payload[field] = None
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, epochs=3, resume_from=True)


def test_resume_extends_one_history_table_rather_than_starting_a_second(tmp_path):
    _run(tmp_path, epochs=2)
    _run(tmp_path, epochs=4, resume_from=True)

    text = (tmp_path / "logs" / "history.csv").read_text().strip().splitlines()
    assert sum(1 for line in text if line.startswith("epoch,")) == 1
    assert len(text) == 5                       # one header + four epochs


def test_resume_restores_the_optimizer_state(tmp_path):
    """Adam moments carry the run's history; dropping them is a different run."""
    first = _run(tmp_path, epochs=3)
    saved_steps = [
        int(s["step"]) for s in first.optimizer.state.values() if "step" in s
    ]

    second = _run(tmp_path, epochs=4, resume_from=True)
    resumed_steps = [
        int(s["step"]) for s in second.optimizer.state.values() if "step" in s
    ]
    assert saved_steps and resumed_steps
    assert min(resumed_steps) > min(saved_steps), "optimizer restarted from zero"


def test_resume_restores_the_lr_schedule(tmp_path):
    """A restarted schedule would put the continuation back at the initial LR."""
    first = _run(tmp_path, epochs=3)
    lr_at_stop = first.optimizer.param_groups[0]["lr"]

    second = _run(tmp_path, epochs=4, resume_from=True)
    assert second.scheduler.last_epoch > first.scheduler.last_epoch
    assert second.optimizer.param_groups[0]["lr"] <= lr_at_stop


def test_resume_restores_the_early_stopping_counter(tmp_path):
    """Otherwise the continuation trains past where the run would have stopped."""
    _run(tmp_path, epochs=4, patience=100)
    payload = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)
    saved = payload["early_stopping"]["counter"]

    trainer = Trainer(_config(tmp_path, 5, patience=100))
    trainer.fit(_Tiny(), _batches(), _batches(2), resume_from=True)
    assert trainer.early_stopping.counter >= saved


def test_resume_keeps_the_best_score_so_a_worse_epoch_cannot_overwrite_best(tmp_path):
    _run(tmp_path, epochs=3)
    best_before = torch.load(
        tmp_path / "last.pth", map_location="cpu", weights_only=False
    )["best_metric"]

    trainer = _run(tmp_path, epochs=4, resume_from=True)
    assert trainer.checkpoint_manager.best_score <= best_before


# ---- opting in --------------------------------------------------------------

def test_no_resume_flag_starts_a_clean_run(tmp_path):
    """The default must stay byte-identical to the published protocol."""
    _run(tmp_path, epochs=3)
    trainer = _run(tmp_path, epochs=2)          # no resume_from

    assert trainer.start_epoch == 0
    assert [int(r["epoch"]) for r in _history_rows(tmp_path)] == [0, 1]


def test_resume_true_on_a_fresh_directory_just_starts(tmp_path):
    """The form a sweep wants: idempotent without testing for the file."""
    trainer = _run(tmp_path, epochs=2, resume_from=True)
    assert trainer.start_epoch == 0 and trainer.resumed is False


def test_an_explicit_missing_checkpoint_is_an_error(tmp_path):
    """Unlike --resume, naming a checkpoint that isn't there is a mistake."""
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, epochs=2, resume_from=str(tmp_path / "nope.pth"))


def test_best_pth_is_rejected_as_a_resume_source(tmp_path):
    """best.pth holds bare weights -- resuming from it would silently discard
    the optimizer, the schedule and the patience window."""
    _run(tmp_path, epochs=2)
    with pytest.raises(ValueError, match="not a resumable checkpoint"):
        _run(tmp_path, epochs=3, resume_from=str(tmp_path / "best.pth"))


# ---- what the run records ---------------------------------------------------

def test_checkpoint_carries_everything_a_resume_needs(tmp_path):
    _run(tmp_path, epochs=2)
    payload = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "early_stopping", "best_metric", "best_epoch", "history", "epoch_rows",
                "rng_state", "resume_contract"):
        assert key in payload, f"last.pth is missing {key}"


def test_resume_rejects_a_checkpoint_from_another_seed(tmp_path):
    first_config = _config(tmp_path, 2)
    first_config["logging"]["run_meta"] = {
        "model_key": "tiny", "variant_key": "test", "seed": 26,
        "aggregate": False,
    }
    Trainer(first_config).fit(_Tiny(), _batches(), _batches(2))

    second_config = _config(tmp_path, 4)
    second_config["logging"]["run_meta"] = {
        "model_key": "tiny", "variant_key": "test", "seed": 105,
        "aggregate": False,
    }
    with pytest.raises(ValueError, match="benchmark contract mismatch.*identity"):
        Trainer(second_config).fit(
            _Tiny(), _batches(), _batches(2), resume_from=True
        )


def test_split_resume_matches_an_uninterrupted_run(tmp_path):
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"

    uninterrupted = _run(full_dir, epochs=4)
    _run(split_dir, epochs=2)
    resumed = _run(split_dir, epochs=4, resume_from=True)

    assert resumed.history == uninterrupted.history
    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name


def test_resume_restores_the_random_streams(tmp_path):
    """Dropout masks and shuffle order come from the RNG. A resume that does not
    restore it produces a different run than the uninterrupted one it claims to
    continue -- which on this benchmark silently moves the published number."""
    _run(tmp_path, epochs=2)
    saved = torch.load(tmp_path / "last.pth", map_location="cpu",
                       weights_only=False)["rng_state"]

    torch.manual_seed(999)                      # scramble every global stream
    random.seed(999)
    np.random.seed(999)
    assert not torch.equal(torch.get_rng_state(), saved["torch_cpu"])

    # epochs == the epoch already reached, so the loop body never runs and the
    # only thing under test is the restore itself.
    trainer = Trainer(_config(tmp_path, 2))
    trainer.fit(_Tiny(), _batches(), _batches(2), resume_from=True)

    assert torch.equal(torch.get_rng_state(), saved["torch_cpu"])
    assert random.getstate() == saved["python"]
    assert np.random.get_state()[1].tolist() == saved["numpy"][1].tolist()


def test_an_incomplete_legacy_checkpoint_is_refused(tmp_path):
    """Better to refuse than to resume badly: a checkpoint from before the RNG
    state was captured cannot reproduce the run it continues."""
    _run(tmp_path, epochs=2)
    path = tmp_path / "last.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("rng_state")
    torch.save(payload, path)

    with pytest.raises(ValueError, match="rng_state"):
        _run(tmp_path, epochs=3, resume_from=True)


def test_epoch_rows_carry_the_added_telemetry(tmp_path):
    _run(tmp_path, epochs=2)
    row = _history_rows(tmp_path)[0]
    for column in ("samples_per_s", "gpu_mem_peak_mb", "early_stop_counter",
                   "batch_loss_min", "batch_loss_max", "val_value_mse",
                   "val_weight_mse", "lr_group0"):
        assert column in row, f"history.csv is missing {column}"
    assert float(row["samples_per_s"]) > 0


def test_per_channel_val_metrics_do_not_change_the_selection_metric(tmp_path):
    """val_value_mse / val_weight_mse are reported; val_loss still selects."""
    trainer = _run(tmp_path, epochs=2)
    row = _history_rows(tmp_path)[-1]
    combined = float(row["val_loss"])
    halves = (float(row["val_value_mse"]) + float(row["val_weight_mse"])) / 2.0
    assert combined == pytest.approx(halves, rel=1e-6)
    assert trainer.history["val_loss"][-1] == pytest.approx(combined, rel=1e-9)


def test_status_reaches_a_terminal_state_on_a_clean_finish(tmp_path):
    _run(tmp_path, epochs=2)
    status = json.loads((tmp_path / "logs" / "status.json").read_text())
    assert status["state"] == "finished"
    assert status["epochs_run"] == 2
    assert status["model"] == "_Tiny"


@pytest.mark.parametrize("already_stopped", [False, True])
def test_explicit_resume_carries_selected_weights_to_new_directory(tmp_path, already_stopped):
    source, destination = tmp_path / "source", tmp_path / "continued"
    def config(path, epochs):
        cfg = _config(path, epochs, patience=1 if already_stopped else 100,
                      embedded_contract=True)
        cfg["training"]["lr"] = 0.0
        cfg["logging"]["run_contract"]["training_config"]["lr"] = 0.0
        return cfg
    Trainer(config(source, 2)).fit(_Tiny(), _batches(), _batches(2))
    original_bytes = (source / "best.pth").read_bytes()
    selected = torch.load(source / "best.pth", weights_only=False)
    resumed = Trainer(config(destination, 4))
    resumed.fit(_Tiny(), _batches(), _batches(2), resume_from=source / "last.pth")
    actual = torch.load(destination / "best.pth", weights_only=False)
    assert actual["epoch"] == selected["epoch"] == 0
    for key, value in selected["model_state_dict"].items():
        assert torch.equal(actual["model_state_dict"][key], value)
    assert actual["run_contract"] == resumed.run_contract
    assert (destination / "artifacts/training_completion.json").is_file()
    continued = torch.load(destination / "last.pth", weights_only=False)
    assert continued["run_contract"] == resumed.run_contract
    inventory = json.loads((destination / "checkpoint_manifest.json").read_text())
    assert {entry["role"] for entry in inventory["files"]} == {"best", "resume"}
    assert (source / "best.pth").read_bytes() == original_bytes


def test_bad_selected_resume_source_is_rejected_before_publication(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "continued"
    _run(source, epochs=2)
    selected = torch.load(source / "best.pth", weights_only=False)
    selected["score"] += 1
    torch.save(selected, source / "best.pth")
    with pytest.raises(ValueError, match="selected checkpoint does not match resume source"):
        _run(destination, epochs=4, resume_from=source / "last.pth")
    assert not (destination / "best.pth").exists()
    assert not (destination / "logs").exists()
