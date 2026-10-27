import hashlib
import json
import os
import stat
from types import SimpleNamespace

import pytest

from src.utils.run_artifacts import append_event, atomic_text, write_checkpoint_manifest


def test_manifest_requires_files_and_records_content_hashes(tmp_path):
    atomic_text(tmp_path / "best.pth", "checkpoint bytes")
    result = write_checkpoint_manifest(tmp_path, {"best": "best.pth"},
                                       kind="test", run_contract={"seed": 26})
    assert result == json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert result["files"] == [{"role": "best", "path": "best.pth", "bytes": 16,
                                "sha256": hashlib.sha256(b"checkpoint bytes").hexdigest()}]
    original = (tmp_path / "checkpoint_manifest.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        write_checkpoint_manifest(tmp_path, {"best": "missing.pth"},
                                  kind="test", run_contract={})
    assert (tmp_path / "checkpoint_manifest.json").read_bytes() == original


def test_manifest_rejects_empty_or_outside_artifacts(tmp_path):
    for files in ({}, {"best": "../outside.pth"}):
        with pytest.raises(ValueError):
            write_checkpoint_manifest(tmp_path, files, kind="test", run_contract={})
    (tmp_path / "empty.pth").touch()
    with pytest.raises(ValueError, match="Empty"):
        write_checkpoint_manifest(tmp_path, {"best": "empty.pth"},
                                  kind="test", run_contract={})


def test_atomic_write_failure_preserves_old_file(tmp_path, monkeypatch):
    import src.utils.run_artifacts as artifacts
    path = tmp_path / "status.json"
    atomic_text(path, "original")
    def fail(*args):
        raise OSError("disk error")
    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError, match="disk error"):
        atomic_text(path, "replacement")
    assert path.read_text() == "original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["status.json"]


def test_events_append_and_encode_nonfinite_as_null(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, "started", attempt_id="one")
    append_event(path, "failed", loss=float("nan"))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in events] == ["started", "failed"]
    assert events[1]["loss"] is None
    assert all(row["timestamp"].endswith("Z") for row in events)


def test_checkpoint_write_failure_preserves_previous_weights(tmp_path, monkeypatch):
    import torch
    from src.training.callbacks import atomic_torch_save
    path = tmp_path / "last.pth"
    atomic_torch_save({"weights": torch.tensor([3.0])}, path)
    original = path.read_bytes()
    def fail(payload, stream):
        stream.write(b"partial checkpoint")
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save({}, path)
    assert path.read_bytes() == original
    assert torch.load(path, weights_only=True)["weights"].item() == 3.0
    assert [file.name for file in tmp_path.iterdir()] == ["last.pth"]


def test_s4_local_logger_records_epochs_and_failure(tmp_path):
    pytest.importorskip("pytorch_lightning")
    omega = pytest.importorskip("omegaconf")
    import torch
    from src.utils.s4_logging import BenchmarkRunLogger
    config = omega.OmegaConf.create({"train": {"seed": 26},
                                    "dataset": {"_name_": "s4_embeddings"}})
    callback = BenchmarkRunLogger(tmp_path, config)
    trainer = SimpleNamespace(is_global_zero=True, max_epochs=3, ckpt_path=None,
                              current_epoch=0, global_step=2,
                              callback_metrics={"val/loss": torch.tensor(0.25)},
                              optimizers=[SimpleNamespace(param_groups=[{"lr": 1e-4}])])
    module = SimpleNamespace(device="cpu")
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_end(trainer, module, None, None, 1)
    callback.on_train_epoch_end(trainer, module)
    epoch = json.loads((tmp_path / "epochs.jsonl").read_text())
    assert epoch["metrics"]["val/loss"] == 0.25
    assert epoch["learning_rates"] == [1e-4]
    callback.on_exception(trainer, module, RuntimeError("diagnostic failure"))
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "failed" and status["epoch"] == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["traceback"].endswith("RuntimeError: diagnostic failure\n")


def test_s4_nonzero_rank_does_not_write(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from src.utils.s4_logging import BenchmarkRunLogger
    callback = BenchmarkRunLogger(tmp_path, None)
    trainer = SimpleNamespace(is_global_zero=False)
    callback.on_fit_start(trainer, None)
    callback.on_train_epoch_end(trainer, None)
    callback.on_exception(trainer, None, RuntimeError())
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory synchronization")
@pytest.mark.parametrize("kind", ["text", "torch", "tree"])
def test_replacement_is_followed_by_directory_sync(tmp_path, monkeypatch, kind):
    import src.utils.run_artifacts as artifacts
    from src.training.callbacks import atomic_torch_save
    from src.models.tabular_common import save_tree_bundle
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def sync(descriptor):
        events.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        return real_sync(descriptor)

    def replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "fsync", sync)
    monkeypatch.setattr(artifacts.os, "replace", replace)
    if kind == "text":
        atomic_text(tmp_path / "status.json", "{}")
    elif kind == "torch":
        atomic_torch_save({"epoch": 1}, tmp_path / "last.pth")
    else:
        save_tree_bundle(tmp_path / "best_xgb", {"value": 1}, {"weight": 2}, {})
    assert events[0] == "file" and events[-1] == "directory"
    assert "replace" in events
    for index, event in enumerate(events):
        if event == "replace":
            assert "directory" in events[index + 1:]


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory synchronization")
@pytest.mark.parametrize("kind", ["text", "torch", "tree"])
def test_directory_sync_failure_is_not_reported_as_success(tmp_path, monkeypatch, kind):
    import src.utils.run_artifacts as artifacts
    from src.training.callbacks import atomic_torch_save
    from src.models.tabular_common import save_tree_bundle
    real_sync = os.fsync

    def sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory sync failed")
        return real_sync(descriptor)

    monkeypatch.setattr(artifacts.os, "fsync", sync)
    with pytest.raises(OSError, match="directory sync failed"):
        if kind == "text":
            atomic_text(tmp_path / "status.json", "{}")
        elif kind == "torch":
            atomic_torch_save({"epoch": 1}, tmp_path / "last.pth")
        else:
            save_tree_bundle(tmp_path / "best_xgb", {}, {}, {})
