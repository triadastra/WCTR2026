"""The run log is the only record a killed run leaves behind, so the guarantees
it makes are pinned here: artifacts land on disk each epoch rather than at the
end, a run's logger cannot capture another run's output, and a broken tracker
cannot take training down with it."""

import json
import logging

import pytest

from src.utils.logging import TrainingLogger, setup_logging
from src.utils.trackers import Tracker, TrackerGroup, build_trackers


def _logger(tmp_path, name="run_a", **kwargs):
    return TrainingLogger(log_dir=str(tmp_path / name), experiment_name=name, **kwargs)


# ---- crash safety -----------------------------------------------------------

def test_history_and_metrics_are_written_before_the_run_ends(tmp_path):
    """The whole point: kill a run mid-flight and its record still exists."""
    log = _logger(tmp_path)
    log.log_epoch(0, {"train_loss": 1.0, "val_loss": 2.0})
    log.log_epoch(1, {"train_loss": 0.5, "val_loss": 1.5})

    # No close(), no save_metrics() -- simulating a process that died here.
    assert log.history_path.exists()
    assert log.metrics_path.exists()

    rows = log.history_path.read_text().strip().splitlines()
    assert rows[0].split(",")[:3] == ["epoch", "phase", "train_loss"]
    assert len(rows) == 3                      # header + 2 epochs

    recorded = json.loads(log.metrics_path.read_text())
    assert [r["epoch"] for r in recorded] == [0, 1]
    assert recorded[1]["val_loss"] == 1.5


def test_status_heartbeat_tracks_progress_and_terminal_state(tmp_path):
    log = _logger(tmp_path, run_meta={"seed": 26})

    initial = json.loads(log.status_path.read_text())
    assert initial["state"] == "running"
    assert initial["seed"] == 26
    assert initial["pid"] > 0

    log.update_status(epoch=3, val_loss=0.04)
    mid = json.loads(log.status_path.read_text())
    assert mid["epoch"] == 3 and mid["state"] == "running"

    log.close(state="failed", summary={"error": "CUDA OOM"})
    final = json.loads(log.status_path.read_text())
    assert final["state"] == "failed"
    assert final["error"] == "CUDA OOM"
    assert "finished_at" in final


def test_a_fresh_run_does_not_inherit_a_previous_attempts_history(tmp_path):
    """Two different trainings must not concatenate into one CSV."""
    first = _logger(tmp_path)
    first.log_epoch(0, {"train_loss": 1.0, "val_loss": 2.0})
    first.close()

    second = _logger(tmp_path)                 # same dir, resumed=False
    second.log_epoch(0, {"train_loss": 9.0, "val_loss": 9.0})

    rows = second.history_path.read_text().strip().splitlines()
    assert len(rows) == 2                      # header + exactly one epoch
    assert json.loads(second.metrics_path.read_text())[0]["train_loss"] == 9.0


def test_a_resumed_run_extends_the_existing_history(tmp_path):
    first = _logger(tmp_path)
    first.log_epoch(0, {"train_loss": 1.0, "val_loss": 2.0})
    first.close()

    second = _logger(tmp_path, resumed=True)
    second.log_epoch(1, {"train_loss": 0.5, "val_loss": 1.5})

    rows = second.history_path.read_text().strip().splitlines()
    assert len(rows) == 3                      # header + both epochs, one table
    assert rows[0].startswith("epoch,phase,train_loss")


def test_history_columns_are_fixed_by_the_first_row(tmp_path):
    """A late extra key must not shift every column in the file."""
    log = _logger(tmp_path)
    log.log_epoch(0, {"train_loss": 1.0, "val_loss": 2.0})
    log.log_epoch(1, {"train_loss": 0.5, "val_loss": 1.5, "surprise": 7.0})

    rows = log.history_path.read_text().strip().splitlines()
    assert rows[0] == "epoch,phase,train_loss,val_loss"
    assert len(rows[1].split(",")) == len(rows[2].split(",")) == 4


# ---- logger isolation -------------------------------------------------------

def test_run_logger_does_not_hijack_the_root_logger(tmp_path):
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    before = list(root.handlers)

    _logger(tmp_path).log_info("hello")

    assert root.handlers == before, "constructing a run logger cleared root's handlers"
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.NullHandler)]


def test_two_runs_in_one_process_keep_separate_files(tmp_path):
    """A sweep builds many loggers; the second must not swallow the first."""
    a = _logger(tmp_path, name="run_a")
    b = _logger(tmp_path, name="run_b")

    a.log_info("belongs-to-a")
    b.log_info("belongs-to-b")

    text_a = (a.log_dir / "run_a.log").read_text()
    text_b = (b.log_dir / "run_b.log").read_text()
    assert "belongs-to-a" in text_a and "belongs-to-b" not in text_a
    assert "belongs-to-b" in text_b and "belongs-to-a" not in text_b


def test_same_experiment_name_in_different_directories_is_isolated(tmp_path):
    a = TrainingLogger(str(tmp_path / "a"), experiment_name="same")
    b = TrainingLogger(str(tmp_path / "b"), experiment_name="same")
    a.log_info("only-a")
    b.log_info("only-b")
    a.close()
    b.close()
    assert "only-a" in (tmp_path / "a/same.log").read_text()
    assert "only-a" not in (tmp_path / "b/same.log").read_text()


def test_attempt_events_survive_restart_and_capture_traceback(tmp_path):
    first = _logger(tmp_path)
    try:
        raise RuntimeError("test failure")
    except RuntimeError:
        first.log_exception("training failed")
    first.close("failed")
    second = _logger(tmp_path, resumed=True)
    second.close()
    rows = [json.loads(line) for line in first.events_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "failed", "started", "finished"]
    assert rows[0]["attempt_id"] != rows[2]["attempt_id"]
    assert "Traceback" in (first.log_dir / "run_a.log").read_text()


def test_batch_heartbeat_is_throttled_without_losing_epoch_status(tmp_path, monkeypatch):
    import src.utils.logging as logging_module
    log = _logger(tmp_path)
    ticks = iter([100.0, 110.0, 131.0])
    monkeypatch.setattr(logging_module.time, "monotonic", lambda: next(ticks))
    log.heartbeat(batch=1)
    log.heartbeat(batch=2)
    assert json.loads(log.status_path.read_text())["batch"] == 1
    log.heartbeat(batch=3)
    assert json.loads(log.status_path.read_text())["batch"] == 3
    log.close()


def test_library_logging_does_not_leak_into_the_run_log(tmp_path):
    log = _logger(tmp_path)
    logging.getLogger("some.third.party").warning("noise from a dependency")
    assert "noise from a dependency" not in (log.log_dir / "run_a.log").read_text()


def test_log_epoch_is_silent_by_default(tmp_path):
    """The trainer already writes a denser line; emitting here too doubled
    every run log for no added information."""
    log = _logger(tmp_path)
    log.log_epoch(0, {"train_loss": 1.0, "val_loss": 2.0})
    assert "Epoch 0 [train]" not in (log.log_dir / "run_a.log").read_text()

    log.log_epoch(1, {"train_loss": 1.0, "val_loss": 2.0}, emit=True)
    assert "Epoch 1 [train]" in (log.log_dir / "run_a.log").read_text()


def test_setup_logging_replaces_handlers_rather_than_stacking_them(tmp_path):
    name = "dup.check"
    setup_logging(log_file=str(tmp_path / "a.log"), log_to_file=True, name=name)
    setup_logging(log_file=str(tmp_path / "b.log"), log_to_file=True, name=name)
    assert len(logging.getLogger(name).handlers) == 2      # one console + one file


# ---- trackers ---------------------------------------------------------------

class _Exploding(Tracker):
    name = "boom"

    def log_metrics(self, step, metrics):
        raise RuntimeError("dashboard is down")


class _Recording(Tracker):
    name = "recorder"

    def __init__(self):
        self.seen = []

    def log_metrics(self, step, metrics):
        self.seen.append((step, metrics))


def test_a_failing_tracker_is_dropped_not_raised(tmp_path):
    """Losing a dashboard is an inconvenience; losing a 28-hour run is not."""
    good = _Recording()
    group = TrackerGroup([_Exploding(), good])

    group.log_metrics(0, {"loss": 1.0})        # must not raise
    group.log_metrics(1, {"loss": 0.5})

    assert len(group) == 1                     # the exploding one was removed
    assert [s for s, _ in good.seen] == [0, 1]


def test_unknown_and_missing_backends_degrade_to_no_ops(tmp_path):
    group = build_trackers(["not_a_real_backend"], str(tmp_path))
    assert len(group) == 0
    group.log_metrics(0, {"loss": 1.0})        # no-op, no raise


def test_no_trackers_requested_is_a_no_op(tmp_path):
    assert len(build_trackers(None, str(tmp_path))) == 0
    assert len(build_trackers([], str(tmp_path))) == 0


def test_tracker_receives_epoch_metrics(tmp_path):
    log = _logger(tmp_path)
    recorder = _Recording()
    log.trackers = TrackerGroup([recorder])

    log.log_epoch(0, {"train_loss": 1.0, "val_loss": 2.0})

    step, metrics = recorder.seen[0]
    assert step == 0
    assert metrics["train/train_loss"] == 1.0
    assert metrics["train/val_loss"] == 2.0
