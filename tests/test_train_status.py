"""The monitor's job is to tell you which runs need attention, so the case that
matters is the one no other tool surfaces: a run whose heartbeat stopped but
whose own record still says "running"."""

import json
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_status as ts  # noqa: E402


NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


def _stamp(seconds_ago):
    return (NOW - timedelta(seconds=seconds_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(root, name, **fields):
    run_dir = root / name / "logs"
    run_dir.mkdir(parents=True)
    payload = {
        "run": name,
        "state": "running",
        "pid": 1,
        "host": "somewhere-else",       # keeps the PID liveness check out of it
        "started_at": _stamp(600),
        "updated_at": _stamp(5),
        "epoch": 10,
        "epochs_total": 100,
        "epoch_time_s": 6.0,
    }
    payload.update(fields)
    (run_dir / "status.json").write_text(json.dumps(payload))
    return run_dir / "status.json"


# ---- discovery --------------------------------------------------------------

def test_finds_runs_anywhere_in_the_tree(tmp_path):
    _write(tmp_path, "lstm_embeddings_seed26")
    _write(tmp_path, "s4_onehot_seed105")
    assert len(ts.find_runs(tmp_path)) == 2


def test_unreadable_status_is_skipped_not_fatal(tmp_path):
    _write(tmp_path, "good_run")
    bad = tmp_path / "bad_run" / "logs"
    bad.mkdir(parents=True)
    (bad / "status.json").write_text("{ truncated")   # a half-written file

    runs = ts.find_runs(tmp_path)
    assert [r["run"] for r in runs] == ["good_run"]


def test_empty_tree_renders_without_crashing(tmp_path):
    assert "no runs found" in ts.render(ts.build_rows([], NOW, 5.0), color=False)


# ---- liveness ---------------------------------------------------------------

def test_a_recent_heartbeat_is_running(tmp_path):
    run = json.loads(_write(tmp_path, "r", updated_at=_stamp(5)).read_text())
    state, _ = ts.classify(run, NOW, stale_factor=5.0)
    assert state == "running"


def test_a_run_that_stopped_writing_is_stalled(tmp_path):
    """6s epochs and no heartbeat for 10 minutes: the log simply stopped
    growing, which nothing else reports."""
    run = json.loads(_write(tmp_path, "r", updated_at=_stamp(600)).read_text())
    state, age = ts.classify(run, NOW, stale_factor=5.0)
    assert state == "stalled"
    assert age == pytest.approx(600, abs=2)


def test_staleness_scales_with_the_runs_own_epoch_time(tmp_path):
    """A 5-minute epoch is not stalled at 60s; a 6-second one is. Any fixed
    threshold would be wrong for most arms in the sweep."""
    slow = json.loads(_write(tmp_path, "slow", epoch_time_s=300.0,
                             updated_at=_stamp(400)).read_text())
    fast = json.loads(_write(tmp_path, "fast", epoch_time_s=6.0,
                             updated_at=_stamp(400)).read_text())
    assert ts.classify(slow, NOW, 5.0)[0] == "running"
    assert ts.classify(fast, NOW, 5.0)[0] == "stalled"


def test_a_run_before_its_first_epoch_gets_a_grace_period(tmp_path):
    """Data loading and model construction take minutes; a run with no epoch
    time yet must not be flagged the moment it starts."""
    run = json.loads(_write(tmp_path, "r", epoch=0, updated_at=_stamp(120),
                            epoch_time_s=None).read_text())
    assert ts.classify(run, NOW, 5.0)[0] == "running"


def test_terminal_states_are_reported_as_is(tmp_path):
    for reported in ("finished", "failed", "interrupted"):
        run = json.loads(
            _write(tmp_path, reported, state=reported, updated_at=_stamp(99999)).read_text()
        )
        # Age is irrelevant once a run is terminal -- a finished run from
        # yesterday is not "stalled".
        assert ts.classify(run, NOW, 5.0)[0] == reported


def test_a_dead_local_process_is_failed_not_stalled(tmp_path):
    """A PID that no longer exists on THIS box is proof, not inference."""
    run = json.loads(_write(tmp_path, "r", host=socket.gethostname(),
                            pid=2 ** 22, updated_at=_stamp(5)).read_text())
    assert ts.classify(run, NOW, 5.0)[0] == "failed"


def test_a_pid_from_another_host_is_not_probed(tmp_path):
    """PID 999999 on the GPU box says nothing about PID 999999 here."""
    run = json.loads(_write(tmp_path, "r", host="gpu-box-7", pid=2 ** 22,
                            updated_at=_stamp(5)).read_text())
    assert ts.classify(run, NOW, 5.0)[0] == "running"


def test_unparseable_timestamp_is_unknown_not_a_crash(tmp_path):
    run = json.loads(_write(tmp_path, "r", updated_at="not-a-timestamp").read_text())
    assert ts.classify(run, NOW, 5.0)[0] == "unknown"


# ---- presentation -----------------------------------------------------------

def test_runs_needing_attention_sort_first(tmp_path):
    _write(tmp_path, "z_finished", state="finished")
    _write(tmp_path, "a_running", updated_at=_stamp(5))
    _write(tmp_path, "m_failed", state="failed")
    _write(tmp_path, "q_stalled", updated_at=_stamp(9999))

    rows = ts.build_rows(ts.find_runs(tmp_path), NOW, 5.0)
    assert [r["state"] for r in rows] == ["failed", "stalled", "running", "finished"]


def test_table_reports_progress_and_the_early_stop_counter(tmp_path):
    _write(tmp_path, "lstm_embeddings_seed26", epoch=12, epochs_total=5000,
           val_loss=0.0501, best_val=0.0498, early_stop_counter=7,
           early_stop_patience=50)
    out = ts.render(ts.build_rows(ts.find_runs(tmp_path), NOW, 5.0), color=False)
    assert "lstm_embeddings_seed26" in out
    assert "12/5000" in out
    assert "0.049800" in out
    assert "7/50" in out


def test_a_failed_run_shows_its_error(tmp_path):
    _write(tmp_path, "oom_run", state="failed", error="RuntimeError: CUDA OOM")
    out = ts.render(ts.build_rows(ts.find_runs(tmp_path), NOW, 5.0), color=False)
    assert "CUDA OOM" in out


def test_nonfinite_batches_are_called_out(tmp_path):
    _write(tmp_path, "diverging", nonfinite_batches=3)
    out = ts.render(ts.build_rows(ts.find_runs(tmp_path), NOW, 5.0), color=False)
    assert "3 non-finite batch(es)" in out


def test_state_filter_narrows_the_snapshot(tmp_path):
    _write(tmp_path, "a", state="finished")
    _write(tmp_path, "b", updated_at=_stamp(5))
    assert len(ts.snapshot(tmp_path, 5.0, state_filter="finished")) == 1
