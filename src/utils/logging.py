"""Logging utilities.

``TrainingLogger`` is the run's black box. Everything it records is written to
disk **as it happens**, not accumulated and dumped at the end, because the runs
this repo produces are long (the published sweep has single arms over 24h) and a
run that is killed, OOMs, or loses its box must still leave behind a complete
per-epoch record up to the moment it died.

Per run it maintains five artifacts under ``log_dir``:

``<experiment_name>.log``  human-readable text log
``history.csv``            one row per epoch, appended and flushed each epoch
``metrics.json``           the same rows as JSON, rewritten atomically each epoch
``status.json``            single-object heartbeat for the live monitor
                           (``scripts/train_status.py``)
``events.jsonl``           append-only start/end/failure events across attempts

Optional TensorBoard / W&B mirrors are attached through ``src.utils.trackers``
and never affect the files above.
"""

import csv
import io
import json
import logging
import os
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .trackers import build_trackers
from .run_artifacts import atomic_text, append_event


def _utc_now() -> str:
    """Timestamp for machine-read artifacts. UTC because runs are dispatched
    across boxes and a local-time heartbeat cannot be ordered against another
    machine's."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, text: str) -> None:
    """Replace ``path`` in one step.

    The monitor polls these files while training writes them. A plain
    ``open(w)`` leaves a window where the reader sees a truncated file and
    raises; writing a sibling and renaming makes the swap atomic on POSIX, so a
    reader sees either the old content or the new one.
    """
    atomic_text(path, text)


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = False,
    name: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (default: INFO)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        name: Logger name. ``None`` configures the root logger, which is
            appropriate for a script's global setup but NOT for a per-run
            logger -- see the note below.
        propagate: Whether records also travel to ancestor loggers. Off by
            default for named loggers so a run's file does not also collect
            every third-party library's output.

    Returns:
        Configured logger instance

    Note:
        Handlers are replaced, not appended, so calling this twice on the same
        name does not double every line. That is also why a per-run logger must
        pass ``name``: configuring the *root* logger per run means a second
        Trainer in the same process silently redirects the first one's output
        into the second one's file.
    """
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(log_level)
    logger.propagate = propagate if name else True

    # Close what we replace. Dropping the reference alone leaks the file
    # descriptor, which matters when a sweep builds hundreds of loggers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:                                  # noqa: BLE001
            pass

    formatter = logging.Formatter(
        fmt="%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = time.gmtime

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Logger specifically for training with metrics tracking.

    Writes every artifact incrementally so an interrupted run keeps its record.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        run_meta: Optional[Dict[str, Any]] = None,
        trackers: Optional[Iterable[str]] = None,
        tracker_options: Optional[Dict[str, Any]] = None,
        resumed: bool = False,
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (for log file naming)
            run_meta: Identifying fields (model, variant, seed, ...) copied into
                ``status.json`` so the monitor can label a run without parsing
                its directory name.
            trackers: Optional experiment-tracking backends ("tensorboard",
                "wandb"). Missing packages degrade to a no-op.
            tracker_options: Per-backend kwargs, e.g. ``{"wandb": {...}}``.
            resumed: True when continuing a previous run, so existing history is
                appended to rather than truncated.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        log_file = self.log_dir / f"{experiment_name}.log"

        # A per-run NAMED logger: two Trainers in one process keep separate
        # files, and library records do not leak into the run log.
        self.logger = setup_logging(
            log_file=str(log_file),
            log_to_console=True,
            log_to_file=True,
            name=f"train.{self.log_dir.resolve()}.{experiment_name}",
        )

        self.metrics_history = []
        self.history_path = self.log_dir / "history.csv"
        self.metrics_path = self.log_dir / "metrics.json"
        self.status_path = self.log_dir / "status.json"
        self.events_path = self.log_dir / "events.jsonl"
        self.attempt_id = uuid.uuid4().hex
        self._last_heartbeat = 0.0

        self._fieldnames = None
        self._resumed = resumed
        if not resumed:
            # A fresh run owns its artifacts; leftovers from a previous attempt
            # would silently concatenate two different trainings into one CSV.
            for stale in (self.history_path, self.metrics_path):
                stale.unlink(missing_ok=True)
        elif self.metrics_path.exists():
            try:
                existing_metrics = json.loads(self.metrics_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot resume malformed metrics history {self.metrics_path}: {exc}"
                ) from exc
            if not isinstance(existing_metrics, list):
                raise ValueError(
                    f"cannot resume {self.metrics_path}: expected a list of epoch rows"
                )
            self.metrics_history = existing_metrics

        self._status = {
            "run": experiment_name,
            "state": "running",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "resumed": bool(resumed),
            "attempt_id": self.attempt_id,
        }
        if run_meta:
            self._status.update(run_meta)
        self._write_status()
        append_event(self.events_path, "started", **self._status)

        self.trackers = build_trackers(
            trackers, str(self.log_dir), experiment_name, tracker_options, self.logger,
        )
        if len(self.trackers):
            self.logger.info("trackers  : %s", ", ".join(self.trackers.names))

    # ---- text ---------------------------------------------------------------

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_exception(self, message: str) -> None:
        self.logger.exception(message)

    def heartbeat(self, **fields: Any) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= 30:
            self.update_status(**fields)
            self._last_heartbeat = now

    # ---- per-epoch record ---------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        metrics: dict,
        phase: str = "train",
        emit: bool = False,
    ) -> None:
        """
        Record metrics for an epoch and flush them to disk.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            phase: Phase (train/val/test)
            emit: Also write a text line. Off by default -- the trainer already
                writes a denser, better-formatted line per epoch, and emitting
                here as well doubled the size of every run log for no added
                information.
        """
        if emit:
            metric_str = " | ".join(
                f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            self.logger.info(f"Epoch {epoch} [{phase}] - {metric_str}")

        record = {"epoch": epoch, "phase": phase, **metrics}
        self.metrics_history.append(record)

        # Flush both machine-readable artifacts now, not at the end of training.
        self._append_history_row(record)
        self.save_metrics()

        self.trackers.log_metrics(
            epoch,
            {f"{phase}/{k}": float(v) for k, v in metrics.items()
             if isinstance(v, (int, float))},
        )

    def _append_history_row(self, record: dict) -> None:
        """Append one CSV row, flushed and synced to disk.

        The column set is fixed by the first row of the run: a later row with a
        new key would otherwise shift every column silently. Extra keys are
        dropped and missing ones written empty, which keeps the file parseable
        no matter what a caller passes.
        """
        new_file = self._fieldnames is None
        if new_file:
            self._fieldnames = list(record.keys())
            # On resume, preserve all old and new columns. If the schema grew,
            # rewrite the existing rows once with the union header; appending a
            # second header would make standard CSV readers treat it as data.
            if self._resumed and self.history_path.exists():
                try:
                    with open(self.history_path, newline="") as fh:
                        reader = csv.DictReader(fh)
                        existing = reader.fieldnames
                        existing_rows = list(reader)
                    if existing:
                        union = list(existing)
                        union.extend(k for k in self._fieldnames if k not in union)
                        self._fieldnames = union
                        if union != existing:
                            buffer = io.StringIO(newline="")
                            writer = csv.DictWriter(buffer, fieldnames=union)
                            writer.writeheader()
                            writer.writerows(existing_rows)
                            _atomic_write(self.history_path, buffer.getvalue())
                        new_file = False
                except OSError:
                    pass

        row = {k: record.get(k, "") for k in self._fieldnames}
        with open(self.history_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())

    def save_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Save metrics history to JSON file.

        Args:
            save_path: Path to save metrics (default: log_dir/metrics.json)
        """
        path = Path(save_path) if save_path else self.metrics_path
        _atomic_write(path, json.dumps(self.metrics_history, indent=2, default=str))

    # ---- heartbeat ----------------------------------------------------------

    def update_status(self, **fields: Any) -> None:
        """Merge fields into ``status.json`` and stamp ``updated_at``.

        The monitor uses the stamp for liveness: a run whose heartbeat is older
        than a few of its own epochs is stalled, which is otherwise invisible
        until someone notices the log stopped growing.
        """
        self._status.update(fields)
        self._status["updated_at"] = _utc_now()
        self._write_status()

    def _write_status(self) -> None:
        _atomic_write(self.status_path, json.dumps(self._status, indent=2, default=str))

    # ---- teardown -----------------------------------------------------------

    def close(self, state: str = "finished", summary: Optional[Dict[str, Any]] = None) -> None:
        """Mark the run terminal and release the trackers.

        Args:
            state: "finished", "failed", or "interrupted". The monitor colours
                on this, so a crashed run must not be left reading "running".
            summary: Final values (best score, epochs run, wall time).
        """
        if summary:
            self._status.update(summary)
            self.trackers.log_summary(summary)
        self.update_status(state=state, finished_at=_utc_now())
        append_event(self.events_path, state, **self._status)
        self.trackers.close()

        # Release the file handle: a sweep constructs one logger per run in a
        # single process, and open handlers accumulate until it runs out.
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:                              # noqa: BLE001
                pass
