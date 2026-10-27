"""Optional experiment-tracking backends (TensorBoard, Weights & Biases).

The file artifacts written by ``TrainingLogger`` are the source of truth and are
always produced. These backends are *additive* mirrors, so a run must behave
identically whether or not they are installed or enabled:

* an unknown / disabled backend is a no-op, not an error;
* a missing package logs one warning and degrades to a no-op;
* a backend that raises mid-run is disabled rather than allowed to take the
  training run down with it. Losing a dashboard is an inconvenience; losing a
  28-hour run to a logging exception is not acceptable.

Nothing here is imported at module scope beyond the standard library, so adding
this module does not add a dependency to the project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


class Tracker:
    """Uniform interface every backend implements."""

    name = "tracker"

    def log_config(self, config: Dict[str, Any]) -> None:  # pragma: no cover - base
        pass

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:  # pragma: no cover - base
        pass

    def log_summary(self, summary: Dict[str, Any]) -> None:  # pragma: no cover - base
        pass

    def close(self) -> None:  # pragma: no cover - base
        pass


class TensorBoardTracker(Tracker):
    """Mirror scalars into a TensorBoard event file under ``<log_dir>/tb``."""

    name = "tensorboard"

    def __init__(self, log_dir: str, run_name: Optional[str] = None):
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        path = Path(log_dir) / "tb"
        if run_name:
            path = path / run_name
        path.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(path))

    def log_config(self, config: Dict[str, Any]) -> None:
        # add_hparams wants a flat mapping of scalars/strings and insists on a
        # metric dict, which we do not have before the first epoch. Text is
        # lossless and always renders, so the run's configuration is readable
        # from the dashboard without inventing a placeholder metric.
        lines = [f"{k}: {v}" for k, v in sorted(config.items())]
        self._writer.add_text("config", "\n\n".join(lines), 0)

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)
        self._writer.flush()

    def log_summary(self, summary: Dict[str, Any]) -> None:
        lines = [f"{k}: {v}" for k, v in sorted(summary.items())]
        self._writer.add_text("summary", "\n\n".join(lines), 0)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class WandbTracker(Tracker):
    """Mirror scalars into a Weights & Biases run."""

    name = "wandb"

    def __init__(
        self,
        log_dir: str,
        run_name: Optional[str] = None,
        project: str = "wctr2026",
        entity: Optional[str] = None,
        mode: str = "online",
    ):
        import wandb  # noqa: PLC0415

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            dir=str(log_dir),
            mode=mode,
            reinit=True,
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        self._run.config.update(config, allow_val_change=True)

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        self._run.log(metrics, step=step)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        for key, value in summary.items():
            self._run.summary[key] = value

    def close(self) -> None:
        self._run.finish()


_BACKENDS = {
    "tensorboard": TensorBoardTracker,
    "tb": TensorBoardTracker,
    "wandb": WandbTracker,
    "wb": WandbTracker,
}


class TrackerGroup(Tracker):
    """Fan one call out to every enabled backend, swallowing their failures.

    A backend that raises is dropped for the rest of the run and reported once.
    Retrying every epoch would turn a broken dashboard into thousands of log
    lines, and re-raising would make a mirror able to kill the run it mirrors.
    """

    name = "group"

    def __init__(self, trackers: List[Tracker], logger: Optional[logging.Logger] = None):
        self._trackers = list(trackers)
        self._log = logger or logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self._trackers)

    @property
    def names(self) -> List[str]:
        return [t.name for t in self._trackers]

    def _fan_out(self, method: str, *args) -> None:
        for tracker in list(self._trackers):
            try:
                getattr(tracker, method)(*args)
            except Exception as exc:                       # noqa: BLE001
                self._trackers.remove(tracker)
                self._log.warning(
                    "tracker '%s' failed on %s (%s); disabled for this run",
                    tracker.name, method, exc,
                )

    def log_config(self, config: Dict[str, Any]) -> None:
        self._fan_out("log_config", config)

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        self._fan_out("log_metrics", step, metrics)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        self._fan_out("log_summary", summary)

    def close(self) -> None:
        self._fan_out("close")


def build_trackers(
    backends,
    log_dir: str,
    run_name: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> TrackerGroup:
    """Construct the enabled backends, skipping any that cannot be created.

    Args:
        backends: Backend names (list, or a comma-separated string). Empty /
            None disables tracking entirely.
        log_dir: Run log directory; backends write beneath it.
        run_name: Display name for the run.
        options: Per-backend keyword arguments, e.g.
            ``{"wandb": {"project": "wctr2026"}}``.
        logger: Where to report backends that could not be constructed.

    Returns:
        A ``TrackerGroup``. It is a no-op when nothing could be enabled, so
        callers never need to branch on whether tracking is on.
    """
    log = logger or logging.getLogger(__name__)
    if not backends:
        return TrackerGroup([], log)
    if isinstance(backends, str):
        backends = [b.strip() for b in backends.split(",") if b.strip()]

    options = options or {}
    built: List[Tracker] = []
    for name in backends:
        key = str(name).strip().lower()
        cls = _BACKENDS.get(key)
        if cls is None:
            log.warning("unknown tracker backend '%s'; skipped", name)
            continue
        try:
            built.append(cls(log_dir=log_dir, run_name=run_name, **options.get(key, {})))
        except ImportError as exc:
            log.warning(
                "tracker '%s' requested but its package is not installed (%s); "
                "skipped -- file logging is unaffected", key, exc,
            )
        except Exception as exc:                           # noqa: BLE001
            log.warning("tracker '%s' failed to initialise (%s); skipped", key, exc)
    return TrackerGroup(built, log)
