"""Local observability adapter; does not change the authors' S4 layers."""
import json
import math
import os
import socket
import time
import traceback
import uuid
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf

try:
    from .run_artifacts import append_event, atomic_text, utc_now
except ImportError:
    from run_artifacts import append_event, atomic_text, utc_now


class BenchmarkRunLogger(pl.Callback):
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        self.last_heartbeat = 0.0
        self.status = {}

    def update(self, **fields):
        self.status.update(fields, updated_at=utc_now())
        atomic_text(self.directory / "status.json",
                    json.dumps(self.status, indent=2, allow_nan=False))

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.status = {
            "attempt_id": uuid.uuid4().hex, "state": "running", "phase": "setup",
            "host": socket.gethostname(), "pid": os.getpid(), "started_at": utc_now(),
            "seed": self.config.train.seed, "model": self.config.dataset._name_,
            "device": str(pl_module.device), "epochs_total": trainer.max_epochs,
            "resumed": bool(trainer.ckpt_path),
            "run": self.directory.parent.parent.name,
        }
        self.update()
        append_event(self.directory / "events.jsonl", "started", **self.status)
        atomic_text(self.directory / "resolved_config.json", json.dumps(
            OmegaConf.to_container(self.config, resolve=True), indent=2, default=str))

    def heartbeat(self, trainer, batch_idx, phase):
        now = time.monotonic()
        if trainer.is_global_zero and now - self.last_heartbeat >= 30:
            self.update(epoch=trainer.current_epoch + 1, global_step=trainer.global_step,
                        batch=batch_idx, phase=phase)
            self.last_heartbeat = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.heartbeat(trainer, batch_idx, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0):
        if not trainer.sanity_checking:
            self.heartbeat(trainer, batch_idx, "validation")

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            metrics[str(key)] = value if math.isfinite(value) else None
        rates = [float(group["lr"]) for optimizer in trainer.optimizers
                 for group in optimizer.param_groups]
        row = {"epoch": trainer.current_epoch + 1, "global_step": trainer.global_step,
               "epoch_time_s": time.monotonic() - self.epoch_started,
               "learning_rates": rates, "metrics": metrics}
        append_event(self.directory / "epochs.jsonl", "epoch",
                     attempt_id=self.status["attempt_id"], **row)
        selection = getattr(trainer, "checkpoint_callback", None)
        score = getattr(selection, "best_model_score", None)
        summary = {"train_loss": metrics.get("train/loss"),
                   "val_loss": metrics.get("val/loss"), "lr": rates[0] if rates else None,
                   "best_val": float(score) if score is not None else None}
        for callback in getattr(trainer, "callbacks", []):
            if hasattr(callback, "wait_count"):
                summary.update(early_stop_counter=callback.wait_count,
                               early_stop_patience=callback.patience)
        self.update(**row, **summary, phase="epoch_end")

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.update(phase="publishing_artifacts")

    def finish(self):
        self.update(state="finished", finished_at=utc_now())
        append_event(self.directory / "events.jsonl", "finished", **self.status)

    def on_exception(self, trainer, pl_module, exception):
        if not trainer.is_global_zero:
            return
        state = "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed"
        self.update(state=state, error=f"{type(exception).__name__}: {exception}",
                    finished_at=utc_now())
        append_event(self.directory / "events.jsonl", state, **self.status,
                     traceback="".join(traceback.format_exception(
                         type(exception), exception, exception.__traceback__)))
