"""
Unified trainer class for all models.

Handles training loop, validation, checkpointing, and early stopping.
"""

import math
import random
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from ..utils.training_completion import completion_path, write_completion
from .losses import MSELoss, MaskedMSELoss
from .callbacks import EarlyStopping, CheckpointManager, atomic_torch_save


def _fmt_duration(seconds: float) -> str:
    """Compact h/m/s for log lines (ETA, totals)."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
from ..utils.logging import TrainingLogger
from ..utils.run_artifacts import write_checkpoint_manifest


class Trainer:
    """
    Unified trainer for all deep learning models.

    Supports:
    - Multiple optimizers (Adam, AdamW, SGD)
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Mixed precision training (AMP)
    - Gradient clipping
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.device = self._get_device()

        # Training parameters
        training_cfg = config.get("training", {})
        self.batch_size = training_cfg.get("batch_size", 32)
        self.epochs = training_cfg.get("epochs", 200)
        self.lr = training_cfg.get("lr", 1e-3)
        self.weight_decay = training_cfg.get("weight_decay", 0.0)
        # Adam/AdamW moment hyperparameters. The config has always declared
        # these; they were previously never read, which silently pinned the
        # torch defaults. The defaults here equal those torch defaults, so
        # wiring them is a no-op for every recorded run.
        self.betas = tuple(training_cfg.get("betas", (0.9, 0.999)))
        self.eps = float(training_cfg.get("eps", 1e-8))
        # False reproduces the pre-exemption behaviour; see _param_groups.
        self.weight_decay_exempt = training_cfg.get("weight_decay_exempt", True)
        self.patience = training_cfg.get("patience", 10)
        self.min_delta = training_cfg.get("min_delta", 1e-4)

        # Optimizer and scheduler
        self.optimizer_name = training_cfg.get("optimizer", "adam")
        self.scheduler_name = training_cfg.get("scheduler", "reduce_on_plateau")
        # Scale with the early-stopping patience: the plateau scheduler must not
        # anneal the LR to nothing before early stopping has a chance to fire.
        self.scheduler_patience = training_cfg.get("scheduler_patience", 25)
        self.scheduler_factor = training_cfg.get("scheduler_factor", 0.5)
        self.grad_clip = training_cfg.get("grad_clip")
        self.grad_clip_norm = training_cfg.get("grad_clip_norm", 1.0)

        # Mixed precision
        self.use_amp = training_cfg.get("use_amp", False)
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Checkpointing
        ckpt_cfg = config.get("checkpointing", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "outputs/runs"))
        self.save_best_only = ckpt_cfg.get("save_best_only", True)
        self.monitor = ckpt_cfg.get("monitor", "val_loss")
        self.mode = ckpt_cfg.get("mode", "min")
        self.save_last = ckpt_cfg.get("save_last", True)

        # The implementation below always computes/returns validation MSE; it
        # does not dynamically look up an arbitrary metric named by `monitor`.
        # Accepting other values previously made the configuration lie and, in
        # mode=max, deliberately selected the worst-loss checkpoint.
        if self.monitor != "val_loss" or self.mode != "min":
            raise ValueError(
                "Trainer only supports checkpointing monitor='val_loss' with "
                "mode='min'"
            )
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"training.epochs must be a positive integer, got {self.epochs!r}")
        if not isinstance(self.patience, int) or self.patience <= 0:
            raise ValueError(
                f"training.patience must be a positive integer, got {self.patience!r}"
            )
        if not math.isfinite(float(self.lr)) or self.lr < 0:
            raise ValueError(f"training.lr must be finite and non-negative, got {self.lr!r}")

        # Logging
        log_cfg = config.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "outputs/logs")
        self.experiment_name = log_cfg.get("experiment_name")
        # Optional dashboards. Default empty: the file artifacts are the source
        # of truth, and a published run must not depend on a service being up.
        self.trackers_cfg = log_cfg.get("trackers", []) or []
        self.tracker_options = log_cfg.get("tracker_options", {}) or {}
        # Identifying fields for status.json, so the monitor can label a run
        # without inferring anything from its directory name.
        self.run_meta = log_cfg.get("run_meta", {}) or {}
        # Immutable semantic contract written by scripts/train.py after data
        # loading.  It is embedded into best/native checkpoints so inference
        # cannot silently relabel a checkpoint from a different run.
        self.run_contract = log_cfg.get("run_contract", {}) or {}

        # Resume state, set by fit(resume_from=...). Off by default: a default
        # run must stay byte-identical to the published protocol.
        self.start_epoch = 0
        self.resumed = False

        # State
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.checkpoint_manager = None
        self.logger = None

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def _get_device(self) -> torch.device:
        """Get device from config."""
        training_cfg = self.config.get("training", {})
        device_str = training_cfg.get("device", "cuda")
        requested = torch.device(device_str)
        if requested.type not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported training device: {device_str}")
        if requested.type == "cuda" and torch.cuda.is_available():
            return requested
        elif requested.type == "mps" and torch.backends.mps.is_available():
            return requested
        else:
            # Native estimators do not use torch's training device. Neural
            # setup checks this fallback before any training/log publication.
            return torch.device("cpu")

    def setup(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        initialize_logger: bool = True,
    ) -> None:
        """
        Setup trainer with model and data.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        requested = torch.device(self.config.get("training", {}).get("device", "cuda"))
        if requested.type != self.device.type:
            raise RuntimeError(
                f"Requested training device {requested} is unavailable; refusing "
                "to silently train on CPU under a different run contract. "
                "Set training.device=cpu explicitly for a CPU run."
            )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Detect combo-window models (LSTMComboCA*, GRUComboCA*) — these
        # take (B, L, G, F) input, output (B, G, 2), and need a masked
        # group loss so padded series don't contaminate the gradient.
        self.combo = bool(getattr(model, "requires_combo_loader", False))

        # Create optimizer
        self._create_optimizer()

        # Create scheduler
        self._create_scheduler()

        # Create loss function — masked variant for combo models.
        self.criterion = MaskedMSELoss() if self.combo else MSELoss()

        # Create early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta,
            mode=self.mode,
        )

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=self.save_best_only,
        )

        if initialize_logger:
            self._initialize_logger()

    def _initialize_logger(self) -> None:
        """Publish logs only after a requested resume has been accepted."""
        model = self.model
        train_loader = self.train_loader
        self.logger = TrainingLogger(
            log_dir=self.log_dir,
            experiment_name=self.experiment_name,
            run_meta={
                "model": type(model).__name__,
                "variant": getattr(model, "variant", None),
                "num_params": int(model.get_num_params()),
                "device": str(self.device),
                "epochs_total": self.epochs,
                **self.run_meta,
            },
            trackers=self.trackers_cfg,
            tracker_options=self.tracker_options,
            resumed=self.resumed,
        )
        self.logger.trackers.log_config({
            "model": type(model).__name__,
            "variant": getattr(model, "variant", None),
            "num_params": int(model.get_num_params()),
            "optimizer": self.optimizer_name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler_name,
            "epochs": self.epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "batch_size": getattr(train_loader, "batch_size", None),
            "amp": self.use_amp,
            "grad_clip": self.grad_clip_norm if self.grad_clip else None,
            **self.run_meta,
        })

        self.logger.log_info(f"Trainer initialized on device: {self.device}")
        self.logger.log_info(f"Model parameters: {model.get_num_params():,}")
        self.logger.log_info(f"Model configuration: {self.config.get('model', {})}")
        self.logger.log_info(
            f"Model behavior: attention guards=v2; "
            f"bidirectional={getattr(model, 'bidirectional', False)}; "
            f"positional_dropout={getattr(getattr(getattr(model, 'pos_encoding', None), 'dropout', None), 'p', None)}"
        )
        for module_name, module in model.named_modules():
            version = getattr(module, "architecture_version", None)
            if version:
                self.logger.log_info(f"Architecture {module_name}: {version}")
            # Route numerical guard warnings to this run's logger.
            if module.__class__.__name__ in (
                "CrossAttention", "IdentityCrossAttention", "LatentCrossAttention",
                "MultiHeadAttention", "CrossAttentionModule",
            ):
                module.attention_warning_logger = self.logger.log_warning

    def _param_groups(self):
        """Honor the two per-parameter optimizer conventions the SSMs use.

        Passing ``model.parameters()`` straight to the optimizer silently
        overrides both, which applies weight decay to state-space parameters
        their authors deliberately exempt -- on an SSM that pulls the state
        matrix A toward zero and degrades the dynamics the model is built on.

        * ``_no_weight_decay`` (Mamba / timm convention) -- Mamba-2 marks
          ``A_log``, ``D`` and ``dt_bias``; see external/mamba/.../mamba2.py.
        * ``_optim`` dict (S4 convention) -- ``S4Kernel.register()`` attaches
          ``{"lr": ..., "weight_decay": 0.0}`` to A/B/dt; this mirrors what
          external/s4/train.py::configure_optimizers does for the hydra runs,
          so an S4 trained here matches one trained there. Note C is registered
          with ``wd=None``, so it keeps the global decay on BOTH sides.

        Only what each method's own authors specify is honored. The usual
        AdamW extra -- exempting every 1-D parameter (norm gains, biases) --
        is deliberately NOT applied: no author of these models asked for it,
        so adding it would be us imposing a third convention on architectures
        that already come from different papers with different recipes. The
        RNN/Transformer arms therefore keep exactly the optimizer treatment
        their published runs used, and only the SSM arms change.
        """
        if not self.weight_decay_exempt:
            # Ablation arm: decay everything, reproducing runs recorded before
            # the exemptions existed. Kept because the reported winner (flat
            # Mamba-2, 0.0493) was trained this way, and weight decay on A_log
            # is a shrinkage prior toward a common timescale rather than a bug
            # -- on 144 training months that may genuinely help. Untested
            # either way, so it is a switch, not an assumption.
            return [{"params": [p for p in self.model.parameters()
                                if p.requires_grad]}]

        decay, no_decay, special = [], [], {}
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if hasattr(param, "_optim"):
                special.setdefault(frozenset(param._optim.items()), []).append(param)
            elif getattr(param, "_no_weight_decay", False):
                no_decay.append(param)
            else:
                decay.append(param)

        groups = [{"params": decay}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        for key, params in special.items():
            groups.append({"params": params, **dict(key)})

        if len(groups) > 1:
            def _count(g):
                n = sum(q.numel() for q in g["params"])
                return f"{len(g['params']):>3} tensors / {n:>9,} params"

            print("  Optimizer parameter groups:")
            print(f"    decayed          {_count(groups[0])}"
                  f"   weight_decay={self.weight_decay}")
            idx = 1
            if no_decay:
                print(f"    no weight decay  {_count(groups[idx])}"
                      f"   author-marked (Mamba A_log, D, dt_bias)")
                idx += 1
            for group in groups[idx:]:
                overrides = " ".join(f"{k}={v}" for k, v in group.items()
                                     if k != "params")
                # S4 registers C with lr=None, wd=None -> an empty _optim dict,
                # so it forms its own group that simply inherits the defaults.
                # Kept rather than merged, to mirror the hydra grouping exactly.
                print(f"    special          {_count(group)}   "
                      f"{overrides or 'inherits defaults (S4 C)'}")

        return groups

    def _create_optimizer(self) -> None:
        """Create optimizer."""
        params = self._param_groups()
        # betas/eps apply to the Adam family only; SGD takes neither.
        if self.optimizer_name == "adam":
            self.optimizer = Adam(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = SGD(
                params,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _create_scheduler(self) -> None:
        """Create learning rate scheduler."""
        if self.scheduler_name == "reduce_on_plateau":
            # Read from config, not hardcoded: these were pinned at 0.5/5 while
            # base.yaml carried its own scheduler_factor/scheduler_patience that
            # nothing ever read. They must also track the early-stopping
            # patience -- at patience=50 with a plateau patience of 5 the LR is
            # halved ~10 times (1e-3 -> 1e-6) long before early stopping fires,
            # so the last ~30 epochs of the wait train at an LR of effectively
            # zero and "patience 50" stops being a convergence criterion.
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
        elif self.scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
            )
        elif self.scheduler_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            self.scheduler = None

    def _forward_and_loss(self, batch: Dict[str, Any], detail: bool = False):
        """Run model + loss for a single batch. Branches on self.combo.

        Per-series path: input (B, L, F); model gets state/comm/flow IDs;
        loss is plain MSE on (B, 2) preds vs (B, 2) targets.

        Combo path: input (B, L, G, F); model gets group_mask; loss is
        masked MSE that ignores padded slots so the optimizer doesn't
        chase zeros for missing series. State/comm/flow IDs are not
        present in combo batches and are not used by combo models.

        Args:
            batch: One loader batch.
            detail: Also return the per-channel (value / weight) squared-error
                sums. Reporting only -- the returned ``loss`` is unchanged and
                remains the sole quantity that trains and selects the model.
                Off on the training path so the hot loop is untouched.

        Returns:
            ``loss``, or ``(loss, parts)`` when ``detail`` is set.
        """
        x_numeric = batch["x_numeric"].to(self.device)
        target_value = batch["target_value"].to(self.device)
        target_weight = batch["target_weight"].to(self.device)

        if self.combo:
            group_mask = batch["group_mask"].to(self.device)
            predictions = self.model(x_numeric, group_mask=group_mask)  # (B, G, 2)
            loss = self.criterion(predictions, target_value, target_weight, group_mask)
        else:
            state_ids = batch["state_ids"].to(self.device)
            comm_ids = batch["comm_ids"].to(self.device)
            flow_ids = batch["flow_ids"].to(self.device)
            target = torch.stack([target_value, target_weight], dim=-1)
            predictions = self.model(x_numeric, state_ids, comm_ids, flow_ids)
            loss = self.criterion(predictions, target)

        if not detail:
            return loss

        # Channel split, computed the same way the loss reduces: the combo path
        # counts only unmasked slots, so a padded batch is not diluted by zeros.
        value_se = (predictions[..., 0] - target_value) ** 2
        weight_se = (predictions[..., 1] - target_weight) ** 2
        if self.combo:
            mask = group_mask.to(value_se.dtype)
            parts = {
                "value_se": float((value_se * mask).detach().sum().item()),
                "weight_se": float((weight_se * mask).detach().sum().item()),
                "n": float(mask.detach().sum().item()),
            }
        else:
            parts = {
                "value_se": float(value_se.detach().sum().item()),
                "weight_se": float(weight_se.detach().sum().item()),
                "n": float(value_se.numel()),
            }
        return loss, parts

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_se = 0.0
        target_count = 0.0
        num_batches = 0
        num_samples = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        nonfinite_batches = 0
        batch_loss_min = float("inf")
        batch_loss_max = float("-inf")

        # Peak allocation is per-epoch, not per-run: the interesting number is
        # "did this epoch come close to the card", and a run-lifetime peak set
        # by epoch 1 hides a later climb.
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        epoch_start = time.time()
        for batch_index, batch in enumerate(self.train_loader):
            if self.logger is not None:
                self.logger.heartbeat(phase="train", batch=batch_index,
                                      batches_total=len(self.train_loader))
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, parts = self._forward_and_loss(batch, detail=True)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"non-finite training loss before optimizer step: {loss.item()!r}"
                    )
                self.scaler.scale(loss).backward()
                # Always unscale so the reported norm is in real units; with
                # grad_clip off the "clip" at inf is a no-op that just measures.
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm if self.grad_clip else float("inf"),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, parts = self._forward_and_loss(batch, detail=True)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"non-finite training loss before optimizer step: {loss.item()!r}"
                    )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm if self.grad_clip else float("inf"),
                )
                if not math.isfinite(float(grad_norm)):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"non-finite gradient norm before optimizer step: {float(grad_norm)!r}"
                    )
                self.optimizer.step()

            batch_loss = loss.item()
            if not math.isfinite(batch_loss):
                nonfinite_batches += 1
            else:
                batch_loss_min = min(batch_loss_min, batch_loss)
                batch_loss_max = max(batch_loss_max, batch_loss)
            total_se += parts["value_se"] + parts["weight_se"]
            target_count += parts["n"]
            num_batches += 1
            num_samples += int(batch["x_numeric"].shape[0])
            gn = float(grad_norm)
            if math.isfinite(gn):
                grad_norm_sum += gn
                grad_norm_max = max(grad_norm_max, gn)

        if num_batches == 0 or target_count <= 0:
            raise ValueError("training loader produced no valid targets")

        elapsed = max(time.time() - epoch_start, 1e-9)

        # Picked up by fit() for the per-epoch log line and history.csv.
        self._last_epoch_stats = {
            "grad_norm_mean": grad_norm_sum / max(num_batches, 1),
            "grad_norm_max": grad_norm_max,
            "nonfinite_batches": nonfinite_batches,
            "num_batches": num_batches,
            "num_samples": num_samples,
            "samples_per_s": num_samples / elapsed,
            # Within-epoch spread: a mean that looks healthy while max climbs is
            # the early signature of a diverging batch, and the mean alone hides
            # it until the whole epoch goes bad.
            "batch_loss_min": batch_loss_min if math.isfinite(batch_loss_min) else float("nan"),
            "batch_loss_max": batch_loss_max if math.isfinite(batch_loss_max) else float("nan"),
            "gpu_mem_peak_mb": (
                torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                if self.device.type == "cuda" else 0.0
            ),
            # One entry per optimizer group: the SSM arms deliberately run A/B/dt
            # at their own LR and weight decay, and a single global LR reading
            # cannot show whether that survived a scheduler step.
            "group_lrs": [g["lr"] for g in self.optimizer.param_groups],
        }
        # Exact per-target/channel MSE.  A mean of batch means overweights a
        # short final batch (or a combo batch with fewer valid slots).
        return total_se / (2.0 * target_count)

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model. Returns average validation loss.

        The returned scalar is the batch-mean of the SAME criterion used for
        training, and remains the only quantity early stopping and checkpoint
        selection see. The per-channel split it also records
        (``self._last_val_stats``) is reported, never selected on.
        """
        self.model.eval()
        num_batches = 0
        value_se = weight_se = count = 0.0
        for batch_index, batch in enumerate(self.val_loader):
            if self.logger is not None:
                self.logger.heartbeat(phase="validation", batch=batch_index,
                                      batches_total=len(self.val_loader))
            loss, parts = self._forward_and_loss(batch, detail=True)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite validation loss: {loss.item()!r}")
            num_batches += 1
            value_se += parts["value_se"]
            weight_se += parts["weight_se"]
            count += parts["n"]

        if num_batches == 0 or count <= 0:
            raise ValueError("validation loader produced no valid targets")
        denom = count
        self._last_val_stats = {
            "val_value_mse": value_se / denom,
            "val_weight_mse": weight_se / denom,
            "val_targets": count,
        }
        return (value_se + weight_se) / (2.0 * denom)

    def _fit_sklearn(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Train a sklearn-style model (e.g., XGBoost) that provides its own
        fit()/predict() API and does not use back-propagation.

        Args:
            model: Model with ``requires_training = False``
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Logger only — no optimizer / scheduler / early-stopping needed
        log_cfg = self.config.get("logging", {})
        self.logger = TrainingLogger(
            log_dir=log_cfg.get("log_dir", "outputs/logs"),
            experiment_name=log_cfg.get("experiment_name"),
            run_meta={
                "model": model.__class__.__name__,
                "variant": getattr(model, "variant", None),
                "device": str(self.device),
                "kind": "sklearn",
                # These fit in one call, so the monitor should show them as a
                # single step rather than as a run stuck at epoch 0 of 5000.
                "epochs_total": 1,
                **self.run_meta,
            },
            trackers=self.trackers_cfg,
            tracker_options=self.tracker_options,
        )
        self.logger.log_info(
            f"Trainer initialized for sklearn-style model: {model.__class__.__name__}"
        )

        self.logger.log_info(f"Tree parameters: {getattr(model, 'lgbm_params', getattr(model, 'xgb_params', getattr(model, 'rf_params', {})))}")
        start_time = time.time()

        # Train via sklearn-style API
        model.fit(train_loader, val_loader, verbose=True)

        if hasattr(model, "evaluate_loader"):
            train_loss = float(model.evaluate_loader(train_loader)["mse"])
        else:
            train_preds, train_targets = model.predict_loader(train_loader)
            train_loss = float(np.mean((train_preds - train_targets) ** 2))

        # Validate — prefer model-specific metrics when available (XGBoost
        # reports both MAE and MSE in z-space, matching the paper tables).
        if hasattr(model, "evaluate_loader"):
            val_metrics = model.evaluate_loader(val_loader)
            val_loss = val_metrics["mse"]
        else:
            preds, targets = model.predict_loader(val_loader)
            val_loss = float(np.mean((preds - targets) ** 2))
            val_metrics = {"mse": val_loss}

        val_loss = float(val_loss)
        for split, loss in (("training", train_loss), ("validation", val_loss)):
            if not math.isfinite(loss) or loss < 0:
                raise ValueError(f"Invalid native-model {split} MSE: {loss}")

        total_time = time.time() - start_time
        self.logger.log_info(f"Training completed in {total_time:.2f} seconds")
        self.logger.log_info(f"Validation MSE: {val_loss:.6f}")
        if "mae" in val_metrics:
            self.logger.log_info(
                "Validation MAE: "
                f"{val_metrics['mae']:.6f} "
                f"(value={val_metrics['value_mae']:.6f}, "
                f"weight={val_metrics['weight_mae']:.6f})"
            )

        # Save checkpoint using the model's native save method
        ckpt_cfg = self.config.get("checkpointing", {})
        save_dir = Path(ckpt_cfg.get("save_dir", "outputs/runs"))
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir / "best_xgb"), run_contract=self.run_contract)

        self.history = {
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "lr": [0.0],
        }
        # One "epoch" so these runs land in history.csv / metrics.json with the
        # same schema as the neural arms, and the monitor can render them.
        self.logger.log_epoch(0, {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": 0.0,
            "epoch_time_s": total_time,
            "is_best": 1.0,
            **{k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        })
        self.logger.close(state="finished", summary={
            "epochs_run": 1,
            "best_val": val_loss,
            "best_epoch": 1,
            "stopped_early": False,
            "wall_time_s": round(total_time, 1),
        })
        return self.history

    def _read_resume_checkpoint(self, resume_from) -> Dict[str, Any]:
        """Load a resume checkpoint, or return ``{}`` when not resuming.

        Called before model setup and deferred logger initialization. Logging
        starts only after the checkpoint has been validated and restored.

        ``resume_from=True`` means "continue this run if it left a ``last.pth``,
        otherwise start clean" -- the form a sweep wants, since it makes the
        runner idempotent without the caller testing for the file.
        """
        if not resume_from:
            return {}
        path = self.save_dir / "last.pth" if resume_from is True else Path(resume_from)
        if not path.exists():
            if resume_from is not True:
                raise FileNotFoundError(f"resume checkpoint not found: {path}")
            return {}
        payload = torch.load(path, map_location=self.device, weights_only=False)
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "early_stopping",
            "best_metric",
            "best_epoch",
            "history",
            "epoch_rows",
            "rng_state",
            "resume_contract",
            "run_contract",
        }
        missing = required.difference(payload) if isinstance(payload, dict) else required
        if missing:
            raise ValueError(
                f"{path} is not a resumable checkpoint "
                f"(incomplete; missing: {', '.join(sorted(missing))}). Legacy last.pth and "
                "best.pth files do not contain enough state for a faithful resume."
            )
        optimizer_state = payload["optimizer_state_dict"]
        if (
            not isinstance(optimizer_state, dict)
            or not isinstance(optimizer_state.get("state"), dict)
            or not isinstance(optimizer_state.get("param_groups"), list)
            or not optimizer_state["param_groups"]
        ):
            raise ValueError("resume checkpoint has invalid or missing optimizer state")
        self.resumed = True
        self._resume_path = path
        return payload

    @staticmethod
    def _contract_value(value):
        """Convert configuration values into stable, comparable primitives."""
        if isinstance(value, dict):
            return {
                str(key): Trainer._contract_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [Trainer._contract_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    @staticmethod
    def _loader_contract(loader) -> Dict[str, Any]:
        """Describe the loader properties that change the optimization path."""
        try:
            batches = len(loader)
        except TypeError:
            batches = None
        dataset = getattr(loader, "dataset", None)
        try:
            samples = len(dataset) if dataset is not None else None
        except TypeError:
            samples = None
        return {
            "batch_size": getattr(loader, "batch_size", None),
            "batches": batches,
            "samples": samples,
            "drop_last": getattr(loader, "drop_last", None),
        }

    def _resume_contract(self) -> Dict[str, Any]:
        """Identity and protocol a checkpoint is allowed to continue.

        ``epochs`` is deliberately recorded but not compared below: increasing
        the ceiling is a supported continuation for plateau/step schedules. A
        cosine schedule is tied to its original horizon and is checked exactly.
        """
        training = dict(self.config.get("training", {}) or {})
        epochs_total = training.pop("epochs", self.epochs)
        identity_keys = ("model_key", "variant_key", "seed", "aggregate")
        contract = {
            "version": 1,
            "identity": {key: self.run_meta.get(key) for key in identity_keys},
            "model": {
                "class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "variant": getattr(self.model, "variant", None),
                "num_params": int(self.model.get_num_params()),
                "combo": bool(self.combo),
                "config": self.config.get("model", {}) or {},
            },
            "data": self.config.get("data", {}) or {},
            "training": training,
            "selection": {
                "monitor": self.monitor,
                "mode": self.mode,
            },
            "loaders": {
                "train": self._loader_contract(self.train_loader),
                "val": self._loader_contract(self.val_loader),
            },
            "epochs_total": epochs_total,
        }
        return self._contract_value(contract)

    def _validate_resume_contract(self, saved: Dict[str, Any]) -> None:
        """Refuse state from another benchmark arm or training protocol."""
        current = self._resume_contract()
        saved_comparable = dict(saved)
        current_comparable = dict(current)
        saved_epochs = saved_comparable.pop("epochs_total", None)
        current_epochs = current_comparable.pop("epochs_total", None)

        mismatched = [
            key for key in current_comparable
            if saved_comparable.get(key) != current_comparable[key]
        ]
        mismatched.extend(key for key in saved_comparable if key not in current_comparable)
        if self.scheduler_name == "cosine" and saved_epochs != current_epochs:
            mismatched.append("epochs_total")
        if (type(saved_epochs) is not int or type(current_epochs) is not int
                or current_epochs < saved_epochs):
            mismatched.append("epochs_total")
        if mismatched:
            fields = ", ".join(sorted(set(mismatched)))
            raise ValueError(
                "resume checkpoint benchmark contract mismatch "
                f"({fields}); refusing to mix benchmark arms or protocols"
            )

    def _restore_from_checkpoint(self, payload: Dict[str, Any]):
        """Put optimizer / scheduler / early-stopping back where they were.

        Restoring weights alone is not a resume: the optimizer moments, the LR
        the scheduler had reached, and the early-stopping counter all decide
        what the next epochs do. Dropping them restarts the schedule and can
        train past the point the original run would have stopped.

        Returns:
            ``(start_epoch, best_metric, best_epoch, epoch_rows)``
        """
        if not payload:
            return 0, (float("inf") if self.mode == "min" else float("-inf")), -1, []

        self._validate_resume_contract(payload["resume_contract"])
        saved_run_contract = self._contract_value(payload.get("run_contract"))
        current_run_contract = self._contract_value(self.run_contract)
        # Increasing a non-cosine epoch ceiling is a supported continuation.
        # The resume contract above already enforces every optimization field
        # and separately pins the cosine horizon; ignore only this duplicate
        # provenance field when comparing the embedded full run contract.
        for contract in (saved_run_contract, current_run_contract):
            training = contract.get("training_config") if isinstance(contract, dict) else None
            if isinstance(training, dict):
                training.pop("epochs", None)
        if saved_run_contract != current_run_contract:
            raise ValueError(
                "resume checkpoint run contract mismatch; refusing to resume "
                "against different source data or experiment provenance"
            )
        if self.scheduler is not None and (
            not isinstance(payload.get("scheduler_state_dict"), dict)
            or not payload["scheduler_state_dict"]
        ):
            raise ValueError("resume checkpoint has invalid or missing scheduler state")
        if self.scaler is not None and self.scaler.is_enabled() and (
            not isinstance(payload.get("scaler_state_dict"), dict)
            or not payload["scaler_state_dict"]
        ):
            raise ValueError("resume checkpoint has invalid or missing AMP scaler state")
        self.model.load_state_dict(payload["model_state_dict"])
        if payload.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and payload.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if self.scaler is not None and payload.get("scaler_state_dict"):
            self.scaler.load_state_dict(payload["scaler_state_dict"])

        self._restore_rng_state(payload["rng_state"])

        es = payload.get("early_stopping") or {}
        if es:
            self.early_stopping.counter = es.get("counter", 0)
            self.early_stopping.best_score = es.get("best_score")
            self.early_stopping.early_stop = es.get("early_stop", False)

        best_metric = payload.get(
            "best_metric", float("inf") if self.mode == "min" else float("-inf")
        )
        best_epoch = payload.get("best_epoch", -1)
        # The checkpoint records the epoch it was taken at; the next one to run
        # is the one after it.
        start_epoch = int(payload.get("epoch", -1)) + 1
        epoch_rows = list(payload.get("epoch_rows") or [])

        # CheckpointManager compares against its own best or it would overwrite
        # a better best.pth with the first epoch of the continuation.
        self.checkpoint_manager.best_score = best_metric
        for key in ("train_loss", "val_loss", "lr"):
            self.history[key] = list(payload.get("history", {}).get(key, []))

        return start_epoch, best_metric, best_epoch, epoch_rows

    def _capture_rng_state(self) -> Dict[str, Any]:
        """Capture every random stream that can affect the next epoch."""
        loader_generator = getattr(getattr(self, "train_loader", None), "generator", None)
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "train_loader": (
                loader_generator.get_state() if loader_generator is not None else None
            ),
            "torch_cuda": None,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        """Restore the random streams captured by :meth:`_capture_rng_state`."""
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())

        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state.cpu(), self.device)

        loader_state = state.get("train_loader")
        loader_generator = getattr(getattr(self, "train_loader", None), "generator", None)
        if loader_state is not None:
            if loader_generator is None:
                raise ValueError(
                    "resume checkpoint contains DataLoader RNG state, but the current "
                    "training loader has no generator"
                )
            loader_generator.set_state(loader_state.cpu())

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from=None,
        on_ready=None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            resume_from: ``None`` to train from scratch (default), ``True`` to
                continue from ``<save_dir>/last.pth`` when it exists, or an
                explicit checkpoint path.

        Returns:
            Training history
        """
        # Sklearn-style branch (e.g. XGBoost). These models expose no optimizer
        # or incremental-training state, so never pretend an explicit resume was
        # honored. The optional --resume form may still start clean when there
        # is no checkpoint, as documented by the CLI.
        if getattr(model, "requires_training", True) is False:
            if resume_from:
                path = self.save_dir / "last.pth" if resume_from is True else Path(resume_from)
                if resume_from is not True and not path.exists():
                    raise FileNotFoundError(f"resume checkpoint not found: {path}")
                if path.exists():
                    raise ValueError(
                        f"resume is not supported for sklearn-style model "
                        f"{model.__class__.__name__}: {path}"
                    )
            if on_ready is not None:
                on_ready()
            return self._fit_sklearn(model, train_loader, val_loader)

        # Read the checkpoint before setup; defer all logging until restoration
        # succeeds so rejected resumes cannot overwrite existing run metadata.
        resume_payload = self._read_resume_checkpoint(resume_from)

        # Setup
        self.setup(model, train_loader, val_loader, initialize_logger=False)

        start_epoch, best_metric, best_epoch, epoch_rows = self._restore_from_checkpoint(
            resume_payload
        )
        if resume_payload:
            # last.pth contains the final weights, which need not be the selected
            # weights. Validate its sibling before publishing any resumed logs.
            source_best = Path(self._resume_path).parent / "best.pth"
            selected = torch.load(source_best, map_location="cpu", weights_only=False)
            selected_contract = self._contract_value(selected.get("run_contract"))
            resume_contract = self._contract_value(resume_payload["run_contract"])
            if self.scheduler_name != "cosine":
                old_training = selected_contract.get("training_config", {}) if isinstance(selected_contract, dict) else {}
                saved_training = resume_contract.get("training_config", {})
                old_limit = old_training.pop("epochs", None)
                saved_limit = saved_training.pop("epochs", None)
                if old_limit != saved_limit and (
                    type(old_limit) is not int or type(saved_limit) is not int or old_limit > saved_limit
                ):
                    raise ValueError("selected checkpoint has incompatible resume epoch budget")
            if (selected.get("epoch") != best_epoch
                    or selected.get("score") != best_metric
                    or selected_contract != resume_contract):
                raise ValueError("selected checkpoint does not match resume source")
            if source_best.resolve() != (self.save_dir / "best.pth").resolve():
                atomic_torch_save(selected, self.save_dir / "best.pth")
        # A rejected resume must leave the old status/history/logs untouched.
        # Tracker initialization may use random numbers: preserve the restored
        # training streams around it so continuation stays reproducible.
        restored_rng = self._capture_rng_state() if resume_payload else None
        self._initialize_logger()
        if restored_rng is not None:
            self._restore_rng_state(restored_rng)
            self.logger.log_info(
                f"resumed   : {getattr(self, '_resume_path', '?')} at epoch {start_epoch} "
                f"| best val {best_metric:.6f} @ epoch {best_epoch + 1} "
                f"| early-stop counter {self.early_stopping.counter}/{self.patience}"
            )
        # Publish sidecar provenance only after checkpoint compatibility and
        # optimizer/RNG restoration succeed. A rejected resume must not relabel
        # an existing checkpoint with a different data/configuration snapshot.
        if on_ready is not None:
            on_ready()
        # Invalidate prior completion only after resume compatibility succeeds.
        completion_path(self.save_dir / "best.pth").unlink(missing_ok=True)
        self.start_epoch = start_epoch
        rows_before = len(epoch_rows)      # rows carried in from a resume

        # Training loop
        start_time = time.time()
        self._log_run_header(model, train_loader, val_loader)
        stopped_early = False

        if self.early_stopping.early_stop:
            stopped_early = True
            self.logger.log_info(
                "checkpoint had already reached early stopping; no additional epochs run"
            )

        epoch_range = () if stopped_early else range(start_epoch, self.epochs)
        for epoch in epoch_range:
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            if self.scheduler_name == "reduce_on_plateau":
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            stats = getattr(self, "_last_epoch_stats", {})
            improved = val_loss < best_metric
            delta = val_loss - best_metric if best_epoch >= 0 else float("nan")
            epoch_time = time.time() - epoch_start
            eta = epoch_time * (self.epochs - epoch - 1)

            # Save checkpoint
            if improved:
                best_metric = val_loss
                best_epoch = epoch
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    run_contract=self.run_contract,
                )

            # Update early stopping before logging so every artifact reports the
            # counter produced by this epoch rather than the previous one.
            stop_now = self.early_stopping(val_loss)

            # last.pth every epoch when configured. The old wiring never wrote
            # it: CheckpointManager only ran on improvement and its save_last
            # branch was gated behind `not save_best_only`, so base.yaml's
            # `save_last: true` had no effect.
            #
            # The payload carries everything a resume needs, not just weights:
            # optimizer moments, where the LR schedule had got to, the AMP
            # scaler, the early-stopping counter, the best-so-far, and the
            # epoch rows. Restoring weights alone silently restarts the
            # schedule and the patience window.
            # One dense line per epoch: losses, whether it beat the best and by
            # how much, lr, gradient scale, throughput, timing and ETA. Enough
            # to diagnose divergence, plateaus, LR-schedule behaviour and a
            # slow-down from the log alone.
            val_stats = getattr(self, "_last_val_stats", {})
            self.logger.log_info(
                f"epoch {epoch + 1:>3}/{self.epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} "
                f"{'*best' if improved else ' ' * 5} "
                f"{'' if best_epoch < 0 or epoch == best_epoch else f'({delta:+.6f})':>13} | "
                f"lr {lr:.3e} | "
                f"grad {stats.get('grad_norm_mean', float('nan')):.3f} "
                f"(max {stats.get('grad_norm_max', float('nan')):.3f}) | "
                f"{stats.get('num_batches', 0)} batches | "
                f"{stats.get('samples_per_s', float('nan')):.0f} smp/s | "
                f"mem {stats.get('gpu_mem_peak_mb', 0.0):.0f}MB | "
                f"es {self.early_stopping.counter}/{self.patience} | "
                f"{epoch_time:.1f}s | eta {_fmt_duration(eta)}"
            )
            if stats.get("nonfinite_batches"):
                self.logger.log_warning(
                    f"epoch {epoch + 1}: {stats['nonfinite_batches']} batch(es) "
                    f"produced a non-finite loss"
                )

            # Structured record -> history.csv + metrics.json, flushed now.
            record = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "epoch_time_s": epoch_time,
                "grad_norm_mean": stats.get("grad_norm_mean", float("nan")),
                "grad_norm_max": stats.get("grad_norm_max", float("nan")),
                "batch_loss_min": stats.get("batch_loss_min", float("nan")),
                "batch_loss_max": stats.get("batch_loss_max", float("nan")),
                "nonfinite_batches": float(stats.get("nonfinite_batches", 0)),
                "num_batches": float(stats.get("num_batches", 0)),
                "samples_per_s": stats.get("samples_per_s", float("nan")),
                "gpu_mem_peak_mb": stats.get("gpu_mem_peak_mb", 0.0),
                "early_stop_counter": float(self.early_stopping.counter),
                "is_best": float(improved),
                # Reported, never selected on: the driving loss stays val_loss.
                "val_value_mse": val_stats.get("val_value_mse", float("nan")),
                "val_weight_mse": val_stats.get("val_weight_mse", float("nan")),
            }
            for i, group_lr in enumerate(stats.get("group_lrs", [])):
                record[f"lr_group{i}"] = group_lr
            self.logger.log_epoch(epoch, record)

            epoch_rows.append({"epoch": epoch + 1, **record})

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            # last.pth every epoch when configured. The old wiring never wrote
            # it: CheckpointManager only ran on improvement and its save_last
            # branch was gated behind `not save_best_only`, so base.yaml's
            # `save_last: true` had no effect.
            #
            # The payload carries everything a resume needs, not just weights:
            # optimizer moments, where the LR schedule had got to, the AMP
            # scaler, the early-stopping counter, the best-so-far and the epoch
            # rows. Restoring weights alone silently restarts both the schedule
            # and the patience window.
            if self.save_last:
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": (
                            self.scheduler.state_dict() if self.scheduler is not None else None
                        ),
                        "scaler_state_dict": (
                            self.scaler.state_dict() if self.scaler is not None else None
                        ),
                        "early_stopping": {
                            "counter": self.early_stopping.counter,
                            "best_score": self.early_stopping.best_score,
                            "early_stop": self.early_stopping.early_stop,
                        },
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "history": self.history,
                        "epoch_rows": epoch_rows,
                        "rng_state": self._capture_rng_state(),
                        "resume_contract": self._resume_contract(),
                        "run_contract": self.run_contract,
                        "score": val_loss,
                    },
                    self.save_dir / "last.pth",
                )

            # Heartbeat for scripts/train_status.py. Written every epoch so a
            # stalled run is visible as a stale timestamp rather than as a log
            # that quietly stopped growing.
            self.logger.update_status(
                epoch=epoch + 1,
                epochs_total=self.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val=best_metric,
                best_epoch=best_epoch + 1,
                lr=lr,
                epoch_time_s=round(epoch_time, 3),
                eta_s=round(eta, 1),
                samples_per_s=round(stats.get("samples_per_s", float("nan")), 1),
                gpu_mem_peak_mb=round(stats.get("gpu_mem_peak_mb", 0.0), 1),
                grad_norm_mean=stats.get("grad_norm_mean", float("nan")),
                nonfinite_batches=stats.get("nonfinite_batches", 0),
                early_stop_counter=self.early_stopping.counter,
                early_stop_patience=self.patience,
            )

            if stop_now:
                stopped_early = True
                self.logger.log_info(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement > {self.min_delta} for {self.patience} epochs)"
                )
                break

        # Training complete
        total_time = time.time() - start_time
        epochs_total = len(epoch_rows)          # including any resumed from disk
        epochs_this_run = epochs_total - rows_before
        if resume_payload:
            self._rebind_selected_checkpoint_after_resume(best_epoch, best_metric)
        write_completion(
            self.save_dir / "best.pth", self.run_contract,
            epochs_run=epochs_total, best_epoch=best_epoch + 1,
            best_val=best_metric, stopped_early=stopped_early,
            patience_counter=self.early_stopping.counter,
        )
        files = {"best": "best.pth"}
        if self.save_last:
            if resume_payload and not epochs_this_run:
                atomic_torch_save({**resume_payload, "run_contract": self.run_contract,
                                   "resume_contract": self._resume_contract()},
                                  self.save_dir / "last.pth")
            files["resume"] = "last.pth"
        write_checkpoint_manifest(self.save_dir, files, kind="native_neural",
                                  run_contract=self.run_contract)
        self.logger.log_info("-" * 78)
        self.logger.log_info(
            f"finished  : {epochs_total}/{self.epochs} epochs in "
            f"{_fmt_duration(total_time)}"
            f"{f' ({epochs_this_run} this session)' if rows_before else ''}"
            f"{' (early stop)' if stopped_early else ''}"
        )
        self.logger.log_info(
            f"best      : val {best_metric:.6f} at epoch {best_epoch + 1}"
        )
        if epochs_this_run:
            self.logger.log_info(
                f"final     : train {epoch_rows[-1]['train_loss']:.6f} | "
                f"val {epoch_rows[-1]['val_loss']:.6f}"
            )
            self.logger.log_info(
                f"mean epoch: {total_time / epochs_this_run:.1f}s"
            )
        self.logger.log_info(f"history   : {self.logger.history_path}")
        self.logger.log_info("=" * 78)

        self.logger.close(state="finished", summary={
            "epochs_run": epochs_total,
            "epochs_this_session": epochs_this_run,
            "best_val": best_metric,
            "best_epoch": best_epoch + 1,
            "stopped_early": stopped_early,
            "wall_time_s": round(total_time, 1),
        })

        return self.history

    def _rebind_selected_checkpoint_after_resume(self, best_epoch, best_metric):
        """Carry an unchanged selected model into an accepted longer run.

        Only the non-cosine epoch ceiling may differ. Preserve the original
        selection contract explicitly, and never replace its model tensors
        with the final (possibly worse) training weights.
        """
        path = self.save_dir / "best.pth"
        selected = torch.load(path, map_location="cpu", weights_only=False)
        if selected.get("epoch") != best_epoch or selected.get("score") != best_metric:
            raise ValueError("selected checkpoint does not match resumed best epoch/score")
        old = selected.get("run_contract")
        if old == self.run_contract:
            return
        old_comparable = self._contract_value(old)
        new_comparable = self._contract_value(self.run_contract)
        old_training = old_comparable.get("training_config") if isinstance(old_comparable, dict) else None
        new_training = new_comparable.get("training_config") if isinstance(new_comparable, dict) else None
        if not isinstance(old_training, dict) or not isinstance(new_training, dict):
            raise ValueError("selected checkpoint has incompatible resume provenance")
        old_limit = old_training.pop("epochs", None)
        new_limit = new_training.pop("epochs", None)
        if (self.scheduler_name == "cosine" or old_comparable != new_comparable
                or type(old_limit) is not int or type(new_limit) is not int
                or new_limit < old_limit):
            raise ValueError("selected checkpoint has incompatible resume provenance")
        selected.setdefault("selection_origin_run_contract", old)
        selected["run_contract"] = self.run_contract
        atomic_torch_save(selected, path)

    def _log_run_header(self, model, train_loader, val_loader) -> None:
        """Log the full run configuration before the first epoch.

        Every number that shapes the run is written once, so a log file is
        self-describing: reading it later tells you what was trained, on what,
        with which hyperparameters -- without needing the config tree.
        """
        log = self.logger.log_info
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log("=" * 78)
        log(f"model     : {type(model).__name__}"
            f"  variant={getattr(model, 'variant', '?')}")
        log(f"params    : {total:,} total | {trainable:,} trainable")
        for name, value in (
            ("d_model/hidden", getattr(model, "d_model", None)
             or getattr(model, "hidden_size", None)),
            ("scan_schedule", getattr(getattr(model, "grid", None),
                                      "scan_schedule", None)),
            ("scan layers", len(getattr(getattr(model, "grid", None),
                                        "layer_pos", []) or []) or None),
            ("depth_policy", getattr(model, "depth_policy", None)),
            ("group encoder", type(getattr(model, "group_encoder", None)).__name__
             if getattr(model, "group_encoder", None) is not None else None),
            ("axial input dim", getattr(getattr(model, "axial", None),
                                        "in_proj", None) is not None
             and model.axial.in_proj.in_features or None),
        ):
            if value is not None:
                log(f"{name:<10}: {value}")
        log(f"device    : {self.device}  amp={self.use_amp}  "
            f"grad_clip={self.grad_clip_norm if self.grad_clip else 'off'}")
        log(f"optimizer : {self.optimizer_name}  lr={self.lr}  "
            f"weight_decay={self.weight_decay}  scheduler={self.scheduler_name}")
        log(f"schedule  : {self.epochs} epochs max  "
            f"early_stop(patience={self.patience}, min_delta={self.min_delta}, "
            f"monitor={self.monitor}/{self.mode})")
        try:
            log(f"data      : {len(train_loader)} train batches | "
                f"{len(val_loader)} val batches | "
                f"batch_size={getattr(train_loader, 'batch_size', '?')}")
        except TypeError:                      # loader without __len__
            log(f"data      : batch_size={getattr(train_loader, 'batch_size', '?')}")
        log("-" * 78)

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        atomic_torch_save(checkpoint, path)
        self.logger.log_info(f"Checkpoint saved to {path}")
