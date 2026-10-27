"""Canonical Experiment 2 membership and exact training contracts."""
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .config import get_data_config, load_config
from .experiment_manifest import load_manifest, exp1_source_domain
from .run_contract import build_run_contract


@dataclass(frozen=True)
class Exp2Run:
    model: str
    variant: str
    seed: int
    kind: str = "native"
    experiment: str = ""

    @property
    def directory_name(self):
        arm = f"s4_{self.experiment}" if self.kind == "s4" else f"{self.model}_{self.variant}"
        return f"{arm}_seed{self.seed}"


def expected_runs(manifest, seeds=None):
    seeds = tuple(map(int, manifest["PAPER_SEEDS"].split())) if seeds is None else tuple(seeds)
    runs = []
    for seed in seeds:
        for model in manifest["EXP2_BASIC_MODELS"].split():
            runs.extend(Exp2Run(model, v, seed) for v in manifest["BASIC_VARIANTS"].split())
        for model in manifest["EXP2_COMBO_MODELS"].split():
            runs.extend(Exp2Run(model, v, seed) for v in ("cross_attention", "film_attention"))
        for model in manifest["EXP2_AXIAL_MODELS"].split():
            runs.extend(Exp2Run(model, v, seed) for v in manifest["EXP2_AXIAL_VARIANTS"].split())
        runs.extend(Exp2Run(model, "none", seed, "baseline") for model in manifest["BASELINES"].split())
        for experiment in manifest["EXP2_S4_EXPERIMENTS"].split():
            from .s4_checkpoint_provenance import s4_experiment_identity
            model, variant = s4_experiment_identity(experiment)
            runs.append(Exp2Run(model, variant, seed, "s4", experiment))
    if len({r.directory_name for r in runs}) != len(runs):
        raise ValueError("Duplicate Experiment 2 run directories")
    return tuple(runs)


def s4_overrides(repo):
    config = load_config(str(Path(repo) / "config/base.yaml"))
    t = config["training"]
    data = get_data_config(config)
    if (data["input_len"], data["lag_count"], data.get("train_target_start")) != (36, 12, None):
        raise ValueError("Experiment 2 requires window=36, lags=12, and no shortened training calendar")
    if not data.get("cohort_from_train_only") or not data.get("tabular_align_targets"):
        raise ValueError("Experiment 2 requires train-only cohort and matched tree target dates")
    if t["optimizer"] != "adamw" or t["scheduler"] != "reduce_on_plateau" or t.get("use_amp"):
        raise ValueError("Experiment 2 currently requires AdamW, plateau scheduling and fp32")
    if not t.get("weight_decay_exempt"):
        raise ValueError("Experiment 2 S4 recipe requires explicit SSM decay exemptions")
    state = load_config(str(Path(repo) / "config/models/s4.yaml"))["model"]["d_state"]
    return [
        f"trainer.max_epochs={t['epochs']}",
        "+callbacks.early_stopping.monitor=val/loss", "+callbacks.early_stopping.mode=min",
        f"+callbacks.early_stopping.patience={t['patience']}",
        f"+callbacks.early_stopping.min_delta={t['min_delta']}",
        f"scheduler.patience={t['scheduler_patience']}", f"scheduler.factor={t['scheduler_factor']}",
        f"optimizer.lr={t['lr']}", f"optimizer.weight_decay={t['weight_decay']}",
        f"+trainer.gradient_clip_val={t['grad_clip_norm'] if t['grad_clip'] else 0.0}",
        "+trainer.deterministic=warn", "trainer.val_check_interval=1.0",
        "trainer.precision=32", "train.test=false", "train.optimizer_param_grouping=null",
        # Inherited kernel LR=1e-3 otherwise silently overrides shared 1e-4.
        "model.layer.lr=null", f"model.layer.d_state={state}",
    ]


def resolved_s4_config(repo, run):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path(repo).resolve() / "external/s4/configs"), version_base=None):
        config = compose(config_name="config", overrides=[
            f"experiment={run.experiment}", f"train.seed={run.seed}",
            "wandb=null", "loader.num_workers=0", *s4_overrides(repo),
        ])
    return OmegaConf.to_container(config, resolve=True)


def expected_contract(repo, run, manifest=None):
    repo = Path(repo)
    manifest = manifest or load_manifest(repo / "config/experiment_manifest.env")
    domain = exp1_source_domain(manifest)
    cfg = load_config(str(repo / "config/base.yaml"))
    if run.kind == "s4":
        from src.data.dataloader import DataConfig
        from .s4_checkpoint_provenance import s4_run_contract
        from .provenance import CORRECTED_NUM_COMBOS
        data = get_data_config(cfg)
        data["aggregate"] = False
        pipe = SimpleNamespace(config=DataConfig(**data), kept_combos=[()] * CORRECTED_NUM_COMBOS,
                               source_fingerprint=manifest["PAPER_SOURCE_FINGERPRINT"], source_domain=domain)
        return s4_run_contract(resolved_s4_config(repo, run), SimpleNamespace(pipe=pipe))
    for path in (repo / f"config/models/{run.model}.yaml", repo / f"config/variants/{run.variant}.yaml"):
        if path.is_file():
            cfg = load_config(str(path), cfg)
    if (run.variant.startswith(("cross_attention_", "matched_axial_"))
            or (run.model in ("lstm", "gru") and run.variant in ("cross_attention", "film_attention"))):
        cfg["training"]["batch_size"] = 1
    if run.model in ("xgboost", "lightgbm", "random_forest"):
        cfg["model"]["random_state"] = run.seed
    if run.kind == "baseline":
        cfg["model"] = {"name": run.model, "variant": "none", "window":
                        int(manifest["PAPER_MOVING_AVERAGE_WINDOW"]) if run.model == "moving_average" else None}
    return build_run_contract(cfg, model=run.model, variant=run.variant, seed=run.seed,
                              aggregate=False, source_fingerprint=manifest["PAPER_SOURCE_FINGERPRINT"],
                              source_domain=domain)


def comparable_s4_contract(contract):
    # external/s4/train.py stamps each callback config with `_name_` equal to
    # its own key before the checkpoint contract is captured; a freshly
    # composed config has no such key. Only that self-echo is dropped: a
    # `_name_` naming a different registry entry selects a different class.
    if not isinstance(contract, dict):
        return contract
    callbacks = (contract.get("training_config") or {}).get("callbacks")
    if not isinstance(callbacks, dict):
        return contract
    contract = copy.deepcopy(contract)
    for key, callback in contract["training_config"]["callbacks"].items():
        if isinstance(callback, dict) and callback.get("_name_") == key:
            del callback["_name_"]
    return contract


def validate_contract(payload, repo, roster=None):
    # roster(manifest, seeds) -> runs; the linear-head rerun passes its own.
    roster = roster or expected_runs
    manifest = load_manifest(Path(repo) / "config/experiment_manifest.env")
    candidates = [r for r in roster(manifest, [payload.get("seed")])
                  if (r.model, r.variant) == (payload.get("model"), payload.get("variant"))]
    if len(candidates) != 1:
        return False, "not an Experiment 2 arm"
    expected = expected_contract(repo, candidates[0], manifest)
    if candidates[0].kind == "s4":
        actual, expected = comparable_s4_contract(payload.get("run_contract")), comparable_s4_contract(expected)
    else:
        actual = payload.get("run_contract")
    if actual != expected:
        return False, "Experiment 2 model/data/training/selection contract differs from canonical recipe"
    return True, "current"
