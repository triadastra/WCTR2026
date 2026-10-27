"""Require terminal, checkpoint-bound evidence for every trained Exp2 arm."""
from pathlib import Path
from .exp1_completion import BASELINES, validate_exp1_completion_artifact
from .s4_completion import read_s4_completion


def validate_exp2_completion_artifact(payload, metrics_path):
    if payload["model"] in BASELINES:
        return
    contract = payload["run_contract"]
    if "loss_reduction" not in contract["training_config"]:
        return validate_exp1_completion_artifact(payload, metrics_path)
    checkpoint = Path(metrics_path).parent / "hydra/checkpoints/val/loss.ckpt"
    receipt = read_s4_completion(checkpoint, contract)
    if receipt != payload.get("training_completion"):
        raise ValueError("Exp2 S4 metrics completion evidence differs from selected checkpoint")
