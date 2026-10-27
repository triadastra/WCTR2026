"""Probe the real launch imports in isolated interpreters before any training."""
from pathlib import Path
import subprocess
import sys


def check_runtime(repo, *, include_exp2=True):
    repo = Path(repo).resolve()
    modules = ["torch", "src.models.lstm", "src.models.transformer", "src.models.s4"]
    if include_exp2:
        modules += ["pytorch_lightning", "hydra", "xgboost", "lightgbm", "wandb",
                    "triton", "src.models.mamba", "src.models.mamba_nd"]
    probes = [(name, repo, "import importlib,sys; importlib.import_module(sys.argv[1])", [name])
              for name in modules]
    probes.append(("S4 source provenance", repo,
                   "import runpy; runpy.run_path('scripts/check_s4_upstream.py', run_name='__main__')", []))
    # Native S4 now constructs the authors' modular backbone lazily. Importing
    # our adapter alone does not exercise its Lightning/Hydra/vendor imports.
    probes.append(("modular S4 construction", repo,
                   "from src.models.s4_modular import create_modular_s4; "
                   "create_modular_s4(d_model=8, n_layers=1, d_state=4); "
                   "create_modular_s4(d_model=8, n_layers=1, d_state=4, bidirectional=True)", []))
    if include_exp2:
        probes.append(("upstream_s4_train", repo / "external/s4", "import train", []))
        # A successful import is insufficient: Lightning 1.9 imports but rejects
        # the recipes' strategy='auto'. Construct without data, loggers or fitting.
        probes.append(("Lightning trainer configuration", repo / "external/s4",
                       "import train; train.pl.Trainer(accelerator='cpu', devices=1, "
                       "strategy='auto', logger=False, enable_checkpointing=False, "
                       "enable_model_summary=False, enable_progress_bar=False)", []))
    probes.append(("CUDA execution", repo,
                   "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                   "x=torch.ones(2,device='cuda'); assert x.sum().item()==2; torch.cuda.synchronize()", []))
    results, errors = {}, []
    for name, cwd, code, args in probes:
        try:
            result = subprocess.run([sys.executable, "-c", code, *args], cwd=cwd,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()[-2000:]
                results[name] = {"passed": False, "detail": detail}
                errors.append(f"{name}: {detail.splitlines()[-1]}")
            else:
                results[name] = {"passed": True}
        except (OSError, subprocess.TimeoutExpired) as exc:
            results[name] = {"passed": False, "detail": str(exc)}
            errors.append(f"{name}: {exc}")
    return {"passed": not errors, "probes": results, "errors": errors}
