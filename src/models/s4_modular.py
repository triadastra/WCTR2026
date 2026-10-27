"""Load the authors' modular S4 without rewriting its absolute src imports.

Both projects own a package called src. Only construction needs the upstream
namespace; the returned objects retain the authors' actual forward methods.
Training checkpoints use state_dicts, not pickled Python model objects.
"""
from contextlib import contextmanager
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import threading
import types

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "external" / "s4"
IMPLEMENTATION = "authors-modular-s4-v1"
_IMPORT_LOCK = threading.RLock()
_VENDOR_MODULES = {}


def _src_modules():
    return {name: module for name, module in sys.modules.copy().items()
            if name == "src" or name.startswith("src.")}


@contextmanager
def modular_s4_imports():
    """Restore the benchmark namespace even if upstream construction fails."""
    with _IMPORT_LOCK:
        previous = _src_modules()
        for name in previous:
            del sys.modules[name]
        if _VENDOR_MODULES:
            sys.modules.update(_VENDOR_MODULES)
        else:
            package = types.ModuleType("src")
            package.__path__ = [str(UPSTREAM_ROOT / "src")]
            package.__package__ = "src"
            package.__spec__ = ModuleSpec("src", loader=None, is_package=True)
            package.__spec__.submodule_search_locations = package.__path__
            sys.modules["src"] = package
        try:
            yield
        except BaseException:
            _VENDOR_MODULES.clear()
            raise
        else:
            _VENDOR_MODULES.update(_src_modules())
        finally:
            for name in _src_modules():
                del sys.modules[name]
            sys.modules.update(previous)


def backbone_config(*, d_model=128, n_layers=4, d_state=64, dropout=0.1,
                    n_ssm=1, dt_transform="softplus", tie_dropout=True,
                    bidirectional=False):
    """Resolve the same Hydra model recipe used by external flat-S4 runs."""
    with initialize_config_dir(config_dir=str(UPSTREAM_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "experiment=s4_embeddings", "wandb=null",
            f"model.d_model={d_model}", f"model.n_layers={n_layers}",
            f"model.dropout={dropout}", f"model.layer.d_state={d_state}",
            f"model.layer.n_ssm={n_ssm}", f"model.layer.dt_transform={dt_transform}",
            f"model.tie_dropout={str(tie_dropout).lower()}",
            f"model.layer.bidirectional={str(bidirectional).lower()}",
            "model.layer.lr=null",
        ])
    model = OmegaConf.to_container(cfg.model, resolve=True)
    model.pop("encoder", None)
    model.pop("decoder", None)
    return model


def create_modular_s4(*, implementation=IMPLEMENTATION, **kwargs):
    if implementation != IMPLEMENTATION:
        raise ValueError(f"Unsupported S4 implementation: {implementation!r}")
    config = backbone_config(**kwargs)
    # Lightning/torchmetrics inspect loaded modules during their first import.
    # Load them before temporarily installing the vendor's src namespace.
    importlib.import_module("pytorch_lightning")
    with modular_s4_imports():
        utils = importlib.import_module("src.utils")
        registry = importlib.import_module("src.utils.registry")
        model = utils.instantiate(registry.model, config)
    model.architecture_version = IMPLEMENTATION
    return model
