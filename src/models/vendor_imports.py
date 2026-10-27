"""Scoped import setup for the vendored Mamba runtime."""
import importlib
from pathlib import Path
import sys
import types


def load_mamba2():
    """Use the intended fallback kernels without poisoning other imports.

    Kernel choices are captured by the imported Mamba modules. Temporary
    module overrides and the search path are restored even if Triton is absent.
    """
    vendor = str(Path(__file__).resolve().parents[2] / "external/mamba")
    missing = object()
    names = ("selective_scan_cuda", "causal_conv1d")
    previous = {name: sys.modules.get(name, missing) for name in names}
    previous_mamba = {name for name in sys.modules
                      if name == "mamba_ssm" or name.startswith("mamba_ssm.")}
    inserted = vendor not in sys.path
    if inserted:
        sys.path.insert(0, vendor)
    try:
        sys.modules.setdefault("selective_scan_cuda", types.ModuleType("selective_scan_cuda"))
        sys.modules["causal_conv1d"] = None
        module = importlib.import_module("mamba_ssm")
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(Path(vendor).resolve()):
            raise ImportError("Mamba must come from the vendored external/mamba source")
        return module.Mamba2
    except BaseException:
        # Python removes the failing module itself, but successfully imported
        # children may survive with references to our temporary kernel stubs.
        # Do not let a retry reuse that partial import graph.
        for name in list(sys.modules):
            if (name == "mamba_ssm" or name.startswith("mamba_ssm.")) and name not in previous_mamba:
                sys.modules.pop(name, None)
        raise
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if inserted:
            sys.path.remove(vendor)
