import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models import vendor_imports
from src.models.s4_modular import modular_s4_imports


@pytest.mark.parametrize("fail", [False, True])
def test_mamba_kernel_overrides_and_path_are_restored(monkeypatch, fail):
    original_conv = SimpleNamespace()
    original_scan = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "causal_conv1d", original_conv)
    monkeypatch.setitem(sys.modules, "selective_scan_cuda", original_scan)
    before = list(sys.path)
    sentinel = object()
    origin = Path(vendor_imports.__file__).resolve().parents[2] / "external/mamba/mamba_ssm/__init__.py"

    def fake_import(name):
        assert name == "mamba_ssm"
        assert sys.modules["causal_conv1d"] is None
        if fail:
            raise ImportError("test missing Triton")
        return SimpleNamespace(Mamba2=sentinel, __file__=str(origin))

    monkeypatch.setattr(vendor_imports.importlib, "import_module", fake_import)
    if fail:
        with pytest.raises(ImportError, match="Triton"):
            vendor_imports.load_mamba2()
    else:
        assert vendor_imports.load_mamba2() is sentinel
    assert sys.path == before
    assert sys.modules["causal_conv1d"] is original_conv
    assert sys.modules["selective_scan_cuda"] is original_scan


def test_mamba_rejects_cached_nonvendored_package(monkeypatch):
    monkeypatch.setattr(vendor_imports.importlib, "import_module", lambda name:
                        SimpleNamespace(__file__="/tmp/unrelated/mamba_ssm/__init__.py"))
    with pytest.raises(ImportError, match="vendored"):
        vendor_imports.load_mamba2()


def test_failed_mamba_import_discards_only_new_partial_modules(monkeypatch):
    retained_name = "mamba_ssm.test_existing_child"
    new_name = "mamba_ssm.test_partial_child"
    retained = SimpleNamespace()
    monkeypatch.setitem(sys.modules, retained_name, retained)
    monkeypatch.delitem(sys.modules, new_name, raising=False)

    def fake_import(name):
        sys.modules[new_name] = SimpleNamespace()
        raise ImportError("missing runtime")

    monkeypatch.setattr(vendor_imports.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match="missing runtime"):
        vendor_imports.load_mamba2()
    assert new_name not in sys.modules
    assert sys.modules[retained_name] is retained


def test_modular_s4_logging_is_rank_zero_only(monkeypatch):
    with modular_s4_imports():
        rank_zero_only = importlib.import_module("src.utils.train").rank_zero_only
    calls = []
    log = rank_zero_only(lambda: calls.append("logged"))
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    log()
    assert not calls
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    log()
    assert calls == ["logged"]
