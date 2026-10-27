import subprocess
from types import SimpleNamespace

from src.utils.runtime_readiness import check_runtime


def test_broken_import_is_not_mistaken_for_installed_runtime(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        bad = command[-1] == "pytorch_lightning"
        return SimpleNamespace(returncode=int(bad), stdout="",
                               stderr="ModuleNotFoundError: No module named 'torchmetrics'" if bad else "")
    monkeypatch.setattr(subprocess, "run", run)
    result = check_runtime(tmp_path)
    assert not result["passed"]
    assert "torchmetrics" in result["errors"][0]
    assert any(str(kwargs["cwd"]).endswith("external/s4") for _, kwargs in calls)
    assert any("torch.cuda.synchronize" in command[2] for command, _ in calls)
    assert any("strategy='auto'" in command[2] for command, _ in calls)


def test_exp3_only_checks_its_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="", stderr=""))
    result = check_runtime(tmp_path, include_exp2=False)
    assert result["passed"]
    assert "src.models.s4" in result["probes"]
    assert "modular S4 construction" in result["probes"]
    assert "S4 source provenance" in result["probes"]
    assert "upstream_s4_train" not in result["probes"]


def test_import_timeout_fails_closed(tmp_path, monkeypatch):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", run)
    assert not check_runtime(tmp_path, include_exp2=False)["passed"]


def test_lazy_s4_dependency_failure_blocks_exp3(tmp_path, monkeypatch):
    def run(command, **kwargs):
        broken = "create_modular_s4" in command[2]
        return SimpleNamespace(returncode=int(broken), stdout="",
                               stderr="ImportError: modular S4 dependency unavailable" if broken else "")
    monkeypatch.setattr(subprocess, "run", run)
    result = check_runtime(tmp_path, include_exp2=False)
    assert not result["passed"]
    assert result["probes"]["src.models.s4"]["passed"]
    assert not result["probes"]["modular S4 construction"]["passed"]


def test_changed_s4_sources_block_runtime_even_if_imports_work(tmp_path, monkeypatch):
    def run(command, **kwargs):
        broken = "check_s4_upstream.py" in command[2]
        return SimpleNamespace(returncode=int(broken), stderr="",
                               stdout="unaudited source/config file" if broken else "")
    monkeypatch.setattr(subprocess, "run", run)
    result = check_runtime(tmp_path, include_exp2=False)
    assert not result["passed"]
    assert not result["probes"]["S4 source provenance"]["passed"]
