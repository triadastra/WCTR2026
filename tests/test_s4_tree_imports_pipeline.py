"""The vendored S4 tree loads src/data/dataloader.py BY PATH from a working
directory where `src` is S4's own package. A package-style import inside the
pipeline file therefore has to survive that context, or every hydra S4/S4ND
train and eval run dies at import -- which is what happened between
2026-09-04 (9275290 added `from src.utils.provenance import ...`) and
2026-09-09. No in-repo test exercised that path; this one does."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
S4 = ROOT / "external" / "s4"


def _import_in_s4_tree(module):
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=S4, capture_output=True, text=True, timeout=600,
    )


@pytest.mark.parametrize("module", [
    "src.dataloaders.trade_unified",
    "src.dataloaders.trade_grid",
])
def test_pipeline_imports_from_the_vendored_s4_tree(module):
    proc = _import_in_s4_tree(module)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        # A missing third-party dependency is an environment gap, not the
        # regression this test guards; the regression names a repo module.
        if "No module named" in tail and "'src." not in tail and "github" not in tail:
            pytest.skip(f"S4 tree dependencies unavailable here: {tail}")
        pytest.fail(f"importing {module} inside external/s4 failed:\n{proc.stderr[-2000:]}")


def test_pipeline_file_loads_standalone_by_path(tmp_path):
    """The exact mechanism trade_unified.py uses, minus the S4 dependencies."""
    code = (
        "import importlib.util, pathlib, sys\n"
        f"p = pathlib.Path({str(ROOT / 'src' / 'data' / 'dataloader.py')!r})\n"
        "spec = importlib.util.spec_from_file_location('github_dataloader', p)\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert m.BENCHMARK_PROTOCOL_VERSION == 'fair-comparison-v5'\n"
        "print('ok')\n"
    )
    # Run from a directory that has NO `src` package on the path at all.
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "ok"
