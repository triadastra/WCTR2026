"""Do not call the S4 runtime verified after constructor-only unit tests."""
from pathlib import Path
import subprocess
import sys


def test_all_six_external_s4_training_checkpoint_and_evaluation_lifecycles():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "tests/smoke_exp2_s4.py")],
                            cwd=root / "external/s4", capture_output=True, text=True,
                            timeout=240)
    assert result.returncode == 0, (
        "S4 lifecycle failed. Use the benchmark dependencies (requirements-benchmark.txt); "
        "Lightning 1.9 does not support the configured strategy='auto'.\n"
        + result.stdout + result.stderr
    )
    for experiment in ("s4_embeddings", "s4_onehot", "s4_embeddings_bidir", "s4_onehot_bidir", "s4nd_2d_state", "s4nd_2d", "s4nd_3d", "s4nd_4d"):
        assert f"SMOKE PASS {experiment}\n" in result.stdout
