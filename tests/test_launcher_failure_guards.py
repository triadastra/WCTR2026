import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_full_benchmark_suppresses_inference_after_training_failure(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts" / "run_full_test.sh", scripts)
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "experiment_manifest.env", tmp_path / "config")
    (tmp_path / "external" / "s4").mkdir(parents=True)
    # A monthly-release root must reach the configured source check without
    # requiring unused raw report directories.
    (tmp_path / "data" / "processed").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$*\" >> \"$FAKE_CALLS\"\n"
        "case \"$*\" in\n"
        "  *run_is_current.py*) exit 1 ;;\n"
        "  *train.py*) exit 1 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    fake_python.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["PYTHON"] = str(fake_python)
    env["PY"] = str(fake_python)
    env["DATA_DIR"] = str(tmp_path)
    env["FAKE_CALLS"] = str(calls)
    result = subprocess.run(
        ["bash", str(scripts / "run_full_test.sh"), "26"],
        cwd=tmp_path,
        env=env,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    invoked = calls.read_text()
    assert "scripts/check_legacy_data.py" in invoked
    assert "scripts/train.py" in invoked
    assert "scripts/inference.py" not in invoked
    assert result.returncode != 0


def test_all_experiment_training_calls_are_failure_guarded():
    source = (ROOT / "scripts" / "run_all_experiments.sh").read_text()
    assert source.count("inference suppressed") >= 2
    assert 'if [ "$FAILED" -ne 0 ]' in source
    assert "no workbook was built" in source


def test_exp3_launcher_pins_shared_targets_and_all_seeds():
    source = (ROOT / "scripts" / "run_all_experiments.sh").read_text()
    manifest = (ROOT / "config/experiment_manifest.env").read_text()
    assert "PAPER_SEEDS='26 105 115'" in manifest
    assert "PAPER_SWEEP_TRAIN_START='2015-01-01'" in manifest
    assert 'SWEEP_SEEDS="${SWEEP_SEEDS:-$PAPER_SEEDS}"' in source
    assert 'for seed in $SWEEP_SEEDS' in source
    assert '--train-target-start "$SWEEP_TRAIN_START"' in source


def test_master_prints_the_canonical_plan_without_starting_training():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_all_experiments.sh"), "--print-plan"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PY": sys.executable},
    )
    assert "Exp 1: 14 neural/SSM arms + 4 stochastic tree arms x 25 seeds" in result.stdout
    assert "Exp 2: 57 arms x 3 seeds" in result.stdout
    assert "Exp 3: 64 grid cells x 3 seeds" in result.stdout


def test_master_has_a_lock_and_exact_manifest_build_gate():
    source = (ROOT / "scripts" / "run_all_experiments.sh").read_text()
    assert 'LOCK_DIR="outputs/.run_all_experiments.lock"' in source
    assert "verify_paper_outputs" in source
    assert 'EXP1_RUNS_DIR="outputs/agg"' in source
    assert 'verify_no_unexpected_metrics "$EXP1_RUNS_DIR" "$expected_agg"' in source
    assert "verify_no_unexpected_metrics outputs/full" in source


def test_master_manifest_matches_the_full_benchmark_launcher():
    from src.utils.experiment_manifest import load_manifest
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    assert manifest["EXP2_BASIC_MODELS"].split() == "lstm gru transformer gpt mamba xgboost lightgbm random_forest".split()
    assert manifest["EXP2_COMBO_MODELS"].split() == ["lstm", "gru", "transformer"]
    for launcher in ("run_all_experiments.sh", "run_full_test.sh"):
        assert "experiment_manifest.env" in (ROOT / "scripts" / launcher).read_text()


def test_paper_mode_rejects_noncanonical_seed_override_before_training():
    env = dict(os.environ)
    env.update({"PIPELINE_MODE": "paper", "SEEDS": "42", "CHECK_GPU": "0", "PY": sys.executable})
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_all_experiments.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "paper mode requires SEEDS='26 105 115'" in result.stderr


def test_s4_launcher_preserves_and_resumes_checkpoints():
    source = (ROOT / "scripts" / "run_full_test.sh").read_text()
    assert 'training_complete' in source
    assert 'train.ckpt=' in source
    assert 'if [ "$RESUME" = "1" ] && [ -f "$last" ]' in source


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_master_launcher_does_not_build_workbook_after_worker_failure(tmp_path, runtime_failure):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts" / "run_all_experiments.sh", scripts)
    (scripts / "run_full_test.sh").write_text(
        "#!/usr/bin/env bash\n"
        "echo run_full_test >> \"$FAKE_CALLS\"\n"
        "exit 1\n"
    )
    (tmp_path / "external" / "s4").mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/experiment_manifest.env", tmp_path / "config")
    (tmp_path / "config" / "base.yaml").write_text("data: {}\n")
    (tmp_path / "data" / "processed").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$*\" >> \"$FAKE_CALLS\"\n"
        "if [ \"${FAIL_RUNTIME:-0}\" = 1 ] && [[ \"$*\" == *--check-runtime* ]]; then exit 1; fi\n"
        "exit 0\n"
    )
    fake_python.chmod(0o755)

    env = dict(os.environ)
    env.update({
        "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        "PY": str(fake_python),
        "FAKE_CALLS": str(calls),
        "FAIL_RUNTIME": "1" if runtime_failure else "0",
        "PIPELINE_MODE": "smoke",
        "SEEDS": "26",
        "EXP1_SEEDS": "26",
        "SWEEP_SEEDS": "26",
        "GPUS": "0",
        "RUN_SWEEP": "0",
        "CHECK_GPU": "0",
        "DATA_DIR": str(tmp_path),
    })
    result = subprocess.run(
        ["bash", str(scripts / "run_all_experiments.sh")],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert calls.exists(), result.stdout + result.stderr
    invoked = calls.read_text()
    assert "scripts/check_legacy_data.py" in invoked
    if runtime_failure:
        assert "--check-runtime" in invoked
        assert "scripts/train.py" not in invoked
        assert "run_full_test" not in invoked
    else:
        assert "run_full_test" in invoked, result.stdout + result.stderr
    assert "scripts/build_paper_results.py" not in invoked


@pytest.mark.parametrize("ignore_term", [False, True])
def test_master_interrupt_stops_training_descendants_before_unlock(tmp_path, ignore_term):
    import signal
    import time

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/run_all_experiments.sh", scripts)
    (scripts / "run_full_test.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/experiment_manifest.env", tmp_path / "config")
    (tmp_path / "config/base.yaml").write_text("data: {}\n")
    for flow in ("imports", "exports"):
        directory = tmp_path / "data" / flow
        directory.mkdir(parents=True)
        (directory / "input.csv").write_text("x\n1\n")
    marker = tmp_path / "training_pids"
    fake = tmp_path / "fake_python"
    fake.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in\n'
        'scripts/run_is_current.py) exit 1 ;;\n'
        'scripts/train.py)\n'
        '  if [ "$IGNORE_TERM" = 1 ]; then trap "" TERM; fi\n'
        '  sleep 60 &\n'
        '  child=$!\n'
        '  printf "%s %s\\n" "$$" "$child" > "$TRAINING_PIDS"\n'
        '  wait "$child" ;;\n'
        '*) exit 0 ;;\n'
        'esac\n'
    )
    fake.chmod(0o755)
    env = {**os.environ, "PY": str(fake), "TRAINING_PIDS": str(marker),
           "DATA_DIR": str(tmp_path), "PIPELINE_MODE": "smoke", "CHECK_GPU": "0",
           "SEEDS": "26", "EXP1_SEEDS": "26", "SWEEP_SEEDS": "26",
           "GPUS": "0", "RUN_SWEEP": "0", "IGNORE_TERM": "1" if ignore_term else "0"}

    def running(pid):
        result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                capture_output=True, text=True)
        return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")

    process = subprocess.Popen(["bash", str(scripts / "run_all_experiments.sh")],
                               cwd=tmp_path, env=env, start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    children = []
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert process.poll() is None, "controller exited before launching test training"
            time.sleep(0.05)
        assert marker.exists(), "training did not launch"
        children = [int(value) for value in marker.read_text().split()]
        assert len(children) == 2 and all(running(pid) for pid in children)
        process.send_signal(signal.SIGTERM)
        if ignore_term:
            time.sleep(0.2)
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) != 0
        deadline = time.monotonic() + 2
        while any(running(pid) for pid in children) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(running(pid) for pid in children), "orphaned training descendants"
        assert not (tmp_path / "outputs/.run_all_experiments.lock").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for pid in children:
            if running(pid):
                os.kill(pid, signal.SIGKILL)
