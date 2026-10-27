"""The user-facing S4 table must cover exactly the canonical experiment roster."""
import importlib.util
import copy
from pathlib import Path

import pytest

from src.utils.experiment_manifest import load_manifest, exp1_expected_runs
from src.utils.exp2_protocol import expected_runs
from src.utils.exp3_protocol import coordinates, run_name

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s4_inventory", ROOT / "scripts/report_s4_experiments.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture(scope="module")
def rows():
    return report.inventory()


def test_table_covers_every_canonical_s4_run_exactly_once(rows):
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    expected = {(1, run.directory_name, run.seed) for run in exp1_expected_runs(manifest)
                if run.model in {"s4", "s4_bidir"}}
    expected |= {(2, run.directory_name, run.seed) for run in expected_runs(manifest)
                 if run.model in {"s4", "s4_bidir", "s4nd"}}
    expected |= {(3, run_name(model, w, lag, seed), seed) for model, w, lag, seed in coordinates()
                 if model in {"s4", "s4_bidir"}}
    actual = [(row["experiment"], run["directory"], run["seed"]) for row in rows for run in row["runs"]]
    assert len(actual) == len(set(actual)) == 238
    assert set(actual) == expected
    assert len(rows) == 50
    assert {e: sum(len(r["runs"]) for r in rows if r["experiment"] == e)
            for e in (1, 2, 3)} == {1: 100, 2: 42, 3: 96}


def test_reported_shared_settings_match_actual_contracts(rows):
    for row in rows:
        assert (row["layers"], row["width"], row["state_size"], row["n_ssm"], row["dropout"]) == (4, 128, 64, 1, 0.1)
        contract = row["representative_contract"]
        training = contract["training_config"]
        if row["runner"] == "external/s4/train.py":
            assert training["optimizer"]["_name_"] == "adamw"
            assert training["optimizer"]["lr"] == training["optimizer"]["weight_decay"] == 1e-4
            assert training["trainer"]["max_epochs"] == 5000
            assert training["trainer"]["precision"] == 32
            assert training["trainer"]["gradient_clip_val"] == 1.0
            assert training["callbacks"]["early_stopping"]["patience"] == 50
            assert training["callbacks"]["early_stopping"]["min_delta"] == 1e-4
            assert training["scheduler"]["patience"] == 15
            assert training["scheduler"]["factor"] == 0.5
            assert contract["model_config"]["model"]["layer"]["lr"] is None
        else:
            assert contract["model_config"]["implementation"] == "authors-modular-s4-v1"
            assert training["optimizer"] == "adamw"
            assert training["lr"] == training["weight_decay"] == 1e-4
            assert training["epochs"] == 5000 and training["patience"] == 50
            assert training["min_delta"] == 1e-4
            assert training["scheduler_patience"] == 15 and training["scheduler_factor"] == 0.5
            assert training["grad_clip_norm"] == 1.0 and not training["use_amp"]
        assert row["bidirectional"] == (row["model"] in {"s4nd", "s4_bidir"})
    nd = {r["variant"]: r for r in rows if r["model"] == "s4nd"}
    assert {v: r["kernel_axes"] for v, r in nd.items()} == {"2d_state": 2, "2d": 3, "3d": 3, "4d": 4}


def test_published_config_table_is_current(rows):
    assert (ROOT / "docs/S4_EXPERIMENTS.md").read_text() == report.markdown(rows)


def test_directions_are_paired_without_private_tuning(rows):
    def key(row):
        return row["experiment"], row["variant"], row["window"], row["lags"]
    causal = {key(row): row for row in rows if row["model"] == "s4"}
    bidirectional = {key(row): row for row in rows if row["model"] == "s4_bidir"}
    assert causal.keys() == bidirectional.keys()
    for coordinate, row in causal.items():
        other = bidirectional[coordinate]
        assert [r["seed"] for r in row["runs"]] == [r["seed"] for r in other["runs"]]
        assert not ({r["directory"] for r in row["runs"]} & {r["directory"] for r in other["runs"]})
        before, after = row["representative_contract"], other["representative_contract"]
        for field in ("data_config", "training_config", "selection", "source_fingerprint", "source_domain"):
            assert before[field] == after[field], (coordinate, field)
        architecture = copy.deepcopy(before["model_config"])
        if "model" in architecture:
            architecture["model"]["layer"]["bidirectional"] = True
        else:
            architecture["name"] = "s4_bidir"
            architecture["bidirectional"] = True
        assert architecture == after["model_config"]
