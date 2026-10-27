"""Read and validate the canonical experiment manifest.

The manifest is a deliberately small ``.env`` subset shared with Bash.  This
module parses it as data; it never sources or evaluates it.  Keeping the paper
arm/seed population here prevents the launcher and workbook builder from
quietly describing different experiments.
"""
from __future__ import annotations

import re
import shlex
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


MANIFEST_RELATIVE_PATH = Path("config/experiment_manifest.env")
_ASSIGNMENT = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)=(?P<value>.*)$")
_REQUIRED = {
    "PAPER_SEEDS",
    "PAPER_EXP1_SEEDS",
    "PAPER_EXP1_BASELINE_SEED",
    "PAPER_EXP1_INPUT_LEN",
    "PAPER_EXP1_LAG_COUNT",
    "PAPER_EXP1_TRAIN_TARGET_START",
    "PAPER_EXP1_BATCH_SIZE",
    "PAPER_MOVING_AVERAGE_WINDOW",
    "PAPER_STATES",
    "PAPER_SOURCE_FINGERPRINT",
    "PAPER_SOURCE_START",
    "PAPER_SOURCE_END",
    "PAPER_TARGET_MEASURE",
    "PAPER_DUPLICATE_POLICY",
    "PAPER_COMMODITY_UNIVERSE",
    "PAPER_HS2_CODES",
    "PAPER_DATA_START",
    "PAPER_DATA_END",
    "PAPER_TRAIN_END",
    "PAPER_VAL_START",
    "PAPER_VAL_END",
    "PAPER_TEST_START",
    "PAPER_TEST_END",
    "PAPER_NORMALIZATION_METHOD",
    "PAPER_SWEEP_MODELS",
    "PAPER_WINDOWS",
    "PAPER_LAGS",
    "PAPER_SWEEP_BATCH",
    "PAPER_SWEEP_TRAIN_START",
    "BASIC_VARIANTS",
    "EXP1_NEURAL_MODELS",
    "EXP1_TREE_MODELS",
    "EXP2_BASIC_MODELS",
    "EXP2_COMBO_MODELS",
    "EXP2_AXIAL_MODELS",
    "EXP2_AXIAL_VARIANTS",
    "EXP2_S4_EXPERIMENTS",
    "BASELINES",
}


@dataclass(frozen=True, order=True)
class Exp1Run:
    model: str
    variant: str
    seed: int
    kind: str

    @property
    def directory_name(self) -> str:
        return f"{self.model}_{self.variant}_seed{self.seed}"


def _one_shell_word(raw: str, *, key: str, path: Path) -> str:
    try:
        words = shlex.split(raw, comments=False, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid {key} assignment in {path}: {exc}") from exc
    if len(words) != 1:
        raise ValueError(
            f"{key} in {path} must contain exactly one quoted shell word"
        )
    return words[0]


def load_manifest(path: Path | str) -> dict[str, str]:
    """Parse the non-executable KEY='value' manifest subset."""
    path = Path(path)
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if not match:
            raise ValueError(f"invalid manifest line {path}:{number}: {raw_line!r}")
        key = match.group("key")
        if key in values:
            raise ValueError(f"duplicate manifest key {key!r} in {path}")
        value = _one_shell_word(match.group("value"), key=key, path=path)
        # The same file is sourced by Bash. Keep the accepted data language
        # intentionally smaller than shell syntax so command/parameter
        # expansion cannot hide inside an otherwise valid-looking assignment.
        if re.fullmatch(r"[A-Za-z0-9_./ -]+", value) is None:
            raise ValueError(
                f"unsafe character in {key} assignment in {path}"
            )
        values[key] = value
    missing = sorted(_REQUIRED.difference(values))
    unknown = sorted(set(values).difference(_REQUIRED))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise ValueError(f"invalid experiment manifest {path}: {'; '.join(detail)}")
    return values


def _tokens(manifest: Mapping[str, str], key: str) -> tuple[str, ...]:
    values = tuple(manifest[key].split())
    if not values:
        raise ValueError(f"manifest key {key} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"manifest key {key} contains duplicate values")
    return values


def _integer_tokens(manifest: Mapping[str, str], key: str) -> tuple[int, ...]:
    values = _tokens(manifest, key)
    try:
        parsed = tuple(int(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"manifest key {key} must contain integers") from exc
    if any(value < 0 for value in parsed):
        raise ValueError(f"manifest key {key} must contain non-negative integers")
    return parsed


def exp1_expected_runs(manifest: Mapping[str, str]) -> tuple[Exp1Run, ...]:
    """Return the complete, unique Experiment 1 paper population."""
    stochastic_seeds = _integer_tokens(manifest, "PAPER_EXP1_SEEDS")
    baseline_seeds = _integer_tokens(manifest, "PAPER_EXP1_BASELINE_SEED")
    if len(baseline_seeds) != 1:
        raise ValueError("PAPER_EXP1_BASELINE_SEED must contain exactly one seed")
    variants = _tokens(manifest, "BASIC_VARIANTS")
    runs = [
        Exp1Run(model, variant, seed, "stochastic")
        for model in _tokens(manifest, "EXP1_NEURAL_MODELS")
        for variant in variants
        for seed in stochastic_seeds
    ]
    runs.extend(
        Exp1Run(model, variant, seed, "tree")
        for model in _tokens(manifest, "EXP1_TREE_MODELS")
        for variant in variants
        for seed in stochastic_seeds
    )
    runs.extend(
        Exp1Run(model, "none", seed, "baseline")
        for model in _tokens(manifest, "BASELINES")
        for seed in baseline_seeds
    )
    names = [run.directory_name for run in runs]
    if len(names) != len(set(names)):
        raise ValueError("Experiment 1 manifest produces duplicate run directories")
    return tuple(sorted(runs))


def exp1_protocol(manifest: Mapping[str, str]) -> dict[str, object]:
    target_start = manifest["PAPER_EXP1_TRAIN_TARGET_START"]
    return {
        "input_len": int(manifest["PAPER_EXP1_INPUT_LEN"]),
        "lag_count": int(manifest["PAPER_EXP1_LAG_COUNT"]),
        "train_target_start": None if target_start == "null" else target_start,
        "batch_size": int(manifest["PAPER_EXP1_BATCH_SIZE"]),
        "moving_average_window": int(manifest["PAPER_MOVING_AVERAGE_WINDOW"]),
        "states": _tokens(manifest, "PAPER_STATES"),
        "source_fingerprint": manifest["PAPER_SOURCE_FINGERPRINT"],
        "source_start": manifest["PAPER_SOURCE_START"],
        "source_end": manifest["PAPER_SOURCE_END"],
        "target_measure": manifest["PAPER_TARGET_MEASURE"],
        "duplicate_policy": manifest["PAPER_DUPLICATE_POLICY"],
        "commodity_universe": manifest["PAPER_COMMODITY_UNIVERSE"],
        "hs2_codes": _tokens(manifest, "PAPER_HS2_CODES"),
        "start": manifest["PAPER_DATA_START"],
        "end": manifest["PAPER_DATA_END"],
        "train_end": manifest["PAPER_TRAIN_END"],
        "val_start": manifest["PAPER_VAL_START"],
        "val_end": manifest["PAPER_VAL_END"],
        "test_start": manifest["PAPER_TEST_START"],
        "test_end": manifest["PAPER_TEST_END"],
        "normalization_method": manifest["PAPER_NORMALIZATION_METHOD"],
    }


def exp1_source_domain(manifest: Mapping[str, str]) -> dict[str, object]:
    """Return the exact source population/measure/calendar for Exp 1."""
    protocol = exp1_protocol(manifest)
    return {
        "states": list(protocol["states"]),
        "target_measure": protocol["target_measure"],
        "duplicate_audit": {
            "policy": protocol["duplicate_policy"],
            "groups": 2250 if protocol["duplicate_policy"] == "legacy_sum" else 0,
            "rows": 4500 if protocol["duplicate_policy"] == "legacy_sum" else 0,
            "excess_rows": 2250 if protocol["duplicate_policy"] == "legacy_sum" else 0,
            "conflicting_target_groups": 2198 if protocol["duplicate_policy"] == "legacy_sum" else 0,
        },
        "commodity_universe": protocol["commodity_universe"],
        "hs2_codes": list(protocol["hs2_codes"]),
        "source_start": protocol["source_start"],
        "source_end": protocol["source_end"],
        "grid_start": protocol["start"],
        "grid_end": protocol["end"],
        "train_end": protocol["train_end"],
        "val_start": protocol["val_start"],
        "val_end": protocol["val_end"],
        "test_start": protocol["test_start"],
        "test_end": protocol["test_end"],
    }


def validate_exp1_config(manifest: Mapping[str, str], config: Mapping[str, object]) -> None:
    """Fail before GPU work when YAML and the paper manifest disagree."""
    from src.utils.config import get_data_config

    expected = exp1_protocol(manifest)
    data = get_data_config(dict(config))
    training = config.get("training")
    if not isinstance(data, Mapping) or not isinstance(training, Mapping):
        raise ValueError("resolved Exp 1 config requires data and training mappings")
    config_fields = {
        "input_len": data.get("input_len"),
        "lag_count": data.get("lag_count"),
        "train_target_start": data.get("train_target_start"),
        "start": str(data.get("start"))[:10],
        "end": str(data.get("end"))[:10],
        "train_end": str(data.get("train_end"))[:10],
        "val_start": str(data.get("val_start"))[:10],
        "val_end": str(data.get("val_end"))[:10],
        "test_start": str(data.get("test_start"))[:10],
        "test_end": str(data.get("test_end"))[:10],
        "normalization_method": data.get("normalization_method"),
        "target_measure": data.get("target_measure"),
        "duplicate_policy": data.get("duplicate_policy"),
        "commodity_universe": data.get("commodity_universe"),
        "batch_size": training.get("batch_size"),
    }
    mismatches = {
        key: (config_fields[key], expected[key])
        for key in config_fields
        if config_fields[key] != expected[key]
    }
    expected_states = data.get("expected_states")
    if expected_states is None or tuple(expected_states) != expected["states"]:
        mismatches["expected_states"] = (expected_states, expected["states"])
    expected_hs2_codes = data.get("expected_hs2_codes")
    if expected_hs2_codes is None or tuple(expected_hs2_codes) != expected["hs2_codes"]:
        mismatches["expected_hs2_codes"] = (
            expected_hs2_codes, expected["hs2_codes"]
        )
    pinned_source = data.get("expected_source_fingerprint")
    if pinned_source != expected["source_fingerprint"]:
        mismatches["expected_source_fingerprint"] = (
            pinned_source, expected["source_fingerprint"]
        )
    for config_key, expected_key in (
        ("expected_source_start", "source_start"),
        ("expected_source_end", "source_end"),
    ):
        actual = str(data.get(config_key))[:10]
        if actual != expected[expected_key]:
            mismatches[config_key] = (actual, expected[expected_key])
    if data.get("normalize") is not True:
        mismatches["normalize"] = (data.get("normalize"), True)
    if mismatches:
        detail = ", ".join(
            f"{key}={actual!r} (manifest {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"resolved Exp 1 YAML conflicts with paper manifest: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the canonical experiment manifest")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    manifest = load_manifest(repo / MANIFEST_RELATIVE_PATH)
    # Import lazily so the data-only manifest parser has no YAML dependency.
    from src.utils.config import load_config

    config = load_config(str(repo / "config" / "base.yaml"))
    config = load_config(str(repo / "config" / "exp1.yaml"), config)
    validate_exp1_config(manifest, config)
    runs = exp1_expected_runs(manifest)
    print(
        f"validated manifest: Exp 1 has {len(runs)} runs across "
        f"{len({(run.model, run.variant) for run in runs})} arms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
