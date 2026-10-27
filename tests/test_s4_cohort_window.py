"""The vendored S4 dataloaders must freeze the cohort on training data.

`external/s4/**/shared_dataloader.py::build_grid` applies the same >=95%-nonzero
sparse-combo filter as the canonical `TradeDataPipeline`. It used to measure
nonzero density over the FULL timeline, which lets val/test targets decide which
series are evaluated -- survivorship bias, since the full-timeline window can
only ever drop series that were dense through train and thinned afterwards
(918 train-eligible combos vs 903 full-timeline on the real dataset).

These tests pin the corrected behaviour:

  1. the filter measures on `Time <= cohort_train_end` only, so a series that
     dies during val/test is still evaluated;
  2. calling `build_grid` with the filter active but no train boundary raises
     instead of silently falling back to the full timeline;
  3. every call site in `external/s4` passes a train boundary.

Run:  pytest tests/test_s4_cohort_window.py -v
"""
import importlib.util
import pathlib
import re
import sys

import pandas as pd
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_COPIES = {
    "s4_root": _REPO / "external" / "s4" / "shared_dataloader.py",
    "s4_pkg": _REPO / "external" / "s4" / "src" / "dataloaders" / "shared_dataloader.py",
}

START = "2020-01-01"
END = "2021-12-01"
TRAIN_END = "2020-12-01"   # 12 train months, 12 post-train months


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=sorted(_COPIES), ids=sorted(_COPIES))
def sdl(request, monkeypatch):
    monkeypatch.delenv("S4_MIN_NONZERO_FRAC", raising=False)
    path = _COPIES[request.param]
    if not path.exists():
        pytest.skip(f"{path} not present")
    return _load(request.param, path)


def _panel() -> pd.DataFrame:
    """Three combos: always dense, dense-then-dead, and sparse during train.

    Rows are only emitted for nonzero months; `build_grid` zero-fills the rest.
    """
    months = pd.date_range(START, END, freq="MS")
    train_months = [m for m in months if m <= pd.Timestamp(TRAIN_END)]
    rows = []

    def add(commodity, when):
        for t in when:
            rows.append({
                "State": "TX", "Commodity": commodity, "Country": "MX",
                "Time": t, "Import/Export": "Import",
                "Value": 100.0, "Weight": 10.0,
            })

    add("dense", months)                      # nonzero throughout
    add("thins_late", train_months)           # dense in train, dead in val/test
    add("sparse_train", train_months[::2])    # 50% nonzero during train
    return pd.DataFrame(rows)


def _kept(grid: pd.DataFrame) -> set:
    return set(grid["Commodity"].unique())


def test_cohort_is_measured_on_train_slice_only(sdl):
    """A series that dies after train_end stays in the evaluated population."""
    grid = sdl.build_grid(_panel(), START, END, TRAIN_END)
    assert _kept(grid) == {"dense", "thins_late"}, (
        "the sparse filter must judge series on the train slice only — "
        "'thins_late' is dense through train_end and must survive, and "
        "'sparse_train' must be dropped"
    )


def test_full_timeline_window_is_unreachable(sdl):
    """No argument may restore the survivorship-biased full-timeline window."""
    with pytest.raises(ValueError, match="survivorship"):
        sdl.build_grid(_panel(), START, END)


def test_train_boundary_must_overlap_the_grid(sdl):
    with pytest.raises(ValueError, match="cohort_train_end"):
        sdl.build_grid(_panel(), START, END, "1999-01-01")


def test_filter_can_still_be_disabled_wholesale(sdl, monkeypatch):
    """S4_MIN_NONZERO_FRAC=0 (paper-style all-combos) needs no train boundary."""
    monkeypatch.setenv("S4_MIN_NONZERO_FRAC", "0")
    grid = sdl.build_grid(_panel(), START, END)
    assert _kept(grid) == {"dense", "thins_late", "sparse_train"}


def test_real_data_cohort_matches_the_canonical_pipeline(sdl):
    """The vendored loaders must select the same 918 lanes as TradeDataPipeline.

    Both must exclude the burn-in year: `add_lags` drops it, so counting its
    months in the filter would judge series on rows no model ever sees (that
    alone cost 9 lanes here — 909 instead of 918).
    """
    imports = _REPO / "data" / "imports"
    exports = _REPO / "data" / "exports"
    if not list(imports.glob("*.csv")) or not list(exports.glob("*.csv")):
        pytest.skip("real dataset not present under data/imports & data/exports")
    cfg = sdl.SharedConfig()
    panel = pd.concat(
        [sdl.read_folder(str(imports), "Import"), sdl.read_folder(str(exports), "Export")],
        ignore_index=True,
    )
    grid = sdl.build_grid(panel, cfg.start, cfg.end, cfg.split_train_end)
    assert grid.groupby(sdl.KEYS).ngroups == 918, (
        "the vendored S4 cohort must match the canonical train-only cohort "
        "(918 combos); see tests/test_data_pipeline.py::KEPT_COMBOS"
    )


CALL = re.compile(r"\b(?:_flat_)?build_grid\(")


def test_every_s4_call_site_passes_a_train_boundary():
    """A new caller must not be able to reintroduce the full-timeline cohort."""
    offenders = []
    for path in sorted((_REPO / "external" / "s4").rglob("*.py")):
        if path.name == "tx.py":
            continue  # its own build_grid applies no cohort filter at all
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if "def build_grid" in line or not CALL.search(line):
                continue
            if "split_train_end" not in line and "cohort_train_end" not in line:
                offenders.append(f"{path.relative_to(_REPO)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "build_grid call sites without a train boundary (the sparse-combo "
        "cohort would be chosen with val/test in view):\n" + "\n".join(offenders)
    )
