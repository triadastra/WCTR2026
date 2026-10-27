"""
Unified data pipeline for time series forecasting.

Implements TradeDataPipeline class that loads, preprocesses, normalizes,
and creates train/val/test splits for merchandise trade data.
"""

import hashlib
import io
import json
import math
import re
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader



def _load_sibling(module_name: str, relative_path: str):
    """Import a repo module by PATH when the package import is unavailable.

    ``external/s4/src/dataloaders/trade_unified.py`` and ``trade_grid.py`` load
    this file with ``importlib.util.spec_from_file_location`` -- as a bare
    module, from a working directory where ``src`` is the vendored S4 package.
    There ``from src.utils.provenance import ...`` resolves against S4's own
    ``src/utils`` and fails, and a relative import has no parent package. That
    broke every hydra S4/S4ND train and eval run between 2026-09-04 and
    2026-09-09 (ModuleNotFoundError: src.utils.provenance). Resolve the sibling
    by path from this file's location instead, and keep the normal import for
    the in-package case so tests and tooling see one module object.
    """
    import importlib
    import importlib.util as _ilu
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("src"):
            raise
    target = Path(__file__).resolve().parents[1] / relative_path
    spec = _ilu.spec_from_file_location(f"_github_{target.stem}", target)
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK_PROTOCOL_VERSION = _load_sibling(
    "src.utils.benchmark_protocol", "utils/benchmark_protocol.py"
).BENCHMARK_PROTOCOL_VERSION


# Column name constants
VALUE_CANON = "Value"
WEIGHT_CANON = "Weight"
KEYS = ["State", "Commodity", "Import/Export"]
SOURCE_REPORT_KEY_COLUMNS = [
    "State", "Commodity", "Country", "Import/Export", "Time"
]

# Raw column name mappings
EXPORT_VALUE_TOTAL = "Vessel Total Exports Value ($US)"
EXPORT_WEIGHT_TOTAL = "Vessel Total Exports SWT (kg)"
EXPORT_VALUE_CONTAINERIZED = "Containerized Vessel Total Exports Value ($US)"
EXPORT_WEIGHT_CONTAINERIZED = "Containerized Vessel Total Exports SWT (kg)"
IMPORT_VALUE_TOTAL = "Vessel Customs Value (Gen) ($US)"
IMPORT_WEIGHT_TOTAL = "Vessel SWT (Gen) (kg)"
IMPORT_VALUE_CONTAINERIZED = "Customs Containerized Vessel Value (Gen) ($US)"
IMPORT_WEIGHT_CONTAINERIZED = "Containerized Vessel SWT (Gen) (kg)"

# Backward-compatible names used by a few callers/tests. They now intentionally
# identify the canonical total-vessel export target, not the old mixed target.
EXPORT_VALUE_RAW = EXPORT_VALUE_TOTAL
EXPORT_WEIGHT_RAW = EXPORT_WEIGHT_TOTAL
IMPORT_VALUE_PRI = IMPORT_VALUE_TOTAL
IMPORT_VALUE_ALT = IMPORT_VALUE_CONTAINERIZED
IMPORT_WEIGHT_PRI = IMPORT_WEIGHT_TOTAL
IMPORT_WEIGHT_ALT = IMPORT_WEIGHT_CONTAINERIZED

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    source_format: str = "reports"
    aggregated_csv: str = "data/processed/hs2_monthly_aggregated.csv"
    imports_dir: str = "data/imports"
    exports_dir: str = "data/exports"
    # Optional paper-data pin. When set, load_data fails if any consumed CSV
    # name or byte changes. Custom/synthetic DataConfig instances may omit it.
    expected_source_fingerprint: Optional[str] = None
    expected_states: Optional[List[str]] = None
    expected_hs2_codes: Optional[List[str]] = None
    expected_source_start: Optional[str] = None
    expected_source_end: Optional[str] = None
    # The supplied shipment reference covers vessel bills of lading, not only
    # rows carrying a container number. Keep import/export target semantics
    # symmetric. The previous loader silently mixed total-vessel imports with
    # containerized-only exports, so its aggregate was not one defined measure.
    target_measure: str = "total_vessel"
    # Explicit historical reproduction only; never silently deduplicate.
    duplicate_policy: str = "error"
    # The checked-in reports contain all 97 represented two-digit HS chapters.
    # No agricultural HS filter is applied; state that scope explicitly.
    commodity_universe: str = "all_hs_chapters"
    start: str = "2009-01-01"
    end: str = "2025-05-01"
    input_len: int = 36
    lag_count: int = 12
    normalize: bool = True
    normalization_method: str = "minmax"

    # Sparse-combo filtering. The dense monthly grid fills no-trade months
    # with 0, so a large fraction of (State, Commodity, Flow) combos are
    # structurally sparse (mostly zero). Those zeros make relative metrics
    # (MAPE) explode and add no signal. When ``filter_sparse`` is True we
    # keep only combos whose RAW Value is nonzero in at least
    # ``min_nonzero_frac`` of months. This DIVERGES from paper included/
    # (which used every combo): the Phase 7 equivalence test must set
    # filter_sparse=False; real runs use the default. See
    # docs/BEHAVIOR_DIFFERENCES.md.
    filter_sparse: bool = True
    min_nonzero_frac: float = 0.95

    # Window the sparse-combo filter measures over. True (default) measures
    # nonzero density on the TRAIN slice only (``Time <= train_end``) and
    # freezes the cohort before val/test are touched. False formerly enabled
    # the pre-fix behaviour of measuring over the full timeline, which is
    # survivorship-biased: it can only ever DROP series that were dense
    # through train and thinned during val/test (measured on this dataset:
    # 918 train-eligible combos vs 903 under the full timeline, with zero
    # combos going the other way). False is rejected to prevent leakage.
    cohort_from_train_only: bool = True
    # Only the explicitly isolated historical reproduction profile may inspect
    # the full timeline when selecting its cohort. Merely using the preserved
    # mixed-measure source must never opt canonical runs into that behavior.
    allow_legacy_full_timeline: bool = False

    # Cross-family target alignment for the TABULAR (tree) arms.
    #
    # A tree row is (feature row t-1 -> target t), so it needs 2 rows of
    # history and can emit a target from the second grid month onward
    # (2010-02). A windowed neural sample needs ``input_len`` rows behind it
    # and cannot emit before 2013-01. On this dataset that is 153,306 tree
    # train targets against 121,176 neural ones -- trees fit on ~26% more
    # supervised targets than the arms they are benchmarked against.
    #
    # True restricts the tree TRAIN extraction to the same first target month
    # the windowed arms get, so both families fit on an identical target set.
    # It does NOT give trees a window and does not touch their features: the
    # feature row is still t-1 with its lag_count lags. Val and test are
    # already aligned by their explicit target_time_start, so this only ever
    # affects training.
    #
    # It does NOT equalize history REACH, which is a separate axis and not
    # fixable this way: a tree row reaches t-13 (t-1 plus 12 lags) while a
    # neural window reaches t-48 (36 rows, each carrying 12 lags). That axis
    # moves with lag_count and must be reported, not silently matched.
    #
    # Default False: flipping it changes what every tree arm trains on and so
    # invalidates existing tree runs. See docs/BEHAVIOR_DIFFERENCES.md.
    tabular_align_targets: bool = False

    # Opt-in matched-information tree arm. Flatten the exact neural window.
    tabular_feature_mode: str = "last_row"
    batch_by_target_month: bool = False
    train_drop_last: bool = False

    aggregate: bool = False  # Test 1: collapse all combos into one summed series

    # Optional inclusive lower bound for TRAIN targets. Earlier rows remain
    # available as history. Sensitivity grids use this to keep the supervised
    # target calendar fixed while input_len and lag_count vary.
    train_target_start: Optional[str] = None

    # Data splits
    train_end: str = "2023-12-01"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-01"
    test_start: str = "2025-01-01"
    test_end: str = "2025-05-01"

    def __post_init__(self) -> None:
        """Reject ambiguous or leakage-prone data contracts up front."""
        if self.source_format not in {"reports", "monthly"}:
            raise ValueError("source_format must be reports or monthly")
        if self.source_format == "monthly" and (
            not self.aggregated_csv or self.target_measure != "legacy_mixed"
            or self.duplicate_policy != "error" or not self.expected_source_fingerprint
        ):
            raise ValueError("monthly source requires a pinned legacy_mixed dataset and duplicate_policy=error")
        for field in (
            "normalize", "filter_sparse", "cohort_from_train_only",
            "allow_legacy_full_timeline",
            "tabular_align_targets", "aggregate", "train_drop_last", "batch_by_target_month",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a boolean")

        if isinstance(self.input_len, bool) or not isinstance(self.input_len, int):
            raise TypeError("input_len must be an integer")
        if isinstance(self.lag_count, bool) or not isinstance(self.lag_count, int):
            raise TypeError("lag_count must be an integer")
        if self.input_len <= 0:
            raise ValueError("input_len must be positive")
        if self.lag_count < 0:
            raise ValueError("lag_count must be non-negative")
        if not math.isfinite(float(self.min_nonzero_frac)):
            raise ValueError("min_nonzero_frac must be finite")
        if not 0.0 <= float(self.min_nonzero_frac) <= 1.0:
            raise ValueError("min_nonzero_frac must be between 0 and 1")

        method = str(self.normalization_method).strip().lower()
        if self.normalize and method != "minmax":
            raise ValueError(
                "normalize=True currently supports only "
                "normalization_method='minmax'"
            )
        # Keep persisted/resolved configuration truthful when scaling is off.
        self.normalization_method = method if self.normalize else "none"

        self.target_measure = str(self.target_measure).strip().lower()
        if self.target_measure not in {"total_vessel", "containerized_vessel", "legacy_mixed"}:
            raise ValueError(
                "target_measure must be total_vessel, containerized_vessel, or legacy_mixed"
            )
        if self.duplicate_policy not in {"error", "legacy_sum"}:
            raise ValueError("duplicate_policy must be error or legacy_sum")
        if self.duplicate_policy == "legacy_sum" and (
            self.target_measure != "legacy_mixed" or not self.expected_source_fingerprint
        ):
            raise ValueError("legacy_sum requires legacy_mixed and a pinned source fingerprint")
        if self.allow_legacy_full_timeline and not (
            self.cohort_from_train_only is False
            and self.target_measure == "legacy_mixed"
            and self.duplicate_policy == "legacy_sum"
            and self.expected_source_fingerprint
        ):
            raise ValueError("allow_legacy_full_timeline requires the explicit pinned legacy reproduction profile")
        self.commodity_universe = str(self.commodity_universe).strip().lower()
        if self.commodity_universe != "all_hs_chapters":
            raise ValueError(
                "commodity_universe currently supports only 'all_hs_chapters'; "
                "an agricultural subset requires an explicit, reviewed HS-code list"
            )

        if self.expected_source_fingerprint is not None:
            expected = str(self.expected_source_fingerprint).lower()
            if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError("expected_source_fingerprint must be a SHA-256 hex digest")
            self.expected_source_fingerprint = expected
        if self.expected_states is not None:
            if (
                not isinstance(self.expected_states, (list, tuple))
                or not self.expected_states
                or any(not isinstance(state, str) or not state.strip()
                       for state in self.expected_states)
            ):
                raise ValueError("expected_states must be a non-empty list of state names")
            normalized_states = [state.strip() for state in self.expected_states]
            if len(set(normalized_states)) != len(normalized_states):
                raise ValueError("expected_states contains duplicates")
            self.expected_states = sorted(normalized_states)
        if self.expected_hs2_codes is not None:
            if (
                not isinstance(self.expected_hs2_codes, (list, tuple))
                or not self.expected_hs2_codes
            ):
                raise ValueError("expected_hs2_codes must be a non-empty list")
            codes = [str(code).strip().zfill(2) for code in self.expected_hs2_codes]
            if any(re.fullmatch(r"\d{2}", code) is None for code in codes):
                raise ValueError("expected_hs2_codes must contain two-digit codes")
            if len(set(codes)) != len(codes):
                raise ValueError("expected_hs2_codes contains duplicates")
            self.expected_hs2_codes = sorted(codes)

        date_fields = (
            "start", "end", "train_end", "val_start", "val_end",
            "test_start", "test_end",
        )
        dates = {name: self._month(name, getattr(self, name)) for name in date_fields}
        source_dates = {}
        for name in ("expected_source_start", "expected_source_end"):
            value = getattr(self, name)
            if value is not None:
                source_dates[name] = self._month(name, value)
        if (
            len(source_dates) == 2
            and source_dates["expected_source_start"] > source_dates["expected_source_end"]
        ):
            raise ValueError("expected_source_start must be on or before expected_source_end")
        if dates["start"] > dates["train_end"]:
            raise ValueError("start must be on or before train_end")
        if dates["train_end"] >= dates["val_start"]:
            raise ValueError("train and validation splits overlap or are out of order")
        if dates["val_start"] > dates["val_end"]:
            raise ValueError("val_start must be on or before val_end")
        if dates["val_end"] >= dates["test_start"]:
            raise ValueError("validation and test splits overlap or are out of order")
        if dates["test_start"] > dates["test_end"]:
            raise ValueError("test_start must be on or before test_end")
        if dates["test_end"] > dates["end"]:
            raise ValueError("test_end must be on or before end")

        if self.train_target_start not in (None, ""):
            target_start = self._month("train_target_start", self.train_target_start)
            if not dates["start"] <= target_start <= dates["train_end"]:
                raise ValueError(
                    "train_target_start must fall between start and train_end"
                )

    @staticmethod
    def _month(name: str, value: Any) -> pd.Timestamp:
        if value is None:
            raise ValueError(f"{name} is required")
        try:
            result = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} is not a valid date: {value!r}") from exc
        if pd.isna(result) or result.tzinfo is not None:
            raise ValueError(f"{name} must be a timezone-naive monthly date")
        if (
            result.day != 1 or result.hour != 0 or result.minute != 0
            or result.second != 0 or result.microsecond != 0
            or result.nanosecond != 0
        ):
            raise ValueError(f"{name} must be the first day of a month")
        return result


class TradeDataPipeline:
    """
    Unified data pipeline for merchandise trade forecasting.

    This class handles:
    - Loading CSV data from imports/exports directories
    - Creating complete monthly timeline grid
    - Adding lag features and seasonal encodings
    - Computing hierarchical normalization stats
    - Creating train/val/test splits
    - Building PyTorch dataloaders
    """

    def __init__(self, config: Union[DataConfig, Dict[str, Any]]):
        """
        Initialize data pipeline.

        Args:
            config: DataConfig instance or config dictionary
        """
        if isinstance(config, dict):
            self.config = DataConfig(**config)
        else:
            self.config = config

        # Internal state
        self.df_full = None
        self.stats = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.feat_cols = None
        self.kept_combos = None
        self.source_fingerprint = None
        self.source_files = []
        self.source_domain = None
        self.df_metric_grid = None

        # Mappings
        self.state2id = {}
        self.comm2id = {}
        self.flow2id = {}

    def parse_month(self, s) -> pd.Timestamp:
        """Parse an explicit month label without heuristic date guessing."""
        if pd.isna(s):
            return pd.NaT
        s = str(s).strip()

        iso = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-01)?", s)
        if iso:
            try:
                return pd.Timestamp(int(iso.group(1)), int(iso.group(2)), 1)
            except ValueError:
                return pd.NaT

        short = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})", s)
        if short:
            month = MONTH_MAP.get(short.group(2).capitalize())
            if month is not None:
                return pd.Timestamp(2000 + int(short.group(1)), month, 1)
        return pd.NaT

    def _numeric_clean(self, series: pd.Series) -> pd.Series:
        """Clean numeric input without turning malformed values into no-trade zeros.

        Empty source cells represent absent reported trade and remain zero, but
        any non-empty value that cannot be parsed is a data error.
        """
        missing = series.isna() | series.astype("string").str.strip().eq("")
        text = series.astype("string").str.replace(",", "", regex=False).str.strip()
        numeric = pd.to_numeric(text, errors="coerce")
        bad = numeric.isna() & ~missing
        if bad.any():
            examples = series[bad].astype(str).drop_duplicates().head(5).tolist()
            raise ValueError(
                f"Malformed numeric values in {series.name!r}: {examples} "
                f"({int(bad.sum())} row(s))"
            )
        numeric = numeric.fillna(0.0).astype(float)
        nonfinite = ~np.isfinite(numeric.to_numpy())
        if nonfinite.any():
            raise ValueError(
                f"Non-finite numeric values in {series.name!r}: "
                f"{int(nonfinite.sum())} row(s)"
            )
        negative = numeric < 0
        if negative.any():
            examples = numeric[negative].drop_duplicates().head(5).tolist()
            raise ValueError(
                f"Negative trade values in {series.name!r}: {examples} "
                f"({int(negative.sum())} row(s))"
            )
        return numeric

    @staticmethod
    def _missing_text(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("")

    def _raw_measure_columns(self, *, exports: bool) -> tuple[str, str]:
        """Return explicitly selected measures, including historical mixed mode."""
        if self.config.target_measure == "legacy_mixed":
            return (
                (EXPORT_VALUE_CONTAINERIZED, EXPORT_WEIGHT_CONTAINERIZED)
                if exports else ("Vessel Value ($US)", "Vessel SWT (kg)")
            )
        if self.config.target_measure == "total_vessel":
            return (
                (EXPORT_VALUE_TOTAL, EXPORT_WEIGHT_TOTAL)
                if exports else (IMPORT_VALUE_TOTAL, IMPORT_WEIGHT_TOTAL)
            )
        return (
            (EXPORT_VALUE_CONTAINERIZED, EXPORT_WEIGHT_CONTAINERIZED)
            if exports else (IMPORT_VALUE_CONTAINERIZED, IMPORT_WEIGHT_CONTAINERIZED)
        )

    def read_folder(self, folder: str, label: str) -> pd.DataFrame:
        """
        Read all CSV files from a folder.

        Args:
            folder: Path to folder containing CSV files
            label: "imports" or "exports"

        Returns:
            Combined DataFrame
        """
        files = sorted(Path(folder).glob("*.csv"), key=lambda path: path.name)
        if not files:
            raise FileNotFoundError(f"No CSV files in {folder}")

        dfs = []
        flow_label = "imports" if label.lower().startswith("import") else "exports"
        for path in files:
            fp = str(path)
            # A skipped malformed row changes the monthly target without any
            # trace. Fail so the source file can be repaired instead.
            raw = path.read_bytes()
            self.source_files.append({
                "flow": flow_label,
                "name": path.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
            # Parse the exact bytes that were hashed; reading the path a second
            # time would leave a race where provenance and training saw
            # different file revisions.
            df = pd.read_csv(io.BytesIO(raw), on_bad_lines="error", dtype=str)

            # Ensure required columns
            for k in ["State", "Commodity", "Country", "Time"]:
                if k not in df.columns:
                    raise ValueError(f"{fp} missing column: {k}")
                missing = self._missing_text(df[k])
                if missing.any():
                    raise ValueError(
                        f"{fp} contains {int(missing.sum())} missing {k} value(s)"
                    )
                df[k] = df[k].astype(str).str.strip()

            df["Time"] = df["Time"].apply(self.parse_month)
            if df["Time"].isna().any():
                raise ValueError(
                    f"{fp} contains {int(df['Time'].isna().sum())} unparseable "
                    "or missing Time value(s)"
                )

            is_export = label.lower().startswith("export")
            value_column, weight_column = self._raw_measure_columns(exports=is_export)
            for column in (value_column, weight_column):
                if column not in df.columns:
                    raise ValueError(
                        f"{fp} missing {self.config.target_measure} column: {column}"
                    )
            df[VALUE_CANON] = self._numeric_clean(df[value_column])
            df[WEIGHT_CANON] = self._numeric_clean(df[weight_column])

            df["Import/Export"] = "Import" if label.lower().startswith("import") else "Export"
            keep = ["State", "Commodity", "Country", "Time", "Import/Export", VALUE_CANON, WEIGHT_CANON]
            dfs.append(df[keep])

        return pd.concat(dfs, ignore_index=True)

    def read_monthly(self) -> pd.DataFrame:
        """Read the frozen monthly targets directly; never repair them on load."""
        path = Path(self.config.aggregated_csv).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        payload = path.read_bytes()
        self.source_files = [{"flow": "monthly", "name": path.name,
                              "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]
        # Parse the very bytes hashed above, avoiding a read/hash race.
        import io
        df = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
        required = KEYS + ["Time", VALUE_CANON, WEIGHT_CANON]
        if set(required).difference(df.columns):
            raise ValueError("Monthly source is missing required target/key columns")
        for col in KEYS:
            if self._missing_text(df[col]).any() or df[col].ne(df[col].str.strip()).any():
                raise ValueError(f"Monthly source contains empty or untrimmed {col}")
        if not df["Import/Export"].isin(["Import", "Export"]).all():
            raise ValueError("Monthly source contains an invalid flow")
        if not df.Time.str.fullmatch(r"\d{4}-\d{2}-01").all():
            raise ValueError("Monthly source Time must be YYYY-MM-01")
        df["Time"] = pd.to_datetime(df.Time, format="%Y-%m-%d", errors="raise")
        if df.duplicated(KEYS + ["Time"]).any():
            raise ValueError("Monthly source contains duplicate monthly keys; refusing to aggregate")
        for col in (VALUE_CANON, WEIGHT_CANON):
            if self._missing_text(df[col]).any():
                raise ValueError(f"Monthly source contains missing {col}; zero filling belongs in the release")
            df[col] = self._numeric_clean(df[col])
        if df.empty:
            raise ValueError("Monthly source is empty")
        months = pd.date_range(df.Time.min(), df.Time.max(), freq="MS")
        if not df.groupby(KEYS).size().eq(len(months)).all():
            raise ValueError("Monthly source has an incomplete monthly calendar")
        return df[required].copy()

    def load_data(self) -> pd.DataFrame:
        """
        Load data from imports and exports directories.

        Returns:
            Combined DataFrame with all trade data
        """
        # Load imports and exports. Reset the manifest so repeated calls on one
        # pipeline describe this invocation rather than accumulating entries.
        self.source_files = []
        if self.config.source_format == "monthly":
            monthly = self.read_monthly()
            df_imports = monthly[monthly["Import/Export"] == "Import"]
            df_exports = monthly[monthly["Import/Export"] == "Export"]
        else:
            df_imports = self.read_folder(self.config.imports_dir, "imports")
            df_exports = self.read_folder(self.config.exports_dir, "exports")

        manifest = json.dumps(
            {"version": 1, "files": self.source_files},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        self.source_fingerprint = hashlib.sha256(manifest).hexdigest()
        expected = self.config.expected_source_fingerprint
        if expected is not None and self.source_fingerprint != expected:
            raise ValueError(
                "Source data fingerprint mismatch: expected "
                f"{expected}, got {self.source_fingerprint}. Refusing to run "
                "the paper protocol on different CSV bytes."
            )

        # Combine
        df_full = pd.concat([df_imports, df_exports], ignore_index=True)

        # Country is the finest retained dimension in these USA Trade Online
        # extracts.  Multiple rows at this exact report grain have no column
        # explaining how they differ, so summing them would silently
        # double-count trade and choosing either row would be arbitrary.
        duplicate_rows = df_full.duplicated(
            KEYS + ["Time"] if self.config.source_format == "monthly"
            else SOURCE_REPORT_KEY_COLUMNS, keep=False
        )
        self.duplicate_audit = {
            "policy": self.config.duplicate_policy,
            "groups": 0, "rows": 0, "excess_rows": 0,
            "conflicting_target_groups": 0,
        }
        if duplicate_rows.any():
            duplicate_groups = (
                df_full.loc[duplicate_rows]
                .groupby(
                    SOURCE_REPORT_KEY_COLUMNS,
                    dropna=False,
                    observed=True,
                )
                .size()
                .sort_values(ascending=False)
            )
            examples = [
                dict(zip(SOURCE_REPORT_KEY_COLUMNS, key))
                for key in duplicate_groups.index[:3]
            ]
            conflicts = df_full.loc[duplicate_rows].groupby(
                SOURCE_REPORT_KEY_COLUMNS, dropna=False, observed=True
            )[[VALUE_CANON, WEIGHT_CANON]].nunique().gt(1).any(axis=1)
            self.duplicate_audit.update({
                "groups": len(duplicate_groups),
                "rows": int(duplicate_rows.sum()),
                "excess_rows": int((duplicate_groups - 1).sum()),
                "conflicting_target_groups": int(conflicts.sum()),
            })
            message = (
                "Source snapshot contains ambiguous duplicate report-grain "
                f"keys: {len(duplicate_groups):,} group(s), "
                f"{int(duplicate_rows.sum()):,} row(s) at "
                f"{SOURCE_REPORT_KEY_COLUMNS}. Examples: {examples}. "
                "Resolve source lineage, or explicitly select pinned legacy reproduction."
            )
            if self.config.duplicate_policy == "error":
                raise ValueError(message)
            warnings.warn(
                message + " Legacy reproduction retains and sums ALL records; "
                "this is not corrective deduplication or validated ground truth.",
                RuntimeWarning, stacklevel=2,
            )

        states = sorted(df_full["State"].unique().tolist())
        commodity_codes = sorted({
            match.group(1)
            for value in df_full["Commodity"].unique()
            if (match := re.match(r"^(\d{2})(?:\s|$)", str(value))) is not None
        })
        malformed_commodities = sorted({
            str(value) for value in df_full["Commodity"].unique()
            if re.match(r"^(\d{2})(?:\s|$)", str(value)) is None
        })
        if malformed_commodities:
            raise ValueError(
                "Commodity labels do not begin with a two-digit HS chapter: "
                f"{malformed_commodities[:5]}"
            )
        source_start = pd.Timestamp(df_full["Time"].min())
        source_end = pd.Timestamp(df_full["Time"].max())
        if self.config.expected_states is not None and states != self.config.expected_states:
            raise ValueError(
                f"Source state domain mismatch: expected {self.config.expected_states}, "
                f"got {states}"
            )
        if (
            self.config.expected_hs2_codes is not None
            and commodity_codes != self.config.expected_hs2_codes
        ):
            raise ValueError(
                "Source HS2 domain mismatch: expected "
                f"{self.config.expected_hs2_codes}, got {commodity_codes}"
            )
        for field, observed in (
            ("expected_source_start", source_start),
            ("expected_source_end", source_end),
        ):
            expected_date = getattr(self.config, field)
            if expected_date is not None and observed != pd.Timestamp(expected_date):
                raise ValueError(
                    f"Source time domain mismatch: {field}={expected_date}, "
                    f"observed {observed.date()}"
                )
        self.source_domain = {
            "states": states,
            "target_measure": self.config.target_measure,
            "duplicate_audit": self.duplicate_audit,
            "commodity_universe": self.config.commodity_universe,
            "hs2_codes": commodity_codes,
            "source_start": source_start.strftime("%Y-%m-%d"),
            "source_end": source_end.strftime("%Y-%m-%d"),
            "grid_start": pd.Timestamp(self.config.start).strftime("%Y-%m-%d"),
            "grid_end": pd.Timestamp(self.config.end).strftime("%Y-%m-%d"),
            "train_end": pd.Timestamp(self.config.train_end).strftime("%Y-%m-%d"),
            "val_start": pd.Timestamp(self.config.val_start).strftime("%Y-%m-%d"),
            "val_end": pd.Timestamp(self.config.val_end).strftime("%Y-%m-%d"),
            "test_start": pd.Timestamp(self.config.test_start).strftime("%Y-%m-%d"),
            "test_end": pd.Timestamp(self.config.test_end).strftime("%Y-%m-%d"),
        }

        print(f"Loaded {len(df_imports)} import rows and {len(df_exports)} export rows")
        print(f"Total combined rows: {len(df_full)}")
        print(f"Source fingerprint: sha256:{self.source_fingerprint}")

        self.df_full = df_full
        return df_full

    def build_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete monthly timeline grid.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with complete monthly timeline
        """
        required = set(KEYS + ["Time", VALUE_CANON, WEIGHT_CANON])
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

        start = pd.Timestamp(self.config.start)
        end = pd.Timestamp(self.config.end)
        in_scope = df[(df["Time"] >= start) & (df["Time"] <= end)].copy()
        if in_scope.empty:
            raise ValueError(
                f"No source rows fall within configured range "
                f"{start.date()}..{end.date()}"
            )
        if in_scope[KEYS + ["Time"]].isna().any().any():
            raise ValueError("Input data contains missing series keys or timestamps")
        values = in_scope[[VALUE_CANON, WEIGHT_CANON]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Input data contains non-finite Value/Weight values")
        if (values < 0).any():
            raise ValueError("Input data contains negative Value/Weight values")

        if self.config.source_format == "monthly":
            # The release already supplies the complete lattice. Do not sum or
            # fill again, including if a caller mutates df_full after loading.
            if df.duplicated(KEYS + ["Time"]).any():
                raise ValueError("Duplicate monthly keys in preaggregated grid")
            months = pd.date_range(start, end, freq="MS")
            expected = len(df[KEYS].drop_duplicates()) * len(months)
            if len(in_scope) != expected or not in_scope.groupby(KEYS).size().eq(len(months)).all():
                raise ValueError("Incomplete preaggregated grid calendar")
            grid = in_scope[KEYS + ["Time", VALUE_CANON, WEIGHT_CANON]].copy()
            grid["month"] = grid.Time.dt.month
            grid["sin_month"] = np.sin(2 * np.pi * grid["month"] / 12)
            grid["cos_month"] = np.cos(2 * np.pi * grid["month"] / 12)
            return grid

        # Only canonical additive measures belong in the monthly aggregation.
        # Extra numeric fields can be ratios, precomputed statistics, or IDs.
        num_cols = [VALUE_CANON, WEIGHT_CANON]
        grouped = (
            in_scope.groupby(KEYS + ["Time"], dropna=False)[num_cols]
            .sum().reset_index()
        )

        # Full monthly timeline
        full_time = pd.date_range(start=self.config.start, end=self.config.end, freq="MS")
        time_df = pd.DataFrame({"Time": full_time, "key": 1})
        # The historical grid retained even series observed only before the
        # experiment start (as all-zero rows within the fixed calendar).
        # Preserve that domain only in the explicit reproduction profile.
        key_source = df if self.config.duplicate_policy == "legacy_sum" else grouped
        unique_keys = key_source[KEYS].drop_duplicates().assign(key=1)

        grid = unique_keys.merge(time_df, on="key", how="outer").drop(columns="key")
        grid = grid.merge(grouped, on=KEYS + ["Time"], how="left")

        for c in num_cols:
            grid[c] = grid[c].fillna(0.0)
        if not np.isfinite(grid[num_cols].to_numpy()).all():
            raise ValueError("Monthly aggregation produced non-finite values")

        # Add month encodings
        grid["month"] = grid["Time"].dt.month
        grid["sin_month"] = np.sin(2 * np.pi * grid["month"] / 12)
        grid["cos_month"] = np.cos(2 * np.pi * grid["month"] / 12)

        return grid

    def filter_sparse_combos(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Drop structurally-sparse (State, Commodity, Flow) combos.

        Keeps only combos whose RAW ``Value`` is nonzero in at least
        ``config.min_nonzero_frac`` of months, measured over the TRAIN
        slice only (``Time <= config.train_end``) and then frozen, so the
        evaluated population is fixed before val/test are looked at.

        This was previously measured over the full timeline on the grounds
        that a population definition is not a fitted statistic. That
        argument does not hold: the full-timeline filter is strictly
        one-directional against test-period attrition -- on this dataset it
        drops 15 combos that the train-only filter keeps and adds none,
        and those 15 are exactly the series that thin out during 2024-2025
        (post-2023 nonzero fraction 0.35-0.94 against a median of 1.00 for
        the kept cohort). Letting future target availability decide which
        series are evaluated is survivorship bias, so the cohort is now
        chosen on train data alone. ``cohort_from_train_only=False`` is now rejected.

        Must be called on the raw grid BEFORE normalization, so the
        nonzero test is against real dollar values and not z-scores.

        MEASUREMENT WINDOW: ``[config.start, config.train_end]``, deliberately
        NOT narrowed to the rows that survive ``add_lags``. The vendored
        ``external/s4/shared_dataloader.build_grid`` does exclude its burn-in
        year, and the two agree on this dataset (both select 918) because that
        copy starts at 2008-01 and drops 2008, landing on the same
        2009-01..2023-12 window this one uses. Narrowing here to
        ``start + lag_count`` would break something more important:
        ``lag_count`` is the swept axis in Exp 3 ({6, 12, 24, 36}), so a
        lag-dependent window would give each sweep cell a DIFFERENT evaluated
        population and make the cells non-comparable. A fixed window keeps the
        cohort invariant to ``lag_count`` and ``input_len``, which is the
        property the sweep needs. Pinned by
        ``tests/test_audit_fixes.py::test_cohort_is_invariant_to_lag_and_window``.

        NOTE: diverges from paper included/, which used every combo. The
        kept set is deterministic given (grid, threshold), so inference
        reconstructs the identical population. Disable via
        ``filter_sparse=False`` for the equivalence test.
        """
        thr = self.config.min_nonzero_frac
        train_only = getattr(self.config, "cohort_from_train_only", True)
        legacy = (self.config.allow_legacy_full_timeline is True
                  and self.config.target_measure == "legacy_mixed"
                  and self.config.duplicate_policy == "legacy_sum"
                  and bool(self.config.expected_source_fingerprint))
        if train_only is not True and not legacy:
            raise ValueError(
                "cohort_from_train_only must be True: selecting series using "
                "validation/test values causes data leakage."
            )
        measure = grid[grid["Time"] <= pd.Timestamp(self.config.train_end)] if train_only else grid
        window = f"train only, <= {pd.Timestamp(self.config.train_end).date()}" if train_only else "explicit legacy full timeline"
        if measure.empty:
            raise ValueError(
                "Sparse-combo filter has no train rows to measure over: "
                f"no grid month is <= train_end={self.config.train_end}."
            )
        nz_frac = measure.groupby(KEYS)[VALUE_CANON].apply(lambda s: float((s != 0).mean()))
        keep_idx = nz_frac[nz_frac >= thr].index
        before = nz_frac.shape[0]
        after = keep_idx.shape[0]
        if after == 0:
            raise ValueError(
                f"Sparse-combo filter dropped ALL {before} combos at "
                f"min_nonzero_frac={thr} over {window}. Lower the threshold."
            )
        keep_df = keep_idx.to_frame(index=False)
        filtered = grid.merge(keep_df, on=KEYS, how="inner")
        self.kept_combos = sorted(map(tuple, keep_df.itertuples(index=False, name=None)))
        print(
            f"Sparse-combo filter (>= {thr:.0%} nonzero raw Value, {window}): "
            f"kept {after}/{before} combos, dropped {before - after}"
        )
        return filtered

    def add_lags(self, df: pd.DataFrame, n_lags: Optional[int] = None) -> pd.DataFrame:
        """
        Add lag features.

        Args:
            df: Input DataFrame
            n_lags: Number of lags (default from config)

        Returns:
            DataFrame with lag features
        """
        if n_lags is None:
            n_lags = self.config.lag_count
        if isinstance(n_lags, bool) or not isinstance(n_lags, int) or n_lags < 0:
            raise ValueError("n_lags must be a non-negative integer")

        df = df.sort_values(KEYS + ["Time"]).copy()
        for k in range(1, n_lags + 1):
            df[f"value_lag_{k}"] = df.groupby(KEYS)[VALUE_CANON].shift(k).fillna(0.0)
            df[f"weight_lag_{k}"] = df.groupby(KEYS)[WEIGHT_CANON].shift(k).fillna(0.0)

        # Burn-in: drop ONLY the leading rows whose lag columns would be
        # fabricated by the .fillna(0.0) above — the rows whose deepest lag
        # reaches before config.start. That is exactly ``n_lags`` months.
        #
        # The further ``input_len`` months of history the first target needs
        # are RESERVED, not deleted. TradeDataset already refuses to emit a
        # target until a full window fits behind it
        # (``range(len(group) - input_len)``), so the window requirement is
        # enforced there and must not be applied a second time here.
        #
        #   contract:  first valid target == start + n_lags + input_len months
        #   (start 2009-01, lag 12, window 36  ->  2013-01)
        #
        # This previously dropped ``input_len + n_lags`` months at YEAR
        # granularity. Two defects followed. (1) It applied the window
        # requirement twice — the first target slid from 2013-01 to 2016-01,
        # costing 36 target months on every series (~27% of the training set).
        # (2) Year granularity is wrong whenever lag_count is not a multiple
        # of 12 or the start is not January: at lag_count=6 it evaluated to a
        # zero-year drop and let rows with fabricated lags into the first
        # windows. Exp 3 sweeps lag_count over {6, 12, 24, 36}, so both paths
        # are live. Pinned by tests/test_burn_in_window.py.
        first_valid = pd.Timestamp(self.config.start) + pd.DateOffset(months=n_lags)
        df = df[df["Time"] >= first_valid]

        return df

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns.

        Returns:
            List of feature column names
        """
        n_lags = self.config.lag_count
        core_feats = (
            [VALUE_CANON, WEIGHT_CANON]
            + [f"value_lag_{k}" for k in range(1, n_lags + 1)]
            + [f"weight_lag_{k}" for k in range(1, n_lags + 1)]
            + ["sin_month", "cos_month"]
        )
        return core_feats

    def compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute per-combo min/max normalization statistics from training data.

        One (min, max) pair per exact (State, Commodity, Flow) combo — no
        borrowing from broader groups. The 95% nonzero filter ensures every
        kept combo has enough real observations to compute meaningful stats.
        """
        s_exact = (
            df.groupby(KEYS)[[VALUE_CANON, WEIGHT_CANON]]
            .agg(["min", "max"])
            .reset_index()
        )
        s_exact.columns = ["State", "Commodity", "Import/Export",
                           "val_min", "val_max", "wt_min", "wt_max"]
        if s_exact.empty:
            raise ValueError(
                "Cannot compute normalization statistics from an empty train split"
            )
        # If min == max (constant series) the range is 0 — normalize to 0.
        # Guard by setting range to 1 so division is safe; denorm recovers
        # the constant by multiplying 0 * 1 + min = min.
        for col in ("val_min", "val_max", "wt_min", "wt_max"):
            s_exact[col] = s_exact[col].fillna(0.0)
        return {"exact": s_exact}

    def normalize_minmax(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        """Apply per-combo min-max normalization: x_norm = (x - min) / (max - min).

        Maps training range to [0, 1]. Val/test values outside training range
        can exceed [0, 1] — that is intentional and correct (no clipping).
        Constant series (max == min) are mapped to 0.
        """
        out = df.sort_values(KEYS + ["Time"]).copy()
        out = out.merge(
            stats["exact"], on=KEYS, how="left", validate="many_to_one"
        )
        stat_cols = ["val_min", "val_max", "wt_min", "wt_max"]
        if out[stat_cols].isna().any().any():
            missing = out.loc[
                out[stat_cols].isna().any(axis=1), KEYS
            ].drop_duplicates()
            raise ValueError(
                "Missing train normalization statistics for series: "
                f"{missing.head(5).to_dict(orient='records')}"
            )

        val_range = (out["val_max"] - out["val_min"]).replace(0, 1.0)
        wt_range  = (out["wt_max"]  - out["wt_min"]).replace(0, 1.0)

        out[VALUE_CANON]  = (out[VALUE_CANON]  - out["val_min"]) / val_range
        out[WEIGHT_CANON] = (out[WEIGHT_CANON] - out["wt_min"])  / wt_range

        out = out.drop(columns=stat_cols)
        if not np.isfinite(out[[VALUE_CANON, WEIGHT_CANON]].to_numpy()).all():
            raise ValueError("Normalization produced non-finite values")
        return out

    def build_id_maps(self, df: pd.DataFrame):
        """
        Build ID mappings for categorical features.

        Args:
            df: DataFrame with categorical features
        """
        states = sorted(df["State"].dropna().unique().tolist())
        comms = sorted(df["Commodity"].dropna().unique().tolist())
        flows = sorted(df["Import/Export"].dropna().unique().tolist())

        self.state2id = {s: i for i, s in enumerate(states)}
        self.comm2id = {c: i for i, c in enumerate(comms)}
        self.flow2id = {f: i for i, f in enumerate(flows)}

        print(f"Built mappings: {len(states)} states, {len(comms)} commodities, {len(flows)} flows")

    def build_combo_maps(self, df: pd.DataFrame):
        """
        Build mapping from (State, Commodity, Import/Export) combinations
        to integer slot indices. Required by combo-window models
        (CrossAttention / FiLM variants) which need a fixed G dimension.

        Mirrors ``paper included/shared_dataloader.py::build_combo_maps``.
        Should be called on the TRAIN slice (or full grid) so val/test
        do not introduce previously-unseen combos.
        """
        unique = (
            df[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
        )
        self.combo2id = {
            (row["State"], row["Commodity"], row["Import/Export"]): i
            for i, row in unique.iterrows()
        }
        self.id2combo = {i: k for k, i in self.combo2id.items()}
        self.num_combos = len(self.combo2id)
        print(f"Built combo map: {self.num_combos} unique (state, commodity, flow) combos")

        # Lattice coordinates for the axial (S4ND-style 2d/3d/4d) combo models:
        # map each combo's g_idx -> (state_idx, commodity_idx, flow_idx) over the
        # categories actually present in the kept combos. lattice_dims = (S,C,Flow).
        states = sorted({k[0] for k in self.combo2id})
        comms = sorted({k[1] for k in self.combo2id})
        flows = sorted({k[2] for k in self.combo2id})
        s2i = {s: i for i, s in enumerate(states)}
        c2i = {c: i for i, c in enumerate(comms)}
        f2i = {f: i for i, f in enumerate(flows)}
        coords = np.zeros((self.num_combos, 3), dtype=np.int64)
        for combo, g in self.combo2id.items():
            coords[g] = (s2i[combo[0]], c2i[combo[1]], f2i[combo[2]])
        self.combo_coords = coords
        self.lattice_dims = (len(states), len(comms), len(flows))

    def _aggregate_grid(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Collapse the whole grid into one summed monthly series (Test 1)."""
        agg = grid.groupby("Time", as_index=False)[[VALUE_CANON, WEIGHT_CANON]].sum()
        agg["State"] = "ALL"
        agg["Commodity"] = "ALL"
        agg["Import/Export"] = "ALL"
        agg["month"] = agg["Time"].dt.month
        agg["sin_month"] = np.sin(2 * np.pi * agg["month"] / 12)
        agg["cos_month"] = np.cos(2 * np.pi * agg["month"] / 12)
        self.kept_combos = [("ALL", "ALL", "ALL")]
        print(f"[aggregate] collapsed to 1 summed series over {len(agg)} months")
        return agg

    def create_splits(self):
        """
        Create train/val/test splits.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        boundaries = [pd.Timestamp(getattr(self.config, name)) for name in (
            "train_end", "val_start", "val_end", "test_start", "test_end"
        )]
        if (any(pd.isna(t) for t in boundaries) or not (
            boundaries[0] < boundaries[1] <= boundaries[2]
            < boundaries[3] <= boundaries[4]
        )):
            raise ValueError(
                "Split dates must satisfy train_end < val_start <= val_end "
                "< test_start <= test_end; overlapping targets cause data leakage."
            )
        if self.config.cohort_from_train_only is not True and not (
            self.config.allow_legacy_full_timeline is True
            and self.config.target_measure == "legacy_mixed"
            and self.config.duplicate_policy == "legacy_sum"
            and self.config.expected_source_fingerprint
        ):
            raise ValueError("cohort_from_train_only must be True to prevent data leakage.")

        if self.df_full is None:
            raise RuntimeError("Must call load_data() first")

        # Build grid (raw monthly panel, no lags yet)
        grid = self.build_grid(self.df_full)

        # Drop structurally-sparse combos on the RAW grid, before stats /
        # normalization, so the nonzero test is on real dollar values.
        if getattr(self.config, "aggregate", False):
            grid = self._aggregate_grid(grid)
        elif self.config.filter_sparse:
            grid = self.filter_sparse_combos(grid)

        # Create splits based on time
        t_train_end = pd.Timestamp(self.config.train_end)
        t_val_start = pd.Timestamp(self.config.val_start)
        t_val_end = pd.Timestamp(self.config.val_end)
        t_test_start = pd.Timestamp(self.config.test_start)
        t_test_end = pd.Timestamp(self.config.test_end)

        # IMPORTANT: normalize BEFORE adding lags. If we add lags first,
        # the lag columns hold raw dollar values (up to ~3e8 for high-volume
        # commodities) while Value/Weight get min-max scaled — feeding two scales
        # of input into the model. RNNs tolerate this via their Linear input
        # projection learning a tiny scale, but the Transformer's attention
        # softmax silently overflows to NaN. Paper-included normalizes first,
        # then computes lags, so lag columns are lags of min-max scaled values
        # (matches `paper included/shared_dataloader.py` ordering).

        # Compute normalization stats on the TRAIN slice only (no leakage)
        train_slice = grid[grid["Time"] <= t_train_end]
        self.stats = self.compute_stats(train_slice)

        # Apply normalization to the FULL grid using train stats.
        if self.config.normalize:
            grid = self.normalize_minmax(grid, self.stats)

            tr = grid[grid["Time"] <= t_train_end]
            print(f"[Train] minmax ~ Value [{tr[VALUE_CANON].min():.3f}, {tr[VALUE_CANON].max():.3f}]"
                  f" | Weight [{tr[WEIGHT_CANON].min():.3f}, {tr[WEIGHT_CANON].max():.3f}]")

        # Scale-free metrics need the complete in-sample history.  Preserve it
        # before lag burn-in removes the first lag_count history-only months.
        self.df_metric_grid = grid.copy()

        # Add lags AFTER normalization, so lag columns hold scaled values.
        grid = self.add_lags(grid)

        # Get feature columns (now that lag columns exist)
        self.feat_cols = self.get_feature_columns()

        # The "split DataFrames" returned here are reporting/inspection
        # views only — they are NOT what get_dataloaders feeds to
        # TradeDataset. See `self.df_full_grid` for the master.
        self.df_full_grid = grid
        df_train = grid[grid["Time"] <= t_train_end].copy()
        df_val = grid[(grid["Time"] >= t_val_start) & (grid["Time"] <= t_val_end)].copy()
        df_test = grid[(grid["Time"] >= t_test_start) & (grid["Time"] <= t_test_end)].copy()

        print(f"Train: {len(df_train)} rows until {t_train_end.date()}")
        print(f"Val: {len(df_val)} rows from {t_val_start.date()} to {t_val_end.date()}")
        print(f"Test: {len(df_test)} rows from {t_test_start.date()} to {t_test_end.date()}")

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        # Store time boundaries so get_dataloaders can build target-time
        # filtered datasets without re-parsing the config.
        self._t_train_end = t_train_end
        self._t_val_start = t_val_start
        self._t_val_end = t_val_end
        self._t_test_start = t_test_start
        self._t_test_end = t_test_end

        return df_train, df_val, df_test

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        generator: Optional[torch.Generator] = None,
        worker_init_fn: Optional[Any] = None,
        combo: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch dataloaders.

        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle_train: Whether to shuffle training data
            generator: torch.Generator for reproducible shuffling. Pass the
                generator returned by ``src.utils.seeding.set_seed(seed)``.
            worker_init_fn: Callable seeded per worker. Pass
                ``src.utils.seeding.seed_worker`` to make NumPy/random
                deterministic inside workers.
            combo: If True, return dataloaders over a ``TradeComboDataset``
                with batches shaped ``(B, L, G, F)`` plus a ``group_mask``.
                Required by ``cross_attention`` / ``film_attention`` model
                variants (which set ``requires_combo_loader = True``).
                If False (default), returns per-series ``TradeDataset``
                loaders shaped ``(B, L, F)``.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.df_train is None:
            raise RuntimeError("Must call create_splits() first")

        # Build ID maps from the train slice (so val/test do not introduce
        # new categorical values).
        self.build_id_maps(self.df_train)
        train_target_start = (
            pd.Timestamp(self.config.train_target_start)
            if self.config.train_target_start is not None else None
        )
        if combo:
            self.build_combo_maps(self.df_train)

        # All three datasets read the FULL grid; the target_time filters
        # decide which samples each split actually emits. This is what
        # lets a 5-month test set still produce real samples whose 36-month
        # windows reach into the train period.
        #
        # Normally `df_full_grid` is set by `create_splits()`. Tests and
        # other callers that bypass `create_splits()` (e.g. by setting
        # `df_train/val/test` directly on the pipeline) fall back to
        # rebuilding the panel from the union of the three splits.
        if getattr(self, "df_full_grid", None) is not None:
            full = self.df_full_grid
            t_train_end = self._t_train_end
            t_val_start = self._t_val_start
            t_val_end = self._t_val_end
            t_test_start = self._t_test_start
            t_test_end = self._t_test_end
        else:
            full = pd.concat(
                [self.df_train, self.df_val, self.df_test],
                ignore_index=True,
            ).drop_duplicates(subset=KEYS + ["Time"]).sort_values(KEYS + ["Time"])
            # Derive boundaries from the actual split dfs rather than config
            # defaults, since callers in this path set the splits directly
            # and may not match config dates (e.g. synthetic test data).
            t_train_end = self.df_train["Time"].max()
            t_val_start = self.df_val["Time"].min() if len(self.df_val) else None
            t_val_end = self.df_val["Time"].max() if len(self.df_val) else None
            t_test_start = self.df_test["Time"].min() if len(self.df_test) else None
            t_test_end = self.df_test["Time"].max() if len(self.df_test) else None
        if combo:
            # Combo path: all-groups-per-window dataset, used by the
            # cross_attention / film_attention model variants.
            ds_kwargs = dict(
                feat_cols=self.feat_cols,
                input_len=self.config.input_len,
                combo2id=self.combo2id,
            )
            train_ds = TradeComboDataset(full, **ds_kwargs,
                                          target_time_start=train_target_start,
                                          target_time_end=t_train_end)
            val_ds = TradeComboDataset(full, **ds_kwargs,
                                        target_time_start=t_val_start,
                                        target_time_end=t_val_end)
            test_ds = TradeComboDataset(full, **ds_kwargs,
                                         target_time_start=t_test_start,
                                         target_time_end=t_test_end)
            active_collate_fn = combo_collate_fn
        else:
            train_ds = TradeDataset(
                full, self.feat_cols, self.config.input_len,
                self.state2id, self.comm2id, self.flow2id,
                target_time_start=train_target_start, target_time_end=t_train_end,
            )
            # Cross-family target alignment (DataConfig.tabular_align_targets).
            # Tree arms read this off the dataset in
            # tabular_common.extract_from_trade_dataset and clamp their TRAIN
            # targets to it. Derived from the windowed samples themselves --
            # the earliest target the neural arms were actually offered --
            # rather than by date arithmetic, so it stays correct for any
            # start / lag_count / input_len. No effect on the windowed arms.
            if getattr(self.config, "tabular_align_targets", False):
                train_ds.tabular_target_time_start = (
                    min(s["target_time"] for s in train_ds.samples)
                    if train_ds.samples else None
                )
            val_ds = TradeDataset(
                full, self.feat_cols, self.config.input_len,
                self.state2id, self.comm2id, self.flow2id,
                target_time_start=t_val_start, target_time_end=t_val_end,
            )
            test_ds = TradeDataset(
                full, self.feat_cols, self.config.input_len,
                self.state2id, self.comm2id, self.flow2id,
                target_time_start=t_test_start, target_time_end=t_test_end,
            )
            active_collate_fn = collate_fn

        # Create dataloaders. The collate fn (chosen above) stacks per-sample
        # dicts and renames keys to the plural form the trainer/model expects.
        # Without it the default collate produces singular keys and downstream
        # code silently breaks.
        #
        # Clamp oversized batches for predictable memory/use reporting.  The
        # loader deliberately keeps its final partial batch: dropping it
        # silently changes the supervised population (Exp 1 had 132 targets
        # at batch 64, so neural arms omitted four while trees used all 132).
        if len(train_ds) == 0:
            raise ValueError(
                "Train dataset produced 0 samples — check input_len and the "
                "split boundaries."
            )
        # Inference may request a larger evaluation batch. Training still
        # contains exactly one target month in the combo representation.
        train_bs = 1 if self.config.batch_by_target_month and combo else batch_size
        if len(train_ds) < train_bs:
            print(
                f"[warn] train dataset has {len(train_ds)} samples < "
                f"batch_size {batch_size}; clamping train batch to "
                f"{len(train_ds)}"
            )
            train_bs = len(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=train_bs, shuffle=shuffle_train,
            drop_last=self.config.train_drop_last, num_workers=num_workers, pin_memory=True,
            collate_fn=active_collate_fn, generator=generator,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=active_collate_fn, worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=active_collate_fn, worker_init_fn=worker_init_fn,
        )

        if self.config.batch_by_target_month:
            if self.config.train_drop_last:
                raise ValueError("Month batching must not drop training targets")
            if combo:
                if train_bs != 1:
                    raise ValueError("Month batching requires combo batch_size=1")
            else:
                TargetMonthBatchSampler = _load_sibling(
                    "src.data.sampling", "data/sampling.py"
                ).TargetMonthBatchSampler
                train_loader = DataLoader(
                    train_ds,
                    batch_sampler=TargetMonthBatchSampler(
                        train_ds.samples, shuffle=shuffle_train, generator=generator,
                    ),
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=active_collate_fn, worker_init_fn=worker_init_fn,
                    generator=generator,
                )

        if self.config.tabular_feature_mode not in ("last_row", "window"):
            raise ValueError("tabular_feature_mode must be last_row or window")
        for dataset in (train_ds, val_ds, test_ds):
            dataset.tabular_feature_mode = self.config.tabular_feature_mode

        mode = "combo" if combo else "per-series"
        print(f"Created dataloaders ({mode}): {len(train_loader)} train, "
              f"{len(val_loader)} val, {len(test_loader)} test batches")

        return train_loader, val_loader, test_loader

    def save_artifacts(self, out_dir: str):
        """
        Save data artifacts for later use.

        Args:
            out_dir: Output directory
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        artifacts = out_path / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        # Save stats
        if self.stats:
            stats_json = {"exact": self.stats["exact"].to_dict(orient="list")}
            (artifacts / "series_stats.json").write_text(json.dumps(stats_json, indent=2))

        # Save ID maps
        maps = {
            "state2id": self.state2id,
            "comm2id": self.comm2id,
            "flow2id": self.flow2id,
            "num_states": len(self.state2id),
            "num_commodities": len(self.comm2id),
            "num_flows": len(self.flow2id),
        }
        (artifacts / "id_maps.json").write_text(json.dumps(maps, ensure_ascii=False, indent=2))

        # Save features
        if self.feat_cols:
            (artifacts / "features.json").write_text(json.dumps({"feat_cols": self.feat_cols}, indent=2))

        # Save the kept-combo population (sparse filter) for transparency.
        if self.kept_combos is not None:
            (artifacts / "kept_combos.json").write_text(json.dumps({
                "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                "min_nonzero_frac": self.config.min_nonzero_frac,
                "cohort_from_train_only": getattr(
                    self.config, "cohort_from_train_only", True
                ),
                "cohort_measured_through": (
                    self.config.train_end
                    if getattr(self.config, "cohort_from_train_only", True)
                    else self.config.end
                ),
                "num_combos": len(self.kept_combos),
                "combos": [list(c) for c in self.kept_combos],
                "source_fingerprint": self.source_fingerprint,
                "source_domain": self.source_domain,
            }, indent=2))

        if self.source_fingerprint is not None:
            (artifacts / "source_data.json").write_text(json.dumps({
                "fingerprint_algorithm": "sha256-manifest-v1",
                "source_fingerprint": self.source_fingerprint,
                "source_domain": self.source_domain,
                "files": self.source_files,
            }, indent=2))

        # Save normalized data
        if self.df_train is not None:
            self.df_train.to_csv(artifacts / "train_normalized.csv", index=False)
        if self.df_val is not None:
            self.df_val.to_csv(artifacts / "val_normalized.csv", index=False)
        if self.df_test is not None:
            self.df_test.to_csv(artifacts / "test_normalized.csv", index=False)

        print(f"Saved artifacts to {artifacts}")


class TradeDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.

    Each sample contains a sequence of length ``input_len`` with features
    for a specific (State, Commodity, Flow) combination, plus the target
    row at ``window_end + 1``.

    The dataset is always fed the FULL normalized monthly grid. Which
    samples are emitted is controlled by ``target_time_start`` /
    ``target_time_end``: a sample is emitted iff the target row's ``Time``
    falls within ``[target_time_start, target_time_end]``. This is the
    only correct way to construct val/test sets — a 12-month val slice
    cannot fit a 36-month window inside itself, so val/test windows MUST
    reach back into earlier splits for historical context.

    Train: pass ``target_time_end=train_end`` to keep targets within train.
    Val:   pass ``[val_start, val_end]``; windows reach back into train.
    Test:  pass ``[test_start, test_end]``; windows reach back into train+val.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        input_len: int,
        state2id: Dict[str, int],
        comm2id: Dict[str, int],
        flow2id: Dict[str, int],
        target_time_start: Optional[pd.Timestamp] = None,
        target_time_end: Optional[pd.Timestamp] = None,
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with time series data (typically the full
                normalized grid, not a temporally pre-sliced subset).
            feat_cols: List of feature column names
            input_len: Input sequence length
            state2id: State to ID mapping
            comm2id: Commodity to ID mapping
            flow2id: Flow to ID mapping
            target_time_start: Optional inclusive lower bound on target
                time. If None, no lower bound.
            target_time_end: Optional inclusive upper bound on target
                time. If None, no upper bound.
        """
        self.df = df.sort_values(KEYS + ["Time"]).copy()
        self.feat_cols = feat_cols
        self.input_len = input_len
        self.state2id = state2id
        self.comm2id = comm2id
        self.flow2id = flow2id
        self.target_time_start = target_time_start
        self.target_time_end = target_time_end

        # Build sequences
        self.samples = self._build_sequences()

    def _build_sequences(self) -> List[Dict[str, Any]]:
        """Build sequences from DataFrame, filtered by target-time range."""
        samples = []

        grouped = self.df.groupby(KEYS)
        for (state, comm, flow), group in grouped:
            if len(group) < self.input_len + 1:
                continue  # Skip groups that are too short

            group = group.sort_values("Time").reset_index(drop=True)
            times = group["Time"].to_numpy()

            # Create sliding windows
            for i in range(len(group) - self.input_len):
                target_idx = i + self.input_len
                target_time = times[target_idx]

                # Target-time filter — this is what makes val/test work:
                # a 5-month test set still emits samples whose 36-month
                # windows reach into train.
                if self.target_time_start is not None and target_time < self.target_time_start:
                    continue
                if self.target_time_end is not None and target_time > self.target_time_end:
                    continue

                # Input sequence
                x = group.iloc[i : i + self.input_len]

                # Target (next time step)
                y = group.iloc[target_idx]

                samples.append({
                    "x_numeric": torch.tensor(x[self.feat_cols].values, dtype=torch.float32),
                    "state_id": self.state2id[state],
                    "comm_id": self.comm2id[comm],
                    "flow_id": self.flow2id[flow],
                    "target_value": torch.tensor(y[VALUE_CANON], dtype=torch.float32),
                    "target_weight": torch.tensor(y[WEIGHT_CANON], dtype=torch.float32),
                    "target_time": pd.Timestamp(target_time),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "x_numeric": sample["x_numeric"],
            "state_id": torch.tensor(sample["state_id"], dtype=torch.long),
            "comm_id": torch.tensor(sample["comm_id"], dtype=torch.long),
            "flow_id": torch.tensor(sample["flow_id"], dtype=torch.long),
            "target_value": sample["target_value"],
            "target_weight": sample["target_weight"],
            "target_time": sample["target_time"],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of samples

    Returns:
        Batched tensors
    """
    x_numeric = torch.stack([item["x_numeric"] for item in batch])
    state_ids = torch.stack([item["state_id"] for item in batch])
    comm_ids = torch.stack([item["comm_id"] for item in batch])
    flow_ids = torch.stack([item["flow_id"] for item in batch])
    target_value = torch.stack([item["target_value"] for item in batch])
    target_weight = torch.stack([item["target_weight"] for item in batch])

    return {
        "x_numeric": x_numeric,
        "state_ids": state_ids,
        "comm_ids": comm_ids,
        "flow_ids": flow_ids,
        "target_value": target_value,
        "target_weight": target_weight,
        # Timestamps can't be stacked into a tensor; pass through as a list.
        "target_time": [item["target_time"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Combo-window dataset for CA-only / CA+FiLM model variants.
#
# Mirrors `paper included/shared_dataloader.py::ComboWindowDataset`. Where
# `TradeDataset` produces per-series windows of shape (L, F), this dataset
# produces all-groups-at-once windows of shape (L, G, F) plus a per-group
# validity mask. Combo models need this shape so they can cross-attend
# across groups at each timestep.
#
# Same target_time_start / target_time_end filtering convention as
# TradeDataset, so val/test samples reach back into earlier splits.
# ---------------------------------------------------------------------------

class TradeComboDataset(Dataset):
    """All-groups-per-window dataset for combo-window models.

    Each sample is one window ending at ``target_time``, containing
    features for EVERY (state, commodity, flow) combination. Combinations
    that have no data at a given timestep are zero-filled and flagged
    in the mask.

    Sample dict:
        x_numeric:    (L, G, F) -- window of features per group per timestep
        group_mask:   (G,)      -- 1 if combo is valid at the target time
        target_value: (G,)      -- target Value per combo
        target_weight:(G,)      -- target Weight per combo
        target_time:  scalar    -- the target Timestamp (kept for inference)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        input_len: int,
        combo2id: Dict[tuple, int],
        target_time_start: Optional[pd.Timestamp] = None,
        target_time_end: Optional[pd.Timestamp] = None,
    ):
        self.feat_cols = feat_cols
        self.input_len = input_len
        self.combo2id = combo2id
        self.G = len(combo2id)
        self.F = len(feat_cols)

        df = df.sort_values(KEYS + ["Time"]).copy()
        self.times: List[pd.Timestamp] = sorted(df["Time"].unique().tolist())
        self.T = len(self.times)
        self.time2idx = {t: i for i, t in enumerate(self.times)}

        self.value_idx = feat_cols.index(VALUE_CANON)
        self.weight_idx = feat_cols.index(WEIGHT_CANON)

        # Pre-build (T, G, F) data tensor and (T, G) mask.
        self.data = np.zeros((self.T, self.G, self.F), dtype=np.float32)
        self.mask = np.zeros((self.T, self.G), dtype=np.float32)
        for combo_key, gdf in df.groupby(KEYS, sort=False):
            if combo_key not in combo2id:
                continue
            g_idx = combo2id[combo_key]
            gdf = gdf.sort_values("Time")
            t_indices = gdf["Time"].map(self.time2idx).to_numpy(dtype=np.int64)
            valid = ~np.isnan(t_indices.astype(float))
            t_indices = t_indices[valid]
            feats = gdf[feat_cols].to_numpy(dtype=np.float32)[valid]
            keep = np.isfinite(feats).all(axis=1)
            t_indices = t_indices[keep]
            feats = feats[keep]
            self.data[t_indices, g_idx, :] = feats
            self.mask[t_indices, g_idx] = 1.0

        # Sample indices: target time index t such that a full window
        # [t-L, t) fits AND target_time lies in the requested range.
        self.target_time_start = target_time_start
        self.target_time_end = target_time_end
        self.samples: List[int] = []
        for t in range(self.T):
            if t < self.input_len:
                continue
            tt = self.times[t]
            if target_time_start is not None and tt < target_time_start:
                continue
            if target_time_end is not None and tt > target_time_end:
                continue
            self.samples.append(t)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.samples[idx]
        x = self.data[t - self.input_len : t, :, :]   # (L, G, F)
        target_row = self.data[t, :, :]               # (G, F)
        group_mask = self.mask[t, :]                  # (G,)
        return {
            "x_numeric": torch.from_numpy(x.copy()),
            "group_mask": torch.from_numpy(group_mask.copy()),
            "target_value": torch.from_numpy(target_row[:, self.value_idx].copy()),
            "target_weight": torch.from_numpy(target_row[:, self.weight_idx].copy()),
            "target_time": self.times[t],
        }


def combo_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate combo-window samples.

    Stacks (L, G, F) -> (B, L, G, F); (G,) -> (B, G); etc. Keeps
    ``target_time`` as a list of Timestamps (not a tensor) for inference.
    """
    return {
        "x_numeric":    torch.stack([b["x_numeric"]    for b in batch]),
        "group_mask":   torch.stack([b["group_mask"]   for b in batch]),
        "target_value": torch.stack([b["target_value"] for b in batch]),
        "target_weight":torch.stack([b["target_weight"] for b in batch]),
        "target_time":  [b["target_time"] for b in batch],
    }
