#!/usr/bin/env python3
"""Download the frozen Experiment 1 source panel from U.S. Census.

The downloader uses the report endpoint called by Census USA Trade Online.
It deliberately fixes the scientific scope in code: eight states, World
Total, every two-digit code exposed by the HS state report, both trade flows,
and 2008-01 through 2025-05.  The experiment grid still begins in 2009-01;
the preceding year is retained only as source/history coverage.

Requests are small, independently validated year/flow batches.  Census omits
some state/HS/month combinations with no reported vessel trade, so the output
contains an explicit zero row for every missing combination.  Nothing is
published until every batch has succeeded and passed validation.

The endpoint does not require a Census API key, account, cookie, or bearer
token.  It is an official Census service but is the USA Trade Online web
application's report endpoint rather than a versioned public Census Data API;
the audit receipt records that distinction and the exact request contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.census_scope import (  # noqa: E402
    CENSUS_STATE_ITEMS,
    HS2_CODES,
    HS2_LABELS,
    HS2_UNIVERSE_SHA256,
)

CENSUS_ENDPOINT = (
    "https://usatradeonline.census.gov/api/reports/build-totals-report"
)
CENSUS_PRODUCT_URL = (
    "https://www.census.gov/foreign-trade/reference/products/catalog/"
    "usatradeonline.html"
)
CENSUS_HISTORY_URL = (
    "https://www.census.gov/foreign-trade/statistics/historical/hs.html"
)
CENSUS_USER_GUIDE_URL = (
    "https://www.census.gov/foreign-trade/data/"
    "USA_Trade_Online_Reimagined_User_Guide.pdf"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "census_ground_truth_download"
DEFAULT_USER_AGENT = "WCTR2026-CensusGroundTruthDownloader/1.0"

SOURCE_START = "2008-01"
SOURCE_END = "2025-05"
EXPERIMENT_GRID_START = "2009-01"
WORLD_TOTAL_CODE = "1"
WORLD_TOTAL_LABEL = "World Total"

DIMENSION_COLUMNS = (
    "State",
    "Commodity",
    "Country",
    "Time",
)

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
UTO_DOCUMENTED_REPORT_CELL_LIMIT = 2_000_000


class CensusDownloadError(RuntimeError):
    """Raised when Census returns an incomplete or invalid report."""


@dataclass(frozen=True)
class StateSpec:
    code: str
    name: str


@dataclass(frozen=True)
class FlowSpec:
    name: str
    dataset: str
    response_value_field: str
    response_weight_field: str


@dataclass(frozen=True)
class HttpResult:
    body: bytes
    content_type: str


@dataclass(frozen=True)
class BatchResult:
    values: Mapping[tuple[str, str, str], tuple[int, int]]
    returned_rows: int
    zero_filled_rows: int
    total_nodes: int | None
    total_queries: int | None


STATES = tuple(StateSpec(code, name) for code, name in CENSUS_STATE_ITEMS)

FLOWS = (
    FlowSpec(
        name="imports",
        dataset="hs_state_i",
        response_value_field="Vessel Customs Value (Gen) ($US)",
        response_weight_field="Vessel SWT (Gen) (kg)",
    ),
    FlowSpec(
        name="exports",
        dataset="hs_state_e",
        response_value_field="Vessel Total Exports Value ($US)",
        response_weight_field="Vessel Total Exports SWT (kg)",
    ),
)

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable, human-readable JSON bytes without a wall-clock field."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def iter_months(start: str = SOURCE_START, end: str = SOURCE_END) -> tuple[str, ...]:
    """Return an inclusive sequence of YYYYMM Census report identifiers."""
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    if start_dt > end_dt:
        raise ValueError("start month must not follow end month")
    months: list[str] = []
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        months.append(f"{year:04d}{month:02d}")
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return tuple(months)


ALL_MONTHS = iter_months()


def months_for_year(year: int) -> tuple[str, ...]:
    return tuple(month for month in ALL_MONTHS if int(month[:4]) == year)


def build_request(flow: FlowSpec, months: Sequence[str]) -> dict[str, Any]:
    """Build one complete, unpaginated USA Trade Online report request."""
    if not months:
        raise ValueError("a Census request must include at least one month")
    years = {month[:4] for month in months}
    if len(years) != 1:
        raise ValueError("Census requests must be batched by one calendar year")
    if any(month not in ALL_MONTHS for month in months):
        raise ValueError("request month is outside the frozen source calendar")
    return {
        "commodities": list(HS2_CODES),
        "countries": [WORLD_TOTAL_CODE],
        "states": [state.code for state in STATES],
        "time": list(months),
        "measures": [
            flow.response_value_field,
            flow.response_weight_field,
        ],
        "dataset": flow.dataset,
        "paginateBy": "commodities",
        "exportData": "Y",
    }


def source_columns(flow: FlowSpec) -> tuple[str, ...]:
    """Return loader-compatible official measure columns for one flow."""
    return (
        *DIMENSION_COLUMNS,
        flow.response_value_field,
        flow.response_weight_field,
    )


def _header_content_type(headers: Any) -> str:
    if headers is None:
        return ""
    if hasattr(headers, "get_content_type"):
        return str(headers.get_content_type())
    raw = headers.get("Content-Type", "") if hasattr(headers, "get") else ""
    return str(raw).split(";", 1)[0].strip().lower()


class CensusClient:
    """Small retrying HTTP client with a strict response-size bound."""

    RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        endpoint: str = CENSUS_ENDPOINT,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 90.0,
        max_attempts: int = 4,
        retry_base_seconds: float = 2.0,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds must be non-negative")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if "\n" in user_agent or "\r" in user_agent:
            raise ValueError("user_agent must not contain a newline")
        self.endpoint = endpoint
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.max_response_bytes = max_response_bytes
        self.opener = opener
        self.sleeper = sleeper

    def _retry_delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 60.0)
            except ValueError:
                pass
        return self.retry_base_seconds * (2 ** (attempt - 1))

    def post(self, request_payload: Mapping[str, Any]) -> HttpResult:
        body = canonical_json_bytes(request_payload)
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
        )

        for attempt in range(1, self.max_attempts + 1):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    raw = response.read(self.max_response_bytes + 1)
                    if status != 200:
                        raise CensusDownloadError(
                            f"Census returned unexpected HTTP status {status}"
                        )
                    if len(raw) > self.max_response_bytes:
                        raise CensusDownloadError(
                            "Census response exceeded the configured byte limit"
                        )
                    content_type = _header_content_type(
                        getattr(response, "headers", None)
                    )
                    if content_type != "application/json":
                        raise CensusDownloadError(
                            "Census returned non-JSON content type "
                            f"{content_type or '<missing>'!r}"
                        )
                    return HttpResult(body=raw, content_type=content_type)
            except urllib.error.HTTPError as exc:
                error_body = exc.read(512).decode("utf-8", errors="replace")
                if exc.code not in self.RETRYABLE_STATUS or attempt == self.max_attempts:
                    raise CensusDownloadError(
                        f"Census HTTP {exc.code}: {error_body!r}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                self.sleeper(self._retry_delay(attempt, retry_after))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.max_attempts:
                    raise CensusDownloadError(
                        f"Census request failed after {attempt} attempts: {exc}"
                    ) from exc
                self.sleeper(self._retry_delay(attempt))

        raise AssertionError("unreachable retry loop")


def decode_response(body: bytes) -> Mapping[str, Any]:
    try:
        decoded = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CensusDownloadError("Census response is not valid UTF-8") from exc
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise CensusDownloadError("Census response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise CensusDownloadError("Census response root must be a JSON object")
    return payload


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise CensusDownloadError(f"{field} is not a reported integer: {value!r}")
    text = str(value).strip().replace(",", "")
    if not text:
        raise CensusDownloadError(f"{field} is blank in a returned row")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise CensusDownloadError(f"{field} is malformed: {value!r}") from exc
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise CensusDownloadError(
            f"{field} must be a finite non-negative integer, got {value!r}"
        )
    return int(number)


def _response_month(value: Any) -> str:
    if not isinstance(value, str):
        raise CensusDownloadError(f"time_id must be text, got {value!r}")
    for pattern in ("%B %Y", "%b %Y"):
        try:
            parsed = datetime.strptime(value.strip(), pattern)
            return f"{parsed.year:04d}{parsed.month:02d}"
        except ValueError:
            continue
    raise CensusDownloadError(f"unrecognized Census time_id: {value!r}")


def _commodity_code(value: Any) -> str:
    if not isinstance(value, str):
        raise CensusDownloadError(
            f"commodities_label must be text, got {value!r}"
        )
    match = re.match(r"^(\d{2})(?:\s|$)", value.strip())
    if match is None:
        raise CensusDownloadError(
            f"cannot recover HS2 code from commodities_label {value!r}"
        )
    return match.group(1)


def validate_and_zero_fill_batch(
    *,
    flow: FlowSpec,
    months: Sequence[str],
    response: Mapping[str, Any],
    states: Sequence[StateSpec] = STATES,
    commodity_codes: Sequence[str] = HS2_CODES,
) -> BatchResult:
    """Validate one response and fill all omitted state/HS2/month rows with 0."""
    message = response.get("message")
    if message not in (None, ""):
        raise CensusDownloadError(f"Census report message: {message!r}")
    data = response.get("data")
    if not isinstance(data, list):
        raise CensusDownloadError("Census response is missing its data array")

    if "totalPages" in response:
        total_pages = response["totalPages"]
        if (
            isinstance(total_pages, bool)
            or not isinstance(total_pages, int)
            or total_pages != 1
        ):
            raise CensusDownloadError(
                f"totalPages={total_pages!r}; exportData=Y was not complete"
            )
    if "totalVerticalPages" in response:
        # The live service returns 0 for full multi-commodity export responses
        # and 1 for some one-row responses.  Values above 1 mean the response
        # contract has changed or the result may be vertically truncated.
        vertical_pages = response["totalVerticalPages"]
        if (
            isinstance(vertical_pages, bool)
            or not isinstance(vertical_pages, int)
            or vertical_pages not in (0, 1)
        ):
            raise CensusDownloadError(
                f"totalVerticalPages={vertical_pages!r}; exportData=Y was not complete"
            )

    total_nodes = response.get("totalNodes")
    if total_nodes is not None:
        if isinstance(total_nodes, bool) or not isinstance(total_nodes, int):
            raise CensusDownloadError("totalNodes must be an integer when present")
        if total_nodes != len(data):
            raise CensusDownloadError(
                f"totalNodes={total_nodes} but data contains {len(data)} rows"
            )
    total_queries = response.get("totalQueries")
    if total_queries is not None and (
        isinstance(total_queries, bool) or not isinstance(total_queries, int)
    ):
        raise CensusDownloadError("totalQueries must be an integer when present")

    state_by_name = {state.name: state.code for state in states}
    expected_keys = {
        (state.code, commodity, month)
        for state in states
        for commodity in commodity_codes
        for month in months
    }
    observed: dict[tuple[str, str, str], tuple[int, int]] = {}

    required = {
        "state_label",
        "commodities_label",
        "countries_label",
        "time_id",
        flow.response_value_field,
        flow.response_weight_field,
    }
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise CensusDownloadError(f"Census data row {index} is not an object")
        missing = required.difference(row)
        if missing:
            raise CensusDownloadError(
                f"Census data row {index} is missing fields: {sorted(missing)}"
            )
        state_name = row["state_label"]
        if state_name not in state_by_name:
            raise CensusDownloadError(
                f"unexpected state_label in row {index}: {state_name!r}"
            )
        if row["countries_label"] != WORLD_TOTAL_LABEL:
            raise CensusDownloadError(
                "expected countries_label='World Total', got "
                f"{row['countries_label']!r}"
            )
        key = (
            state_by_name[state_name],
            _commodity_code(row["commodities_label"]),
            _response_month(row["time_id"]),
        )
        if key not in expected_keys:
            raise CensusDownloadError(f"Census returned out-of-scope row key {key}")
        if key in observed:
            raise CensusDownloadError(f"Census returned duplicate row key {key}")
        observed[key] = (
            _nonnegative_integer(
                row[flow.response_value_field], field=flow.response_value_field
            ),
            _nonnegative_integer(
                row[flow.response_weight_field], field=flow.response_weight_field
            ),
        )

    if not observed:
        raise CensusDownloadError("Census returned zero rows for a full-year batch")
    if not any(value or weight for value, weight in observed.values()):
        raise CensusDownloadError("Census full-year batch contains no positive measure")

    filled = {
        key: observed.get(key, (0, 0))
        for key in sorted(expected_keys)
    }
    return BatchResult(
        values=filled,
        returned_rows=len(observed),
        zero_filled_rows=len(expected_keys) - len(observed),
        total_nodes=total_nodes,
        total_queries=total_queries,
    )


def _canonical_batch_rows(
    values: Mapping[tuple[str, str, str], tuple[int, int]],
) -> bytes:
    lines = [
        f"{state},{commodity},{month},{value},{weight}\n"
        for (state, commodity, month), (value, weight) in sorted(values.items())
    ]
    return "".join(lines).encode("ascii")


def download_all(
    client: CensusClient,
    *,
    delay_seconds: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    dict[str, dict[tuple[str, str, str], tuple[int, int]]],
    list[dict[str, Any]],
]:
    """Download every frozen year/flow batch, returning only validated data."""
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")
    values_by_flow: dict[
        str, dict[tuple[str, str, str], tuple[int, int]]
    ] = {flow.name: {} for flow in FLOWS}
    receipts: list[dict[str, Any]] = []
    first_request = True

    for year in range(2008, 2026):
        months = months_for_year(year)
        for flow in FLOWS:
            if not first_request and delay_seconds:
                time.sleep(delay_seconds)
            first_request = False
            request_payload = build_request(flow, months)
            if progress is not None:
                progress(f"Downloading {flow.name} {year} ({len(months)} month(s))")
            http = client.post(request_payload)
            response = decode_response(http.body)
            batch = validate_and_zero_fill_batch(
                flow=flow,
                months=months,
                response=response,
            )
            overlap = values_by_flow[flow.name].keys() & batch.values.keys()
            if overlap:
                raise CensusDownloadError(
                    f"overlapping year batches returned {len(overlap)} duplicate keys"
                )
            values_by_flow[flow.name].update(batch.values)
            normalized = _canonical_batch_rows(batch.values)
            receipts.append({
                "flow": flow.name,
                "year": year,
                "months": list(months),
                "request": request_payload,
                "request_sha256": sha256_bytes(canonical_json_bytes(request_payload)),
                "response_bytes": len(http.body),
                "response_sha256": sha256_bytes(http.body),
                "normalized_rows_sha256": sha256_bytes(normalized),
                "returned_rows": batch.returned_rows,
                "zero_filled_rows": batch.zero_filled_rows,
                "emitted_rows": len(batch.values),
                "total_nodes": batch.total_nodes,
                "total_queries": batch.total_queries,
            })

    expected_per_flow = len(STATES) * len(HS2_CODES) * len(ALL_MONTHS)
    for flow in FLOWS:
        observed = len(values_by_flow[flow.name])
        if observed != expected_per_flow:
            raise CensusDownloadError(
                f"{flow.name} assembled {observed} rows; expected {expected_per_flow}"
            )
    return values_by_flow, receipts


def _source_csv_bytes(
    *,
    flow: FlowSpec,
    state: StateSpec,
    values: Mapping[tuple[str, str, str], tuple[int, int]],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(source_columns(flow))
    expected = 0
    for commodity in HS2_CODES:
        for month in ALL_MONTHS:
            expected += 1
            try:
                value, weight = values[(state.code, commodity, month)]
            except KeyError as exc:
                raise CensusDownloadError(
                    f"missing assembled source row {(state.code, commodity, month)}"
                ) from exc
            writer.writerow([
                state.name,
                HS2_LABELS[commodity],
                WORLD_TOTAL_LABEL,
                f"{month[:4]}-{month[4:]}-01",
                value,
                weight,
            ])
    if expected != len(HS2_CODES) * len(ALL_MONTHS):
        raise AssertionError("unexpected source row count")
    return output.getvalue().encode("utf-8")


def build_artifacts(
    values_by_flow: Mapping[
        str, Mapping[tuple[str, str, str], tuple[int, int]]
    ],
    batch_receipts: Sequence[Mapping[str, Any]],
) -> dict[Path, bytes]:
    """Render source CSVs and their deterministic audit receipt."""
    artifacts: dict[Path, bytes] = {}
    file_receipts: list[dict[str, Any]] = []
    rows_per_file = len(HS2_CODES) * len(ALL_MONTHS)
    for flow in FLOWS:
        if flow.name not in values_by_flow:
            raise CensusDownloadError(f"missing assembled flow {flow.name!r}")
        for state in STATES:
            relative = Path(flow.name) / f"{state.name}.csv"
            payload = _source_csv_bytes(
                flow=flow,
                state=state,
                values=values_by_flow[flow.name],
            )
            artifacts[relative] = payload
            file_receipts.append({
                "path": relative.as_posix(),
                "flow": flow.name,
                "state_code": state.code,
                "state_name": state.name,
                "rows": rows_per_file,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            })

    max_full_year_rows = len(STATES) * len(HS2_CODES) * 12
    receipt = {
        "schema_version": 1,
        "source": {
            "publisher": "U.S. Census Bureau",
            "product": "USA Trade Online",
            "endpoint": CENSUS_ENDPOINT,
            "endpoint_status": (
                "official USA Trade Online application endpoint; not a "
                "versioned Census Data API contract"
            ),
            "authentication": "none (no API key, login, cookie, or bearer token)",
            "product_url": CENSUS_PRODUCT_URL,
            "historical_coverage_url": CENSUS_HISTORY_URL,
            "user_guide_url": CENSUS_USER_GUIDE_URL,
        },
        "scope": {
            "states": [
                {"code": state.code, "name": state.name} for state in STATES
            ],
            "country_code": WORLD_TOTAL_CODE,
            "country_label": WORLD_TOTAL_LABEL,
            "hs_level": 2,
            "hs2_codes": list(HS2_CODES),
            "hs2_labels": dict(HS2_LABELS),
            "hs2_universe_sha256": HS2_UNIVERSE_SHA256,
            "source_start": f"{SOURCE_START}-01",
            "source_end": f"{SOURCE_END}-01",
            "experiment_grid_start": f"{EXPERIMENT_GRID_START}-01",
            "flows": [flow.name for flow in FLOWS],
            "months": len(ALL_MONTHS),
            "rows_per_flow": len(STATES) * len(HS2_CODES) * len(ALL_MONTHS),
        },
        "request_contract": {
            "batching": "one POST per calendar year and flow",
            "batch_count": len(batch_receipts),
            "max_full_year_dimension_rows": max_full_year_rows,
            "max_full_year_measure_cells": max_full_year_rows * 2,
            "documented_report_cell_limit": UTO_DOCUMENTED_REPORT_CELL_LIMIT,
            "exportData": "Y",
            "paginateBy": "commodities",
        },
        "measure_mapping": {
            flow.name: {
                "dataset": flow.dataset,
                "response_value": flow.response_value_field,
                "response_weight": flow.response_weight_field,
                "output_value": flow.response_value_field,
                "output_weight": flow.response_weight_field,
            }
            for flow in FLOWS
        },
        "output": {
            "format": "UTF-8 CSV with LF line endings",
            "columns_by_flow": {
                flow.name: list(source_columns(flow)) for flow in FLOWS
            },
            "sort": ["flow", "state", "Commodity", "Time"],
            "missing_combinations": (
                "explicit zero value and zero weight after each returned batch "
                "passes schema, domain, uniqueness, and completeness checks"
            ),
            "files": sorted(file_receipts, key=lambda item: item["path"]),
        },
        "batches": list(batch_receipts),
    }
    artifacts[Path("download_receipt.json")] = canonical_json_bytes(receipt)
    return artifacts


def _existing_tree_matches(output_dir: Path, artifacts: Mapping[Path, bytes]) -> bool:
    expected_paths = set(artifacts)
    actual_paths = {
        path.relative_to(output_dir)
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        return False
    return all((output_dir / path).read_bytes() == payload for path, payload in artifacts.items())


def publish_artifacts(output_dir: Path, artifacts: Mapping[Path, bytes]) -> str:
    """Publish atomically; never replace an existing differing directory."""
    output_dir = output_dir.resolve()
    protected = {
        (ROOT / "data").resolve(),
        (ROOT / "data" / "imports").resolve(),
        (ROOT / "data" / "exports").resolve(),
    }
    if output_dir in protected:
        raise CensusDownloadError(
            f"refusing to write directly over protected tracked data path {output_dir}"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise CensusDownloadError(f"output path is not a directory: {output_dir}")
        if _existing_tree_matches(output_dir, artifacts):
            return "Verified unchanged"
        raise CensusDownloadError(
            f"output directory already exists and differs: {output_dir}; "
            "choose a new --output-dir after reviewing the existing receipt"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ))
    try:
        for relative, payload in sorted(
            artifacts.items(), key=lambda item: item[0].as_posix()
        ):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return "Wrote"


def _scope_summary() -> str:
    years = len({month[:4] for month in ALL_MONTHS})
    full_rows = len(STATES) * len(HS2_CODES) * len(ALL_MONTHS) * len(FLOWS)
    return (
        f"{len(STATES)} states, {len(HS2_CODES)} HS2 codes, "
        f"{len(ALL_MONTHS)} months, {len(FLOWS)} flows, {years * len(FLOWS)} "
        f"year/flow requests, {full_rows:,} output rows"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "New destination directory. Existing differing directories are never "
            "overwritten (default: %(default)s)."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Polite delay between successful Census requests (default: 1).",
    )
    parser.add_argument(
        "--contact",
        help="Optional email/URL appended to User-Agent to identify the operator.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the immutable request scope without making network requests.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Frozen Census scope: {_scope_summary()}")
    print(
        f"Source {SOURCE_START}..{SOURCE_END}; experiment grid remains "
        f"{EXPERIMENT_GRID_START}..{SOURCE_END}"
    )
    if args.dry_run:
        return 0

    user_agent = DEFAULT_USER_AGENT
    if args.contact:
        if "\n" in args.contact or "\r" in args.contact:
            print("ERROR: --contact must not contain a newline", file=sys.stderr)
            return 2
        user_agent += f" (contact: {args.contact.strip()})"
    try:
        client = CensusClient(
            user_agent=user_agent,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
            retry_base_seconds=args.retry_base_seconds,
        )
        values, batches = download_all(
            client,
            delay_seconds=args.delay_seconds,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        artifacts = build_artifacts(values, batches)
        action = publish_artifacts(args.output_dir, artifacts)
    except (CensusDownloadError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    receipt = artifacts[Path("download_receipt.json")]
    print(
        f"{action} {args.output_dir.resolve()} with 16 CSVs; "
        f"receipt sha256:{sha256_bytes(receipt)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
