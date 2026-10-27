"""Offline tests for the frozen official-Census source downloader."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from scripts.download_census_ground_truth import (
    ALL_MONTHS,
    CENSUS_ENDPOINT,
    FLOWS,
    HS2_CODES,
    HS2_LABELS,
    HS2_UNIVERSE_SHA256,
    SOURCE_END,
    SOURCE_START,
    STATES,
    CensusClient,
    CensusDownloadError,
    FlowSpec,
    StateSpec,
    _source_csv_bytes,
    build_request,
    months_for_year,
    publish_artifacts,
    source_columns,
    validate_and_zero_fill_batch,
)


def _flow(name: str) -> FlowSpec:
    return next(flow for flow in FLOWS if flow.name == name)


def _row(
    flow: FlowSpec,
    *,
    state: str = "Arkansas",
    commodity: str = "01 Live Animals",
    month: str = "January 2008",
    value: object = "12",
    weight: object = "34",
) -> dict[str, object]:
    return {
        "state_label": state,
        "commodities_label": commodity,
        "countries_label": "World Total",
        "time_id": month,
        flow.response_value_field: value,
        flow.response_weight_field: weight,
    }


def _response(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "message": None,
        "data": rows,
        "totalPages": 1,
        "totalVerticalPages": 1,
        "totalNodes": len(rows),
        "totalQueries": 1,
        "totalRows": 0,
        "csvFile": "",
    }


def test_scope_is_exactly_frozen_and_year_flow_requests_are_small():
    assert [state.code for state in STATES] == [
        "AR", "IL", "IA", "LA", "MN", "MS", "MO", "WI"
    ]
    assert [state.name for state in STATES] == [
        "Arkansas", "Illinois", "Iowa", "Louisiana", "Minnesota",
        "Mississippi", "Missouri", "Wisconsin",
    ]
    assert (SOURCE_START, SOURCE_END) == ("2008-01", "2025-05")
    assert len(ALL_MONTHS) == 209
    assert ALL_MONTHS[0] == "200801"
    assert ALL_MONTHS[-1] == "202505"
    assert len(months_for_year(2008)) == 12
    assert len(months_for_year(2025)) == 5

    assert len(HS2_CODES) == 98
    assert HS2_CODES[:3] == ("01", "02", "03")
    assert HS2_CODES[-3:] == ("97", "98", "99")
    assert "77" not in HS2_CODES
    assert HS2_LABELS["01"] == "01 Live Animals"
    assert HS2_LABELS["99"] == "99 Special Import Provisions, Nesoi"
    assert (
        HS2_UNIVERSE_SHA256
        == "4225ada3e41d21a6f3993def7cd64f530e303bb2cd7ead4612454e7fb0635969"
    )

    for flow in FLOWS:
        request = build_request(flow, months_for_year(2008))
        assert request["dataset"] == flow.dataset
        assert request["commodities"] == list(HS2_CODES)
        assert request["states"] == [state.code for state in STATES]
        assert request["countries"] == ["1"]
        assert request["exportData"] == "Y"
        assert "page" not in request
        assert "itemsPerPage" not in request
        assert len(request["states"]) * len(request["commodities"]) * len(
            request["time"]
        ) * len(request["measures"]) == 18_816


def test_batch_response_is_validated_then_missing_keys_are_zero_filled():
    flow = _flow("imports")
    states = (StateSpec("AR", "Arkansas"), StateSpec("IA", "Iowa"))
    result = validate_and_zero_fill_batch(
        flow=flow,
        months=("200801", "200802"),
        response=_response([_row(flow)]),
        states=states,
        commodity_codes=("01", "02"),
    )

    assert result.returned_rows == 1
    assert result.zero_filled_rows == 7
    assert len(result.values) == 8
    assert result.values[("AR", "01", "200801")] == (12, 34)
    assert result.values[("IA", "02", "200802")] == (0, 0)

    live_shape = _response([_row(flow)])
    live_shape["totalVerticalPages"] = 0
    assert validate_and_zero_fill_batch(
        flow=flow,
        months=("200801",),
        response=live_shape,
        states=(StateSpec("AR", "Arkansas"),),
        commodity_codes=("01",),
    ).returned_rows == 1


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda flow, response: response["data"].append(_row(flow)),
            "duplicate row key",
        ),
        (
            lambda flow, response: response["data"][0].update(
                {"state_label": "California"}
            ),
            "unexpected state_label",
        ),
        (
            lambda flow, response: response["data"][0].update(
                {flow.response_value_field: "not-a-number"}
            ),
            "is malformed",
        ),
        (
            lambda flow, response: response.update({"totalPages": 2}),
            "was not complete",
        ),
        (
            lambda flow, response: response.update({"message": "failed"}),
            "report message",
        ),
    ],
)
def test_batch_response_fails_closed_on_ambiguous_or_incomplete_data(mutate, match):
    flow = _flow("exports")
    response = _response([_row(flow)])
    mutate(flow, response)
    response["totalNodes"] = len(response["data"])
    with pytest.raises(CensusDownloadError, match=match):
        validate_and_zero_fill_batch(
            flow=flow,
            months=("200801",),
            response=response,
            states=(StateSpec("AR", "Arkansas"),),
            commodity_codes=("01",),
        )


def test_source_csv_is_deterministic_and_uses_official_flow_specific_headers():
    state = StateSpec("AR", "Arkansas")
    values = {
        ("AR", commodity, month): (0, 0)
        for commodity in HS2_CODES
        for month in ALL_MONTHS
    }
    values[("AR", "01", "200801")] = (12, 34)

    for flow in FLOWS:
        first = _source_csv_bytes(flow=flow, state=state, values=values)
        second = _source_csv_bytes(flow=flow, state=state, values=dict(
            reversed(list(values.items()))
        ))
        assert first == second
        lines = first.decode("utf-8").splitlines()
        assert lines[0].split(",") == list(source_columns(flow))
        assert len(lines) == len(HS2_CODES) * len(ALL_MONTHS) + 1
        assert lines[1] == "Arkansas,01 Live Animals,World Total,2008-01-01,12,34"

    assert source_columns(_flow("imports"))[-2:] == (
        "Vessel Customs Value (Gen) ($US)", "Vessel SWT (Gen) (kg)"
    )
    assert source_columns(_flow("exports"))[-2:] == (
        "Vessel Total Exports Value ($US)",
        "Vessel Total Exports SWT (kg)",
    )


def test_publish_is_atomic_idempotent_and_never_replaces_different_bytes(tmp_path):
    destination = tmp_path / "census"
    artifacts = {
        Path("imports/Arkansas.csv"): b"header\nrow\n",
        Path("download_receipt.json"): b"{}\n",
    }
    assert publish_artifacts(destination, artifacts) == "Wrote"
    assert publish_artifacts(destination, artifacts) == "Verified unchanged"
    with pytest.raises(CensusDownloadError, match="already exists and differs"):
        publish_artifacts(
            destination,
            {**artifacts, Path("download_receipt.json"): b'{"changed":true}\n'},
        )


class _Headers(dict):
    def get_content_type(self) -> str:
        return self["Content-Type"].split(";", 1)[0]


class _FakeResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.headers = _Headers({"Content-Type": "application/json; charset=utf-8"})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


def test_http_client_retries_transient_status_and_sends_no_authentication_header():
    calls = []
    waits = []
    payload = {"test": True}
    response_body = json.dumps(_response([{"placeholder": True}])).encode()

    def opener(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                CENSUS_ENDPOINT,
                503,
                "busy",
                {"Retry-After": "0"},
                io.BytesIO(b"busy"),
            )
        return _FakeResponse(response_body)

    client = CensusClient(opener=opener, sleeper=waits.append)
    result = client.post(payload)
    assert result.body == response_body
    assert waits == [0.0]
    assert len(calls) == 2
    sent_headers = {key.lower(): value for key, value in calls[-1][0].header_items()}
    assert sent_headers["content-type"] == "application/json"
    assert sent_headers["accept"] == "application/json"
    assert "authorization" not in sent_headers
    assert json.loads(calls[-1][0].data) == payload
