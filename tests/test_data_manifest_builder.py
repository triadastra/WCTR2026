import json
from pathlib import Path

import pytest

from scripts.build_data_manifest import _canonical_json, build_manifest
from src.census_scope import CENSUS_STATE_ITEMS, HS2_CODES


def _receipt(path: Path) -> None:
    path.write_text(json.dumps({
        "scope": {
            "states": [
                {"code": code, "name": name} for code, name in CENSUS_STATE_ITEMS
            ],
            "hs2_codes": list(HS2_CODES),
            "source_start": "2008-01-01",
            "source_end": "2025-05-01",
        }
    }))


def test_manifest_hashes_exact_flow_state_file_set_and_receipt(tmp_path: Path):
    for flow in ("imports", "exports"):
        folder = tmp_path / "data" / flow
        folder.mkdir(parents=True)
        for _, state in CENSUS_STATE_ITEMS:
            (folder / f"{state}.csv").write_text(f"{flow},{state}\n")
    receipt = tmp_path / "receipt.json"
    _receipt(receipt)

    first = build_manifest(tmp_path, receipt)
    second = build_manifest(tmp_path, receipt)
    assert _canonical_json(first) == _canonical_json(second)
    assert len(first["files"]) == 16
    assert len(first["source_fingerprint"]) == 64
    assert first["source_receipt"]["sha256"]

    (tmp_path / "data" / "imports" / "extra.csv").write_text("extra")
    with pytest.raises(ValueError, match="exactly"):
        build_manifest(tmp_path, receipt)


def test_manifest_rejects_receipt_with_expanded_scope(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    _receipt(receipt)
    data = json.loads(receipt.read_text())
    data["scope"]["states"].append({"code": "CA", "name": "California"})
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="eight-state"):
        build_manifest(tmp_path, receipt)
