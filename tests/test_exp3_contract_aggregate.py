"""Exp 3 expected contracts must match what per-lane sweep training records."""
from src.data.dataloader import DataConfig
from src.utils.config import get_data_config
from src.utils.exp3_protocol import coordinates, expected_contract, resolved_config


def test_resolved_config_leaves_aggregate_to_the_dataconfig_default():
    cfg = resolved_config(*coordinates()[0])
    assert "aggregate" not in cfg["data"]
    data = get_data_config(cfg)
    fields = DataConfig.__dataclass_fields__
    assert DataConfig(**{k: v for k, v in data.items() if k in fields}).aggregate is False


def test_expected_contract_keeps_per_lane_identity():
    contract = expected_contract(*coordinates()[0])
    assert contract["identity"]["aggregate"] is False
    assert "aggregate" not in contract["data_config"]
