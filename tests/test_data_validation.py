import pandas as pd
import pytest

from src.data.dataloader import DataConfig, TradeDataPipeline


def _pipeline():
    return TradeDataPipeline(DataConfig())


def test_numeric_clean_allows_missing_but_rejects_malformed_values():
    series = pd.Series(["1,234", None, "", " 5.5 "], name="Value")
    assert _pipeline()._numeric_clean(series).tolist() == [1234.0, 0.0, 0.0, 5.5]

    malformed = pd.Series(["1", "not-a-number"], name="Value")
    with pytest.raises(ValueError, match="Malformed numeric"):
        _pipeline()._numeric_clean(malformed)
