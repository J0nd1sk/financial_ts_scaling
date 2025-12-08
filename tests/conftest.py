"""Pytest fixtures for data pipeline tests."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def mock_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic OHLCV data for all yfinance downloads."""
    dates = pd.date_range(start="1993-01-29", end=pd.Timestamp(datetime.now().date()), freq="D")
    values = pd.Series(range(len(dates)), index=dates, dtype=float)
    data = pd.DataFrame(
        {
            "Open": 100 + values,
            "High": 100.5 + values,
            "Low": 99.5 + values,
            "Close": 100.2 + values,
            "Volume": (1_000_000 + values * 1_000).astype(int),
        },
        index=dates,
    )
    data.index.name = "Date"

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, period: str = "max") -> pd.DataFrame:
            return data.copy()

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda symbol: FakeTicker(symbol))

