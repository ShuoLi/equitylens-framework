"""Tests for reference engines."""

import pandas as pd
import pytest

from equitylens.data_sources import YFinanceSource
from equitylens.engines import MomentumEngine, QualityEngine, ValueEngine
from equitylens.plugin import DataSource


class MockDataSource(DataSource):
    """Controllable mock for engine tests."""

    def __init__(self):
        self._prices = None
        self._fundamentals = None

    def set_prices(self, df):
        self._prices = df

    def set_fundamentals(self, df):
        self._fundamentals = df

    def prices(self, tickers, start, end):
        if self._prices is not None:
            return self._prices
        return pd.DataFrame()

    def fundamentals(self, tickers, as_of=None):
        if self._fundamentals is not None:
            return self._fundamentals
        return pd.DataFrame()


class TestMomentumEngine:
    def test_basic_scoring(self):
        # Build synthetic price data with clear momentum leaders
        dates = pd.date_range("2025-01-01", "2025-09-18", freq="D")
        tickers = ["WINNER", "LOSER"]
        frames = []
        for t in tickers:
            if t == "WINNER":
                px = [100 * (1.001 ** i) for i in range(len(dates))]  # steady uptrend
            else:
                px = [100 * (0.999 ** i) for i in range(len(dates))]  # steady downtrend
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": t,
                        "open": px,
                        "high": px,
                        "low": px,
                        "close": px,
                        "volume": [1e6] * len(dates),
                        "adj_close": px,
                    }
                )
            )
        prices = pd.concat(frames, ignore_index=True)

        ds = MockDataSource()
        ds.set_prices(prices)

        engine = MomentumEngine(lookback_days=126, skip_days=5)
        out = engine.compute(ds, tickers, as_of="2025-09-18")

        assert not out.scores.empty
        assert out.scores["WINNER"] > out.scores["LOSER"]
        assert out.scores.between(0, 100).all()


class TestQualityEngine:
    def test_basic_scoring(self):
        fund = pd.DataFrame(
            {
                "ticker": ["HIGH_Q", "LOW_Q"],
                "roe": [0.30, 0.05],
                "roa": [0.15, 0.02],
                "gross_margin": [0.60, 0.20],
                "operating_margin": [0.40, 0.05],
                "revenue_growth": [0.15, -0.05],
                "earnings_growth": [0.20, -0.10],
                "debt_to_equity": [0.30, 2.0],
                "current_ratio": [2.5, 0.8],
            }
        )
        ds = MockDataSource()
        ds.set_fundamentals(fund)

        engine = QualityEngine()
        out = engine.compute(ds, ["HIGH_Q", "LOW_Q"], as_of="2025-09-18")

        assert not out.scores.empty
        assert out.scores["HIGH_Q"] > out.scores["LOW_Q"]


class TestValueEngine:
    def test_forward_pe_preferred(self):
        fund = pd.DataFrame(
            {
                "ticker": ["CHEAP", "EXPENSIVE"],
                "pe_forward": [10.0, 50.0],
                "pe_trailing": [12.0, 60.0],
                "peg": [0.8, 3.0],
                "ps": [1.5, 10.0],
                "pb": [2.0, 15.0],
            }
        )
        ds = MockDataSource()
        ds.set_fundamentals(fund)

        engine = ValueEngine(use_forward_pe=True)
        out = engine.compute(ds, ["CHEAP", "EXPENSIVE"], as_of="2025-09-18")

        assert not out.scores.empty
        assert out.scores["CHEAP"] > out.scores["EXPENSIVE"]
