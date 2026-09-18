"""Tests for core plugin interfaces."""

import pandas as pd
import pytest

from equitylens import Engine, EngineOutput, Pipeline
from equitylens.plugin import DataSource


class MockDataSource(DataSource):
    """Minimal data source for testing."""

    def prices(self, tickers, start, end):
        dates = pd.date_range(start, end, freq="D")
        data = {
            "date": list(dates) * len(tickers),
            "ticker": [t for t in tickers for _ in dates],
            "open": [100.0] * len(dates) * len(tickers),
            "high": [101.0] * len(dates) * len(tickers),
            "low": [99.0] * len(dates) * len(tickers),
            "close": [100.0] * len(dates) * len(tickers),
            "volume": [1_000_000] * len(dates) * len(tickers),
            "adj_close": [100.0] * len(dates) * len(tickers),
        }
        return pd.DataFrame(data)

    def fundamentals(self, tickers, as_of=None):
        return pd.DataFrame(
            {
                "ticker": tickers,
                "pe_forward": [15.0] * len(tickers),
                "roe": [0.20] * len(tickers),
                "revenue_growth": [0.10] * len(tickers),
            }
        )

    def alternative(self, tickers, kind, as_of=None):
        return None


class DummyEngine(Engine):
    name = "dummy"

    def compute(self, data, universe, as_of):
        return EngineOutput(
            scores=pd.Series(50.0, index=universe),
            raw_metrics=pd.DataFrame({"x": [1] * len(universe)}, index=universe),
            confidence=pd.Series(1.0, index=universe),
            metadata={"as_of": as_of},
        )


class TestPipeline:
    def test_register_and_run(self):
        ds = MockDataSource()
        p = Pipeline(ds)
        p.register(DummyEngine(), weight=1.0)
        scores = p.run(["AAPL", "MSFT"], as_of="2025-09-18")
        assert len(scores) == 2
        assert scores["AAPL"] == 50.0

    def test_duplicate_engine_raises(self):
        ds = MockDataSource()
        p = Pipeline(ds)
        p.register(DummyEngine(), weight=1.0)
        with pytest.raises(ValueError):
            p.register(DummyEngine(), weight=0.5)

    def test_no_engines_raises(self):
        ds = MockDataSource()
        p = Pipeline(ds)
        with pytest.raises(ValueError):
            p.run(["AAPL"], as_of="2025-09-18")
