"""Plugin example: custom Engine that uses Google Trends data."""

from __future__ import annotations

import pandas as pd

from equitylens import DataSource, Engine, EngineOutput


class GoogleTrendsEngine(Engine):
    """Example engine using search volume as a sentiment proxy.

    This is a demonstration of the plugin interface. In production,
    you would fetch real Google Trends data via pytrends or similar.
    """

    name = "google_trends"

    def __init__(self, lookback_weeks: int = 12):
        self.lookback_weeks = lookback_weeks

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        # Fetch alternative data if the DataSource provides it
        trends = data.alternative(universe, kind="google_trends", as_of=as_of)
        if trends is None or trends.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"warning": "no trends data available"},
            )

        # Example logic: rising search volume = increasing attention
        # In practice you would normalize, detrend, etc.
        trends = trends.set_index("ticker")
        scores = trends["trend_slope"].rank(pct=True) * 100

        return EngineOutput(
            scores=scores,
            raw_metrics=trends,
            confidence=pd.Series(0.5, index=scores.index),  # low confidence for alt data
            metadata={"lookback_weeks": self.lookback_weeks, "as_of": as_of},
        )
