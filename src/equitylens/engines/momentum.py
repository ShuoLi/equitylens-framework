"""Momentum engine — cross-sectional price momentum."""

from __future__ import annotations

import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput


class MomentumEngine(Engine):
    """Cross-sectional momentum scorer.

    Scores each stock by its risk-adjusted return over `lookback_days`,
    normalized to 0-100 percentile within the universe.

    Parameters
    ----------
    lookback_days : int
        Momentum lookback window (default 126 = ~6 months).
    skip_days : int
        Skip most recent N days (momentum crash avoidance, default 5).
    """

    name = "momentum"

    def __init__(self, lookback_days: int = 126, skip_days: int = 5):
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        # Fetch extra data to cover the lookback window
        start = pd.Timestamp(as_of) - pd.Timedelta(days=int(self.lookback_days * 1.5))
        prices = data.prices(universe, start=str(start), end=as_of)

        if prices.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "no price data"},
            )

        # Pivot to wide format: rows=date, cols=ticker, values=adj_close
        px = prices.pivot_table(index="date", columns="ticker", values="adj_close")

        # Momentum: return over lookback, skipping recent days
        end_idx = px.index.get_loc(px.index[px.index <= as_of][-1])
        lookback_start = max(0, end_idx - self.lookback_days - self.skip_days)
        skip_start = max(0, end_idx - self.skip_days)

        past_px = px.iloc[lookback_start]
        recent_px = px.iloc[skip_start]
        current_px = px.iloc[end_idx]

        # Total return over the window, avoiding the most recent `skip_days`
        momentum = (recent_px / past_px - 1).dropna()
        volatility = px.pct_change().iloc[lookback_start:end_idx].std().dropna()

        # Risk-adjusted momentum (Sharpe-like)
        sharpe = (momentum / volatility).replace([float("inf"), -float("inf")], 0).dropna()

        # Normalize to 0-100 percentile
        if len(sharpe) == 0:
            scores = pd.Series(dtype=float)
        else:
            scores = sharpe.rank(pct=True) * 100

        raw = pd.DataFrame(
            {
                "momentum_return": momentum,
                "volatility": volatility,
                "risk_adj_momentum": sharpe,
            }
        )

        # Confidence: fraction of lookback period with valid prices
        conf = px.notna().mean().reindex(universe).fillna(0)

        return EngineOutput(
            scores=scores,
            raw_metrics=raw,
            confidence=conf,
            metadata={
                "lookback_days": self.lookback_days,
                "skip_days": self.skip_days,
                "as_of": as_of,
                "n_scored": len(scores),
            },
        )
