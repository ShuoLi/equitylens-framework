"""Reversal engine — short-term mean reversion on recent losers.

Longs recent 1-week / 1-month losers (expected to bounce); recent winners
score low (exit signal). Returns are vol-normalized so a -3% move on a
50%-vol stock doesn't score the same as -3% on a 15%-vol stock.

Designed to run alongside the momentum engine: the two naturally offset
(momentum chases trends, reversal fades them).

References: Lehmann (1990); Jegadeesh (1990); Da, Liu & Schaumburg (2014).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput
from equitylens.utils import percentile_rank, winsorize

TRADING_DAYS_WEEK = 5
TRADING_DAYS_MONTH = 22


class ReversalEngine(Engine):
    """Short-term reversal scorer.

    Parameters
    ----------
    lookback_days : int
        Calendar days of price history to fetch (default 150, covers the
        1-month window plus the EWMA volatility estimate).
    """

    name = "reversal"

    def __init__(self, lookback_days: int = 150):
        self.lookback_days = lookback_days

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        start = pd.Timestamp(as_of) - pd.Timedelta(days=self.lookback_days)
        prices = data.prices(universe, start=str(start), end=as_of)
        if prices.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "no price data"},
            )

        px = prices.pivot_table(index="date", columns="ticker", values="adj_close")
        px = px.reindex(columns=universe)

        rows = [
            {"ticker": t, **self._compute_metrics(px[t].dropna())} for t in universe
        ]
        metrics = pd.DataFrame(rows).set_index("ticker")

        if metrics.empty:
            return EngineOutput(scores=pd.Series(dtype=float))

        # Vol-normalized reversal, winsorized then inverted percentile-ranked:
        # a bigger (more negative) normalized loss = higher score.
        scored = pd.DataFrame(index=metrics.index)
        for col in ("rev_1w_norm", "rev_1m_norm"):
            valid = metrics[col].dropna()
            if not valid.empty:
                scored[col] = percentile_rank(winsorize(metrics[col]), higher_is_better=False)

        if scored.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                raw_metrics=metrics,
                metadata={"error": "insufficient price history"},
            )

        scores = scored.mean(axis=1, skipna=True).clip(0, 100).dropna()

        # Confidence: fraction of lookback with valid prices
        conf = px.notna().mean().reindex(universe).fillna(0)

        return EngineOutput(
            scores=scores,
            raw_metrics=metrics,
            confidence=conf,
            metadata={
                "as_of": as_of,
                "n_scored": len(scores),
            },
        )

    def _compute_metrics(self, s: pd.Series) -> dict:
        out = {
            "rev_1w": np.nan,
            "rev_1m": np.nan,
            "rev_1w_norm": np.nan,
            "rev_1m_norm": np.nan,
        }
        if len(s) < TRADING_DAYS_WEEK + 1:
            return out

        end = float(s.iloc[-1])
        wk_ago = float(s.iloc[-TRADING_DAYS_WEEK - 1]) if len(s) >= TRADING_DAYS_WEEK + 1 else np.nan
        mon_ago = float(s.iloc[-TRADING_DAYS_MONTH - 1]) if len(s) >= TRADING_DAYS_MONTH + 1 else np.nan

        if not np.isnan(wk_ago) and wk_ago > 0:
            out["rev_1w"] = (end - wk_ago) / wk_ago
        if not np.isnan(mon_ago) and mon_ago > 0:
            out["rev_1m"] = (end - mon_ago) / mon_ago

        # EWMA vol (60-day half-life), annualized
        if len(s) >= TRADING_DAYS_MONTH:
            daily_ret = s.pct_change().dropna()
            ewma_vol = float(daily_ret.ewm(com=60).std().iloc[-1]) * np.sqrt(252)
            if ewma_vol > 0:
                if not np.isnan(out["rev_1w"]):
                    out["rev_1w_norm"] = out["rev_1w"] / ewma_vol
                if not np.isnan(out["rev_1m"]):
                    out["rev_1m_norm"] = out["rev_1m"] / ewma_vol
        return out
