"""Quality engine — fundamental quality scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput


class QualityEngine(Engine):
    """Score stocks on fundamental quality.

    Uses ROE, ROA, gross margin, operating margin, low debt, and revenue growth.
    Each metric is cross-sectionally ranked to 0-100, then averaged.

    Parameters
    ----------
    weights : dict, optional
        Override metric weights. Defaults balance profitability + growth + safety.
    require_positive_earnings : bool
        If True, stocks with negative earnings get score = 0.
    """

    name = "quality"

    DEFAULT_WEIGHTS = {
        "roe": 0.20,
        "roa": 0.10,
        "gross_margin": 0.15,
        "operating_margin": 0.15,
        "revenue_growth": 0.15,
        "earnings_growth": 0.10,
        "debt_to_equity_inv": 0.10,  # inverse: lower debt = higher score
        "current_ratio": 0.05,
    }

    def __init__(self, weights: dict | None = None, require_positive_earnings: bool = False):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.require_positive_earnings = require_positive_earnings

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        fund = data.fundamentals(universe, as_of=as_of)
        if fund.empty:
            return EngineOutput(scores=pd.Series(dtype=float))

        fund = fund.set_index("ticker")

        # Derive inverse debt-to-equity (so higher = better)
        if "debt_to_equity" in fund.columns:
            fund["debt_to_equity_inv"] = 1 / (1 + fund["debt_to_equity"].clip(lower=0))

        # Rank each metric to 0-100
        ranked = pd.DataFrame(index=fund.index)
        for metric in self.weights:
            if metric not in fund.columns:
                continue
            col = fund[metric].replace([np.inf, -np.inf], np.nan)
            ranked[metric] = col.rank(pct=True) * 100

        # Weighted composite
        available = [m for m in self.weights if m in ranked.columns]
        if not available:
            return EngineOutput(scores=pd.Series(dtype=float))

        w = pd.Series({m: self.weights[m] for m in available})
        w = w / w.sum()

        scores = (ranked[available] * w).sum(axis=1)
        scores = scores.dropna()

        # Optional: zero out negative earners
        if self.require_positive_earnings and "earnings_growth" in fund.columns:
            neg = fund["earnings_growth"] < 0
            scores = scores.mask(neg.reindex(scores.index).fillna(False), 0)

        # Confidence: fraction of metrics available
        conf = fund[available].notna().mean(axis=1)

        return EngineOutput(
            scores=scores,
            raw_metrics=fund[available],
            confidence=conf,
            metadata={"as_of": as_of, "metrics_used": available},
        )
