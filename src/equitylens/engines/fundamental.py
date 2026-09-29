"""Fundamental engine — accounting deep-dive, independent of price.

Purely fundamentals-based scoring using EDGAR-derived metrics (served
PIT-safe by EdgarSource, or any source exposing the same columns):

  1. Capital Efficiency   (30%) — ROIC, FCF growth
  2. Earnings Quality     (25%) — CFO/NI consistency, accruals ratio
  3. Growth Quality       (20%) — 3Y revenue CAGR, earnings consistency
  4. Competitive Position (15%) — margin stability, R&D intensity, SGA efficiency
  5. Shareholder Return   (10%) — buyback yield, Piotroski F-Score

Sub-scores 1/3/4 rank within sector when a `sector` column is available
(capital efficiency isn't comparable across industries); 2/5 rank globally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput
from equitylens.utils import percentile_rank, sector_percentile_rank, winsorize

WEIGHTS = {"cap": 0.30, "eq": 0.25, "gq": 0.20, "cp": 0.15, "sr": 0.10}

GROUPS = {
    "cap": {"roic": True, "fcf_growth_1y": True},
    "eq": {"cfo_ni_consistency": True, "accruals_ratio": False},
    "gq": {"revenue_growth_3y": True, "earnings_consistency": False},
    "cp": {"gross_margin_stability": False, "rd_intensity": True, "sga_efficiency": False},
    "sr": {"buyback_yield": True, "piotroski_f": True},
}

SECTOR_RELATIVE = {"cap": True, "eq": False, "gq": True, "cp": True, "sr": False}

METRIC_COLUMNS = [m for group in GROUPS.values() for m in group]


class FundamentalEngine(Engine):
    """EDGAR-style fundamental scorer.

    Works with any DataSource whose fundamentals() returns the derived
    metric columns (EdgarSource computes them point-in-time safe).
    """

    name = "fundamental"

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        fund = data.fundamentals(universe, as_of=as_of)
        if fund.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "no fundamental data"},
            )

        fund = fund.set_index("ticker").reindex(universe)

        sub_scores = pd.DataFrame(index=fund.index)
        for group, metrics in GROUPS.items():
            sub_scores[group] = self._factor_score(
                fund, metrics, sector_relative=SECTOR_RELATIVE[group]
            )

        weighted_sum = pd.Series(0.0, index=fund.index)
        total_weight = pd.Series(0.0, index=fund.index)
        for group, w in WEIGHTS.items():
            mask = sub_scores[group].notna()
            weighted_sum += sub_scores[group].fillna(0) * w * mask
            total_weight += w * mask
        scores = (weighted_sum / total_weight.replace(0, np.nan)).clip(0, 100).dropna()

        # Confidence: fraction of the 10 metric columns available
        available = [m for m in METRIC_COLUMNS if m in fund.columns]
        conf = fund[available].notna().mean(axis=1) if available else pd.Series(0.0, index=fund.index)

        raw = fund.join(sub_scores.rename(columns={g: f"fund_{g}_score" for g in GROUPS}))

        return EngineOutput(
            scores=scores,
            raw_metrics=raw,
            confidence=conf,
            metadata={
                "as_of": as_of,
                "n_scored": len(scores),
                "weights": WEIGHTS,
            },
        )

    def _factor_score(
        self,
        df: pd.DataFrame,
        metrics: dict[str, bool],
        sector_relative: bool,
    ) -> pd.Series:
        """Winsorize, then percentile-rank (sector-relative if requested and a
        sector column exists), then average. metrics = {col: higher_is_better}."""
        scores = []
        for col, hib in metrics.items():
            if col not in df.columns or df[col].isna().all():
                continue
            s = winsorize(df[col])
            if sector_relative and "sector" in df.columns:
                tmp = df.assign(**{col: s})
                ranked = sector_percentile_rank(tmp, col, higher_is_better=hib)
            else:
                ranked = percentile_rank(s, higher_is_better=hib)
            scores.append(ranked)
        if not scores:
            return pd.Series(dtype=float)
        return pd.concat(scores, axis=1).mean(axis=1, skipna=True).clip(0, 100)
