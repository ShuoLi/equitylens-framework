"""Sentiment engine — analyst consensus, rating momentum, earnings, insider.

Components (each 0-100, NaN when its dataset is unavailable):
  1. Analyst consensus   — buy %, neutral 50 when coverage < 2 analysts
  2. Rating momentum     — net upgrades (2/3 weight on 30d, 1/3 on 90d)
  3. Earnings quality    — avg surprise %, ±10% maps to 0..100
  4. Insider sentiment   — net $ value, ±$10M maps to 0..100
  5. Catalyst            — earnings within 14d boosts/penalizes by surprise history

Data comes from `alternative(kind="sentiment")` — FinnhubSource serves it.
Tickers with no data at all get a NaN score (not a neutral 50), so the
pipeline blend can distinguish "no opinion" from "average opinion".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput

COMPONENTS = [
    "analyst_consensus_score",
    "rating_momentum_score",
    "earnings_quality_score",
    "insider_sentiment_score",
    "catalyst_score",
]


class SentimentEngine(Engine):
    """Analyst / earnings / insider sentiment scorer.

    Expects a data source whose `alternative(tickers, kind="sentiment")`
    returns one row per ticker with columns:
      total_buy, total_hold, total_sell,
      upgrades_30d, downgrades_30d, upgrades_90d, downgrades_90d, num_analysts,
      avg_earnings_surprise_pct,
      days_until_earnings, earnings_imminent,
      insider_net_value
    (FinnhubSource produces exactly this shape.)
    """

    name = "sentiment"

    def __init__(self, min_analysts: int = 2):
        self.min_analysts = min_analysts

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        alt = data.alternative(universe, kind="sentiment", as_of=as_of)
        if alt is None or alt.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "no sentiment data (source returned None)"},
            )

        alt = alt.set_index("ticker").reindex(universe)

        components = pd.DataFrame(index=universe)
        components["analyst_consensus_score"] = self._score_analyst_consensus(alt)
        components["rating_momentum_score"] = self._score_rating_momentum(alt)
        components["earnings_quality_score"] = self._score_earnings_quality(alt)
        components["insider_sentiment_score"] = self._score_insider_sentiment(alt)
        components["catalyst_score"] = self._score_catalyst(alt)

        # Composite: mean over available components; NaN only if none available
        scores = components.mean(axis=1, skipna=True).clip(0, 100)

        # Confidence: fraction of the 5 component datasets present
        has_data = pd.DataFrame(index=universe)
        has_data["analyst"] = alt.get("num_analysts", pd.Series(index=universe)).fillna(0) > 0
        has_data["rating"] = alt.get("upgrades_30d", pd.Series(index=universe)).fillna(0) > 0
        has_data["earnings"] = alt.get("avg_earnings_surprise_pct", pd.Series(index=universe)).notna()
        has_data["insider"] = (
            alt.get("insider_buy_count", pd.Series(index=universe)).fillna(0)
            + alt.get("insider_sell_count", pd.Series(index=universe)).fillna(0)
        ) > 0
        has_data["calendar"] = alt.get("days_until_earnings", pd.Series(index=universe)).notna()
        conf = has_data.mean(axis=1)

        raw = alt.join(components)

        return EngineOutput(
            scores=scores,
            raw_metrics=raw,
            confidence=conf,
            metadata={
                "as_of": as_of,
                "n_scored": int(scores.notna().sum()),
            },
        )

    # -- component scorers -----------------------------------------------------

    def _score_analyst_consensus(self, df: pd.DataFrame) -> pd.Series:
        if "analyst_buy_pct" in df.columns:
            buy_pct = df["analyst_buy_pct"]
        elif {"total_buy", "total_hold", "total_sell"}.issubset(df.columns):
            total = df["total_buy"] + df["total_hold"] + df["total_sell"]
            buy_pct = (df["total_buy"] / total.replace(0, np.nan)) * 100
        else:
            return pd.Series(np.nan, index=df.index)

        has_coverage = df.get("num_analysts", pd.Series(0, index=df.index)).fillna(0) >= self.min_analysts
        total = df.get("total_buy", 0) + df.get("total_hold", 0) + df.get("total_sell", 0)
        has_coverage = has_coverage | (total >= self.min_analysts)
        return buy_pct.where(has_coverage).clip(0, 100)

    def _score_rating_momentum(self, df: pd.DataFrame) -> pd.Series:
        if "upgrades_30d" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        def col(name):
            s = df.get(name)
            return s.fillna(0) if s is not None else pd.Series(0.0, index=df.index)

        net_30d = col("upgrades_30d") - col("downgrades_30d")
        net_90d = col("upgrades_90d") - col("downgrades_90d")
        net_combined = net_30d * 0.67 + net_90d * 0.33
        # -5 net downgrades = 0, 0 = 50, +5 = 100
        return (50 + net_combined * 10).clip(0, 100)

    def _score_earnings_quality(self, df: pd.DataFrame) -> pd.Series:
        if "avg_earnings_surprise_pct" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        surprise = df["avg_earnings_surprise_pct"]
        # -10% = 0, 0% = 50, +10% = 100
        return (50 + surprise.fillna(0) * 5).clip(0, 100).where(surprise.notna())

    def _score_insider_sentiment(self, df: pd.DataFrame) -> pd.Series:
        if "insider_net_value" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        net_value = df["insider_net_value"]
        scaled = (net_value / 1e7).clip(-1, 1)
        return (50 + scaled.fillna(0) * 50).clip(0, 100).where(net_value.notna())

    def _score_catalyst(self, df: pd.DataFrame) -> pd.Series:
        if "days_until_earnings" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        days = df["days_until_earnings"]
        imminent = days <= 14
        score = pd.Series(50.0, index=df.index)
        if "avg_earnings_surprise_pct" in df.columns:
            surprise = df["avg_earnings_surprise_pct"].fillna(0)
            score = score + imminent.fillna(False) * (surprise * 2).clip(-20, 20)
        return score.clip(0, 100).where(days.notna())
