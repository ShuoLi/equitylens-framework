"""Technical engine — price action, trend structure, volume, relative strength.

Sub-scores:
  1. Trend Structure     (35%) — MA relationship, 50MA slope, golden cross
  2. Momentum Quality    (30%) — Bollinger %B (capped at 1.0)
  3. Volume Confirmation (20%) — OBV slope, volume ratio
  4. Relative Strength   (15%) — 3M/6M return vs universe median

RSI is used only as an overbought-entry-risk cap (RSI > 75 with %B > 1.0
pulls the score toward 50), not as a selection factor — cross-sectional RSI
IC is low at monthly horizons.

Note: this overlaps MomentumEngine by design (trend-following inputs), but
the signals differ: momentum uses risk-adjusted total return; technical uses
trend structure + volume confirmation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput
from equitylens.utils import percentile_rank, winsorize

logger = logging.getLogger(__name__)

WEIGHTS = {"trend": 0.35, "mom": 0.30, "vol": 0.20, "rs": 0.15}

_EPS = 1e-10


class TechnicalEngine(Engine):
    """Technical signal scorer.

    Parameters
    ----------
    lookback_days : int
        Calendar days of OHLCV history (default 500 — covers the 200MA and
        the 52-week range).
    """

    name = "technical"

    def __init__(self, lookback_days: int = 500):
        self.lookback_days = lookback_days

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        start = pd.Timestamp(as_of) - pd.Timedelta(days=self.lookback_days)
        prices = data.prices(universe, start=str(start), end=as_of)
        if prices.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "no price data"},
            )

        close = prices.pivot_table(index="date", columns="ticker", values="adj_close")
        volume = prices.pivot_table(index="date", columns="ticker", values="volume")

        indicators = pd.DataFrame(
            [self._compute_indicators(t, close, volume) for t in universe],
        ).set_index("ticker")

        # Sub-scores (global percentile — technical signals are market-wide)
        trend = self._factor_score(
            indicators,
            {
                "tech_above_50ma": True,
                "tech_above_200ma": True,
                "tech_golden_cross": True,
                "tech_ma50_slope": True,  # winsorized below
            },
        )

        indicators["tech_bb_pct_b_capped"] = indicators["tech_bb_pct_b"].clip(upper=1.0)
        mom = self._factor_score(indicators, {"tech_bb_pct_b_capped": True})

        vol = self._factor_score(
            indicators,
            {
                "tech_obv_slope": True,
                "tech_volume_ratio": True,
            },
        )

        # Relative strength vs universe median (3M / 6M total return)
        rs = self._relative_strength(close)

        sub = pd.DataFrame({"trend": trend, "mom": mom, "vol": vol, "rs": rs})
        weighted_sum = pd.Series(0.0, index=universe)
        total_weight = pd.Series(0.0, index=universe)
        for col, w in WEIGHTS.items():
            mask = sub[col].notna()
            weighted_sum += sub[col].fillna(0) * w * mask
            total_weight += w * mask
        raw_score = (weighted_sum / total_weight.replace(0, np.nan)).clip(0, 100)

        # Overbought penalty: RSI > 75 AND %B > 1.0 → short-term extension
        # risk. Caps the score toward 50 (at RSI=90 + %B>1: ~18% reduction).
        rsi = indicators["tech_rsi"].fillna(50)
        bb = indicators["tech_bb_pct_b"].fillna(0)
        severity = ((rsi - 75).clip(lower=0) / 25) * (bb - 1.0).clip(lower=0)
        raw_score = raw_score * (1 - (severity * 0.40).clip(upper=0.40))

        scores = raw_score.clip(0, 100).dropna()

        # Confidence: fraction of lookback with valid prices
        conf = close.notna().mean().reindex(universe).fillna(0)

        return EngineOutput(
            scores=scores,
            raw_metrics=indicators,
            confidence=conf,
            metadata={
                "as_of": as_of,
                "n_scored": len(scores),
                "weights": WEIGHTS,
            },
        )

    # -- indicators -----------------------------------------------------------

    def _compute_indicators(self, ticker: str, close: pd.DataFrame, volume: pd.DataFrame) -> dict:
        row = {"ticker": ticker}
        defaults = {
            "tech_above_50ma": np.nan,
            "tech_above_200ma": np.nan,
            "tech_golden_cross": np.nan,
            "tech_ma50_slope": np.nan,
            "tech_rsi": np.nan,
            "tech_bb_pct_b": np.nan,
            "tech_obv_slope": np.nan,
            "tech_volume_ratio": np.nan,
            "tech_atr_pct": np.nan,
            "tech_pct_from_52w_high": np.nan,
            "tech_pct_from_52w_low": np.nan,
        }
        row.update(defaults)

        try:
            c = close[ticker].dropna() if ticker in close.columns else pd.Series(dtype=float)
            if len(c) < 30:
                return row

            ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
            last_price, last_ma50, last_ma200 = c.iloc[-1], ma50.iloc[-1], ma200.iloc[-1]

            if not np.isnan(last_ma50):
                row["tech_above_50ma"] = 1.0 if last_price > last_ma50 else 0.0
            if not np.isnan(last_ma200):
                row["tech_above_200ma"] = 1.0 if last_price > last_ma200 else 0.0
            if not np.isnan(last_ma50) and not np.isnan(last_ma200):
                row["tech_golden_cross"] = 1.0 if last_ma50 > last_ma200 else 0.0

            # 50MA slope over last 20 days, as % of price
            if len(ma50.dropna()) >= 20:
                row["tech_ma50_slope"] = (ma50.iloc[-1] - ma50.iloc[-20]) / (ma50.iloc[-20] + _EPS)

            row["tech_rsi"] = self._rsi(c, period=14)

            # Bollinger %B (20, 2)
            if len(c) >= 20:
                ma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
                upper, lower = ma20 + 2 * std20, ma20 - 2 * std20
                band_width = (upper - lower).iloc[-1]
                if band_width > 0:
                    row["tech_bb_pct_b"] = (last_price - lower.iloc[-1]) / band_width
                    row["tech_atr_pct"] = band_width / last_price

            # 52-week range
            window = c.iloc[-252:] if len(c) >= 252 else c
            hi, lo = window.max(), window.min()
            row["tech_pct_from_52w_high"] = (last_price - hi) / hi
            row["tech_pct_from_52w_low"] = (last_price - lo) / (lo + _EPS)

            # OBV slope + volume ratio
            if ticker in volume.columns:
                v = volume[ticker].reindex(c.index).fillna(0)
                obv = self._obv(c, v)
                if len(obv) >= 20:
                    row["tech_obv_slope"] = (obv.iloc[-1] - obv.iloc[-20]) / (
                        abs(obv.iloc[-20]) + _EPS
                    )
                if len(v) >= 50:
                    v50 = v.iloc[-50:].mean()
                    if v50 > 0:
                        row["tech_volume_ratio"] = v.iloc[-20:].mean() / v50
        except Exception as e:
            logger.debug(f"Technical indicator error for {ticker}: {e}")

        return row

    def _relative_strength(self, close: pd.DataFrame) -> pd.Series:
        """3M/6M total return minus the universe median, percentile-ranked."""
        ret_3m = close.iloc[-1] / close.iloc[-64] - 1 if len(close) >= 64 else None
        ret_6m = close.iloc[-1] / close.iloc[-126] - 1 if len(close) >= 126 else None

        scores = []
        for rets in (ret_3m, ret_6m):
            if rets is None:
                continue
            rs = (rets - rets.median()).replace([np.inf, -np.inf], np.nan).dropna()
            if not rs.empty:
                scores.append(percentile_rank(winsorize(rs)))
        if not scores:
            return pd.Series(dtype=float)
        return pd.concat(scores, axis=1).mean(axis=1, skipna=True).clip(0, 100)

    @staticmethod
    def _factor_score(df: pd.DataFrame, metrics: dict[str, bool]) -> pd.Series:
        """Winsorize (except binary flags) then percentile-rank and average."""
        scores = []
        for col in metrics:
            if col not in df.columns or df[col].isna().all():
                continue
            s = df[col]
            # Binary flags and bounded ratios need no winsorizing
            if not s.dropna().isin([0.0, 1.0]).all():
                s = winsorize(s)
            scores.append(percentile_rank(s))
        if not scores:
            return pd.Series(dtype=float)
        return pd.concat(scores, axis=1).mean(axis=1, skipna=True).clip(0, 100)

    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> float:
        if len(prices) < period + 1:
            return np.nan
        delta = prices.diff().dropna()
        avg_gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
        avg_loss = (-delta).clip(lower=0).rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        return 100 - (100 / (1 + avg_gain / avg_loss))

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()
