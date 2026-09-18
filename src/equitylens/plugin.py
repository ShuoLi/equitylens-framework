"""Plugin interfaces for EquityLens.

All plugins implement one of these protocols. The framework handles
orchestration, caching, point-in-time enforcement, and attribution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

@runtime_checkable
class DataSource(Protocol):
    """Anything that can fetch raw market data for a universe of tickers.

    Implementations must be point-in-time safe: `fundamentals(tickers, as_of)`
    must return only data that was public knowledge as of that date.
    """

    def prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: date, ticker, open, high, low, close, volume, adj_close
            Index: MultiIndex (date, ticker) or flat with date + ticker cols.
        """
        ...

    def fundamentals(
        self,
        tickers: list[str],
        as_of: str | None = None,
    ) -> pd.DataFrame:
        """Fundamental data.

        Parameters
        ----------
        tickers : list[str]
        as_of : str, optional
            If provided, return only data filed on or before this date
            (YYYY-MM-DD). This is the point-in-time constraint.

        Returns
        -------
        pd.DataFrame
            One row per ticker. Column names are free-form; engines
            document which columns they require.
        """
        ...

    def alternative(
        self,
        tickers: list[str],
        kind: str,
        as_of: str | None = None,
    ) -> pd.DataFrame | None:
        """Optional alternative data (news, insider, sentiment, etc.).

        Parameters
        ----------
        kind : str
            Identifier for the data type, e.g. "news", "insider_trades".
        """
        ...


# ---------------------------------------------------------------------------
# Engines (Signal Generators)
# ---------------------------------------------------------------------------

@dataclass
class EngineOutput:
    """Standardized result from any Engine.compute() call.

    Attributes
    ----------
    scores : pd.Series
        index=ticker, values=0-100 (or any comparable scale).
    raw_metrics : pd.DataFrame
        All intermediate calculations. Useful for debugging and attribution.
    confidence : pd.Series
        index=ticker, values=0-1. Data-quality confidence, not signal strength.
    metadata : dict
        Engine version, run timestamp, config snapshot, etc.
    """

    scores: pd.Series
    raw_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence: pd.Series = field(default_factory=pd.Series)
    metadata: dict = field(default_factory=dict)


class Engine(ABC):
    """Base class for all signal engines.

    Subclasses must:
      1. Set `name` (unique string identifier)
      2. Implement `compute(data, universe, as_of)`
      3. Be point-in-time safe: only use data available at `as_of`

    The framework calls `compute()` during pipeline runs and backtests.
    Engines should be stateless; store config in __init__, not in attributes
    mutated during compute().
    """

    name: str = "base"

    @abstractmethod
    def compute(
        self,
        data: DataSource,
        universe: list[str],
        as_of: str,
    ) -> EngineOutput:
        """Generate scores for universe as of a specific date.

        Parameters
        ----------
        data : DataSource
            Injected data source. Do NOT call yfinance/EDGAR directly.
        universe : list[str]
            Tickers to score.
        as_of : str
            YYYY-MM-DD. Only use data available on or before this date.

        Returns
        -------
        EngineOutput
        """
        ...

    def validate(self, output: EngineOutput) -> list[str]:
        """Optional sanity checks. Return list of warning strings."""
        warnings = []
        if output.scores.empty:
            warnings.append(f"{self.name}: empty scores")
        if not output.scores.between(0, 100).all():
            warnings.append(f"{self.name}: scores outside 0-100 range")
        return warnings


# ---------------------------------------------------------------------------
# Portfolio Rules
# ---------------------------------------------------------------------------

class PortfolioRule(ABC):
    """How to turn composite scores into target portfolio weights.

    Implementations should be turnover-aware: the framework passes
    current weights so rules can penalize excessive trading.
    """

    @abstractmethod
    def target_weights(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        """Return target portfolio weights.

        Parameters
        ----------
        scores : pd.Series
            Composite scores from Pipeline.run(), index=ticker.
        prices : pd.DataFrame
            Recent price data for volatility/liquidity checks.
        current_weights : pd.Series
            Current portfolio weights (0 for new portfolio).

        Returns
        -------
        pd.Series
            Target weights, index=ticker, values that sum to ~1.0.
            Tickers with 0 weight are excluded.
        """
        ...
