"""Backtest engine — walk-forward simulation with PIT enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from equitylens.pipeline import Pipeline
from equitylens.plugin import PortfolioRule

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series       # portfolio value over time
    returns: pd.Series            # period returns
    weights_history: pd.DataFrame # weights at each rebalance
    turnover: pd.Series           # turnover at each rebalance
    scores_history: pd.DataFrame  # composite scores at each rebalance
    stats: dict                   # summary statistics


class BacktestEngine:
    """Walk-forward backtest with point-in-time enforcement.

    Parameters
    ----------
    pipeline : Pipeline
        Configured pipeline (data source + engines registered).
    portfolio_rule : PortfolioRule
        Rule for constructing target weights.
    rebalance_freq : str
        pandas offset alias: "W" (weekly), "2W", "M", etc.
    transaction_cost_bps : float
        Cost per unit of turnover, in basis points.
    max_lookback_days : int
        Extra calendar days to fetch prices before backtest start
        so the first rebalance has enough data.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        portfolio_rule: PortfolioRule,
        rebalance_freq: str = "W",
        transaction_cost_bps: float = 10.0,
        max_lookback_days: int = 200,
    ):
        self.pipeline = pipeline
        self.portfolio_rule = portfolio_rule
        self.rebalance_freq = rebalance_freq
        self.transaction_cost_bps = transaction_cost_bps / 10000  # bps → decimal
        self.max_lookback_days = max_lookback_days

    def run(
        self,
        universe: list[str],
        start: str,
        end: str,
        initial_capital: float = 100_000,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Parameters
        ----------
        universe : list[str]
        start, end : str
            YYYY-MM-DD. `start` is the first rebalance date.
        initial_capital : float

        Returns
        -------
        BacktestResult
        """
        dates = self._rebalance_dates(start, end)
        logger.info(f"Backtest: {len(dates)} rebalances from {dates[0]} to {dates[-1]}")

        # Fetch full price history once (data source handles PIT)
        px_start = pd.Timestamp(start) - pd.Timedelta(days=self.max_lookback_days)
        prices = self.pipeline.data_source.prices(universe, str(px_start), end)
        if prices.empty:
            raise ValueError("No price data returned for backtest period")
        px = prices.pivot_table(index="date", columns="ticker", values="adj_close")
        px.index = pd.to_datetime(px.index)

        # Initialize
        cash = initial_capital
        holdings = pd.Series(0.0, index=universe)  # shares
        prev_weights = pd.Series(0.0, index=universe)
        equity_history = []
        weights_history = []
        turnover_history = []
        scores_history = []

        for i, as_of in enumerate(dates):
            as_of_str = str(as_of.date())
            logger.info(f"[{i+1}/{len(dates)}] Rebalance @ {as_of_str}")

            # 1. Get composite scores (PIT enforced by Pipeline)
            scores = self.pipeline.run(universe, as_of=as_of_str)

            # 2. Build target weights
            current_px = px.loc[:as_of].iloc[-1]
            target_w = self.portfolio_rule.target_weights(scores, prices, prev_weights)
            if target_w.empty:
                logger.warning(f"No positions selected @ {as_of_str}")
                continue

            # 3. Compute turnover and transaction costs
            aligned_target = target_w.reindex(universe).fillna(0)
            turnover = float((aligned_target - prev_weights).abs().sum() / 2)
            cost = turnover * self.transaction_cost_bps * (cash + (holdings * current_px).sum())

            # 4. Rebalance to target weights
            portfolio_value = cash + (holdings * current_px).sum()
            portfolio_value -= cost
            cash = 0.0

            # Allocate to target weights
            for ticker in universe:
                if ticker in target_w.index and target_w[ticker] > 0:
                    target_value = portfolio_value * target_w[ticker]
                    if current_px.get(ticker, 0) > 0:
                        holdings[ticker] = target_value / current_px[ticker]
                    else:
                        holdings[ticker] = 0.0
                else:
                    holdings[ticker] = 0.0

            # 5. Track
            equity_history.append({"date": as_of, "equity": portfolio_value})
            weights_history.append({"date": as_of, **aligned_target.to_dict()})
            turnover_history.append({"date": as_of, "turnover": turnover})
            scores_history.append({"date": as_of, **scores.to_dict()})
            prev_weights = aligned_target

        # Build result objects
        eq_df = pd.DataFrame(equity_history).set_index("date")["equity"]
        ret = eq_df.pct_change().dropna()
        w_df = pd.DataFrame(weights_history).set_index("date")
        t_df = pd.DataFrame(turnover_history).set_index("date")["turnover"]
        s_df = pd.DataFrame(scores_history).set_index("date")

        stats = self._compute_stats(eq_df, ret, t_df)
        return BacktestResult(
            equity_curve=eq_df,
            returns=ret,
            weights_history=w_df,
            turnover=t_df,
            scores_history=s_df,
            stats=stats,
        )

    def _rebalance_dates(self, start: str, end: str) -> pd.DatetimeIndex:
        dates = pd.date_range(start, end, freq=self.rebalance_freq)
        return dates

    @staticmethod
    def _compute_stats(
        equity: pd.Series,
        returns: pd.Series,
        turnover: pd.Series,
    ) -> dict[str, Any]:
        n_periods = len(returns)
        if n_periods == 0:
            return {}

        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        periods_per_year = 52  # weekly rebalance default

        # CAGR
        years = n_periods / periods_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe (annualized, risk-free = 0)
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0

        # Max drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Win rate
        win_rate = (returns > 0).mean()

        # Turnover
        avg_turnover = turnover.mean()
        annual_turnover = avg_turnover * periods_per_year

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_weekly_turnover": avg_turnover,
            "annual_turnover": annual_turnover,
            "n_periods": n_periods,
            "final_equity": equity.iloc[-1],
        }
