"""Pipeline orchestrator — the main entry point for users."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from equitylens.cache import Cache
from equitylens.plugin import DataSource, Engine, EngineOutput, PortfolioRule

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    engine: Engine
    weight: float
    min_confidence: float = 0.0  # skip engine if mean confidence below this


class Pipeline:
    """Orchestrates data → engines → composite scores → portfolio.

    Parameters
    ----------
    data_source : DataSource
        Injected data source (yfinance, Polygon, custom, etc.)
    cache : Cache, optional
        SQLite cache for engine outputs. Pass None to disable.

    Example
    -------
    >>> from equitylens import Pipeline, YFinanceSource
    >>> from equitylens.engines import MomentumEngine, QualityEngine
    >>> pipeline = Pipeline(YFinanceSource())
    >>> pipeline.register(MomentumEngine(), weight=0.3)
    >>> pipeline.register(QualityEngine(), weight=0.7)
    >>> scores = pipeline.run(["AAPL", "MSFT"], as_of="2025-09-18")
    """

    def __init__(self, data_source: DataSource, cache: Cache | None = None):
        self.data_source = data_source
        self.cache = cache
        self._engines: dict[str, EngineConfig] = {}
        self._portfolio_rule: PortfolioRule | None = None
        self._ic_history: dict[str, pd.Series] = {}  # engine_name → IC time series

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, engine: Engine, weight: float, min_confidence: float = 0.0):
        """Register an engine with a blending weight."""
        if engine.name in self._engines:
            raise ValueError(f"Engine '{engine.name}' already registered")
        self._engines[engine.name] = EngineConfig(
            engine=engine, weight=weight, min_confidence=min_confidence
        )
        logger.info(f"Registered engine '{engine.name}' weight={weight}")

    def set_portfolio_rule(self, rule: PortfolioRule):
        """Set the portfolio construction rule."""
        self._portfolio_rule = rule

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(
        self,
        universe: list[str],
        as_of: str,
        blend_method: str = "weighted_average",
    ) -> pd.Series:
        """Run all engines and return composite scores.

        Parameters
        ----------
        universe : list[str]
        as_of : str
            YYYY-MM-DD point-in-time date.
        blend_method : str
            "weighted_average" | "ic_weighted" (requires IC history)

        Returns
        -------
        pd.Series
            Composite scores, index=ticker, sorted descending.
        """
        outputs = self._run_engines(universe, as_of)
        composite = self._blend(outputs, method=blend_method)
        return composite.sort_values(ascending=False)

    def _run_engines(
        self, universe: list[str], as_of: str
    ) -> dict[str, EngineOutput]:
        outputs = {}
        for name, cfg in self._engines.items():
            # Try cache
            if self.cache:
                cached = self.cache.get("engine", name, as_of, hash(tuple(universe)))
                if cached is not None:
                    logger.debug(f"Cache hit: {name} @ {as_of}")
                    outputs[name] = cached
                    continue

            # Compute
            logger.info(f"Computing engine '{name}' for {len(universe)} tickers @ {as_of}")
            output = cfg.engine.compute(self.data_source, universe, as_of)

            # Validate
            warnings = cfg.engine.validate(output)
            for w in warnings:
                logger.warning(f"[{name}] {w}")

            # Confidence gate
            if cfg.min_confidence > 0 and not output.confidence.empty:
                mean_conf = output.confidence.mean()
                if mean_conf < cfg.min_confidence:
                    logger.warning(
                        f"[{name}] mean confidence {mean_conf:.2f} < {cfg.min_confidence}, skipping"
                    )
                    continue

            # Cache
            if self.cache:
                self.cache.set("engine", name, as_of, hash(tuple(universe)), value=output)

            outputs[name] = output

        return outputs

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------

    def _blend(
        self, outputs: dict[str, EngineOutput], method: str = "weighted_average"
    ) -> pd.Series:
        if not outputs:
            raise ValueError("No engines produced output")

        # Collect all scores into a DataFrame
        score_df = pd.DataFrame({name: out.scores for name, out in outputs.items()})

        if method == "weighted_average":
            weights = {
                name: self._engines[name].weight for name in outputs
            }
            # Normalize weights to sum to 1
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            composite = sum(score_df[name] * w for name, w in weights.items())
            return composite

        elif method == "ic_weighted":
            # Weight by historical IC (Information Coefficient)
            ic_weights = {}
            for name in outputs:
                ic_series = self._ic_history.get(name)
                if ic_series is not None and len(ic_series) > 0:
                    # Use mean IC, clamped to positive
                    ic_weights[name] = max(ic_series.mean(), 0.01)
                else:
                    # Fallback to configured weight if no IC history
                    ic_weights[name] = self._engines[name].weight

            total = sum(ic_weights.values())
            ic_weights = {k: v / total for k, v in ic_weights.items()}
            logger.info(f"IC weights: {ic_weights}")

            composite = sum(score_df[name] * w for name, w in ic_weights.items())
            return composite

        else:
            raise ValueError(f"Unknown blend method: {method}")

    # ------------------------------------------------------------------
    # IC history (for ic_weighted blending)
    # ------------------------------------------------------------------

    def record_ic(self, engine_name: str, ic_value: float, date: str):
        """Record an IC observation for an engine."""
        if engine_name not in self._ic_history:
            self._ic_history[engine_name] = pd.Series(dtype=float)
        self._ic_history[engine_name].loc[date] = ic_value

    def load_ic_history(self, engine_name: str, series: pd.Series):
        """Bulk-load IC history from a Series (index=date, values=IC)."""
        self._ic_history[engine_name] = series

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def build_portfolio(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Turn composite scores into target weights via the portfolio rule."""
        if self._portfolio_rule is None:
            raise ValueError("No portfolio rule set. Call set_portfolio_rule() first.")
        if current_weights is None:
            current_weights = pd.Series(0.0, index=scores.index)
        return self._portfolio_rule.target_weights(scores, prices, current_weights)
