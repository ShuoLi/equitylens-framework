"""Portfolio rules — turn scores into target weights."""

from __future__ import annotations

import pandas as pd

from equitylens.plugin import PortfolioRule


class EqualWeightTopN(PortfolioRule):
    """Hold top N stocks, equal weight.

    Parameters
    ----------
    n : int
        Number of holdings.
    min_score : float, optional
        Only hold stocks with composite score >= this threshold.
    """

    def __init__(self, n: int = 20, min_score: float | None = None):
        self.n = n
        self.min_score = min_score

    def target_weights(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        s = scores.copy()
        if self.min_score is not None:
            s = s[s >= self.min_score]
        top = s.nlargest(self.n)
        if len(top) == 0:
            return pd.Series(dtype=float)
        w = pd.Series(1.0 / len(top), index=top.index)
        return w


class ScoreWeightedTopN(PortfolioRule):
    """Hold top N stocks, weight proportional to score.

    Parameters
    ----------
    n : int
        Number of holdings.
    max_weight : float
        Cap any single position at this weight.
    min_score : float, optional
        Only hold stocks with composite score >= this threshold.
    """

    def __init__(self, n: int = 20, max_weight: float = 0.10, min_score: float | None = None):
        self.n = n
        self.max_weight = max_weight
        self.min_score = min_score

    def target_weights(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        s = scores.copy()
        if self.min_score is not None:
            s = s[s >= self.min_score]
        top = s.nlargest(self.n)
        if len(top) == 0:
            return pd.Series(dtype=float)

        # Weight proportional to score, capped
        w = top / top.sum()
        w = w.clip(upper=self.max_weight)
        # Re-normalize after capping
        w = w / w.sum()
        return w


class TurnoverAwareTopN(PortfolioRule):
    """Hold top N, but penalize turnover from current portfolio.

    Combines score-proportional weights with a penalty for deviating
    from current weights. This reduces transaction costs.

    Parameters
    ----------
    n : int
        Number of holdings.
    turnover_penalty : float
        0 = ignore current weights, 1 = never trade. Default 0.3.
    max_weight : float
        Cap any single position.
    """

    def __init__(
        self,
        n: int = 20,
        turnover_penalty: float = 0.3,
        max_weight: float = 0.10,
    ):
        self.n = n
        self.turnover_penalty = turnover_penalty
        self.max_weight = max_weight

    def target_weights(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        top = scores.nlargest(self.n)
        if len(top) == 0:
            return pd.Series(dtype=float)

        # Score-proportional target
        score_w = top / top.sum()

        # Blend with current weights to reduce turnover
        if self.turnover_penalty > 0 and len(current_weights) > 0:
            aligned_current = current_weights.reindex(score_w.index).fillna(0)
            blended = (1 - self.turnover_penalty) * score_w + self.turnover_penalty * aligned_current
        else:
            blended = score_w

        blended = blended.clip(upper=self.max_weight)
        blended = blended / blended.sum()
        return blended
