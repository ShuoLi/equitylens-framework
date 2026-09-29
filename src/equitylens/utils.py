"""Cross-sectional ranking utilities.

All scores are normalized 0-100 where higher = better. NaN inputs stay NaN.
"""

from __future__ import annotations

import pandas as pd


def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank each value as a percentile (0-100) within the series.

    NaN values receive NaN (not 50).
    """
    if series.dropna().empty:
        return pd.Series(float("nan"), index=series.index)

    ranks = series.rank(pct=True, na_option="keep") * 100
    if not higher_is_better:
        ranks = 100 - ranks
    return ranks.clip(0, 100)


def sector_percentile_rank(
    df: pd.DataFrame,
    col: str,
    sector_col: str = "sector",
    higher_is_better: bool = True,
) -> pd.Series:
    """Rank each value within its sector. Falls back to global rank when a
    sector has fewer than 5 valid stocks."""
    result = pd.Series(float("nan"), index=df.index)

    for _, group in df.groupby(sector_col):
        if group[col].notna().sum() >= 5:
            result.loc[group.index] = percentile_rank(group[col], higher_is_better)
        else:
            result.loc[group.index] = percentile_rank(df[col], higher_is_better).loc[group.index]

    return result


def winsorize(series: pd.Series, limits: float = 0.02) -> pd.Series:
    """Clip at the given quantile on each tail to remove outlier distortion."""
    valid = series.dropna()
    if valid.empty:
        return series.copy()
    lo, hi = valid.quantile(limits), valid.quantile(1 - limits)
    return series.clip(lower=lo, upper=hi)
