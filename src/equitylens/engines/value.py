"""Value engine — multi-metric valuation scoring."""

from __future__ import annotations

import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput


class ValueEngine(Engine):
    """Score stocks on cheapness.

    Combines forward P/E, PEG, P/S, and P/B. Each is inverted (cheaper = higher)
    and cross-sectionally ranked. Handles negative earnings gracefully by
    excluding those metrics for affected stocks.

    Parameters
    ----------
    use_forward_pe : bool
        Prefer forward P/E over trailing (default True). Forward P/E is
        less distorted by one-time charges but depends on analyst estimates.
    max_peg : float
        Cap PEG at this value to avoid outlier dominance.
    """

    name = "value"

    def __init__(self, use_forward_pe: bool = True, max_peg: float = 5.0):
        self.use_forward_pe = use_forward_pe
        self.max_peg = max_peg

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        fund = data.fundamentals(universe, as_of=as_of)
        if fund.empty:
            return EngineOutput(scores=pd.Series(dtype=float))

        fund = fund.set_index("ticker")
        metrics_used = []

        # --- P/E (forward preferred, trailing fallback) ---
        pe_col = "pe_forward" if self.use_forward_pe else "pe_trailing"
        if pe_col not in fund.columns:
            pe_col = "pe_trailing" if "pe_trailing" in fund.columns else None

        if pe_col:
            pe = fund[pe_col].copy()
            # Negative P/E = unprofitable → score 0 on this metric
            pe = pe.where(pe > 0)
            fund["pe_inv"] = 1 / pe
            metrics_used.append("pe_inv")

        # --- PEG ---
        if "peg" in fund.columns:
            peg = fund["peg"].clip(upper=self.max_peg)
            fund["peg_inv"] = 1 / peg.where(peg > 0)
            metrics_used.append("peg_inv")

        # --- P/S ---
        if "ps" in fund.columns:
            ps = fund["ps"]
            fund["ps_inv"] = 1 / ps.where(ps > 0)
            metrics_used.append("ps_inv")

        # --- P/B ---
        if "pb" in fund.columns:
            pb = fund["pb"]
            fund["pb_inv"] = 1 / pb.where(pb > 0)
            metrics_used.append("pb_inv")

        # Rank each inverted metric to 0-100
        ranked = pd.DataFrame(index=fund.index)
        for m in metrics_used:
            ranked[m] = fund[m].rank(pct=True) * 100

        if not metrics_used:
            return EngineOutput(scores=pd.Series(dtype=float))

        # Equal-weight average of available metrics
        scores = ranked.mean(axis=1).dropna()
        conf = ranked.notna().mean(axis=1)

        return EngineOutput(
            scores=scores,
            raw_metrics=fund[metrics_used],
            confidence=conf,
            metadata={"as_of": as_of, "metrics_used": metrics_used},
        )
