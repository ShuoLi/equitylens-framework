"""YFinance data source — free, no API key, good for development."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from equitylens.plugin import DataSource


class YFinanceSource(DataSource):
    """Fetch prices and fundamentals from Yahoo Finance via yfinance.

    Note: yfinance is not point-in-time safe for fundamentals (it returns
    latest filings, not historical). For PIT-correct backtests, use a
    source that tracks filing dates (e.g., SEC EDGAR, Polygon, SimFin).
    """

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
        # Normalize to long format
        frames = []
        for t in tickers:
            if t not in data.columns.levels[0]:
                continue
            df = data[t].copy()
            df["ticker"] = t
            df = df.reset_index().rename(columns={"Date": "date"})
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fundamentals(self, tickers: list[str], as_of: str | None = None) -> pd.DataFrame:
        """Fetch latest fundamentals. NOT point-in-time safe."""
        rows = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                rows.append(
                    {
                        "ticker": t,
                        "pe_trailing": info.get("trailingPE"),
                        "pe_forward": info.get("forwardPE"),
                        "peg": info.get("pegRatio"),
                        "ps": info.get("priceToSalesTrailing12Months"),
                        "pb": info.get("priceToBook"),
                        "roe": info.get("returnOnEquity"),
                        "roa": info.get("returnOnAssets"),
                        "gross_margin": info.get("grossMargins"),
                        "operating_margin": info.get("operatingMargins"),
                        "debt_to_equity": info.get("debtToEquity"),
                        "current_ratio": info.get("currentRatio"),
                        "revenue_growth": info.get("revenueGrowth"),
                        "earnings_growth": info.get("earningsGrowth"),
                        "market_cap": info.get("marketCap"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                    }
                )
            except Exception as e:
                print(f"[YFinanceSource] {t}: {e}")
        return pd.DataFrame(rows)

    def alternative(self, tickers: list[str], kind: str, as_of: str | None = None):
        return None  # yfinance doesn't provide alternative data
