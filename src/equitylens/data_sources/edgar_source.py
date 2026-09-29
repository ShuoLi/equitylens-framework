"""SEC EDGAR data source — point-in-time safe fundamentals from 10-K filings.

Fetches structured financials from EDGAR's XBRL company facts API and derives
fundamental metrics (ROIC, revenue CAGR, Piotroski F, buyback yield, ...).
Every metric is filtered by filing date, so `fundamentals(tickers, as_of)`
only uses data that was public knowledge as of that date (45-day lag after
filing, the standard PIT conservatism for 10-K data).

Also serves 10-K narrative text (MD&A + Risk Factors) via `alternative()`
for the text engine.

Requires the `edgar` extra: pip install equitylens[edgar]
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from equitylens.plugin import DataSource

logger = logging.getLogger(__name__)

EDGAR_HEADERS = {"User-Agent": "equitylens research@equitylens.local"}
RATE_LIMIT_SLEEP = 0.12  # ~8 req/s, under SEC's 10/s limit
DEFAULT_LAG_DAYS = 45

# Concept aliases: tried in order, first non-empty match wins.
CONCEPT_ALIASES = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomer",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "CapitalExpenditureDiscontinuedOperations",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
    "total_debt": [
        "DebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "total_assets": ["Assets"],
    "gross_profit": ["GrossProfit"],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAndShortTermInvestments",
    ],
}

N_YEARS = 15  # annual values to keep (allows backtests back to ~2010)

METRIC_COLUMNS = [
    "revenue_growth_3y",
    "fcf_growth_1y",
    "roic",
    "buyback_yield",
    "piotroski_f",
    "gross_margin_stability",
    "cfo_ni_consistency",
    "rd_intensity",
    "sga_efficiency",
    "earnings_consistency",
    "accruals_ratio",
]


# ---------------------------------------------------------------------------
# PIT helpers (module level for testability)
# ---------------------------------------------------------------------------


def is_observable(
    filing_date: str | None,
    as_of: datetime.date | None = None,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> bool:
    """True only if `filing_date + lag_days <= as_of`.

    Standard practice for 10-K data is a 45-day lag after the filing date
    before the data is considered point-in-time safe. If as_of is None (live
    scoring), everything already filed is public. If as_of is set, a missing
    or unparseable filing_date means observability CANNOT be proven — exclude
    it (returning True there is a look-ahead hole, not "conservative").
    """
    if as_of is None:
        return True
    if not filing_date:
        return False
    try:
        filed = datetime.date.fromisoformat(filing_date)
    except ValueError:
        return False
    return (filed + datetime.timedelta(days=lag_days)) <= as_of


def _extract_annual_with_dates(
    facts_usegaap: dict, concepts: list[str], n: int = N_YEARS
) -> list[tuple]:
    """Extract (end_date, filing_date, value) triples for the first matching
    concept, deduplicated by fiscal year (most recently filed wins)."""
    for concept in concepts:
        if concept not in facts_usegaap:
            continue
        units_block = facts_usegaap[concept].get("units", {})
        entries = units_block.get("USD") or units_block.get("shares") or []
        if not entries:
            continue

        annual = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
        by_fy: dict[int, dict] = {}
        for e in annual:
            fy = e.get("fy")
            if fy is None:
                continue
            if fy not in by_fy or e.get("filed", "") > by_fy[fy].get("filed", ""):
                by_fy[fy] = e

        sorted_entries = sorted(by_fy.values(), key=lambda e: e.get("end", ""))
        result = [(e["end"], e.get("filed", ""), e["val"]) for e in sorted_entries[-n:]]
        if result:
            return result
    return []


def _cagr(start: float, end: float, years: float) -> float | None:
    if not start or start <= 0 or years <= 0:
        return None
    try:
        return (end / start) ** (1 / years) - 1
    except (OverflowError, ZeroDivisionError, ValueError):
        return None


def _derived_metrics(fin: dict, current_ratio: float = float("nan")) -> dict:
    """Derive fundamental metrics from a PIT-filtered {metric: [(date, val)]} dict.

    Ports the EDGAR deep-dive metrics: revenue CAGR, FCF growth, true ROIC,
    buyback yield, full Piotroski F, margin stability, CFO/NI consistency,
    R&D intensity, SGA efficiency, earnings consistency, accruals ratio.
    """
    out = {k: np.nan for k in METRIC_COLUMNS}
    if not fin:
        return out

    def vals(key):
        return [v for _, v in fin.get(key, [])]

    # Revenue CAGR 3Y
    rev = fin.get("revenue", [])
    if len(rev) >= 4:
        cagr = _cagr(rev[-4][1], rev[-1][1], 3)
        if cagr is not None:
            out["revenue_growth_3y"] = cagr

    # Revenue consistency: coefficient of variation (lower = more predictable)
    if len(rev) >= 3:
        v = vals("revenue")
        mean_rev = float(np.mean(v))
        if mean_rev > 0:
            out["earnings_consistency"] = float(np.std(v) / mean_rev)

    # FCF growth 1Y
    cfo, cap = fin.get("cfo", []), fin.get("capex", [])
    if len(cfo) >= 2 and len(cap) >= 2:
        fcf_now = cfo[-1][1] - abs(cap[-1][1])  # capex is negative in EDGAR
        fcf_prev = cfo[-2][1] - abs(cap[-2][1])
        if fcf_prev != 0:
            out["fcf_growth_1y"] = (fcf_now - fcf_prev) / abs(fcf_prev)

    # True ROIC: NOPAT / (equity + debt - cash), 21% US corp tax
    op_inc, assets = fin.get("operating_income", []), fin.get("total_assets", [])
    debt, equity, cash = fin.get("total_debt", []), fin.get("total_equity", []), fin.get("cash", [])
    if op_inc and assets and debt and equity:
        nopat = op_inc[-1][1] * (1 - 0.21)
        inv_capital = equity[-1][1] + debt[-1][1] - (cash[-1][1] if cash else 0)
        if inv_capital > 0:
            out["roic"] = nopat / inv_capital

    # Accruals ratio: (NI - CFO) / assets. Lower (more cash-backed) = better.
    ni = fin.get("net_income", [])
    if ni and cfo and assets:
        out["accruals_ratio"] = (ni[-1][1] - cfo[-1][1]) / assets[-1][1]

    # Buyback yield from shares outstanding delta
    shares = fin.get("shares_outstanding", [])
    if len(shares) >= 2:
        s_now, s_prev = shares[-1][1], shares[-2][1]
        out["buyback_yield"] = (s_prev - s_now) / s_prev if (s_prev > 0 and s_now < s_prev) else 0.0

    # Piotroski F (9 points, scaled to points available)
    out["piotroski_f"] = _piotroski(fin, current_ratio)

    # Gross margin stability (std of margins, lower = better)
    gp = fin.get("gross_profit", [])
    if len(gp) >= 3 and len(rev) >= 3:
        n = min(len(gp), len(rev))
        margins = [gp[-i][1] / rev[-i][1] for i in range(1, n + 1) if rev[-i][1] > 0]
        if len(margins) >= 2:
            out["gross_margin_stability"] = float(np.std(margins))

    # CFO/NI consistency (median ratio over available years)
    if len(cfo) >= 3 and len(ni) >= 3:
        n = min(len(cfo), len(ni))
        ratios = [cfo[-i][1] / ni[-i][1] for i in range(1, n + 1) if ni[-i][1] > 0]
        if ratios:
            out["cfo_ni_consistency"] = float(np.median(ratios))

    # R&D intensity / SGA efficiency
    rd = fin.get("rd_expense", [])
    if rd and rev and rev[-1][1] > 0:
        out["rd_intensity"] = abs(rd[-1][1]) / rev[-1][1]
    sga = fin.get("sga_expense", [])
    if sga and rev and rev[-1][1] > 0:
        out["sga_efficiency"] = abs(sga[-1][1]) / rev[-1][1]

    return out


def _piotroski(fin: dict, current_ratio: float) -> float:
    """9-point Piotroski F-Score from annual EDGAR data, scaled to the
    points that could actually be computed. NaN if no data."""
    ni = fin.get("net_income", [])
    cfo = fin.get("cfo", [])
    assets = fin.get("total_assets", [])
    debt = fin.get("total_debt", [])
    gp = fin.get("gross_profit", [])
    rev = fin.get("revenue", [])
    shares = fin.get("shares_outstanding", [])

    def ratio(num, den, idx=-1):
        try:
            n, d = num[idx][1], den[idx][1]
            return n / d if d else np.nan
        except (IndexError, ZeroDivisionError):
            return np.nan

    score, possible = 0, 0

    # Profitability (4)
    roa_now = ratio(ni, assets)
    if not np.isnan(roa_now):
        possible += 1
        score += roa_now > 0
    if cfo:
        possible += 1
        score += cfo[-1][1] > 0
    roa_prev = ratio(ni, assets, -2) if len(ni) >= 2 and len(assets) >= 2 else np.nan
    if not np.isnan(roa_now) and not np.isnan(roa_prev):
        possible += 1
        score += roa_now > roa_prev
    if ni and cfo:
        possible += 1
        score += cfo[-1][1] > ni[-1][1]

    # Leverage / liquidity (3)
    lev_now = ratio(debt, assets)
    lev_prev = ratio(debt, assets, -2) if len(debt) >= 2 and len(assets) >= 2 else np.nan
    if not np.isnan(lev_now) and not np.isnan(lev_prev):
        possible += 1
        score += lev_now < lev_prev
    if not np.isnan(current_ratio):
        possible += 1
        score += current_ratio > 1.5
    if len(shares) >= 2:
        possible += 1
        score += shares[-1][1] <= shares[-2][1] * 1.01  # 1% tolerance

    # Operating efficiency (2)
    gm_now = ratio(gp, rev)
    gm_prev = ratio(gp, rev, -2) if len(gp) >= 2 and len(rev) >= 2 else np.nan
    if not np.isnan(gm_now) and not np.isnan(gm_prev):
        possible += 1
        score += gm_now > gm_prev
    at_now = ratio(rev, assets)
    at_prev = ratio(rev, assets, -2) if len(rev) >= 2 and len(assets) >= 2 else np.nan
    if not np.isnan(at_now) and not np.isnan(at_prev):
        possible += 1
        score += at_now > at_prev

    if possible == 0:
        return np.nan
    return round(score * 9 / possible)


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------


class EdgarSource(DataSource):
    """Point-in-time safe fundamentals + 10-K text from SEC EDGAR.

    Parameters
    ----------
    price_source : DataSource, optional
        EDGAR has no daily prices. Delegate `prices()` to this source
        (e.g. YFinanceSource). If omitted, prices() raises.
    lag_days : int
        Days after a filing date before its data is considered observable.
    current_ratios : dict | callable, optional
        {ticker: current_ratio} (or callable(ticker) -> float) for Piotroski
        F6. Without it, F6 is skipped and the F-score scales to 8 points.
    fetcher : callable, optional
        (url) -> dict | None JSON getter. Defaults to requests; injectable
        for tests. Must respect SEC rate limits in production.
    user_agent : str
        SEC requires a declared User-Agent on all requests.

    Example
    -------
    >>> from equitylens.data_sources import EdgarSource, YFinanceSource
    >>> src = EdgarSource(price_source=YFinanceSource())
    >>> fund = src.fundamentals(["AAPL", "MSFT"], as_of="2023-06-30")
    """

    def __init__(
        self,
        price_source: DataSource | None = None,
        lag_days: int = DEFAULT_LAG_DAYS,
        current_ratios: dict | callable | None = None,
        fetcher=None,
        user_agent: str = "equitylens research@equitylens.local",
    ):
        self._price_source = price_source
        self._lag_days = lag_days
        self._current_ratios = current_ratios
        self._fetcher = fetcher or self._http_get_json
        self._headers = {"User-Agent": user_agent}
        self._cik_map: dict[str, str] | None = None
        self._facts_cache: dict[str, dict] = {}
        self._text_cache: dict[str, dict | None] = {}

    # -- prices ----------------------------------------------------------

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self._price_source is None:
            raise NotImplementedError(
                "EDGAR provides no daily prices; construct EdgarSource(price_source=...)"
            )
        return self._price_source.prices(tickers, start, end)

    # -- fundamentals -----------------------------------------------------

    def fundamentals(self, tickers: list[str], as_of: str | None = None) -> pd.DataFrame:
        as_of_date = datetime.date.fromisoformat(as_of) if as_of else None
        rows = []
        for ticker in tickers:
            fin = self._pit_financials(ticker, as_of_date)
            current_ratio = self._lookup_current_ratio(ticker)
            metrics = _derived_metrics(fin, current_ratio)
            metrics["ticker"] = ticker
            rows.append(metrics)
        return pd.DataFrame(rows)

    def _lookup_current_ratio(self, ticker: str) -> float:
        cr = self._current_ratios
        if cr is None:
            return float("nan")
        if callable(cr):
            try:
                return float(cr(ticker))
            except (TypeError, ValueError):
                return float("nan")
        return float(cr.get(ticker, float("nan")))

    def _pit_financials(self, ticker: str, as_of: datetime.date | None) -> dict:
        dated = self._annual_financials_with_dates(ticker)
        out = {}
        for metric, series in dated.items():
            obs = [
                (end, val)
                for (end, filed, val) in series
                if is_observable(filed, as_of=as_of, lag_days=self._lag_days)
            ]
            if obs:
                out[metric] = obs
        return out

    def _annual_financials_with_dates(self, ticker: str) -> dict:
        if ticker in self._facts_cache:
            facts = self._facts_cache[ticker]
        else:
            facts = self._fetch_company_facts(ticker)
            self._facts_cache[ticker] = facts
        if not facts:
            return {}
        usegaap = facts.get("facts", {}).get("us-gaap", {})
        return {
            metric: _extract_annual_with_dates(usegaap, concepts)
            for metric, concepts in CONCEPT_ALIASES.items()
            if concepts
        }

    def _fetch_company_facts(self, ticker: str) -> dict:
        cik = self._get_cik(ticker)
        if not cik:
            return {}
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        data = self._fetcher(url, headers=self._headers)
        return data or {}

    def _get_cik(self, ticker: str) -> str | None:
        if self._cik_map is None:
            url = "https://www.sec.gov/files/company_tickers.json"
            data = self._fetcher(url, headers=self._headers) or {}
            self._cik_map = {
                v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()
            }
        return self._cik_map.get(ticker.upper())

    # -- alternative: 10-K narrative text ---------------------------------

    def alternative(self, tickers: list[str], kind: str, as_of: str | None = None):
        """kind="10k_mda" returns one row per (ticker, filing) with columns
        ticker, year, filing_date, period_end, mda_text, risk_factors,
        word_count — the two most recent filings observable at as_of."""
        if kind != "10k_mda":
            return None
        return self._ten_k_texts(tickers, as_of)

    def _ten_k_texts(self, tickers: list[str], as_of: str | None) -> pd.DataFrame:
        as_of_date = datetime.date.fromisoformat(as_of) if as_of else None
        latest_year = (as_of_date or datetime.date.today()).year
        rows = []
        for ticker in tickers:
            filings = []
            for year in range(latest_year, latest_year - 3, -1):
                filing = self._fetch_10k_text(ticker, year)
                if filing and is_observable(filing["filing_date"], as_of_date, self._lag_days):
                    filings.append(filing)
                if len(filings) >= 2:
                    break
            rows.extend(filings)
        return pd.DataFrame(rows)

    def _fetch_10k_text(self, ticker: str, year: int) -> dict | None:
        key = f"{ticker}_{year}"
        if key in self._text_cache:
            return self._text_cache[key]
        filing = self._search_10k(ticker, year)
        if filing:
            text = self._fetch_filing_text(filing)
            if text:
                filing.update(
                    {
                        "ticker": ticker,
                        "year": year,
                        "full_text": text,
                        "word_count": len(text.split()),
                        "mda_text": _extract_mda(text),
                        "risk_factors": _extract_risk_factors(text),
                    }
                )
                self._text_cache[key] = filing
                return filing
        self._text_cache[key] = None
        return None

    def _search_10k(self, ticker: str, year: int) -> dict | None:
        query = {
            "q": f'ticker:{ticker} form:"10-K"',
            "dateRange": "custom",
            "startdt": f"{year}-01-01",
            "enddt": f"{year}-12-31",
            "page": 1,
            "pageSize": 10,
        }
        data = self._fetcher(
            "https://efts.sec.gov/LATEST/search-index",
            headers=self._headers,
            method="POST",
            json=query,
        )
        if not data:
            return None
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return None
        hit = hits[0]["_source"]
        return {
            "cik": (hit.get("ciks") or [""])[0],
            "accession_number": hit.get("adsh"),
            "filing_date": hit.get("file_date"),
            "period_end": hit.get("period_ending"),
            "form": hit.get("form"),
        }

    def _fetch_filing_text(self, filing: dict) -> str | None:
        cik = filing["cik"]
        accession = filing["accession_number"].replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
        index_data = self._fetcher(index_url, headers=self._headers)
        if not index_data:
            return None

        doc = None
        for f in index_data.get("directory", {}).get("item", []):
            name = f.get("name", "").lower()
            if name.endswith((".htm", ".html")) or name == "filing.txt":
                doc = f["name"]
                break
        if not doc:
            return None

        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
        html = self._fetch_text(url, headers=self._headers)
        if not html:
            return None
        return _HTMLTextExtractor().feed(html)

    # -- default fetchers --------------------------------------------------

    @staticmethod
    def _http_get_json(url: str, headers: dict | None = None, method: str = "GET", json=None):
        import requests

        resp = requests.request(method, url, headers=headers, json=json, timeout=30)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT_SLEEP)
        return resp.json()

    @staticmethod
    def _fetch_text(url: str, headers: dict | None = None) -> str | None:
        import requests

        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT_SLEEP)
        return resp.text


# ---------------------------------------------------------------------------
# 10-K section extraction
# ---------------------------------------------------------------------------


_SKIP_TAGS = {"script", "style", "head"}


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text (stdlib only). Strips script/style content and
    collapses whitespace — sufficient for filing narrative text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def feed(self, data: str) -> str:  # type: ignore[override]
        super().feed(data)
        text = re.sub(r"\s+", " ", " ".join(self._parts))
        return text.strip()

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)


def _extract_mda(text: str) -> str | None:
    """Extract Item 7 (MD&A) from cleaned 10-K text."""
    start_patterns = [
        r"ITEM\s*7\.?\s*[\-\—]?\s*MANAGEMENT[\'\']?S\s*DISCUSSION\s*AND\s*ANALYSIS",
        r"MANAGEMENT[\'\']?S\s*DISCUSSION\s*AND\s*ANALYSIS\s*OF\s*FINANCIAL",
        r"ITEM\s*7\s*MANAGEMENT[\'\']?S\s*DISCUSSION",
    ]
    end_patterns = [
        r"ITEM\s*7A\.?\s*[\-\—]?\s*QUANTITATIVE\s*AND\s*QUALITATIVE",
        r"ITEM\s*8\.?\s*[\-\—]?\s*FINANCIAL\s*STATEMENTS",
        r"ITEM\s*8\s*FINANCIAL",
    ]
    return _extract_section(text, start_patterns, end_patterns, fallback_len=50_000, cap=30_000)


def _extract_risk_factors(text: str) -> str | None:
    """Extract Item 1A (Risk Factors) from cleaned 10-K text."""
    start_patterns = [
        r"ITEM\s*1A\.?\s*[\-\—]?\s*RISK\s*FACTORS",
        r"RISK\s*FACTORS",
    ]
    end_patterns = [
        r"ITEM\s*1B\.?\s*[\-\—]?\s*UNRESOLVED\s*STAFF\s*COMMENTS",
        r"ITEM\s*2\.?\s*[\-\—]?\s*PROPERTIES",
    ]
    return _extract_section(text, start_patterns, end_patterns, fallback_len=30_000, cap=20_000)


def _extract_section(
    text: str,
    start_patterns: list[str],
    end_patterns: list[str],
    fallback_len: int,
    cap: int,
) -> str | None:
    text_upper = text.upper()
    start_pos = None
    for pattern in start_patterns:
        match = re.search(pattern, text_upper)
        if match:
            start_pos = match.start()
            break
    if start_pos is None:
        return None

    end_pos = None
    for pattern in end_patterns:
        match = re.search(pattern, text_upper[start_pos + 1000 :])
        if match:
            end_pos = start_pos + 1000 + match.start()
            break
    if end_pos is None:
        end_pos = start_pos + fallback_len

    section = re.sub(r"\s+", " ", text[start_pos:end_pos].strip())
    return section[:cap]
