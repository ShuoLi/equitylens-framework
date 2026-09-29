"""Finnhub data source — analyst, earnings, and insider alternative data.

Serves the `alternative()` interface used by the sentiment engine:
  kind="sentiment"   one row per ticker combining all five datasets below
  kind="analyst"     buy/hold/sell counts + 30d/90d upgrade/downgrade deltas
  kind="price_target" mean/median/high/low consensus targets
  kind="earnings_surprises" last 4 quarters, avg surprise, beat/miss counts
  kind="earnings_calendar" next earnings date, days until (relative to as_of)
  kind="insider"     net shares/value, buy/sell counts (last 3 months)

Free tier: 60 requests/minute (enforced internally). Requests are cached
per endpoint for the lifetime of the instance; pair with Pipeline(cache=...)
for persistent engine-level caching.

PIT notes: insider transactions and earnings surprises are filtered to rows
dated on or before as_of. Analyst recommendations are filtered to monthly
periods ending on or before as_of. Consensus price targets are a current
snapshot on the free tier — treat as live-only data, not backtest-safe.

Requires the `finnhub` extra: pip install equitylens[finnhub]
"""

from __future__ import annotations

import datetime
import logging
import time
from collections import deque

import pandas as pd

from equitylens.plugin import DataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"
FREE_TIER_RPM = 60

SENTIMENT_KINDS = {
    "analyst",
    "price_target",
    "earnings_surprises",
    "earnings_calendar",
    "insider",
    "sentiment",
}


class FinnhubSource(DataSource):
    """Alternative (analyst/earnings/insider) data from Finnhub.

    Parameters
    ----------
    api_key : str
        Finnhub API key (free tier works).
    price_source : DataSource, optional
        Finnhub free tier has no OHLCV here; delegate prices() to this source.
    fundamentals_source : DataSource, optional
        Delegate fundamentals() to this source.
    rate_per_minute : int
        Self-imposed request cap (default 60 = free tier).
    request_fn : callable, optional
        (method, url, params, headers) -> JSON. Injectable for tests;
        defaults to requests with the built-in rate limiter.
    """

    def __init__(
        self,
        api_key: str,
        price_source: DataSource | None = None,
        fundamentals_source: DataSource | None = None,
        rate_per_minute: int = FREE_TIER_RPM,
        request_fn=None,
    ):
        self.api_key = api_key
        self._price_source = price_source
        self._fundamentals_source = fundamentals_source
        self._max_per_window = rate_per_minute
        self._request_times: deque[float] = deque()
        self._forbidden: set[str] = set()
        self._cache: dict[tuple, object] = {}
        self._request_fn = request_fn

    # -- delegated interface ----------------------------------------------

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self._price_source is None:
            raise NotImplementedError(
                "FinnhubSource has no price feed; construct with price_source=..."
            )
        return self._price_source.prices(tickers, start, end)

    def fundamentals(self, tickers: list[str], as_of: str | None = None) -> pd.DataFrame:
        if self._fundamentals_source is None:
            raise NotImplementedError(
                "FinnhubSource has no fundamentals feed; construct with fundamentals_source=..."
            )
        return self._fundamentals_source.fundamentals(tickers, as_of=as_of)

    # -- alternative data ---------------------------------------------------

    def alternative(self, tickers: list[str], kind: str, as_of: str | None = None):
        if kind not in SENTIMENT_KINDS:
            return None
        as_of_date = datetime.date.fromisoformat(as_of) if as_of else None

        if kind == "sentiment":
            rows = [self._sentiment_row(t, as_of_date) for t in tickers]
            return pd.DataFrame(rows)

        builders = {
            "analyst": self._analyst_frame,
            "price_target": self._price_target_frame,
            "earnings_surprises": self._earnings_frame,
            "earnings_calendar": self._calendar_frame,
            "insider": self._insider_frame,
        }
        return builders[kind](tickers, as_of_date)

    def _sentiment_row(self, ticker: str, as_of: datetime.date | None) -> dict:
        row = {"ticker": ticker}
        row.update(self._analyst_metrics(ticker, as_of))
        row.update(self._price_target_metrics(ticker))
        row.update(self._earnings_metrics(ticker, as_of))
        row.update(self._calendar_metrics(ticker, as_of))
        row.update(self._insider_metrics(ticker, as_of))
        return row

    # -- analysts -----------------------------------------------------------

    def _analyst_frame(self, tickers: list[str], as_of: datetime.date | None) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": t, **self._analyst_metrics(t, as_of)} for t in tickers]
        )

    def _analyst_metrics(self, ticker: str, as_of: datetime.date | None) -> dict:
        data = self._get("stock/recommendation", {"symbol": ticker}, ("rec", ticker))
        out = {
            "total_buy": 0,
            "total_hold": 0,
            "total_sell": 0,
            "upgrades_30d": 0,
            "downgrades_30d": 0,
            "upgrades_90d": 0,
            "downgrades_90d": 0,
            "num_analysts": 0,
        }
        if not isinstance(data, list):
            return out

        periods = [d for d in data if isinstance(d, dict)]
        if as_of is not None:
            periods = [d for d in periods if _period_end(d.get("period", "")) <= as_of]
        if not periods:
            return out

        periods.sort(key=lambda d: d.get("period", ""))
        latest = periods[-1]
        out["total_buy"] = latest.get("buy", 0) + latest.get("strongBuy", 0)
        out["total_hold"] = latest.get("hold", 0)
        out["total_sell"] = latest.get("sell", 0) + latest.get("strongSell", 0)
        out["num_analysts"] = out["total_buy"] + out["total_hold"] + out["total_sell"]

        def net(d):
            bullish = d.get("buy", 0) + d.get("strongBuy", 0)
            bearish = d.get("sell", 0) + d.get("strongSell", 0)
            return bullish, bearish

        if len(periods) >= 2:
            lb_now, bb_now = net(latest)
            lb_prev, bb_prev = net(periods[-2])
            out["upgrades_30d"] = max(0, lb_now - lb_prev)
            out["downgrades_30d"] = max(0, bb_now - bb_prev)
        if len(periods) >= 4:
            lb_now, bb_now = net(latest)
            lb_old, bb_old = net(periods[-4])
            out["upgrades_90d"] = max(0, lb_now - lb_old)
            out["downgrades_90d"] = max(0, bb_now - bb_old)
        return out

    def _price_target_frame(self, tickers: list[str], as_of: datetime.date | None) -> pd.DataFrame:
        return pd.DataFrame([{"ticker": t, **self._price_target_metrics(t)} for t in tickers])

    def _price_target_metrics(self, ticker: str) -> dict:
        data = self._get("stock/price-target", {"symbol": ticker}, ("pt", ticker))
        if not isinstance(data, dict):
            return {"target_mean": None, "target_median": None, "target_high": None,
                    "target_low": None, "num_price_targets": 0}
        return {
            "target_mean": data.get("targetMean"),
            "target_median": data.get("targetMedian"),
            "target_high": data.get("targetHigh"),
            "target_low": data.get("targetLow"),
            "num_price_targets": data.get("numberOfAnalysts", 0),
        }

    # -- earnings -----------------------------------------------------------

    def _earnings_frame(self, tickers: list[str], as_of: datetime.date | None) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": t, **self._earnings_metrics(t, as_of)} for t in tickers]
        )

    def _earnings_metrics(self, ticker: str, as_of: datetime.date | None) -> dict:
        data = self._get("stock/earnings", {"symbol": ticker}, ("earn", ticker))
        out = {"avg_earnings_surprise_pct": None, "beat_count": 0, "miss_count": 0}
        if not isinstance(data, list):
            return out

        quarters = sorted(
            (d for d in data if isinstance(d, dict)),
            key=lambda d: d.get("period", ""),
            reverse=True,
        )
        if as_of is not None:
            quarters = [q for q in quarters if _period_end(q.get("period", "")) <= as_of]
        surprises = []
        for q in quarters[:4]:
            actual, estimate = q.get("actual"), q.get("estimate")
            if actual is not None and estimate not in (None, 0):
                surprises.append((actual - estimate) / abs(estimate) * 100)
        if surprises:
            out["avg_earnings_surprise_pct"] = sum(surprises) / len(surprises)
            out["beat_count"] = sum(1 for s in surprises if s > 0)
            out["miss_count"] = sum(1 for s in surprises if s < 0)
        return out

    def _calendar_frame(self, tickers: list[str], as_of: datetime.date | None) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": t, **self._calendar_metrics(t, as_of)} for t in tickers]
        )

    def _calendar_metrics(self, ticker: str, as_of: datetime.date | None) -> dict:
        ref = as_of or datetime.date.today()
        data = self._get(
            "calendar/earnings",
            {"symbol": ticker, "from": ref.isoformat(), "to": _plus_days(ref, 90)},
            ("cal", ticker, ref.isoformat()),
        )
        out = {"next_earnings_date": None, "days_until_earnings": None, "earnings_imminent": False}
        if not isinstance(data, dict):
            return out
        dates = []
        for e in data.get("earningsCalendar", []):
            d = _parse_date(e.get("date"))
            if d and d >= ref:
                dates.append(d)
        if not dates:
            return out
        nxt = min(dates)
        days = (nxt - ref).days
        return {
            "next_earnings_date": nxt.isoformat(),
            "days_until_earnings": days,
            "earnings_imminent": days <= 14,
        }

    # -- insider --------------------------------------------------------------

    def _insider_frame(self, tickers: list[str], as_of: datetime.date | None) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": t, **self._insider_metrics(t, as_of)} for t in tickers]
        )

    def _insider_metrics(self, ticker: str, as_of: datetime.date | None) -> dict:
        ref = as_of or datetime.date.today()
        data = self._get(
            "stock/insider-transactions",
            {"symbol": ticker, "from": _minus_days(ref, 90)},
            ("ins", ticker, ref.isoformat()),
        )
        out = {"insider_net_shares": 0, "insider_net_value": 0.0,
               "insider_buy_count": 0, "insider_sell_count": 0}
        if not isinstance(data, dict):
            return out
        for t in data.get("data", [])[:20]:
            change = t.get("change", 0) or 0
            if change == 0:
                continue
            if as_of is not None:
                tx_date = _parse_date(t.get("transactionDate"))
                if tx_date and tx_date > as_of:
                    continue
            price = t.get("transactionPrice", 0) or 0
            out["insider_net_shares"] += change
            out["insider_net_value"] += change * price
            if change > 0:
                out["insider_buy_count"] += 1
            else:
                out["insider_sell_count"] += 1
        return out

    # -- HTTP plumbing ---------------------------------------------------------

    def _get(self, endpoint: str, params: dict, cache_key: tuple):
        if endpoint in self._forbidden:
            return None
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.api_key:
            logger.warning("FinnhubSource: no API key configured")
            return None

        data = self._do_request(endpoint, params)
        if data is not None:
            self._cache[cache_key] = data
        return data

    def _do_request(self, endpoint: str, params: dict):
        """Returns parsed JSON, or None (marking forbidden endpoints on 403/empty)."""
        if self._request_fn is not None:
            return self._request_fn(endpoint, params, {"X-Finnhub-Token": self.api_key})

        import requests

        self._rate_limit()
        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, params=params, headers={"X-Finnhub-Token": self.api_key},
                                timeout=30)
            resp.raise_for_status()
            if not resp.text.strip():
                self._forbidden.add(endpoint)  # not available on this plan
                return None
            return resp.json()
        except ValueError:
            self._forbidden.add(endpoint)
            return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.warning(f"Finnhub 403 on {endpoint} — skipping this endpoint")
                self._forbidden.add(endpoint)
            else:
                logger.warning(f"Finnhub API error ({endpoint}): {e}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Finnhub API error ({endpoint}): {e}")
            return None

    def _rate_limit(self):
        now = time.time()
        while self._request_times and self._request_times[0] < now - 60:
            self._request_times.popleft()
        if len(self._request_times) >= self._max_per_window:
            sleep_time = self._request_times[0] - (now - 60) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_times.append(time.time())


# ---------------------------------------------------------------------------
# date helpers
# ---------------------------------------------------------------------------


def _parse_date(s: str | None) -> datetime.date | None:
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _period_end(period: str) -> datetime.date:
    """Finnhub periods are 'YYYY-MM' — map to the last day of that month."""
    try:
        year, month = int(period[:4]), int(period[5:7])
        if month == 12:
            return datetime.date(year, 12, 31)
        return datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    except (ValueError, IndexError):
        return datetime.date.max  # unparseable → excluded by PIT filters


def _plus_days(d: datetime.date, n: int) -> str:
    return (d + datetime.timedelta(days=n)).isoformat()


def _minus_days(d: datetime.date, n: int) -> str:
    return (d - datetime.timedelta(days=n)).isoformat()
