"""Tests for ported engines and data sources."""

import numpy as np
import pandas as pd

from equitylens.data_sources.edgar_source import (
    EdgarSource,
    _derived_metrics,
    _extract_annual_with_dates,
    is_observable,
)
from equitylens.data_sources.finnhub_source import FinnhubSource
from equitylens.engines import (
    FundamentalEngine,
    ReversalEngine,
    SentimentEngine,
    TechnicalEngine,
    TextEngine,
)
from equitylens.plugin import DataSource


class MockDataSource(DataSource):
    """Controllable mock: prices, fundamentals, and alternative data."""

    def __init__(self):
        self._prices = None
        self._fundamentals = None
        self._alternative = None

    def set_prices(self, df):
        self._prices = df

    def set_fundamentals(self, df):
        self._fundamentals = df

    def set_alternative(self, df):
        self._alternative = df

    def prices(self, tickers, start, end):
        if self._prices is not None:
            return self._prices
        return pd.DataFrame()

    def fundamentals(self, tickers, as_of=None):
        if self._fundamentals is not None:
            return self._fundamentals
        return pd.DataFrame()

    def alternative(self, tickers, kind, as_of=None):
        if self._alternative is not None:
            return self._alternative
        return None


def _price_frame(dates, ticker, adj_close, volume=None):
    volume = volume if volume is not None else [1e6] * len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": adj_close,
            "high": adj_close,
            "low": adj_close,
            "close": adj_close,
            "volume": volume,
            "adj_close": adj_close,
        }
    )


class TestReversalEngine:
    def test_recent_loser_scores_higher(self):
        # 4 months of prices; CRASH falls hard in the last week, RALLY rises
        dates = pd.date_range("2025-05-01", "2025-09-18", freq="B")
        n = len(dates)
        crash = [100.0] * (n - 5) + [100, 97, 94, 90, 85]
        rally = [100.0] * (n - 5) + [100, 103, 106, 110, 115]
        prices = pd.concat(
            [_price_frame(dates, "CRASH", crash), _price_frame(dates, "RALLY", rally)],
            ignore_index=True,
        )

        ds = MockDataSource()
        ds.set_prices(prices)

        out = ReversalEngine().compute(ds, ["CRASH", "RALLY"], as_of="2025-09-18")

        assert not out.scores.empty
        assert out.scores["CRASH"] > out.scores["RALLY"]
        assert out.scores.between(0, 100).all()


class TestTechnicalEngine:
    def test_uptrend_scores_higher(self):
        # UPTREND: steady rise, high volume. DOWNTREND: steady decline.
        dates = pd.date_range("2023-06-01", "2025-09-18", freq="B")
        n = len(dates)
        uptrend = [100 * (1.0008 ** i) for i in range(n)]
        downtrend = [100 * (0.9992 ** i) for i in range(n)]
        up_vol = [1e6 + i * 100 for i in range(n)]
        down_vol = [2e6 - i * 100 for i in range(n)]

        prices = pd.concat(
            [
                _price_frame(dates, "UPTREND", uptrend, up_vol),
                _price_frame(dates, "DOWNTREND", downtrend, down_vol),
            ],
            ignore_index=True,
        )

        ds = MockDataSource()
        ds.set_prices(prices)

        out = TechnicalEngine().compute(ds, ["UPTREND", "DOWNTREND"], as_of="2025-09-18")

        assert not out.scores.empty
        assert out.scores["UPTREND"] > out.scores["DOWNTREND"]
        assert out.scores.between(0, 100).all()
        # Uptrend should be above its 50MA/200MA with a golden cross
        assert out.raw_metrics.loc["UPTREND", "tech_above_50ma"] == 1.0


class TestSentimentEngine:
    def _sentiment_frame(self):
        return pd.DataFrame(
            [
                {
                    "ticker": "LOVED",
                    "total_buy": 8,
                    "total_hold": 2,
                    "total_sell": 0,
                    "upgrades_30d": 3,
                    "downgrades_30d": 0,
                    "upgrades_90d": 5,
                    "downgrades_90d": 0,
                    "num_analysts": 10,
                    "avg_earnings_surprise_pct": 8.0,
                    "days_until_earnings": 60,
                    "earnings_imminent": False,
                    "insider_net_value": 5e6,
                },
                {
                    "ticker": "HATED",
                    "total_buy": 0,
                    "total_hold": 1,
                    "total_sell": 9,
                    "upgrades_30d": 0,
                    "downgrades_30d": 4,
                    "upgrades_90d": 0,
                    "downgrades_90d": 6,
                    "num_analysts": 10,
                    "avg_earnings_surprise_pct": -8.0,
                    "days_until_earnings": 60,
                    "earnings_imminent": False,
                    "insider_net_value": -8e6,
                },
            ]
        )

    def test_bullish_sentiment_scores_higher(self):
        ds = MockDataSource()
        ds.set_alternative(self._sentiment_frame())

        out = SentimentEngine().compute(ds, ["LOVED", "HATED"], as_of="2025-09-18")

        assert not out.scores.empty
        assert out.scores["LOVED"] > out.scores["HATED"]
        assert out.scores.between(0, 100).all()
        assert out.confidence["LOVED"] > 0.5

    def test_no_data_returns_empty(self):
        ds = MockDataSource()
        out = SentimentEngine().compute(ds, ["AAPL"], as_of="2025-09-18")
        assert out.scores.empty
        assert "error" in out.metadata

    def test_uncovered_ticker_is_nan_not_neutral(self):
        frame = self._sentiment_frame()
        frame.loc[frame["ticker"] == "HATED", "num_analysts"] = 0
        frame.loc[frame["ticker"] == "HATED", "total_buy"] = 0
        frame.loc[frame["ticker"] == "HATED", "total_hold"] = 0
        frame.loc[frame["ticker"] == "HATED", "total_sell"] = 0
        ds = MockDataSource()
        ds.set_alternative(frame)

        out = SentimentEngine().compute(ds, ["HATED"], as_of="2025-09-18")
        # consensus component must be NaN (not 50) when coverage is absent
        assert pd.isna(out.raw_metrics.loc["HATED", "analyst_consensus_score"])


class TestFundamentalEngine:
    def test_quality_business_scores_higher(self):
        fund = pd.DataFrame(
            [
                {
                    "ticker": "COMPOUNDER",
                    "roic": 0.35,
                    "fcf_growth_1y": 0.25,
                    "cfo_ni_consistency": 1.4,
                    "accruals_ratio": -0.05,
                    "revenue_growth_3y": 0.18,
                    "earnings_consistency": 0.08,
                    "gross_margin_stability": 0.02,
                    "rd_intensity": 0.12,
                    "sga_efficiency": 0.15,
                    "buyback_yield": 0.03,
                    "piotroski_f": 8,
                    "sector": "Tech",
                },
                {
                    "ticker": "DECLINER",
                    "roic": -0.02,
                    "fcf_growth_1y": -0.40,
                    "cfo_ni_consistency": 0.4,
                    "accruals_ratio": 0.25,
                    "revenue_growth_3y": -0.10,
                    "earnings_consistency": 0.60,
                    "gross_margin_stability": 0.20,
                    "rd_intensity": 0.0,
                    "sga_efficiency": 0.55,
                    "buyback_yield": 0.0,
                    "piotroski_f": 2,
                    "sector": "Tech",
                },
            ]
        )
        ds = MockDataSource()
        ds.set_fundamentals(fund)

        out = FundamentalEngine().compute(ds, ["COMPOUNDER", "DECLINER"], as_of="2025-09-18")

        assert not out.scores.empty
        assert out.scores["COMPOUNDER"] > out.scores["DECLINER"]
        assert out.scores.between(0, 100).all()
        assert out.confidence["COMPOUNDER"] == 1.0


class _StubAnalyzer:
    """Duck-typed FinBERT stub for TextEngine tests."""

    def __init__(self, scores: dict):
        self._scores = scores  # text-prefix -> score

    def is_available(self):
        return True

    def analyze_long_text(self, text):
        for prefix, score in self._scores.items():
            if text.startswith(prefix):
                return {"score": score, "confidence": 0.9, "label": "Positive" if score > 0 else "Negative"}
        return {"score": 0.0, "confidence": 0.5, "label": "Neutral"}


class TestTextEngine:
    def _filing_frame(self):
        return pd.DataFrame(
            [
                # OPTIMISTIC: improving tone YoY, small MD&A rewrite
                {"ticker": "OPTIMISTIC", "year": 2023, "filing_date": "2024-02-15",
                 "mda_text": "badold " * 100, "risk_factors": "risky " * 50, "word_count": 40000},
                {"ticker": "OPTIMISTIC", "year": 2024, "filing_date": "2025-02-15",
                 "mda_text": "goodnew " * 100, "risk_factors": "fine " * 50, "word_count": 45000},
                # PESSIMISTIC: deteriorating tone
                {"ticker": "PESSIMISTIC", "year": 2023, "filing_date": "2024-02-15",
                 "mda_text": "goodold " * 100, "risk_factors": "fine " * 50, "word_count": 40000},
                {"ticker": "PESSIMISTIC", "year": 2024, "filing_date": "2025-02-15",
                 "mda_text": "badnew " * 100, "risk_factors": "risky " * 50, "word_count": 45000},
            ]
        )

    def test_tone_trend_scores_higher(self):
        analyzer = _StubAnalyzer({"goodnew": 0.8, "goodold": 0.6, "badnew": -0.8, "badold": -0.6})
        ds = MockDataSource()
        ds.set_alternative(self._filing_frame())

        out = TextEngine(analyzer=analyzer).compute(
            ds, ["OPTIMISTIC", "PESSIMISTIC"], as_of="2025-09-18"
        )

        assert not out.scores.empty
        assert out.scores["OPTIMISTIC"] > out.scores["PESSIMISTIC"]
        assert out.scores.between(0, 100).all()

    def test_unavailable_analyzer_returns_empty(self):
        class _Unavailable:
            def is_available(self):
                return False

        ds = MockDataSource()
        ds.set_alternative(self._filing_frame())
        out = TextEngine(analyzer=_Unavailable()).compute(ds, ["OPTIMISTIC"], as_of="2025-09-18")
        assert out.scores.empty
        assert "error" in out.metadata


class TestEdgarPIT:
    def test_is_observable(self):
        import datetime

        d = datetime.date(2023, 6, 30)
        assert is_observable("2023-02-01", as_of=d, lag_days=45) is True
        # Filed only 10 days before as_of → not yet observable
        assert is_observable("2023-06-20", as_of=d, lag_days=45) is False
        # Live mode: everything observable
        assert is_observable("2023-06-20", as_of=None) is True
        # Missing filing date with as_of → cannot prove → exclude
        assert is_observable("", as_of=d) is False
        assert is_observable("garbage", as_of=d) is False

    def test_extract_annual_with_dates_dedupes_by_fy(self):
        facts = {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fp": "FY", "fy": 2022, "end": "2022-12-31",
                         "filed": "2023-02-01", "val": 100},
                        {"form": "10-K", "fp": "FY", "fy": 2022, "end": "2022-12-31",
                         "filed": "2023-03-15", "val": 101},  # amended, wins
                        {"form": "10-K", "fp": "FY", "fy": 2023, "end": "2023-12-31",
                         "filed": "2024-02-10", "val": 120},
                        {"form": "10-Q", "fp": "Q2", "fy": 2023, "end": "2023-06-30",
                         "filed": "2023-08-01", "val": 55},  # quarterly, ignored
                    ]
                }
            }
        }
        out = _extract_annual_with_dates(facts, ["Revenues"])
        assert out == [("2022-12-31", "2023-03-15", 101), ("2023-12-31", "2024-02-10", 120)]

    def test_derived_metrics(self):
        fin = {
            "revenue": [("2020-12-31", 100.0), ("2021-12-31", 110.0),
                        ("2022-12-31", 125.0), ("2023-12-31", 140.0)],
            "operating_income": [("2023-12-31", 40.0)],
            "net_income": [("2023-12-31", 30.0)],
            "cfo": [("2021-12-31", 20.0), ("2022-12-31", 24.0), ("2023-12-31", 35.0)],
            "capex": [("2021-12-31", -5.0), ("2022-12-31", -5.0), ("2023-12-31", -6.0)],
            "total_assets": [("2022-12-31", 200.0), ("2023-12-31", 210.0)],
            "total_debt": [("2023-12-31", 50.0)],
            "total_equity": [("2023-12-31", 120.0)],
            "cash": [("2023-12-31", 20.0)],
            "gross_profit": [("2021-12-31", 50.0), ("2022-12-31", 55.0), ("2023-12-31", 62.0)],
            "shares_outstanding": [("2022-12-31", 100.0), ("2023-12-31", 98.0)],
            "rd_expense": [("2023-12-31", -10.0)],
            "sga_expense": [("2023-12-31", -15.0)],
        }
        m = _derived_metrics(fin)
        assert abs(m["revenue_growth_3y"] - ((140 / 100) ** (1 / 3) - 1)) < 1e-9
        assert m["roic"] > 0
        assert m["buyback_yield"] == 0.02
        assert m["accruals_ratio"] == (30 - 35) / 210
        assert not np.isnan(m["piotroski_f"])

    def test_fundamentals_respects_as_of(self):
        """A filing too new for as_of must not leak into derived metrics."""
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"form": "10-K", "fp": "FY", "fy": 2020, "end": "2020-12-31",
                                 "filed": "2021-02-15", "val": 90.0},
                                {"form": "10-K", "fp": "FY", "fy": 2021, "end": "2021-12-31",
                                 "filed": "2022-02-15", "val": 100.0},
                                {"form": "10-K", "fp": "FY", "fy": 2022, "end": "2022-12-31",
                                 "filed": "2023-02-15", "val": 1000.0},
                                {"form": "10-K", "fp": "FY", "fy": 2023, "end": "2023-12-31",
                                 "filed": "2024-02-15", "val": 1100.0},
                            ]
                        }
                    },
                }
            }
        }

        def fetcher(url, headers=None, method="GET", json=None):
            if "company_tickers" in url:
                return {"0": {"cik_str": 12345, "ticker": "ACME"}}
            return facts

        src = EdgarSource(fetcher=fetcher)
        # FY2022 filed 2023-02-15 + 45d → observable from 2023-04-01
        early = src.fundamentals(["ACME"], as_of="2023-01-01")
        late = src.fundamentals(["ACME"], as_of="2023-06-01")

        assert early.empty is False
        # Early: FY2020+FY2021 observable (2 years) → CV needs 3 → NaN
        assert np.isnan(early.iloc[0]["earnings_consistency"])
        # Late: FY2020..FY2022 observable (3 years) → CV computable
        assert not np.isnan(late.iloc[0]["earnings_consistency"])

    def test_prices_requires_delegate(self):
        src = EdgarSource()
        try:
            src.prices(["AAPL"], "2024-01-01", "2024-02-01")
            raised = False
        except NotImplementedError:
            raised = True
        assert raised


class TestFinnhubSource:
    def _source(self):
        responses = {
            ("stock/recommendation", "GOOD"): [
                {"period": "2025-05", "buy": 3, "strongBuy": 2, "hold": 2, "sell": 1, "strongSell": 0},
                {"period": "2025-06", "buy": 5, "strongBuy": 3, "hold": 1, "sell": 0, "strongSell": 0},
            ],
            ("stock/price-target", "GOOD"): {
                "targetMean": 150.0, "targetMedian": 145.0,
                "targetHigh": 180.0, "targetLow": 120.0, "numberOfAnalysts": 9,
            },
            ("stock/earnings", "GOOD"): [
                {"period": "2024-12-31", "actual": 1.5, "estimate": 1.3},
                {"period": "2025-03-31", "actual": 1.7, "estimate": 1.6},
            ],
            ("calendar/earnings", "GOOD"): {
                "earningsCalendar": [{"date": "2025-12-15", "hour": "bmo"}]
            },
            ("stock/insider-transactions", "GOOD"): {
                "data": [
                    {"change": 1000, "transactionPrice": 100.0, "transactionDate": "2025-08-01"},
                    {"change": -500, "transactionPrice": 110.0, "transactionDate": "2025-09-01"},
                ]
            },
        }

        def request_fn(endpoint, params, headers):
            return responses.get((endpoint, params["symbol"]))

        return FinnhubSource(api_key="test-key", request_fn=request_fn)

    def test_sentiment_frame_shape(self):
        src = self._source()
        df = src.alternative(["GOOD"], kind="sentiment", as_of="2025-09-18")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["num_analysts"] == 9  # 5 buy + 3 strongBuy + 1 hold
        assert row["upgrades_30d"] == 3  # bullish 8→5+3=8 vs 3+2=5
        assert row["avg_earnings_surprise_pct"] > 0
        assert row["insider_net_value"] == 1000 * 100.0 - 500 * 110.0
        assert row["days_until_earnings"] == (pd.Timestamp("2025-12-15") - pd.Timestamp("2025-09-18")).days

    def test_pit_filter_excludes_future_periods(self):
        src = self._source()
        # Before the second recommendation month and the recent earnings:
        # only 2025-05 rec period and 2024-12 surprise survive.
        df = src.alternative(["GOOD"], kind="sentiment", as_of="2025-05-31")
        row = df.iloc[0]
        assert row["num_analysts"] == 8  # 3 buy + 2 strongBuy + 2 hold + 1 sell
        assert row["upgrades_30d"] == 0  # no earlier period to compare
        # Insider transactions dated after 2025-05-31 are excluded
        assert row["insider_net_value"] == 0

    def test_unknown_kind_returns_none(self):
        src = self._source()
        assert src.alternative(["GOOD"], kind="nope", as_of="2025-09-18") is None
