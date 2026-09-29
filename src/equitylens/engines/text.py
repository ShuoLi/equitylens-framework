"""Text engine — 10-K MD&A sentiment via FinBERT.

Sub-scores:
  1. MD&A sentiment        — FinBERT polarity of current-year MD&A
  2. Sentiment trend       — score change vs prior year
  3. Text change magnitude — large MD&A rewrites score lower (consistency)
  4. Risk factor sentiment — FinBERT polarity of Item 1A
  5. Disclosure quality    — full-filing word count (very short = concerning)

Data comes from `alternative(kind="10k_mda")` — EdgarSource serves it
point-in-time safe (only filings observable at as_of). The analyzer is
duck-typed: anything with is_available() and analyze_long_text() works,
so tests can inject a stub and heavy FinBERT deps stay optional
(pip install equitylens[text]).

Academic grounding: Loughran & McDonald (2011); Yang et al. (2020) FinBERT.
"""

from __future__ import annotations

import difflib
import logging

import numpy as np
import pandas as pd

from equitylens.plugin import DataSource, Engine, EngineOutput

logger = logging.getLogger(__name__)


class TextEngine(Engine):
    """10-K narrative sentiment scorer.

    Parameters
    ----------
    analyzer : object, optional
        FinBERT-like analyzer (is_available(), analyze_long_text(text) ->
        {"score": -1..1, "confidence": 0..1, "label": str}). Defaults to a
        lazily-constructed FinBERTAnalyzer; if transformers/torch are
        missing, the engine returns empty scores.
    """

    name = "text"

    def __init__(self, analyzer=None):
        self._analyzer = analyzer
        self._analyzer_tried = analyzer is not None

    def compute(self, data: DataSource, universe: list[str], as_of: str) -> EngineOutput:
        analyzer = self._get_analyzer()
        if analyzer is None or not analyzer.is_available():
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "finbert unavailable (pip install equitylens[text])"},
            )

        alt = data.alternative(universe, kind="10k_mda", as_of=as_of)
        if alt is None or alt.empty:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "no 10-K text data (source returned None)"},
            )

        alt = alt.sort_values("filing_date")
        rows = []
        for ticker in universe:
            filings = alt[alt["ticker"] == ticker]
            if filings.empty:
                continue
            current = filings.iloc[-1]
            previous = filings.iloc[-2] if len(filings) >= 2 else None
            rows.append(self._build_metrics(ticker, current, previous, analyzer))

        if not rows:
            return EngineOutput(
                scores=pd.Series(dtype=float),
                metadata={"error": "no filings found for universe"},
            )

        df = pd.DataFrame(rows).set_index("ticker").reindex(universe)

        components = pd.DataFrame(index=df.index)
        components["mda_sentiment_score"] = self._score_mda_sentiment(df)
        components["sentiment_trend_score"] = self._score_sentiment_trend(df)
        components["text_change_score"] = self._score_text_change(df)
        components["risk_sentiment_score"] = self._score_risk_sentiment(df)
        components["disclosure_quality_score"] = self._score_disclosure_quality(df)

        scores = components.mean(axis=1, skipna=True).clip(0, 100)

        # Confidence: analyzer confidence on the current MD&A
        conf = df["mda_confidence"].fillna(0).clip(0, 1)

        return EngineOutput(
            scores=scores,
            raw_metrics=df.join(components),
            confidence=conf,
            metadata={
                "as_of": as_of,
                "n_scored": int(scores.notna().sum()),
            },
        )

    # -- analyzer ------------------------------------------------------------

    def _get_analyzer(self):
        if not self._analyzer_tried:
            self._analyzer_tried = True
            try:
                from equitylens.nlp.finbert import FinBERTAnalyzer

                self._analyzer = FinBERTAnalyzer()
            except Exception as e:
                logger.warning(f"FinBERT analyzer unavailable: {e}")
                self._analyzer = None
        return self._analyzer

    # -- metrics per ticker ----------------------------------------------------

    def _build_metrics(self, ticker: str, current: pd.Series, previous: pd.Series | None, analyzer) -> dict:
        row = {"ticker": ticker}
        current_mda = current.get("mda_text") or ""
        previous_mda = previous.get("mda_text") if previous is not None else ""

        if current_mda:
            sentiment = analyzer.analyze_long_text(current_mda)
            row["mda_sentiment"] = sentiment.get("score", 0.0)
            row["mda_confidence"] = sentiment.get("confidence", 0.0)
            row["mda_label"] = sentiment.get("label", "Unknown")
        else:
            row["mda_sentiment"] = np.nan
            row["mda_confidence"] = 0.0
            row["mda_label"] = "Unknown"

        # Sentiment trend vs prior year
        if current_mda and previous_mda:
            prev_sentiment = analyzer.analyze_long_text(previous_mda)
            row["sentiment_shift"] = row["mda_sentiment"] - prev_sentiment.get("score", 0.0)
        else:
            row["sentiment_shift"] = 0.0

        # MD&A change magnitude (1 - similarity), 0-100
        if current_mda and previous_mda:
            similarity = difflib.SequenceMatcher(None, previous_mda, current_mda).ratio()
            row["mda_diff_score"] = (1 - similarity) * 100
        else:
            row["mda_diff_score"] = 0.0

        # Risk factor sentiment
        risk_text = current.get("risk_factors") or ""
        if risk_text:
            row["risk_sentiment"] = analyzer.analyze_long_text(risk_text).get("score", 0.0)
        else:
            row["risk_sentiment"] = np.nan

        row["total_word_count"] = current.get("word_count", 0) or 0
        row["filing_year"] = current.get("year")
        row["filing_date"] = current.get("filing_date")
        return row

    # -- component scorers -------------------------------------------------------

    @staticmethod
    def _score_mda_sentiment(df: pd.DataFrame) -> pd.Series:
        if "mda_sentiment" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        sentiment = df["mda_sentiment"].fillna(0)
        confidence = df.get("mda_confidence", pd.Series(1.0, index=df.index)).fillna(0)
        # -1..+1 → 0..100, downweighted toward neutral by analyzer confidence
        score = 50 + sentiment * 50 * confidence
        return score.clip(0, 100).where(df["mda_sentiment"].notna())

    @staticmethod
    def _score_sentiment_trend(df: pd.DataFrame) -> pd.Series:
        if "sentiment_shift" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        shift = df["sentiment_shift"].fillna(0)
        return (50 + shift * 50).clip(0, 100)

    @staticmethod
    def _score_text_change(df: pd.DataFrame) -> pd.Series:
        if "mda_diff_score" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        diff = df["mda_diff_score"].fillna(0)
        # Major rewrites = concern: 0% change = 100, >=50% change = 0
        return (100 - diff * 2).clip(0, 100)

    @staticmethod
    def _score_risk_sentiment(df: pd.DataFrame) -> pd.Series:
        if "risk_sentiment" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        risk = df["risk_sentiment"]
        return (50 + risk.fillna(0) * 50).clip(0, 100).where(risk.notna())

    @staticmethod
    def _score_disclosure_quality(df: pd.DataFrame) -> pd.Series:
        if "total_word_count" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        wc = df["total_word_count"].fillna(0)
        score = 40 + (wc / 2000).clip(0, 30)
        score = score.where(wc >= 10_000, wc / 200)
        score = score.where(wc <= 150_000, 80)
        return score.clip(0, 100)
