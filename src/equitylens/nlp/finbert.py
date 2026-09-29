"""FinBERT sentiment analyzer (ProsusAI/finbert).

Lazy-loads transformers/torch on first use. If either is missing (or the
model fails to load), is_available() returns False and engines that depend
on it return empty scores instead of crashing.

Requires the `text` extra: pip install equitylens[text]
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
MAX_SEQUENCE_LENGTH = 512
CHUNK_OVERLAP = 100

_LABELS = ["positive", "neutral", "negative"]
_SCORE_MAP = {"negative": -1, "neutral": 0, "positive": 1}


class FinBERTAnalyzer:
    """Financial sentiment analyzer using FinBERT.

    Provides sentiment classification (positive/neutral/negative), a -1..+1
    score, and label confidence. Analyzes long texts via overlapping chunks.
    """

    def __init__(self, device: str | None = None):
        self._tokenizer = None
        self._model = None
        self._device = device or "cpu"

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            logger.warning(
                "transformers/torch not installed — FinBERT unavailable "
                "(pip install equitylens[text]); text engines will return empty scores"
            )
            return

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            logger.info(f"Loading FinBERT model on {self._device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
            self._model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
            self._model.to(self._device)
            self._model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self._tokenizer = None
            self._model = None

    def is_available(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    # -- public API ---------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """Sentiment of a single text (auto-truncated to 3000 chars)."""
        if not self.is_available():
            return self._empty_result()
        if not text or len(text.strip()) < 10:
            return self._empty_result()

        text = text[:3000]
        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            return self._result_from_probs(probs)
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            return self._empty_result()

    def analyze_long_text(self, text: str, chunk_size: int = 2000) -> dict:
        """Long-text sentiment: chunk with overlap, aggregate weighted by confidence."""
        if not self.is_available():
            return self._empty_result()
        if not text:
            return self._empty_result()
        if len(text) <= chunk_size:
            return self.analyze(text)

        results = [
            self.analyze(chunk)
            for chunk in self._chunk(text, chunk_size)[:5]
        ]
        results = [r for r in results if r["label"] != "Unknown"]
        return self._aggregate(results)

    def analyze_mda_changes(self, old_mda: str, new_mda: str) -> dict:
        """Sentiment change between two MD&A versions."""
        old_result = self.analyze_long_text(old_mda) if old_mda else self._empty_result()
        new_result = self.analyze_long_text(new_mda) if new_mda else self._empty_result()

        shift = new_result["score"] - old_result["score"]
        if shift > 0.3:
            trend = "improving"
        elif shift < -0.3:
            trend = "deteriorating"
        else:
            trend = "stable"

        return {
            "old_sentiment": old_result,
            "new_sentiment": new_result,
            "sentiment_shift": shift,
            "sentiment_trend": trend,
        }

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _chunk(text: str, chunk_size: int) -> list[str]:
        """Split into sentence-boundary-respecting overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                for i in range(end, max(start, end - 200), -1):
                    if text[i - 1] in ".?!":
                        end = i
                        break
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
        return chunks

    def _result_from_probs(self, probs) -> dict:
        pred_idx = int(np.argmax(probs))
        label = _LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        return {
            "label": label.capitalize(),
            "score": confidence if label == "positive" else (-confidence if label == "negative" else 0.0),
            "confidence": confidence,
            "probabilities": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2]),
            },
        }

    @staticmethod
    def _aggregate(results: list[dict]) -> dict:
        if not results:
            return FinBERTAnalyzer._empty_result()
        total_weight = sum(r["confidence"] for r in results)
        avg_score = sum(r["score"] * r["confidence"] for r in results) / total_weight
        if avg_score > 0.2:
            label = "Positive"
        elif avg_score < -0.2:
            label = "Negative"
        else:
            label = "Neutral"
        return {
            "label": label,
            "score": avg_score,
            "confidence": sum(r["confidence"] for r in results) / len(results),
            "chunk_count": len(results),
            "probabilities": None,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {"label": "Unknown", "score": 0.0, "confidence": 0.0, "probabilities": None}
