# EquityLens Framework

**Pluggable multi-factor equity research.** Bring your own data sources, signal
engines, and portfolio rules — the framework handles orchestration, caching,
point-in-time correctness, and attribution.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

Most quant research code rots the same way: hardcoded API calls, silent
look-ahead bias, and monoliths you can't test. This framework is built to
prevent exactly that. It was extracted from a working 7-engine research
application (S&P 1000 scoring, SEC filing parsing, FinBERT 10-K sentiment)
and rebuilt around three protocols — `DataSource`, `Engine`, `PortfolioRule`
— with correctness enforced at the interface level rather than by convention.

## Design principles

1. **Point-in-time by default** — every data interface accepts `as_of`;
   implementations must respect it. `EdgarSource` filters every metric by SEC
   filing date with a 45-day observability lag, so backtests can't see filings
   that didn't exist yet.
2. **Inject, don't import** — engines receive a `DataSource` and never call
   yfinance/EDGAR/Finnhub directly. Swap data vendors without touching a signal.
3. **Stateless engines** — config in `__init__`, no mutation during `compute()`.
4. **Confidence-aware** — every engine reports per-ticker data quality, and the
   pipeline can gate on it. A missing dataset yields *no* score, never a fake
   neutral one.
5. **Attribution-friendly** — intermediate metrics are preserved in every
   `EngineOutput`, so you can always answer "why did this stock score high?"

## Quick start

```bash
git clone https://github.com/ShuoLi/equitylens-framework.git
cd equitylens-framework
pip install -e .
```

```python
from equitylens import Pipeline
from equitylens.data_sources import YFinanceSource
from equitylens.engines import MomentumEngine, QualityEngine, ValueEngine

pipeline = Pipeline(YFinanceSource())
pipeline.register(MomentumEngine(lookback_days=126), weight=0.30)
pipeline.register(QualityEngine(), weight=0.45)
pipeline.register(ValueEngine(use_forward_pe=True), weight=0.25)

scores = pipeline.run(["AAPL", "MSFT", "NVDA", "GOOGL"], as_of="2025-09-18")
print(scores)
```

More complete examples — including backtesting with turnover-aware rules and a
custom alternative-data engine — live in [`examples/`](examples/).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ DataSource  │────▶│   Engine(s)  │────▶│  Portfolio   │
│  (plugin)   │     │   (plugins)  │     │   Rule       │
│ yfinance /  │     │ momentum /   │     │  (plugin)    │
│ edgar /     │     │ quality /    │     │ equal weight │
│ finnhub /   │     │ value /      │     │ score weight │
│ polygon /.. │     │ reversal /   │     │ turnover-    │
│             │     │ technical /  │     │ aware        │
│             │     │ sentiment /  │     └──────────────┘
│             │     │ fundamental /│            │
│             │     │ text / ...   │            │
└─────────────┘     └──────────────┘            │
       │                   │                    │
       └───────────────────┴────────────────────┘
                           │
                    ┌──────────────┐
                    │   Pipeline   │
                    │  orchestrate │
                    │  cache / PIT │
                    │  blend / IC  │
                    └──────────────┘
```

## Built-in engines

Eight engines, all stateless, all returning `EngineOutput(scores, raw_metrics,
confidence, metadata)`:

| Engine | Signal family | Needs |
|--------|---------------|-------|
| `MomentumEngine` | Risk-adjusted momentum with skip-days (crash avoidance) | `prices()` |
| `ReversalEngine` | Vol-normalized 1W/1M mean reversion (offsets momentum) | `prices()` |
| `TechnicalEngine` | MA structure, Bollinger %B, OBV/volume confirmation, relative strength | `prices()` |
| `QualityEngine` | ROE, margins, growth, leverage | `fundamentals()` |
| `ValueEngine` | Forward P/E, PEG, P/S, P/B with outlier caps | `fundamentals()` |
| `FundamentalEngine` | EDGAR deep-dive: ROIC, accruals, Piotroski F, margin stability | EDGAR-derived fundamentals |
| `SentimentEngine` | Analyst consensus, upgrade/downgrade momentum, earnings surprises, insider flow | `alternative("sentiment")` |
| `TextEngine` | FinBERT 10-K MD&A sentiment + year-over-year tone shift | `alternative("10k_mda")` + `[text]` extra |

Every engine degrades gracefully: if its data isn't available (missing API key,
optional extra not installed), it returns empty scores with a note in
`metadata["error"]` and the remaining engines still blend. One broken
dependency never crashes a pipeline.

## Built-in data sources

| Source | prices | fundamentals | alternative | PIT status |
|--------|--------|--------------|-------------|------------|
| `YFinanceSource` | ✓ | ✓ | — | Prices ✓; fundamentals **not** (latest filings only) |
| `EdgarSource` | delegates | ✓ filing-date filtered | 10-K MD&A + risk factors | **Yes** — 45-day lag after filing |
| `FinnhubSource` | delegates | delegates | analyst / earnings / insider | Historical rows filtered to ≤ `as_of`; consensus targets are live-only |

Install optional dependencies per source:

```bash
pip install -e ".[edgar]"     # SEC EDGAR fundamentals + 10-K text
pip install -e ".[finnhub]"   # analyst/earnings/insider data
pip install -e ".[text]"      # FinBERT (transformers + torch)
```

## Blending and confidence gating

Engines blend by fixed weights or by their historical Information Coefficient,
and can be skipped when data quality is too low:

```python
pipeline.register(MomentumEngine(), weight=0.30, min_confidence=0.6)

# Record IC observations as you accumulate them
pipeline.record_ic("momentum", ic_value=0.04, date="2025-09-12")

scores = pipeline.run(universe, as_of="2025-09-18", blend_method="ic_weighted")
```

## Backtesting

Walk-forward simulation with transaction costs, reusing the same pipeline —
so a backtest scores exactly what live scoring would:

```python
from equitylens.backtest import BacktestEngine
from equitylens.portfolio import TurnoverAwareTopN

bt = BacktestEngine(
    pipeline=pipeline,
    portfolio_rule=TurnoverAwareTopN(n=20, turnover_penalty=0.3),
    rebalance_freq="W",
    transaction_cost_bps=10,
)
result = bt.run(universe, start="2024-01-01", end="2025-09-01")
print(result.stats)
```

## Writing plugins

Any `DataSource`, `Engine`, or `PortfolioRule` that satisfies the protocol
drops straight into the pipeline — and can be distributed as a pip package
with entry points for auto-discovery. Full guide with worked examples (Polygon
source, earnings-surprise engine, risk-parity rule): [`docs/PLUGINS.md`](docs/PLUGINS.md).

Architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Honest limitations

- `YFinanceSource` fundamentals have look-ahead bias in backtests (it returns
  the latest filing regardless of `as_of`). For PIT-correct fundamentals, use
  `EdgarSource` — that's why it exists.
- Finnhub free-tier consensus price targets are a current snapshot; treat them
  as live-only, not backtest-safe.
- Universe membership is the caller's responsibility — survivorship-free
  point-in-time index membership is on the roadmap.
- No regime detection yet (the research app this was extracted from used an
  HMM macro layer; porting it is on the roadmap).

## Roadmap

- Regime-conditional engine weights (HMM macro layer)
- HRP / style-factor-neutral portfolio construction rules
- Point-in-time universe membership (survivorship bias)
- CI pipeline with coverage reporting

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/        # 22 tests, no network access required
```

Tests cover every engine against synthetic data with known orderings, plus the
point-in-time machinery itself — including a test that a filing too new for
`as_of` cannot leak into derived metrics.

## License

MIT
