# EquityLens Framework

Pluggable multi-factor equity research. Bring your own data source, signal engines, and portfolio rules — the framework handles orchestration, caching, point-in-time correctness, and attribution.

## Quick Start

```bash
pip install equitylens
```

```python
from equitylens import Pipeline
from equitylens.data_sources import YFinanceSource
from equitylens.engines import MomentumEngine, QualityEngine, ValueEngine
from equitylens.portfolio import EqualWeightTopN

pipeline = Pipeline(YFinanceSource())
pipeline.register(MomentumEngine(lookback_days=126), weight=0.30)
pipeline.register(QualityEngine(), weight=0.45)
pipeline.register(ValueEngine(use_forward_pe=True), weight=0.25)

scores = pipeline.run(["AAPL", "MSFT", "NVDA", "GOOGL"], as_of="2025-09-18")
print(scores)
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ DataSource  │────▶│   Engine(s)  │────▶│  Portfolio   │
│  (plugin)   │     │   (plugins)  │     │   Rule       │
│ yfinance /  │     │ momentum /   │     │  (plugin)    │
│ edgar /     │     │ quality /    │     │ equal weight │
│ finnhub /   │     │ value /      │     │ score weight │
│ polygon /.. │     │ reversal /   │     └──────────────│
│             │     │ technical /  │            │
│             │     │ sentiment /  │            │
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

## Built-in Engines and Sources

| Component | What it needs | Notes |
|-----------|---------------|-------|
| `MomentumEngine`, `ReversalEngine`, `TechnicalEngine` | `prices()` | work with any price source |
| `QualityEngine`, `ValueEngine` | `fundamentals()` | any source with standard columns |
| `FundamentalEngine` | `fundamentals()` with EDGAR-derived columns | pair with `EdgarSource` for PIT safety |
| `SentimentEngine` | `alternative(kind="sentiment")` | pair with `FinnhubSource` (`pip install equitylens[finnhub]`) |
| `TextEngine` | `alternative(kind="10k_mda")` + analyzer | `EdgarSource` + `equitylens[text]` (FinBERT); returns empty scores without them |
| `EdgarSource` | — | `pip install equitylens[edgar]`; PIT-safe via SEC filing dates |
| `FinnhubSource` | API key | rate-limited to 60 req/min; delegates `prices()`/`fundamentals()` to a wrapped source |

Engines return empty scores (with a note in `metadata["error"]`) when their
data isn't available, so a missing API key or optional extra never crashes a
pipeline — the remaining engines still blend.

## Plugin Interfaces

### DataSource

```python
from equitylens import DataSource

class MyDataSource(DataSource):
    def prices(self, tickers, start, end):
        return pd.DataFrame(...)  # date, ticker, open, high, low, close, volume, adj_close

    def fundamentals(self, tickers, as_of=None):
        return pd.DataFrame(...)  # one row per ticker, PIT-safe if as_of given

    def alternative(self, tickers, kind, as_of=None):
        return None  # optional
```

### Engine

```python
from equitylens import Engine, EngineOutput

class MyEngine(Engine):
    name = "my_signal"

    def compute(self, data, universe, as_of):
        px = data.prices(universe, start="2025-01-01", end=as_of)
        # ... compute signal ...
        return EngineOutput(
            scores=pd.Series({...}),
            raw_metrics=pd.DataFrame({...}),
            confidence=pd.Series({...}),
            metadata={"version": "1.0"},
        )
```

### PortfolioRule

```python
from equitylens import PortfolioRule

class MyRule(PortfolioRule):
    def target_weights(self, scores, prices, current_weights):
        return pd.Series({...})  # weights sum to ~1.0
```

## Backtesting

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

## Point-in-Time Correctness

The framework enforces PIT at the `DataSource` interface level. When `as_of` is passed to `fundamentals()` or `prices()`, implementations must only return data that was public knowledge on that date.

- `YFinanceSource` is **NOT** PIT-safe for fundamentals (returns latest filings).
- For PIT-correct backtests, use a source that tracks filing dates: SEC EDGAR, Polygon, SimFin, etc.

## IC-Weighted Blending

```python
# Record IC observations as you get them
pipeline.record_ic("momentum", ic_value=0.04, date="2025-09-12")
pipeline.record_ic("quality", ic_value=0.06, date="2025-09-12")

# Blend by IC instead of fixed weights
scores = pipeline.run(universe, as_of="2025-09-18", blend_method="ic_weighted")
```

## License

MIT
