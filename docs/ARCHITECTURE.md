# Architecture

## Design Principles

1. **PIT by default** — every data interface accepts `as_of`; implementations must respect it
2. **Inject, don't import** — engines receive a `DataSource`, never call yfinance/EDGAR directly
3. **Stateless engines** — config in `__init__`, no mutation during `compute()`
4. **Confidence-aware** — every engine reports data quality; pipeline can gate on it
5. **Attribution-friendly** — `raw_metrics` preserved for debugging and factor attribution

## Module Map

```
src/equitylens/
├── plugin.py          # DataSource, Engine, EngineOutput, PortfolioRule protocols
├── pipeline.py        # Pipeline orchestrator (register, run, blend, portfolio)
├── cache.py           # SQLite cache with TTL
├── data_sources/
│   └── yfinance_source.py   # Reference implementation (NOT PIT-safe for fundamentals)
├── engines/
│   ├── momentum.py          # Cross-sectional price momentum
│   ├── quality.py           # Fundamental quality (ROE, margins, growth, low debt)
│   └── value.py             # Multi-metric valuation (fwd P/E, PEG, P/S, P/B)
├── portfolio/
│   └── rules.py             # EqualWeightTopN, ScoreWeightedTopN, TurnoverAwareTopN
└── backtest/
    └── engine.py            # Walk-forward simulation with transaction costs
```

## Data Flow

```
User calls pipeline.run(universe, as_of)
         │
         ▼
    ┌────────────┐
    │  Pipeline  │
    └────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Engine A   Engine B   ...   (each receives DataSource + universe + as_of)
    │         │
    ▼         ▼
EngineOutput({scores, raw_metrics, confidence, metadata})
    │         │
    └────┬────┘
         ▼
    ┌────────────┐
    │   Blend    │   weighted_average | ic_weighted
    └────────────┘
         │
         ▼
    pd.Series (composite scores)
```

## Blending Methods

| Method | Description | When to use |
|--------|-------------|-------------|
| `weighted_average` | Fixed weights from `pipeline.register()` | Initial development, no IC history |
| `ic_weighted` | Weights proportional to historical IC | Production, after accumulating IC observations |

## Caching

- Key: `("engine", engine_name, as_of, universe_hash)`
- TTL: configurable per Cache instance (default 72h)
- Storage: SQLite BLOB via pickle

## PIT Enforcement

The framework enforces PIT at the interface level:

- `DataSource.fundamentals(tickers, as_of)` — must return only data filed ≤ `as_of`
- `DataSource.prices(tickers, start, end)` — end ≤ `as_of` is caller's responsibility
- `Engine.compute(data, universe, as_of)` — receives `as_of`, should pass to data calls

`YFinanceSource` is explicitly **not** PIT-safe for fundamentals. It returns latest filings. For backtests, use a PIT-safe source or accept the lookahead bias.
