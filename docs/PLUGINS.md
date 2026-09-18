# How to Write a Plugin

## DataSource Plugin

Create a new file, e.g., `my_polygon_source.py`:

```python
import pandas as pd
from polygon import RESTClient
from equitylens import DataSource

class PolygonSource(DataSource):
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)

    def prices(self, tickers, start, end):
        # Fetch and normalize to: date, ticker, open, high, low, close, volume, adj_close
        ...

    def fundamentals(self, tickers, as_of=None):
        # Fetch and return one row per ticker
        # If as_of is provided, only return data filed on or before that date
        ...
```

Then inject it:

```python
from equitylens import Pipeline
pipeline = Pipeline(PolygonSource("your-api-key"))
```

## Engine Plugin

```python
from equitylens import Engine, EngineOutput

class EarningsSurpriseEngine(Engine):
    name = "earnings_surprise"

    def __init__(self, surprise_window_days: int = 5):
        self.surprise_window_days = surprise_window_days

    def compute(self, data, universe, as_of):
        # Fetch data via the injected DataSource
        alt = data.alternative(universe, kind="earnings_dates", as_of=as_of)
        if alt is None:
            return EngineOutput(scores=pd.Series(dtype=float))

        # Compute surprise magnitude
        alt["surprise"] = (alt["actual_eps"] - alt["consensus_eps"]).abs()
        scores = alt.groupby("ticker")["surprise"].mean().rank(pct=True) * 100

        return EngineOutput(
            scores=scores,
            raw_metrics=alt,
            confidence=pd.Series(0.8, index=scores.index),
            metadata={"window": self.surprise_window_days},
        )
```

Register it:

```python
pipeline.register(EarningsSurpriseEngine(), weight=0.10)
```

## PortfolioRule Plugin

```python
from equitylens import PortfolioRule

class RiskParityTopN(PortfolioRule):
    def __init__(self, n: int = 20):
        self.n = n

    def target_weights(self, scores, prices, current_weights):
        top = scores.nlargest(self.n)
        # Inverse-volatility weighting
        vol = prices.pivot_table(index="date", columns="ticker", values="adj_close").pct_change().std()
        inv_vol = 1 / vol.reindex(top.index)
        w = inv_vol / inv_vol.sum()
        return w
```

## Distributing as a Package

Others can `pip install` your plugin and use entry points for auto-discovery:

```toml
# pyproject.toml
[project.entry-points."equitylens.engines"]
earnings_surprise = "my_package:EarningsSurpriseEngine"
```

```toml
[project.entry-points."equitylens.data_sources"]
polygon = "my_package:PolygonSource"
```
