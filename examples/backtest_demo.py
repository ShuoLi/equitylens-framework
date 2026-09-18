"""Reference implementation: backtest three-factor model over 2024-2025."""

from equitylens import Pipeline
from equitylens.backtest import BacktestEngine
from equitylens.data_sources import YFinanceSource
from equitylens.engines import MomentumEngine, QualityEngine, ValueEngine
from equitylens.portfolio import TurnoverAwareTopN

UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "V"]

pipeline = Pipeline(YFinanceSource())
pipeline.register(MomentumEngine(lookback_days=126, skip_days=5), weight=0.30)
pipeline.register(QualityEngine(), weight=0.45)
pipeline.register(ValueEngine(use_forward_pe=True), weight=0.25)

rule = TurnoverAwareTopN(n=5, turnover_penalty=0.3, max_weight=0.25)

bt = BacktestEngine(
    pipeline=pipeline,
    portfolio_rule=rule,
    rebalance_freq="W",
    transaction_cost_bps=10,
)

if __name__ == "__main__":
    result = bt.run(UNIVERSE, start="2024-01-01", end="2025-09-01")
    print("=== Backtest Stats ===")
    for k, v in result.stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
