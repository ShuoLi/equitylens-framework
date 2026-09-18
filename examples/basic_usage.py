"""Reference implementation: three-factor model with weekly rebalance."""

from equitylens import Pipeline
from equitylens.backtest import BacktestEngine
from equitylens.data_sources import YFinanceSource
from equitylens.engines import MomentumEngine, QualityEngine, ValueEngine
from equitylens.portfolio import TurnoverAwareTopN

# Universe: liquid large-caps
UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "V"]

# Build pipeline
pipeline = Pipeline(YFinanceSource())
pipeline.register(MomentumEngine(lookback_days=126, skip_days=5), weight=0.30)
pipeline.register(QualityEngine(require_positive_earnings=False), weight=0.45)
pipeline.register(ValueEngine(use_forward_pe=True), weight=0.25)

# Score as of today
if __name__ == "__main__":
    scores = pipeline.run(UNIVERSE, as_of="2025-09-18")
    print("=== Composite Scores ===")
    print(scores.round(1).to_string())

    # Build target portfolio
    rule = TurnoverAwareTopN(n=5, turnover_penalty=0.3, max_weight=0.25)
    prices = pipeline.data_source.prices(UNIVERSE, start="2025-06-01", end="2025-09-18")
    weights = pipeline.build_portfolio(scores, prices)
    print("\n=== Target Weights ===")
    print(weights.round(3).to_string())
