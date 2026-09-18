"""EquityLens Framework — pluggable multi-factor equity research.

Core abstractions:
    DataSource     — anything that fetches raw market data
    Engine         — anything that generates scores from data
    PortfolioRule  — anything that turns scores into positions
    Pipeline       — orchestrates the above with PIT correctness
"""

from equitylens.plugin import DataSource, Engine, EngineOutput, PortfolioRule
from equitylens.pipeline import Pipeline
from equitylens.cache import Cache

__version__ = "0.1.0"
