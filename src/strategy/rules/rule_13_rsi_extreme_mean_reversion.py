from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, SellSignal, PairData, TickData

MarketData = dict[str, PairData]

RULE_ID = "rule_13_rsi_extreme_mean_reversion