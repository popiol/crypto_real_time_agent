from __future__ import annotations
import statistics
import numpy as np
from src.agent.models import BuySignal, SellSignal, PairData

RULE_ID = "rule_08_roc_momentum_v2"

# Configuration parameters
N_