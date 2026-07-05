from __future__ import annotations
import statistics
from src.agent.models import BuySignal, SellSignal, PairData

RULE_ID = "rule_01_spread_compression_v2"

# Parameters from pseudocode
WINDOW_SPREAD_AVG = 60 # e.g., 1-minute bars, 1 hour average
THRESHOLD_SPREAD_COMPRESSION = 0.15 # Percentage decrease from average
THRESHOLD_SPREAD_EXPANSION = 0.15 # Percentage increase from average
VOLUME_WINDOW = 5 # Number of recent bars to check volume
VOLUME_SURGE_MULTIPLIER = 1.5 # Volume must be X times its recent average

MarketData = dict[str, PairData]

def enhanced_spread_compression_with_volume_confirmation(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    # Minimum ticks required for calculations:
    # We need WINDOW_SPREAD_AVG ticks for historical spread average.
    # We need VOLUME_WINDOW ticks for historical volume average.
    # We need 1 tick for the current state.
    # We need an additional tick for price comparison over VOLUME_WINDOW (current vs. VOLUME_WINDOW ticks ago).
    # So, total = max(WINDOW_SPREAD_AVG, VOLUME_WINDOW) + 1 (for history) + 1 (for current) = max(WINDOW_SPREAD_AVG, VOLUME_WINDOW) + 2
    min_required_ticks = max(WINDOW_SPREAD_AVG, VOLUME_WINDOW) + 2

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < min_required_ticks:
            continue

        # Extract current tick data
        current_tick = ticks[-1]
        current_volume = current_tick.volume
        current_spread = current_tick.spread_rel # This is (ask-bid)/bid

        # Get recent history for spread calculation (excluding current tick)
        # Slices from `ticks[-WINDOW_SPREAD_AVG-1]` up to `ticks[-2]` (total WINDOW_SPREAD_AVG ticks)
        recent_spread_history = ticks[-(WINDOW_SPREAD_AVG + 1):-1]
        if not recent_spread_history:
            continue # Should not happen