from __future__ import annotations
import numpy as np
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, OrderBook # OrderBook is implicitly used via Tick.order_book

# Parameters based on the rule idea and pseudocode:
# OBI_LOOKBACK_TICKS: Window (in number of ticks) for checking recent OBI consistency.
# A shorter window captures immediate trends.
OBI_LOOKBACK_TICKS = 10 

# SIGNAL_PERCENTAGE_THRESHOLD: Minimum percentage (0.0-1.0) of recent OBIs
# that must exceed the adaptive threshold to generate a signal.
SIGNAL_PERCENTAGE_THRESHOLD = 0.7 

# STD_DEV_LOOKBACK_TICKS: Window (in number of ticks) for calculating OBI standard deviation.
# A longer window provides a more stable measure of volatility.
# Assuming 1 tick/second, 60 ticks is 1 minute of data.
STD_DEV_LOOKBACK_TICKS = 60 

# THRESHOLD_MULTIPLIER: Multiplier for the OBI standard deviation to set the adaptive thresholds.
# A higher multiplier makes the rule less sensitive (requires larger deviations).
THRESHOLD_MULTIPLIER = 1.5 

# Minimum number of ticks required for the rule to have enough data for both lookback windows.
MIN_TICKS = max(OBI_LOOKBACK_TICKS, STD_DEV_LOOKBACK_TICKS)


def _imbalance(tick: Tick) -> float | None:
    """
    Calculates the Order Book Imbalance (OBI) ratio for a given tick.
    OBI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume).
    Returns None if total volume is zero to avoid division by zero.
    Prioritizes order book depth data if available, falls back to ticker bid/ask volumes.
    """
    if tick.order_book is not None:
        # Sum volumes from order book levels if available
        bid_vol = sum(lvl.volume for lvl in tick.order_book.bids)
        ask_vol = sum(lvl.volume for lvl in tick.order_book.asks)
    else:
        # Fallback to direct bid/ask volumes from the ticker if order book is not provided
        bid_vol = tick.bid_volume
        ask_vol = tick.ask_volume

    total_volume = bid_vol + ask_vol
    if total_volume == 0:
        return None  # Cannot calculate imbalance if there's no volume
    return (bid_vol - ask_vol) / total_volume


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on an adaptive threshold for Order Book Imbalance (OBI).

    The rule calculates OBI for recent ticks and compares them against thresholds
    that are dynamically set as a multiple of the historical OBI standard deviation.
    A signal is generated if a sufficient percentage of recent OBIs consistently
    exceeds these adaptive thresholds.

    Args:
        data: A MarketData object containing hot (recent ticks), warm (hourly candles),
              and cold (monthly data) for various currency pairs.

    Returns:
        A list of BuySignal or SellSignal objects indicating trading opportunities.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        
        # Ensure we have enough historical ticks for both lookback windows.
        if len(ticks) < MIN_TICKS:
            continue

        # Extract ticks for the standard deviation calculation window.
        # This window provides the historical context for OBI volatility.
        long_term_ticks = ticks[-STD_DEV_LOOKBACK_TICKS:]
        
        # Calculate OBIs for the standard deviation window.
        # Filter out any None values (where total volume was zero) as they cannot be used in std dev.
        all_long_term_obis = [_imbalance(t) for t in long_term_ticks]
        long_term_obis_float = [obi for obi in all_long_term_obis if obi is not None]

        # We need enough valid OBI observations to calculate a meaningful standard deviation.
        # If the number of valid OBIs is less than the required window size, we skip.
        # Also, at least two points are typically needed for a non-zero standard deviation calculation (ddof=1).
        if len(long_term_obis_float) < STD_DEV_LOOKBACK_TICKS or len(long_term_obis_float) < 2:
            continue
        
        # Calculate the standard deviation of OBI over the longer lookback window.
        # Using numpy for numerical stability and performance. ddof=1 for sample standard deviation.
        obi_std_dev = np.std(long_term_obis_float, ddof=1)

        # Set adaptive thresholds based on the standard deviation.
        # If obi_std_dev is zero (meaning OBI has not varied), thresholds are also zero.
        # In this scenario, any non-zero imbalance would be considered significant.
        buy_threshold = THRESHOLD_MULTIPLIER * obi_std_dev
        sell_threshold = -THRESHOLD_MULTIPLIER * obi_std_dev

        # Extract ticks for the recent consistency check window.
        # This window is used to check if current OBI behavior consistently exceeds thresholds.
        recent_ticks = ticks[-OBI_LOOKBACK_TICKS:]

        # Calculate OBIs for the recent consistency check window.
        # Filter out any None values.
        all_recent_obis = [_imbalance(t) for t in recent_ticks]
        recent_obis_float = [obi for obi in all_recent_obis if obi is not None]

        # Ensure we have enough valid OBI observations for the recent window.
        if len(recent_obis_float) < OBI_LOOKBACK_TICKS:
            continue
        
        # Handle the edge case where recent_obis_float might be empty after filtering,
        # although the `len` check above should largely prevent this if OBI_LOOKBACK_TICKS > 0.
        if not recent_obis_float:
            continue

        # Count how many recent OBIs exceed the adaptive thresholds.
        positive_exceed_count = sum(1 for obi_value in recent_obis_float if obi_value > buy_threshold)
        negative_exceed_count = sum(1 for obi_value in recent_obis_float if obi_value < sell_threshold)

        # Calculate the percentage of recent OBIs that exceeded the thresholds.
        positive_percentage = positive_exceed_count / len(recent_obis_float)
        negative_percentage = negative_exceed_count / len(recent_obis_float)

        # Get the timestamp and price of the most recent tick for the signal.
        ts = ticks[-1].polled_at
        price = ticks[-1].last_price

        # Check for signal conditions.
        # A Buy signal is generated if a sufficient percentage of recent OBIs
        # indicate buying pressure (positive imbalance above threshold).
        if positive_percentage >= SIGNAL_PERCENTAGE_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price))
        # A Sell signal is generated if a sufficient percentage of recent OBIs
        # indicate selling pressure (negative imbalance below threshold).
        elif negative_percentage >= SIGNAL_PERCENTAGE_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price))

    return signals