from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle

# Parameters for Bollinger Bands (inherited from rule_02_bollinger_band_v4)
K = 2.5  # Standard deviation multiplier
WINDOW = 20  # Periods for SMA and SD calculation

def calculate_bollinger_bands(closes: list[float], k_multiplier: float) -> tuple[float, float, float]:
    """
    Calculates Simple Moving Average (SMA), Standard Deviation (SD),
    Lower Bollinger Band (LB), and Upper Bollinger Band (UB) for a given list of closing prices.
    Returns (SMA, LB, UB). If SD is zero, LB and UB are equal to SMA.
    """
    if len(closes) < 2:  # statistics.stdev requires at least 2 data points
        # Return default values indicating an invalid calculation or insufficient data
        return 0.0, 0.0, 0.0

    mean = statistics.mean(closes)
    std = statistics.stdev(closes)

    if std == 0:
        # If standard deviation is zero, bands are flat at the mean
        return mean, mean, mean

    lower_band = mean - k_multiplier * std
    upper_band = mean + k_multiplier * std
    return mean, lower_band, upper_band

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # We need WINDOW candles for the 'current' period's bands and close,
        # and WINDOW candles for the 'previous' period's bands and close.
        # This means we need at least WINDOW + 1 candles in total to have both
        # current_close and previous_close, and their respective WINDOW-length contexts.
        if len(pair_data.warm) < WINDOW + 1:
            continue

        all_closes = [c.close for c in pair_data.warm]

        # --- Current Period Calculations ---
        # The latest WINDOW candles for current bands calculation
        current_period_closes_for_bands = all_closes[-(WINDOW):]
        current_close = all_closes[-1]
        current_timestamp = pair_data.warm[-1].hour # Timestamp for the signal

        # Calculate bands for the current period
        # Note: The pseudocode suggests std_dev_multiplier = 2, but as this modifies rule_02_bollinger_band_v4,
        # which uses K=2.5, we maintain K=2.5 for consistency with the base rule being modified.
        current_sma, current_lb, current_ub = calculate_bollinger_bands(current_period_closes_for_bands, K)

        # --- Previous Period Calculations ---
        # The WINDOW candles ending one period ago for previous bands calculation
        previous_period_closes_for_bands = all_closes[-(WINDOW+1):-1]
        previous_close = all_closes[-2]

        # Calculate bands for the previous period
        previous_sma, previous_lb, previous_ub = calculate_bollinger_bands(previous_period_closes_for_bands, K)

        # --- Re-entry Reversal Logic (Modification of v4) ---
        # Buy Signal Logic: Price drops below lower band, then closes back inside
        # Condition 1: Previous close was below the previous lower band (breach)
        # Condition 2: Current close is above the previous lower band (re-entry)
        if previous_close < previous_lb and current_close > previous_lb:
            signals.append(BuySignal(pair=pair, timestamp=current_timestamp, price=current_close))
        # Sell Signal Logic: Price rises above upper band, then closes back inside
        # Condition 1: Previous close was above the previous upper band (breach)
        # Condition 2: Current close is below the previous upper band (re-entry)
        elif previous_close > previous_ub and current_close < previous_ub:
            signals.append(SellSignal(pair=pair, timestamp=current_timestamp, price=current_close))

    return signals