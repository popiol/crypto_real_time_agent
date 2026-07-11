from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, PairData, SellSignal

K = 2.5  # Standard deviation multiplier, inherited from rule_02_bollinger_band_v2
WINDOW = 20  # Periods for SMA and SD calculation, as per pseudocode

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
        current_sma, current_lb, current_ub = calculate_bollinger_bands(current_period_closes_for_bands, K)

        # Skip if bands could not be calculated meaningfully (e.g., insufficient data or zero std dev)
        # Note: calculate_bollinger_bands handles std=0 by returning flat bands (LB=UB=SMA),
        # which means no signal will be generated if current_close is within (or equal to) them.
        # We still need to ensure the calculation was valid (e.g., not all zeros from insufficient data)
        if current_sma == 0.0 and current_lb == 0.0 and current_ub == 0.0 and len(current_period_closes_for_bands) < 2:
            continue

        # --- Previous Period Calculations ---
        # The WINDOW candles ending one period ago for previous bands calculation
        previous_period_closes_for_bands = all_closes[-(WINDOW+1):-1]
        previous_close = all_closes[-2]

        # Calculate bands for the previous period
        previous_sma, previous_lb, previous_ub = calculate_bollinger_bands(previous_period_closes_for_bands, K)

        # Skip if bands could not be calculated meaningfully
        if previous_sma == 0.0 and previous_lb == 0.0 and previous_ub == 0.0 and len(previous_period_closes_for_bands) < 2:
            continue

        # --- Two-Period Confirmation Logic ---
        # Buy signal: Price closed below lower band for two consecutive periods
        if previous_close < previous_lb and current_close < current_lb:
            signals.append(BuySignal(pair=pair, timestamp=current_timestamp, price=current_close))
        # Sell signal: Price closed above upper band for two consecutive periods
        elif previous_close > previous_ub and current_close > current_ub:
            signals.append(SellSignal(pair=pair, timestamp=current_timestamp, price=current_close))

    return signals