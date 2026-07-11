from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, SellSignal, PairData, Tick, WarmCandle


# --- Configuration Constants ---
K_BASE = 2.0  # Original fixed K multiplier for Bollinger Bands
ALPHA = 0.5   # Sensitivity parameter for adapting K. Higher alpha means K changes more dynamically.
K_MIN_BOUND = 1.0 # Minimum allowed value for adaptive K
K_MAX_BOUND = 3.0 # Maximum allowed value for adaptive K

MIN_TICKS_BB = 10 # Minimum number of hot ticks required for Bollinger Band calculation
MIN_CANDLES_VOL = 6 # Minimum number of warm hourly candles required for volatility assessment


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Adaptive Bollinger Band Multiplier trading rule.

    This rule enhances the standard Bollinger Band logic by dynamically adjusting
    the standard deviation multiplier (K) based on recent market volatility.
    In periods of higher hourly volatility relative to short-term tick volatility,
    K increases to widen the bands, reducing false signals. Conversely, in periods
    of lower volatility, K decreases to tighten the bands, capturing more subtle
    mean-reversion opportunities.

    Args:
        data: A MarketData dictionary containing current and historical price data
              for various currency pairs.

    Returns:
        A list of BuySignal or SellSignal objects based on the adaptive Bollinger Band logic.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient hot (tick) data for Bollinger Band calculation
        if not pair_data.hot or len(pair_data.hot) < MIN_TICKS_BB:
            continue

        # Extract closing prices from hot ticks for Bollinger Band calculation
        # The 'hot' data represents the most recent, high-frequency price movements.
        closes_hot = [t.last_price for t in pair_data.hot]
        mean_hot = statistics.mean(closes_hot)
        
        # Calculate standard deviation for the Bollinger Bands based on hot data
        # This represents the short-term volatility for the bands themselves.
        try:
            std_hot = statistics.stdev(closes_hot)
        except statistics.StatisticsError: # Handles cases where std cannot be calculated (e.g., all prices are identical)
            std_hot = 0.0

        if std_hot == 0:
            # If there's no price movement in the hot data, bands would be infinitely tight.
            # Skip to avoid division by zero or nonsensical signals.
            continue

        # --- Adaptive K Calculation ---
        k_adaptive = K_BASE # Default to base K

        # Assess recent market volatility using warm (hourly candle) data
        # The 'warm' data provides a longer-term perspective on recent volatility.
        if len(pair_data.warm) >= MIN_CANDLES_VOL:
            closes_warm = [c.close for c in pair_data.warm]
            
            # Calculate standard deviation from warm data as the "current volatility" metric
            try:
                std_warm = statistics.stdev(closes_warm)
            except statistics.StatisticsError:
                std_warm = 0.0

            if std_warm > 0:
                # Calculate the volatility ratio: hourly volatility relative to short-term tick volatility
                # This ratio determines how much K should adapt.
                volatility_ratio = std_warm / std_hot

                # Apply the adaptive formula: K_adaptive = K_base * (1 + alpha * (current_volatility_ratio - 1))
                # If volatility_ratio > 1, K increases. If < 1, K decreases.
                k_adaptive = K_BASE * (1 + ALPHA * (volatility_ratio - 1))

                # Clamp K_adaptive within defined bounds to prevent extreme values
                k_adaptive = max(K_MIN_BOUND, min(K_MAX_BOUND, k_adaptive))
        
        # --- Signal Generation with Adaptive K ---
        current_price = pair_data.hot[-1].last_price
        timestamp = pair_data.hot[-1].polled_at

        # Calculate adaptive Bollinger Bands
        upper_band_adaptive = mean_hot + k_adaptive * std_hot
        lower_band_adaptive = mean_hot - k_adaptive * std_hot

        # Generate signals based on price crossing the adaptive bands
        if current_price < lower_band_adaptive:
            # Price below lower band suggests downward overextension, expect reversion up
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        elif current_price > upper_band_adaptive:
            # Price above upper band suggests upward overextension, expect reversion down
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals