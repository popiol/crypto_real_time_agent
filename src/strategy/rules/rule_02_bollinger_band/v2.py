from __future__ import annotations
import numpy as np
# statistics module is available but numpy is preferred for array operations
# import statistics 
from src.agent.models import BuySignal, SellSignal, PairData

RULE_ID = "rule_02_bollinger_band_v2"

# Rule Parameters
BB_WINDOW_LENGTH = 20
BB_STD_DEV_MULTIPLIER = 2.0
VOL_WINDOW_LENGTH = 5
VOL_SURGE_MULTIPLIER = 1.5

# Minimum ticks required for all calculations (Bollinger Bands and Average Volume)
MIN_TICKS = max(BB_WINDOW_LENGTH, VOL_WINDOW_LENGTH)

MarketData = dict[str, PairData]

def _calculate_sma(data_array: np.ndarray, window: int) -> float:
    """Calculates the Simple Moving Average of the last 'window' elements."""
    # The check for len(data_array) < window is handled by MIN_TICKS in the main function,
    # ensuring data_array[-window:] is always valid.
    return float(np.mean(data_array[-window:]))

def _calculate_std_dev(data_array: np.ndarray, window: int) -> float:
    """Calculates the Standard Deviation of the last 'window' elements."""
    # The check for len(data_array) < window is handled by MIN_TICKS in the main function,
    # ensuring data_array[-window:] is always valid.
    return float(np.std(data_array[-window:]))

def bollinger_band_v2(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band with Volume Confirmation trading rule.

    This rule emits a Buy signal when the price falls below the lower Bollinger Band
    AND there is a significant surge in trading volume.
    Conversely, it emits a Sell signal when the price rises above the upper Bollinger Band
    AND there is a significant surge in trading volume.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot

        # Ensure we have enough data points for all calculations
        if len(ticks) < MIN_TICKS:
            continue

        # Extract prices and volumes into numpy arrays for efficient calculation
        prices = np.array([t.last_price for t in ticks])
        volumes = np.array([t.volume for t in ticks])

        # Calculate Bollinger Bands components
        sma_prices = _calculate_sma(prices, BB_WINDOW_LENGTH)
        std_dev_prices = _calculate_std_dev(prices, BB_WINDOW_LENGTH)

        upper_band = sma_prices + (std_dev_prices * BB_STD_DEV_MULTIPLIER)
        lower_band = sma_prices - (std_dev_prices * BB_STD_DEV_MULTIPLIER)

        # Calculate Average Volume
        avg_volume = _calculate_sma(volumes, VOL_WINDOW_LENGTH)

        # Get current price and volume (most recent tick)
        current_price = prices[-1]
        current_volume = volumes[-1]

        # Generate Signals
        # Buy signal condition: Price below lower band AND current volume is a surge
        if current_price < lower_band and current_volume > (avg_volume * VOL_SURGE_MULTIPLIER):
            signals.append(BuySignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=ticks[-1].polled_at,
                price=current_price,
            ))
        # Sell signal condition: Price above upper band AND current volume is a surge
        elif current_price > upper_band and current_volume > (avg_volume * VOL_SURGE_MULTIPLIER):
            signals.append(SellSignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=ticks[-1].polled_at,
                price=current_price,
            ))

    return signals