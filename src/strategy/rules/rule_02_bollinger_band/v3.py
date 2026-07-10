from __future__ import annotations

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal


# --- Parameters for Bollinger Bands ---
BB_PERIOD = 20  # Number of candles for Bollinger Band calculation (e.g., 20 for 20-period SMA)
BB_STD_DEV = 2.0  # Standard deviation multiplier for Bollinger Bands (e.g., 2 standard deviations)

# --- Parameters for RSI ---
RSI_PERIOD = 14  # Number of candles for RSI calculation (e.g., 14 periods)
RSI_OVERSOLD_THRESHOLD = 30  # RSI level below which an asset is considered oversold
RSI_OVERBOUGHT_THRESHOLD = 70  # RSI level above which an asset is considered overbought

# Minimum candles required for both calculations:
# BB_PERIOD candles for BB's SMA and StdDev.
# RSI_PERIOD + 1 candles for RSI (to get RSI_PERIOD price changes).
MIN_CANDLES_REQUIRED = max(BB_PERIOD, RSI_PERIOD + 1)


def calculate_rsi(prices: np.ndarray, period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for a given series of prices.
    Assumes prices are ordered from oldest to newest.
    
    Args:
        prices (np.ndarray): A numpy array of closing prices.
        period (int): The number of periods to use for RSI calculation.

    Returns:
        float: The latest RSI value, or np.nan if not enough data.
    """
    if len(prices) < period + 1:
        return np.nan  # Not enough data for RSI calculation

    # Calculate price changes (differences)
    # diffs[i] = prices[i+1] - prices[i]
    diffs = np.diff(prices)

    # Separate gains and losses
    gains = np.maximum(0, diffs)
    losses = np.maximum(0, -diffs)

    # Calculate initial average gain and loss over the first 'period' differences
    # These correspond to prices[1]...prices[period+1] relative to prices[0]...prices[period]
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Handle cases where initial avg_loss is zero
    if avg_loss == 0:
        if avg_gain == 0:
            # No movement in the initial period, RSI is neutral
            rs = 1.0  # Leads to RSI = 50
        else:
            # Pure gains, no losses in the initial period, RSI is 100
            rs = np.inf
    else:
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Apply smoothing for subsequent periods if more data points are available
    # We only need the latest RSI value, so we iterate through remaining diffs
    for i in range(period, len(gains)):
        current_gain = gains[i]
        current_loss = losses[i]

        # RSI smoothing formula
        avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
        avg_loss = ((avg_loss * (period - 1)) + current_loss) / period

        if avg_loss == 0:
            if avg_gain == 0:
                rs = 1.0  # Leads to RSI = 50
            else:
                rs = np.inf  # Leads to RSI = 100
        else:
            rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
            
    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on Bollinger Bands with RSI confirmation.

    A Buy signal is generated when the price drops below the lower Bollinger Band
    and the RSI indicates an oversold condition (below RSI_OVERSOLD_THRESHOLD).

    A Sell signal is generated when the price rises above the upper Bollinger Band
    and the RSI indicates an overbought condition (above RSI_OVERBOUGHT_THRESHOLD).

    Args:
        data (MarketData): A dictionary of market data for various currency pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated buy or sell signals.
    """
    signals: list[BuySignal | SellSignal] = []
    
    # Define rule_id for generated signals based on the provided idea_id
    rule_id = "26936aab-3890-418d-be60-0e0f83b0b27d"

    for pair, pair_data in data.items():
        # Ensure we have enough hot data for current price and timestamp
        # And enough warm candles for BB and RSI calculations
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES_REQUIRED:
            continue

        # Get the latest 'hot' tick for the current price and timestamp
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        # --- Prepare data for Bollinger Bands and RSI ---
        # Take the most recent 'MIN_CANDLES_REQUIRED' closes from warm data.
        # 'warm' data is assumed to be ordered from oldest to newest.
        relevant_closes = np.array([c.close for c in pair_data.warm[-MIN_CANDLES_REQUIRED:]])

        # --- Calculate Bollinger Bands ---
        # Bollinger Bands are calculated on the last BB_PERIOD closes from `relevant_closes`.
        # Since `MIN_CANDLES_REQUIRED` is at least `BB_PERIOD`, this slice is valid.
        bb_period_closes = relevant_closes[-BB_PERIOD:]
        
        mean = np.mean(bb_period_closes)
        std = np.std(bb_period_closes)

        # If standard deviation is zero, the price hasn't moved, so no band (or flat band).
        # This indicates a non-tradable condition, so we skip.
        if std == 0:
            continue

        lower_band = mean - BB_STD_DEV * std
        upper_band = mean + BB_STD_DEV * std

        # --- Calculate RSI ---
        # RSI needs 'RSI_PERIOD + 1' closes for calculation.
        # Since `MIN_CANDLES_REQUIRED` is at least `RSI_PERIOD + 1`, this slice is valid.
        rsi_closes = relevant_closes[-(RSI_PERIOD + 1):]
        rsi_value = calculate_rsi(rsi_closes, RSI_PERIOD)

        # If RSI could not be calculated (e.g., due to extreme flat prices leading to division by zero
        # not handled by the specific logic in `calculate_rsi`), skip.
        if np.isnan(rsi_value):
            continue

        # --- Generate Signals with RSI Confirmation ---
        if current_price < lower_band and rsi_value < RSI_OVERSOLD_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id=rule_id))
        elif current_price > upper_band and rsi_value > RSI_OVERBOUGHT_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id=rule_id))

    return signals