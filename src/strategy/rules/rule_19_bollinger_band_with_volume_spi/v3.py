from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters
PERIOD_BB = 20
STD_DEV_BB = 2.0
PERIOD_RSI = 14
OVERSOLD_RSI = 30
OVERBOUGHT_RSI = 70
PERIOD_VOLUME_MA = 50
VOLUME_MULTIPLIER = 1.5

# Unique identifier for this rule
RULE_ID = "247ea6bd-d3b2-4dbf-9cd5-6fbf047983a1"

def calculate_rsi(close_prices: np.ndarray, period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for the last data point
    in the provided `close_prices` array.

    Args:
        close_prices (np.ndarray): An array of historical close prices.
        period (int): The period for RSI calculation (e.g., 14).

    Returns:
        float: The RSI value for the most recent close price, or np.nan if
               there is insufficient data.
    """
    # RSI requires at least `period + 1` prices to calculate `period` changes.
    if len(close_prices) < period + 1:
        return np.nan

    # Use the relevant subset of prices for RSI calculation: the last `period + 1` prices.
    relevant_prices = close_prices[-(period + 1):]

    # Calculate price changes (deltas)
    deltas = np.diff(relevant_prices)

    # Separate gains and losses
    # Gains are positive deltas, losses are absolute values of negative deltas
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate the average gain and average loss over the specified period.
    # For the last RSI value, we average the gains/losses over the last `period` changes.
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    # Calculate Relative Strength (RS)
    if avg_loss == 0:
        # Handle cases with no losses: RSI is 100 if there were gains, else 50 (flat/no movement)
        rs = np.inf if avg_gain > 0 else 1.0 # If avg_gain is also 0, rs=1, rsi=50
    else:
        rs = avg_gain / avg_loss

    # Calculate RSI
    if rs == np.inf:
        rsi = 100.0
    else:
        rsi = 100 - (100 / (1 + rs))

    return rsi

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on a Bollinger Band breach, confirmed by
    an adaptive volume threshold and an RSI overbought/oversold condition.

    A Buy signal is emitted when:
    1. The price closes below the lower Bollinger Band.
    2. The current volume exceeds an adaptive moving average of volume by a multiplier.
    3. The Relative Strength Index (RSI) is oversold (< OVERSOLD_RSI).

    A Sell signal is emitted when:
    1. The price closes above the upper Bollinger Band.
    2. The current volume exceeds an adaptive moving average of volume by a multiplier.
    3. The Relative Strength Index (RSI) is overbought (> OVERBOUGHT_RSI).
    """
    signals: list[BuySignal | SellSignal] = []

    # Determine the minimum number of candles required for all indicator calculations
    # Bollinger Bands (BB) need PERIOD_BB candles.
    # RSI needs PERIOD_RSI + 1 candles (to calculate PERIOD_RSI price changes).
    # Volume Moving Average (MA) needs PERIOD_VOLUME_MA candles.
    min_required_candles = max(PERIOD_BB, PERIOD_RSI + 1, PERIOD_VOLUME_MA)

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # Ensure enough historical candles are available for all calculations
        if len(candles) < min_required_candles:
            continue

        # Extract close prices and volumes from the candles
        close_prices = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Get the data for the most recent candle for signal generation
        latest_candle: WarmCandle = candles[-1]
        current_close = latest_candle.close
        current_volume = latest_candle.volume
        current_timestamp = latest_candle.hour

        # 1. Calculate Bollinger Bands (BB)
        # Use the last PERIOD_BB close prices for SMA and Standard Deviation
        bb_period_closes = close_prices[-PERIOD_BB:]
        sma_bb = np.mean(bb_period_closes)
        std_dev_bb = np.std(bb_period_closes)
        upper_band = sma_bb + (STD_DEV_BB * std_dev_bb)
        lower_band = sma_bb - (STD_DEV_BB * std_dev_bb)

        # 2. Calculate Relative Strength Index (RSI)
        rsi_value = calculate_rsi(close_prices, PERIOD_RSI)
        
        # If RSI calculation resulted in NaN (e.g., insufficient data within calculate_rsi),
        # skip signal generation for this pair.
        if np.isnan(rsi_value):
            continue

        # 3. Calculate Adaptive Volume Threshold
        # Use the last PERIOD_VOLUME_MA volumes for the moving average
        volume_ma_data = volumes[-PERIOD_VOLUME_MA:]
        volume_ma = np.mean(volume_ma_data)
        volume_threshold = volume_ma * VOLUME_MULTIPLIER

        # Generate Signals based on the rule's conditions
        # Buy signal condition: Price below lower BB, high volume, and RSI oversold
        if (current_close < lower_band and
                current_volume > volume_threshold and
                rsi_value < OVERSOLD_RSI):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))
        # Sell signal condition: Price above upper BB, high volume, and RSI overbought
        elif (current_close > upper_band and
                current_volume > volume_threshold and
                rsi_value > OVERBOUGHT_RSI):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id=RULE_ID
            ))

    return signals