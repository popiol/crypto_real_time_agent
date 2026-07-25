from __future__ import annotations
import statistics
import math
from datetime import datetime
from typing import List, Union

from pydantic import BaseModel, Field

# Import models from the expected path within the project structure.
# This assumes `src.agent.models` is available in the environment.
# If running standalone, these models would need to be defined directly here.
try:
    from src.agent.models import BuySignal, SellSignal, MarketData, WarmCandle
except ImportError:
    # Fallback for local testing or environments where src.agent.models is not accessible.
    # These are minimal definitions needed for the rule to function.
    class Tick(BaseModel):
        pair: str
        polled_at: datetime
        last_price: float
        bid_price: float
        bid_volume: float
        ask_price: float
        ask_volume: float
        volume_24h: float = 0.0
        mid_price: float
        spread_abs: float
        spread_rel: float
        order_book: dict | None = None

    class WarmCandle(BaseModel):
        hour: datetime
        open_price: float
        high: float
        low: float
        close: float
        avg_spread_rel: float = 0.0
        volume: float = Field(default=0.0)

    class ColdMonth(BaseModel):
        month: str
        min_price: float
        max_price: float
        avg_price: float
        avg_daily_spread: float
        candle_count: int
        last_candle_hour: datetime

    class PairData(BaseModel):
        hot: list[Tick] = Field(default=[])
        warm: list[WarmCandle] = Field(default=[])
        cold: list[ColdMonth] = Field(default=[])

    class BuySignal(BaseModel):
        pair: str
        timestamp: datetime
        price: float
        rule_id: str = ""
        confidence: float | None = None

    class SellSignal(BaseModel):
        pair: str
        timestamp: datetime
        price: float
        rule_id: str = ""
        confidence: float | None = None

    MarketData = dict[str, PairData]


# --- Helper functions for indicator calculations ---

def _calculate_ema(data: list[float], period: int) -> list[float]:
    """
    Calculates Exponential Moving Average (EMA) for a list of float values.
    Returns a list of EMA values, starting from the point where enough data is available.
    """
    if len(data) < period:
        return []

    ema_values = [0.0] * len(data)
    alpha = 2 / (period + 1)

    # Initialize the first EMA value with a Simple Moving Average (SMA)
    initial_sma = sum(data[:period]) / period
    ema_values[period - 1] = initial_sma

    # Calculate subsequent EMA values
    for i in range(period, len(data)):
        ema_values[i] = (data[i] - ema_values[i-1]) * alpha + ema_values[i-1]

    # Return only the valid EMA values, which start from index `period - 1`
    return ema_values[period - 1:]


def calculate_rsi(candles: list[WarmCandle], period: int = 14) -> list[float]:
    """
    Calculates the Relative Strength Index (RSI) for a list of WarmCandle objects.
    Returns a list of RSI values, starting from the point where enough data is available.
    """
    if len(candles) < period + 1:
        return []

    gains = [0.0] * len(candles)
    losses = [0.0] * len(candles)

    # Calculate price changes, positive changes (gains), and absolute negative changes (losses)
    for i in range(1, len(candles)):
        change = candles[i].close - candles[i-1].close
        if change > 0:
            gains[i] = change
        else:
            losses[i] = abs(change)

    avg_gains = [0.0] * len(candles)
    avg_losses = [0.0] * len(candles)
    rsi_values = [0.0] * len(candles)

    # Calculate the first average gain/loss using a Simple Moving Average over the period
    # We sum from index 1 as `gains[0]` and `losses[0]` are always 0.
    avg_gains[period] = sum(gains[1 : period + 1]) / period
    avg_losses[period] = sum(losses[1 : period + 1]) / period

    # Calculate subsequent average gain/loss using an EMA-like smoothing (Wilder's method)
    for i in range(period + 1, len(candles)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RSI values
    for i in range(period, len(candles)): # RSI values are valid from `period` index onwards
        if avg_losses[i] == 0:
            rsi_values[i] = 100.0 # If no losses, RS is infinite, RSI is 100
        else:
            rs = avg_gains[i] / avg_losses[i]
            rsi_values[i] = 100 - (100 / (1 + rs))

    # Return only the valid RSI values, which start from index `period`
    return rsi_values[period:]


def calculate_keltner_channel_position(candles: list[WarmCandle], ema_period: int = 20, atr_period: int = 10) -> list[float]:
    """
    Calculates the position of the current close price relative to the Keltner Channel center line.
    The rule's condition `Keltner Channel Position < 0` implies `close < center_line`.
    Thus, this function returns `(close - center_line)`. The `atr_period` is not used in this specific calculation
    as only the center line's relation to price matters for the rule.
    """
    if len(candles) < ema_period:
        return []

    close_prices = [c.close for c in candles]
    ema_closes = _calculate_ema(close_prices, ema_period)

    if not ema_closes:
        return []

    positions = []
    # The `ema_closes` list's first element corresponds to `candles[ema_period - 1]`.
    # We align the close price with its corresponding EMA value.
    for i in range(len(ema_closes)):
        current_candle_close = candles[ema_period - 1 + i].close
        position = current_candle_close - ema_closes[i]
        positions.append(position)

    return positions


def calculate_volume_anomaly(candles: list[WarmCandle], period: int = 20) -> list[float]:
    """
    Calculates the volume anomaly as `current_volume / SMA(previous_period_volumes)`.
    Requires `period` previous candles to calculate the SMA, plus the current candle,
    thus `period + 1` total candles are needed for the first anomaly value.
    """
    if len(candles) < period + 1: # Need 'period' candles for SMA + current candle
        return []

    volumes = [c.volume for c in candles]
    
    anomalies = []
    # Loop starts from `period` index to ensure there are `period` previous candles available for SMA
    for i in range(period, len(volumes)):
        # Calculate SMA of the previous `period` volumes (candles from `i-period` to `i-1`)
        previous_volumes = volumes[i - period : i]
        
        # This check is a safeguard; it should not be triggered if `i >= period` and `len(volumes)` is sufficient.
        if not previous_volumes: 
            anomalies.append(0.0) # Append a default value if data is unexpectedly missing
            continue
        
        sma_prev_volumes = sum(previous_volumes) / period
        
        # Calculate anomaly: current volume divided by the average of previous volumes
        if sma_prev_volumes > 0:
            anomalies.append(volumes[i] / sma_prev_volumes)
        else:
            anomalies.append(0.0) # Avoid division by zero if average previous volume is zero

    return anomalies


# --- Rule Implementation ---

# Minimum number of warm candles required to calculate all indicators.
# RSI(14) needs 14 + 1 = 15 candles.
# Keltner Center (EMA 20) needs 20 candles.
# Volume Anomaly (period 20, comparing current to previous 20) needs 20 + 1 = 21 candles.
# The highest requirement dictates the minimum.
MIN_CANDLES_REQUIRED = 21

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Mean Reversion Buy: Oversold Below Keltner Center with Strong Volume' rule.
    Initiates a buy trade when the asset's hourly closing price is below the Keltner Channel
    center line (Keltner Channel Position < 0), combined with an oversold Relative Strength Index (RSI < 35),
    and strong hourly trading volume indicated by a Volume Anomaly greater than 1.0.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure sufficient data for all indicator calculations
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        current_close = warm_candles[-1].close
        current_timestamp = warm_candles[-1].hour

        # Calculate indicators
        hourly_rsi_series = calculate_rsi(warm_candles, period=14)
        if not hourly_rsi_series:
            continue
        hourly_rsi = hourly_rsi_series[-1]

        hourly_keltner_position_series = calculate_keltner_channel_position(warm_candles, ema_period=20, atr_period=10)
        if not hourly_keltner_position_series:
            continue
        hourly_keltner_position = hourly_keltner_position_series[-1]

        hourly_volume_anomaly_series = calculate_volume_anomaly(warm_candles, period=20)
        if not hourly_volume_anomaly_series:
            continue
        hourly_volume_anomaly = hourly_volume_anomaly_series[-1]

        # Apply the rule's conditions for a BUY signal
        # 1. Price is below Keltner Channel center line (Keltner Position < 0)
        # 2. RSI indicates oversold conditions (RSI < 35)
        # 3. Volume shows strong activity (Volume Anomaly > 1.0)
        if (hourly_keltner_position < 0 and
            hourly_rsi < 35 and
            hourly_volume_anomaly > 1.0):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_close,
                rule_id="mean_reversion_buy_oversold_kc",
                reason="Oversold below Keltner Channel center with strong volume, anticipating mean reversion."
            ))

    return signals