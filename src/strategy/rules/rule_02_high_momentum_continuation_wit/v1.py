from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime

# Assuming these models are available from src.agent.models
# The actual system will provide these.
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# --- Constants ---
RSI_PERIOD = 24  # Lookback period for RSI
KCH_PERIOD = 20  # Lookback period for Keltner Channels (SMA and ATR)
KCH_MULTIPLIER = 2.0  # Multiplier for ATR in Keltner Channels

# Minimum data requirements for indicator calculations
# RSI(N) needs N+1 closes. ATR(N) needs N+1 candles.
MIN_CANDLES_FOR_INDICATORS = max(RSI_PERIOD, KCH_PERIOD) + 1
MIN_TICKS = 1  # Need at least one tick for current price, bid, ask

# --- Helper Functions for Indicators ---

def calculate_sma(data: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average for the last 'period' values."""
    if len(data) < period:
        return None
    return float(np.mean(data[-period:]))

def calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI) for the last candle."""
    if len(closes) < period + 1:
        return None

    # Calculate price differences
    diffs = np.diff(closes)
    
    # Separate gains and losses
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, np.abs(diffs), 0)

    # Calculate initial average gain and loss (simple moving average for the first 'period' values)
    # These are for the first `period` entries in `gains`/`losses` array (which corresponds to `closes[1]` to `closes[period]`)
    avg_gain_initial = np.mean(gains[:period])
    avg_loss_initial = np.mean(losses[:period])

    # Initialize lists to hold smoothed averages
    avg_gains_smoothed = [0.0] * len(gains)
    avg_losses_smoothed = [0.0] * len(losses)

    # Set the first smoothed averages
    avg_gains_smoothed[period - 1] = avg_gain_initial
    avg_losses_smoothed[period - 1] = avg_loss_initial

    # Apply Wilder's smoothing for subsequent averages
    for i in range(period, len(gains)):
        avg_gains_smoothed[i] = (avg_gains_smoothed[i - 1] * (period - 1) + gains[i]) / period
        avg_losses_smoothed[i] = (avg_losses_smoothed[i - 1] * (period - 1) + losses[i]) / period

    # Use the last calculated smoothed averages for the final RSI
    final_avg_gain = avg_gains_smoothed[-1]
    final_avg_loss = avg_losses_smoothed[-1]

    if final_avg_loss == 0:
        return 100.0 if final_avg_gain > 0 else 50.0  # Handle division by zero
    
    rs = final_avg_gain / final_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    """Calculates the Average True Range (ATR) for the last candle."""
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < period: # We need `period` TRs to calculate the SMA
        return None
        
    return float(np.mean(true_ranges[-period:])) # Simple Moving Average of True Ranges for ATR

def calculate_keltner_channels(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    period: int,
    atr_multiplier: float
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Calculates Keltner Channels bands (Middle, Upper, Lower) and the ATR value.
    Returns (middle_band, upper_band, lower_band, atr_value).
    """
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return None, None, None, None
    
    middle_band = calculate_sma(closes, period)
    if middle_band is None:
        return None, None, None, None

    atr_value = calculate_atr(highs, lows, closes, period)
    if atr_value is None:
        return None, None, None, None

    upper_band = middle_band + (atr_multiplier * atr_value)
    lower_band = middle_band - (atr_multiplier * atr_value)
    
    return middle_band, upper_band, lower_band, atr_value

# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'High Momentum Continuation with Volume and Liquidity Filter' rule.
    Initiates a long position when an asset exhibits extreme overbought conditions (RSI > 80)
    and strong price deviation above Keltner Channels, confirmed by significant volume
    and low bid-ask spread. Exits when momentum wanes or price retreats below Keltner upper band.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # --- Data Validation ---
        if len(pair_data.hot) < MIN_TICKS:
            continue
        if len(pair_data.warm) < MIN_CANDLES_FOR_INDICATORS:
            continue

        latest_tick = pair_data.hot[-1]
        latest_candle = pair_data.warm[-1]
        
        current_price = latest_tick.last_price
        current_bid = latest_tick.bid_price
        current_ask = latest_tick.ask_price
        
        # Extract candle data for indicator calculations
        closes = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]
        volumes = [c.volume for c in pair_data.warm]

        # --- Indicator Calculations ---

        # 1. Spread Percentage
        spread_pct = ((current_ask - current_bid) / current_price) * 100 if current_price else 0.0

        # 2. Hourly Volume Average Deviation
        # Average of all *past* volumes, excluding the latest candle
        past_volumes = volumes[:-1]
        avg_volume_past = sum(past_volumes) / len(past_volumes) if len(past_volumes) > 0 else 0.0
        current_volume = latest_candle.volume
        
        volume_deviation = 0.0
        if avg_volume_past > 0:
            volume_deviation = (current_volume - avg_volume_past) / avg_volume_past
        elif current_volume > 0: # If past volume is zero, and current is positive, treat as high deviation
            volume_deviation = 1.0
        # If both are zero, volume_deviation remains 0.0

        # 3. RSI (24h)
        rsi_24 = calculate_rsi(closes, RSI_PERIOD)
        if rsi_24 is None:
            continue

        # 4. Keltner Channels & Keltner Channels Percentage
        # Keltner Channels Percentage is defined as (latest_candle_close - upper_band) / atr_value
        # This measures how many ATRs the close is above the upper band.
        _, upper_band, _, atr_value = calculate_keltner_channels(
            closes, highs, lows, KCH_PERIOD, KCH_MULTIPLIER
        )
        if upper_band is None or atr_value is None or atr_value == 0:
            continue
        
        keltner_percentage = (latest_candle.close - upper_band) / atr_value

        # --- Signal Conditions ---

        # BUY Signal: High momentum continuation
        if (rsi_24 > 80 and
            keltner_percentage > 4 and
            spread_pct < 0.2 and
            volume_deviation > 0.5):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_tick.polled_at,
                price=current_price,
                rule_id="momentum_surge_breakout",
                confidence=1.0, # Deterministic signal
            ))

        # SELL Signal: Momentum wanes or price retreats (to close existing long position)
        # This condition is independent of the buy signal for the current tick,
        # as it serves as an exit for an already open position.
        if (rsi_24 < 70 and
            keltner_percentage < 2):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=latest_tick.polled_at,
                price=current_price,
                rule_id="momentum_surge_breakout",
                confidence=1.0, # Deterministic signal
            ))

    return signals