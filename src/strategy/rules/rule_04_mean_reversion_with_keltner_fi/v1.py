import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick


def calculate_sma(closes: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


def calculate_ema(closes: list[float], period: int) -> float | None:
    """Calculates the Exponential Moving Average."""
    if len(closes) < period:
        return None

    # Initial EMA is the SMA of the first 'period' closes
    # We need to compute EMA iteratively from the beginning of the relevant data window
    # to get the correct EMA for the latest candle.
    
    # Ensure there are enough candles to calculate the first SMA
    if len(closes) < period:
        return None

    # Calculate initial SMA for the first 'period' candles to start the EMA series
    current_ema = np.mean(closes[:period])
    
    multiplier = 2 / (period + 1)
    
    # Apply EMA formula for subsequent candles
    for i in range(period, len(closes)):
        current_ema = (closes[i] - current_ema) * multiplier + current_ema
    
    return float(current_ema)


def calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:  # Need at least period + 1 candles for the first change
        return None

    # Calculate price changes
    deltas = np.diff(closes)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Initial average gain/loss (SMA over the first 'period' changes)
    # The deltas array has len(closes) - 1 elements.
    # To get 'period' changes, we need len(closes) - 1 >= period, so len(closes) >= period + 1.
    initial_avg_gain = np.mean(gains[:period])
    initial_avg_loss = np.mean(losses[:period])

    avg_gain = initial_avg_gain
    avg_loss = initial_avg_loss

    # Calculate subsequent average gain/loss using smoothing (Wilder's smoothing method, similar to EMA)
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else (0.0 if avg_gain == 0 else None) # Handle division by zero
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    """Calculates the Average True Range (ATR)."""
    if len(highs) < period + 1:  # Need previous close for TR calculation
        return None

    true_ranges = []
    for i in range(1, len(closes)):  # Start from the second candle to have a previous close
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        true_ranges.append(tr)
    
    if len(true_ranges) < period: # Not enough TR values to calculate initial SMA
        return None

    # Initial ATR is SMA of the first 'period' true ranges
    current_atr = np.mean(true_ranges[:period])

    # Subsequent ATRs are smoothed using an EMA-like formula (Wilder's smoothing)
    for i in range(period, len(true_ranges)):
        current_atr = ((current_atr * (period - 1)) + true_ranges[i]) / period
    
    return float(current_atr)


def calculate_keltner_lower_band(highs: list[float], lows: list[float], closes: list[float], period: int, multiplier: float) -> float | None:
    """Calculates the Keltner Channel Lower Band."""
    # Keltner Channel Middle Band is typically an EMA
    middle_band = calculate_ema(closes, period)
    if middle_band is None:
        return None

    # Keltner Channel uses ATR for bandwidth
    atr = calculate_atr(highs, lows, closes, period)
    if atr is None:
        return None

    return middle_band - (multiplier * atr)


def calculate_bollinger_band_width(closes: list[float], period: int, stddev_multiplier: float) -> float | None:
    """Calculates the Bollinger Band Width."""
    if len(closes) < period:
        return None

    current_closes = np.array(closes[-period:])
    
    middle_band = np.mean(current_closes)
    std_dev = np.std(current_closes)

    # Calculate Upper and Lower Bands
    upper_band = middle_band + (stddev_multiplier * std_dev)
    lower_band = middle_band - (stddev_multiplier * std_dev)

    if middle_band == 0:  # Avoid division by zero
        return None
    
    return (upper_band - lower_band) / middle_band


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "mr_filtered_keltner_exit_v1"

    for pair, pair_data in data.items():
        # Check for sufficient warm data for indicators
        # Keltner ATR needs period+1 candles for TR values, and then 'period' TR values for its SMA.
        # So, total `period + 1` candles (e.g., 20+1 = 21) are needed for `highs`, `lows`, `closes`.
        # RSI(14) needs 14+1=15 candles. EMA/SMA(20) need 20 candles.
        # The highest requirement is 21 candles for `data.warm`.
        if len(pair_data.warm) < 21:
            continue

        # Check for sufficient hot data for current spread and price
        if not pair_data.hot:
            continue

        closes = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]
        
        last_close_price = closes[-1]
        current_tick = pair_data.hot[-1]
        current_timestamp = current_tick.polled_at
        current_price = current_tick.last_price  # Price for signal is last_price from hot data

        # Calculate current bid-ask spread percentage
        # Check for bid_price == 0 to avoid division by zero
        if current_tick.bid_price == 0:
            continue
        current_bid_ask_spread_pct = (current_tick.ask_price - current_tick.bid_price) / current_tick.bid_price

        # --- Calculate Indicators ---
        rsi_14 = calculate_rsi(closes, period=14)
        ema_20 = calculate_ema(closes, period=20)
        sma_20 = calculate_sma(closes, period=20)
        keltner_lower_20_2atr = calculate_keltner_lower_band(highs, lows, closes, period=20, multiplier=2.0)
        bollinger_band_width_20_2sd = calculate_bollinger_band_width(closes, period=20, stddev_multiplier=2.0)

        # If any essential indicator is None (due to insufficient data or calculation error), skip this pair
        if any(val is None for val in [rsi_14, ema_20, sma_20, keltner_lower_20_2atr, bollinger_band_width_20_2sd]):
            continue

        # --- Calculate Deviations ---
        # Check for zero denominators before calculating deviations to prevent division by zero
        if ema_20 == 0 or sma_20 == 0 or keltner_lower_20_2atr == 0:
            continue

        price_dev_ema = (last_close_price - ema_20) / ema_20
        price_dev_sma = (last_close_price - sma_20) / sma_20
        price_dev_keltner_lower = (last_close_price - keltner_lower_20_2atr) / keltner_lower_20_2atr

        # --- Buy Signal Conditions ---
        CONDITION_1_RSI_OVERSOLD = (35 <= rsi_14 <= 44)
        CONDITION_2_PRICE_DIP_EMA = (-0.01 <= price_dev_ema <= -0.001)
        CONDITION_3_PRICE_DIP_SMA = (-0.01 <= price_dev_sma <= -0.001)
        CONDITION_4_PRICE_DIP_KELTNER = (-0.015 <= price_dev_keltner_lower <= -0.001)
        # Pseudocode has `Bollinger_Band_Width_20_2SD < 3.5`. This is a very high upper bound, but implemented as specified.
        CONDITION_5_MODERATE_VOLATILITY = (0.005 < bollinger_band_width_20_2sd < 3.5)
        CONDITION_6_LOW_SPREAD = (current_bid_ask_spread_pct < 0.0005)

        if (CONDITION_1_RSI_OVERSOLD and
                CONDITION_2_PRICE_DIP_EMA and
                CONDITION_3_PRICE_DIP_SMA and
                CONDITION_4_PRICE_DIP_KELTNER and
                CONDITION_5_MODERATE_VOLATILITY and
                CONDITION_6_LOW_SPREAD):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=rule_id,
                confidence=1.0  # Default confidence
            ))

        # --- Sell Signal Conditions (for existing long positions) ---
        # Stop Loss: Price breaks significantly below Keltner Lower Band
        CONDITION_STOP_LOSS_KELTNER = (price_dev_keltner_lower < -0.02)
        # Profit Target: Price has mean-reverted towards EMA (positive or very slightly negative deviation)
        CONDITION_PROFIT_TAKE_EMA = (price_dev_ema > -0.0005)

        if CONDITION_STOP_LOSS_KELTNER or CONDITION_PROFIT_TAKE_EMA:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id=rule_id,
                confidence=1.0  # Default confidence
            ))

    return signals