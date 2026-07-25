from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle # noqa: F401 (WarmCandle is used in type hints)

# --- Helper Functions ---

def _calculate_sma(closes: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average for the last 'period' closes."""
    if len(closes) < period:
        return None
    return statistics.mean(closes[-period:])

def _calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    """Calculates the Average True Range for the last 'period' True Ranges.
    Requires at least (period + 1) candles to calculate 'period' True Ranges
    (each TR needs current and previous close).
    """
    if len(highs) != len(lows) or len(highs) != len(closes) or len(highs) < period + 1:
        return None

    tr_values = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        close = closes[i]
        prev_close = closes[i-1]
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    
    # tr_values now has len(closes) - 1 elements.
    # We need at least 'period' of these to calculate the average.
    if len(tr_values) < period:
        return None
        
    return statistics.mean(tr_values[-period:])

def _calculate_rsi(closes: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index for the last close in the provided list.
    Requires at least (period + 1) closes to compute initial averages and the final RSI value.
    """
    if len(closes) < period + 1:
        return None

    gains = [0.0] * (len(closes) - 1)
    losses = [0.0] * (len(closes) - 1)

    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains[i-1] = change
        else:
            losses[i-1] = abs(change)

    # Initial average gain/loss over the first 'period' changes
    initial_avg_gain = sum(gains[:period]) / period
    initial_avg_loss = sum(losses[:period]) / period

    # Apply Wilder's smoothing for subsequent changes up to the last one
    # The last change is at index len(gains) - 1.
    # The initial averages cover changes up to index period - 1.
    # So we apply smoothing for changes from index 'period' to len(gains) - 1.
    current_avg_gain = initial_avg_gain
    current_avg_loss = initial_avg_loss

    for i in range(period, len(gains)):
        current_avg_gain = (current_avg_gain * (period - 1) + gains[i]) / period
        current_avg_loss = (current_avg_loss * (period - 1) + losses[i]) / period

    if current_avg_loss == 0:
        return 100.0 if current_avg_gain > 0 else 0.0 # Avoid division by zero, handle extreme cases
    
    rs = current_avg_gain / current_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Trading Rule ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    
    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure enough data for robust indicator calculation + current/previous candles
        # RSI(14) needs 14+1=15 closes for prev_candle. This means we need 15 candles in the slice.
        # Keltner (SMA 20, ATR 10) needs:
        #   SMA(20): 20 closes for prev_candle. This means we need 20 candles in the slice.
        #   ATR(10): 10+1=11 closes for prev_candle (to get 10 TRs, need 11 candles for 10 changes).
        #   Total for KC: max(20, 11) = 20 candles in the slice for prev_candle.
        # So, for indicators on prev_candle, we need max(15, 20) = 20 candles in the slice.
        # Plus 1 for current_candle. Total = 20 candles (for indicators on prev_candle) + 1 (prev_candle itself) + 1 (current_candle) = 22.
        # No, the slice is the history *before* current_candle.
        # To get indicators for prev_candle (warm_candles[-2]):
        #   RSI(14): needs 15 candles: warm_candles[-16] to warm_candles[-2].
        #   KC (SMA 20, ATR 10): needs 20 candles: warm_candles[-21] to warm_candles[-2].
        # So, the longest history needed for prev_candle indicators is 20 candles.
        # For the rule, we need `prev_candle` (`warm_candles[-2]`) and `current_candle` (`warm_candles[-1]`).
        # This means `len(warm_candles)` must be at least 2 (for current and prev candles)
        # PLUS the longest lookback for indicators, which is 20 for KC on `prev_candle`.
        # So `len(warm_candles)` must be at least `20 + 1 = 21` (20 historical + `prev_candle` itself).
        # The pseudocode specified `len(data.warm) < 24`, so we use this as the minimum.
        MIN_CANDLES_REQUIRED = 24 
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        prev_candle = warm_candles[-2]
        current_candle = warm_candles[-1]
        
        # --- Calculate RSI(14) for previous candle (data.warm[-2]) ---
        rsi_period = 14
        # Slice for RSI: (period + 1) candles ending at prev_candle.
        # E.g., for prev_candle (index -2), need 15 candles: warm_candles[-16] to warm_candles[-2].
        rsi_candles_slice = warm_candles[-(rsi_period + 2):-1] 
        # Check if slice itself has enough data, though main check should cover this.
        if len(rsi_candles_slice) < rsi_period + 1: 
             continue
        rsi_closes = [c.close for c in rsi_candles_slice]
        prev_rsi = _calculate_rsi(rsi_closes, rsi_period)
        if prev_rsi is None: # Should not happen if data length is sufficient
            continue

        # --- Calculate Keltner Channel (20-period SMA, 2.0 * 10-period ATR) for previous candle (data.warm[-2]) ---
        kc_period = 20
        atr_period = 10
        # Slice for KC: max(kc_period, atr_period + 1) candles ending at prev_candle.
        # E.g., for prev_candle (index -2), need 20 candles: warm_candles[-21] to warm_candles[-2].
        kc_candles_slice_len = max(kc_period, atr_period + 1)
        kc_candles_slice = warm_candles[-(kc_candles_slice_len + 1):-1]
        if len(kc_candles_slice) < kc_candles_slice_len: # Should not happen if data length is sufficient
            continue

        kc_closes = [c.close for c in kc_candles_slice]
        kc_highs = [c.high for c in kc_candles_slice]
        kc_lows = [c.low for c in kc_candles_slice]

        kc_middle_band = _calculate_sma(kc_closes, kc_period)
        atr_value = _calculate_atr(kc_highs, kc_lows, kc_closes, atr_period)
        
        if kc_middle_band is None or atr_value is None: # Should not happen if data length is sufficient
            continue

        kc_lower_band = kc_middle_band - (2.0 * atr_value)

        # Condition 1: Deeply oversold on previous candle
        oversold_condition = (prev_rsi < 30) and (prev_candle.close < kc_lower_band)

        # Condition 2: Bullish rebound on current candle (current close > 0.5% above previous close)
        rebound_condition = (current_candle.close > prev_candle.close * 1.005)

        if oversold_condition and rebound_condition:
            # Ensure there's at least one tick for entry_price
            if not hot_ticks:
                continue
            
            # Generate a long signal
            signals.append(BuySignal(
                pair=pair,
                timestamp=hot_ticks[-1].polled_at,
                price=hot_ticks[-1].last_price,
                rule_id="deep_oversold_reversal_bounce",
                stop_loss=current_candle.low,
                take_profit=hot_ticks[-1].last_price * 1.02 # Example: 2% target
            ))
            
    return signals