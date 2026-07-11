from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2
MFI_PERIOD = 14
VOLUME_SMA_PERIOD = 20
BBW_SMA_PERIOD = 20
MFI_OVERSOLD_THRESHOLD = 20
MFI_OVERBOUGHT_THRESHOLD = 80
PIN_BAR_WICK_BODY_RATIO = 2.0
PIN_BAR_OPPOSITE_WICK_BODY_RATIO_MAX = 0.5

# Rule ID for signals
RULE_ID = "104ffeaf-b78e-4821-ba05-a02088e670a4"

def _calculate_bollinger_bands(closes: np.ndarray, period: int, std_dev: float):
    """Calculates Bollinger Bands (Upper, Middle, Lower) for the last 'period' closes."""
    if len(closes) < period:
        return None, None, None
    
    middle_band = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, middle_band, lower_band

def _calculate_mfi(candles_slice: list[WarmCandle], period: int):
    """
    Calculates the Money Flow Index (MFI) for the last candle in the provided slice.
    The slice must contain at least `period` candles.
    """
    if len(candles_slice) < period:
        return None

    # We need `period` candles to calculate MFI for the last candle in the slice.
    # The MFI calculation compares current TP with previous TP, so we need `period`
    # (current candle) + `period` (previous candle) points for summing positive/negative money flow.
    # The `candles_slice` passed here should be exactly `period` long to calculate MFI for its last element.
    
    typical_prices = np.array([(c.high + c.low + c.close) / 3 for c in candles_slice])
    money_flows = np.array([tp * c.volume for tp, c in zip(typical_prices, candles_slice)])

    positive_mf_sum = 0.0
    negative_mf_sum = 0.0

    # Iterate over the `period` candles provided to sum positive/negative money flow.
    # The loop considers the current candle `i` and the previous candle `i-1`.
    # It starts from the second candle in the slice (index 1) up to the last candle.
    for i in range(1, len(candles_slice)):
        if typical_prices[i] > typical_prices[i-1]:
            positive_mf_sum += money_flows[i]
        elif typical_prices[i] < typical_prices[i-1]:
            negative_mf_sum += money_flows[i]
    
    if negative_mf_sum == 0:
        return 100.0 # MFI is 100 if no negative money flow
    
    money_ratio = positive_mf_sum / negative_mf_sum
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

def _is_bullish_rejection_pin_bar(candle: WarmCandle, bb_lower: float) -> bool:
    """Detects a bullish pin bar (hammer-like) rejecting the lower Bollinger Band."""
    # Condition: Low touches/breaches lower band, but closes above it (rejection)
    if not (candle.low <= bb_lower and candle.close > bb_lower):
        return False

    body_size = abs(candle.close - candle.open_price)
    lower_wick = min(candle.open_price, candle.close) - candle.low
    upper_wick = candle.high - max(candle.open_price, candle.close)

    if body_size > 0:
        if lower_wick >= body_size * PIN_BAR_WICK_BODY_RATIO and upper_wick <= body_size * PIN_BAR_OPPOSITE_WICK_BODY_RATIO_MAX:
            return True
    elif body_size == 0: # Doji-like pin bar
        if lower_wick > 0 and upper_wick <= lower_wick * PIN_BAR_OPPOSITE_WICK_BODY_RATIO_MAX:
            return True
            
    return False

def _is_bearish_rejection_pin_bar(candle: WarmCandle, bb_upper: float) -> bool:
    """Detects a bearish pin bar (shooting star-like) rejecting the upper Bollinger Band."""
    # Condition: High touches/breaches upper band, but closes below it (rejection)
    if not (candle.high >= bb_upper and candle.close < bb_upper):
        return False

    body_size = abs(candle.close - candle.open_price)
    upper_wick = candle.high - max(candle.open_price, candle.close)
    lower_wick = min(candle.open_price, candle.close) - candle.low

    if body_size > 0:
        if upper_wick >= body_size * PIN_BAR_WICK_BODY_RATIO and lower_wick <= body_size * PIN_BAR_OPPOSITE_WICK_BODY_RATIO_MAX:
            return True
    elif body_size == 0: # Doji-like pin bar
        if upper_wick > 0 and lower_wick <= upper_wick * PIN_BAR_OPPOSITE_WICK_BODY_RATIO_MAX:
            return True
            
    return False

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    # Determine minimum required candles for all indicators
    # MFI needs MFI_PERIOD candles for current MFI, and MFI_PERIOD candles for previous MFI.
    # To get both, we need MFI_PERIOD + 1 total candles.
    # BBW_SMA needs BBW_SMA_PERIOD BBW values. Each BBW value needs BB_PERIOD candles.
    # So, we need BB_PERIOD + BBW_SMA_PERIOD - 1 total candles for BBW_SMA calculation.
    min_candles_required = max(BB_PERIOD, MFI_PERIOD + 1, VOLUME_SMA_PERIOD, BB_PERIOD + BBW_SMA_PERIOD - 1)

    for pair, pair_data in data.items():
        candles = pair_data.warm
        if len(candles) < min_candles_required:
            continue

        # Extract relevant data series
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        current_candle = candles[-1]

        # 1. Calculate Bollinger Bands for all historical points to derive BBW history
        all_bb_uppers = []
        all_bb_middles = []
        all_bb_lowers = []
        # Iterate from the first point where BB can be calculated up to the current candle
        for i in range(BB_PERIOD - 1, len(closes)):
            upper, middle, lower = _calculate_bollinger_bands(closes[:i+1], BB_PERIOD, BB_STD_DEV)
            all_bb_uppers.append(upper)
            all_bb_middles.append(middle)
            all_bb_lowers.append(lower)
        
        # We need at least BBW_SMA_PERIOD BB values to calculate BBW_SMA.
        # `all_bb_uppers` (and others) will have `len(candles) - BB_PERIOD + 1` elements.
        if len(all_bb_uppers) < BBW_SMA_PERIOD:
            continue

        # Get BBs for the current candle (last in the series)
        bb_upper = all_bb_uppers[-1]
        bb_middle = all_bb_middles[-1]
        bb_lower = all_bb_lowers[-1]

        # 2. Calculate MFI for current and previous candles
        # current_mfi uses the last MFI_PERIOD candles
        current_mfi = _calculate_mfi(candles[-MFI_PERIOD:], MFI_PERIOD)
        # previous_mfi uses MFI_PERIOD candles ending one period earlier
        previous_mfi = _calculate_mfi(candles[-MFI_PERIOD-1:-1], MFI_PERIOD)

        if current_mfi is None or previous_mfi is None:
            continue # Not enough data for MFI calculation (should be caught by min_candles_required)

        # 3. Calculate Volume SMA
        if len(volumes) < VOLUME_SMA_PERIOD: # Should be covered by min_candles_required
            continue
        volume_sma = np.mean(volumes[-VOLUME_SMA_PERIOD:])
        current_volume = current_candle.volume

        # 4. Calculate Bollinger Band Width (BBW) and BBW_SMA
        all_bbw = []
        for i in range(len(all_bb_uppers)):
            if all_bb_middles[i] != 0: 
                all_bbw.append((all_bb_uppers[i] - all_bb_lowers[i]) / all_bb_middles[i])
            else:
                # Handle cases where middle band could be zero (e.g., if prices are extremely low or data error)
                all_bbw.append(0.0) 

        if len(all_bbw) < BBW_SMA_PERIOD: # Should be covered by min_candles_required
            continue

        current_bbw = all_bbw[-1]
        bbw_sma = np.mean(all_bbw[-BBW_SMA_PERIOD:])
        
        # 5. Candlestick Pattern Detection
        is_bullish_rejection_pin_bar = _is_bullish_rejection_pin_bar(current_candle, bb_lower)
        is_bearish_rejection_pin_bar = _is_bearish_rejection_pin_bar(current_candle, bb_upper)

        # 6. Signal Conditions
        buy_signal = False
        sell_signal = False

        # Buy Signal
        if (is_bullish_rejection_pin_bar and
            current_mfi < MFI_OVERSOLD_THRESHOLD and
            previous_mfi < MFI_OVERSOLD_THRESHOLD and # MFI was also oversold previously
            current_mfi > previous_mfi and             # MFI showing signs of reversal (moving up)
            current_volume > volume_sma and
            current_bbw > bbw_sma):                     # Higher than average volatility
            buy_signal = True

        # Sell Signal
        if (is_bearish_rejection_pin_bar and
            current_mfi > MFI_OVERBOUGHT_THRESHOLD and
            previous_mfi > MFI_OVERBOUGHT_THRESHOLD and # MFI was also overbought previously
            current_mfi < previous_mfi and              # MFI showing signs of reversal (moving down)
            current_volume > volume_sma and
            current_bbw > bbw_sma):                     # Higher than average volatility
            sell_signal = True

        if buy_signal:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=RULE_ID
            ))
        elif sell_signal:
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_candle.close,
                rule_id=RULE_ID
            ))

    return signals