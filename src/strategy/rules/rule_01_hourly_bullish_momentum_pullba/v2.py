from __future__ import annotations
import math
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle

# --- Parameters for Adjustment (Broadened from previous) ---
MIN_BULLISH_BODY_TO_RANGE_RATIO = 0.50
MIN_BULLISH_CANDLE_BODY_PERCENT_OF_ATR = 1.2
MAX_PULLBACK_PERCENT_FROM_HIGH = 0.04
MIN_PULLBACK_PERCENT_FROM_HIGH = 0.002
HOT_VOL_LOOKBACK_TICKS = 60
HOT_SPREAD_LOOKBACK_TICKS = 30
ATR_LOOKBACK_WARM_CANDLES = 10 # Hourly candles for ATR, must be < 24
MAX_ACCEPTABLE_SPREAD_PERCENT = 0.002 # 0.2% of current price
MAX_HOT_VOLATILITY_MULTIPLIER_VS_PREV = 1.2 # Current hot vol should not be more than 20% higher than previous window

RULE_ID = "fix-bullish-pullback-broaden"

def calculate_atr(warm_data: list[WarmCandle], lookback: int) -> float | None:
    """
    Calculates Average True Range (ATR) for a given lookback period.
    Requires `lookback + 1` candles for `lookback` TR values.
    """
    if len(warm_data) < lookback + 1:
        return None

    true_ranges = []
    # Loop backwards from the latest candle to include `lookback` candles for TR calculation.
    # For `i=1`, current_candle is `warm_data[-1]`, prev_candle is `warm_data[-2]`.
    # For `i=lookback`, current_candle is `warm_data[-lookback]`, prev_candle is `warm_data[-(lookback+1)]`.
    for i in range(1, lookback + 1):
        current_candle = warm_data[-i]
        
        # prev_candle is the one chronologically *before* current_candle for TR calculation
        prev_candle = warm_data[-(i + 1)] # This index is guaranteed to exist by the initial len check

        high_low = current_candle.high - current_candle.low
        
        # Calculate True Range (TR)
        tr = max(high_low, abs(current_candle.high - prev_candle.close), abs(current_candle.low - prev_candle.close))
        true_ranges.append(tr)

    if not true_ranges: # Should not happen if len check passes and lookback > 0
        return None
    
    return sum(true_ranges) / len(true_ranges)

def calculate_hot_volatility(ticks_window: list[Tick], lookback: int) -> float | None:
    """
    Calculates standard deviation of log returns for a given lookback period
    from a provided list of ticks.
    Requires `lookback + 1` ticks for `lookback` log returns.
    """
    if len(ticks_window) < lookback + 1:
        return None

    log_returns = []
    # Ticks are ordered chronologically. We need `lookback` returns.
    # The ticks_window is already sliced to contain the necessary ticks.
    # Example: if lookback=60, ticks_window should have 61 ticks.
    # We calculate returns from ticks_window[0] vs ticks_window[1] up to ticks_window[59] vs ticks_window[60].
    for i in range(1, lookback + 1):
        current_tick = ticks_window[-i]
        previous_tick = ticks_window[-i-1] 

        if previous_tick.last_price > 0 and current_tick.last_price > 0:
            log_returns.append(math.log(current_tick.last_price / previous_tick.last_price))
        # If prices are invalid (e.g., zero), skip this return.
        # This means the effective lookback for volatility might be smaller.

    if len(log_returns) < 2: # statistics.stdev requires at least 2 data points
        return 0.0 # Return 0.0 to avoid errors and allow comparisons, effectively low volatility

    return statistics.stdev(log_returns)

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # 1. Ensure sufficient warm data for ATR and candle analysis
        # Need ATR_LOOKBACK_WARM_CANDLES for ATR calculation + 1 for the latest_warm_candle itself.
        # Total needed: ATR_LOOKBACK_WARM_CANDLES + 1.
        if len(pair_data.warm) < ATR_LOOKBACK_WARM_CANDLES + 1:
            continue

        latest_warm_candle = pair_data.warm[-1]
        
        # Calculate Average True Range (ATR) from warm data
        avg_true_range = calculate_atr(pair_data.warm, ATR_LOOKBACK_WARM_CANDLES)
        if avg_true_range is None or avg_true_range <= 0: # ATR must be positive for meaningful comparison
            continue

        # 2. Condition: Strong Bullish Candle (latest_warm_candle)
        is_strong_bullish_candle = False
        if latest_warm_candle.close > latest_warm_candle.open_price:
            candle_body_size = latest_warm_candle.close - latest_warm_candle.open_price
            candle_range = latest_warm_candle.high - latest_warm_candle.low

            if candle_range > 0 and (candle_body_size / candle_range) >= MIN_BULLISH_BODY_TO_RANGE_RATIO:
                if candle_body_size >= MIN_BULLISH_CANDLE_BODY_PERCENT_OF_ATR * avg_true_range:
                    is_strong_bullish_candle = True

        if not is_strong_bullish_candle:
            continue

        # 3. Ensure sufficient hot data for current price and volatility/spread
        # For current volatility: HOT_VOL_LOOKBACK_TICKS ticks for returns, plus one prior tick for the first return. So HOT_VOL_LOOKBACK_TICKS + 1.
        # For previous volatility: another HOT_VOL_LOOKBACK_TICKS ticks for returns, plus one prior. So (HOT_VOL_LOOKBACK_TICKS * 2) + 1.
        # For spread: HOT_SPREAD_LOOKBACK_TICKS ticks for spread calculation.
        required_hot_ticks_for_vol = (HOT_VOL_LOOKBACK_TICKS * 2) + 1
        required_hot_ticks_for_spread = HOT_SPREAD_LOOKBACK_TICKS 
        
        if len(pair_data.hot) < max(required_hot_ticks_for_vol, required_hot_ticks_for_spread):
            continue

        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        high_of_strong_candle = latest_warm_candle.high
        open_of_strong_candle = latest_warm_candle.open_price

        # 4. Condition: Slight Price Pullback
        # Price must be below the high but above the open of the strong candle
        # And within a defined, broader pullback percentage range
        is_slight_pullback = False
        if current_price < high_of_strong_candle and current_price > open_of_strong_candle:
            if high_of_strong_candle > 0: # Avoid division by zero
                pullback_percentage = (high_of_strong_candle - current_price) / high_of_strong_candle
                if MIN_PULLBACK_PERCENT_FROM_HIGH <= pullback_percentage <= MAX_PULLBACK_PERCENT_FROM_HIGH:
                    is_slight_pullback = True

        if not is_slight_pullback:
            # If the conditions for a buy signal are not met, check for a sell signal.
            # A price drop below the strong candle's open indicates a strong invalidation of the bullish thesis.
            if current_price < open_of_strong_candle:
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=current_timestamp,
                    price=current_price,
                    rule_id=RULE_ID,
                    confidence=1.0
                ))
            continue # If not a pullback, no buy signal

        # 5. Condition: Confirmation via Hot Volatility and Spread (less restrictive)
        # Check average bid-ask spread
        total_spread = 0.0
        count_spreads = 0
        
        # Iterate over the last HOT_SPREAD_LOOKBACK_TICKS ticks for spread
        for i in range(1, HOT_SPREAD_LOOKBACK_TICKS + 1):
            tick = pair_data.hot[-i]
            if tick.bid_price is not None and tick.ask_price is not None and tick.bid_price > 0 and tick.ask_price > 0:
                total_spread += (tick.ask_price - tick.bid_price)
                count_spreads += 1
        
        avg_bid_ask_spread_hot = total_spread / count_spreads if count_spreads > 0 else 0.0

        if current_price > 0 and avg_bid_ask_spread_hot > (current_price * MAX_ACCEPTABLE_SPREAD_PERCENT):
            continue # Spread is too wide, indicating uncertainty or low liquidity

        # Check for non-spiking volatility (current volatility not significantly higher than previous)
        # Slicing for recent hot volatility: last HOT_VOL_LOOKBACK_TICKS + 1 ticks
        recent_hot_vol = calculate_hot_volatility(pair_data.hot[-HOT_VOL_LOOKBACK_TICKS-1:], HOT_VOL_LOOKBACK_TICKS)
        
        # Slicing for previous hot volatility: the HOT_VOL_LOOKBACK_TICKS + 1 ticks before the recent window
        prev_hot_vol = calculate_hot_volatility(
            pair_data.hot[-(HOT_VOL_LOOKBACK_TICKS * 2)-1 : -HOT_VOL_LOOKBACK_TICKS], 
            HOT_VOL_LOOKBACK_TICKS
        )
        
        # If volatility calculation failed (e.g., not enough valid price points in window)
        if recent_hot_vol is None or prev_hot_vol is None:
            continue

        # Only filter if volatility is spiking significantly
        # If prev_hot_vol is 0, it indicates extreme stability or insufficient data for previous window std dev.
        # In such cases, we don't want to trigger a false positive for "spiking".
        if prev_hot_vol > 0 and recent_hot_vol > prev_hot_vol * MAX_HOT_VOLATILITY_MULTIPLIER_VS_PREV:
            continue # Volatility is spiking excessively, not a calm pullback

        # If all conditions met, generate a BuySignal
        signals.append(BuySignal(
            pair=pair,
            timestamp=current_timestamp,
            price=current_price,
            rule_id=RULE_ID,
            confidence=1.0
        ))

    return signals