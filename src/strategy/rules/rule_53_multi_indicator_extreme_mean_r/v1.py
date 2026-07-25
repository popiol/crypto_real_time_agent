from datetime import datetime

# Assuming these are available from the environment based on problem description
from src.agent.models import BuySignal, SellSignal, MarketData, Tick, WarmCandle, ColdMonth

# --- Constants ---
RSI_PERIOD = 14
K_EMA_PERIOD = 20 # Period for Keltner Channel's EMA (middle band)
ATR_PERIOD = 10 # Period for Keltner Channel's ATR (width)

# Minimum data required for calculations.
# RSI(14) needs 15 candles (14 price changes for initial avg + current candle).
# EMA(20) needs 20 candles for its initial calculation (if using SMA for first N values).
# ATR(10) needs 11 candles (10 True Range values, each needing current and previous candle).
MIN_WARM_CANDLES = max(RSI_PERIOD + 1, K_EMA_PERIOD, ATR_PERIOD + 1)

MIN_TICKS_FOR_PRICE = 1 # Need at least one tick for current price
MIN_COLD_MONTHS = 1 # Need at least one cold month entry for monthly mean deviation

# --- Helper Functions ---

def _calculate_rsi(candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI) for the last candle's close price."""
    if len(candles) < period + 1:
        return None

    # Calculate price changes (deltas)
    price_changes = [candles[i].close - candles[i-1].close for i in range(1, len(candles))]

    # Initial average gain/loss over the first 'period' changes (SMA)
    initial_gains = [max(0, change) for change in price_changes[:period]]
    initial_losses = [abs(min(0, change)) for change in price_changes[:period]]

    avg_gain = sum(initial_gains) / period
    avg_loss = sum(initial_losses) / period

    # Apply EMA smoothing for subsequent periods to get the RSI value for the last candle
    for i in range(period, len(price_changes)):
        current_gain = max(0, price_changes[i])
        current_loss = abs(min(0, price_changes[i]))
        
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period

    if avg_loss == 0:
        # If there are no losses, RSI is 100 (if gains exist) or 50 (if flat market)
        return 100.0 if avg_gain > 0 else 50.0 
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def _calculate_keltner_deviation(candles: list[WarmCandle], current_price: float, ema_period: int, atr_period: int) -> float | None:
    """
    Calculates Keltner Channel Deviation for the current price.
    Deviation = (current_price - Keltner Middle Band) / Average True Range (ATR)
    """
    if len(candles) < max(ema_period, atr_period + 1):
        return None

    # --- Calculate Average True Range (ATR) ---
    true_ranges = []
    # True Range (TR) calculation requires the previous candle's close, so start from index 1.
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    
    if len(true_ranges) < atr_period: 
        return None

    # Initial ATR (Simple Moving Average of the first 'atr_period' True Ranges)
    current_atr = sum(true_ranges[:atr_period]) / atr_period
    
    # Subsequent ATR values using EMA smoothing
    alpha_atr = 2 / (atr_period + 1)
    for i in range(atr_period, len(true_ranges)):
        current_atr = (true_ranges[i] * alpha_atr) + (current_atr * (1 - alpha_atr))

    if current_atr == 0: # Avoid division by zero if ATR is flatlined (no volatility)
        return None

    # --- Calculate Exponential Moving Average (EMA) for Keltner Middle Band ---
    close_prices = [c.close for c in candles]
    
    if len(close_prices) < ema_period:
        return None

    # Initial EMA (Simple Moving Average of the first 'ema_period' close prices)
    current_ema = sum(close_prices[:ema_period]) / ema_period

    # Subsequent EMA values using smoothing
    alpha_ema = 2 / (ema_period + 1)
    for i in range(ema_period, len(close_prices)):
        current_ema = (close_prices[i] * alpha_ema) + (current_ema * (1 - alpha_ema))

    # --- Calculate Keltner Channel Deviation ---
    deviation = (current_price - current_ema) / current_atr
    return deviation

def _calculate_price_to_monthly_mean_deviation(current_price: float, cold_data: list[ColdMonth], current_datetime: datetime) -> float | None:
    """
    Calculates Price to Monthly Mean Deviation using the formula:
    (current_price - monthly_avg_price) / (monthly_max_price - monthly_min_price)
    This normalizes the deviation relative to the monthly price range.
    """
    if not cold_data:
        return None

    # Find the ColdMonth entry corresponding to the current month
    current_month_str = current_datetime.strftime("%Y-%m")
    relevant_month_data = None
    for cm in cold_data:
        if cm.month == current_month_str:
            relevant_month_data = cm
            break
    
    if relevant_month_data is None:
        return None

    monthly_avg_price = relevant_month_data.avg_price
    monthly_min_price = relevant_month_data.min_price
    monthly_max_price = relevant_month_data.max_price

    price_range = monthly_max_price - monthly_min_price

    # Handle cases where monthly price range is zero (e.g., very stable month or insufficient data)
    if price_range == 0:
        # If current_price is exactly the monthly_avg_price, deviation is 0.
        # Otherwise, if range is 0 but price is different, deviation is infinite, so return None.
        return 0.0 if current_price == monthly_avg_price else None
    
    deviation = (current_price - monthly_avg_price) / price_range
    return deviation


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates BUY/SELL signals based on the "Multi-Indicator Extreme Mean Reversion Entry" rule.
    This rule targets highly overextended market conditions for mean reversion by
    combining Hourly RSI, Keltner Channel Deviation, and Price to Monthly Mean Deviation.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        warm_candles = pair_data.warm
        cold_months = pair_data.cold

        # --- Data Validation ---
        # Ensure sufficient data is available for current price and indicator calculations
        if not ticks or len(ticks) < MIN_TICKS_FOR_PRICE:
            continue
        
        current_tick = ticks[-1]
        current_price = current_tick.last_price
        current_timestamp = current_tick.polled_at

        if not warm_candles or len(warm_candles) < MIN_WARM_CANDLES:
            continue
        
        if not cold_months or len(cold_months) < MIN_COLD_MONTHS:
            continue

        # --- Indicator Calculations ---
        hourly_rsi = _calculate_rsi(warm_candles, RSI_PERIOD)
        if hourly_rsi is None:
            continue

        keltner_deviation = _calculate_keltner_deviation(warm_candles, current_price, K_EMA_PERIOD, ATR_PERIOD)
        if keltner_deviation is None:
            continue
        
        price_to_monthly_mean_deviation = _calculate_price_to_monthly_mean_deviation(current_price, cold_months, current_timestamp)
        if price_to_monthly_mean_deviation is None:
            continue

        # --- Apply Trading Rule Logic ---
        # BUY Signal: HourlyRSI < 30 AND KeltnerChannelDeviation < -1.0 AND PriceToMonthlyMeanDeviation < -1.0
        if (hourly_rsi < 30 and 
            keltner_deviation < -1.0 and 
            price_to_monthly_mean_deviation < -1.0):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="multi_indicator_mean_reversion_v1",
                confidence=1.0 # Placeholder confidence
            ))
        
        # SELL Signal: HourlyRSI > 70 AND KeltnerChannelDeviation > 4.0 AND PriceToMonthlyMeanDeviation > 4.0
        elif (hourly_rsi > 70 and 
              keltner_deviation > 4.0 and 
              price_to_monthly_mean_deviation > 4.0):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="multi_indicator_mean_reversion_v1",
                confidence=1.0 # Placeholder confidence
            ))

    return signals