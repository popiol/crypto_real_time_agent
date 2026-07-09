from __future__ import annotations

import cmath
import math
import statistics
from datetime import datetime

# Assuming these models are available in the execution environment
from pydantic import BaseModel, Field


# --- Data Models (Copied from description for self-contained module) ---
class Tick(BaseModel):
    """A single poll snapshot for one currency pair."""

    pair: str
    polled_at: datetime

    # Last trade
    last_price: float

    # Best bid / ask from Ticker
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float

    # 24-hour rolling volume in base currency (from Kraken Ticker v[1])
    volume_24h: float = 0.0

    # Derived
    mid_price: float
    spread_abs: float  # ask - bid
    spread_rel: float  # (ask - bid) / mid  * 100  (%)

    # Top-5 order book (from Depth endpoint)
    order_book: dict | None = None # Using dict as a placeholder for OrderBook


class WarmCandle(BaseModel):
    hour: datetime
    open_price: float
    high: float
    low: float
    close: float
    avg_spread_rel: float = 0.0
    # IMPORTANT: The pseudocode for VWAP requires 'VOLUME'.
    # The provided WarmCandle model *does not* include a 'volume' attribute.
    # For this implementation, we assume a 'volume' attribute exists on WarmCandle
    # or is made available through an extension. Without it, VWAP cannot be calculated.
    volume: float = 0.0  # Placeholder: This needs actual data to be effective.


class ColdMonth(BaseModel):
    month: str  # "YYYY-MM"
    min_price: float
    max_price: float
    avg_price: float
    avg_daily_spread: float
    candle_count: int
    last_candle_hour: datetime


class PairData(BaseModel):
    hot: list[Tick] = Field(
        default=[],
        description="TTL-capped; ~300 ticks at 1 poll/sec with default 300s retention",
    )
    warm: list[WarmCandle] = Field(
        default=[], description="At most 24 entries (last 24 hourly candles)"
    )
    cold: list[ColdMonth] = Field(
        default=[], description="One entry per calendar month; unbounded"
    )


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

# --- Rule Parameters ---
# FFT Parameters (adapted from pseudocode and existing rule)
# MIN_CANDLES: Minimum warm candles required for FFT analysis.
MIN_CANDLES = 12
# DOMINANT_CYCLE_THRESHOLD: Minimum amplitude of the dominant cycle relative to price std.
# From pseudocode (0.5), replaces existing rule's AMPLITUDE_THRESHOLD (0.3).
DOMINANT_CYCLE_THRESHOLD = 0.5
# TROUGH_THRESHOLD: Cosine of phase must be below -TROUGH_THRESHOLD for a trough,
# and above TROUGH_THRESHOLD for a peak. From existing rule.
TROUGH_THRESHOLD = 0.7

# VWAP Parameters
VWAP_PERIOD = 14

# Stochastic Oscillator Parameters
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
STOCHASTIC_OVERSOLD = 20
STOCHASTIC_OVERBOUGHT = 80


# --- Helper Functions ---

def _detrend(series: list[float]) -> list[float]:
    """
    Removes the least-squares linear trend from a series.
    This helps the DFT reflect cycles rather than overall price drift.
    """
    n = len(series)
    if n < 2:
        return [0.0] * n
    
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(series)
    
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, series))
    den = sum((xi - x_mean) ** 2 for xi in xs)
    
    slope = num / den if den else 0.0
    intercept = y_mean - slope * x_mean
    
    return [yi - (slope * xi + intercept) for xi, yi in zip(xs, series)]


def _dft(series: list[float]) -> list[complex]:
    """
    Computes the Discrete Fourier Transform (DFT) of a series.
    A naive O(n^2) implementation, sufficient for small window sizes (e.g., <= 24).
    """
    n = len(series)
    if n == 0:
        return []
    
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
        for k in range(n)
    ]


def _calculate_vwap(
    closes: list[float], highs: list[float], lows: list[float], volumes: list[float], period: int
) -> list[float]:
    """
    Calculates the Volume Weighted Average Price (VWAP) for each point in the series.
    VWAP is typically calculated using (High + Low + Close) / 3 as the price.
    Returns a list of VWAP values, with 0.0 for initial periods where data is insufficient.
    """
    if len(closes) < period or period <= 0:
        return [0.0] * len(closes)

    vwaps = []
    for i in range(len(closes)):
        if i < period - 1:
            vwaps.append(0.0)  # Not enough data for the initial period
            continue

        period_closes = closes[i - period + 1 : i + 1]
        period_highs = highs[i - period + 1 : i + 1]
        period_lows = lows[i - period + 1 : i + 1]
        period_volumes = volumes[i - period + 1 : i + 1]

        typical_prices = [(h + l + c) / 3 for h, l, c in zip(period_highs, period_lows, period_closes)]
        
        sum_pv = sum(tp * v for tp, v in zip(typical_prices, period_volumes))
        sum_v = sum(period_volumes)

        if sum_v > 0:
            vwaps.append(sum_pv / sum_v)
        else:
            # Fallback to the last typical price if no volume in the period
            # This can happen if all volumes are 0.0, which might occur with placeholder data.
            vwaps.append(typical_prices[-1] if typical_prices else 0.0)

    return vwaps


def _calculate_stochastic(
    closes: list[float], highs: list[float], lows: list[float], k_period: int, d_period: int
) -> tuple[list[float], list[float]]:
    """
    Calculates the Stochastic Oscillator (%K and %D).
    %K = ((Current Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    %D = Simple Moving Average of %K over d_period.
    Returns two lists: %K values and %D values.
    """
    if len(closes) < k_period or k_period <= 0 or d_period <= 0:
        return [0.0] * len(closes), [0.0] * len(closes)

    percent_k: list[float] = []
    for i in range(len(closes)):
        if i < k_period - 1:
            percent_k.append(0.0)  # Not enough data for the initial %K calculation
            continue

        period_closes = closes[i - k_period + 1 : i + 1]
        period_highs = highs[i - k_period + 1 : i + 1]
        period_lows = lows[i - k_period + 1 : i + 1]

        current_close = period_closes[-1]
        lowest_low = min(period_lows)
        highest_high = max(period_highs)

        range_hl = highest_high - lowest_low
        if range_hl > 0:
            k = ((current_close - lowest_low) / range_hl) * 100
        else:
            k = 50.0  # Default to mid-range if no price movement in the period
        percent_k.append(k)

    # Calculate %D as SMA of %K
    percent_d: list[float] = []
    for i in range(len(percent_k)):
        if i < d_period - 1:
            percent_d.append(0.0)  # Not enough data for the initial %D calculation
            continue
        
        # Only average non-zero %K values that have been calculated
        valid_k_values = [val for val in percent_k[i - d_period + 1 : i + 1] if val != 0.0]
        if valid_k_values:
            percent_d.append(statistics.mean(valid_k_values))
        else:
            percent_d.append(0.0)

    return percent_k, percent_d


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'FFT Cycle with VWAP Cross and Stochastic Confirmation' trading rule.

    This rule generates Buy signals when:
    - The dominant FFT cycle indicates a trough.
    - The current price crosses above VWAP.
    - The Stochastic Oscillator (%K) indicates an oversold condition.

    It generates Sell signals when:
    - The dominant FFT cycle indicates a peak.
    - The current price crosses below VWAP.
    - The Stochastic Oscillator (%K) indicates an overbought condition.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candles for analysis and at least one hot tick for current price/timestamp.
        # The FFT_WINDOW_SIZE implicitly becomes len(pair_data.warm) as per the original rule's context.
        if len(pair_data.warm) < MIN_CANDLES or not pair_data.hot:
            continue

        # Extract data from warm candles
        closes = [c.close for c in pair_data.warm]
        highs = [c.high for c in pair_data.warm]
        lows = [c.low for c in pair_data.warm]
        volumes = [c.volume for c in pair_data.warm] # Critical assumption: WarmCandle has 'volume'

        n = len(closes)
        if n == 0:
            continue

        # --- 1. FFT Cycle Detection ---
        detrended_prices = _detrend(closes)
        X = _dft(detrended_prices)

        # Find dominant cycle (excluding DC component k=0 and higher frequencies > n/2)
        if n < 2: # Need at least 2 points for a cycle
             continue
        
        # Ensure there's a valid range for k_star (i.e., n//2 >= 1)
        if n // 2 < 1:
            continue

        try:
            # k_star is the frequency with the largest amplitude in the first half of the spectrum
            k_star = max(range(1, n // 2 + 1), key=lambda k: abs(X[k]))
        except ValueError: # This can happen if range(1, n//2 + 1) is empty
            continue

        amplitude = 2 * abs(X[k_star]) / n
        
        # Calculate standard deviation for amplitude thresholding
        price_std = statistics.stdev(closes) if len(closes) > 1 else 0.0

        # Filter out noise-driven detections: dominant cycle amplitude must be significant
        if price_std == 0 or amplitude / price_std < DOMINANT_CYCLE_THRESHOLD:
            continue

        # Calculate the phase of the dominant cycle at the latest data point (index n-1)
        # φ = 2π · k* · (N-1) / N + angle(X[k*])
        phase = 2 * math.pi * k_star * (n - 1) / n + cmath.phase(X[k_star])
        cos_phase = math.cos(phase)

        cycle_phase_status: str | None = None
        if cos_phase < -TROUGH_THRESHOLD:
            cycle_phase_status = "trough"
        elif cos_phase > TROUGH_THRESHOLD:
            cycle_phase_status = "peak"
        
        if cycle_phase_status is None:
            continue # No clear peak or trough detected based on threshold

        # --- 2. VWAP Calculation ---
        vwaps = _calculate_vwap(closes, highs, lows, volumes, VWAP_PERIOD)
        if not vwaps or vwaps[-1] == 0.0: # Ensure VWAP was successfully calculated
            continue
        latest_vwap = vwaps[-1]

        # --- 3. Stochastic Oscillator Calculation ---
        stoch_k_values, stoch_d_values = _calculate_stochastic(
            closes, highs, lows, STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD
        )
        # Ensure Stochastic %K was successfully calculated and is not the initial 0.0 placeholder
        if not stoch_k_values or stoch_k_values[-1] == 0.0: 
            continue
        latest_stoch_k = stoch_k_values[-1]

        # --- Get Latest Market Data (from hot ticks) ---
        ts = pair_data.hot[-1].polled_at
        latest_close = pair_data.hot[-1].last_price

        # --- Generate Signals based on combined conditions ---
        if (
            cycle_phase_status == "trough"
            and latest_close > latest_vwap
            and latest_stoch_k < STOCHASTIC_OVERSOLD
        ):
            signals.append(BuySignal(pair=pair, timestamp=ts, price=latest_close))
        elif (
            cycle_phase_status == "peak"
            and latest_close < latest_vwap
            and latest_stoch_k > STOCHASTIC_OVERBOUGHT
        ):
            signals.append(SellSignal(pair=pair, timestamp=ts, price=latest_close))

    return signals