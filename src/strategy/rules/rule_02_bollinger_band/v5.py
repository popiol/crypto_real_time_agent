from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle


# Bollinger Band parameters
BB_PERIOD = 20  # 20-period Simple Moving Average and Standard Deviation for prices
K = 2.0         # Multiplier for Standard Deviation (width of Bollinger Bands)

# Volume confirmation parameters
VOL_PERIOD = 20 # Period for Simple Moving Average and Standard Deviation for volume
VOLUME_K = 1.5  # Multiplier for Standard Deviation for volume confirmation threshold

# State management for Bollinger Band breach status per pair
# This dictionary stores the last known breach status for each currency pair.
# It persists across calls to the `signal` function, allowing the rule to track
# whether a band breach has occurred and is awaiting re-entry confirmation.
_band_breach_status: dict[str, str] = {}

# Status codes for clarity
_STATUS_NONE = "none"           # Price is within bands, or no active breach to confirm
_STATUS_BELOW_LOWER = "below_lower" # Price has dropped below the lower band
_STATUS_ABOVE_UPPER = "above_upper" # Price has risen above the upper band


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Bollinger Band Reversal with Volume Confirmation trading rule.

    A Buy signal is generated when the price, after dropping below the lower
    Bollinger Band, subsequently closes at or above it, AND this reversal is
    accompanied by significantly higher than average trading volume.

    Similarly, a Sell signal is generated when the price, after rising above the
    upper band, subsequently closes at or below it, AND this reversal is
    accompanied by significantly higher than average trading volume.

    This rule refines a previous Bollinger Band strategy by adding a volume
    confirmation mechanism to improve signal reliability.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure sufficient data for Bollinger Band and Volume calculations.
        # We need at least BB_PERIOD warm candles for price bands and
        # VOL_PERIOD hot ticks for volume analysis.
        if len(warm_candles) < BB_PERIOD or len(hot_ticks) < VOL_PERIOD:
            _band_breach_status[pair] = _STATUS_NONE
            continue

        # --- 1-4. Calculate Bollinger Bands using `warm` (hourly) candles. ---
        # Use the last BB_PERIOD candles for calculations
        closes = [c.close for c in warm_candles[-BB_PERIOD:]]
        
        # Calculate mean and standard deviation for prices
        mean_price = statistics.mean(closes)
        std_price = statistics.stdev(closes) if len(closes) > 1 else 0.0

        if std_price == 0:
            # If standard deviation is zero, Bollinger Bands are collapsed to the mean.
            # No meaningful signals can be generated. Reset state and skip.
            _band_breach_status[pair] = _STATUS_NONE
            continue

        lower_band = mean_price - K * std_price
        upper_band = mean_price + K * std_price

        # Get the current price and timestamp from the latest tick in `hot` data.
        current_tick = hot_ticks[-1]
        current_price = current_tick.last_price
        ts = current_tick.polled_at

        # --- 5-7. Calculate Volume Confirmation Metrics. ---
        # The `volume_24h` from `Tick` is a rolling 24-hour volume.
        # We use the latest `volume_24h` as current volume and calculate its
        # average and standard deviation over the last VOL_PERIOD ticks' `volume_24h` values.
        current_volume = current_tick.volume_24h
        
        # Use the last VOL_PERIOD ticks' volume_24h for calculations
        historical_volumes_24h = [t.volume_24h for t in hot_ticks[-VOL_PERIOD:]]

        # Calculate mean and standard deviation for volumes
        mean_volume = statistics.mean(historical_volumes_24h)
        std_volume = statistics.stdev(historical_volumes_24h) if len(historical_volumes_24h) > 1 else 0.0

        # Define volume confirmation threshold
        volume_threshold_met = False
        if std_volume > 0:
            # Current volume must be significantly higher than average + 1.5 * SD
            volume_threshold_met = current_volume > (mean_volume + VOLUME_K * std_volume)
        elif mean_volume > 0:
            # If std_volume is 0 (all historical volumes are identical and non-zero),
            # check if current volume is strictly greater than this constant mean.
            volume_threshold_met = current_volume > mean_volume
        # If mean_volume is 0 (all historical volumes are zero), volume_threshold_met remains False.

        # Retrieve the current breach status for this pair.
        # Default to _STATUS_NONE if no prior state exists.
        current_breach_status = _band_breach_status.get(pair, _STATUS_NONE)

        # Apply the Bollinger Band Reversal with Volume Confirmation logic.
        if current_price < lower_band:
            # Price has dropped below the lower band, indicating a potential oversold condition.
            # Update status to await re-entry confirmation.
            _band_breach_status[pair] = _STATUS_BELOW_LOWER
        elif current_price > upper_band:
            # Price has risen above the upper band, indicating a potential overbought condition.
            # Update status to await re-entry confirmation.
            _band_breach_status[pair] = _STATUS_ABOVE_UPPER
        else:  # current_price is within or at the bands (lower_band <= current_price <= upper_band)
            if current_breach_status == _STATUS_BELOW_LOWER:
                # Price was previously below the lower band and has now closed back inside.
                # This is a potential buy signal. Check volume confirmation.
                if volume_threshold_met:
                    signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price, rule_id="BollingerBandReversalWithVolume"))
                _band_breach_status[pair] = _STATUS_NONE  # Reset status after potential signal or failed confirmation
            elif current_breach_status == _STATUS_ABOVE_UPPER:
                # Price was previously above the upper band and has now closed back inside.
                # This is a potential sell signal. Check volume confirmation.
                if volume_threshold_met:
                    signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price, rule_id="BollingerBandReversalWithVolume"))
                _band_breach_status[pair] = _STATUS_NONE  # Reset status after potential signal or failed confirmation
            else:
                # Price is within bands, and there was no active breach to confirm.
                _band_breach_status[pair] = _STATUS_NONE

    return signals