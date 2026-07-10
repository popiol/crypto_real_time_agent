from __future__ import annotations

import statistics
from datetime import datetime

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle, ColdMonth


K = 2.0
MIN_CANDLES = 10

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
    Implements the Bollinger Band Re-entry Confirmation trading rule.

    A Buy signal is generated only after the price has first dropped below the lower
    Bollinger Band and subsequently closes *above* or *at* the lower band (i.e., re-enters).
    A Sell signal is generated only after the price has first risen above the upper
    Bollinger Band and subsequently closes *below* or *at* the upper band (i.e., re-enters).

    This rule enhances a basic Bollinger Band strategy by requiring confirmation of
    reversal, aiming to reduce false signals.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient data for Bollinger Band calculation.
        # If data is insufficient, reset the breach status for this pair to avoid
        # stale state triggering false signals when data eventually becomes available.
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            _band_breach_status[pair] = _STATUS_NONE
            continue

        # Calculate Bollinger Bands using `warm` (hourly) candles.
        closes = [c.close for c in pair_data.warm]
        if len(closes) < MIN_CANDLES:
            _band_breach_status[pair] = _STATUS_NONE
            continue

        mean = statistics.mean(closes)
        
        # Calculate standard deviation. Handle cases where it might be zero
        # (e.g., all closes are identical), which would make bands collapse.
        if len(closes) > 1:
            std = statistics.stdev(closes)
        else:
            # If only one or zero closes, standard deviation is undefined or zero.
            # Treat as zero for practical purposes.
            std = 0.0

        if std == 0:
            # If standard deviation is zero, Bollinger Bands are collapsed to the mean.
            # No meaningful signals can be generated. Reset state and skip.
            _band_breach_status[pair] = _STATUS_NONE
            continue

        lower_band = mean - K * std
        upper_band = mean + K * std

        # Get the current price and timestamp from the latest tick in `hot` data.
        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # Retrieve the current breach status for this pair.
        # Default to _STATUS_NONE if no prior state exists.
        current_breach_status = _band_breach_status.get(pair, _STATUS_NONE)

        # Apply the Bollinger Band Re-entry Confirmation logic based on the pseudocode.
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
                # This confirms the mean-reversion for a buy signal.
                signals.append(BuySignal(pair=pair, timestamp=ts, price=current_price))
                _band_breach_status[pair] = _STATUS_NONE  # Reset status after signal generation
            elif current_breach_status == _STATUS_ABOVE_UPPER:
                # Price was previously above the upper band and has now closed back inside.
                # This confirms the mean-reversion for a sell signal.
                signals.append(SellSignal(pair=pair, timestamp=ts, price=current_price))
                _band_breach_status[pair] = _STATUS_NONE  # Reset status after signal generation
            else:
                # Price is within bands, and there was no active breach to confirm.
                # Or, a breach was confirmed in a previous iteration and the status was already reset.
                _band_breach_status[pair] = _STATUS_NONE

    return signals