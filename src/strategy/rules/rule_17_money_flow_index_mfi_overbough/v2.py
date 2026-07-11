from __future__ import annotations
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Rule specific constants
MFI_PERIOD = 14
MFI_OVERBOUGHT = 70  # Adjusted from 80 to 70
MFI_OVERSOLD = 30    # Adjusted from 20 to 30
RULE_ID = "1739a028-6130-436d-8c7e-441d051e0c4c" # Updated rule ID


def _calculate_mfi(candles: list[WarmCandle], period: int) -> float | None:
    """
    Calculates the Money Flow Index (MFI) for the last candle in the provided list.
    Requires at least `period + 1` candles for calculation.
    """
    if len(candles) < period + 1:
        return None

    typical_prices: list[float] = [(c.high + c.low + c.close) / 3 for c in candles]
    raw_money_flow: list[float] = [tp * c.volume for tp, c in zip(typical_prices, candles)]

    positive_money_flow = 0.0
    negative_money_flow = 0.0

    # Iterate over the last `period` candles to sum positive and negative money flow
    # For each candle `i` in this window, compare its TP with `i-1`'s TP.
    # The loop starts from `len(candles) - period`, which is the first candle
    # in the `period`-long window that ends at `candles[-1]`.
    for i in range(len(candles) - period, len(candles)):
        current_tp = typical_prices[i]
        previous_tp = typical_prices[i - 1]
        rmf = raw_money_flow[i]

        if current_tp > previous_tp:
            positive_money_flow += rmf
        elif current_tp < previous_tp:
            negative_money_flow += rmf

    money_ratio: float
    if negative_money_flow == 0:
        if positive_money_flow == 0:
            # Both PMF and NMF are zero, implying no price movement with volume.
            # MFI is typically 50 in this case (neutral).
            money_ratio = 1.0
        else:
            # All money flow is positive, NMF is zero. MFI tends to 100.
            money_ratio = float('inf')
    else:
        money_ratio = positive_money_flow / negative_money_flow

    mfi: float
    if money_ratio == float('inf'):
        mfi = 100.0
    elif money_ratio == 0.0:  # This implies PMF is 0 and NMF > 0
        mfi = 0.0
    else:
        mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates Buy/Sell signals based on the Money Flow Index (MFI)
    overbought/oversold conditions, using adjusted thresholds (30/70).

    A Buy signal is generated when MFI drops below MFI_OVERSOLD (30).
    A Sell signal is generated when MFI rises above MFI_OVERBOUGHT (70).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm

        # We need at least MFI_PERIOD + 1 candles to calculate MFI for the last candle.
        # This allows for `MFI_PERIOD` comparisons of (TP_current vs TP_previous).
        if len(candles) < MFI_PERIOD + 1:
            continue

        mfi = _calculate_mfi(candles, MFI_PERIOD)

        if mfi is None:
            continue  # Should not happen if the length check above is correct

        latest_candle = candles[-1]
        timestamp = latest_candle.hour
        price = latest_candle.close

        # Generate Buy Signal: MFI falls below the oversold threshold (30)
        if mfi < MFI_OVERSOLD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=RULE_ID,
                confidence=1.0  # Default confidence
            ))
        # Generate Sell Signal: MFI rises above the overbought threshold (70)
        elif mfi > MFI_OVERBOUGHT:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=RULE_ID,
                confidence=1.0  # Default confidence
            ))

    return signals