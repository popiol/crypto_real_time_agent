from __future__ import annotations
from src.agent.models import BuySignal, MarketData, SellSignal

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "reversal_failed_breakout_v1"
    
    for pair, pair_data in data.items():
        # Ensure we have recent price data
        if not pair_data.hot:
            continue
        
        last_price = pair_data.hot[-1].last_price
        polled_at = pair_data.hot[-1].polled_at

        # Check for sufficient warm data for 8-hour range
        if len(pair_data.warm) < 8:
            continue
        
        eight_hour_candles = pair_data.warm[-8:]
        eight_hour_high = max([c.high for c in eight_hour_candles])
        eight_hour_low = min([c.low for c in eight_hour_candles])

        # Check for monthly average price data
        if not pair_data.cold:
            continue
        
        monthly_avg_price = pair_data.cold[-1].avg_price

        # SHORT signal condition
        if last_price > eight_hour_high and last_price > monthly_avg_price:
            signals.append(SellSignal(
                pair=pair,
                timestamp=polled_at,
                price=last_price,
                rule_id=rule_id,
                reason="Reversal of failed 8-hour range breakout buy signal"
            ))
        # LONG signal condition
        elif last_price < eight_hour_low and last_price < monthly_avg_price:
            signals.append(BuySignal(
                pair=pair,
                timestamp=polled_at,
                price=last_price,
                rule_id=rule_id,
                reason="Reversal of failed 8-hour range breakout sell signal"
            ))
            
    return signals