"""Virtual portfolio — tracks cash, positions, and pending limit orders.

Runs on every data pull cycle:
  1. Fill any pending orders whose limit price has been reached.
  2. Find the best rule by recent_avg_gain_pct from rule_evaluation.json.
  3. If the best rule exceeds the configured threshold, place new orders
     from its signals (buy 1/10 of current capital per signal).
  4. Persist state to portfolio/portfolio.json and append fills to portfolio/transactions.json.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.models import AppConfig, BuySignal, SellSignal, Tick

logger = logging.getLogger(__name__)


class Position(BaseModel):
    position_id: str
    pair: str
    rule_id: str
    quantity: float
    buy_price: float
    value: float = 0.0
    opened_at: datetime


class Order(BaseModel):
    order_id: str
    direction: Literal["buy", "sell"]
    pair: str
    rule_id: str
    limit_price: float
    quantity: float
    value: float
    position_id: str | None = None
    created_at: datetime


class Transaction(BaseModel):
    transaction_id: str
    pair: str
    rule_id: str
    quantity: float
    buy_price: float
    sell_price: float
    revenue: float
    gain_pct: float
    opened_at: datetime
    closed_at: datetime


class Portfolio(BaseModel):
    cash: float
    value: float = 0.0
    positions: list[Position] = Field(default_factory=list)
    pending_orders: list[Order] = Field(default_factory=list)

    def capital(self, current_prices: dict[str, float]) -> float:
        positions_value = sum(
            p.quantity * current_prices[p.pair]
            for p in self.positions
            if p.pair in current_prices
        )
        return self.cash + positions_value


def run_cycle(
    ticks: list[Tick],
    signals: list[BuySignal | SellSignal],
    config: AppConfig,
) -> None:
    if not ticks:
        return
    now = max(t.polled_at for t in ticks)
    portfolio = _load(config)
    current_prices = {t.pair: t.last_price for t in ticks}

    new_transactions = _fill_orders(portfolio, current_prices, now, config.portfolio_fee)
    _close_stale_positions(portfolio, current_prices, now)

    best_rule = _find_best_rule(config.state_dir, config.portfolio_min_recent_gain)
    if best_rule:
        _place_orders(portfolio, signals, best_rule, now, current_prices)
    else:
        logger.debug("No rule meets recent gain threshold (%.4f); no new orders", config.portfolio_min_recent_gain)

    for pos in portfolio.positions:
        pos.value = pos.quantity * current_prices.get(pos.pair, 0.0)
    portfolio.value = portfolio.capital(current_prices)

    _save(portfolio, config)
    if new_transactions:
        _append_transactions(new_transactions, config)


def _fill_orders(
    portfolio: Portfolio,
    current_prices: dict[str, float],
    now: datetime,
    fee: float,
) -> list[Transaction]:
    filled_ids: set[str] = set()
    transactions: list[Transaction] = []

    for order in portfolio.pending_orders:
        price = current_prices.get(order.pair)
        if price is None:
            continue

        if order.direction == "buy" and price < order.limit_price:
            cost = order.quantity * order.limit_price * (1 + fee)
            if portfolio.cash < cost:
                logger.warning(
                    "Skipping BUY fill for %s: insufficient cash (%.2f < %.2f)",
                    order.pair, portfolio.cash, cost,
                )
                continue
            portfolio.cash -= cost
            portfolio.positions.append(Position(
                position_id=str(uuid.uuid4()),
                pair=order.pair,
                rule_id=order.rule_id,
                quantity=order.quantity,
                buy_price=order.limit_price,
                opened_at=now,
            ))
            filled_ids.add(order.order_id)
            logger.info(
                "Filled BUY  %s  qty=%.6f @ %.4f  cost=%.2f (fee=%.2f)  cash_remaining=%.2f",
                order.pair, order.quantity, order.limit_price,
                cost, cost - order.quantity * order.limit_price, portfolio.cash,
            )

        elif order.direction == "sell" and price > order.limit_price:
            pos = next((p for p in portfolio.positions if p.position_id == order.position_id), None)
            revenue = order.quantity * order.limit_price * (1 - fee)
            portfolio.cash += revenue
            portfolio.positions = [p for p in portfolio.positions if p.position_id != order.position_id]
            filled_ids.add(order.order_id)
            logger.info(
                "Filled SELL %s  qty=%.6f @ %.4f  revenue=%.2f (fee=%.2f)  cash=%.2f",
                order.pair, order.quantity, order.limit_price,
                revenue, order.quantity * order.limit_price - revenue, portfolio.cash,
            )
            if pos is not None:
                gain_pct = (order.limit_price - pos.buy_price) / pos.buy_price
                transactions.append(Transaction(
                    transaction_id=str(uuid.uuid4()),
                    pair=order.pair,
                    rule_id=order.rule_id,
                    quantity=order.quantity,
                    buy_price=pos.buy_price,
                    sell_price=order.limit_price,
                    revenue=revenue,
                    gain_pct=gain_pct,
                    opened_at=pos.opened_at,
                    closed_at=now,
                ))

    portfolio.pending_orders = [o for o in portfolio.pending_orders if o.order_id not in filled_ids]
    return transactions


def _close_stale_positions(portfolio: Portfolio, current_prices: dict[str, float], now: datetime) -> None:
    cutoff = now - timedelta(hours=24)
    for pos in portfolio.positions:
        if pos.opened_at > cutoff:
            continue
        price = current_prices.get(pos.pair)
        if price is None:
            continue
        portfolio.pending_orders = [
            o for o in portfolio.pending_orders
            if not (o.direction == "sell" and o.position_id == pos.position_id)
        ]
        portfolio.pending_orders.append(Order(
            order_id=str(uuid.uuid4()),
            direction="sell",
            pair=pos.pair,
            rule_id=pos.rule_id,
            limit_price=price,
            quantity=pos.quantity,
            value=pos.quantity * price,
            position_id=pos.position_id,
            created_at=now,
        ))
        logger.info(
            "Auto-closing stale position %s %s (held >24h): qty=%.6f @ %.4f",
            pos.position_id[:8], pos.pair, pos.quantity, price,
        )


def _place_orders(
    portfolio: Portfolio,
    signals: list[BuySignal | SellSignal],
    best_rule: str,
    now: datetime,
    current_prices: dict[str, float],
) -> None:
    for signal in signals:
        if signal.rule_id != best_rule:
            continue
        if isinstance(signal, BuySignal):
            _place_buy(portfolio, signal, now, current_prices)
        elif isinstance(signal, SellSignal):
            _place_sell(portfolio, signal, now)


def _place_buy(
    portfolio: Portfolio,
    signal: BuySignal,
    now: datetime,
    current_prices: dict[str, float],
) -> None:
    if len(portfolio.positions) >= 10:
        return
    spend = portfolio.capital(current_prices) / 10
    committed = sum(o.value for o in portfolio.pending_orders if o.direction == "buy")
    if spend <= 0 or portfolio.cash - committed < spend:
        return
    qty = spend / signal.price
    portfolio.pending_orders.append(Order(
        order_id=str(uuid.uuid4()),
        direction="buy",
        pair=signal.pair,
        rule_id=signal.rule_id,
        limit_price=signal.price,
        quantity=qty,
        value=qty * signal.price,
        created_at=now,
    ))
    logger.info("Placed BUY  %s  qty=%.6f @ %.4f  spend=%.2f", signal.pair, qty, signal.price, spend)


def _place_sell(portfolio: Portfolio, signal: SellSignal, now: datetime) -> None:
    pos = next((p for p in portfolio.positions if p.pair == signal.pair), None)
    if pos is None:
        return
    portfolio.pending_orders = [
        o for o in portfolio.pending_orders
        if not (o.direction == "sell" and o.position_id == pos.position_id)
    ]
    portfolio.pending_orders.append(Order(
        order_id=str(uuid.uuid4()),
        direction="sell",
        pair=signal.pair,
        rule_id=signal.rule_id,
        limit_price=signal.price,
        quantity=pos.quantity,
        value=pos.quantity * signal.price,
        position_id=pos.position_id,
        created_at=now,
    ))
    logger.info("Placed SELL %s  qty=%.6f @ %.4f", signal.pair, pos.quantity, signal.price)


def _find_best_rule(state_dir: str, min_gain: float) -> str | None:
    path = Path(state_dir) / "rule_evaluation.json"
    if not path.exists():
        return None
    try:
        rules = json.loads(path.read_text(encoding="utf-8")).get("rules", [])
        eligible = [r for r in rules if r.get("recent_avg_gain_pct", 0.0) > min_gain]
        if not eligible:
            return None
        return max(eligible, key=lambda r: r["recent_avg_gain_pct"])["rule_id"]
    except Exception:
        logger.warning("Could not read rule_evaluation.json for portfolio", exc_info=True)
        return None


def _portfolio_dir(config: AppConfig) -> Path:
    return Path(config.data_dir) / "portfolio"


def _load(config: AppConfig) -> Portfolio:
    path = _portfolio_dir(config) / "portfolio.json"
    if path.exists():
        try:
            return Portfolio.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse portfolio.json; resetting to initial cash")
    return Portfolio(cash=config.portfolio_initial_capital)


def _save(portfolio: Portfolio, config: AppConfig) -> None:
    path = _portfolio_dir(config) / "portfolio.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")


def _append_transactions(transactions: list[Transaction], config: AppConfig) -> None:
    path = _portfolio_dir(config) / "transactions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse transactions.json; starting fresh")
    existing.extend(t.model_dump(mode="json") for t in transactions)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
