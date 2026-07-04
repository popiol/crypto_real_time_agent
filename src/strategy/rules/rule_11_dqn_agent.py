"""Rule 11 — Reinforcement learning: Deep Q-Network (DQN).

Action space: A = {0 = hold, 1 = buy, 2 = sell}

The network approximates Q(s, a) for all actions simultaneously:
    output shape: (3,)  →  [Q(s, hold), Q(s, buy), Q(s, sell)]

Each cycle:

  1. Label + train (Bellman update)
     For each pending experience (s, a, r=0, recorded_at) that is ≥ 24 h old:
       - If a = buy:  r = clip((P_24h − P_0) / P_0 − PROVISION, −MAX_R, +MAX_R)
       - If a = sell: r = clip((P_0 − P_24h) / P_0 − PROVISION, −MAX_R, +MAX_R)
       - If a = hold: r = 0
       - Fetch current state s′ from warm-tier data
       - Compute Q_target = Q(s) with only the taken action updated:
             Q_target[a] = r + γ · max_a′ Q(s′)
       - train_on_batch(s, Q_target)   ← MSE only on the taken action's head

  2. Record (ε-greedy)
     At most once per pair per hour, select an action:
       - With probability ε: random action (explore)
       - Otherwise: a = argmax Q(s)   (exploit)
     Save (pair, recorded_at, price_entry, state s, action a).
     ε decays from 1.0 → 0.1 over training steps.

  3. Infer
     Emit BuySignal when argmax Q(s) == 1, SellSignal when argmax Q(s) == 2.
     Inference suppressed until MIN_TRAINING_STEPS accumulated.
     Confidence = σ(Q(s, chosen) − Q(s, hold))  ∈ (0, 1).

State files:
    data/dqn_model/         TensorFlow SavedModel
    data/dqn_pending.ndjson pending experiences
    data/dqn_state.json     {"training_steps": N}
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.agent.models import BuySignal, PairData, SellSignal, WarmCandle

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = logging.getLogger(__name__)

RULE_ID = "dqn_buy_signal"

WINDOW_SIZE = 20  # warm candles per state
N_FEATURES = 2  # (close_norm, roc) per candle
N_ACTIONS = 3
GAMMA = 0.9  # discount factor
EPS_START = 1.0  # initial exploration rate
EPS_MIN = 0.1  # minimum exploration rate
EPS_DECAY = 0.995  # per-step multiplicative decay
PROVISION = 0.005  # round-trip trading cost deducted from buy reward
FORECAST_HORIZON = timedelta(hours=24)
MAX_R = 0.10  # clip rewards to [−10%, +10%]
MIN_TRAINING_STEPS = 20
RECORD_INTERVAL = timedelta(hours=1)
SAVE_EVERY = 10

_MODEL_PATH = Path("data") / "dqn_model"
_PENDING_PATH = Path("data") / "dqn_pending.ndjson"
_STATE_PATH = Path("data") / "dqn_state.json"

MarketData = dict[str, PairData]

# ── Module-level state ────────────────────────────────────────────────────────
_model: Any = None
_training_steps: int = 0
_last_recorded: dict[str, datetime] = {}


# ── Model ─────────────────────────────────────────────────────────────────────


def _build_model():
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_SIZE * N_FEATURES,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(N_ACTIONS),  # linear: [Q(hold), Q(buy)]
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def _load_model_and_state() -> None:
    global _model, _training_steps
    import tensorflow as tf

    if _MODEL_PATH.exists():
        try:
            _model = tf.keras.models.load_model(str(_MODEL_PATH))
            logger.info("DQN model loaded from %s", _MODEL_PATH)
        except Exception:
            logger.exception("Failed to load DQN model; starting fresh")
            _model = _build_model()
    else:
        logger.info("No DQN model at %s; initialising fresh network", _MODEL_PATH)
        _model = _build_model()

    if _STATE_PATH.exists():
        try:
            _training_steps = json.loads(_STATE_PATH.read_text())["training_steps"]
        except Exception:
            _training_steps = 0


def _save_model_and_state() -> None:
    try:
        _model.save(str(_MODEL_PATH))
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"training_steps": _training_steps}))
    except Exception:
        logger.exception("Failed to save DQN model")


# ── State vector ──────────────────────────────────────────────────────────────


def _state_vec(candles: list[WarmCandle]) -> list[float]:
    """Flatten WINDOW_SIZE candles into a 1-D state vector."""
    closes = [c.close for c in candles]
    mean_close = statistics.mean(closes) or 1.0
    flat: list[float] = []
    for i, close in enumerate(closes):
        prev = closes[i - 1]
        roc = (closes[i] - prev) / prev if i > 0 and prev != 0.0 else 0.0
        flat.extend([close / mean_close, roc])
    return flat


def _current_state(pair_data: PairData) -> list[float] | None:
    if len(pair_data.warm) < WINDOW_SIZE + 1:
        return None
    return _state_vec(pair_data.warm[-(WINDOW_SIZE + 1) :][1:])


# ── Exploration rate ──────────────────────────────────────────────────────────


def _epsilon() -> float:
    return max(EPS_MIN, EPS_START * (EPS_DECAY**_training_steps))


# ── Pending experiences ───────────────────────────────────────────────────────


def _load_pending() -> list[dict]:
    if not _PENDING_PATH.exists():
        return []
    records: list[dict] = []
    for line in _PENDING_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _save_pending(records: list[dict]) -> None:
    _PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in records)
    _PENDING_PATH.write_text(text + ("\n" if records else ""), encoding="utf-8")


# ── Bellman update ────────────────────────────────────────────────────────────


def _label_and_train(
    pending: list[dict],
    current_prices: dict[str, float],
    current_states: dict[str, list[float]],
) -> list[dict]:
    global _training_steps
    import numpy as np

    now = datetime.now(timezone.utc)
    remaining: list[dict] = []

    for rec in pending:
        recorded_at = datetime.fromisoformat(rec["recorded_at"])

        if now - recorded_at < FORECAST_HORIZON:
            remaining.append(rec)
            continue

        pair = rec["pair"]
        if pair not in current_prices or pair not in current_states:
            remaining.append(rec)  # retry next cycle
            continue

        action = rec["action"]

        if action == 1:
            raw_r = (current_prices[pair] - rec["price"]) / rec["price"] - PROVISION
            r = max(-MAX_R, min(MAX_R, raw_r))
        elif action == 2:
            raw_r = (rec["price"] - current_prices[pair]) / rec["price"] - PROVISION
            r = max(-MAX_R, min(MAX_R, raw_r))
        else:
            r = 0.0

        s = np.array([rec["state"]], dtype=np.float32)  # (1, 40)
        s_prime = np.array([current_states[pair]], dtype=np.float32)  # (1, 40)

        # Q(s) and Q(s') from current network
        q_s = _model(s, training=False).numpy()[0]  # shape (2,)
        q_s_prime = _model(s_prime, training=False).numpy()[0]  # shape (2,)

        # Bellman target: only update the head for the taken action
        q_target = q_s.copy()
        q_target[action] = r + GAMMA * float(q_s_prime.max())

        loss = float(_model.train_on_batch(s, np.array([q_target], dtype=np.float32)))
        _training_steps += 1

        if _training_steps % SAVE_EVERY == 0:
            logger.debug(
                "DQN step %d — loss %.6f  ε=%.3f",
                _training_steps,
                loss,
                _epsilon(),
            )
            _save_model_and_state()

    return remaining


# ── ε-greedy record ───────────────────────────────────────────────────────────


def _add_records(pending: list[dict], data: MarketData, now: datetime) -> list[dict]:
    import numpy as np

    for pair, pair_data in data.items():
        s = _current_state(pair_data)
        if s is None or not pair_data.hot:
            continue

        last = _last_recorded.get(pair)
        if last is not None and now - last < RECORD_INTERVAL:
            continue

        # ε-greedy action selection
        if random.random() < _epsilon():
            action = random.randint(0, N_ACTIONS - 1)
        else:
            X = np.array([s], dtype=np.float32)
            q = _model(X, training=False).numpy()[0]
            action = int(q.argmax())

        pending.append(
            {
                "pair": pair,
                "recorded_at": now.isoformat(),
                "price": pair_data.hot[-1].last_price,
                "state": s,
                "action": action,
            }
        )
        _last_recorded[pair] = now

    return pending


# ── Inference ─────────────────────────────────────────────────────────────────


def _infer(data: MarketData) -> list[BuySignal | SellSignal]:
    if _training_steps < MIN_TRAINING_STEPS:
        return []

    import numpy as np

    signals: list[BuySignal | SellSignal] = []
    for pair, pair_data in data.items():
        s = _current_state(pair_data)
        if s is None or not pair_data.hot:
            continue

        X = np.array([s], dtype=np.float32)
        q = _model(X, training=False).numpy()[0]  # [Q(hold), Q(buy), Q(sell)]
        best = int(q.argmax())

        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price
        confidence = 1.0 / (1.0 + math.exp(-(float(q[best]) - float(q[0]))))

        if best == 1:
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price, confidence=confidence))
        elif best == 2:
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price, confidence=confidence))

    return signals


# ── Entry point ───────────────────────────────────────────────────────────────


def dqn_buy_signal(data: MarketData) -> list[BuySignal | SellSignal]:
    if _model is None:
        _load_model_and_state()

    now = datetime.now(timezone.utc)

    current_prices = {
        pair: pd.hot[-1].last_price for pair, pd in data.items() if pd.hot
    }
    current_states = {
        pair: s for pair, pd in data.items() if (s := _current_state(pd)) is not None
    }

    pending = _load_pending()
    pending = _label_and_train(pending, current_prices, current_states)
    pending = _add_records(pending, data, now)
    _save_pending(pending)

    return _infer(data)
