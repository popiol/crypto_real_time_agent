"""SQLite connection management and schema."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_db(data_dir: str):
    path = Path(data_dir) / "assets" / "agent.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(con)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS hot_ticks (
            pair        TEXT NOT NULL,
            polled_at   TEXT NOT NULL,
            last_price  REAL NOT NULL,
            bid_price   REAL NOT NULL,
            bid_volume  REAL NOT NULL,
            ask_price   REAL NOT NULL,
            ask_volume  REAL NOT NULL,
            volume_24h  REAL NOT NULL DEFAULT 0.0,
            mid_price   REAL NOT NULL,
            spread_abs  REAL NOT NULL,
            spread_rel  REAL NOT NULL,
            order_book  TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_hot_pair_time ON hot_ticks(pair, polled_at)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS warm_candles (
            pair            TEXT NOT NULL,
            hour            TEXT NOT NULL,
            open_price      REAL NOT NULL,
            high            REAL NOT NULL,
            low             REAL NOT NULL,
            close           REAL NOT NULL,
            avg_spread_rel  REAL NOT NULL DEFAULT 0.0,
            volume          REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (pair, hour)
        )
    """)
    con.execute(
        "ALTER TABLE warm_candles ADD COLUMN volume REAL NOT NULL DEFAULT 0.0"
        if not _column_exists(con, "warm_candles", "volume") else "SELECT 1"
    )
    con.execute("""
        CREATE TABLE IF NOT EXISTS cold_months (
            pair              TEXT NOT NULL,
            month             TEXT NOT NULL,
            min_price         REAL NOT NULL,
            max_price         REAL NOT NULL,
            avg_price         REAL NOT NULL,
            avg_daily_spread  REAL NOT NULL,
            candle_count      INTEGER NOT NULL,
            last_candle_hour  TEXT NOT NULL,
            PRIMARY KEY (pair, month)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            signal_id        TEXT PRIMARY KEY,
            direction        TEXT NOT NULL,
            pair             TEXT NOT NULL,
            rule_id          TEXT NOT NULL,
            emitted_at       TEXT NOT NULL,
            price_at_signal  REAL NOT NULL,
            confidence       REAL,
            evaluated_at     TEXT,
            exit_price       REAL,
            exit_reason      TEXT,
            gain_pct         REAL,
            gain_24h_pct     REAL,
            max_gain_24h_pct REAL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_signals_pair ON signals(pair, direction)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_signals_rule ON signals(rule_id)")
    con.commit()


def _column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)
