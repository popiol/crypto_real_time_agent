"""Kraken REST API wrapper.

Terminology used throughout this module:

- **currency**: a single tradable asset identified by its clean symbol
  (e.g. ``"XBT"``, ``"ETH"``, ``"ADA"``).  Kraken internally prefixes
  crypto asset IDs with ``X`` and fiat asset IDs with ``Z`` (e.g. ``XXBT``,
  ``ZUSD``); this wrapper strips those one-letter prefixes whenever values
  are returned to callers.

- **pair**: a trading pair, always *<currency>/USD* in this wrapper
  (e.g. ``"XBT/USD"``).  Internally Kraken identifies pairs by an
  *altname* such as ``XBTUSD``; that value is used in API calls.

- **cash**: the USD balance held in the account.  All buys spend cash and
  all sells return cash.

.. note::
    Kraken calls Bitcoin ``XBT``, not ``BTC``.  This wrapper surfaces
    whichever symbol Kraken publishes in the pair's WebSocket name
    (``wsname``), so ``"XBT"`` is the correct currency symbol here.
"""

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests

_BASE_URL = "https://api.kraken.com/0"

# Kraken's internal asset ID for US Dollar.
_USD_KRAKEN_ID = "ZUSD"


def _to_float(value: object, default: float = 0.0) -> float:
    """Parse a Kraken numeric field (often a string) into a float.

    Returns *default* for missing or unparseable values.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairInfo:
    """Metadata for a single currency/USD trading pair."""

    altname: str  # Kraken altname used in order API calls, e.g. "XBTUSD"
    wsname: str  # WebSocket name, e.g. "XBT/USD"
    currency: str  # Clean base symbol, e.g. "XBT"
    kraken_base_id: str  # Kraken internal asset ID, e.g. "XXBT"
    pair_decimals: int  # Decimal places for prices
    lot_decimals: int  # Decimal places for order volume


@dataclass(frozen=True)
class Holding:
    """A single portfolio position."""

    currency: str  # Clean currency symbol, e.g. "XBT"
    volume: float  # Units held
    price_usd: float | None  # Last traded USD price (None if unavailable)

    @property
    def value_usd(self) -> float | None:
        """Current USD value of the position, or None if price is unavailable."""
        if self.price_usd is None:
            return None
        return self.volume * self.price_usd


@dataclass(frozen=True)
class TickerSnapshot:
    """Current market data for a single currency/USD pair."""

    currency: str  # Clean currency symbol, e.g. "XBT"
    price_usd: float  # Last traded price in USD
    volume_today: float  # 24-hour volume in base currency (resets daily)


@dataclass(frozen=True)
class OpenOrder:
    """A single open order on the exchange, with parsed and typed fields.

    Callers use these attributes directly instead of digging through Kraken's
    nested ``descr`` dict.
    """

    txid: str  # Kraken transaction ID
    type: str  # "buy" or "sell"
    ordertype: str  # "limit", "market", etc.
    currency: str | None  # Clean base symbol, e.g. "XBT"; None if not a known USD pair
    price: float  # Limit price in USD (0.0 for market orders)
    volume: float  # Ordered volume in base currency
    volume_executed: float  # Filled volume so far in base currency
    status: str  # "open", "pending", etc.

    @property
    def remaining_volume(self) -> float:
        """Unfilled portion of the order in base currency."""
        return max(0.0, self.volume - self.volume_executed)


# ── Exceptions ─────────────────────────────────────────────────────────────────


class KrakenError(Exception):
    """Raised when the Kraken API returns one or more error strings."""


# ── Client ─────────────────────────────────────────────────────────────────────


class KrakenClient:
    """Thin wrapper around the Kraken REST API.

    Authentication uses the standard API-Key / API-Sign scheme.  Obtain a
    key pair from your Kraken account settings and pass them to the
    constructor.

    All public-facing methods use *clean* currency symbols (e.g. ``"XBT"``
    instead of the internal ``"XXBT"``).  USD is the only cash currency;
    every buy spends USD and every sell returns USD.

    Pair metadata is fetched once from the public AssetPairs endpoint and
    cached for the lifetime of the client instance.
    """

    def __init__(self, api_key: str = "", api_secret: str = "") -> None:
        """
        Parameters
        ----------
        api_key:
            Kraken API key string.  May be omitted when only public endpoints
            are needed.
        api_secret:
            Base64-encoded Kraken API secret string (as shown in account
            settings).  May be omitted when only public endpoints are needed.
        """
        self._api_key = api_key
        self._api_secret = base64.b64decode(api_secret) if api_secret else b""
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        # currency symbol → PairInfo; populated lazily on first use.
        self._usd_pairs: dict[str, PairInfo] | None = None

    @classmethod
    def from_secrets_file(
        cls, path: str | Path = "kraken_secrets.json"
    ) -> "KrakenClient":
        """Construct a client from a JSON secrets file.

        The file must contain ``api_key`` and ``secret_key`` fields, e.g.::

            {
                "api_key": "...",
                "secret_key": "..."
            }

        Parameters
        ----------
        path:
            Path to the secrets file.  Defaults to ``kraken_secrets.json``
            in the current working directory.
        """
        secrets = json.loads(Path(path).read_text())
        return cls(api_key=secrets["api_key"], api_secret=secrets["secret_key"])

    # ── Internal: HTTP helpers ─────────────────────────────────────────────

    @staticmethod
    def _nonce() -> str:
        """Return a unique, monotonically increasing nonce string."""
        return str(int(time.time() * 1_000_000))

    def _sign(self, endpoint_path: str, data: dict) -> str:
        """Compute the HMAC-SHA512 API-Sign header value.

        Kraken signature scheme:
            HMAC-SHA512(
                uri_path + SHA256(nonce_str + url_encoded_post_body),
                base64_decoded_secret
            )
        """
        nonce = data["nonce"]
        postdata = urllib.parse.urlencode(data)
        hash_input = (nonce + postdata).encode()
        message = endpoint_path.encode() + hashlib.sha256(hash_input).digest()
        mac = hmac.new(self._api_secret, message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _public_get(self, path: str, params: dict | None = None) -> dict:
        url = f"{_BASE_URL}/public/{path}"
        resp = self._session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if body.get("error"):
            raise KrakenError(body["error"])
        return body["result"]

    def _private_post(self, path: str, data: dict | None = None) -> dict:
        if data is None:
            data = {}
        data["nonce"] = self._nonce()
        endpoint_path = f"/0/private/{path}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "API-Key": self._api_key,
            "API-Sign": self._sign(endpoint_path, data),
        }
        url = f"{_BASE_URL}/private/{path}"
        resp = self._session.post(
            url,
            data=urllib.parse.urlencode(data),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("error"):
            raise KrakenError(body["error"])
        return body["result"]

    # ── Internal: pair catalogue ───────────────────────────────────────────

    def _load_usd_pairs(self) -> dict[str, PairInfo]:
        """Fetch all *currency*/USD spot pairs and return a currency → PairInfo
        mapping.  Results are cached after the first call."""
        if self._usd_pairs is not None:
            return self._usd_pairs

        raw = self._public_get("AssetPairs")
        pairs: dict[str, PairInfo] = {}
        for _key, info in raw.items():
            wsname: str = info.get("wsname", "")
            if not wsname.endswith("/USD"):
                continue
            currency = wsname.split("/")[0]
            pairs[currency] = PairInfo(
                altname=info["altname"],
                wsname=wsname,
                currency=currency,
                kraken_base_id=info["base"],
                pair_decimals=info["pair_decimals"],
                lot_decimals=info["lot_decimals"],
            )

        self._usd_pairs = pairs
        return pairs

    def _pair_for(self, currency: str) -> PairInfo:
        """Return the PairInfo for *currency*/USD, raising ValueError if unknown."""
        pairs = self._load_usd_pairs()
        if currency not in pairs:
            raise ValueError(f"No USD pair found for currency '{currency}'.")
        return pairs[currency]

    @staticmethod
    def _strip_kraken_asset_prefix(kraken_id: str) -> str:
        """Strip the single-letter X (crypto) or Z (fiat) prefix that Kraken
        prepends to internal asset IDs.

        Examples::

            XXBT  →  XBT
            ZUSD  →  USD
            XETH  →  ETH
            ADA   →  ADA   (no prefix, returned as-is)

        The rule applied is: strip the leading character only if it is ``X``
        or ``Z`` *and* the second character is an uppercase letter, which
        distinguishes prefixed IDs (``XXBT``) from assets that happen to
        start with X or Z (none in practice, but guarded against).
        """
        if len(kraken_id) > 1 and kraken_id[0] in ("X", "Z") and kraken_id[1].isupper():
            return kraken_id[1:]
        return kraken_id

    # ── Account ────────────────────────────────────────────────────────────

    def get_cash(self) -> float:
        """Return the current USD (cash) balance."""
        result = self._private_post("Balance")
        return float(result.get(_USD_KRAKEN_ID, 0.0))

    def _fetch_tickers(
        self, altname_to_currency: dict[str, str]
    ) -> dict[str, TickerSnapshot]:
        """Internal: call the Ticker endpoint and return TickerSnapshot per currency.

        *altname_to_currency* maps Kraken pair altnames to clean currency symbols.
        Kraken may key its response by either the altname or the internal pair ID;
        both are handled via case-insensitive comparison with optional X/Z-prefix
        stripping.
        """
        if not altname_to_currency:
            return {}
        pair_param = ",".join(altname_to_currency)
        raw = self._public_get("Ticker", params={"pair": pair_param})
        snapshots: dict[str, TickerSnapshot] = {}
        for ticker_key, ticker_data in raw.items():
            price = float(ticker_data["c"][0])
            volume = float(ticker_data["v"][0])  # volume_today (daily reset)
            # Match response key → altname → currency symbol.
            currency = None
            for altname, sym in altname_to_currency.items():
                if ticker_key.upper() == altname.upper():
                    currency = sym
                    break
            if currency is None:
                # Fallback: strip X/Z prefix from ticker key and re-match.
                stripped = (
                    self._strip_kraken_asset_prefix(ticker_key[:4]) + ticker_key[4:]
                )
                for altname, sym in altname_to_currency.items():
                    if stripped.upper() == altname.upper():
                        currency = sym
                        break
            if currency is not None:
                snapshots[currency] = TickerSnapshot(
                    currency=currency, price_usd=price, volume_today=volume
                )
        return snapshots

    def _get_usd_prices(self, currencies: list[str]) -> dict[str, float]:
        """Return the last traded USD price for each currency in *currencies*."""
        pairs = self._load_usd_pairs()
        altname_to_currency = {pairs[c].altname: c for c in currencies if c in pairs}
        snapshots = self._fetch_tickers(altname_to_currency)
        return {sym: s.price_usd for sym, s in snapshots.items()}

    def get_tickers(self, currencies: list[str]) -> dict[str, TickerSnapshot]:
        """Return current price and volume for each currency in *currencies*.

        Only currencies with a known USD pair are included in the result.
        """
        pairs = self._load_usd_pairs()
        altname_to_currency = {pairs[c].altname: c for c in currencies if c in pairs}
        return self._fetch_tickers(altname_to_currency)

    def get_all_usd_tickers(self) -> dict[str, TickerSnapshot]:
        """Return current price and volume for every tradable currency/USD pair.

        This makes a single request to the Ticker endpoint with all known
        USD-pair altnames.  Use this when you need market data for all assets
        at once (e.g. for opportunity scoring across the full universe).
        """
        pairs = self._load_usd_pairs()
        altname_to_currency = {p.altname: p.currency for p in pairs.values()}
        return self._fetch_tickers(altname_to_currency)

    def get_raw_usd_tickers(self) -> dict:
        """Return the raw Kraken Ticker API response for all currency/USD pairs.

        Unlike :meth:`get_all_usd_tickers`, this returns the unprocessed dict
        straight from the API (altname → ticker fields ``a``, ``b``, ``c``,
        etc.), suitable for writing directly to raw snapshot JSON files.
        No API credentials are required.
        """
        pairs = self._load_usd_pairs()
        altnames = [p.altname for p in pairs.values()]
        return self._public_get("Ticker", params={"pair": ",".join(altnames)})

    def get_holdings(self, min_value_usd: float = 1.0) -> list[Holding]:
        """Return non-USD spot positions whose current USD value meets the threshold.

        Staking and Earn balance variants (asset IDs that contain a ``.``,
        such as ``XBT.F`` or ``ETH.S``) are excluded because they cannot
        be traded directly.

        Parameters
        ----------
        min_value_usd:
            Minimum position value in USD to include.  Dust positions below
            this threshold are silently dropped.  Defaults to ``1.0``.

        Returns
        -------
        list[Holding]
            One :class:`Holding` per position, sorted by currency symbol.
            Each entry contains ``currency``, ``volume``, ``price_usd``, and
            the derived ``value_usd`` property.
        """
        result = self._private_post("Balance")
        usd_pairs = self._load_usd_pairs()

        # Build a reverse map: Kraken internal asset ID → clean symbol.
        id_to_symbol: dict[str, str] = {
            p.kraken_base_id: p.currency for p in usd_pairs.values()
        }

        # First pass: collect all non-zero non-USD spot balances.
        raw_holdings: dict[str, float] = {}
        for kraken_id, balance_str in result.items():
            if kraken_id == _USD_KRAKEN_ID:
                continue
            if "." in kraken_id:  # skip staking / earn variants
                continue
            amount = float(balance_str)
            if amount == 0.0:
                continue
            symbol = id_to_symbol.get(
                kraken_id, self._strip_kraken_asset_prefix(kraken_id)
            )
            raw_holdings[symbol] = amount

        # Second pass: fetch current prices and drop dust positions.
        # Assets with no USD pair (price is None) are excluded because they
        # cannot be traded and their value cannot be determined.
        prices = self._get_usd_prices(list(raw_holdings))
        holdings: list[Holding] = []
        for symbol, amount in raw_holdings.items():
            price = prices.get(symbol)
            if price is not None and amount * price >= min_value_usd:
                holdings.append(
                    Holding(currency=symbol, volume=amount, price_usd=price)
                )

        return sorted(holdings, key=lambda h: h.currency)

    # ── Orders ─────────────────────────────────────────────────────────────

    def place_market_buy(self, currency: str, usd_amount: float) -> str:
        """Place a market buy order for *currency* spending *usd_amount* USD.

        The ``viqc`` order flag instructs Kraken to interpret the volume as
        quote-currency units (USD), so no manual price conversion is needed.

        Parameters
        ----------
        currency:
            Clean currency symbol, e.g. ``"XBT"``.
        usd_amount:
            Amount of USD to spend.

        Returns
        -------
        str
            Kraken transaction ID (``txid``) of the placed order.
        """
        pair = self._pair_for(currency)
        result = self._private_post(
            "AddOrder",
            {
                "ordertype": "market",
                "type": "buy",
                "volume": str(usd_amount),
                "pair": pair.altname,
                "oflags": "viqc",
            },
        )
        return result["txid"][0]

    def place_market_sell(self, currency: str, volume: float) -> str:
        """Place a market sell order, selling *volume* units of *currency* for USD.

        Parameters
        ----------
        currency:
            Clean currency symbol, e.g. ``"XBT"``.
        volume:
            Number of currency units to sell (in base-currency terms).

        Returns
        -------
        str
            Kraken transaction ID (``txid``) of the placed order.
        """
        pair = self._pair_for(currency)
        result = self._private_post(
            "AddOrder",
            {
                "ordertype": "market",
                "type": "sell",
                "volume": str(volume),
                "pair": pair.altname,
            },
        )
        return result["txid"][0]

    def place_limit_buy(
        self,
        currency: str,
        price: float,
        volume: float,
        expiry_seconds: int = 86400,
    ) -> str:
        """Place a limit buy order for *volume* units of *currency* at *price* USD.

        Parameters
        ----------
        currency:
            Clean currency symbol, e.g. ``"XBT"``.
        price:
            Limit price in USD (maximum price willing to pay).
        volume:
            Number of currency units to buy (base currency).
        expiry_seconds:
            Order lifetime in seconds (default 24 hours).

        Returns
        -------
        str
            Kraken transaction ID (``txid``) of the placed order.
        """
        pair = self._pair_for(currency)
        result = self._private_post(
            "AddOrder",
            {
                "ordertype": "limit",
                "type": "buy",
                "price": f"{price:.{pair.pair_decimals}f}",
                "volume": str(volume),
                "pair": pair.altname,
                "expiretm": f"+{expiry_seconds}",
            },
        )
        return result["txid"][0]

    def place_limit_sell(
        self,
        currency: str,
        price: float,
        volume: float,
        expiry_seconds: int = 86400,
    ) -> str:
        """Place a limit sell order for *volume* units of *currency* at *price* USD.

        Parameters
        ----------
        currency:
            Clean currency symbol, e.g. ``"XBT"``.
        price:
            Limit price in USD (minimum price willing to accept).
        volume:
            Number of currency units to sell (base currency).
        expiry_seconds:
            Order lifetime in seconds (default 24 hours).

        Returns
        -------
        str
            Kraken transaction ID (``txid``) of the placed order.
        """
        pair = self._pair_for(currency)
        result = self._private_post(
            "AddOrder",
            {
                "ordertype": "limit",
                "type": "sell",
                "price": f"{price:.{pair.pair_decimals}f}",
                "volume": str(volume),
                "pair": pair.altname,
                "expiretm": f"+{expiry_seconds}",
            },
        )
        return result["txid"][0]

    def amend_order(
        self,
        txid: str,
        currency: str,
        *,
        price: float | None = None,
        volume: float | None = None,
    ) -> str:
        """Amend an open order's limit price and/or volume in place.

        Unlike cancel-and-replace this keeps the same ``txid``.  Pass only the
        fields you want to change.

        Parameters
        ----------
        txid:
            Transaction ID of the order to amend.
        currency:
            Clean currency symbol, used to format *price*/*volume* to the pair's
            precision.
        price:
            New limit price in USD, or None to leave it unchanged.
        volume:
            New order volume in base currency, or None to leave it unchanged.

        Returns
        -------
        str
            Kraken amend ID for the operation.
        """
        pair = self._pair_for(currency)
        data: dict = {"txid": txid}
        if price is not None:
            data["limit_price"] = f"{price:.{pair.pair_decimals}f}"
        if volume is not None:
            data["order_qty"] = f"{volume:.{pair.lot_decimals}f}"
        result = self._private_post("AmendOrder", data)
        return result.get("amend_id", "")

    def get_order_status(self, txid: str) -> dict:
        """Return the raw order info dict for *txid*.

        The returned dict includes at minimum:

        - ``status``: ``"pending"``, ``"open"``, ``"closed"``,
          ``"canceled"``, or ``"expired"``
        - ``vol``: ordered volume (base currency)
        - ``vol_exec``: executed volume (base currency)
        - ``cost``: total cost in USD
        - ``fee``: total fee in USD
        - ``price``: average execution price in USD

        Parameters
        ----------
        txid:
            Kraken transaction ID returned by :meth:`place_market_buy` or
            :meth:`place_market_sell`.
        """
        result = self._private_post("QueryOrders", {"txid": txid})
        if txid not in result:
            raise KrakenError(f"txid '{txid}' not found in QueryOrders response")
        return result[txid]

    def get_open_orders(self) -> list[OpenOrder]:
        """Return all currently open orders as typed :class:`OpenOrder` objects.

        Kraken's nested ``descr`` structure is parsed here so callers never see
        it: the trading pair is resolved to a clean currency symbol and every
        numeric field is converted to a float.
        """
        result = self._private_post("OpenOrders", {})
        orders: dict = result.get("open", {})

        # Resolve whichever pair form Kraken returns ("XBTUSD" or "XBT/USD")
        # back to a clean currency symbol.
        pairs = self._load_usd_pairs()
        pair_to_currency: dict[str, str] = {}
        for currency, info in pairs.items():
            pair_to_currency[info.altname.upper()] = currency
            pair_to_currency[info.wsname.replace("/", "").upper()] = currency

        parsed: list[OpenOrder] = []
        for txid, info in orders.items():
            descr = info.get("descr", {})
            pair_key = (
                str(descr.get("pair", info.get("pair", ""))).replace("/", "").upper()
            )
            parsed.append(
                OpenOrder(
                    txid=txid,
                    type=str(descr.get("type", "")),
                    ordertype=str(descr.get("ordertype", "")),
                    currency=pair_to_currency.get(pair_key),
                    price=_to_float(descr.get("price", info.get("price", 0.0))),
                    volume=_to_float(info.get("vol", 0.0)),
                    volume_executed=_to_float(info.get("vol_exec", 0.0)),
                    status=str(info.get("status", "")),
                )
            )
        return parsed

    # ── Precision ──────────────────────────────────────────────────────────

    def get_volume_precision(self, currency: str) -> int:
        """Return the number of decimal places for order volume (base currency).

        Use this to round order volumes before placing orders to avoid
        Kraken rejecting them for excess precision.
        """
        return self._pair_for(currency).lot_decimals

    def get_price_precision(self, currency: str) -> int:
        """Return the number of decimal places for prices.

        Use this to round price values before placing limit orders.
        """
        return self._pair_for(currency).pair_decimals


# ── CLI ────────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.kraken",
        description="Manual test CLI for the Kraken API wrapper.",
    )
    parser.add_argument(
        "--secrets",
        default="kraken_secrets.json",
        metavar="FILE",
        help="Path to the secrets JSON file (default: kraken_secrets.json).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cash", help="Print current USD cash balance.")

    sub.add_parser("holdings", help="Print all non-USD spot holdings.")

    p_buy = sub.add_parser("buy", help="Place a market buy order (spend USD).")
    p_buy.add_argument("currency", help="Currency symbol, e.g. XBT.")
    p_buy.add_argument("usd_amount", type=float, help="USD amount to spend.")
    p_buy.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the order details without submitting.",
    )

    p_sell = sub.add_parser("sell", help="Place a market sell order (receive USD).")
    p_sell.add_argument("currency", help="Currency symbol, e.g. XBT.")
    p_sell.add_argument("volume", type=float, help="Volume to sell (base currency).")
    p_sell.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the order details without submitting.",
    )

    p_order = sub.add_parser("order-status", help="Query an order by txid.")
    p_order.add_argument("txid", help="Kraken transaction ID.")

    p_prec = sub.add_parser(
        "precision", help="Show volume and price precision for a currency."
    )
    p_prec.add_argument("currency", help="Currency symbol, e.g. XBT.")

    args = parser.parse_args()
    client = KrakenClient.from_secrets_file(args.secrets)

    if args.command == "cash":
        print(f"USD cash: {client.get_cash():.2f}")

    elif args.command == "holdings":
        holdings = client.get_holdings()
        if not holdings:
            print("No holdings.")
        else:
            width = max(len(h.currency) for h in holdings)
            print(
                f"  {'Currency':<{width}}  {'Volume':<20}  {'Price (USD)':>14}  {'Value (USD)':>12}"
            )
            print(f"  {'-' * width}  {'-' * 20}  {'-' * 14}  {'-' * 12}")
            for h in holdings:
                price_str = (
                    f"{h.price_usd:>14.4f}"
                    if h.price_usd is not None
                    else f"{'N/A':>14}"
                )
                value_str = (
                    f"{h.value_usd:>12.2f}"
                    if h.value_usd is not None
                    else f"{'N/A':>12}"
                )
                print(
                    f"  {h.currency:<{width}}  {h.volume:<20}  {price_str}  {value_str}"
                )

    elif args.command == "buy":
        pair = client._pair_for(args.currency)
        print(
            f"{'[DRY RUN] ' if args.dry_run else ''}"
            f"Market BUY  {args.currency}/USD  spend ${args.usd_amount:.2f}"
            f"  (pair altname: {pair.altname})"
        )
        if not args.dry_run:
            txid = client.place_market_buy(args.currency, args.usd_amount)
            print(f"Order placed. txid: {txid}")

    elif args.command == "sell":
        pair = client._pair_for(args.currency)
        print(
            f"{'[DRY RUN] ' if args.dry_run else ''}"
            f"Market SELL {args.volume} {args.currency}  for USD"
            f"  (pair altname: {pair.altname})"
        )
        if not args.dry_run:
            txid = client.place_market_sell(args.currency, args.volume)
            print(f"Order placed. txid: {txid}")

    elif args.command == "order-status":
        info = client.get_order_status(args.txid)
        status = info.get("status", "unknown")
        descr = info.get("descr", {})
        print(f"txid   : {args.txid}")
        print(f"status : {status}")
        print(f"order  : {descr.get('order', '-')}")
        print(f"vol    : {info.get('vol', '-')}")
        print(f"vol_exec: {info.get('vol_exec', '-')}")
        print(f"cost   : {info.get('cost', '-')}")
        print(f"fee    : {info.get('fee', '-')}")
        print(f"price  : {info.get('price', '-')}")

    elif args.command == "precision":
        vol = client.get_volume_precision(args.currency)
        price = client.get_price_precision(args.currency)
        print(f"{args.currency}  volume precision: {vol} decimals")
        print(f"{args.currency}  price  precision: {price} decimals")


if __name__ == "__main__":
    _cli()
