import os
import re
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import requests
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
LINE_TOKEN    = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_IDS = [uid.strip() for uid in os.environ.get("LINE_USER_ID", "").split(",") if uid.strip()]
TD_API_KEY    = os.environ.get("TWELVEDATA_API_KEY", "")
AV_API_KEY    = os.environ.get("ALPHAVANTAGE_API_KEY", "")
TW_TZ = pytz.timezone("Asia/Taipei")

PRICE_TTL = int(os.environ.get("PRICE_CACHE_SECONDS", "3600"))
HIST_TTL  = int(os.environ.get("HIST_CACHE_SECONDS",  "43200"))

TD_BASE = "https://api.twelvedata.com"
AV_BASE = "https://www.alphavantage.co/query"

TW_SYMBOLS = {
    "0050":   "0050.TW",
    "00631L": "00631L.TW",
    "00662":  "00662.TW",
    "00865B": "00865B.TW",
}
US_SYMBOLS = {
    "QQQ": ("QQQ", "NASDAQ"),
    "QLD": ("QLD", ""),
}
# Yahoo Finance-only US stocks (no Twelve Data quota needed)
YF_US_SYMBOLS = {
    "GOOG": "GOOG",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "MSTR": "MSTR",
}
SYMBOL_NAMES = {
    "0050":   "元大台灣50",
    "QQQ":    "Invesco QQQ ETF",
    "00631L": "元大台灣50正2",
    "QLD":    "ProShares Ultra QQQ",
    "00662":  "富邦NASDAQ",
    "00865B": "國泰US短期公債",
    "GOOG":   "Alphabet (Google)",
    "NVDA":   "NVIDIA",
    "TSLA":   "Tesla",
    "MSTR":   "MicroStrategy",
}

# ─── Cache ────────────────────────────────────────────────────────────────────
_cache = {
    "stocks":      {},
    "histories":   {},
    "hist_ts":     0.0,
    "price_ts":    0.0,
    "vix_ts":      0.0,
    "chips_ts":      0.0,
    "usdtwd":        None,
    "usdtwd_ts":     0.0,
    "fear_greed":       None,
    "fear_greed_ts":    0.0,
    "global_quotes":    {},
    "global_quotes_ts": 0.0,
    "sector":           [],
    "sector_ts":        0.0,
    "refreshing":  False,
    "vix":         None,
    "foreign_net": None,
    "tw_chips":    None,
}
VIX_TTL    = 1800   # 30 min — VIX updates intraday but not per-minute
CHIPS_TTL  = 7200   # 2 hours — TWSE chips published once daily ~14:30
USDTWD_TTL = 1800   # 30 min
FG_TTL     = 3600   # 1 hour — CNN Fear & Greed updates a few times daily
GLOBAL_TTL = 1800   # 30 min
SECTOR_TTL = 1800   # 30 min
GLOBAL_SYMBOLS = {
    "^TWII": "台灣加權",
    "^N225": "日經225",
    "^GSPC": "S&P 500",
    "^IXIC": "那斯達克",
    "SOXX":  "費半ETF",
}
SECTOR_SYMBOLS = {
    "0050.TW": {"name": "台灣大盤",   "group": "TW"},
    "0052.TW": {"name": "電子(0052)", "group": "TW"},
    "0055.TW": {"name": "金融(0055)", "group": "TW"},
    "0056.TW": {"name": "高股息(0056)","group": "TW"},
    "^GSPC":   {"name": "S&P 500",   "group": "US"},
    "SOXX":    {"name": "費半(SOXX)", "group": "US"},
    "XLK":     {"name": "科技(XLK)", "group": "US"},
    "XLF":     {"name": "金融(XLF)", "group": "US"},
    "XLE":     {"name": "能源(XLE)", "group": "US"},
}
_refresh_lock = threading.Lock()
_line_users: dict = {}

ALERT_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "alert_history.json")

def _load_alert_history() -> list:
    try:
        with open(ALERT_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_alert_history():
    try:
        with open(ALERT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(alert_history, f, ensure_ascii=False)
    except Exception as exc:
        logger.error("Alert history save error: %s", exc)

alert_history: list = _load_alert_history()

# Search cache: {ticker: {"entry": dict, "hist": DataFrame, "hist_ts": float, "price_ts": float}}
_search_cache: dict = {}
SEARCH_HIST_TTL  = 43200   # 12 hours — history data rarely changes intraday
SEARCH_PRICE_TTL = 300     # 5 minutes — keep quote reasonably fresh

def _start_search_cache_cleanup():
    def _run():
        while True:
            time.sleep(300)
            now = time.time()
            expired = [k for k, v in list(_search_cache.items())
                       if now - v.get("hist_ts", 0) > SEARCH_HIST_TTL]
            for k in expired:
                _search_cache.pop(k, None)
            if expired:
                logger.info("Search cache cleanup: evicted %d stale entries", len(expired))
    threading.Thread(target=_run, daemon=True).start()

_start_search_cache_cleanup()


# ─── Technical Indicators ─────────────────────────────────────────────────────
def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # Pure uptrend (avg_loss = 0, avg_gain > 0) → no losses → RSI should be 100, not NaN
    pure_up = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~pure_up, 100.0)
    return rsi


def _calc_macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist


def _calc_kd(df: pd.DataFrame, period: int = 9):
    lo  = df["Low"].rolling(period).min()
    hi  = df["High"].rolling(period).max()
    rsv = (df["Close"] - lo) / (hi - lo).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k   = rsv.ewm(com=2, adjust=False).mean()   # alpha=1/3 smoothing
    d   = k.ewm(com=2, adjust=False).mean()
    return k, d


# ─── Divergence Detection ─────────────────────────────────────────────────────
def _find_peaks(series: pd.Series, order: int = 7) -> list[int]:
    """Local maxima indices (0-based) with at least `order` bars on each side."""
    arr = series.to_numpy()
    n   = len(arr)
    if n < order * 2 + 1:
        return []
    idx     = np.arange(order, n - order)
    windows = np.lib.stride_tricks.as_strided(
        arr, shape=(len(idx), 2 * order + 1), strides=(arr.strides[0], arr.strides[0])
    )
    mask = (arr[idx] == windows.max(axis=1)) & (arr[idx] > arr[idx - 1]) & (arr[idx] > arr[idx + 1])
    return idx[mask].tolist()


def _find_troughs(series: pd.Series, order: int = 7) -> list[int]:
    """Local minima indices (0-based)."""
    arr = series.to_numpy()
    n   = len(arr)
    if n < order * 2 + 1:
        return []
    idx     = np.arange(order, n - order)
    windows = np.lib.stride_tricks.as_strided(
        arr, shape=(len(idx), 2 * order + 1), strides=(arr.strides[0], arr.strides[0])
    )
    mask = (arr[idx] == windows.min(axis=1)) & (arr[idx] < arr[idx - 1]) & (arr[idx] < arr[idx + 1])
    return idx[mask].tolist()


def _calc_divergence(close: pd.Series, rsi: pd.Series,
                     lookback: int = 60, order: int = 7) -> dict:
    """
    Bearish (頂部背離): price higher-high but RSI lower-high → fake breakout risk.
    Bullish (底部背離): price lower-low but RSI higher-low  → exhausted selling.
    Returns dict with keys: bearish, bullish, desc.
    """
    empty = {"bearish": False, "bullish": False, "desc": "正常"}

    cl = close.tail(lookback).reset_index(drop=True)
    rs = rsi.tail(lookback).reset_index(drop=True).ffill().fillna(50)

    if len(cl) < order * 3:
        return empty

    price_peaks   = _find_peaks(cl,   order)
    price_troughs = _find_troughs(cl, order)

    bearish = False
    bullish = False

    # Bearish: price makes higher-high, RSI makes lower-high (≥2 pt gap)
    if len(price_peaks) >= 2:
        p1, p2 = price_peaks[-2], price_peaks[-1]
        if cl.iloc[p2] > cl.iloc[p1] and rs.iloc[p2] < rs.iloc[p1] - 2:
            bearish = True

    # Bullish: price makes lower-low, RSI makes higher-low (≥2 pt gap)
    if len(price_troughs) >= 2:
        t1, t2 = price_troughs[-2], price_troughs[-1]
        if cl.iloc[t2] < cl.iloc[t1] and rs.iloc[t2] > rs.iloc[t1] + 2:
            bullish = True

    if bearish:
        desc = "頂部背離：價格創高但RSI未跟上，假突破風險"
    elif bullish:
        desc = "底部背離：價格破低但RSI止穩，底部訊號浮現"
    else:
        desc = "正常"

    return {"bearish": bearish, "bullish": bullish, "desc": desc}


# ─── Taiwan stock helpers ─────────────────────────────────────────────────────
def _fetch_tw_quotes() -> dict:
    parts = []
    for s in TW_SYMBOLS:
        parts.append(f"tse_{s}.tw")
        parts.append(f"otc_{s}.tw")
    ex_ch = "|".join(parts)
    try:
        resp = requests.get(
            "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
            params={"ex_ch": ex_ch, "json": "1", "delay": "0"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        result = {}
        for item in resp.json().get("msgArray", []):
            code  = item.get("c", "").upper()
            z     = item.get("z", "-")
            y     = item.get("y", "-")
            price = float(z) if z not in ("-", "") else None
            prev  = float(y) if y not in ("-", "") else None
            if prev is None:
                continue
            if price is None:
                # Market closed / not yet traded — skip so Yahoo Finance fallback is used
                continue
            pz_raw  = item.get("pz", "0")
            nav_est = float(pz_raw) if pz_raw and pz_raw not in ("0", "-", "") else None
            premium_pct = round((price - nav_est) / nav_est * 100, 2) if nav_est else None
            result[code] = {
                "price":          price,
                "prev_close":     prev,
                "daily_change":   (price - prev) / prev * 100 if prev else 0.0,
                "is_market_open": True,
                "nav_est":        nav_est,
                "premium_pct":    premium_pct,
            }
        logger.info("TWSE quotes fetched: %s", list(result.keys()))
        return result
    except Exception as exc:
        logger.error("TWSE quote error: %s", exc)
        return {}


def _fetch_twse_history(stock_no: str, months: int = 14) -> pd.DataFrame | None:
    from datetime import date
    rows = []
    today = date.today()
    for i in range(months):
        m = today.month - i
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        try:
            resp = requests.get(
                "https://www.twse.com.tw/exchangeReport/STOCK_DAY",
                params={"response": "json", "date": f"{y}{m:02d}01", "stockNo": stock_no},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            data = resp.json()
            if data.get("stat") != "OK":
                continue
            for row in data.get("data", []):
                close_str = row[6].replace(",", "")
                if close_str in ("--", ""):
                    continue
                high_str = row[4].replace(",", "")
                low_str  = row[5].replace(",", "")
                parts_d  = row[0].split("/")
                close_f  = float(close_str)
                rows.append({
                    "date":  f"{int(parts_d[0]) + 1911}-{parts_d[1]}-{parts_d[2]}",
                    "Close": close_f,
                    "High":  float(high_str) if high_str not in ("--", "") else close_f,
                    "Low":   float(low_str)  if low_str  not in ("--", "") else close_f,
                })
            time.sleep(0.3)
        except Exception as exc:
            logger.warning("TWSE history %s %s: %s", stock_no, f"{y}{m:02d}", exc)

    if not rows:
        logger.error("TWSE no history for %s", stock_no)
        return None
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["date"])
    df = df.sort_index()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    logger.info("TWSE history %s: %d rows", stock_no, len(df))
    return df


# ─── Yahoo Finance helper (with retry) ───────────────────────────────────────
def _yahoo_get(url: str, params: dict) -> dict | None:
    """GET a Yahoo Finance chart URL with up to 3 retries and exponential backoff."""
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params,
                                headers={"User-Agent": "Mozilla/5.0"},
                                timeout=20)
            data = resp.json()
            if data.get("chart", {}).get("error"):
                logger.warning("Yahoo chart error (attempt %d): %s", attempt + 1,
                               data["chart"]["error"])
            else:
                return data
        except Exception as exc:
            logger.warning("Yahoo request error (attempt %d): %s", attempt + 1, exc)
        if attempt < 2:
            time.sleep(2 ** attempt)   # 1s, 2s
    logger.error("Yahoo request failed after 3 attempts: %s", url)
    return None


# ─── US stock helpers (Twelve Data) ──────────────────────────────────────────
_td_last_req = 0.0
TD_MIN_INTERVAL = 8.0   # free tier: 8 req/min


def _td_throttle():
    """Sleep just enough to respect Twelve Data free-tier rate limit."""
    global _td_last_req
    elapsed = time.time() - _td_last_req
    if elapsed < TD_MIN_INTERVAL:
        time.sleep(TD_MIN_INTERVAL - elapsed)
    _td_last_req = time.time()


def _td_get(endpoint: str, params: dict) -> dict | None:
    if not TD_API_KEY:
        logger.error("TWELVEDATA_API_KEY not set")
        return None
    try:
        params["apikey"] = TD_API_KEY
        resp = requests.get(f"{TD_BASE}/{endpoint}", params=params, timeout=30)
        data = resp.json()
        if data.get("status") == "error" or "code" in data:
            logger.error("TD error [%s]: %s", endpoint, data.get("message", data))
            return None
        return data
    except Exception as exc:
        logger.error("TD request error [%s]: %s", endpoint, exc)
        return None


def _fetch_td_history(sym: str, exchange: str) -> pd.DataFrame | None:
    params = {"symbol": sym, "interval": "1day", "outputsize": "500"}
    if exchange:
        params["exchange"] = exchange
    data = _td_get("time_series", params)
    if not data:
        return None
    values = data.get("values", [])
    if not values:
        return None
    df = pd.DataFrame(values)
    df.index = pd.to_datetime(df["datetime"])
    df = df.sort_index()
    df["Close"] = df["close"].astype(float)
    df["High"]  = df["high"].astype(float)
    df["Low"]   = df["low"].astype(float)
    df["MA200"] = df["Close"].rolling(200).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    logger.info("TD history fetched %s (%d rows)", sym, len(df))
    return df


def _fetch_yahoo_history(ticker: str) -> pd.DataFrame | None:
    try:
        data   = _yahoo_get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                            {"interval": "1d", "range": "2y"})
        result = (data or {}).get("chart", {}).get("result", [])
        if not result:
            logger.warning("Yahoo history empty for %s", ticker)
            return None
        r          = result[0]
        timestamps = r.get("timestamp", [])
        quote      = r.get("indicators", {}).get("quote", [{}])[0]
        closes     = quote.get("close", [])
        highs      = quote.get("high",  [])
        lows       = quote.get("low",   [])
        rows = []
        for ts, c, h, lo in zip(timestamps, closes, highs, lows):
            if c is None:
                continue
            rows.append({
                "date":  pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d"),
                "Close": float(c),
                "High":  float(h)  if h  is not None else float(c),
                "Low":   float(lo) if lo is not None else float(c),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"])
        df = df.sort_index()
        df["MA200"] = df["Close"].rolling(200).mean()
        df["MA50"]  = df["Close"].rolling(50).mean()
        logger.info("Yahoo history %s: %d rows", ticker, len(df))
        return df
    except Exception as exc:
        logger.error("Yahoo history error %s: %s", ticker, exc)
        return None


def _fetch_yahoo_quote(ticker: str) -> dict | None:
    try:
        data   = _yahoo_get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                            {"interval": "1d", "range": "5d"})
        result = (data or {}).get("chart", {}).get("result", [])
        if not result:
            return None
        meta   = result[0].get("meta", {})
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        closes = [c for c in closes if c is not None]
        price  = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
        if not price:
            return None
        # Yahoo often omits previousClose for TW stocks — fall back to second-to-last close
        prev = meta.get("previousClose")
        if prev is None and len(closes) >= 2:
            prev = closes[-2]
        if prev is None:
            prev = price
        return {
            "price":          float(price),
            "prev_close":     float(prev),
            "daily_change":   (float(price) - float(prev)) / float(prev) * 100 if prev else 0.0,
            "is_market_open": meta.get("marketState") == "REGULAR",
            "short_name":     meta.get("shortName") or meta.get("longName", ""),
        }
    except Exception as exc:
        logger.error("Yahoo quote error %s: %s", ticker, exc)
        return None


def _fetch_td_quote(sym: str, exchange: str) -> dict | None:
    params = {"symbol": sym}
    if exchange:
        params["exchange"] = exchange
    data = _td_get("quote", params)
    if not data:
        return None
    try:
        return {
            "price":          float(data["close"]),
            "prev_close":     float(data["previous_close"]),
            "daily_change":   float(data["percent_change"]),
            "is_market_open": data.get("is_market_open", False),
        }
    except (KeyError, ValueError) as exc:
        logger.error("TD quote parse %s: %s", sym, exc)
        return None


# ─── VIX & 外資買賣超 ──────────────────────────────────────────────────────────
def _fetch_vix() -> dict | None:
    try:
        data   = _yahoo_get("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
                            {"interval": "1d", "range": "10d"})
        result = (data or {}).get("chart", {}).get("result", [])
        if not result:
            return None
        meta  = result[0].get("meta", {})
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        prev  = meta.get("previousClose") or price
        if not price:
            return None
        return {
            "price":        round(float(price), 2),
            "prev_close":   round(float(prev),  2),
            "daily_change": round((float(price) - float(prev)) / float(prev) * 100, 2) if prev else 0.0,
        }
    except Exception as exc:
        logger.error("VIX fetch error: %s", exc)
        return None


def _fetch_foreign_net() -> dict | None:
    """Fetch 三大法人 daily net buy/sell from TWSE (unit: NT dollars)."""
    try:
        resp = requests.get(
            "https://www.twse.com.tw/fund/BFI82U",
            params={"response": "json", "type": "day"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        data = resp.json()
        if data.get("stat") != "OK":
            logger.warning("TWSE BFI82U stat: %s", data.get("stat"))
            return None
        result = {"date": data.get("date", "")}
        for row in data.get("data", []):
            name    = row[0].strip()
            net_str = row[3].replace(",", "").replace("+", "").strip()
            try:
                net = int(net_str)
            except ValueError:
                net = None  # "-" means data unavailable, not zero net
            if "外資及陸資" in name:
                result["foreign"] = net
            elif "投信" in name:
                result["trust"] = net
            elif name == "自營商":   # subtotal row, not 自行買賣 or 避險
                result["dealer"] = net
        logger.info("Foreign net fetched: %s", result)
        return result
    except Exception as exc:
        logger.error("Foreign net fetch error: %s", exc)
        return None


# ─── USD/TWD 匯率 ─────────────────────────────────────────────────────────────
def _fetch_usdtwd() -> dict | None:
    try:
        data   = _yahoo_get("https://query1.finance.yahoo.com/v8/finance/chart/USDTWD=X",
                            {"interval": "1d", "range": "5d"})
        result = (data or {}).get("chart", {}).get("result", [])
        if not result:
            return None
        meta  = result[0].get("meta", {})
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        prev  = meta.get("previousClose") or price
        if not price:
            return None
        return {
            "rate":   round(float(price), 3),
            "prev":   round(float(prev),  3),
            "change": round((float(price) - float(prev)) / float(prev) * 100, 2) if prev else 0.0,
        }
    except Exception as exc:
        logger.error("USDTWD fetch error: %s", exc)
        return None


# ─── CNN 恐貪指數 ──────────────────────────────────────────────────────────────
def _fetch_fear_greed() -> dict | None:
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
            },
            timeout=10,
        )
        data  = resp.json()
        fg    = data.get("fear_and_greed", {})
        score = fg.get("score")
        if score is None:
            return None
        return {
            "score":     round(float(score), 1),
            "rating":    fg.get("rating", ""),
            "prev_week": round(float(fg["previous_1_week"]), 1) if fg.get("previous_1_week") else None,
        }
    except Exception as exc:
        logger.error("Fear & Greed fetch error: %s", exc)
        return None


# ─── 配息資料 ─────────────────────────────────────────────────────────────────
def _fetch_yahoo_dividends(ticker: str) -> list[dict]:
    try:
        data = _yahoo_get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            {"interval": "1d", "range": "10y", "events": "dividends"},
        )
        result = (data or {}).get("chart", {}).get("result", [])
        if not result:
            return []
        raw = result[0].get("events", {}).get("dividends", {})
        divs = [
            {"date": pd.Timestamp(int(ts), unit="s").strftime("%Y-%m-%d"),
             "amount": float(v.get("amount", 0))}
            for ts, v in raw.items() if v.get("amount", 0) > 0
        ]
        divs.sort(key=lambda x: x["date"])
        logger.info("Dividends %s: %d events", ticker, len(divs))
        return divs
    except Exception as exc:
        logger.error("Dividend fetch error %s: %s", ticker, exc)
        return []


def _run_dividend_sim(hist: pd.DataFrame, dividends: list[dict],
                      capital: float, start_str: str) -> dict:
    hist_c = hist.dropna(subset=["Close"])
    if re.match(r"^\d{4}-\d{2}$", start_str):
        start_dt = pd.Timestamp(f"{start_str}-01")
    else:
        td = date.today()
        start_dt = pd.Timestamp(td.year - 5, 1, 1)

    hist_f = hist_c[hist_c.index >= start_dt]
    if hist_f.empty:
        return {"error": "指定期間內無資料，請調整起始月份"}

    start_price  = float(hist_f["Close"].iloc[0])
    drip_shares  = capital / start_price
    nodiv_shares = capital / start_price
    nodiv_cash   = 0.0
    drip_total_div = nodiv_total_div = 0.0
    div_events: list = []

    drip_chart: list  = []
    nodiv_chart: list = []

    price_map = {idx.strftime("%Y-%m-%d"): float(row["Close"])
                 for idx, row in hist_f.iterrows()}

    for d_str, price in sorted(price_map.items()):
        for div in dividends:
            if div["date"] != d_str or div["amount"] <= 0:
                continue
            amt = div["amount"]
            drip_div   = drip_shares  * amt
            nodiv_div  = nodiv_shares * amt
            drip_shares += drip_div / price
            drip_total_div  += drip_div
            nodiv_cash      += nodiv_div
            nodiv_total_div += nodiv_div
            div_events.append({
                "date":             d_str,
                "amount_per_share": round(amt, 4),
                "drip_total":       round(drip_div, 2),
                "shares_bought":    round(drip_div / price, 4),
                "price":            round(price, 2),
            })
        drip_chart.append( {"d": d_str, "v": round(drip_shares  * price)})
        nodiv_chart.append({"d": d_str, "v": round(nodiv_shares * price + nodiv_cash)})

    final_price = float(hist_f["Close"].iloc[-1])
    days = max((hist_f.index[-1] - hist_f.index[0]).days, 1)
    drip_final  = drip_shares  * final_price
    nodiv_final = nodiv_shares * final_price + nodiv_cash
    drip_ret    = (drip_final  - capital) / capital * 100
    nodiv_ret   = (nodiv_final - capital) / capital * 100

    return {
        "capital":       capital,
        "start_date":    hist_f.index[0].strftime("%Y-%m-%d"),
        "end_date":      hist_f.index[-1].strftime("%Y-%m-%d"),
        "days":          days,
        "start_price":   round(start_price, 2),
        "final_price":   round(final_price, 2),
        "drip_final":    round(drip_final),
        "drip_shares":   round(drip_shares, 4),
        "drip_reinvested": round(drip_total_div),
        "drip_return":   round(drip_ret,  2),
        "drip_cagr":     round(((drip_final  / capital) ** (365.0 / days) - 1) * 100, 2),
        "nodiv_final":   round(nodiv_final),
        "nodiv_cash":    round(nodiv_cash),
        "nodiv_return":  round(nodiv_ret,  2),
        "nodiv_cagr":    round(((nodiv_final / capital) ** (365.0 / days) - 1) * 100, 2),
        "advantage":     round(drip_final - nodiv_final),
        "advantage_pct": round(drip_ret - nodiv_ret, 2),
        "div_count":     len(div_events),
        "div_events":    div_events[-12:],
        "drip_chart":    drip_chart,
        "nodiv_chart":   nodiv_chart,
    }


# ─── 產業輪動 ──────────────────────────────────────────────────────────────────
def _fetch_sector_perf(ticker: str, name: str, group: str) -> dict | None:
    try:
        data = _yahoo_get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            {"interval": "1d", "range": "1y"},
        )
        result = (data or {}).get("chart", {}).get("result", [])
        if not result:
            return None
        r          = result[0]
        timestamps = r.get("timestamp", [])
        closes     = r.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        valid      = [(pd.Timestamp(ts, unit="s").normalize(), float(c))
                      for ts, c in zip(timestamps, closes) if c is not None]
        if len(valid) < 10:
            return None

        dates, prices = zip(*valid)
        last = prices[-1]

        def _pct_since(n_days: int) -> float | None:
            cutoff = dates[-1] - pd.Timedelta(days=n_days)
            base = next((p for dt, p in zip(dates, prices) if dt <= cutoff), None)
            return round((last - base) / base * 100, 2) if base else None

        year_start = pd.Timestamp(dates[-1].year, 1, 1)
        ytd_prices = [p for dt, p in zip(dates, prices) if dt >= year_start]
        ytd = round((last - ytd_prices[0]) / ytd_prices[0] * 100, 2) if ytd_prices else None

        return {
            "ticker":   ticker,
            "name":     name,
            "group":    group,
            "price":    round(last, 2),
            "chg_1w":   _pct_since(7),
            "chg_1m":   _pct_since(30),
            "chg_3m":   _pct_since(90),
            "chg_ytd":  ytd,
        }
    except Exception as exc:
        logger.error("Sector perf error %s: %s", ticker, exc)
        return None


def _fetch_all_sector_perf() -> list:
    order = list(SECTOR_SYMBOLS.keys())
    result = []
    with ThreadPoolExecutor(max_workers=len(SECTOR_SYMBOLS)) as pool:
        futs = {pool.submit(_fetch_sector_perf, t, m["name"], m["group"]): t
                for t, m in SECTOR_SYMBOLS.items()}
        for fut in as_completed(futs):
            try:
                v = fut.result()
                if v:
                    result.append(v)
            except Exception as exc:
                logger.warning("Sector perf error: %s", exc)
    result.sort(key=lambda x: order.index(x["ticker"]) if x["ticker"] in order else 99)
    logger.info("Sector rotation fetched: %d symbols", len(result))
    return result


# ─── 全球大盤即時報價 ─────────────────────────────────────────────────────────────
def _fetch_global_quotes() -> dict:
    result = {}
    with ThreadPoolExecutor(max_workers=len(GLOBAL_SYMBOLS)) as pool:
        futs = {pool.submit(_fetch_yahoo_quote, ticker): (ticker, name)
                for ticker, name in GLOBAL_SYMBOLS.items()}
        for fut in as_completed(futs):
            ticker, name = futs[fut]
            try:
                q = fut.result()
                if q:
                    result[ticker] = {**q, "name": name}
            except Exception as exc:
                logger.warning("Global quote error %s: %s", ticker, exc)
    logger.info("Global quotes: %s", list(result.keys()))
    return result


# ─── SOP 回測 ────────────────────────────────────────────────────────────────
def _run_sop_backtest(hist: pd.DataFrame, capital: float) -> dict:
    df = hist[["Close", "MA200"]].copy().dropna(subset=["MA200"])
    if len(df) < 30:
        return {"error": "歷史資料不足（需至少 30 個交易日含 MA200）"}
    df["daily_chg"] = df["Close"].pct_change() * 100
    df["vs_ma200"]  = (df["Close"] - df["MA200"]) / df["MA200"] * 100

    cash        = capital * 0.20
    shares      = (capital * 0.80) / float(df["Close"].iloc[0])
    events      = []
    pv_list     = []
    below3_n    = 0
    above3_n    = 0
    bought_lvls: set = set()   # smile levels bought in current holding period
    SMILE_THRS  = [-8, -10, -15, -20, -25, -30]

    for i, (date_idx, row) in enumerate(df.iterrows()):
        price  = float(row["Close"])
        vs_ma  = float(row["vs_ma200"]) if pd.notna(row["vs_ma200"]) else 0
        d_chg  = float(row["daily_chg"]) if i > 0 and pd.notna(row["daily_chg"]) else 0
        date_s = date_idx.strftime("%Y-%m-%d")
        exited_today = False

        below3_n = below3_n + 1 if vs_ma < -3 else 0
        above3_n = above3_n + 1 if vs_ma >  3 else 0

        # SOP03: retreat
        if shares > 0 and (d_chg <= -5 or below3_n >= 3):
            cash += shares * price
            events.append({
                "date": date_s, "type": "SELL", "price": round(price, 2),
                "desc": f"SOP03 {'單日暴跌' + str(round(d_chg,1)) + '%' if d_chg <= -5 else '連3日跌破年線-3%'} → 清倉",
            })
            shares = 0; bought_lvls = set(); exited_today = True

        # SOP04: smile buying levels
        for j, thr in enumerate(SMILE_THRS):
            if vs_ma <= thr and (j == len(SMILE_THRS) - 1 or vs_ma > SMILE_THRS[j + 1]):
                if thr not in bought_lvls and cash >= capital * 0.03:
                    bought_lvls.add(thr)
                    amt = min(cash * 0.05, cash)
                    shares += amt / price; cash -= amt
                    events.append({
                        "date": date_s, "type": "BUY", "price": round(price, 2),
                        "desc": f"SOP04 微笑佈局 年線{thr}% → 加碼 5% 資金",
                    })
                break

        # SOP05: recovery
        if not exited_today and shares == 0 and above3_n >= 3 and cash > 0:
            new_sh = (cash * 0.90) / price
            events.append({
                "date": date_s, "type": "BUY", "price": round(price, 2),
                "desc": f"SOP05 反攻號角連3日 → 重倉進場 90% 資金",
            })
            shares = new_sh; cash *= 0.10; bought_lvls = set()

        pv_list.append({"d": date_s, "v": round(shares * price + cash)})

    final_price = float(df["Close"].iloc[-1])
    final_val   = shares * final_price + cash
    ret_pct     = (final_val - capital) / capital * 100
    days        = max((df.index[-1] - df.index[0]).days, 1)
    cagr        = ((final_val / capital) ** (365.0 / days) - 1) * 100

    bnh_sh   = (capital * 0.80) / float(df["Close"].iloc[0])
    bnh_val  = bnh_sh * final_price + capital * 0.20
    bnh_ret  = (bnh_val - capital) / capital * 100
    bnh_cagr = ((bnh_val / capital) ** (365.0 / days) - 1) * 100

    return {
        "capital":     capital,
        "final_value": round(final_val),
        "return_pct":  round(ret_pct, 2),
        "cagr":        round(cagr, 2),
        "bnh_return":  round(bnh_ret, 2),
        "bnh_cagr":    round(bnh_cagr, 2),
        "events":      events,
        "chart":       pv_list,
        "start_date":  df.index[0].strftime("%Y-%m-%d"),
        "end_date":    df.index[-1].strftime("%Y-%m-%d"),
        "days":        days,
        "trades":      len(events),
    }


# ─── 台股籌碼：三大法人 + 融資融券 ───────────────────────────────────────────────
def _fetch_tw_chips() -> dict | None:
    """Fetch per-stock 三大法人 (T86) and 融資融券 (MI_MARGN) for TW ETFs."""
    def _int(s: str) -> int:
        try:
            return int(s.replace(",", "").replace("+", "").strip())
        except (ValueError, AttributeError):
            return 0

    result: dict = {}

    _twse_headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept":          "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-TW,zh;q=0.9",
        "Referer":         "https://www.twse.com.tw/zh/trading/fund/T86.html",
        "X-Requested-With": "XMLHttpRequest",
    }

    # T86: per-stock 外資 / 投信 / 自營商 buy-sell net (shares)
    try:
        today = datetime.now(TW_TZ).strftime("%Y%m%d")
        resp = requests.get(
            "https://www.twse.com.tw/fund/T86",
            params={"response": "json", "date": today, "selectType": "ALLBUT0999"},
            headers=_twse_headers,
            timeout=15,
        )
        if not resp.text.strip():
            raise ValueError("empty response from T86")
        data = resp.json()
        if data.get("stat") == "OK":
            fields = data.get("fields", [])
            # Locate columns — use exact match where names overlap as substrings
            def _col(name, exact=False):
                for i, f in enumerate(fields):
                    if (f == name) if exact else (name in f):
                        return i
                return -1
            idx_foreign = _col("外陸資買賣超股數")          # 外陸資(不含外資自營商)
            idx_fdealer = _col("外資自營商買賣超")           # 外資自營商
            idx_trust   = _col("投信買賣超")
            idx_dealer  = _col("自營商買賣超股數", exact=True)  # total dealer, exact!
            idx_total   = len(fields) - 1                   # 三大法人合計 is always last
            for row in data.get("data", []):
                code = row[0].strip()
                if code in TW_SYMBOLS:
                    foreign_net = ((_int(row[idx_foreign]) if idx_foreign >= 0 else 0) +
                                   (_int(row[idx_fdealer]) if idx_fdealer >= 0 else 0))
                    result[code] = {
                        "foreign_net": foreign_net,
                        "trust_net":   _int(row[idx_trust])  if idx_trust  >= 0 else 0,
                        "dealer_net":  _int(row[idx_dealer]) if idx_dealer >= 0 else 0,
                        "total_net":   _int(row[idx_total]),
                        "date":        data.get("date", ""),
                    }
        logger.info("T86 chips: %s", list(result.keys()))
    except Exception as exc:
        logger.error("T86 fetch error: %s", exc)

    # MI_MARGN: per-stock 融資 / 融券 balance (unit: 張=1000 shares)
    try:
        resp = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN",
            headers={**_twse_headers, "Referer": "https://openapi.twse.com.tw/"},
            timeout=15,
        )
        if not resp.text.strip():
            raise ValueError("empty response from MI_MARGN")
        for row in resp.json():
            code = row.get("股票代號", "").strip()
            if code in TW_SYMBOLS:
                margin_bal  = _int(row.get("融資今日餘額", ""))
                margin_prev = _int(row.get("融資前日餘額", ""))
                short_bal   = _int(row.get("融券今日餘額", ""))
                short_prev  = _int(row.get("融券前日餘額", ""))
                result.setdefault(code, {}).update({
                    "margin_bal":  margin_bal,
                    "margin_chg":  margin_bal - margin_prev,
                    "short_bal":   short_bal,
                    "short_chg":   short_bal - short_prev,
                })
        logger.info("MI_MARGN chips: %s", [k for k in result if "margin_bal" in result[k]])
    except Exception as exc:
        logger.error("MI_MARGN fetch error: %s", exc)

    return result or None


# ─── Build stock entry ────────────────────────────────────────────────────────
def _build_stock_entry(name: str, hist: pd.DataFrame, quote: dict) -> dict:
    current   = quote["price"]
    prev      = quote["prev_close"]
    daily_chg = quote["daily_change"]

    ma200 = float(hist["MA200"].iloc[-1]) if not pd.isna(hist["MA200"].iloc[-1]) else None
    ma50  = float(hist["MA50"].iloc[-1])  if not pd.isna(hist["MA50"].iloc[-1])  else None
    pct_from_ma200 = (current - ma200) / ma200 * 100 if ma200 else None

    last3: list[float] = []
    for i in range(-3, 0):
        try:
            p = float(hist["Close"].iloc[i])
            m = float(hist["MA200"].iloc[i])
            if not np.isnan(m):
                last3.append((p - m) / m * 100)
        except Exception:
            pass

    # RSI 14
    rsi_series = _calc_rsi(hist["Close"])
    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None

    # MACD 12/26/9
    macd_s, signal_s, _ = _calc_macd(hist["Close"])
    macd_val = float(macd_s.iloc[-1])   if not pd.isna(macd_s.iloc[-1])   else None
    macd_sig = float(signal_s.iloc[-1]) if not pd.isna(signal_s.iloc[-1]) else None
    macd_cross = None
    macd_trend = None
    if len(macd_s) >= 2 and macd_val is not None and macd_sig is not None:
        curr_diff = macd_val - macd_sig
        prev_macd = float(macd_s.iloc[-2])
        prev_sig  = float(signal_s.iloc[-2])
        prev_diff = prev_macd - prev_sig
        macd_trend = "bull" if curr_diff > 0 else "bear"
        if not np.isnan(prev_diff):
            if prev_diff < 0 and curr_diff >= 0:
                macd_cross = "golden"
            elif prev_diff > 0 and curr_diff <= 0:
                macd_cross = "dead"

    # KD 9 (needs High/Low)
    k_val = d_val = None
    if "High" in hist.columns and "Low" in hist.columns:
        k_s, d_s = _calc_kd(hist)
        k_val = float(k_s.iloc[-1]) if not pd.isna(k_s.iloc[-1]) else None
        d_val = float(d_s.iloc[-1]) if not pd.isna(d_s.iloc[-1]) else None

    # Divergence (RSI vs Price, 60-bar lookback)
    divergence = _calc_divergence(hist["Close"], rsi_series)

    ch = hist.tail(252)
    chart = {
        "dates":  ch.index.strftime("%Y-%m-%d").tolist(),
        "prices": [round(x, 2) for x in ch["Close"].tolist()],
        "ma200":  [None if pd.isna(x) else round(float(x), 2) for x in ch["MA200"]],
        "ma50":   [None if pd.isna(x) else round(float(x), 2) for x in ch["MA50"]],
    }

    return {
        "name":           SYMBOL_NAMES.get(name, name),
        "price":          round(current, 2),
        "prev_close":     round(prev, 2),
        "daily_change":   round(daily_chg, 2),
        "ma200":          round(ma200, 2) if ma200 else None,
        "ma50":           round(ma50,  2) if ma50  else None,
        "pct_from_ma200": round(pct_from_ma200, 2) if pct_from_ma200 is not None else None,
        "above_ma200":    current > ma200 if ma200 else None,
        "last3_vs_ma":    [round(p, 2) for p in last3],
        "rsi":            round(rsi, 2)      if rsi      is not None else None,
        "macd_val":       round(macd_val, 4) if macd_val is not None else None,
        "macd_signal":    round(macd_sig, 4) if macd_sig is not None else None,
        "macd_cross":     macd_cross,
        "macd_trend":     macd_trend,
        "k_val":       round(k_val, 2) if k_val is not None else None,
        "d_val":       round(d_val, 2) if d_val is not None else None,
        "divergence":  divergence,
        "chart":       chart,
        "nav_est":        round(quote["nav_est"], 2)     if quote.get("nav_est")     else None,
        "premium_pct":    round(quote["premium_pct"], 2) if quote.get("premium_pct") is not None else None,
        "is_market_open": quote.get("is_market_open", False),
    }


# ─── Refresh logic ────────────────────────────────────────────────────────────
def _do_refresh(refresh_hist: bool, refresh_price: bool):
    if refresh_hist:
        logger.info("Refreshing histories and market data (parallel)…")

        def _tw_hist(name, av_sym):
            h = _fetch_yahoo_history(av_sym)
            if h is None:
                logger.warning("Yahoo hist failed for %s, trying TWSE…", name)
                h = _fetch_twse_history(av_sym.replace(".TW", ""))
            return name, h

        def _yf_hist(name, ticker):
            return name, _fetch_yahoo_history(ticker)

        # Phase 1: all non-TD fetches in parallel
        with ThreadPoolExecutor(max_workers=12) as pool:
            tw_futs   = [pool.submit(_tw_hist, n, s)      for n, s   in TW_SYMBOLS.items()]
            yf_futs   = [pool.submit(_yf_hist, n, t)      for n, t   in YF_US_SYMBOLS.items()]
            misc_futs = {
                pool.submit(_fetch_vix):           "vix",
                pool.submit(_fetch_foreign_net):   "foreign_net",
                pool.submit(_fetch_tw_chips):      "chips",
                pool.submit(_fetch_usdtwd):        "usdtwd",
                pool.submit(_fetch_fear_greed):    "fear_greed",
                pool.submit(_fetch_global_quotes): "global_quotes",
            }
            for fut in as_completed(tw_futs + yf_futs):
                name, hist = fut.result()
                if hist is not None:
                    _cache["histories"][name] = hist
            for fut, key in misc_futs.items():
                val = fut.result()
                if val:
                    if key == "chips":
                        _cache["tw_chips"] = val; _cache["chips_ts"] = time.time()
                    elif key == "global_quotes":
                        _cache["global_quotes"] = val; _cache["global_quotes_ts"] = time.time()
                    else:
                        _cache[key] = val

        # Phase 2: TD histories (rate-limited, must stay sequential)
        logger.info("Refreshing US histories (Twelve Data)…")
        for name, (sym, exch) in US_SYMBOLS.items():
            _td_throttle()
            hist = _fetch_td_history(sym, exch)
            if hist is None:
                logger.warning("TD hist failed for %s, trying Yahoo…", name)
                hist = _fetch_yahoo_history(sym)
            if hist is not None:
                _cache["histories"][name] = hist

        _cache["hist_ts"] = time.time()

    if refresh_price:
        logger.info("Refreshing prices (parallel)…")
        now_t = time.time()

        def _tw_entry(name, av_sym, tw_quotes):
            quote = tw_quotes.get(name)
            if quote is None:
                logger.warning("TWSE quote missing for %s, trying Yahoo…", name)
                quote = _fetch_yahoo_quote(av_sym)
            hist = _cache["histories"].get(name)
            return name, (_build_stock_entry(name, hist, quote) if quote and hist is not None else None)

        def _yf_entry(name, ticker):
            quote = _fetch_yahoo_quote(ticker)
            hist  = _cache["histories"].get(name)
            return name, (_build_stock_entry(name, hist, quote) if quote and hist is not None else None)

        do_vix    = now_t - _cache["vix_ts"]           > VIX_TTL
        do_chips  = now_t - _cache["chips_ts"]         > CHIPS_TTL
        do_usd    = now_t - _cache["usdtwd_ts"]        > USDTWD_TTL
        do_fg     = now_t - _cache["fear_greed_ts"]    > FG_TTL
        do_global = now_t - _cache["global_quotes_ts"] > GLOBAL_TTL

        # Phase 1: bulk TWSE quote, then all non-TD entries + optional misc in parallel
        tw_quotes = _fetch_tw_quotes()
        with ThreadPoolExecutor(max_workers=12) as pool:
            entry_futs = (
                [pool.submit(_tw_entry, n, s, tw_quotes) for n, s in TW_SYMBOLS.items()] +
                [pool.submit(_yf_entry, n, t)            for n, t in YF_US_SYMBOLS.items()]
            )
            misc_futs = {}
            if do_vix:    misc_futs[pool.submit(_fetch_vix)]           = "vix"
            if do_chips:  misc_futs[pool.submit(_fetch_tw_chips)]      = "chips"
            if do_usd:    misc_futs[pool.submit(_fetch_usdtwd)]        = "usdtwd"
            if do_fg:     misc_futs[pool.submit(_fetch_fear_greed)]    = "fear_greed"
            if do_global: misc_futs[pool.submit(_fetch_global_quotes)] = "global_quotes"

            for fut in as_completed(entry_futs):
                name, entry = fut.result()
                if entry:
                    _cache["stocks"][name] = entry
            for fut, key in misc_futs.items():
                val = fut.result()
                if val:
                    if key == "chips":
                        _cache["tw_chips"] = val; _cache["chips_ts"] = now_t
                    elif key == "global_quotes":
                        _cache["global_quotes"] = val; _cache["global_quotes_ts"] = now_t
                    elif key == "vix":
                        _cache["vix"] = val; _cache["vix_ts"] = now_t
                    elif key == "usdtwd":
                        _cache["usdtwd"] = val; _cache["usdtwd_ts"] = now_t
                    elif key == "fear_greed":
                        _cache["fear_greed"] = val; _cache["fear_greed_ts"] = now_t

        # Phase 2: TD quotes (rate-limited, must stay sequential)
        logger.info("Refreshing US prices (Twelve Data)…")
        for name, (sym, exch) in US_SYMBOLS.items():
            _td_throttle()
            quote = _fetch_td_quote(sym, exch)
            if quote is None:
                logger.warning("TD quote failed for %s, trying Yahoo…", name)
                quote = _fetch_yahoo_quote(sym)
            hist = _cache["histories"].get(name)
            if quote and hist is not None:
                _cache["stocks"][name] = _build_stock_entry(name, hist, quote)

        _cache["price_ts"] = now_t


def _background_refresh(hist: bool = False, price: bool = True):
    if not _refresh_lock.acquire(blocking=False):
        return
    _cache["refreshing"] = True
    try:
        _do_refresh(refresh_hist=hist, refresh_price=price)
        logger.info("Refresh done (hist=%s price=%s).", hist, price)
    except Exception as exc:
        logger.error("Refresh error: %s", exc)
    finally:
        _cache["refreshing"] = False
        _refresh_lock.release()


def cached_data() -> dict:
    now        = time.time()
    need_hist  = not _cache["histories"] or (now - _cache["hist_ts"])  > HIST_TTL
    need_price = not _cache["stocks"]    or (now - _cache["price_ts"]) > PRICE_TTL
    need_chips = _cache["tw_chips"] is None or (now - _cache["chips_ts"]) > CHIPS_TTL

    if (not _cache["stocks"] or need_hist or need_price or need_chips) and not _cache["refreshing"]:
        is_cold = not _cache["stocks"]
        if is_cold:
            logger.info("Cold start…")
        threading.Thread(
            target=_background_refresh,
            kwargs={"hist": is_cold or need_hist, "price": True},
            daemon=True,
        ).start()

    return _cache["stocks"]


# ─── SOP logic ────────────────────────────────────────────────────────────────
def compute_sop(stocks: dict) -> dict | None:
    tw  = stocks.get("0050")
    qqq = stocks.get("QQQ",  {})
    if not tw:
        return None   # caller renders "等待 0050 資料" instead of fake 正常

    p0050 = tw.get("pct_from_ma200") or 0.0
    pqqq  = qqq.get("pct_from_ma200") or 0.0
    d0050 = tw.get("daily_change") or 0.0
    l3    = tw.get("last3_vs_ma", [])
    above = tw.get("above_ma200", True)
    ma200 = tw.get("ma200") or 0.0
    price = tw.get("price") or 0.0

    rsi        = tw.get("rsi")
    macd_cross = tw.get("macd_cross")
    macd_trend = tw.get("macd_trend")
    k_val      = tw.get("k_val")
    d_val      = tw.get("d_val")
    div        = tw.get("divergence") or {}
    bearish_div = div.get("bearish", False)
    bullish_div = div.get("bullish", False)

    smile_levels = []
    for t in [-8, -10, -15, -20, -25, -30]:
        tp = ma200 * (1 + t / 100) if ma200 else 0
        smile_levels.append({
            "threshold": t,
            "price":     round(tp, 2),
            "triggered": price <= tp if tp else False,
        })

    def s02():
        if pqqq <= -10: return "alert"
        if pqqq <= -5:  return "watch"
        return "normal"

    def s03():
        if (len(l3) >= 3 and all(p <= -3 for p in l3)) or d0050 <= -5:
            return "alert"
        # MACD 死叉 or 空頭趨勢 加入 watch 條件
        if p0050 <= -3 or macd_cross == "dead" or (macd_trend == "bear" and not above):
            return "watch"
        return "normal"

    def s04():
        return "active" if not above else "standby"

    def s05():
        strong = (len(l3) >= 3 and all(p >= 3 for p in l3)) or d0050 >= 5
        if strong and bearish_div:
            return "watch"   # 頂部背離降級：暫緩全力壓正2
        if strong:
            return "alert"
        if p0050 >= 3 and above:
            return "watch"
        return "standby"

    rsi_txt  = f"RSI {rsi:.0f}" if rsi is not None else "RSI -"
    kd_txt   = f"K {k_val:.0f}/D {d_val:.0f}" if k_val is not None else "KD -"
    macd_txt = {"golden": "金叉↑", "dead": "死叉↓"}.get(
        macd_cross or "", "多頭" if macd_trend == "bull" else "空頭" if macd_trend == "bear" else "-"
    )
    # 背離標記（SOP desc 用）
    div04_txt = "　底部背離✓" if bullish_div else ""
    div05_txt = "　⚠頂部背離" if bearish_div else ""

    return {
        "sop01": {
            "name": "建軍配置", "status": "active",
            "desc_rule": "40% 原型 + 40% 正2 + 20% 現金，每年底再平衡",
            "desc_val":  "每年底若線下已出清正2，則不需再平衡",
        },
        "sop02": {
            "name": "雙核雷達", "status": s02(),
            "qqq_pct":    round(pqqq, 2),
            "tw50_above": above,
            "desc_rule": "台股看0050年線；QQQ跌破年線-10% → 正2強制清倉",
            "desc_val":  f"QQQ 年線偏離：{pqqq:+.2f}%",
        },
        "sop03": {
            "name": "撤退機制", "status": s03(),
            "daily_change": round(d0050, 2),
            "last3":        l3,
            "desc_rule": "跌破年線-3%連3天，或單日-5% → 出清正2轉備戰現金",
            "desc_val":  f"0050 日漲跌：{d0050:+.2f}%　MACD {macd_txt}",
        },
        "sop04": {
            "name": "微笑佈局", "status": s04(),
            "pct_from_ma":  round(p0050, 2),
            "smile_levels": smile_levels,
            "desc_rule": "線下只買原型；左側各5%，右側各2%；-30%冬眠",
            "desc_val":  f"年線偏離：{p0050:+.2f}%　{rsi_txt}　{kd_txt}{div04_txt}",
        },
        "sop05": {
            "name": "反攻號角", "status": s05(),
            "pct_from_ma":  round(p0050, 2),
            "daily_change": round(d0050, 2),
            "desc_rule": "站回年線+3%連3天 或 單日+5% → 全數壓正2",
            "desc_val":  f"年線偏離：{p0050:+.2f}%　MACD {macd_txt}　{rsi_txt}{div05_txt}",
        },
    }


def check_alerts(stocks: dict, sop: dict | None) -> list[dict]:
    if not sop:
        return []
    alerts  = []
    now_str = datetime.now(TW_TZ).isoformat()
    tw  = stocks.get("0050", {})
    qqq = stocks.get("QQQ",  {})

    pqqq  = qqq.get("pct_from_ma200") or 0.0
    p0050 = tw.get("pct_from_ma200")  or 0.0
    d0050 = tw.get("daily_change")    or 0.0
    l3    = tw.get("last3_vs_ma", [])
    rsi        = tw.get("rsi")
    macd_cross = tw.get("macd_cross")
    above      = tw.get("above_ma200", True)
    div        = tw.get("divergence") or {}
    bearish_div = div.get("bearish", False)
    bullish_div = div.get("bullish", False)

    # ── 原有 SOP 警報 ──
    if pqqq <= -10:
        alerts.append({"type": "CRITICAL", "code": "02", "ts": now_str,
            "title": "雙核雷達觸發",
            "msg": f"QQQ 跌破年線 {pqqq:.1f}%（閾值 -10%），台股正2 無條件強制清倉！"})

    if len(l3) >= 3 and all(p <= -3 for p in l3):
        alerts.append({"type": "CRITICAL", "code": "03", "ts": now_str,
            "title": "撤退機制（連續3日）",
            "msg": "0050 連續3天跌破年線 -3%，立刻出清正2轉備戰現金！"})

    if d0050 <= -5:
        alerts.append({"type": "CRITICAL", "code": "03", "ts": now_str,
            "title": "撤退機制（單日暴跌）",
            "msg": f"0050 單日跌幅 {d0050:.1f}%（閾值 -5%），立刻出清正2轉備戰現金！"})

    # ── MACD 警報 ──
    if macd_cross == "dead":
        alerts.append({"type": "CRITICAL", "code": "03", "ts": now_str,
            "title": "MACD 死叉預警",
            "msg": "0050 MACD 出現死叉，動能轉弱，搭配其他訊號注意撤退時機"})

    # ── 微笑佈局買點 ──
    for lvl in sop.get("sop04", {}).get("smile_levels", []):
        if lvl["triggered"] and abs(p0050 - lvl["threshold"]) < 0.8:
            alerts.append({"type": "BUY", "code": "04", "ts": now_str,
                "title": f"微笑佈局 {lvl['threshold']}% 買點",
                "msg": f"0050 接近年線 {lvl['threshold']}% 位置（現 {p0050:.1f}%），左側佈局 5%"})

    # ── RSI 超賣確認 ──
    if rsi is not None and rsi < 30 and not above:
        alerts.append({"type": "BUY", "code": "04", "ts": now_str,
            "title": "RSI 超賣確認",
            "msg": f"0050 RSI {rsi:.1f} 進入超賣區（< 30），搭配微笑佈局加碼訊號強化"})

    # ── 反攻號角 ──
    if len(l3) >= 3 and all(p >= 3 for p in l3):
        alerts.append({"type": "BUY", "code": "05", "ts": now_str,
            "title": "反攻號角（連續3日）",
            "msg": "0050 連續3天站回年線+3%，底部原型 + 所有備戰現金全數壓正2！"})

    if d0050 >= 5:
        alerts.append({"type": "BUY", "code": "05", "ts": now_str,
            "title": "反攻號角（單日大漲）",
            "msg": f"0050 單日漲幅 {d0050:.1f}%（閾值 +5%），全數壓正2！"})

    if macd_cross == "golden" and not above:
        alerts.append({"type": "BUY", "code": "05", "ts": now_str,
            "title": "MACD 金叉反攻訊號",
            "msg": "0050 年線下出現 MACD 金叉，動能反轉，留意反攻號角確認"})

    # ── 背離警報 ──
    if bearish_div:
        alerts.append({"type": "CRITICAL", "code": "DIV", "ts": now_str,
            "title": "頂部背離警示（假突破風險）",
            "msg": "0050 價格創新高但 RSI 未跟進，動能衰退，反攻號角暫緩加碼正2"})

    if bullish_div and not above:
        alerts.append({"type": "BUY", "code": "DIV", "ts": now_str,
            "title": "底部背離確認（底部訊號）",
            "msg": "0050 價格破低但 RSI 止穩，下跌動能枯竭，微笑佈局信心加強"})

    # ── VIX 警報 ──
    vix = _cache.get("vix")
    if vix and vix.get("price", 0) > 30:
        alerts.append({"type": "CRITICAL", "code": "VX", "ts": now_str,
            "title": "VIX 恐慌指數偏高",
            "msg": f"VIX {vix['price']:.1f}（> 30），市場高波動期，正2槓桿風險加倍，請謹慎操作"})

    # ── 外資大幅賣超 ──
    fn = _cache.get("foreign_net")
    if fn and fn.get("foreign") is not None and fn["foreign"] < -5_000_000_000:
        sold_yi = abs(fn["foreign"]) / 1e8
        alerts.append({"type": "CRITICAL", "code": "FN", "ts": now_str,
            "title": "外資大幅賣超",
            "msg": f"外資今日賣超約 {sold_yi:.0f} 億元，搭配年線訊號判斷是否撤退"})

    return alerts


# ─── Notification ─────────────────────────────────────────────────────────────
def _send_notification_sync(message: str):
    for uid in LINE_USER_IDS:
        try:
            r = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {LINE_TOKEN}",
                },
                json={"to": uid, "messages": [{"type": "text", "text": message}]},
                timeout=10,
            )
            logger.info("LINE push to %s: %s", uid, r.status_code)
        except Exception as e:
            logger.error("LINE error (uid=%s): %s", uid, e)

def send_notification(message: str):
    if not LINE_TOKEN or not LINE_USER_IDS:
        return
    threading.Thread(target=_send_notification_sync, args=(message,), daemon=True).start()


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def api_dashboard():
    try:
        stocks = cached_data()
        sop    = compute_sop(stocks)
        alerts = check_alerts(stocks, sop)

        new_alerts = False
        for a in alerts:
            # Deduplicate by code+title (ignore ts which changes every call)
            is_dup = any(
                h.get("code") == a.get("code") and h.get("title") == a.get("title")
                for h in alert_history
            )
            if not is_dup:
                alert_history.insert(0, a)
                new_alerts = True
        while len(alert_history) > 50:
            alert_history.pop()
        if new_alerts:
            _save_alert_history()

        return jsonify({
            "stocks":        stocks,
            "sop":           sop,
            "alerts":        alerts,
            "alert_history": alert_history[:20],
            "updated_at":    datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "price_age":     int(time.time() - _cache["price_ts"]) if _cache["price_ts"] > 0 else None,
            "hist_age":      int(time.time() - _cache["hist_ts"])  if _cache["hist_ts"]  > 0 else None,
            "vix":           _cache.get("vix"),
            "foreign_net":   _cache.get("foreign_net"),
            "tw_chips":      _cache.get("tw_chips"),
            "usdtwd":        _cache.get("usdtwd"),
            "fear_greed":    _cache.get("fear_greed"),
            "global_quotes": _cache.get("global_quotes", {}),
        })
    except Exception as exc:
        logger.exception("Dashboard error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/notify", methods=["POST"])
def api_notify():
    body = request.get_json(silent=True) or {}
    msg  = body.get("message", "[測試] 台美股監控系統運作正常")
    send_notification(msg)
    return jsonify({"ok": True})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    _cache["price_ts"] = 0.0
    _cache["hist_ts"]  = 0.0
    threading.Thread(
        target=_background_refresh,
        kwargs={"hist": True, "price": True},
        daemon=True,
    ).start()
    return jsonify({"ok": True, "msg": "Full refresh triggered in background"})


def _line_reply(reply_token: str, message: str):
    if not LINE_TOKEN:
        logger.error("LINE reply skipped: LINE_CHANNEL_ACCESS_TOKEN is not set")
        return
    try:
        r = requests.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"},
            json={"replyToken": reply_token, "messages": [{"type": "text", "text": message}]},
            timeout=10,
        )
        logger.info("LINE reply status: %s body: %s", r.status_code, r.text)
    except Exception as e:
        logger.error("LINE reply error: %s", e)


@app.route("/webhook", methods=["POST"])
def line_webhook():
    body   = request.get_json(silent=True) or {}
    events = body.get("events", [])
    logger.info("Webhook received: %d event(s)", len(events))
    for event in events:
        uid = event.get("source", {}).get("userId", "")
        if uid:
            _line_users[uid] = {"last_seen": datetime.now(TW_TZ).isoformat()}

        if event.get("type") != "message":
            continue
        if event.get("message", {}).get("type") != "text":
            continue

        reply_token = event.get("replyToken", "")
        msg_text    = event.get("message", {}).get("text", "").strip()
        logger.info("Message uid=%s msg=%r", uid, msg_text)

        if uid:
            _line_users[uid]["last_msg"] = msg_text

        if reply_token and msg_text.lower() in ("id", "我的id", "userid", "user id", "my id"):
            _line_reply(reply_token, f"你的 LINE User ID 是：\n{uid}")
    return "OK", 200


@app.route("/api/line-users")
def api_line_users():
    return jsonify({
        "token_configured":      bool(LINE_TOKEN),
        "push_users_configured": LINE_USER_IDS,
        "seen_users":            _line_users,
    })


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().upper()
    if not q or len(q) > 12:
        return jsonify({"error": "請輸入有效的股票代號"}), 400

    is_tw  = bool(re.match(r"^\d", q))
    ticker = f"{q}.TW" if is_tw else q
    now    = time.time()
    cached = _search_cache.get(ticker, {})

    # Fetch history only when cache is cold or expired
    hist = None
    if now - cached.get("hist_ts", 0) < SEARCH_HIST_TTL:
        hist = cached.get("hist")
    if hist is None:
        hist = _fetch_yahoo_history(ticker)
        if hist is None:
            return jsonify({"error": f"「{q}」歷史資料取得失敗"}), 404
        cached["hist"]    = hist
        cached["hist_ts"] = now

    # Fetch quote only when price cache is stale
    price_from_cache = False
    quote = None
    if now - cached.get("price_ts", 0) < SEARCH_PRICE_TTL:
        quote = cached.get("quote")
        if quote:
            price_from_cache = True
    if quote is None:
        quote = _fetch_yahoo_quote(ticker)
        if not quote:
            return jsonify({"error": f"找不到「{q}」，請確認代號正確"}), 404
        cached["quote"]    = quote
        cached["price_ts"] = now

    # Evict oldest entries when cache grows too large
    if len(_search_cache) > 100:
        oldest = min(_search_cache, key=lambda k: _search_cache[k].get("price_ts", 0))
        _search_cache.pop(oldest, None)
    _search_cache[ticker] = cached

    entry = _build_stock_entry(q, hist, quote)
    if entry.get("name") == q:
        entry["name"] = quote.get("short_name") or q
    entry["code"]        = q
    entry["ticker"]      = ticker
    entry["is_tw"]       = is_tw
    entry["from_cache"]  = price_from_cache   # True = served from cache, False = fresh fetch
    return jsonify(entry)


@app.route("/api/dividend-sim")
def api_dividend_sim():
    sym = request.args.get("sym", "0056").upper()
    try:    capital = float(request.args.get("capital", "1000000"))
    except: capital = 1_000_000
    start_str = request.args.get("start", "")

    ticker = f"{sym}.TW" if re.match(r"^\d", sym) else sym
    now    = time.time()
    cache_key = f"div_{ticker}"
    cached = _search_cache.get(cache_key, {})

    hist = cached.get("hist") if now - cached.get("hist_ts", 0) < SEARCH_HIST_TTL else None
    if hist is None:
        hist = _fetch_yahoo_history(ticker)
        if hist is None:
            return jsonify({"error": f"「{sym}」歷史資料取得失敗"}), 404
        cached["hist"] = hist; cached["hist_ts"] = now

    divs = cached.get("divs") if now - cached.get("div_ts", 0) < SEARCH_HIST_TTL else None
    if divs is None:
        divs = _fetch_yahoo_dividends(ticker)
        cached["divs"] = divs; cached["div_ts"] = now

    _search_cache[cache_key] = cached

    if not divs:
        return jsonify({"error": f"「{sym}」找不到配息記錄，請確認為高股息標的"}), 404
    return jsonify(_run_dividend_sim(hist, divs, capital, start_str))


@app.route("/api/sector-rotation")
def api_sector_rotation():
    now = time.time()
    if not _cache.get("sector") or now - _cache.get("sector_ts", 0) > SECTOR_TTL:
        data = _fetch_all_sector_perf()
        if data:
            _cache["sector"]    = data
            _cache["sector_ts"] = now
    return jsonify(_cache.get("sector", []))


@app.route("/api/backtest")
def api_backtest():
    sym = request.args.get("sym", "0050").upper()
    try:
        capital = float(request.args.get("capital", "1000000"))
    except (ValueError, TypeError):
        capital = 1_000_000
    hist = _cache["histories"].get(sym)
    if hist is None:
        return jsonify({"error": f"{sym} 歷史資料未載入，請稍後再試"}), 404
    return jsonify(_run_sop_backtest(hist, capital))


@app.route("/api/dca")
def api_dca():
    sym = request.args.get("sym", "0050").upper()
    try:    monthly = float(request.args.get("monthly", "10000"))
    except: monthly = 10000.0
    start_str = request.args.get("start", "")

    hist = _cache["histories"].get(sym)
    if hist is None:
        return jsonify({"error": f"{sym} 歷史資料未載入"}), 404

    hist_c = hist.dropna(subset=["Close"])
    if hist_c.empty:
        return jsonify({"error": "無有效歷史資料"}), 404

    today = datetime.now(TW_TZ).date()
    if re.match(r"^\d{4}-\d{2}$", start_str):
        sy, sm = int(start_str[:4]), int(start_str[5:7])
    else:
        sy = today.year - 2; sm = today.month

    rows: list = []
    total_invested = 0.0
    total_shares   = 0.0
    y, m = sy, sm

    while (y < today.year) or (y == today.year and m <= today.month):
        month_data = hist_c[(hist_c.index.year == y) & (hist_c.index.month == m)]
        if not month_data.empty:
            price = float(month_data["Close"].iloc[0])
            bought = monthly / price
            total_shares   += bought
            total_invested += monthly
            rows.append({"ym": f"{y}-{m:02d}", "price": round(price, 2),
                         "shares": round(bought, 4), "cum_cost": round(total_invested, 0)})
        m += 1
        if m > 12: m = 1; y += 1

    if not rows:
        return jsonify({"error": "指定期間內無資料，請調整起始月份"}), 404

    current_price = float(hist_c["Close"].iloc[-1])
    current_value = total_shares * current_price
    profit        = current_value - total_invested
    ret_pct       = profit / total_invested * 100 if total_invested else 0
    avg_cost      = total_invested / total_shares  if total_shares   else 0

    span_d = max((date(today.year, today.month, 1) - date(sy, sm, 1)).days, 1)
    cagr   = ((current_value / total_invested) ** (365.0 / span_d) - 1) * 100 if total_invested else 0

    return jsonify({
        "sym":            sym,
        "monthly":        monthly,
        "total_invested": round(total_invested),
        "total_shares":   round(total_shares, 4),
        "current_price":  round(current_price, 2),
        "current_value":  round(current_value),
        "profit":         round(profit),
        "return_pct":     round(ret_pct, 2),
        "cagr":           round(cagr, 2),
        "avg_cost":       round(avg_cost, 2),
        "rows":           rows[-24:],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
