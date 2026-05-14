import os
import re
import json
import time
import logging
import threading
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
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

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "stocks_config.json")

def _load_stocks_config():
    with open(_CONFIG_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    tw_symbols    = cfg["tw_symbols"]
    us_symbols    = {k: (v["symbol"], v["exchange"]) for k, v in cfg["us_td_symbols"].items()}
    us_yf_symbols = cfg["us_yf_symbols"]
    names         = cfg["names"]
    return tw_symbols, us_symbols, us_yf_symbols, names

TW_SYMBOLS, US_SYMBOLS, YF_US_SYMBOLS, SYMBOL_NAMES = _load_stocks_config()

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

    "refreshing":  False,
    "vix":         None,
    "foreign_net": None,
    "tw_chips":    None,
    "taiex_pe":    None,
    "taiex_pe_ts": 0.0,
    "pcr":         None,
    "pcr_ts":      0.0,
    "news":        None,
    "news_ts":     0.0,
}
VIX_TTL    = 1800   # 30 min — VIX updates intraday but not per-minute
CHIPS_TTL  = 7200   # 2 hours — TWSE chips published once daily ~14:30
USDTWD_TTL = 1800   # 30 min
FG_TTL     = 3600   # 1 hour — CNN Fear & Greed updates a few times daily
GLOBAL_TTL = 1800   # 30 min

PE_TTL     = 3600   # 1 hour — TWSE publishes P/E once per trading day
PCR_TTL    = 1800   # 30 min — TAIFEX updates intraday
GLOBAL_SYMBOLS = {
    "^TWII": "台灣加權",
    "^N225": "日經225",
    "^GSPC": "S&P 500",
    "^IXIC": "那斯達克",
    "SOXX":  "費半ETF",
}

_refresh_lock = threading.Lock()
_line_users: dict = {}


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


# ─── 大盤本益比 (TAIEX P/E) ──────────────────────────────────────────────────
def _fetch_taiex_pe() -> dict | None:
    try:
        resp = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_d",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        rows = resp.json()
        pe_vals  = []
        yld_vals = []
        pb_vals  = []
        for row in rows:
            for key, store in (("PEratio", pe_vals), ("DividendYield", yld_vals), ("PBratio", pb_vals)):
                try:
                    v = float(row.get(key, ""))
                    if v > 0:
                        store.append(v)
                except (ValueError, TypeError):
                    pass
        # Filter extreme outliers for P/E (loss-making firms skew the mean)
        pe_vals = [v for v in pe_vals if v < 200]
        if not pe_vals:
            return None
        date_str = rows[0].get("Date", "") if rows else ""
        return {
            "date":       date_str,
            "pe_median":  round(float(np.median(pe_vals)), 2),
            "yld_median": round(float(np.median(yld_vals)), 2) if yld_vals else None,
            "pb_median":  round(float(np.median(pb_vals)),  2) if pb_vals  else None,
            "count":      len(pe_vals),
        }
    except Exception as exc:
        logger.error("TAIEX PE fetch error: %s", exc)
        return None


# ─── 選擇權 Put/Call Ratio (PCR) ──────────────────────────────────────────────
def _fetch_pcr() -> dict | None:
    try:
        resp = requests.get(
            "https://www.taifex.com.tw/cht/3/pcRatio",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.taifex.com.tw/"},
            timeout=15,
        )
        for row_html in re.findall(r'<tr[^>]*>(.*?)</tr>', resp.text, re.DOTALL):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
            cols  = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if len(cols) >= 7 and '/' in (cols[0] if cols else ''):
                def _n(s):
                    try: return float(s.replace(',', ''))
                    except: return None
                return {
                    "date":     cols[0],
                    "put_vol":  _n(cols[1]),
                    "call_vol": _n(cols[2]),
                    "vol_pcr":  _n(cols[3]),
                    "put_oi":   _n(cols[4]),
                    "call_oi":  _n(cols[5]),
                    "oi_pcr":   _n(cols[6]),
                }
        return None
    except Exception as exc:
        logger.error("PCR fetch error: %s", exc)
        return None



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
                pool.submit(_fetch_taiex_pe):      "taiex_pe",
                pool.submit(_fetch_pcr):           "pcr",
            }
            for fut in as_completed(tw_futs + yf_futs):
                name, hist = fut.result()
                if hist is not None:
                    _cache["histories"][name] = hist
            now_t = time.time()
            for fut, key in misc_futs.items():
                val = fut.result()
                if val:
                    if key == "chips":
                        _cache["tw_chips"] = val; _cache["chips_ts"] = now_t
                    elif key == "global_quotes":
                        _cache["global_quotes"] = val; _cache["global_quotes_ts"] = now_t
                    elif key == "taiex_pe":
                        _cache["taiex_pe"] = val; _cache["taiex_pe_ts"] = now_t
                    elif key == "pcr":
                        _cache["pcr"] = val; _cache["pcr_ts"] = now_t
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
        do_pe     = now_t - _cache["taiex_pe_ts"]      > PE_TTL
        do_pcr    = now_t - _cache["pcr_ts"]           > PCR_TTL

        # Phase 1: bulk TWSE quote, then all non-TD entries + optional misc in parallel
        tw_quotes = _fetch_tw_quotes()
        with ThreadPoolExecutor(max_workers=14) as pool:
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
            if do_pe:     misc_futs[pool.submit(_fetch_taiex_pe)]      = "taiex_pe"
            if do_pcr:    misc_futs[pool.submit(_fetch_pcr)]           = "pcr"

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
                    elif key == "taiex_pe":
                        _cache["taiex_pe"] = val; _cache["taiex_pe_ts"] = now_t
                    elif key == "pcr":
                        _cache["pcr"] = val; _cache["pcr_ts"] = now_t

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

        return jsonify({
            "stocks":        stocks,
            "updated_at":    datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "price_age":     int(time.time() - _cache["price_ts"]) if _cache["price_ts"] > 0 else None,
            "hist_age":      int(time.time() - _cache["hist_ts"])  if _cache["hist_ts"]  > 0 else None,
            "vix":           _cache.get("vix"),
            "foreign_net":   _cache.get("foreign_net"),
            "tw_chips":      _cache.get("tw_chips"),
            "usdtwd":        _cache.get("usdtwd"),
            "fear_greed":    _cache.get("fear_greed"),
            "global_quotes": _cache.get("global_quotes", {}),
            "taiex_pe":      _cache.get("taiex_pe"),
            "pcr":           _cache.get("pcr"),
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




def _resolve_hist(sym: str) -> pd.DataFrame | None:
    """Return history DataFrame for any symbol, fetching on-demand if needed."""
    hist = _cache["histories"].get(sym)
    if hist is not None:
        return hist
    is_tw  = bool(re.match(r"^\d", sym))
    ticker = f"{sym}.TW" if is_tw else sym
    now    = time.time()
    cached = _search_cache.get(ticker, {})
    if now - cached.get("hist_ts", 0) < SEARCH_HIST_TTL and cached.get("hist") is not None:
        return cached["hist"]
    hist = _fetch_yahoo_history(ticker)
    if hist is not None:
        cached["hist"]    = hist
        cached["hist_ts"] = now
        if len(_search_cache) > 100:
            oldest = min(_search_cache, key=lambda k: _search_cache[k].get("hist_ts", 0))
            _search_cache.pop(oldest, None)
        _search_cache[ticker] = cached
    return hist



@app.route("/api/monthly-returns")
def api_monthly_returns():
    sym = request.args.get("sym", "0050").upper()
    hist = _resolve_hist(sym)
    if hist is None:
        return jsonify({"error": f"{sym} 歷史資料取得失敗，請確認代號正確"}), 404

    hist_c = hist.dropna(subset=["Close"])
    monthly_last: dict = {}
    for idx, row in hist_c.iterrows():
        monthly_last[(idx.year, idx.month)] = float(row["Close"])

    sorted_months = sorted(monthly_last)
    result = []
    for i, ym in enumerate(sorted_months):
        if i == 0:
            continue
        prev = sorted_months[i - 1]
        ret = (monthly_last[ym] - monthly_last[prev]) / monthly_last[prev] * 100
        result.append({"year": ym[0], "month": ym[1], "return_pct": round(ret, 2)})

    return jsonify({"sym": sym, "data": result})


@app.route("/api/config")
def api_config():
    tw_syms    = list(TW_SYMBOLS.keys())
    us_td_syms = list(US_SYMBOLS.keys())
    us_yf_syms = list(YF_US_SYMBOLS.keys())
    return jsonify({
        "tw_symbols":    tw_syms,
        "us_td_symbols": us_td_syms,
        "us_yf_symbols": us_yf_syms,
        "all_symbols":   tw_syms + us_td_syms + us_yf_syms,
        "names":         SYMBOL_NAMES,
    })




NEWS_TTL = 900   # 15 minutes

@app.route("/api/tw-news")
def api_tw_news():
    now = time.time()
    if _cache["news"] and now - _cache["news_ts"] < NEWS_TTL:
        return jsonify(_cache["news"])

    queries = ["台股", "台灣股市 ETF 投資"]
    items: list = []
    seen: set   = set()

    for q in queries:
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={"q": q, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            root = ET.fromstring(resp.text)
            for item in root.iter("item"):
                title_raw = item.findtext("title", "").strip()
                link      = item.findtext("link",  "").strip()
                pub_str   = item.findtext("pubDate", "")
                src_el    = item.find("source")
                source    = src_el.text.strip() if src_el is not None and src_el.text else ""

                # Remove " - SourceName" suffix from title
                if source and title_raw.endswith(f" - {source}"):
                    title = title_raw[: -len(f" - {source}")].strip()
                else:
                    title = re.sub(r"\s*-\s*[^-\n]{2,40}$", "", title_raw).strip() or title_raw

                if not title or title in seen:
                    continue
                seen.add(title)

                ts = None
                try:
                    ts = int(parsedate_to_datetime(pub_str).timestamp())
                except Exception:
                    pass

                items.append({"title": title, "link": link, "source": source, "ts": ts})
        except Exception as exc:
            logger.error("News fetch error (%s): %s", q, exc)

    items.sort(key=lambda x: x.get("ts") or 0, reverse=True)
    result = items[:24]

    _cache["news"]    = result
    _cache["news_ts"] = now
    return jsonify(result)


@app.route("/api/kline")
def api_kline():
    sym    = request.args.get("sym", "0050").upper()
    period = request.args.get("period", "1m")

    range_map = {"1m": "1mo", "3m": "3mo", "6m": "6mo", "1y": "1y"}
    yf_range  = range_map.get(period, "1mo")

    is_tw  = bool(re.match(r"^\d", sym))
    ticker = f"{sym}.TW" if is_tw else sym

    data = _yahoo_get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        {"interval": "1d", "range": yf_range},
    )
    if not data:
        return jsonify({"error": f"無法取得 {sym} 資料"}), 404

    result = data.get("chart", {}).get("result", [])
    if not result:
        return jsonify({"error": f"{sym} 查無資料，請確認代號"}), 404

    r          = result[0]
    timestamps = r.get("timestamp", [])
    meta       = r.get("meta", {})
    quote      = r.get("indicators", {}).get("quote", [{}])[0]

    opens  = quote.get("open",   [])
    highs  = quote.get("high",   [])
    lows   = quote.get("low",    [])
    closes = quote.get("close",  [])
    vols   = quote.get("volume", [])

    rows = []
    for ts, o, h, lo, c, v in zip(timestamps, opens, highs, lows, closes, vols):
        if c is None:
            continue
        rows.append({
            "date":   pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d"),
            "open":   float(o)  if o  is not None else float(c),
            "high":   float(h)  if h  is not None else float(c),
            "low":    float(lo) if lo is not None else float(c),
            "close":  float(c),
            "volume": int(v)    if v  is not None else 0,
        })

    if not rows:
        return jsonify({"error": "無有效資料"}), 404

    df = pd.DataFrame(rows)
    cl = df["close"]

    df["ma5"]      = cl.rolling(5).mean()
    df["ma20"]     = cl.rolling(20).mean()
    df["ma60"]     = cl.rolling(60).mean()
    std20          = cl.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["rsi"]      = _calc_rsi(cl)

    def nv(v):
        return round(float(v), 2) if pd.notna(v) else None

    last  = df.iloc[-1]
    bb_u  = last["bb_upper"]
    bb_l  = last["bb_lower"]
    cur   = last["close"]
    bb_pct = (
        round((cur - bb_l) / (bb_u - bb_l) * 100, 1)
        if pd.notna(bb_u) and pd.notna(bb_l) and (bb_u - bb_l) > 0
        else None
    )

    return jsonify({
        "sym":   sym,
        "name":  meta.get("shortName") or meta.get("longName", ""),
        "price": nv(cur),
        "dates": df["date"].tolist(),
        "ohlcv": [
            {"x": row["date"],
             "o": round(row["open"],  2),
             "h": round(row["high"],  2),
             "l": round(row["low"],   2),
             "c": round(row["close"], 2)}
            for _, row in df.iterrows()
        ],
        "volumes":   df["volume"].tolist(),
        "ma5":       [nv(v) for v in df["ma5"]],
        "ma20":      [nv(v) for v in df["ma20"]],
        "ma60":      [nv(v) for v in df["ma60"]],
        "bb_upper":  [nv(v) for v in df["bb_upper"]],
        "bb_lower":  [nv(v) for v in df["bb_lower"]],
        "rsi":       [nv(v) for v in df["rsi"]],
        "chips": {
            "ma5":      nv(last["ma5"]),
            "ma20":     nv(last["ma20"]),
            "ma60":     nv(last["ma60"]),
            "rsi":      nv(last["rsi"]),
            "bb_upper": nv(bb_u),
            "bb_lower": nv(bb_l),
            "bb_pct":   bb_pct,
        },
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
