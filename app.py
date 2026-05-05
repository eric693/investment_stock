import os
import time
import logging
import threading
from datetime import datetime
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
SYMBOL_NAMES = {
    "0050":   "元大台灣50",
    "QQQ":    "Invesco QQQ ETF",
    "00631L": "元大台灣50正2",
    "QLD":    "ProShares Ultra QQQ",
    "00662":  "富邦NASDAQ",
    "00865B": "國泰US短期公債",
}

# ─── Cache ────────────────────────────────────────────────────────────────────
_cache = {
    "stocks":      {},
    "histories":   {},
    "hist_ts":     0.0,
    "price_ts":    0.0,
    "refreshing":  False,
    "vix":         None,
    "foreign_net": None,
}
_refresh_lock = threading.Lock()
alert_history: list = []
_line_users: dict = {}


# ─── Technical Indicators ─────────────────────────────────────────────────────
def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


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
    peaks = []
    n = len(series)
    for i in range(order, n - order):
        window = series.iloc[i - order: i + order + 1]
        if (series.iloc[i] == window.max()
                and series.iloc[i] > series.iloc[i - 1]
                and series.iloc[i] > series.iloc[i + 1]):
            peaks.append(i)
    return peaks


def _find_troughs(series: pd.Series, order: int = 7) -> list[int]:
    """Local minima indices (0-based)."""
    troughs = []
    n = len(series)
    for i in range(order, n - order):
        window = series.iloc[i - order: i + order + 1]
        if (series.iloc[i] == window.min()
                and series.iloc[i] < series.iloc[i - 1]
                and series.iloc[i] < series.iloc[i + 1]):
            troughs.append(i)
    return troughs


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
        parts.append(f"tse_{s.lower()}.tw")
        parts.append(f"otc_{s.lower()}.tw")
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
            result[code] = {
                "price":          price,
                "prev_close":     prev,
                "daily_change":   (price - prev) / prev * 100,
                "is_market_open": True,
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


# ─── US stock helpers (Twelve Data) ──────────────────────────────────────────
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
        resp = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"interval": "1d", "range": "2y"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        data   = resp.json()
        result = data.get("chart", {}).get("result", [])
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
        resp = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        data   = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        meta  = result[0].get("meta", {})
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        prev  = meta.get("previousClose") or price
        if not price:
            return None
        return {
            "price":          float(price),
            "prev_close":     float(prev),
            "daily_change":   (float(price) - float(prev)) / float(prev) * 100 if prev else 0.0,
            "is_market_open": meta.get("marketState") == "REGULAR",
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
        resp = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
            params={"interval": "1d", "range": "10d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        data   = resp.json()
        result = data.get("chart", {}).get("result", [])
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
                net = 0
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
        "k_val":      round(k_val, 2) if k_val is not None else None,
        "d_val":      round(d_val, 2) if d_val is not None else None,
        "divergence": divergence,
        "chart":      chart,
    }


# ─── Refresh logic ────────────────────────────────────────────────────────────
def _do_refresh(refresh_hist: bool, refresh_price: bool):
    if refresh_hist:
        logger.info("Refreshing TW histories (Yahoo Finance first)…")
        for name, av_sym in TW_SYMBOLS.items():
            hist = _fetch_yahoo_history(av_sym)
            if hist is None:
                logger.warning("Yahoo history failed for %s, trying TWSE…", name)
                stock_no = av_sym.replace(".TW", "")
                hist = _fetch_twse_history(stock_no)
            if hist is not None:
                _cache["histories"][name] = hist

        logger.info("Refreshing US histories (Twelve Data)…")
        for i, (name, (sym, exch)) in enumerate(US_SYMBOLS.items()):
            if i > 0:
                time.sleep(8)
            hist = _fetch_td_history(sym, exch)
            if hist is not None:
                _cache["histories"][name] = hist

        logger.info("Refreshing VIX and 外資買賣超…")
        vix = _fetch_vix()
        if vix:
            _cache["vix"] = vix
        fn = _fetch_foreign_net()
        if fn:
            _cache["foreign_net"] = fn

        _cache["hist_ts"] = time.time()

    if refresh_price:
        logger.info("Refreshing TW prices (TWSE)…")
        tw_quotes = _fetch_tw_quotes()
        for name, av_sym in TW_SYMBOLS.items():
            quote = tw_quotes.get(name)
            if quote is None:
                logger.warning("TWSE quote missing for %s, trying Yahoo Finance…", name)
                quote = _fetch_yahoo_quote(av_sym)
            hist = _cache["histories"].get(name)
            if quote and hist is not None:
                _cache["stocks"][name] = _build_stock_entry(name, hist, quote)

        logger.info("Refreshing US prices (Twelve Data)…")
        for i, (name, (sym, exch)) in enumerate(US_SYMBOLS.items()):
            if i > 0:
                time.sleep(8)
            quote = _fetch_td_quote(sym, exch)
            hist  = _cache["histories"].get(name)
            if quote and hist is not None:
                _cache["stocks"][name] = _build_stock_entry(name, hist, quote)

        # Refresh VIX on every price cycle too (fast, no auth needed)
        vix = _fetch_vix()
        if vix:
            _cache["vix"] = vix

        _cache["price_ts"] = time.time()


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

    if (not _cache["stocks"] or need_hist or need_price) and not _cache["refreshing"]:
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
def compute_sop(stocks: dict) -> dict:
    tw  = stocks.get("0050", {})
    qqq = stocks.get("QQQ",  {})

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


def check_alerts(stocks: dict, sop: dict) -> list[dict]:
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
    if fn and fn.get("foreign") is not None and fn["foreign"] < -30_000_000_000:
        sold_yi = abs(fn["foreign"]) / 1e8
        alerts.append({"type": "CRITICAL", "code": "FN", "ts": now_str,
            "title": "外資大幅賣超",
            "msg": f"外資今日賣超約 {sold_yi:.0f} 億元，搭配年線訊號判斷是否撤退"})

    return alerts


# ─── Notification ─────────────────────────────────────────────────────────────
def send_notification(message: str):
    if not LINE_TOKEN or not LINE_USER_IDS:
        return
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

        for a in alerts:
            if a not in alert_history:
                alert_history.insert(0, a)
        while len(alert_history) > 50:
            alert_history.pop()

        return jsonify({
            "stocks":        stocks,
            "sop":           sop,
            "alerts":        alerts,
            "alert_history": alert_history[:20],
            "updated_at":    datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "price_age":     int(time.time() - _cache["price_ts"]),
            "hist_age":      int(time.time() - _cache["hist_ts"]),
            "vix":           _cache.get("vix"),
            "foreign_net":   _cache.get("foreign_net"),
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
