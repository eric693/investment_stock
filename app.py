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
LINE_TOKEN  = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_IDS = [uid.strip() for uid in os.environ.get("LINE_USER_ID", "").split(",") if uid.strip()]
TD_API_KEY  = os.environ.get("TWELVEDATA_API_KEY", "")    # US stocks (free plan)
AV_API_KEY  = os.environ.get("ALPHAVANTAGE_API_KEY", "")  # TW stock history
TW_TZ = pytz.timezone("Asia/Taipei")

# Price: every 1h  | History (MA): every 12h
PRICE_TTL = int(os.environ.get("PRICE_CACHE_SECONDS", "3600"))
HIST_TTL  = int(os.environ.get("HIST_CACHE_SECONDS",  "43200"))

TD_BASE = "https://api.twelvedata.com"
AV_BASE = "https://www.alphavantage.co/query"

# Taiwan stocks — TWSE real-time price + Alpha Vantage daily history
TW_SYMBOLS = {
    "0050":   "0050.TW",
    "00631L": "00631L.TW",
    "00662":  "00662.TW",
    "00865B": "00865B.TW",
}
# US stocks — Twelve Data (free plan supports US exchanges)
US_SYMBOLS = {
    "QQQ": ("QQQ", "NASDAQ"),
    "QLD": ("QLD", ""),       # let TD auto-detect exchange
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
    "stocks":     {},
    "histories":  {},
    "hist_ts":    0.0,
    "price_ts":   0.0,
    "refreshing": False,
}
alert_history: list = []
_line_users: dict = {}   # uid -> {last_msg, last_seen}


# ─── Taiwan stock helpers (TWSE + Alpha Vantage) ──────────────────────────────
def _fetch_tw_quotes() -> dict:
    """Batch-fetch all TW real-time prices from TWSE MIS API (no auth needed).
    Sends both tse_ and otc_ prefixes so ETFs on either board are found."""
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
                price = prev  # market closed — use prev close
            result[code] = {
                "price":          price,
                "prev_close":     prev,
                "daily_change":   (price - prev) / prev * 100,
                "is_market_open": z not in ("-", ""),
            }
        logger.info("TWSE quotes fetched: %s", list(result.keys()))
        return result
    except Exception as exc:
        logger.error("TWSE quote error: %s", exc)
        return {}


def _fetch_twse_history(stock_no: str, months: int = 14) -> pd.DataFrame | None:
    """Fetch TW stock history from TWSE official monthly API (free, no auth)."""
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
                parts = row[0].split("/")
                rows.append({
                    "date":  f"{int(parts[0]) + 1911}-{parts[1]}-{parts[2]}",
                    "Close": float(close_str),
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
    df["MA200"]  = df["Close"].rolling(200).mean()
    df["MA50"]   = df["Close"].rolling(50).mean()
    logger.info("TD history fetched %s (%d rows)", sym, len(df))
    return df


def _fetch_yahoo_history(ticker: str) -> pd.DataFrame | None:
    """Fallback: fetch daily history from Yahoo Finance (no auth needed)."""
    try:
        resp = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"interval": "1d", "range": "2y"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            logger.warning("Yahoo history empty for %s", ticker)
            return None
        r = result[0]
        timestamps = r.get("timestamp", [])
        closes = r.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        rows = [
            {"date": pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d"), "Close": float(c)}
            for ts, c in zip(timestamps, closes) if c is not None
        ]
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
    """Fallback: fetch real-time quote from Yahoo Finance."""
    try:
        resp = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        meta = result[0].get("meta", {})
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


# ─── Build entry (same regardless of data source) ────────────────────────────
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
        "chart":          chart,
    }


# ─── Refresh logic ────────────────────────────────────────────────────────────
def _do_refresh(refresh_hist: bool, refresh_price: bool):
    if refresh_hist:
        logger.info("Refreshing TW histories (TWSE monthly API)…")
        for name, av_sym in TW_SYMBOLS.items():
            stock_no = av_sym.replace(".TW", "")
            hist = _fetch_twse_history(stock_no)
            if hist is None:
                logger.warning("TWSE history failed for %s, trying Yahoo Finance…", name)
                hist = _fetch_yahoo_history(av_sym)
            if hist is not None:
                _cache["histories"][name] = hist

        logger.info("Refreshing US histories (Twelve Data)…")
        for i, (name, (sym, exch)) in enumerate(US_SYMBOLS.items()):
            if i > 0:
                time.sleep(8)   # TD free: 8 calls/min
            hist = _fetch_td_history(sym, exch)
            if hist is not None:
                _cache["histories"][name] = hist

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

        _cache["price_ts"] = time.time()


def _background_refresh(hist: bool = False, price: bool = True):
    if _cache["refreshing"]:
        return
    _cache["refreshing"] = True
    try:
        _do_refresh(refresh_hist=hist, refresh_price=price)
        logger.info("Refresh done (hist=%s price=%s).", hist, price)
    except Exception as exc:
        logger.error("Refresh error: %s", exc)
    finally:
        _cache["refreshing"] = False


def cached_data() -> dict:
    now        = time.time()
    need_hist  = not _cache["histories"] or (now - _cache["hist_ts"])  > HIST_TTL
    need_price = not _cache["stocks"]    or (now - _cache["price_ts"]) > PRICE_TTL

    if not _cache["stocks"] and not _cache["refreshing"]:
        logger.info("Cold start…")
        threading.Thread(
            target=_background_refresh,
            kwargs={"hist": True, "price": True},
            daemon=True,
        ).start()
    elif (need_hist or need_price) and not _cache["refreshing"]:
        threading.Thread(
            target=_background_refresh,
            kwargs={"hist": need_hist, "price": need_price},
            daemon=True,
        ).start()

    return _cache["stocks"]


# ─── SOP logic ────────────────────────────────────────────────────────────────
def compute_sop(stocks: dict) -> dict:
    tw   = stocks.get("0050", {})
    qqq  = stocks.get("QQQ",  {})

    p0050 = tw.get("pct_from_ma200") or 0.0
    pqqq  = qqq.get("pct_from_ma200") or 0.0
    d0050 = tw.get("daily_change") or 0.0
    l3    = tw.get("last3_vs_ma", [])
    above = tw.get("above_ma200", True)
    ma200 = tw.get("ma200") or 0.0
    price = tw.get("price") or 0.0

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
        if p0050 <= -3:
            return "watch"
        return "normal"

    def s04():
        return "active" if not above else "standby"

    def s05():
        if (len(l3) >= 3 and all(p >= 3 for p in l3)) or d0050 >= 5:
            return "alert"
        if p0050 >= 3 and above:
            return "watch"
        return "standby"

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
            "desc_val":  f"0050 日漲跌：{d0050:+.2f}%",
        },
        "sop04": {
            "name": "微笑佈局", "status": s04(),
            "pct_from_ma":  round(p0050, 2),
            "smile_levels": smile_levels,
            "desc_rule": "線下只買原型；左側各5%，右側各2%；-30%冬眠",
            "desc_val":  f"0050 年線偏離：{p0050:+.2f}%",
        },
        "sop05": {
            "name": "反攻號角", "status": s05(),
            "pct_from_ma":  round(p0050, 2),
            "daily_change": round(d0050, 2),
            "desc_rule": "站回年線+3%連3天 或 單日+5% → 全數壓正2",
            "desc_val":  f"0050 年線偏離：{p0050:+.2f}%",
        },
    }


def check_alerts(stocks: dict, sop: dict) -> list[dict]:
    alerts = []
    now_str = datetime.now(TW_TZ).isoformat()
    tw  = stocks.get("0050", {})
    qqq = stocks.get("QQQ",  {})

    pqqq  = qqq.get("pct_from_ma200") or 0.0
    p0050 = tw.get("pct_from_ma200")  or 0.0
    d0050 = tw.get("daily_change")    or 0.0
    l3    = tw.get("last3_vs_ma", [])

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

    for lvl in sop.get("sop04", {}).get("smile_levels", []):
        if lvl["triggered"] and abs(p0050 - lvl["threshold"]) < 0.8:
            alerts.append({"type": "BUY", "code": "04", "ts": now_str,
                "title": f"微笑佈局 {lvl['threshold']}% 買點",
                "msg": f"0050 接近年線 {lvl['threshold']}% 位置（現 {p0050:.1f}%），左側佈局 5%"})

    if len(l3) >= 3 and all(p >= 3 for p in l3):
        alerts.append({"type": "BUY", "code": "05", "ts": now_str,
            "title": "反攻號角（連續3日）",
            "msg": "0050 連續3天站回年線+3%，底部原型 + 所有備戰現金全數壓正2！"})

    if d0050 >= 5:
        alerts.append({"type": "BUY", "code": "05", "ts": now_str,
            "title": "反攻號角（單日大漲）",
            "msg": f"0050 單日漲幅 {d0050:.1f}%（閾值 +5%），全數壓正2！"})

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
    body = request.get_json(silent=True) or {}
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
        "token_configured": bool(LINE_TOKEN),
        "push_users_configured": LINE_USER_IDS,
        "seen_users": _line_users,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
