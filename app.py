import os
import time
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────
LINE_TOKEN   = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")
TW_TZ = pytz.timezone("Asia/Taipei")
CACHE_DURATION = int(os.environ.get("CACHE_SECONDS", "300"))   # 5-min cache

SYMBOLS = {
    "0050":    "0050.TW",
    "QQQ":     "QQQ",
    "00631L":  "00631L.TW",   # 台灣50正2
    "QLD":     "QLD",          # QQQ 2x
}
SYMBOL_NAMES = {
    "0050":   "元大台灣50",
    "QQQ":    "Invesco QQQ ETF",
    "00631L": "元大台灣50正2",
    "QLD":    "ProShares Ultra QQQ",
}

# ─── In-memory cache ─────────────────────────────────────────────────────────
_cache: dict = {"data": None, "ts": 0.0}
alert_history: list = []


# ─── Data helpers ─────────────────────────────────────────────────────────────
def _fetch(symbol: str) -> pd.DataFrame | None:
    """Fetch 2 years of OHLCV and compute MA50/MA200."""
    try:
        hist = yf.Ticker(symbol).history(period="2y")
        if hist.empty:
            return None
        hist["MA200"] = hist["Close"].rolling(200).mean()
        hist["MA50"]  = hist["Close"].rolling(50).mean()
        return hist
    except Exception as exc:
        logger.error("Fetch error %s: %s", symbol, exc)
        return None


def _build_stock_entry(name: str, hist: pd.DataFrame) -> dict:
    close   = hist["Close"]
    current = float(close.iloc[-1])
    prev    = float(close.iloc[-2])
    ma200   = float(hist["MA200"].iloc[-1]) if not pd.isna(hist["MA200"].iloc[-1]) else None
    ma50    = float(hist["MA50"].iloc[-1])  if not pd.isna(hist["MA50"].iloc[-1])  else None

    daily_chg       = (current - prev) / prev * 100
    pct_from_ma200  = (current - ma200) / ma200 * 100 if ma200 else None

    # Last 3 sessions vs MA200
    last3: list[float] = []
    for i in range(-3, 0):
        try:
            p = float(close.iloc[i])
            m = float(hist["MA200"].iloc[i])
            if not np.isnan(m):
                last3.append((p - m) / m * 100)
        except Exception:
            pass

    # Chart: last ~252 trading days
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


def get_dashboard_data() -> dict:
    stocks = {}
    for name, sym in SYMBOLS.items():
        hist = _fetch(sym)
        if hist is not None and len(hist) >= 5:
            stocks[name] = _build_stock_entry(name, hist)
    return stocks


def cached_data() -> dict:
    now = time.time()
    if _cache["data"] is None or (now - _cache["ts"]) > CACHE_DURATION:
        logger.info("Refreshing stock data cache…")
        _cache["data"] = get_dashboard_data()
        _cache["ts"]   = now
    return _cache["data"]


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

    # ─ SOP 04 smile levels ─
    smile_levels = []
    for t in [-8, -10, -15, -20, -25, -30]:
        tp = ma200 * (1 + t / 100) if ma200 else 0
        smile_levels.append({
            "threshold": t,
            "price":     round(tp, 2),
            "triggered": price <= tp if tp else False,
        })

    # ─ Status helpers ─
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
            "qqq_pct":   round(pqqq, 2),
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
    if LINE_TOKEN and LINE_USER_ID:
        try:
            r = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {LINE_TOKEN}",
                },
                json={"to": LINE_USER_ID, "messages": [{"type": "text", "text": message}]},
                timeout=10,
            )
            logger.info("LINE push: %s", r.status_code)
        except Exception as e:
            logger.error("LINE error: %s", e)


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

        # Persist new critical alerts
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
            "cache_age":     int(time.time() - _cache["ts"]),
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
    """Force-clear cache and reload data."""
    _cache["ts"] = 0.0
    cached_data()
    return jsonify({"ok": True})


@app.route("/webhook", methods=["POST"])
def line_webhook():
    """LINE Bot webhook endpoint (register in LINE Developers console)."""
    body      = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature", "")
    # Minimal echo: reply with SOP status when user messages
    logger.info("LINE webhook received")
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
