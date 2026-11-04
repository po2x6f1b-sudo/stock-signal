#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株価 売買シグナル通知スクリプト（日足トレンド × 5分足タイミング）
- データ取得: yfinance（日本株は「7203.T」のように末尾に .T）
- 指標: MACD(12,26,9) / RSI(14, Wilder)  ※pandasのみで自前計算（依存が少なく安定）
- 通知: Discord Webhook / LINE Messaging API（どちらか or 両方）
- 自動発注はしません。投資助言ではありません。

使い方:
  python stock_signal_notifier.py --test          # 通知テスト
  python stock_signal_notifier.py                 # 1回だけ判定（cron向け）
  python stock_signal_notifier.py --loop          # 5分ごとに繰り返し（Colab向け）
  python stock_signal_notifier.py --notify-wait   # 「様子見」も通知
"""
import os
import re
import sys
import json
import time
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ========================= 設定 =========================
WATCHLIST = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーG",
    "9984.T": "ソフトバンクG",
}

# 通知先（環境変数 or ここに直接記入）
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
LINE_CHANNEL_TOKEN = os.getenv("LINE_CHANNEL_TOKEN", "")   # LINE Messaging API のチャネルアクセストークン
LINE_USER_ID = os.getenv("LINE_USER_ID", "")               # 送信先のユーザーID（Uから始まる文字列）

# テクニカル設定
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_PERIOD = 14
CROSS_WINDOW = 2          # 直近何本以内のクロスを「発生」とみなすか（5分足）
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STATE_FILE = "signal_state.json"   # 同じシグナルの連続通知を防ぐ
JST = ZoneInfo("Asia/Tokyo")
# ========================================================


# ---------- 指標計算 ----------
def calc_macd(close: pd.Series):
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd, signal, macd - signal


def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


# ---------- データ取得 ----------
def fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: {interval} のデータが取得できません")
    df = df.dropna(subset=["Close"])
    df.index = df.index.tz_convert(JST) if df.index.tz is not None else df.index.tz_localize(JST)
    return df


# ---------- 分析 ----------
def analyze_daily(df: pd.DataFrame) -> dict:
    c = df["Close"]
    macd, sig, hist = calc_macd(c)
    rsi = calc_rsi(c)
    m, s, h, hp = macd.iloc[-1], sig.iloc[-1], hist.iloc[-1], hist.iloc[-2]
    if m > s and h > hp:
        trend = "up"
    elif m < s and h < hp:
        trend = "down"
    else:
        trend = "flat"
    return {
        "trend": trend,
        "macd_ok": m > s,
        "rsi": float(rsi.iloc[-1]),
    }


def find_cross(macd: pd.Series, sig: pd.Series, window: int):
    """直近 window 本以内の最新クロスを返す: (種別, 何本前, 時刻)"""
    diff = macd - sig
    n = len(diff)
    for k in range(window):
        i = n - 1 - k
        if i < 1:
            break
        cur, prev = diff.iloc[i], diff.iloc[i - 1]
        if pd.isna(cur) or pd.isna(prev):
            continue
        if prev <= 0 < cur:
            return "golden", k, diff.index[i]
        if prev >= 0 > cur:
            return "dead", k, diff.index[i]
    return None, None, None


def analyze_intraday(df: pd.DataFrame) -> dict:
    # 形成途中の最新足は除外（だまし防止）
    now = datetime.now(JST)
    if len(df) > 1 and df.index[-1] + pd.Timedelta(minutes=5) > now:
        df = df.iloc[:-1]
    c = df["Close"]
    macd, sig, hist = calc_macd(c)
    rsi = calc_rsi(c)
    cross, bars_ago, cross_time = find_cross(macd, sig, CROSS_WINDOW)
    recent = rsi.iloc[-12:]  # 直近1時間
    rsi_now = float(rsi.iloc[-1])
    rsi_min, rsi_max = float(recent.min()), float(recent.max())

    if rsi_now >= RSI_OVERBOUGHT:
        rsi_state = "過熱圏（買われすぎ）"
    elif rsi_now <= RSI_OVERSOLD:
        rsi_state = "売られすぎ圏"
    elif rsi_min <= 35 and rsi_now > rsi_min:
        rsi_state = "底値圏からの反発"
    elif rsi_max >= 65 and rsi_now < rsi_max:
        rsi_state = "高値圏からの反落"
    else:
        rsi_state = "中立"

    return {
        "cross": cross, "bars_ago": bars_ago, "cross_time": cross_time,
        "macd_above": bool(macd.iloc[-1] > sig.iloc[-1]),
        "rsi": rsi_now, "rsi_min": rsi_min, "rsi_max": rsi_max, "rsi_state": rsi_state,
        "last_close": float(c.iloc[-1]), "last_time": df.index[-1],
        "recent_high": float(df["High"].iloc[-12:].max()),
        "recent_low": float(df["Low"].iloc[-12:].min()),
    }


def decide(d: dict, i: dict) -> dict:
    """日足トレンド × 5分足タイミング → BUY / SELL / WAIT"""
    signal, strength, reason = "WAIT", 0, ""

    if d["trend"] == "up" and i["cross"] == "golden":
        if i["rsi"] >= RSI_OVERBOUGHT:
            reason = "買いクロスだがRSI過熱のため見送り"
        else:
            signal = "BUY"
            strength = 3 if i["rsi_min"] <= 35 else (2 if i["rsi"] < 50 else 1)
    elif d["trend"] == "down" and i["cross"] == "dead":
        if i["rsi"] <= RSI_OVERSOLD:
            reason = "売りクロスだがRSI売られすぎのため見送り"
        else:
            signal = "SELL"
            strength = 3 if i["rsi_max"] >= 65 else (2 if i["rsi"] > 50 else 1)
    elif i["cross"] == "golden" and d["trend"] != "up":
        reason = "5分足は買いクロスだが日足が追随せず（逆張りは見送り）"
    elif i["cross"] == "dead" and d["trend"] != "down":
        reason = "5分足は売りクロスだが日足が追随せず（逆張りは見送り）"
    else:
        reason = "日足と5分足の条件が揃っていません"

    return {"signal": signal, "strength": strength, "reason": reason}


# ---------- メッセージ生成 ----------
def build_message(ticker: str, name: str, d: dict, i: dict, r: dict, markdown: bool = True) -> str:
    code = ticker.replace(".T", "")
    now_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    b = (lambda s: f"**{s}**") if markdown else (lambda s: s)

    head = {
        "BUY":  f"🟢🟢🟢【買い】{code} {name}",
        "SELL": f"🔴🔴🔴【売り】{code} {name}",
        "WAIT": f"⚪【様子見】{code} {name}",
    }[r["signal"]]
    verdict = {
        "BUY":  f"🌟 {b('【強気：買いシグナル発動】')} 🌟",
        "SELL": f"⚠️ {b('【弱気：売りシグナル発動】')} ⚠️",
        "WAIT": f"😴 {b('【様子見：待機】')}",
    }[r["signal"]]
    stars = "★" * r["strength"] + "☆" * (3 - r["strength"])

    trend_txt = {
        "up": "上昇トレンド 📈", "down": "下降トレンド 📉", "flat": "方向感なし ➡️",
    }[d["trend"]]
    macd_txt = "MACD良好" if d["macd_ok"] else "MACD弱含み"

    if i["cross"]:
        kind = "ゴールデンクロス発生 🚀" if i["cross"] == "golden" else "デッドクロス発生 📉"
        ago = f"（{i['cross_time'].strftime('%H:%M')} / 約{i['bars_ago'] * 5}分前）"
        cross_txt = f"{kind} {ago}"
    else:
        cross_txt = "クロスなし（" + ("MACD>シグナル" if i["macd_above"] else "MACD<シグナル") + "）"

    action = {
        "BUY":  f"直近高値（{i['recent_high']:,.0f}円）抜けでの打診買いを検討 / 損切り目安: 直近安値 {i['recent_low']:,.0f}円割れ",
        "SELL": f"直近安値（{i['recent_low']:,.0f}円）割れでの売り・利確を検討 / 損切り目安: 直近高値 {i['recent_high']:,.0f}円超え",
        "WAIT": f"ポジションは見送り。{r['reason']}",
    }[r["signal"]]

    lines = [
        head,
        "━━━━━━━━━━━━━━",
        "🚨 【株価売買シグナル検知】 🚨",
        f"銘柄コード: {code} ({name})",
        f"判定時間: {now_str}　現在値: {i['last_close']:,.0f}円",
        "",
        f"👉 【総合判断】：{verdict}",
    ]
    if r["signal"] != "WAIT":
        lines.append(f"シグナル強度: {stars}")
    lines += [
        "",
        "📊 【テクニカル詳細】",
        f"・日足トレンド: {trend_txt} ({macd_txt} / RSI: {d['rsi']:.0f})",
        f"・5分足状態: {cross_txt}",
        f"・5分足RSI: {i['rsi']:.0f} ({i['rsi_state']})",
        "",
        f"💡 アクション案: {action}",
        "━━━━━━━━━━━━━━",
        "※自動売買ではありません。最終判断はご自身で。",
    ]
    return "\n".join(lines)


# ---------- 通知 ----------
def send_discord(text: str):
    if not DISCORD_WEBHOOK_URL:
        return
    res = requests.post(DISCORD_WEBHOOK_URL, json={"content": text}, timeout=15)
    res.raise_for_status()


def send_line(text: str):
    if not (LINE_CHANNEL_TOKEN and LINE_USER_ID):
        return
    text = text.replace("**", "")  # LINEはMarkdown非対応
    res = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {LINE_CHANNEL_TOKEN}"},
        json={"to": LINE_USER_ID, "messages": [{"type": "text", "text": text[:5000]}]},
        timeout=15,
    )
    res.raise_for_status()


def notify(text: str):
    sent = False
    for fn in (send_discord, send_line):
        try:
            fn(text)
            sent = True
        except Exception as e:
            print(f"通知エラー({fn.__name__}): {e}")
    if not (DISCORD_WEBHOOK_URL or (LINE_CHANNEL_TOKEN and LINE_USER_ID)):
        print("⚠ 通知先が未設定です。内容を画面に表示します。\n")
        print(text)


# ---------- 状態管理（重複通知防止） ----------
def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:
        print(f"状態保存エラー: {e}")


# ---------- メイン処理 ----------
def run_once(notify_wait: bool = False):
    state = load_state()
    for ticker, name in WATCHLIST.items():
        try:
            daily = fetch(ticker, "1d", "6mo")
            intra = fetch(ticker, "5m", "5d")
            d = analyze_daily(daily)
            i = analyze_intraday(intra)
            r = decide(d, i)
            key = f"{r['signal']}:{i['cross_time']}"
            print(f"[{datetime.now(JST):%H:%M}] {ticker} -> {r['signal']} (強度{r['strength']}) {r['reason']}")

            is_new = state.get(ticker) != key
            if (r["signal"] != "WAIT" and is_new) or (r["signal"] == "WAIT" and notify_wait):
                notify(build_message(ticker, name, d, i, r))
                state[ticker] = key
        except Exception as e:
            print(f"{ticker}: エラー {e}")
    save_state(state)


def is_market_open() -> bool:
    now = datetime.now(JST)
    if now.weekday() >= 5:
        return False
    return (9, 0) <= (now.hour, now.minute) <= (15, 30)


def send_test():
    now = datetime.now(JST)
    idx = pd.date_range(end=now, periods=3, freq="5min")
    d = {"trend": "up", "macd_ok": True, "rsi": 58.0}
    i = {"cross": "golden", "bars_ago": 0, "cross_time": idx[-1], "macd_above": True,
         "rsi": 38.0, "rsi_min": 31.0, "rsi_max": 45.0, "rsi_state": "底値圏からの反発",
         "last_close": 3000.0, "last_time": idx[-1], "recent_high": 3020.0, "recent_low": 2985.0}
    r = {"signal": "BUY", "strength": 3, "reason": ""}
    notify("【テスト通知】\n" + build_message("0000.T", "テスト銘柄", d, i, r))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="通知テスト")
    p.add_argument("--loop", action="store_true", help="5分ごとに繰り返し実行")
    p.add_argument("--interval", type=int, default=300, help="ループ間隔（秒）")
    p.add_argument("--notify-wait", action="store_true", help="様子見も通知")
    p.add_argument("--force", action="store_true", help="場外時間でも実行")
    a = p.parse_args(sys.argv[1:] if "ipykernel" not in sys.modules else [])

    if a.test:
        send_test()
    elif a.loop:
        while True:
            if a.force or is_market_open():
                run_once(a.notify_wait)
            else:
                print(f"[{datetime.now(JST):%H:%M}] 場外のため待機中")
            time.sleep(a.interval)
    else:
        if a.force or is_market_open():
            run_once(a.notify_wait)
        else:
            print("場外時間です（--force で強制実行）")
