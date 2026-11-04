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

# ---- デイトレ向け自動スクリーニング ----
USE_SCREENING = True       # True: 下のUNIVERSEから毎回「今日動いている銘柄」を自動選定
TOP_N = 10                 # 自動選定で監視する銘柄数
MIN_STRENGTH = 2           # ★2・★3だけ通知（★1は通知しない。通知数の節約）
MIN_PRICE, MAX_PRICE = 200, 3000    # 株価の範囲（円）
SCREEN_INTERVAL_SEC = 600  # 候補銘柄の再選定間隔（秒）。判定自体は毎回行う
UNIVERSE = {
    "7203.T": "トヨタ", "7267.T": "ホンダ", "7201.T": "日産自動車", "7261.T": "マツダ",
    "6758.T": "ソニーG", "6501.T": "日立", "6503.T": "三菱電機", "6723.T": "ルネサス",
    "6762.T": "TDK", "6981.T": "村田製作所", "6594.T": "ニデック", "6857.T": "アドバンテスト",
    "8035.T": "東京エレクトロン", "6920.T": "レーザーテック", "6526.T": "ソシオネクスト",
    "9984.T": "ソフトバンクG", "9432.T": "NTT", "9433.T": "KDDI", "9434.T": "ソフトバンク",
    "4755.T": "楽天G", "7974.T": "任天堂", "8136.T": "サンリオ",
    "8306.T": "三菱UFJ", "8316.T": "三井住友FG", "8411.T": "みずほFG",
    "8058.T": "三菱商事", "8001.T": "伊藤忠", "8031.T": "三井物産", "8053.T": "住友商事",
    "5401.T": "日本製鉄", "5020.T": "ENEOS", "1605.T": "INPEX",
    "7011.T": "三菱重工", "7012.T": "川崎重工", "7013.T": "IHI",
    "5801.T": "古河電工", "5802.T": "住友電工", "5803.T": "フジクラ",
    "9101.T": "日本郵船", "9104.T": "商船三井", "9107.T": "川崎汽船",
    "4502.T": "武田薬品", "4568.T": "第一三共", "9501.T": "東京電力HD",
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


# ---------- デイトレ候補の自動選定 ----------
def trading_fraction(now=None) -> float:
    """今日の取引時間（9:00-11:30, 12:30-15:30 = 330分）のうち経過した割合"""
    now = now or datetime.now(JST)
    m = now.hour * 60 + now.minute
    start, lunch_s, lunch_e, end = 540, 690, 750, 930
    if m <= start:
        return 0.0
    elapsed = min(m, lunch_s) - start
    if m > lunch_e:
        elapsed += min(m, end) - lunch_e
    return min(elapsed / 330, 1.0)


def pick_candidates():
    """出来高の急増・売買代金・値幅の3つで順位付けし、上位TOP_N銘柄を返す。
    戻り値: ({ticker: 名前}, {ticker: 日足DataFrame})"""
    tickers = list(UNIVERSE)
    raw = yf.download(tickers, period="6mo", interval="1d", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    frac = trading_fraction()
    today = datetime.now(JST).date()
    rows, dailies = [], {}
    for t in tickers:
        try:
            df = raw[t].dropna(subset=["Close"])
            if len(df) < 40:
                continue
            if not (MIN_PRICE <= float(df["Close"].iloc[-1]) <= MAX_PRICE):
                continue
            live = df.index[-1].date() == today
            if live and frac >= 0.1:        # 今日の途中経過を、経過時間で補正して比較
                bar, hist, f = df.iloc[-1], df.iloc[:-1], frac
            elif live:                      # 寄付き直後はデータが少ないので前日で判定
                bar, hist, f = df.iloc[-2], df.iloc[:-2], 1.0
            else:
                bar, hist, f = df.iloc[-1], df.iloc[:-1], 1.0
            avg_vol = float(hist["Volume"].tail(20).mean())
            if avg_vol <= 0:
                continue
            prev_close = float(hist["Close"].iloc[-1])
            rows.append({
                "t": t,
                "vol_ratio": float(bar["Volume"]) / (avg_vol * f),
                "turnover": float(bar["Close"]) * float(bar["Volume"]) / f,
                "range_pct": (float(bar["High"]) - float(bar["Low"])) / prev_close * 100,
            })
            dailies[t] = df
        except Exception:
            continue
    if not rows:
        raise RuntimeError("スクリーニング用データが取得できません")
    r = pd.DataFrame(rows).set_index("t")
    score = sum(r[c].rank(pct=True) for c in ("vol_ratio", "turnover", "range_pct"))
    top = score.sort_values(ascending=False).head(TOP_N).index
    print("本日の候補: " + ", ".join(f"{t.replace('.T', '')}{UNIVERSE[t]}" for t in top))
    return {t: UNIVERSE[t] for t in top}, {t: dailies[t] for t in top}


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


# ---------- 候補キャッシュ・5分足まとめ取得（1分間隔でも通信を少なくする） ----------
_SCREEN = {"at": 0.0, "picked": {}, "dailies": {}}


def get_targets():
    targets = dict(WATCHLIST)
    if USE_SCREENING:
        if time.time() - _SCREEN["at"] >= SCREEN_INTERVAL_SEC:
            try:
                _SCREEN["picked"], _SCREEN["dailies"] = pick_candidates()
                _SCREEN["at"] = time.time()
            except Exception as e:
                print(f"スクリーニング失敗（固定銘柄＋前回の候補で継続）: {e}")
                _SCREEN["at"] = time.time() - SCREEN_INTERVAL_SEC + 60   # 1分後に再試行
        targets.update(_SCREEN["picked"])
    return targets, _SCREEN["dailies"]


def fetch_intraday_batch(tickers) -> dict:
    out = {}
    try:
        raw = yf.download(list(tickers), period="5d", interval="5m", group_by="ticker",
                          auto_adjust=False, progress=False, threads=True)
    except Exception as e:
        print(f"5分足の一括取得に失敗: {e}")
        return out
    for t in tickers:
        try:
            df = raw[t].dropna(subset=["Close"])
            if df.empty:
                continue
            df.index = df.index.tz_convert(JST) if df.index.tz is not None else df.index.tz_localize(JST)
            out[t] = df
        except Exception:
            continue
    return out


# ---------- メイン処理 ----------
def run_once(notify_wait: bool = False):
    state = load_state()
    targets, dailies = get_targets()
    intras = fetch_intraday_batch(list(targets))
    if intras:   # データの遅れを確認するためのログ
        lt = max(df.index[-1] for df in intras.values())
        lag = (datetime.now(JST) - lt.to_pydatetime()).total_seconds() / 60
        print(f"[{datetime.now(JST):%H:%M}] 最新の5分足は{lt:%H:%M}開始（{lag:.0f}分前。5分以内なら遅れなし）")
    for ticker, name in targets.items():
        try:
            daily = dailies.get(ticker)
            if daily is None:
                daily = fetch(ticker, "1d", "6mo")
            intra = intras.get(ticker)
            if intra is None:
                intra = fetch(ticker, "5m", "5d")
            d = analyze_daily(daily)
            i = analyze_intraday(intra)
            r = decide(d, i)
            if r["signal"] != "WAIT" and r["strength"] < MIN_STRENGTH:
                print(f"{ticker}: {r['signal']} 強度{r['strength']} は基準未満のため通知しません")
                continue
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
    t = (now.hour, now.minute)
    return (9, 0) <= t <= (11, 30) or (12, 30) <= t <= (15, 30)


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
    p.add_argument("--until", default="", help="この時刻(JST, HH:MM)でループを終了")
    a = p.parse_args(sys.argv[1:] if "ipykernel" not in sys.modules else [])

    if a.test:
        send_test()
    elif a.loop:
        while True:
            t0 = time.time()
            if a.until and datetime.now(JST).strftime("%H:%M") >= a.until:
                print(f"[{datetime.now(JST):%H:%M}] 終了時刻になりました")
                break
            if a.force or is_market_open():
                run_once(a.notify_wait)
            else:
                print(f"[{datetime.now(JST):%H:%M}] 場外のため待機中")
            time.sleep(max(5, a.interval - (time.time() - t0)))
    else:
        if a.force or is_market_open():
            run_once(a.notify_wait)
        else:
            print("場外時間です（--force で強制実行）")
