#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バックテスト: stock_signal_notifier.py と同じ条件で過去の5分足を再生し、勝率を出す。
- 銘柄: UNIVERSE から、前営業日の売買代金（出来高×株価）上位TOP銘柄を毎日選ぶ
- 売買: ★MIN_STRENGTH以上のシグナルでエントリー → 保有本数(5分足)後に決済（または大引け）
- 「遅れなし」と「データ遅延を想定してN本後にエントリー」の両方を計算
- 「日足フィルターあり/なし」を比較（日足の分析に意味があるかの確認）
使い方: python backtest.py --days 5
注意: 手数料・スリッページ・約定の滑りは考慮していません（--cost で往復コスト%を指定可）
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

import stock_signal_notifier as sn

COLS = ["Open", "High", "Low", "Close", "Volume"]


def to_jst(df):
    df = df.dropna(subset=["Close"]).copy()
    df.index = df.index.tz_convert(sn.JST) if df.index.tz is not None else df.index.tz_localize(sn.JST)
    return df


def to_naive_dates(df):
    df = df.dropna(subset=["Close"]).copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(sn.JST).tz_localize(None)
    df.index = idx.normalize()
    return df


def load():
    tickers = list(sn.UNIVERSE)
    kw = dict(group_by="ticker", auto_adjust=False, progress=False, threads=True)
    d5 = yf.download(tickers, period="60d", interval="5m", **kw)
    d1 = yf.download(tickers, period="1y", interval="1d", **kw)
    intra, daily = {}, {}
    for t in tickers:
        try:
            a, b = to_jst(d5[t]), to_naive_dates(d1[t])
            if len(a) > 400 and len(b) > 60:
                intra[t], daily[t] = a, b
        except Exception:
            continue
    return intra, daily


def fmt(sub):
    if len(sub) == 0:
        return "0件"
    return f"{len(sub)}件 勝率{(sub > 0).mean() * 100:.0f}% 平均{sub.mean():+.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="直近何営業日で検証するか")
    ap.add_argument("--top", type=int, default=20, help="毎日の対象銘柄数（売買代金上位）")
    ap.add_argument("--hold", type=int, default=6, help="保有本数（5分足）。6=30分")
    ap.add_argument("--delay", type=int, default=4, help="データ遅延を想定したエントリー遅れ本数。4≒20分")
    ap.add_argument("--cost", type=float, default=0.0, help="1回あたりの売買コスト(%%)")
    a = ap.parse_args()

    print("データ取得中...")
    intra, daily = load()
    if not intra:
        print("データが取得できませんでした")
        return
    now = dt.datetime.now(sn.JST)
    all_days = sorted({d for df in intra.values() for d in set(df.index.date)})
    target_days = all_days[-a.days:]
    print(f"検証期間: {target_days[0]} 〜 {target_days[-1]}（{len(target_days)}営業日）")

    trades = []      # (variant, delay, side, ret)
    base = []        # 全バーでロングした場合（参考）
    cand_days = 0
    for D in target_days:
        # 前営業日の売買代金上位を選ぶ
        ranks = []
        for t, b in daily.items():
            prev = b[b.index < pd.Timestamp(D)]
            if len(prev) < 40:
                continue
            p = float(prev["Close"].iloc[-1])
            if sn.MIN_PRICE <= p <= sn.MAX_PRICE:
                ranks.append((float(prev["Close"].iloc[-1] * prev["Volume"].iloc[-1]), t))
        picks = [t for _, t in sorted(ranks, reverse=True)[:a.top]]
        cand_days += len(picks)

        for t in picks:
            x5 = intra[t]
            idx = np.where(x5.index.date == D)[0]
            if len(idx) < 30:
                continue
            first, last = idx[0], idx[-1]
            hist = daily[t][daily[t].index < pd.Timestamp(D)][COLS]
            seen = {"あり": set(), "なし": set()}
            for k in idx[6:]:
                end = x5.index[k] + pd.Timedelta(minutes=5)
                if end > now:
                    break
                if end.time() > dt.time(14, 30):
                    break
                part = x5.iloc[first:k + 1]
                today = pd.DataFrame({"Open": [part["Open"].iloc[0]], "High": [part["High"].max()],
                                      "Low": [part["Low"].min()], "Close": [part["Close"].iloc[-1]],
                                      "Volume": [part["Volume"].sum()]}, index=[pd.Timestamp(D)])
                d = sn.analyze_daily(pd.concat([hist, today]))
                i = sn.analyze_intraday(x5.iloc[max(0, k - 329):k + 1])
                close = x5["Close"]

                if k < last:
                    x = min(k + a.hold, last)
                    base.append((float(close.iloc[x] / close.iloc[k] - 1) * 100) - a.cost)

                d_nf = {"trend": {"golden": "up", "dead": "down"}.get(i["cross"], "flat")}
                for variant, dd in (("あり", d), ("なし", d_nf)):
                    r = sn.decide(dd, i)
                    if r["signal"] == "WAIT" or r["strength"] < sn.MIN_STRENGTH:
                        continue
                    key = (r["signal"], i["cross_time"])
                    if key in seen[variant]:
                        continue
                    seen[variant].add(key)
                    sign = 1 if r["signal"] == "BUY" else -1
                    for delay in (0, a.delay):
                        e = k + delay
                        if e >= last:
                            continue
                        x = min(e + a.hold, last)
                        ret = sign * (float(close.iloc[x] / close.iloc[e] - 1) * 100) - a.cost
                        trades.append((variant, delay, r["signal"], ret))

    print(f"\n=== バックテスト結果 ===")
    print(f"対象: 毎日 売買代金上位{a.top}銘柄（延べ{cand_days}銘柄日） / 保有{a.hold * 5}分 / 強度★{sn.MIN_STRENGTH}以上")
    df = pd.DataFrame(trades, columns=["variant", "delay", "side", "ret"])
    for variant in ("あり", "なし"):
        print(f"\n■ 日足フィルター{variant}")
        for delay, label in ((0, "遅れなし（リアルタイム想定）"), (a.delay, f"遅れあり（{a.delay * 5}分後にエントリー）")):
            s = df[(df.variant == variant) & (df.delay == delay)]
            print(f" {label}")
            print(f"   全体: {fmt(s.ret)}")
            print(f"   買い: {fmt(s[s.side == 'BUY'].ret)} / 売り: {fmt(s[s.side == 'SELL'].ret)}")
    b = pd.Series(base)
    print(f"\n参考: 毎バーで買った場合（比較用）: {fmt(b)}")
    print("\n※件数が少ない（目安: 100件未満）と、勝率は偶然でぶれます。手数料・滑りは含みません。")


if __name__ == "__main__":
    main()
