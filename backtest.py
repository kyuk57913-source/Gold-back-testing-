import pandas as pd
import yfinance as yf

# symbol: SL buffer (price points). Nasdaq က ဈေးကြီးလို့ buffer ပိုကြီးထားတယ်
SYMBOLS = {"GC=F": 0.5, "NQ=F": 5.0}

LOOKBACK = 6
SETUP_BARS = 48
MAX_BARS = 96
RR_LIST = [2, 3, 5]


def load(sym):
    raw = yf.download(sym, period="60d", interval="15m", progress=False, auto_adjust=True)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    return raw.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()


def find_sweeps(h4):
    out = []
    for i in range(LOOKBACK, len(h4)):
        prev = h4.iloc[i - LOOKBACK:i]
        c = h4.iloc[i]
        t_end = h4.index[i] + pd.Timedelta(hours=4)
        if c.high > prev.high.max() and c.close < prev.high.max():
            out.append(("short", t_end, c.high))
        elif c.low < prev.low.min() and c.close > prev.low.min():
            out.append(("long", t_end, c.low))
    return out


def find_entries(df, sweeps, buffer):
    out = []
    for side, t0, extreme in sweeps:
        s = df.index.searchsorted(t0)
        end = min(s + SETUP_BARS, len(df) - 1)
        for i in range(s + 2, end):
            a, c = df.iloc[i - 2], df.iloc[i]
            if side == "short" and a.low > c.high:
                mid, sl = (a.low + c.high) / 2, extreme + buffer
            elif side == "long" and a.high < c.low:
                mid, sl = (a.high + c.low) / 2, extreme - buffer
            else:
                continue
            for j in range(i + 1, end + 1):
                hi, lo = df.high.iloc[j], df.low.iloc[j]
                if side == "short" and hi >= sl:
                    break
                if side == "long" and lo <= sl:
                    break
                if (side == "short" and hi >= mid) or (side == "long" and lo <= mid):
                    out.append((j, side, mid, sl))
                    break
            break
    return out


def simulate(df, j, side, entry, sl, rr):
    risk = abs(sl - entry)
    tp = entry + rr * risk if side == "long" else entry - rr * risk
    last = min(j + 1 + MAX_BARS, len(df))
    for k in range(j + 1, last):
        hi, lo = df.high.iloc[k], df.low.iloc[k]
        if side == "long":
            if lo <= sl:
                return -1.0
            if hi >= tp:
                return float(rr)
        else:
            if hi >= sl:
                return -1.0
            if lo <= tp:
                return float(rr)
    close = df.close.iloc[last - 1]
    return ((close - entry) if side == "long" else (entry - close)) / risk


for sym, buffer in SYMBOLS.items():
    print(f"\n===== {sym} =====")
    df = load(sym)
    if df.empty:
        print("No data")
        continue
    h4 = df.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    sweeps = find_sweeps(h4)
    entries = find_entries(df, sweeps, buffer)
    print(f"Data: {df.index[0]} -> {df.index[-1]} | sweeps: {len(sweeps)} | entries: {len(entries)}")
    for rr in RR_LIST:
        r = pd.Series([simulate(df, j, sd, e, sl, rr) for j, sd, e, sl in entries], dtype=float)
        if r.empty:
            print(f"RR {rr}: no trades")
            continue
        eq = r.cumsum()
        dd = (eq.cummax() - eq).max()
        gp, gl = r[r > 0].sum(), -r[r < 0].sum()
        pf = gp / gl if gl else float("inf")
        print(
            f"RR {rr}: trades {len(r)} | win {100 * (r > 0).mean():.0f}% | "
            f"total {r.sum():.1f}R | avg {r.mean():.2f}R | PF {pf:.2f} | maxDD {dd:.1f}R"
    )
