from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import io
import time

app = FastAPI(title="11% Trading API", version="7.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Cache ──────────────────────────────────────────────────────────
_CACHE: dict = {}
CACHE_TTL = 14400  # 4 hours

def cache_get(key):
    if key in _CACHE:
        data, ts = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            return data
        del _CACHE[key]
    return None

def cache_set(key, data):
    _CACHE[key] = (data, time.time())
    if len(_CACHE) > 200:
        oldest = sorted(_CACHE.items(), key=lambda x: x[1][1])[:50]
        for k, _ in oldest:
            del _CACHE[k]

# ── Data Fetchers ──────────────────────────────────────────────────
STOOQ_INTERVAL = {"1d": "d", "1wk": "w", "1mo": "m"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def stooq_ticker(ticker: str) -> str:
    """Convert common tickers to Stooq format"""
    mapping = {
        "BTC-USD": "btc.v", "ETH-USD": "eth.v", "BNB-USD": "bnb.v",
        "SOL-USD": "sol.v", "XRP-USD": "xrp.v", "ADA-USD": "ada.v",
        "DOGE-USD": "doge.v", "AVAX-USD": "avax.v", "LINK-USD": "link.v",
        "DOT-USD": "dot.v", "MATIC-USD": "matic.v",
        "^SPX": "^spx", "^NDX": "^ndx", "^DJI": "^dji",
        "SPY": "spy.us", "QQQ": "qqq.us", "GLD": "gld.us",
        "TLT": "tlt.us", "VXX": "vxx.us", "IWM": "iwm.us",
    }
    t = ticker.upper()
    if t in mapping:
        return mapping[t]
    if "." not in t and not t.startswith("^"):
        return f"{t.lower()}.us"
    return t.lower()

def fetch_stooq(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    stooq_t  = stooq_ticker(ticker)
    freq     = STOOQ_INTERVAL.get(interval, "d")
    url      = f"https://stooq.com/q/d/l/?s={stooq_t}&d1={start.replace('-','')}&d2={end.replace('-','')}&i={freq}"
    res      = requests.get(url, headers=HEADERS, timeout=15)
    if res.status_code != 200 or len(res.content) < 50:
        raise HTTPException(404, f"No data for {ticker}")
    df = pd.read_csv(io.StringIO(res.text))
    if df.empty or "Date" not in df.columns:
        raise HTTPException(404, f"No data for {ticker}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.columns = [c.strip() for c in df.columns]
    rename = {"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}
    df = df.rename(columns=rename)
    df = df.dropna(subset=["Close"])
    return df

def fetch_polygon_fallback(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Polygon.io fallback - daily only"""
    POLY_KEY = "V0HolJbbDoYAR8pAWfprGFbYlzmpG2Mr"
    mult, span = (1, "day") if interval == "1d" else (1, "week")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/{mult}/{span}/{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLY_KEY}"
    res = requests.get(url, timeout=15)
    if res.status_code != 200:
        raise HTTPException(404, f"No data for {ticker}")
    data = res.json()
    if not data.get("results"):
        raise HTTPException(404, f"No data for {ticker}")
    rows = []
    for r in data["results"]:
        rows.append({"Date": pd.Timestamp(r["t"], unit="ms"),
                     "Open": r["o"], "High": r["h"], "Low": r["l"],
                     "Close": r["c"], "Volume": r.get("v", 0)})
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df

def get_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    key = f"{ticker}_{start}_{end}_{interval}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        df = fetch_stooq(ticker, start, end, interval)
        if df.empty:
            raise ValueError("empty")
        cache_set(key, df)
        return df
    except Exception:
        pass
    try:
        df = fetch_polygon_fallback(ticker, start, end, interval)
        cache_set(key, df)
        return df
    except Exception:
        raise HTTPException(404, f"Could not fetch data for {ticker}. Try a different ticker or date range.")

def safe_float(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except: return None

def df_to_ohlcv(df):
    result = []
    for ts, row in df.iterrows():
        result.append({
            "time":   int(pd.Timestamp(ts).timestamp()),
            "open":   safe_float(row.get("Open")),
            "high":   safe_float(row.get("High")),
            "low":    safe_float(row.get("Low")),
            "close":  safe_float(row.get("Close")),
            "volume": safe_float(row.get("Volume")),
        })
    return result

# ── Indicators ─────────────────────────────────────────────────────
def sma(s, p): return s.rolling(p).mean()
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s, fast=12, slow=26, signal=9):
    fl = ema(s, fast); sl = ema(s, slow)
    ml = fl - sl; sig = ema(ml, signal)
    return ml, sig, ml - sig

def bollinger_bands(s, p=20, std=2):
    mid = sma(s, p); st = s.rolling(p).std()
    return mid + std*st, mid, mid - std*st

def atr(df, p=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def supertrend(df, period=10, multiplier=3):
    a = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    close = df["Close"]
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        cu, cl = upper.iloc[i], lower.iloc[i]
        pu, pl = upper.iloc[i-1], lower.iloc[i-1]
        upper.iloc[i] = cu if cu < pu or close.iloc[i-1] > pu else pu
        lower.iloc[i] = cl if cl > pl or close.iloc[i-1] < pl else pl
        if close.iloc[i] > upper.iloc[i-1]: direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]: direction.iloc[i] = -1
        else: direction.iloc[i] = direction.iloc[i-1]
        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return st, direction

# ── Pattern Detection ──────────────────────────────────────────────
def detect_patterns(df: pd.DataFrame) -> list:
    patterns = []
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    dates = [str(d.date()) for d in df.index]
    n = len(close)
    if n < 20:
        return patterns

    def find_peaks(arr, min_dist=5):
        peaks = []
        for i in range(min_dist, len(arr)-min_dist):
            if arr[i] == max(arr[i-min_dist:i+min_dist+1]):
                peaks.append(i)
        return peaks

    def find_troughs(arr, min_dist=5):
        troughs = []
        for i in range(min_dist, len(arr)-min_dist):
            if arr[i] == min(arr[i-min_dist:i+min_dist+1]):
                troughs.append(i)
        return troughs

    peaks   = find_peaks(high)
    troughs = find_troughs(low)

    # Head & Shoulders
    if len(peaks) >= 3:
        for i in range(len(peaks)-2):
            l, h_p, r = peaks[i], peaks[i+1], peaks[i+2]
            lv, hv, rv = high[l], high[h_p], high[r]
            if hv > lv and hv > rv and abs(lv-rv)/hv < 0.05 and (r-l) > 10:
                between_troughs = [t for t in troughs if l < t < r]
                if len(between_troughs) >= 2:
                    neckline = (low[between_troughs[0]] + low[between_troughs[-1]]) / 2
                    target   = neckline - (hv - neckline)
                    patterns.append({
                        "name": "Head & Shoulders",
                        "type": "Reversal",
                        "bias": "Bearish",
                        "reliability": "High",
                        "start_date": dates[l],
                        "end_date": dates[r],
                        "key_level": round(float(neckline), 2),
                        "target": round(float(target), 2),
                        "desc": f"Classic bearish reversal. Neckline at ${neckline:.2f}. Target ${target:.2f}.",
                        "forming": r >= n - 5,
                    })

    # Double Top
    if len(peaks) >= 2:
        for i in range(len(peaks)-1):
            p1, p2 = peaks[i], peaks[i+1]
            v1, v2 = high[p1], high[p2]
            if abs(v1-v2)/max(v1,v2) < 0.03 and (p2-p1) > 8:
                between = [t for t in troughs if p1 < t < p2]
                if between:
                    valley = low[between[0]]
                    target = valley - (max(v1,v2) - valley)
                    patterns.append({
                        "name": "Double Top",
                        "type": "Reversal",
                        "bias": "Bearish",
                        "reliability": "High",
                        "start_date": dates[p1],
                        "end_date": dates[p2],
                        "key_level": round(float(valley), 2),
                        "target": round(float(target), 2),
                        "desc": f"Two peaks at ~${v1:.2f}. Support at ${valley:.2f}. Break = bearish.",
                        "forming": p2 >= n - 5,
                    })

    # Double Bottom
    if len(troughs) >= 2:
        for i in range(len(troughs)-1):
            t1, t2 = troughs[i], troughs[i+1]
            v1, v2 = low[t1], low[t2]
            if abs(v1-v2)/max(v1,v2) < 0.03 and (t2-t1) > 8:
                between = [p for p in peaks if t1 < p < t2]
                if between:
                    peak_val = high[between[0]]
                    target   = peak_val + (peak_val - min(v1,v2))
                    patterns.append({
                        "name": "Double Bottom",
                        "type": "Reversal",
                        "bias": "Bullish",
                        "reliability": "High",
                        "start_date": dates[t1],
                        "end_date": dates[t2],
                        "key_level": round(float(peak_val), 2),
                        "target": round(float(target), 2),
                        "desc": f"Two lows at ~${v1:.2f}. Resistance at ${peak_val:.2f}. Break = bullish.",
                        "forming": t2 >= n - 5,
                    })

    # Bull Flag
    recent = close[-20:]
    if len(recent) >= 10:
        flagpole_move = (recent[5] - recent[0]) / recent[0]
        if flagpole_move > 0.05:
            flag_portion = recent[5:]
            flag_range   = (max(flag_portion) - min(flag_portion)) / recent[5]
            if flag_range < 0.04:
                target = recent[-1] + (recent[5] - recent[0])
                patterns.append({
                    "name": "Bull Flag",
                    "type": "Continuation",
                    "bias": "Bullish",
                    "reliability": "Medium",
                    "start_date": dates[-15],
                    "end_date": dates[-1],
                    "key_level": round(float(max(flag_portion)), 2),
                    "target": round(float(target), 2),
                    "desc": f"Strong move up followed by tight consolidation. Target ${target:.2f} on breakout.",
                    "forming": True,
                })

    # Bear Flag
    if len(recent) >= 10:
        flagpole_move = (recent[0] - recent[5]) / recent[0]
        if flagpole_move > 0.05:
            flag_portion = recent[5:]
            flag_range   = (max(flag_portion) - min(flag_portion)) / recent[5]
            if flag_range < 0.04:
                target = recent[-1] - (recent[0] - recent[5])
                patterns.append({
                    "name": "Bear Flag",
                    "type": "Continuation",
                    "bias": "Bearish",
                    "reliability": "Medium",
                    "start_date": dates[-15],
                    "end_date": dates[-1],
                    "key_level": round(float(min(flag_portion)), 2),
                    "target": round(float(target), 2),
                    "desc": f"Sharp drop followed by tight consolidation. Target ${target:.2f} on breakdown.",
                    "forming": True,
                })

    # Ascending Triangle
    if n >= 20:
        recent_high = high[-20:]
        recent_low  = low[-20:]
        resistance  = max(recent_high)
        touches     = sum(1 for h in recent_high if abs(h - resistance)/resistance < 0.01)
        lows_trend  = np.polyfit(range(len(recent_low)), recent_low, 1)
        if touches >= 2 and lows_trend[0] > 0:
            target = resistance + (resistance - min(recent_low))
            patterns.append({
                "name": "Ascending Triangle",
                "type": "Continuation",
                "bias": "Bullish",
                "reliability": "Medium",
                "start_date": dates[-20],
                "end_date": dates[-1],
                "key_level": round(float(resistance), 2),
                "target": round(float(target), 2),
                "desc": f"Rising support with flat resistance at ${resistance:.2f}. Bullish breakout setup.",
                "forming": True,
            })

    return patterns[-8:]  # return up to 8 patterns

# ── Backtest ────────────────────────────────────────────────────────
def run_backtest(df, strategy, params, capital):
    close = df["Close"]
    signals = pd.Series(0, index=df.index)

    if strategy == "SMA Crossover":
        fast = sma(close, params.get("fast", 20))
        slow = sma(close, params.get("slow", 50))
        signals = pd.Series(np.where(fast > slow, 1, -1), index=df.index)
    elif strategy == "EMA Crossover":
        fast = ema(close, params.get("fast", 12))
        slow = ema(close, params.get("slow", 26))
        signals = pd.Series(np.where(fast > slow, 1, -1), index=df.index)
    elif strategy == "RSI":
        r = rsi(close, params.get("period", 14))
        ob, os_ = params.get("overbought", 70), params.get("oversold", 30)
        pos, sig = 0, []
        for v in r:
            if pd.isna(v): sig.append(0); continue
            if v < os_: pos = 1
            elif v > ob: pos = -1
            sig.append(pos)
        signals = pd.Series(sig, index=df.index)
    elif strategy == "MACD":
        m, s, _ = macd(close, params.get("fast",12), params.get("slow",26), params.get("signal",9))
        signals = pd.Series(np.where(m > s, 1, -1), index=df.index)
    elif strategy == "Bollinger Bands":
        upper, mid, lower = bollinger_bands(close, params.get("period",20), params.get("std",2))
        pos, sig = 0, []
        for c, u, l in zip(close, upper, lower):
            if pd.isna(u): sig.append(0); continue
            if c < l: pos = 1
            elif c > u: pos = -1
            sig.append(pos)
        signals = pd.Series(sig, index=df.index)
    elif strategy == "SuperTrend":
        _, direction = supertrend(df, params.get("period",10), params.get("multiplier",3))
        signals = direction
    elif strategy == "EMA + RSI Filter":
        fast = ema(close, params.get("fast",12))
        slow = ema(close, params.get("slow",26))
        r    = rsi(close, params.get("rsi_period",14))
        trend   = pd.Series(np.where(fast > slow, 1, -1), index=df.index)
        filter_ = pd.Series(np.where(r < 70, 1, np.where(r > 30, 1, 0)), index=df.index)
        signals = trend * filter_

    position, entry_price, equity, trades, balance = 0, 0.0, [capital], [], capital
    for i in range(1, len(df)):
        sig   = signals.iloc[i]
        price = float(close.iloc[i])
        date  = str(df.index[i].date())
        if position == 0 and sig == 1:
            position = balance / price; entry_price = price
        elif position > 0 and sig == -1:
            pnl = (price - entry_price) * position
            trades.append({"date":date,"entry":entry_price,"exit":price,"pnl":round(pnl,2),"pct":round(pnl/(position*entry_price)*100,2)})
            balance = position * price; position = 0
        equity.append(balance + (position * price if position > 0 else 0))

    if position > 0:
        fp = float(close.iloc[-1])
        pnl = (fp - entry_price) * position
        trades.append({"date":str(df.index[-1].date()),"entry":entry_price,"exit":fp,"pnl":round(pnl,2),"pct":round(pnl/(position*entry_price)*100,2)})
        balance = position * fp

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    eq = pd.Series(equity)
    drawdown = ((eq - eq.cummax()) / eq.cummax() * 100)
    ret = eq.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

    return {
        "total_return":  round((balance - capital) / capital * 100, 2),
        "final_equity":  round(balance, 2),
        "total_trades":  len(trades),
        "win_rate":      round(len(wins)/len(trades)*100, 1) if trades else 0,
        "max_drawdown":  round(float(drawdown.min()), 2),
        "sharpe_ratio":  round(sharpe, 2),
        "avg_win":       round(sum(t["pnl"] for t in wins)/len(wins), 2) if wins else 0,
        "avg_loss":      round(sum(t["pnl"] for t in losses)/len(losses), 2) if losses else 0,
        "trades":        trades[-50:],
        "equity_curve":  [round(v, 2) for v in equity],
        "equity_dates":  [str(d.date()) for d in df.index],
    }

# ── Routes ──────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status": "11% API v7.0 running", "data_source": "Stooq + Polygon"}

@app.get("/api/ohlcv")
def get_ohlcv(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), interval: str = Query("1d")):
    df = get_data(ticker, start, end, interval)
    return {"ticker": ticker, "bars": df_to_ohlcv(df)}

@app.get("/api/replay-data")
def replay_data(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), interval: str = Query("1d")):
    df = get_data(ticker, start, end, interval)
    return {"ticker": ticker, "bars": df_to_ohlcv(df)}

@app.get("/api/ticker-info")
def ticker_info(ticker: str = Query(...)):
    """Get basic price info from Stooq"""
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    try:
        df = get_data(ticker, start, end)
        close = df["Close"]
        current = float(close.iloc[-1])
        prev    = float(close.iloc[-2]) if len(close) > 1 else current
        w52high = float(close.tail(252).max())
        w52low  = float(close.tail(252).min())
        r = rsi(close)
        m, s, _ = macd(close)
        volume  = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else None
        avg_vol = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else None
        sma20_val = float(sma(close, 20).iloc[-1]) if len(close) >= 20 else None
        sma50_val = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else None
        sma200_val = float(sma(close, 200).iloc[-1]) if len(close) >= 200 else None
        return {
            "name":          ticker.upper(),
            "sector":        "",
            "current_price": round(current, 2),
            "prev_close":    round(prev, 2),
            "change":        round((current - prev) / prev * 100, 2),
            "52w_high":      round(w52high, 2),
            "52w_low":       round(w52low, 2),
            "avg_volume":    int(avg_vol) if avg_vol else None,
            "volume":        int(volume) if volume else None,
            "rsi":           round(float(r.iloc[-1]), 1) if not pd.isna(r.iloc[-1]) else None,
            "macd":          round(float(m.iloc[-1]), 3) if not pd.isna(m.iloc[-1]) else None,
            "macd_signal":   round(float(s.iloc[-1]), 3) if not pd.isna(s.iloc[-1]) else None,
            "sma20":         round(sma20_val, 2) if sma20_val else None,
            "sma50":         round(sma50_val, 2) if sma50_val else None,
            "sma200":        round(sma200_val, 2) if sma200_val else None,
            "above_sma20":   current > sma20_val if sma20_val else None,
            "above_sma50":   current > sma50_val if sma50_val else None,
            "above_sma200":  current > sma200_val if sma200_val else None,
        }
    except Exception as e:
        raise HTTPException(404, f"Could not fetch data for {ticker}: {str(e)}")

@app.get("/api/patterns")
def get_patterns(
    ticker:   str = Query(...),
    start:    str = Query(...),
    end:      str = Query(...),
    interval: str = Query("1d")
):
    df = get_data(ticker, start, end, interval)
    patterns = detect_patterns(df)
    bars = df_to_ohlcv(df)
    # Calculate indicators for chart overlay
    close = df["Close"]
    sma20_vals  = [safe_float(v) for v in sma(close, 20)]
    sma50_vals  = [safe_float(v) for v in sma(close, 50)]
    rsi_vals    = [safe_float(v) for v in rsi(close)]
    vol_avg     = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
    return {
        "ticker":    ticker,
        "bars":      bars,
        "patterns":  patterns,
        "indicators": {
            "sma20":  sma20_vals,
            "sma50":  sma50_vals,
            "rsi":    rsi_vals,
            "vol_avg": round(vol_avg, 0),
        }
    }

class BacktestRequest(BaseModel):
    ticker: str; start: str; end: str; strategy: str
    capital: float = 10000; params: dict = {}

@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    if not req.ticker: raise HTTPException(400, "Ticker required")
    try:
        datetime.strptime(req.start, "%Y-%m-%d")
        datetime.strptime(req.end, "%Y-%m-%d")
    except: raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")
    if req.start >= req.end: raise HTTPException(400, "Start date must be before end date")
    if req.capital <= 0: raise HTTPException(400, "Capital must be positive")
    df = get_data(req.ticker, req.start, req.end)
    if len(df) < 30: raise HTTPException(400, f"Not enough data. Only {len(df)} bars found. Try a wider date range.")
    result = run_backtest(df, req.strategy, req.params, req.capital)
    return {"ticker": req.ticker, "strategy": req.strategy, **result}

@app.get("/api/screener")
def screener(tickers: str = Query(...)):
    results = []
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    for ticker in tickers.split(",")[:25]:
        ticker = ticker.strip().upper()
        if not ticker: continue
        try:
            df    = get_data(ticker, start, end)
            close = df["Close"]
            if len(close) < 14: continue
            r     = rsi(close)
            m, s, h = macd(close)
            price  = float(close.iloc[-1])
            change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0
            vol    = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else None
            avg_v  = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else None
            rel_vol = round(vol / avg_v, 2) if vol and avg_v and avg_v > 0 else None
            sma20v  = sma(close, 20).iloc[-1]
            sma50v  = sma(close, 50).iloc[-1] if len(close) >= 50 else None
            results.append({
                "ticker":    ticker,
                "price":     round(price, 2),
                "change":    round(change, 2),
                "rsi":       round(float(r.iloc[-1]), 1) if not pd.isna(r.iloc[-1]) else None,
                "macd":      round(float(m.iloc[-1]), 3) if not pd.isna(m.iloc[-1]) else None,
                "signal":    round(float(s.iloc[-1]), 3) if not pd.isna(s.iloc[-1]) else None,
                "volume":    int(vol) if vol else None,
                "rel_vol":   rel_vol,
                "above_sma20": bool(price > sma20v) if not pd.isna(sma20v) else None,
                "above_sma50": bool(price > sma50v) if sma50v and not pd.isna(sma50v) else None,
                "macd_cross": "bullish" if m.iloc[-1] > s.iloc[-1] and m.iloc[-2] < s.iloc[-2] else
                              "bearish" if m.iloc[-1] < s.iloc[-1] and m.iloc[-2] > s.iloc[-2] else "none",
            })
        except: continue
    return {"results": results}

@app.get("/api/correlations")
def correlations(tickers: str = Query(...), start: str = Query(...), end: str = Query(...)):
    closes = {}
    for ticker in tickers.split(",")[:12]:
        ticker = ticker.strip().upper()
        if not ticker: continue
        try:
            df = get_data(ticker, start, end)
            closes[ticker] = df["Close"]
        except: continue
    if len(closes) < 2: raise HTTPException(400, "Need at least 2 valid tickers")
    combined = pd.DataFrame(closes).dropna()
    corr = combined.pct_change().dropna().corr()
    return {"tickers": list(closes.keys()), "matrix": corr.round(3).to_dict()}

@app.get("/api/monte-carlo")
def monte_carlo(ticker: str = Query(...), start: str = Query(...), end: str = Query(...),
                simulations: int = Query(500), days: int = Query(252), capital: float = Query(10000)):
    if capital <= 0: raise HTTPException(400, "Capital must be positive")
    df = get_data(ticker, start, end)
    close = df["Close"]
    returns = close.pct_change().dropna()
    mu, sigma = float(returns.mean()), float(returns.std())
    paths, finals = [], []
    for _ in range(min(simulations, 1000)):
        path = [capital]
        for _ in range(days):
            path.append(path[-1] * (1 + np.random.normal(mu, sigma)))
        paths.append([round(v, 2) for v in path[::5]])
        finals.append(round(path[-1], 2))
    fa = np.array(finals)
    return {
        "simulations":   len(paths),
        "days":          days,
        "paths":         paths[:80],
        "final_median":  round(float(np.median(fa)), 2),
        "final_mean":    round(float(np.mean(fa)), 2),
        "percentile_5":  round(float(np.percentile(fa, 5)), 2),
        "percentile_25": round(float(np.percentile(fa, 25)), 2),
        "percentile_75": round(float(np.percentile(fa, 75)), 2),
        "percentile_95": round(float(np.percentile(fa, 95)), 2),
        "prob_profit":   round(float(np.mean(fa > capital) * 100), 1),
    }

@app.get("/api/market-data")
def market_data():
    """Live market snapshot for dashboard/heatmap"""
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    WATCHLIST = ["SPY","QQQ","IWM","GLD","TLT","XLK","XLV","XLF","XLE","XLY","XLI","XLC"]
    results = []
    for t in WATCHLIST:
        try:
            df = get_data(t, start, end)
            close = df["Close"]
            if len(close) < 2: continue
            chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            results.append({"ticker": t, "price": round(float(close.iloc[-1]),2), "change": round(float(chg),2)})
        except: continue
    return {"data": results, "updated": datetime.now().isoformat()}

@app.get("/api/earnings")
def earnings(ticker: str = Query(...)):
    """Earnings data via Stooq price history"""
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    try:
        df = get_data(ticker, start, end)
        close = df["Close"]
        # Approximate earnings dates by finding large single-day moves
        pct_change = close.pct_change()
        big_moves = pct_change[abs(pct_change) > 0.04].tail(12)
        history = []
        for date, move in big_moves.items():
            history.append({
                "date":    str(date.date()),
                "move":    round(float(move * 100), 2),
                "price":   round(float(close[date]), 2),
                "type":    "Beat" if move > 0 else "Miss",
            })
        history.reverse()
        return {"ticker": ticker.upper(), "history": history, "note": "Large single-day moves (>4%) shown as potential earnings events"}
    except Exception as e:
        raise HTTPException(404, str(e))
