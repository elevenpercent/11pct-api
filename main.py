from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import io
import time

app = FastAPI(title="11% Trading API", version="6.0.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

POLY_KEY  = os.environ.get("POLYGON_KEY", "V0HolJbbDoYAR8pAWfprGFbYlzmpG2Mr")
POLY_BASE = "https://api.polygon.io"

# ── In-memory cache (survives for duration of server run) ────────────────────
_CACHE: dict = {}
CACHE_TTL = 14400  # 4 hours for historical data

def cache_get(key: str):
    if key in _CACHE:
        data, ts = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            return data
        del _CACHE[key]
    return None

def cache_set(key: str, data):
    _CACHE[key] = (data, time.time())

# ── Stooq ticker mapper ───────────────────────────────────────────────────────
def to_stooq_symbol(ticker: str) -> str:
    ticker = ticker.upper().strip()
    # Crypto
    crypto_map = {
        'BTC-USD': 'btc.v', 'ETH-USD': 'eth.v', 'SOL-USD': 'sol.v',
        'BNB-USD': 'bnb.v', 'ADA-USD': 'ada.v', 'XRP-USD': 'xrp.v',
        'DOGE-USD': 'doge.v', 'AVAX-USD': 'avax.v',
    }
    if ticker in crypto_map:
        return crypto_map[ticker]
    # Indices
    index_map = {
        'SPY': 'spy.us', 'QQQ': 'qqq.us', 'DIA': 'dia.us', 'IWM': 'iwm.us',
        '^GSPC': '^spx', '^DJI': '^dji', '^IXIC': '^ndq',
        'VTI': 'vti.us', 'GLD': 'gld.us', 'TLT': 'tlt.us', 'VXX': 'vxx.us',
    }
    if ticker in index_map:
        return index_map[ticker]
    # ETFs and stocks — add .us suffix
    if '-' not in ticker and '^' not in ticker:
        return f"{ticker.lower()}.us"
    return ticker.lower()

# ── Stooq fetcher — 20+ years of free data, no API key ───────────────────────
def fetch_stooq(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    sym = to_stooq_symbol(ticker)

    # Map interval to Stooq format
    interval_map = {"1d": "d", "1wk": "w", "1mo": "m", "1h": "h"}
    stooq_interval = interval_map.get(interval, "d")

    start_fmt = start.replace("-", "")
    end_fmt   = end.replace("-", "")

    url = f"https://stooq.com/q/d/l/?s={sym}&d1={start_fmt}&d2={end_fmt}&i={stooq_interval}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://stooq.com/",
    }

    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise Exception(f"Stooq returned {r.status_code}")

    content = r.text.strip()
    if not content or "No data" in content or len(content) < 50:
        raise Exception(f"No data from Stooq for {ticker}")

    df = pd.read_csv(io.StringIO(content))
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names
    col_map = {
        'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low',
        'Close': 'Close', 'Volume': 'Volume'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if 'Date' not in df.columns:
        raise Exception("Unexpected Stooq response format")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Filter to requested range
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    if df.empty:
        raise Exception(f"No data for {ticker} in range {start} to {end}")

    # Ensure numeric columns
    for col in ['Open','High','Low','Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

    df = df.dropna(subset=['Close'])
    return df

# ── Polygon fallback for data Stooq doesn't have ────────────────────────────
def fetch_polygon(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    if interval == "1d":   mult, span = 1, "day"
    elif interval == "1wk": mult, span = 1, "week"
    elif interval == "1h":  mult, span = 1, "hour"
    else:                   mult, span = 1, "day"

    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker.upper()}/range/{mult}/{span}/{start}/{end}"
    params = {"adjusted":"true","sort":"asc","limit":50000,"apiKey":POLY_KEY}
    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    if not data.get("results"):
        raise Exception(f"No data from Polygon for {ticker}")

    rows = []
    for bar in data["results"]:
        rows.append({
            "Date":   pd.Timestamp(bar["t"], unit="ms"),
            "Open":   float(bar["o"]), "High": float(bar["h"]),
            "Low":    float(bar["l"]), "Close": float(bar["c"]),
            "Volume": float(bar.get("v", 0)),
        })
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df

# ── Main data getter with caching and fallback ───────────────────────────────
def get_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    cache_key = f"{ticker}_{start}_{end}_{interval}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    # Try Stooq first (unlimited history, no IP blocking)
    try:
        df = fetch_stooq(ticker, start, end, interval)
        if not df.empty:
            cache_set(cache_key, df)
            return df
    except Exception as e:
        print(f"Stooq failed for {ticker}: {e}, trying Polygon...")

    # Fallback to Polygon
    try:
        df = fetch_polygon(ticker, start, end, interval)
        if not df.empty:
            cache_set(cache_key, df)
            return df
    except Exception as e:
        print(f"Polygon failed for {ticker}: {e}")

    raise HTTPException(status_code=404, detail=f"No data found for {ticker} between {start} and {end}. Try a US stock ticker like AAPL.")

def safe_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except: return None

def df_to_ohlcv(df):
    return [{
        "time":   int(pd.Timestamp(ts).timestamp()),
        "open":   safe_float(row.get("Open")),
        "high":   safe_float(row.get("High")),
        "low":    safe_float(row.get("Low")),
        "close":  safe_float(row.get("Close")),
        "volume": safe_float(row.get("Volume", 0)),
    } for ts, row in df.iterrows()]

# ── Indicators ────────────────────────────────────────────────────────────────
def sma(s, p): return s.rolling(p).mean()
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def macd(s, fast=12, slow=26, signal=9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m, sig, m - sig

def bollinger(s, p=20, std=2):
    mid = sma(s, p)
    sd = s.rolling(p).std()
    return mid + std*sd, mid, mid - std*sd

def atr_calc(df, p=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def supertrend(df, p=10, m=3):
    a = atr_calc(df, p)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + m * a
    lower = hl2 - m * a
    c = df["Close"]
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        pu, pl = upper.iloc[i-1], lower.iloc[i-1]
        upper.iloc[i] = min(upper.iloc[i], pu) if c.iloc[i-1] <= pu else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], pl) if c.iloc[i-1] >= pl else lower.iloc[i]
        if c.iloc[i] > upper.iloc[i-1]: direction.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i-1]: direction.iloc[i] = -1
        else: direction.iloc[i] = direction.iloc[i-1]
    return direction

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df, strategy, params, capital):
    close = df["Close"]
    signals = pd.Series(0, index=df.index)

    if strategy == "SMA Crossover":
        f, s = sma(close, int(params.get("fast",20))), sma(close, int(params.get("slow",50)))
        signals = pd.Series(np.where(f > s, 1, -1), index=df.index)
    elif strategy == "EMA Crossover":
        f, s = ema(close, int(params.get("fast",12))), ema(close, int(params.get("slow",26)))
        signals = pd.Series(np.where(f > s, 1, -1), index=df.index)
    elif strategy == "RSI":
        r = rsi(close, int(params.get("period",14)))
        ob, os_ = params.get("overbought",70), params.get("oversold",30)
        pos, sig = 0, []
        for v in r:
            if pd.isna(v): sig.append(0); continue
            if v < os_: pos = 1
            elif v > ob: pos = -1
            sig.append(pos)
        signals = pd.Series(sig, index=df.index)
    elif strategy == "MACD":
        m, s, _ = macd(close, int(params.get("fast",12)), int(params.get("slow",26)), int(params.get("signal",9)))
        signals = pd.Series(np.where(m > s, 1, -1), index=df.index)
    elif strategy == "Bollinger Bands":
        upper, _, lower = bollinger(close, int(params.get("period",20)), params.get("std",2))
        pos, sig = 0, []
        for c2, u, l in zip(close, upper, lower):
            if pd.isna(u): sig.append(0); continue
            if c2 < l: pos = 1
            elif c2 > u: pos = -1
            sig.append(pos)
        signals = pd.Series(sig, index=df.index)
    elif strategy == "SuperTrend":
        signals = supertrend(df, int(params.get("period",10)), params.get("multiplier",3))
    elif strategy == "EMA + RSI Filter":
        f = ema(close, int(params.get("fast",12)))
        s = ema(close, int(params.get("slow",26)))
        r = rsi(close, int(params.get("rsi_period",14)))
        signals = pd.Series(np.where((f > s) & (r < 70), 1, np.where((f < s) & (r > 30), -1, 0)), index=df.index)

    position, entry_price, balance = 0, 0.0, capital
    equity, trades = [capital], []

    for i in range(1, len(df)):
        sig = signals.iloc[i]
        price = float(close.iloc[i])
        date = str(df.index[i].date())
        if position == 0 and sig == 1:
            position = balance / price
            entry_price = price
        elif position > 0 and sig == -1:
            exit_val = position * price
            pnl = exit_val - (position * entry_price)
            trades.append({"date":date,"entry":round(entry_price,2),"exit":round(price,2),"pnl":round(pnl,2),"pct":round(pnl/(position*entry_price)*100,2)})
            balance = exit_val
            position = 0
        equity.append(balance + (position * price if position > 0 else 0))

    if position > 0:
        fp = float(close.iloc[-1])
        pnl = position * fp - position * entry_price
        trades.append({"date":str(df.index[-1].date()),"entry":round(entry_price,2),"exit":round(fp,2),"pnl":round(pnl,2),"pct":round(pnl/(position*entry_price)*100,2)})
        balance = position * fp

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    eq     = pd.Series(equity)
    dd     = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    rets   = eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0

    return {
        "total_return":  round((balance - capital) / capital * 100, 2),
        "final_equity":  round(balance, 2),
        "total_trades":  len(trades),
        "win_rate":      round(len(wins)/len(trades)*100, 1) if trades else 0,
        "max_drawdown":  round(float(dd), 2),
        "sharpe_ratio":  round(sharpe, 2),
        "avg_win":       round(sum(t["pnl"] for t in wins)/len(wins), 2) if wins else 0,
        "avg_loss":      round(sum(t["pnl"] for t in losses)/len(losses), 2) if losses else 0,
        "trades":        trades[-50:],
        "equity_curve":  [round(v, 2) for v in equity],
        "equity_dates":  [str(d.date()) for d in df.index],
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "11% API running", "version": "6.0.0", "data": "Stooq (20yr) + Polygon fallback", "cache_size": len(_CACHE)}

@app.get("/api/ohlcv")
def get_ohlcv(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), interval: str = Query("1d")):
    df = get_data(ticker, start, end, interval)
    return {"ticker": ticker, "data": df_to_ohlcv(df), "bars": len(df)}

@app.get("/api/ticker-info")
def ticker_info(ticker: str = Query(...)):
    try:
        r = requests.get(f"{POLY_BASE}/v3/reference/tickers/{ticker.upper()}", params={"apiKey": POLY_KEY}, timeout=15)
        d = r.json().get("results", {})
        # Get recent price from Stooq
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        price = None
        change = None
        try:
            df = get_data(ticker, start, end)
            if len(df) >= 2:
                price = round(float(df["Close"].iloc[-1]), 2)
                change = round(float((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100), 2)
        except: pass
        return {
            "name":          d.get("name", ticker),
            "sector":        d.get("sic_description", ""),
            "market_cap":    d.get("market_cap"),
            "current_price": price,
            "change_pct":    change,
            "ticker":        ticker.upper(),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

class BacktestRequest(BaseModel):
    ticker: str
    start: str
    end: str
    strategy: str
    capital: float = 10000
    params: dict = {}

@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    try:
        df = get_data(req.ticker, req.start, req.end)
        return {"ticker": req.ticker, "strategy": req.strategy, **run_backtest(df, req.strategy, req.params, req.capital)}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/replay-data")
def replay_data(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), interval: str = Query("1d")):
    df = get_data(ticker, start, end, interval)
    return {"ticker": ticker, "bars": df_to_ohlcv(df), "total": len(df)}

@app.get("/api/indicators")
def get_indicators(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), indicator: str = Query(...)):
    df = get_data(ticker, start, end)
    close = df["Close"]
    result = {}
    if indicator == "RSI":   result["rsi"] = [safe_float(v) for v in rsi(close)]
    elif indicator == "MACD":
        m, s, h = macd(close)
        result = {"macd":[safe_float(v) for v in m],"signal":[safe_float(v) for v in s],"histogram":[safe_float(v) for v in h]}
    elif indicator == "BB":
        u, mid, l = bollinger(close)
        result = {"upper":[safe_float(v) for v in u],"mid":[safe_float(v) for v in mid],"lower":[safe_float(v) for v in l]}
    elif indicator == "SMA":
        result = {"sma20":[safe_float(v) for v in sma(close,20)],"sma50":[safe_float(v) for v in sma(close,50)],"sma200":[safe_float(v) for v in sma(close,200)]}
    elif indicator == "EMA":
        result = {"ema12":[safe_float(v) for v in ema(close,12)],"ema26":[safe_float(v) for v in ema(close,26)]}
    elif indicator == "ATR":
        result["atr"] = [safe_float(v) for v in atr_calc(df)]
    result["dates"] = [str(d.date()) for d in df.index]
    return result

@app.get("/api/screener")
def screener(tickers: str = Query(...)):
    results = []
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    for ticker in tickers.split(",")[:30]:
        ticker = ticker.strip().upper()
        try:
            df = get_data(ticker, start, end)
            close = df["Close"]
            r = rsi(close).iloc[-1]
            m, s, _ = macd(close)
            sma20v = sma(close, 20).iloc[-1]
            sma50v = sma(close, 50).iloc[-1]
            atr_v  = atr_calc(df).iloc[-1]
            results.append({
                "ticker":  ticker,
                "price":   round(float(close.iloc[-1]), 2),
                "change":  round(float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100), 2),
                "rsi":     round(float(r), 1) if not np.isnan(r) else None,
                "macd":    round(float(m.iloc[-1]), 3) if not np.isnan(m.iloc[-1]) else None,
                "signal":  round(float(s.iloc[-1]), 3) if not np.isnan(s.iloc[-1]) else None,
                "sma20":   round(float(sma20v), 2) if not np.isnan(sma20v) else None,
                "sma50":   round(float(sma50v), 2) if not np.isnan(sma50v) else None,
                "atr":     round(float(atr_v), 2) if not np.isnan(atr_v) else None,
                "volume":  int(df["Volume"].iloc[-1]) if "Volume" in df.columns and df["Volume"].iloc[-1] > 0 else None,
                "above_sma20": bool(close.iloc[-1] > sma20v) if not np.isnan(sma20v) else None,
                "above_sma50": bool(close.iloc[-1] > sma50v) if not np.isnan(sma50v) else None,
            })
        except: continue
    return {"results": results}

@app.get("/api/correlations")
def correlations(tickers: str = Query(...), start: str = Query(...), end: str = Query(...)):
    closes = {}
    for ticker in tickers.split(",")[:10]:
        ticker = ticker.strip().upper()
        try:
            df = get_data(ticker, start, end)
            closes[ticker] = df["Close"]
        except: continue
    if len(closes) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid tickers")
    corr = pd.DataFrame(closes).dropna().pct_change().dropna().corr()
    return {"tickers": list(closes.keys()), "matrix": corr.round(3).to_dict()}

@app.get("/api/earnings")
def earnings(ticker: str = Query(...)):
    try:
        # Use Polygon for earnings data
        r = requests.get(f"{POLY_BASE}/v2/reference/financials/{ticker.upper()}", params={"apiKey": POLY_KEY, "limit": 8}, timeout=15)
        data = r.json()
        results = data.get("results", [])
        history = []
        for item in results[:8]:
            history.append({
                "fiscalDateEnding": item.get("period_of_report_date", ""),
                "reportedEPS": item.get("financials", {}).get("income_statement", {}).get("basic_earnings_per_share", {}).get("value"),
                "estimatedEPS": None,
                "reportedRevenue": item.get("financials", {}).get("income_statement", {}).get("revenues", {}).get("value"),
            })
        return {"upcoming": {}, "history": history}
    except Exception as e:
        return {"upcoming": {}, "history": [], "error": str(e)}

@app.get("/api/monte-carlo")
def monte_carlo(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), simulations: int = Query(200), days: int = Query(252), capital: float = Query(10000)):
    df = get_data(ticker, start, end)
    rets = df["Close"].pct_change().dropna()
    mu, sigma = float(rets.mean()), float(rets.std())
    paths, finals = [], []
    for _ in range(min(simulations, 500)):
        path = [capital]
        for _ in range(days):
            path.append(path[-1] * (1 + np.random.normal(mu, sigma)))
        paths.append([round(v, 2) for v in path[::5]])
        finals.append(round(path[-1], 2))
    fa = np.array(finals)
    return {
        "simulations":   len(paths),
        "days":          days,
        "paths":         paths[:50],
        "final_median":  round(float(np.median(fa)), 2),
        "final_mean":    round(float(np.mean(fa)), 2),
        "percentile_5":  round(float(np.percentile(fa, 5)), 2),
        "percentile_25": round(float(np.percentile(fa, 25)), 2),
        "percentile_75": round(float(np.percentile(fa, 75)), 2),
        "percentile_95": round(float(np.percentile(fa, 95)), 2),
        "prob_profit":   round(float(np.mean(fa > capital) * 100), 1),
    }
