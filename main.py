from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

app = FastAPI(title="11% Trading API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Bypass Yahoo Finance blocking with a proper session ───────────────────────
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    })
    return s

def get_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    try:
        session = make_session()
        t = yf.Ticker(ticker, session=session)
        df = t.history(start=start, end=end, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            # Try download as fallback
            df = yf.download(ticker, start=start, end=end, interval=interval, 
                           auto_adjust=True, progress=False, session=session)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch {ticker}: {str(e)}")

def safe_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except:
        return None

def df_to_ohlcv(df):
    return [{
        "time":   int(pd.Timestamp(ts).timestamp()),
        "open":   safe_float(row.get("Open")),
        "high":   safe_float(row.get("High")),
        "low":    safe_float(row.get("Low")),
        "close":  safe_float(row.get("Close")),
        "volume": safe_float(row.get("Volume")),
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
        f, s = sma(close, params.get("fast",20)), sma(close, params.get("slow",50))
        signals = pd.Series(np.where(f > s, 1, -1), index=df.index)
    elif strategy == "EMA Crossover":
        f, s = ema(close, params.get("fast",12)), ema(close, params.get("slow",26))
        signals = pd.Series(np.where(f > s, 1, -1), index=df.index)
    elif strategy == "RSI":
        r = rsi(close, params.get("period",14))
        ob, os_ = params.get("overbought",70), params.get("oversold",30)
        pos, sig = 0, []
        for v in r:
            if pd.isna(v): sig.append(0); continue
            if v < os_: pos = 1
            elif v > ob: pos = -1
            sig.append(pos)
        signals = pd.Series(sig, index=df.index)
    elif strategy == "MACD":
        m, s, _ = macd(close)
        signals = pd.Series(np.where(m > s, 1, -1), index=df.index)
    elif strategy == "Bollinger Bands":
        upper, _, lower = bollinger(close)
        pos, sig = 0, []
        for c2, u, l in zip(close, upper, lower):
            if pd.isna(u): sig.append(0); continue
            if c2 < l: pos = 1
            elif c2 > u: pos = -1
            sig.append(pos)
        signals = pd.Series(sig, index=df.index)
    elif strategy == "SuperTrend":
        signals = supertrend(df)
    elif strategy == "EMA + RSI Filter":
        f = ema(close, params.get("fast",12))
        s = ema(close, params.get("slow",26))
        r = rsi(close, params.get("rsi_period",14))
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

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    eq = pd.Series(equity)
    drawdown = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    return {
        "total_return":  round((balance - capital) / capital * 100, 2),
        "final_equity":  round(balance, 2),
        "total_trades":  len(trades),
        "win_rate":      round(len(wins)/len(trades)*100, 1) if trades else 0,
        "max_drawdown":  round(float(drawdown), 2),
        "sharpe_ratio":  round(sharpe, 2),
        "avg_win":       round(sum(t["pnl"] for t in wins)/len(wins), 2) if wins else 0,
        "avg_loss":      round(sum(t["pnl"] for t in losses)/len(losses), 2) if losses else 0,
        "trades":        trades[-50:],
        "equity_curve":  [round(v, 2) for v in equity],
        "equity_dates":  [str(d.date()) for d in df.index],
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status": "11% API running", "version": "2.0.0"}

@app.get("/api/ohlcv")
def get_ohlcv(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), interval: str = Query("1d")):
    df = get_data(ticker, start, end, interval)
    return {"ticker": ticker, "data": df_to_ohlcv(df)}

@app.get("/api/ticker-info")
def ticker_info(ticker: str = Query(...)):
    try:
        session = make_session()
        t = yf.Ticker(ticker, session=session)
        info = t.info
        return {
            "name":          info.get("longName", ticker),
            "sector":        info.get("sector", ""),
            "market_cap":    info.get("marketCap"),
            "pe_ratio":      info.get("trailingPE"),
            "52w_high":      info.get("fiftyTwoWeekHigh"),
            "52w_low":       info.get("fiftyTwoWeekLow"),
            "avg_volume":    info.get("averageVolume"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
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
    return {"ticker": ticker, "bars": df_to_ohlcv(df)}

@app.get("/api/indicators")
def get_indicators(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), indicator: str = Query(...)):
    df = get_data(ticker, start, end)
    close = df["Close"]
    result = {}
    if indicator == "RSI": result["rsi"] = [safe_float(v) for v in rsi(close)]
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
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    for ticker in tickers.split(",")[:20]:
        ticker = ticker.strip().upper()
        try:
            df = get_data(ticker, start, end)
            close = df["Close"]
            r = rsi(close).iloc[-1]
            m, s, _ = macd(close)
            results.append({
                "ticker": ticker,
                "price":  round(float(close.iloc[-1]), 2),
                "change": round(float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100), 2),
                "rsi":    round(float(r), 1) if not np.isnan(r) else None,
                "macd":   round(float(m.iloc[-1]), 3) if not np.isnan(m.iloc[-1]) else None,
                "signal": round(float(s.iloc[-1]), 3) if not np.isnan(s.iloc[-1]) else None,
                "volume": int(df["Volume"].iloc[-1]) if "Volume" in df.columns else None,
            })
        except: continue
    return {"results": results}

@app.get("/api/earnings")
def earnings(ticker: str = Query(...)):
    try:
        session = make_session()
        t = yf.Ticker(ticker, session=session)
        cal = t.calendar
        hist = t.earnings_history
        return {
            "upcoming": cal.to_dict() if cal is not None else {},
            "history":  hist.to_dict(orient="records")[:8] if hist is not None else [],
        }
    except Exception as e:
        return {"upcoming": {}, "history": [], "error": str(e)}

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

@app.get("/api/monte-carlo")
def monte_carlo(ticker: str = Query(...), start: str = Query(...), end: str = Query(...), simulations: int = Query(500), days: int = Query(252), capital: float = Query(10000)):
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
