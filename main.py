from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

app = FastAPI(title="11% Trading API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production set to your Vercel domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def get_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()
    return df


def safe_float(v):
    if v is None: return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    return float(v)


def df_to_ohlcv(df: pd.DataFrame) -> list:
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


# ══════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid   = sma(series, period)
    std   = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def supertrend(df: pd.DataFrame, period=10, multiplier=3):
    a = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    close = df["Close"]
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        prev_upper = upper.iloc[i-1]
        prev_lower = lower.iloc[i-1]
        curr_upper = upper.iloc[i]
        curr_lower = lower.iloc[i]
        upper.iloc[i] = curr_upper if curr_upper < prev_upper or close.iloc[i-1] > prev_upper else prev_upper
        lower.iloc[i] = curr_lower if curr_lower > prev_lower or close.iloc[i-1] < prev_lower else prev_lower
        if close.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return st, direction


# ══════════════════════════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, strategy: str, params: dict, capital: float) -> dict:
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
        ob = params.get("overbought", 70)
        os_ = params.get("oversold", 30)
        pos = 0
        sig = []
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
        pos = 0
        sig = []
        for i, (c, u, l) in enumerate(zip(close, upper, lower)):
            if pd.isna(u): sig.append(0); continue
            if c < l: pos = 1
            elif c > u: pos = -1
            sig.append(pos)
        signals = pd.Series(sig, index=df.index)

    elif strategy == "SuperTrend":
        st, direction = supertrend(df, params.get("period",10), params.get("multiplier",3))
        signals = direction

    elif strategy == "EMA + RSI Filter":
        fast = ema(close, params.get("fast",12))
        slow = ema(close, params.get("slow",26))
        r    = rsi(close, params.get("rsi_period",14))
        trend = pd.Series(np.where(fast > slow, 1, -1), index=df.index)
        filter_ = pd.Series(np.where(r < 70, 1, np.where(r > 30, 1, 0)), index=df.index)
        signals = trend * filter_

    # Simulate trades
    position = 0
    entry_price = 0.0
    equity = [capital]
    trades = []
    balance = capital

    for i in range(1, len(df)):
        sig = signals.iloc[i]
        price = float(close.iloc[i])
        date  = str(df.index[i].date())

        if position == 0 and sig == 1:
            shares = balance / price
            position = shares
            entry_price = price
        elif position > 0 and sig == -1:
            exit_val = position * price
            pnl = exit_val - (position * entry_price)
            trades.append({ "date": date, "entry": entry_price, "exit": price, "pnl": round(pnl, 2), "pct": round(pnl / (position * entry_price) * 100, 2) })
            balance = exit_val
            position = 0

        current_val = balance + (position * price if position > 0 else 0)
        equity.append(current_val)

    # Close any open position
    if position > 0:
        final_price = float(close.iloc[-1])
        exit_val = position * final_price
        pnl = exit_val - (position * entry_price)
        trades.append({ "date": str(df.index[-1].date()), "entry": entry_price, "exit": final_price, "pnl": round(pnl, 2), "pct": round(pnl / (position * entry_price) * 100, 2) })
        balance = exit_val

    final_equity = balance
    total_return = (final_equity - capital) / capital * 100
    wins  = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    # Max drawdown
    eq_series = pd.Series(equity)
    roll_max  = eq_series.cummax()
    drawdown  = (eq_series - roll_max) / roll_max * 100
    max_dd    = float(drawdown.min())

    # Sharpe (simplified)
    returns = pd.Series(equity).pct_change().dropna()
    sharpe  = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    return {
        "total_return":  round(total_return, 2),
        "final_equity":  round(final_equity, 2),
        "total_trades":  len(trades),
        "win_rate":      round(win_rate, 1),
        "max_drawdown":  round(max_dd, 2),
        "sharpe_ratio":  round(sharpe, 2),
        "avg_win":       round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss":      round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
        "trades":        trades[-50:],  # last 50 trades
        "equity_curve":  [round(v, 2) for v in equity],
        "equity_dates":  [str(d.date()) for d in df.index],
    }


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return { "status": "11% API running", "version": "1.0.0" }


@app.get("/api/ohlcv")
def get_ohlcv(
    ticker: str = Query(...),
    start:  str = Query(...),
    end:    str = Query(...),
    interval: str = Query("1d")
):
    df = get_data(ticker, start, end, interval)
    return { "ticker": ticker, "data": df_to_ohlcv(df) }


@app.get("/api/ticker-info")
def ticker_info(ticker: str = Query(...)):
    try:
        t = yf.Ticker(ticker)
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
    except:
        raise HTTPException(status_code=404, detail="Ticker not found")


class BacktestRequest(BaseModel):
    ticker:   str
    start:    str
    end:      str
    strategy: str
    capital:  float = 10000
    params:   dict  = {}


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    try:
        df = get_data(req.ticker, req.start, req.end)
        result = run_backtest(df, req.strategy, req.params, req.capital)
        return { "ticker": req.ticker, "strategy": req.strategy, **result }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/replay-data")
def replay_data(
    ticker:   str = Query(...),
    start:    str = Query(...),
    end:      str = Query(...),
    interval: str = Query("1d")
):
    df = get_data(ticker, start, end, interval)
    return { "ticker": ticker, "bars": df_to_ohlcv(df) }


@app.get("/api/indicators")
def get_indicators(
    ticker:   str = Query(...),
    start:    str = Query(...),
    end:      str = Query(...),
    indicator: str = Query(...)
):
    df   = get_data(ticker, start, end)
    close = df["Close"]
    result = {}

    if indicator == "SMA":
        result["sma20"]  = [safe_float(v) for v in sma(close, 20)]
        result["sma50"]  = [safe_float(v) for v in sma(close, 50)]
        result["sma200"] = [safe_float(v) for v in sma(close, 200)]
    elif indicator == "EMA":
        result["ema12"] = [safe_float(v) for v in ema(close, 12)]
        result["ema26"] = [safe_float(v) for v in ema(close, 26)]
    elif indicator == "RSI":
        result["rsi"] = [safe_float(v) for v in rsi(close)]
    elif indicator == "MACD":
        m, s, h = macd(close)
        result["macd"]      = [safe_float(v) for v in m]
        result["signal"]    = [safe_float(v) for v in s]
        result["histogram"] = [safe_float(v) for v in h]
    elif indicator == "BB":
        u, m, l = bollinger_bands(close)
        result["upper"] = [safe_float(v) for v in u]
        result["mid"]   = [safe_float(v) for v in m]
        result["lower"] = [safe_float(v) for v in l]
    elif indicator == "ATR":
        result["atr"] = [safe_float(v) for v in atr(df)]

    result["dates"] = [str(d.date()) for d in df.index]
    return result


@app.get("/api/screener")
def screener(
    tickers: str = Query(...),  # comma separated
    indicator: str = Query("RSI"),
    period:   str = Query("3mo")
):
    results = []
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    for ticker in tickers.split(",")[:20]:  # max 20
        ticker = ticker.strip().upper()
        try:
            df    = get_data(ticker, start, end)
            close = df["Close"]
            r     = rsi(close).iloc[-1]
            m, s, _ = macd(close)
            price = float(close.iloc[-1])
            change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)

            results.append({
                "ticker":  ticker,
                "price":   round(price, 2),
                "change":  round(change, 2),
                "rsi":     round(float(r), 1) if not np.isnan(r) else None,
                "macd":    round(float(m.iloc[-1]), 3) if not np.isnan(m.iloc[-1]) else None,
                "signal":  round(float(s.iloc[-1]), 3) if not np.isnan(s.iloc[-1]) else None,
                "volume":  int(df["Volume"].iloc[-1]) if "Volume" in df.columns else None,
            })
        except:
            continue

    return { "results": results }


@app.get("/api/earnings")
def earnings(ticker: str = Query(...)):
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        hist = t.earnings_history
        return {
            "upcoming": cal.to_dict() if cal is not None else {},
            "history":  hist.to_dict(orient="records")[:8] if hist is not None else [],
        }
    except Exception as e:
        return { "upcoming": {}, "history": [], "error": str(e) }


@app.get("/api/correlations")
def correlations(
    tickers: str = Query(...),
    start:   str = Query(...),
    end:     str = Query(...)
):
    closes = {}
    for ticker in tickers.split(",")[:10]:
        ticker = ticker.strip().upper()
        try:
            df = get_data(ticker, start, end)
            closes[ticker] = df["Close"]
        except:
            continue
    if len(closes) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid tickers")
    combined = pd.DataFrame(closes).dropna()
    corr = combined.pct_change().dropna().corr()
    return { "tickers": list(closes.keys()), "matrix": corr.round(3).to_dict() }


@app.get("/api/monte-carlo")
def monte_carlo(
    ticker:      str   = Query(...),
    start:       str   = Query(...),
    end:         str   = Query(...),
    simulations: int   = Query(500),
    days:        int   = Query(252),
    capital:     float = Query(10000)
):
    df    = get_data(ticker, start, end)
    close = df["Close"]
    returns = close.pct_change().dropna()
    mu    = float(returns.mean())
    sigma = float(returns.std())

    paths = []
    finals = []
    for _ in range(min(simulations, 1000)):
        path = [capital]
        for _ in range(days):
            path.append(path[-1] * (1 + np.random.normal(mu, sigma)))
        paths.append([round(v, 2) for v in path[::5]])  # sample every 5 days
        finals.append(round(path[-1], 2))

    finals_arr = np.array(finals)
    return {
        "simulations":   len(paths),
        "days":          days,
        "paths":         paths[:50],  # send 50 paths for visualization
        "final_median":  round(float(np.median(finals_arr)), 2),
        "final_mean":    round(float(np.mean(finals_arr)), 2),
        "percentile_5":  round(float(np.percentile(finals_arr, 5)), 2),
        "percentile_25": round(float(np.percentile(finals_arr, 25)), 2),
        "percentile_75": round(float(np.percentile(finals_arr, 75)), 2),
        "percentile_95": round(float(np.percentile(finals_arr, 95)), 2),
        "prob_profit":   round(float(np.mean(finals_arr > capital) * 100), 1),
    }
