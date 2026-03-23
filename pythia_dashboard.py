"""
pythia_dashboard.py
────────────────────────────────────────────────────────────────────────────────
Pythia: Momentum Investing — Interactive Dashboard
────────────────────────────────────────────────────────────────────────────────

Standalone Dash application that sits on top of sp500_prices.duckdb and
reuses every signal / backtest / metric function from Pythia_jupiter.ipynb.

Usage (from repo root):
    uv run pythia_dashboard.py

Then open http://127.0.0.1:8050 in your browser.

Requirements (add with: uv add dash dash-bootstrap-components plotly):
    dash >= 2.17
    dash-bootstrap-components >= 1.6
    plotly >= 5.22
    duckdb, pandas, numpy, scipy, sklearn, lightgbm   (already installed)

Tabs
────
  Tab 1 – Strategy Backtester   : run a single signal with custom parameters
  Tab 2 – Stock Screener        : live momentum ranking of the clean universe
  Tab 3 – Signal Comparison     : run all 4 signals side-by-side
  Tab 4 – Fama-French           : risk-adjust returns against FF3 factors

ⓘ tooltip convention
─────────────────────
  Every chart title and section heading has a small grey ⓘ icon.
  Hovering it shows a dbc.Tooltip with a plain-English explanation.
  No click required, no modal — pure CSS hover via dbc.Tooltip.
────────────────────────────────────────────────────────────────────────────────
"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================
# if running without uv (e.g. plain pip env):
# pip install dash dash-bootstrap-components plotly duckdb pandas numpy scipy scikit-learn lightgbm -q

import io
import time
import warnings
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import duckdb

from scipy import stats

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import roc_auc_score, f1_score
from sklearn.preprocessing   import StandardScaler

import plotly.graph_objects as go
from   plotly.subplots      import make_subplots

import dash
from   dash                       import dcc, html, Input, Output, State, callback
import dash_bootstrap_components  as dbc

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH    = Path("sp500_prices.duckdb")   # DuckDB price database (same dir as this file)
FF_CACHE   = Path("ff3_factors.csv")       # Fama-French cache — written on first use

# ── Finance constants — identical to notebook Section 1 ──────────────────────
TRADING_DAYS         = 252       # exchange-open days per calendar year
RF_ANNUAL            = 0.04      # annual risk-free rate (4 %)
BACKTEST_START       = "2015-01-01"

# ── Backtest defaults ─────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 10        # default flat cost per side in basis points
TOP_DECILE           = 0.10      # top 10 % of ranked stocks → long portfolio
BOTTOM_DECILE        = 0.10      # bottom 10 % → short portfolio

# ── Data quality thresholds — identical to notebook Section 3 ────────────────
MIN_HISTORY_YEARS    = 8
MAX_CONSEC_MISSING   = 5
MAX_DAILY_RETURN     = 0.40

# ── Colour palette — identical to notebook Section 1 ─────────────────────────
# Defined as a dict so every chart always uses the same colour for the same signal
COLOURS = {
    "MOM_12_1"  : "#2196F3",   # blue
    "MOM_3_1"   : "#4CAF50",   # green
    "MA_Cross"  : "#FF9800",   # orange
    "RSI_14"    : "#9C27B0",   # purple
    "Composite" : "#F44336",   # red
    "ML"        : "#00BCD4",   # cyan
    "Benchmark" : "#9E9E9E",   # grey
}

SIGNAL_COLS   = ["MOM_12_1", "MOM_3_1", "MA_Cross", "RSI_14"]
SIGNAL_LABELS = {
    "MOM_12_1"  : "Momentum 12-1",
    "MOM_3_1"   : "Momentum 3-1",
    "MA_Cross"  : "MA Crossover",
    "RSI_14"    : "RSI-14",
    "Composite" : "Composite",
}

# ── Plotly layout defaults — applied to every figure ─────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="'IBM Plex Mono', monospace", size=12, color="#E0E0E0"),
    legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin        = dict(l=50, r=20, t=40, b=40),
    hoverlabel    = dict(bgcolor="#1A1A2E", font_size=12, font_family="'IBM Plex Mono', monospace"),
    xaxis         = dict(gridcolor="#2A2A3E", linecolor="#3A3A5E", zerolinecolor="#3A3A5E"),
    yaxis         = dict(gridcolor="#2A2A3E", linecolor="#3A3A5E", zerolinecolor="#3A3A5E"),
)


# =============================================================================
# 2. DATA LOADERS
# =============================================================================
# All DuckDB queries follow the notebook pattern:
#   con = duckdb.connect(...)  →  query  →  con.close()
# The connection is never left open between function calls.

def load_prices() -> pd.DataFrame:
    """
    Load close prices for the full universe from DuckDB.
    Returns a wide DataFrame: index = date, columns = ticker.
    Applies the same quality filters as notebook Section 3.
    """
    con        = duckdb.connect(str(DB_PATH), read_only=True)
    prices_raw = con.execute("""
        SELECT date, ticker, close
        FROM   daily_prices
        ORDER  BY ticker, date
    """).df()
    con.close()

    prices_raw["date"] = pd.to_datetime(prices_raw["date"])

    # ── Apply quality filters (Section 3 logic) ───────────────────────────────
    # Check 1: sufficient history
    counts = prices_raw.groupby("ticker")["date"].count()
    min_days = int(MIN_HISTORY_YEARS * TRADING_DAYS)
    ok_tickers = counts[counts >= min_days].index

    # Check 2: no abnormal single-day returns
    wide   = prices_raw.pivot(index="date", columns="ticker", values="close")
    rets   = wide.pct_change()
    bad    = (rets.abs() > MAX_DAILY_RETURN).any()
    ok_tickers = ok_tickers.difference(bad[bad].index)

    # Return wide format, clean universe only
    return wide[ok_tickers].sort_index()


def load_benchmark() -> pd.Series:
    """
    Load S&P 500 benchmark (^GSPC) close prices from DuckDB.
    Returns a Series indexed by date.
    Falls back to yfinance if not present in the database.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute("""
        SELECT date, close
        FROM   daily_prices
        WHERE  ticker = '^GSPC'
        ORDER  BY date
    """).df()
    con.close()

    if df.empty:
        # ── Fallback: fetch from yfinance ────────────────────────────────────
        import yfinance as yf
        raw = yf.download("^GSPC", start=BACKTEST_START, auto_adjust=True, progress=False)
        return raw["Close"].rename("Benchmark")

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename("Benchmark")


def fetch_ff3_factors() -> pd.DataFrame:
    """
    Fetch Fama-French 3-factor monthly returns from Kenneth French's website.
    Caches result to FF_CACHE (ff3_factors.csv) — subsequent calls use the cache.

    Returns a DataFrame with columns: [Mkt-RF, SMB, HML, RF] indexed by month-end date.
    """
    # ── Use cache if it exists and is less than 30 days old ──────────────────
    if FF_CACHE.exists():
        age_days = (date.today() - date.fromtimestamp(FF_CACHE.stat().st_mtime)).days
        if age_days < 30:
            df = pd.read_csv(FF_CACHE, index_col=0, parse_dates=True)
            return df

    # ── Fetch from Kenneth French Data Library ────────────────────────────────
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_CSV.zip"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            fname = [n for n in z.namelist() if n.endswith(".CSV")][0]
            raw   = z.read(fname).decode("utf-8")
    except Exception as exc:
        # If offline and cache exists (even stale), use it
        if FF_CACHE.exists():
            return pd.read_csv(FF_CACHE, index_col=0, parse_dates=True)
        raise RuntimeError(f"Cannot fetch Fama-French data and no cache exists: {exc}")

    # ── Parse: skip header lines until the numeric data starts ───────────────
    lines  = raw.strip().split("\n")
    start  = next(i for i, l in enumerate(lines) if l.strip()[:6].isdigit())
    end    = next(
        (i for i, l in enumerate(lines[start:], start) if l.strip() == ""),
        len(lines)
    )
    block  = "\n".join(lines[start:end])
    df     = pd.read_csv(io.StringIO(block), header=None,
                         names=["yyyymm", "Mkt-RF", "SMB", "HML", "RF"])
    df     = df[df["yyyymm"].astype(str).str.len() == 6].copy()
    df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df     = df.set_index("date")[["Mkt-RF", "SMB", "HML", "RF"]]
    df     = df.apply(pd.to_numeric, errors="coerce").dropna() / 100.0  # convert % to decimal

    # ── Write cache ───────────────────────────────────────────────────────────
    df.to_csv(FF_CACHE)
    return df


# =============================================================================
# 3. SIGNAL ENGINES
# =============================================================================
# Pure-Python functions that compute momentum signals on a price matrix.
# All inputs are wide DataFrames (date × ticker).  All outputs are the same shape.

def compute_mom_12_1(prices: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """
    Classic momentum signal: total return over [lookback] months excluding
    the most recent month (skip-month to avoid short-term reversal bias).

    Formula: P(t-1) / P(t-1-lookback) - 1
    """
    # Resample to month-end prices before computing to align with rebalancing
    monthly = prices.resample("ME").last()
    # shift(1) skips the most recent month; shift(1+lookback) is the start
    signal  = monthly.shift(1) / monthly.shift(1 + lookback) - 1
    return signal


def compute_mom_3_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Short-term momentum: 3-month return excluding most recent month."""
    return compute_mom_12_1(prices, lookback=3)


def compute_ma_cross(prices: pd.DataFrame,
                     fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """
    Moving-average crossover signal.
    +1 when fast MA > slow MA (uptrend), -1 when fast MA < slow MA (downtrend).
    Resampled to month-end before returning so it aligns with other signals.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal  = np.sign(fast_ma - slow_ma)
    return signal.resample("ME").last()


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index over [window] trading days.
    RSI > 50 indicates upward momentum; we z-score before ranking so all
    signals are on a comparable scale.
    Resampled to month-end.
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return rsi.resample("ME").last()


def compute_all_signals(prices: pd.DataFrame, lookback: int = 12) -> dict:
    """
    Compute all four signals and a composite (equal-weight average of z-scores).
    Returns a dict: signal_name → wide monthly DataFrame.
    """
    signals = {
        "MOM_12_1" : compute_mom_12_1(prices, lookback),
        "MOM_3_1"  : compute_mom_3_1(prices),
        "MA_Cross" : compute_ma_cross(prices),
        "RSI_14"   : compute_rsi(prices),
    }
    # ── Composite: z-score each signal then average ───────────────────────────
    z_scores = {k: (v - v.mean(axis=1).values[:, None]) /
                    v.std(axis=1).replace(0, np.nan).values[:, None]
                for k, v in signals.items()}
    signals["Composite"] = sum(z_scores.values()) / len(z_scores)
    return signals


# =============================================================================
# 4. BACKTEST ENGINE
# =============================================================================
# Walk-forward monthly backtest — identical logic to notebook Section 6.

def run_backtest(
    prices      : pd.DataFrame,
    signal_name : str,
    lookback    : int   = 12,
    freq        : str   = "ME",
    cost_bps    : float = 10.0,
    long_short  : bool  = True,
) -> pd.DataFrame:
    """
    Walk-forward backtest.

    At each rebalancing date:
      1. Rank all stocks by their signal score
      2. Buy equal-weight top decile (long leg)
      3. Short equal-weight bottom decile (short leg, only if long_short=True)
      4. Hold until next rebalancing date
      5. Deduct transaction cost on both legs at every rebalance

    Returns a DataFrame with columns:
        [strategy_return, benchmark_return, cumulative_strategy, cumulative_benchmark]
    indexed by month-end date.
    """
    # ── Compute signals ───────────────────────────────────────────────────────
    all_sigs = compute_all_signals(prices, lookback)
    sig_df   = all_sigs[signal_name]

    # ── Monthly close prices and returns ─────────────────────────────────────
    monthly_prices  = prices.resample(freq).last()
    monthly_returns = monthly_prices.pct_change()

    # ── Benchmark: equal-weight all clean-universe stocks ────────────────────
    benchmark_ret = monthly_returns.mean(axis=1)

    cost = cost_bps / 10_000.0   # convert bps → decimal

    strategy_returns = []
    dates            = []

    rebal_dates = sig_df.index[sig_df.index >= BACKTEST_START]

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        scores = sig_df.loc[rebal_date].dropna()
        if len(scores) < 20:          # need enough stocks to form deciles
            continue

        n_long  = max(1, int(len(scores) * TOP_DECILE))
        n_short = max(1, int(len(scores) * BOTTOM_DECILE))

        long_tickers  = scores.nlargest(n_long).index.tolist()
        short_tickers = scores.nsmallest(n_short).index.tolist()

        next_date = rebal_dates[i + 1]

        # ── Returns over the holding period ──────────────────────────────────
        # .loc uses label-based slicing — both ends inclusive
        period_rets = monthly_returns.loc[
            (monthly_returns.index > rebal_date) &
            (monthly_returns.index <= next_date)
        ]

        if period_rets.empty:
            continue

        long_ret  = period_rets[long_tickers].mean(axis=1).mean()
        long_ret -= cost   # entry + exit cost split across the period

        if long_short:
            short_ret  = period_rets[short_tickers].mean(axis=1).mean()
            short_ret -= cost
            period_strat_ret = long_ret - short_ret   # long-short portfolio
        else:
            period_strat_ret = long_ret               # long-only portfolio

        strategy_returns.append(period_strat_ret)
        dates.append(next_date)

    if not dates:
        return pd.DataFrame()

    result = pd.DataFrame(
        {"strategy_return"  : strategy_returns,
         "benchmark_return" : benchmark_ret.reindex(dates).values},
        index = dates,
    )

    # ── Cumulative returns (wealth index starting at 1.0) ────────────────────
    result["cumulative_strategy"]  = (1 + result["strategy_return"]).cumprod()
    result["cumulative_benchmark"] = (1 + result["benchmark_return"]).cumprod()

    return result


# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================

def compute_metrics(bt: pd.DataFrame) -> dict:
    """
    Compute standard performance metrics from a backtest result DataFrame.
    Returns a dict with all metrics as floats.
    """
    if bt.empty:
        return {k: float("nan") for k in
                ["ann_return", "sharpe", "max_drawdown", "calmar",
                 "ann_vol", "sortino", "win_rate"]}

    rets = bt["strategy_return"].dropna()
    n    = len(rets)
    if n == 0:
        return {k: float("nan") for k in
                ["ann_return", "sharpe", "max_drawdown", "calmar",
                 "ann_vol", "sortino", "win_rate"]}

    # ── Annualised return and volatility ──────────────────────────────────────
    ann_return = (1 + rets).prod() ** (12 / n) - 1   # monthly → annual (×12 periods/year)
    ann_vol    = rets.std() * np.sqrt(12)

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    rf_monthly = RF_ANNUAL / 12
    sharpe     = (ann_return - RF_ANNUAL) / ann_vol if ann_vol > 0 else float("nan")

    # ── Maximum drawdown ──────────────────────────────────────────────────────
    cum         = bt["cumulative_strategy"]
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # ── Calmar ratio: annualised return / |max drawdown| ──────────────────────
    calmar = ann_return / abs(max_dd) if max_dd != 0 else float("nan")

    # ── Sortino ratio: penalises only downside volatility ─────────────────────
    downside_rets = rets[rets < rf_monthly]
    downside_vol  = downside_rets.std() * np.sqrt(12) if len(downside_rets) > 1 else np.nan
    sortino       = (ann_return - RF_ANNUAL) / downside_vol if downside_vol and downside_vol > 0 else float("nan")

    # ── Win rate: fraction of months with positive return ─────────────────────
    win_rate = (rets > 0).mean()

    return {
        "ann_return"  : ann_return,
        "ann_vol"     : ann_vol,
        "sharpe"      : sharpe,
        "max_drawdown": max_dd,
        "calmar"      : calmar,
        "sortino"     : sortino,
        "win_rate"    : win_rate,
    }


def compute_rolling_sharpe(bt: pd.DataFrame, window: int = 12) -> pd.Series:
    """Rolling annualised Sharpe ratio with a [window]-month rolling window."""
    rets      = bt["strategy_return"].dropna()
    rf_m      = RF_ANNUAL / 12
    roll_ret  = rets.rolling(window).mean()
    roll_vol  = rets.rolling(window).std()
    sharpe    = (roll_ret - rf_m) / roll_vol * np.sqrt(12)
    return sharpe


def compute_drawdown_series(bt: pd.DataFrame) -> pd.Series:
    """Returns the full drawdown time series (always ≤ 0)."""
    cum         = bt["cumulative_strategy"]
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


# =============================================================================
# 6. CHART BUILDERS
# =============================================================================
# Each function returns a go.Figure ready to pass to dcc.Graph.
# All figures share PLOTLY_LAYOUT for visual consistency.

def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the global layout template to a figure."""
    layout = dict(PLOTLY_LAYOUT)
    if title:
        layout["title"] = dict(text=title, font=dict(size=14), x=0.01)
    fig.update_layout(**layout)
    return fig


def fig_cumulative_return(bt: pd.DataFrame, signal_name: str) -> go.Figure:
    """Cumulative wealth index: strategy vs S&P 500 benchmark."""
    fig = go.Figure()
    if bt.empty:
        return _apply_layout(fig)

    fig.add_trace(go.Scatter(
        x    = bt.index,
        y    = bt["cumulative_strategy"],
        name = SIGNAL_LABELS.get(signal_name, signal_name),
        line = dict(color=COLOURS.get(signal_name, "#ffffff"), width=2.5),
        hovertemplate = "%{x|%b %Y}<br>Strategy: $%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x    = bt.index,
        y    = bt["cumulative_benchmark"],
        name = "S&P 500 (B&H)",
        line = dict(color=COLOURS["Benchmark"], width=2, dash="dot"),
        hovertemplate = "%{x|%b %Y}<br>Benchmark: $%{y:.3f}<extra></extra>",
    ))

    # ── Shade outperformance regions ─────────────────────────────────────────
    diff = bt["cumulative_strategy"] - bt["cumulative_benchmark"]
    fig.add_trace(go.Scatter(
        x         = list(bt.index) + list(bt.index[::-1]),
        y         = list(bt["cumulative_strategy"]) + list(bt["cumulative_benchmark"][::-1]),
        fill      = "toself",
        fillcolor = "rgba(33,150,243,0.08)",
        line      = dict(color="rgba(0,0,0,0)"),
        showlegend= False,
        hoverinfo = "skip",
    ))

    fig.update_yaxes(tickprefix="$", tickformat=".2f")
    return _apply_layout(fig)


def fig_rolling_sharpe(bt: pd.DataFrame, window: int = 12) -> go.Figure:
    """Rolling Sharpe ratio with green/red shading."""
    fig = go.Figure()
    if bt.empty:
        return _apply_layout(fig)

    rs = compute_rolling_sharpe(bt, window).dropna()

    # ── Positive shading (Sharpe > 0) ─────────────────────────────────────────
    pos = rs.clip(lower=0)
    fig.add_trace(go.Scatter(
        x=list(rs.index) + list(rs.index[::-1]),
        y=list(pos) + [0] * len(pos),
        fill="toself", fillcolor="rgba(76,175,80,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    # ── Negative shading (Sharpe < 0) ─────────────────────────────────────────
    neg = rs.clip(upper=0)
    fig.add_trace(go.Scatter(
        x=list(rs.index) + list(rs.index[::-1]),
        y=list(neg) + [0] * len(neg),
        fill="toself", fillcolor="rgba(244,67,54,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=rs.index, y=rs,
        name=f"{window}M Rolling Sharpe",
        line=dict(color="#FFD700", width=2),
        hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#FF4444", width=1, dash="dash"))
    fig.add_hline(y=1, line=dict(color="#4CAF50", width=1, dash="dot"),
                  annotation_text="Sharpe = 1", annotation_position="right")

    return _apply_layout(fig)


def fig_drawdown(bt: pd.DataFrame) -> go.Figure:
    """Drawdown profile — filled red area chart."""
    fig = go.Figure()
    if bt.empty:
        return _apply_layout(fig)

    dd = compute_drawdown_series(bt)

    fig.add_trace(go.Scatter(
        x         = dd.index,
        y         = dd * 100,
        fill      = "tozeroy",
        fillcolor = "rgba(244,67,54,0.25)",
        line      = dict(color="#F44336", width=1.5),
        name      = "Drawdown",
        hovertemplate = "%{x|%b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(ticksuffix="%")
    return _apply_layout(fig)


def fig_annual_heatmap(bt: pd.DataFrame, signal_name: str) -> go.Figure:
    """Annual return heatmap: strategy vs benchmark by calendar year."""
    fig = go.Figure()
    if bt.empty:
        return _apply_layout(fig)

    # ── Compute annual returns from monthly ───────────────────────────────────
    annual = bt[["strategy_return", "benchmark_return"]].copy()
    annual.index = pd.to_datetime(annual.index)
    annual["year"] = annual.index.year
    ann_rets = annual.groupby("year").apply(lambda g: (1 + g.drop("year", axis=1)).prod() - 1)

    years    = ann_rets.index.tolist()
    strat    = (ann_rets["strategy_return"] * 100).round(1).tolist()
    bench    = (ann_rets["benchmark_return"] * 100).round(1).tolist()

    z      = [bench, strat]
    labels = ["S&P 500", SIGNAL_LABELS.get(signal_name, signal_name)]
    text   = [[f"{v:.1f}%" for v in row] for row in z]

    fig.add_trace(go.Heatmap(
        z          = z,
        x          = years,
        y          = labels,
        text       = text,
        texttemplate = "%{text}",
        textfont   = dict(size=11),
        colorscale = [
            [0.0, "#C62828"], [0.4, "#EF5350"],
            [0.5, "#1A1A2E"],
            [0.6, "#66BB6A"], [1.0, "#2E7D32"],
        ],
        zmid       = 0,
        showscale  = False,
        hovertemplate = "%{y}<br>%{x}: %{text}<extra></extra>",
    ))

    fig.update_xaxes(tickmode="array", tickvals=years)
    return _apply_layout(fig)


def fig_signal_comparison(results: dict) -> go.Figure:
    """Multi-line cumulative return chart for all signals + benchmark."""
    fig = go.Figure()
    added_benchmark = False

    for sig_name, bt in results.items():
        if bt is None or bt.empty:
            continue

        fig.add_trace(go.Scatter(
            x    = bt.index,
            y    = bt["cumulative_strategy"],
            name = SIGNAL_LABELS.get(sig_name, sig_name),
            line = dict(color=COLOURS.get(sig_name, "#ffffff"), width=2.5),
            hovertemplate = f"{SIGNAL_LABELS.get(sig_name, sig_name)}<br>%{{x|%b %Y}}: $%{{y:.3f}}<extra></extra>",
        ))
        if not added_benchmark:
            fig.add_trace(go.Scatter(
                x    = bt.index,
                y    = bt["cumulative_benchmark"],
                name = "S&P 500 (B&H)",
                line = dict(color=COLOURS["Benchmark"], width=2, dash="dot"),
                hovertemplate = "S&P 500<br>%{x|%b %Y}: $%{y:.3f}<extra></extra>",
            ))
            added_benchmark = True

    fig.update_yaxes(tickprefix="$", tickformat=".2f")
    return _apply_layout(fig)


def fig_metrics_bars(metrics_dict: dict) -> go.Figure:
    """Grouped bar chart comparing key metrics across all signals."""
    signals  = [SIGNAL_LABELS.get(k, k) for k in metrics_dict.keys()]
    colours  = [COLOURS.get(k, "#ffffff") for k in metrics_dict.keys()]
    metric_keys   = ["ann_return", "sharpe", "max_drawdown", "calmar"]
    metric_labels = ["Ann. Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Calmar Ratio"]

    fig = make_subplots(rows=1, cols=4, subplot_titles=metric_labels)

    for col_i, (m_key, m_label) in enumerate(zip(metric_keys, metric_labels), 1):
        vals = []
        for sig_key, mets in metrics_dict.items():
            v = mets.get(m_key, float("nan"))
            if m_key in ("ann_return", "max_drawdown"):
                v = v * 100  # express as percentage
            vals.append(round(v, 2) if not np.isnan(v) else 0)

        bar_colours = [
            "#4CAF50" if v >= 0 else "#F44336"
            for v in vals
        ]

        fig.add_trace(go.Bar(
            x           = signals,
            y           = vals,
            marker_color= bar_colours,
            showlegend  = False,
            hovertemplate = f"%{{x}}<br>{m_label}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=col_i)

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=40))
    return fig


def fig_ff_factor_bars(alpha: float, coefs: dict, conf_ints: dict) -> go.Figure:
    """Horizontal bar chart of FF3 factor exposures with confidence intervals."""
    factors = ["Alpha (ann.)", "Mkt-RF", "SMB", "HML"]
    values  = [alpha * 100, coefs.get("Mkt-RF", 0),
                coefs.get("SMB", 0), coefs.get("HML", 0)]
    errors  = [0,
               conf_ints.get("Mkt-RF", 0),
               conf_ints.get("SMB", 0),
               conf_ints.get("HML", 0)]
    colours = ["#FFD700" if v >= 0 else "#F44336" for v in values]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = factors,
        orientation = "h",
        marker_color= colours,
        error_x     = dict(type="data", array=errors, color="#888"),
        hovertemplate = "%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#888", width=1))
    return _apply_layout(fig)


def fig_rolling_alpha(bt: pd.DataFrame, ff: pd.DataFrame, window: int = 12) -> go.Figure:
    """Rolling 12-month alpha from FF3 regression."""
    fig = go.Figure()
    if bt.empty or ff.empty:
        return _apply_layout(fig)

    monthly = bt["strategy_return"].dropna()
    ff_m    = ff[["Mkt-RF", "SMB", "HML", "RF"]].copy()
    ff_m.index = pd.to_datetime(ff_m.index)

    aligned = monthly.to_frame("ret").join(ff_m, how="inner")
    if aligned.empty:
        return _apply_layout(fig)

    aligned["excess"] = aligned["ret"] - aligned["RF"]
    alphas  = []
    idx     = []

    for end_i in range(window, len(aligned) + 1):
        chunk = aligned.iloc[end_i - window : end_i]
        X     = chunk[["Mkt-RF", "SMB", "HML"]].values
        y     = chunk["excess"].values
        if len(y) < window:
            continue
        X_   = np.column_stack([np.ones(len(X)), X])
        try:
            coef = np.linalg.lstsq(X_, y, rcond=None)[0]
            alphas.append(coef[0] * 12 * 100)   # annualise and express as %
        except Exception:
            alphas.append(np.nan)
        idx.append(aligned.index[end_i - 1])

    roll_alpha = pd.Series(alphas, index=idx)

    fig.add_trace(go.Scatter(
        x    = roll_alpha.index,
        y    = roll_alpha,
        name = f"{window}M Rolling Alpha",
        line = dict(color="#FFD700", width=2),
        fill = "tozeroy",
        fillcolor = "rgba(255,215,0,0.08)",
        hovertemplate = "%{x|%b %Y}<br>Alpha: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#FF4444", width=1, dash="dash"))
    fig.update_yaxes(ticksuffix="%")
    return _apply_layout(fig)


def fig_screener_bars(screener_df: pd.DataFrame, signal_name: str, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top N stocks by signal score."""
    df = screener_df.head(top_n).copy()
    colours = []
    for decile in df["Decile"]:
        # Gradient from green (decile 1) to red (decile 10)
        ratio  = (decile - 1) / 9.0
        r      = int(255 * ratio)
        g      = int(255 * (1 - ratio))
        colours.append(f"rgb({r},{g},60)")

    fig = go.Figure(go.Bar(
        x           = df["Signal Score"].round(3),
        y           = df["Ticker"],
        orientation = "h",
        marker_color= colours[::-1],
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Signal: %{x:.3f}<br>"
            "1M Ret: %{customdata[0]:.1f}%<br>"
            "3M Ret: %{customdata[1]:.1f}%<br>"
            "12M Ret: %{customdata[2]:.1f}%<extra></extra>"
        ),
        customdata = df[["1M Return", "3M Return", "12M Return"]].values[::-1],
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_layout(fig)


# =============================================================================
# 7. SCREENER BUILDER
# =============================================================================

def build_screener(prices: pd.DataFrame, signal_name: str) -> pd.DataFrame:
    """
    Build the current momentum screener table for the clean universe.
    Uses the most recent month's signal scores and trailing return windows.
    Returns a DataFrame ready for display in the screener tab.
    """
    all_sigs = compute_all_signals(prices, lookback=12)
    sig      = all_sigs[signal_name]

    # ── Get the latest signal scores ─────────────────────────────────────────
    latest_scores = sig.iloc[-1].dropna().sort_values(ascending=False)

    # ── Trailing returns ──────────────────────────────────────────────────────
    monthly = prices.resample("ME").last()
    ret_1m  = monthly.pct_change(1).iloc[-1]
    ret_3m  = monthly.pct_change(3).iloc[-1]
    ret_12m = (monthly.shift(1) / monthly.shift(13) - 1).iloc[-1]  # 12-1 skip-month

    # ── RSI current ───────────────────────────────────────────────────────────
    rsi_now = compute_rsi(prices).iloc[-1]

    # ── MA signal current ─────────────────────────────────────────────────────
    ma_now  = compute_ma_cross(prices).iloc[-1]

    # ── Assemble ──────────────────────────────────────────────────────────────
    tickers = latest_scores.index
    df = pd.DataFrame({
        "Ticker"       : tickers,
        "Signal Score" : latest_scores.values,
        "1M Return"    : ret_1m.reindex(tickers).values * 100,
        "3M Return"    : ret_3m.reindex(tickers).values * 100,
        "12M Return"   : ret_12m.reindex(tickers).values * 100,
        "RSI-14"       : rsi_now.reindex(tickers).values.round(1),
        "MA Signal"    : ma_now.reindex(tickers).values,
    }).reset_index(drop=True)

    df["Rank"]   = range(1, len(df) + 1)
    n            = len(df)
    df["Decile"] = pd.cut(df["Rank"], bins=10, labels=range(1, 11)).astype(int)

    col_order = ["Rank", "Ticker", "Signal Score", "1M Return",
                 "3M Return", "12M Return", "RSI-14", "MA Signal", "Decile"]
    return df[col_order].round({"Signal Score": 4, "1M Return": 2,
                                "3M Return": 2, "12M Return": 2})


# =============================================================================
# 8. FAMA-FRENCH REGRESSION
# =============================================================================

def run_ff3_regression(bt: pd.DataFrame, ff: pd.DataFrame) -> dict:
    """
    Run OLS regression of strategy excess returns on FF3 factors.
    Returns alpha (annualised), factor loadings, t-stats, conf intervals, R².
    """
    if bt.empty or ff.empty:
        return {}

    monthly = bt["strategy_return"].dropna()
    ff_m    = ff[["Mkt-RF", "SMB", "HML", "RF"]].copy()
    ff_m.index = pd.to_datetime(ff_m.index)

    aligned = monthly.to_frame("ret").join(ff_m, how="inner")
    if len(aligned) < 12:
        return {}

    aligned["excess"] = aligned["ret"] - aligned["RF"]
    X = aligned[["Mkt-RF", "SMB", "HML"]].values
    y = aligned["excess"].values
    X_ = np.column_stack([np.ones(len(X)), X])

    # ── OLS via lstsq ────────────────────────────────────────────────────────
    coef, residuals, rank, sv = np.linalg.lstsq(X_, y, rcond=None)
    y_hat    = X_ @ coef
    resid    = y - y_hat
    n, k     = len(y), X_.shape[1]
    sigma2   = (resid ** 2).sum() / (n - k)
    var_coef = sigma2 * np.linalg.inv(X_.T @ X_).diagonal()
    se       = np.sqrt(var_coef)
    t_stats  = coef / se

    alpha_monthly = coef[0]
    alpha_annual  = (1 + alpha_monthly) ** 12 - 1

    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    factor_names = ["Mkt-RF", "SMB", "HML"]
    return {
        "alpha_annual"  : alpha_annual,
        "alpha_t"       : t_stats[0],
        "coefs"         : dict(zip(factor_names, coef[1:])),
        "t_stats"       : dict(zip(factor_names, t_stats[1:])),
        "conf_ints_95"  : dict(zip(factor_names, 1.96 * se[1:])),
        "r2"            : r2,
    }


# =============================================================================
# 9. ⓘ TOOLTIP HELPER
# =============================================================================

def info_icon(tooltip_id: str, text: str) -> html.Span:
    """
    Returns an ⓘ icon span + dbc.Tooltip that appears on hover.
    Usage: info_icon("tooltip-cumret", "Explanation text here...")
    """
    return html.Span([
        html.Span(
            "ⓘ",
            id    = tooltip_id,
            style = {
                "fontSize"    : "13px",
                "color"       : "#666",
                "cursor"      : "help",
                "marginLeft"  : "6px",
                "userSelect"  : "none",
            }
        ),
        dbc.Tooltip(
            text,
            target    = tooltip_id,
            placement = "top",
            style     = {
                "maxWidth"  : "320px",
                "fontSize"  : "12px",
                "lineHeight": "1.5",
            }
        ),
    ], style={"display": "inline"})


def section_title(label: str, tooltip_id: str, tooltip_text: str,
                  level: int = 5) -> html.Div:
    """
    Section heading with inline ⓘ tooltip icon.
    level: 5 = <h5>, 6 = <h6>
    """
    Tag = getattr(html, f"H{level}")
    return html.Div([
        Tag(label, style={"display": "inline", "marginRight": "4px",
                          "color": "#E0E0E0", "fontFamily": "'IBM Plex Mono', monospace"}),
        info_icon(tooltip_id, tooltip_text),
    ], style={"marginBottom": "8px", "marginTop": "16px"})


# =============================================================================
# 10. SHARED STYLE TOKENS
# =============================================================================

CARD_STYLE = {
    "backgroundColor": "#12122A",
    "border"         : "1px solid #2A2A4A",
    "borderRadius"   : "8px",
    "padding"        : "16px",
    "marginBottom"   : "16px",
}

KPI_CARD_STYLE = {
    "backgroundColor": "#0D0D1F",
    "border"         : "1px solid #2A2A4A",
    "borderRadius"   : "8px",
    "padding"        : "14px 18px",
    "textAlign"      : "center",
}

LABEL_STYLE = {
    "fontSize"  : "11px",
    "color"     : "#888",
    "letterSpacing": "0.08em",
    "marginBottom": "4px",
    "fontFamily": "'IBM Plex Mono', monospace",
}

VALUE_STYLE = {
    "fontSize"  : "22px",
    "fontWeight": "600",
    "fontFamily": "'IBM Plex Mono', monospace",
    "color"     : "#E0E0E0",
}

BUTTON_STYLE = {
    "width"          : "100%",
    "backgroundColor": "#2196F3",
    "color"          : "#fff",
    "border"         : "none",
    "borderRadius"   : "6px",
    "padding"        : "10px",
    "fontSize"       : "13px",
    "fontFamily"     : "'IBM Plex Mono', monospace",
    "cursor"         : "pointer",
    "marginTop"      : "12px",
    "letterSpacing"  : "0.05em",
}

CONTROL_LABEL_STYLE = {
    "fontSize"  : "11px",
    "color"     : "#AAA",
    "fontFamily": "'IBM Plex Mono', monospace",
    "letterSpacing": "0.06em",
    "marginBottom": "4px",
    "marginTop"  : "12px",
    "display"    : "flex",
    "alignItems" : "center",
}

# =============================================================================
# 11. REUSABLE CONTROL COMPONENTS
# =============================================================================

def signal_dropdown(component_id: str, default: str = "MOM_12_1") -> dcc.Dropdown:
    return dcc.Dropdown(
        id      = component_id,
        options = [{"label": v, "value": k} for k, v in SIGNAL_LABELS.items()],
        value   = default,
        clearable = False,
        style   = {"backgroundColor": "#1A1A2E", "color": "#fff",
                   "border": "1px solid #2A2A4A", "borderRadius": "4px",
                   "fontFamily": "'IBM Plex Mono', monospace", "fontSize": "12px"},
    )


def lookback_slider(component_id: str) -> dcc.Slider:
    return dcc.Slider(
        id    = component_id,
        min=3, max=12, step=None,
        marks = {3: "3M", 6: "6M", 9: "9M", 12: "12M"},
        value = 12,
        tooltip = {"always_visible": False},
    )


def cost_slider(component_id: str) -> dcc.Slider:
    return dcc.Slider(
        id    = component_id,
        min=0, max=50, step=5,
        marks = {i: f"{i}" for i in range(0, 51, 10)},
        value = 10,
        tooltip = {"placement": "bottom", "always_visible": True},
    )


def topn_slider(component_id: str) -> dcc.Slider:
    return dcc.Slider(
        id    = component_id,
        min=20, max=451, step=None,
        marks = {20: "20", 50: "50", 100: "100", 200: "200", 451: "All"},
        value = 50,
        tooltip = {"always_visible": False},
    )


# =============================================================================
# 12. KPI CARD BUILDER
# =============================================================================

def kpi_card(label: str, value_id: str, tooltip_id: str, tooltip_text: str) -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.Div([
                html.Span(label, style=LABEL_STYLE),
                info_icon(tooltip_id, tooltip_text),
            ]),
            html.Div("—", id=value_id, style=VALUE_STYLE),
        ], style=KPI_CARD_STYLE),
        xs=6, md=3,
    )


# =============================================================================
# 13. APP LAYOUT
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets = [
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap",
    ],
    title = "Pythia — Momentum Dashboard",
    suppress_callback_exceptions = True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
header = html.Div(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Span("PYTHIA", style={
                    "fontSize": "22px", "fontWeight": "600",
                    "color": "#2196F3", "fontFamily": "'IBM Plex Mono', monospace",
                    "letterSpacing": "0.15em",
                }),
                html.Span("  ·  Momentum Strategy Dashboard", style={
                    "fontSize": "13px", "color": "#888",
                    "fontFamily": "'IBM Plex Mono', monospace",
                }),
            ], width="auto"),
            dbc.Col([
                html.Span("S&P 500 · 10Y Daily OHLCV · DuckDB", style={
                    "fontSize": "11px", "color": "#555",
                    "fontFamily": "'IBM Plex Mono', monospace",
                    "float": "right", "lineHeight": "32px",
                })
            ]),
        ], align="center"),
    ], fluid=True),
    style={
        "backgroundColor": "#0A0A1A",
        "borderBottom"   : "1px solid #1E1E3A",
        "padding"        : "10px 0",
        "marginBottom"   : "0",
    }
)

# ── Tab 1: Strategy Backtester ─────────────────────────────────────────────────

tab1_controls = html.Div([
    html.Div(style=CARD_STYLE, children=[
        html.Div("BACKTEST PARAMETERS", style={
            "fontSize": "10px", "color": "#555", "letterSpacing": "0.15em",
            "fontFamily": "'IBM Plex Mono', monospace", "marginBottom": "12px",
        }),

        html.Div([
            html.Span("SIGNAL STRATEGY", style=CONTROL_LABEL_STYLE),
            info_icon("tt-signal", "Choose which momentum signal is used to rank stocks each month. "
                      "MOM 12-1 is the academic classic; Composite blends all four signals into a single z-scored rank."),
        ]),
        signal_dropdown("t1-signal"),

        html.Div([
            html.Span("LOOKBACK WINDOW", style=CONTROL_LABEL_STYLE),
            info_icon("tt-lookback", "How many months of history are used to compute the signal. "
                      "12M captures medium-term trend; 3M captures short-term momentum. "
                      "Note: the most recent month is always skipped to avoid reversal bias."),
        ]),
        lookback_slider("t1-lookback"),

        html.Div([
            html.Span("REBALANCING FREQUENCY", style=CONTROL_LABEL_STYLE),
            info_icon("tt-freq", "How often the portfolio is reconstituted. Monthly captures more signal updates "
                      "but incurs more transaction costs. Quarterly is slower but cheaper to run."),
        ]),
        dcc.RadioItems(
            id      = "t1-freq",
            options = [{"label": " Monthly", "value": "ME"},
                       {"label": " Quarterly", "value": "QE"}],
            value   = "ME",
            labelStyle = {"marginRight": "16px", "fontSize": "12px",
                          "fontFamily": "'IBM Plex Mono', monospace", "color": "#CCC"},
        ),

        html.Div([
            html.Span("TRANSACTION COST (BPS)", style=CONTROL_LABEL_STYLE),
            info_icon("tt-cost", "One-way cost per trade in basis points (1 bp = 0.01%). "
                      "Applied to both the long and short legs at every rebalance. "
                      "10 bps is a typical institutional estimate for large-cap S&P 500 stocks."),
        ]),
        cost_slider("t1-cost"),

        html.Div([
            html.Span("PORTFOLIO TYPE", style=CONTROL_LABEL_STYLE),
            info_icon("tt-ls", "Long/Short: buy top decile, short bottom decile — captures the full momentum spread. "
                      "Long-Only: buy top decile only — more realistic for most investors but misses the short alpha."),
        ]),
        dcc.RadioItems(
            id      = "t1-ls",
            options = [{"label": " Long / Short", "value": "ls"},
                       {"label": " Long Only", "value": "lo"}],
            value   = "ls",
            labelStyle = {"marginRight": "16px", "fontSize": "12px",
                          "fontFamily": "'IBM Plex Mono', monospace", "color": "#CCC"},
        ),

        html.Button("▶  RUN BACKTEST", id="t1-run", n_clicks=0, style=BUTTON_STYLE),
    ]),
], style={"position": "sticky", "top": "0"})


tab1_charts = html.Div([
    # ── KPI strip ─────────────────────────────────────────────────────────────
    dbc.Row([
        kpi_card("ANN. RETURN",   "t1-kpi-ret",    "tt-kpi-ret",
                 "Geometric mean annual return over the full backtest period."),
        kpi_card("SHARPE RATIO",  "t1-kpi-sharpe", "tt-kpi-sharpe",
                 "Annualised return minus risk-free rate (4%), divided by annualised volatility. "
                 "Values above 1.0 are generally considered strong."),
        kpi_card("MAX DRAWDOWN",  "t1-kpi-dd",     "tt-kpi-dd",
                 "The largest peak-to-trough decline over the backtest. "
                 "A key measure of worst-case loss — what the strategy did in its worst stretch."),
        kpi_card("CALMAR RATIO",  "t1-kpi-calmar", "tt-kpi-calmar",
                 "Annualised return divided by the absolute max drawdown. "
                 "Rewards strategies that earn well relative to their worst loss."),
    ], className="g-2 mb-3"),

    # ── Chart row 1: Cumulative Return + Rolling Sharpe ────────────────────────
    dbc.Row([
        dbc.Col(html.Div([
            section_title("Cumulative Return", "tt-cumret",
                          "Shows how $1 invested at backtest start would have grown. "
                          "A line above the grey benchmark means the strategy outperformed "
                          "the S&P 500 buy-and-hold. Blue shading highlights outperformance periods."),
            dcc.Loading(dcc.Graph(id="t1-fig-cumret",
                                  config={"displayModeBar": False},
                                  style={"height": "280px"}),
                        type="circle", color="#2196F3"),
        ], style=CARD_STYLE), md=7),

        dbc.Col(html.Div([
            section_title("Rolling 12M Sharpe", "tt-sharpe",
                          "The Sharpe ratio measures return per unit of risk over a rolling "
                          "12-month window. Green shading = periods where the strategy beat cash; "
                          "red shading = underperformed cash. The dotted line marks Sharpe = 1."),
            dcc.Loading(dcc.Graph(id="t1-fig-sharpe",
                                  config={"displayModeBar": False},
                                  style={"height": "280px"}),
                        type="circle", color="#2196F3"),
        ], style=CARD_STYLE), md=5),
    ], className="g-3"),

    # ── Chart row 2: Drawdown + Annual Heatmap ────────────────────────────────
    dbc.Row([
        dbc.Col(html.Div([
            section_title("Drawdown Profile", "tt-dd",
                          "How far the portfolio fell from its highest previous value at each point "
                          "in time. The red filled area makes it easy to see the depth and duration "
                          "of every losing streak."),
            dcc.Loading(dcc.Graph(id="t1-fig-dd",
                                  config={"displayModeBar": False},
                                  style={"height": "220px"}),
                        type="circle", color="#2196F3"),
        ], style=CARD_STYLE), md=5),

        dbc.Col(html.Div([
            section_title("Annual Return Heatmap", "tt-heatmap",
                          "Each cell shows the total return for that calendar year. "
                          "Comparing the strategy row against the S&P 500 row reveals "
                          "which years the momentum signal added value and which years it failed."),
            dcc.Loading(dcc.Graph(id="t1-fig-heatmap",
                                  config={"displayModeBar": False},
                                  style={"height": "220px"}),
                        type="circle", color="#2196F3"),
        ], style=CARD_STYLE), md=7),
    ], className="g-3"),
])

tab1_layout = dbc.Row([
    dbc.Col(tab1_controls, xs=12, md=3),
    dbc.Col(tab1_charts,   xs=12, md=9),
], className="g-3")


# ── Tab 2: Stock Screener ──────────────────────────────────────────────────────

tab2_layout = html.Div([
    # Controls bar
    html.Div(style={**CARD_STYLE, "padding": "12px 16px"}, children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("RANK BY SIGNAL", style=CONTROL_LABEL_STYLE),
                    info_icon("tt-scr-signal",
                              "Stocks are z-scored by the chosen signal and ranked from strongest "
                              "to weakest momentum. Signal Score above +2 means the stock is "
                              "more than 2 standard deviations above the universe average — very strong momentum."),
                ]),
                signal_dropdown("t2-signal"),
            ], md=4),
            dbc.Col([
                html.Div([
                    html.Span("TOP N TO DISPLAY", style=CONTROL_LABEL_STYLE),
                    info_icon("tt-scr-topn",
                              "Limit the screener table and bar chart to the top N ranked stocks. "
                              "Select 'All' to see the full clean universe of 451 stocks."),
                ]),
                topn_slider("t2-topn"),
            ], md=5),
            dbc.Col([
                html.Button("▶  REFRESH SCREENER", id="t2-run", n_clicks=0,
                            style={**BUTTON_STYLE, "marginTop": "22px"}),
            ], md=3),
        ], align="end"),
    ]),

    # Bar chart
    html.Div(style=CARD_STYLE, children=[
        section_title("Top Stocks by Momentum Score", "tt-scr-bars",
                      "Signal score is z-scored across all stocks so values are comparable "
                      "regardless of signal type. Green bars = top-decile stocks (strong buy signals); "
                      "red bars = bottom-decile stocks (strong sell signals for a long/short strategy)."),
        dcc.Loading(
            dcc.Graph(id="t2-fig-bars", config={"displayModeBar": False},
                      style={"height": "420px"}),
            type="circle", color="#2196F3"),
    ]),

    # Table
    html.Div(style=CARD_STYLE, children=[
        section_title("Full Rankings Table", "tt-scr-table",
                      "Sortable table showing every stock in the clean universe ranked by "
                      "momentum signal. Top-decile rows are shaded green; bottom-decile rows red. "
                      "1M / 3M / 12M returns use the skip-month convention (most recent month excluded "
                      "from 12M to avoid short-term reversal contamination)."),
        html.Div(id="t2-table"),
    ]),
])


# ── Tab 3: Signal Comparison ───────────────────────────────────────────────────

tab3_layout = html.Div([
    # Controls
    html.Div(style={**CARD_STYLE, "padding": "12px 16px"}, children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("LOOKBACK WINDOW", style=CONTROL_LABEL_STYLE),
                    info_icon("tt-cmp-lookback",
                              "All four signals use the same lookback window so the comparison is fair. "
                              "The only difference between them is how stocks are ranked each month."),
                ]),
                lookback_slider("t3-lookback"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Span("TRANSACTION COST (BPS)", style=CONTROL_LABEL_STYLE),
                    info_icon("tt-cmp-cost",
                              "Applied identically to all four strategies. "
                              "Higher costs tend to hurt higher-turnover signals more."),
                ]),
                cost_slider("t3-cost"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Span("PORTFOLIO TYPE", style=CONTROL_LABEL_STYLE),
                    info_icon("tt-cmp-ls",
                              "Long/Short captures the full spread between strong and weak momentum stocks. "
                              "Long-Only shows how each signal performs as a stock selection tool on the buy side only."),
                ]),
                dcc.RadioItems(
                    id      = "t3-ls",
                    options = [{"label": " Long / Short", "value": "ls"},
                               {"label": " Long Only", "value": "lo"}],
                    value   = "ls",
                    labelStyle = {"marginRight": "16px", "fontSize": "12px",
                                  "fontFamily": "'IBM Plex Mono', monospace", "color": "#CCC"},
                ),
            ], md=3),
            dbc.Col([
                html.Button("▶  RUN ALL SIGNALS", id="t3-run", n_clicks=0,
                            style={**BUTTON_STYLE, "marginTop": "22px"}),
            ], md=3),
        ], align="end"),
    ]),

    # Loading wrapper covers both charts + table
    dcc.Loading(
        html.Div([
            # Multi-line cumulative return
            html.Div(style=CARD_STYLE, children=[
                section_title("Cumulative Return — All Signals vs Benchmark", "tt-cmp-cumret",
                              "All four momentum signals run on the same universe, same rebalancing "
                              "schedule, and same transaction cost. The only difference is the ranking "
                              "rule applied each month. The grey dotted line is the S&P 500 buy-and-hold benchmark."),
                dcc.Graph(id="t3-fig-cumret", config={"displayModeBar": False},
                          style={"height": "320px"}),
            ]),

            # Metrics bar chart
            html.Div(style=CARD_STYLE, children=[
                section_title("Performance Metrics Comparison", "tt-cmp-bars",
                              "Grouped bars show all four strategies on all four key metrics simultaneously. "
                              "A strategy that is consistently tall across all groups dominates the others. "
                              "Ann. Return and Max Drawdown are shown in percentage points."),
                dcc.Graph(id="t3-fig-bars", config={"displayModeBar": False},
                          style={"height": "260px"}),
            ]),

            # Metrics table
            html.Div(style=CARD_STYLE, children=[
                section_title("Metrics Summary Table", "tt-cmp-table",
                              "One row per signal. Best value in each column is highlighted green; "
                              "worst is highlighted red. Use this to identify which signal offers "
                              "the best risk-adjusted performance under the chosen parameters."),
                html.Div(id="t3-table"),
            ]),
        ]),
        type="circle", color="#2196F3",
    ),
])


# ── Tab 4: Fama-French Risk Adjustment ────────────────────────────────────────

tab4_layout = html.Div([
    # Controls
    html.Div(style={**CARD_STYLE, "padding": "12px 16px"}, children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("SIGNAL STRATEGY", style=CONTROL_LABEL_STYLE),
                    info_icon("tt-ff-signal",
                              "Choose which strategy's returns to risk-adjust. "
                              "The backtest is run at default parameters (ME rebalance, 10bps cost) "
                              "so the analysis is comparable across signals."),
                ]),
                signal_dropdown("t4-signal"),
            ], md=4),
            dbc.Col([
                html.Div([
                    html.Span("LOOKBACK WINDOW", style=CONTROL_LABEL_STYLE),
                    info_icon("tt-ff-lookback",
                              "Lookback window used for the momentum signal in the backtest "
                              "whose returns are then sent through the FF3 regression."),
                ]),
                lookback_slider("t4-lookback"),
            ], md=4),
            dbc.Col([
                html.Button("▶  RUN FF ANALYSIS", id="t4-run", n_clicks=0,
                            style={**BUTTON_STYLE, "marginTop": "22px"}),
            ], md=4),
        ], align="end"),
    ]),

    dcc.Loading(
        html.Div([
            # KPI strip
            dbc.Row([
                kpi_card("ALPHA (ANN.)",   "t4-kpi-alpha",  "tt-ff-alpha",
                         "The annualised return that cannot be explained by the three Fama-French factors. "
                         "Positive alpha = genuine skill beyond known risk premia."),
                kpi_card("ALPHA t-STAT",  "t4-kpi-tstat",  "tt-ff-tstat",
                         "Statistical significance of alpha. A t-statistic above 2.0 means the alpha is "
                         "statistically significant at the 95% confidence level — unlikely to be due to chance."),
                kpi_card("R²",            "t4-kpi-r2",     "tt-ff-r2",
                         "How much of the strategy's return variance is explained by the three factors. "
                         "High R² means the strategy behaves like a known factor portfolio; "
                         "low R² means it is more distinctive."),
            ], className="g-2 mb-3"),

            dbc.Row([
                dbc.Col(html.Div([
                    section_title("Factor Exposures", "tt-ff-bars",
                                  "The Fama-French 3-factor model decomposes returns into: market exposure (Mkt-RF), "
                                  "small-cap tilt (SMB), and value tilt (HML). Error bars show 95% confidence intervals. "
                                  "Bars crossing zero are not statistically significant."),
                    dcc.Graph(id="t4-fig-bars", config={"displayModeBar": False},
                              style={"height": "280px"}),
                ], style=CARD_STYLE), md=5),

                dbc.Col(html.Div([
                    section_title("Rolling 12M Alpha", "tt-ff-rolling",
                                  "Rolling 12-month alpha shows whether the strategy's edge over factor benchmarks "
                                  "has been consistent over time or concentrated in specific regimes. "
                                  "Sustained positive alpha above zero is the goal."),
                    dcc.Graph(id="t4-fig-rolling", config={"displayModeBar": False},
                              style={"height": "280px"}),
                ], style=CARD_STYLE), md=7),
            ], className="g-3"),
        ]),
        type="circle", color="#2196F3",
    ),
])


# ── Tabs container ────────────────────────────────────────────────────────────

tabs = dbc.Tabs([
    dbc.Tab(tab1_layout, label="Strategy Backtester",  tab_id="tab-1",
            label_style={"fontFamily": "'IBM Plex Mono', monospace", "fontSize": "12px"}),
    dbc.Tab(tab2_layout, label="Stock Screener",       tab_id="tab-2",
            label_style={"fontFamily": "'IBM Plex Mono', monospace", "fontSize": "12px"}),
    dbc.Tab(tab3_layout, label="Signal Comparison",    tab_id="tab-3",
            label_style={"fontFamily": "'IBM Plex Mono', monospace", "fontSize": "12px"}),
    dbc.Tab(tab4_layout, label="Fama-French Analysis", tab_id="tab-4",
            label_style={"fontFamily": "'IBM Plex Mono', monospace", "fontSize": "12px"}),
], id="tabs", active_tab="tab-1",
   style={"backgroundColor": "#0A0A1A", "borderBottom": "1px solid #1E1E3A"})


app.layout = html.Div([
    header,
    html.Div(tabs, style={"backgroundColor": "#0D0D22",
                          "minHeight": "100vh", "padding": "0"}),
    dbc.Container(
        html.Div(tabs, style={"display": "none"}),   # dummy, real tabs above
        fluid=True,
        style={"display": "none"}
    ),
], style={"backgroundColor": "#0D0D22", "minHeight": "100vh"})

# Fix: rebuild layout without duplicate
app.layout = html.Div([
    header,
    dbc.Container([tabs], fluid=True,
                  style={"paddingTop": "16px", "paddingBottom": "40px"}),
], style={"backgroundColor": "#0D0D22", "minHeight": "100vh",
          "fontFamily": "'IBM Plex Mono', monospace"})


# =============================================================================
# 14. CALLBACKS
# =============================================================================

# ── Cache: prices loaded once on first callback, reused across all callbacks ──
_prices_cache: pd.DataFrame | None = None

def get_prices() -> pd.DataFrame:
    """Load prices from DuckDB on first call; return cached copy thereafter."""
    global _prices_cache
    if _prices_cache is None:
        _prices_cache = load_prices()
    return _prices_cache


# ── Tab 1: Backtest callback ──────────────────────────────────────────────────

@app.callback(
    Output("t1-fig-cumret",  "figure"),
    Output("t1-fig-sharpe",  "figure"),
    Output("t1-fig-dd",      "figure"),
    Output("t1-fig-heatmap", "figure"),
    Output("t1-kpi-ret",     "children"),
    Output("t1-kpi-sharpe",  "children"),
    Output("t1-kpi-dd",      "children"),
    Output("t1-kpi-calmar",  "children"),
    Output("t1-kpi-ret",     "style"),
    Output("t1-kpi-sharpe",  "style"),
    Output("t1-kpi-dd",      "style"),
    Output("t1-kpi-calmar",  "style"),
    Input("t1-run",      "n_clicks"),
    State("t1-signal",   "value"),
    State("t1-lookback", "value"),
    State("t1-freq",     "value"),
    State("t1-cost",     "value"),
    State("t1-ls",       "value"),
    prevent_initial_call = False,
)
def update_tab1(n_clicks, signal, lookback, freq, cost, ls_mode):
    """
    Main backtest callback for Tab 1.
    Runs the walk-forward backtest and returns all 4 charts + 4 KPI values.
    Triggered by the Run Backtest button (or on initial load with defaults).
    """
    prices     = get_prices()
    long_short = (ls_mode == "ls")

    bt = run_backtest(
        prices      = prices,
        signal_name = signal,
        lookback    = lookback or 12,
        freq        = freq or "ME",
        cost_bps    = cost or 10,
        long_short  = long_short,
    )

    mets = compute_metrics(bt)

    # ── Format KPI values ─────────────────────────────────────────────────────
    def fmt_pct(v):
        return f"{v*100:+.1f}%" if not np.isnan(v) else "—"
    def fmt_float(v):
        return f"{v:.2f}" if not np.isnan(v) else "—"

    ret_str    = fmt_pct(mets["ann_return"])
    sharpe_str = fmt_float(mets["sharpe"])
    dd_str     = fmt_pct(mets["max_drawdown"])
    calmar_str = fmt_float(mets["calmar"])

    # ── KPI colour coding ─────────────────────────────────────────────────────
    def kpi_colour(val, positive_good=True):
        style = dict(VALUE_STYLE)
        if np.isnan(val):
            style["color"] = "#888"
        elif (val > 0) == positive_good:
            style["color"] = "#4CAF50"
        else:
            style["color"] = "#F44336"
        return style

    return (
        fig_cumulative_return(bt, signal),
        fig_rolling_sharpe(bt),
        fig_drawdown(bt),
        fig_annual_heatmap(bt, signal),
        ret_str, sharpe_str, dd_str, calmar_str,
        kpi_colour(mets["ann_return"]),
        kpi_colour(mets["sharpe"]),
        kpi_colour(mets["max_drawdown"], positive_good=False),
        kpi_colour(mets["calmar"]),
    )


# ── Tab 2: Screener callback ──────────────────────────────────────────────────

@app.callback(
    Output("t2-fig-bars", "figure"),
    Output("t2-table",    "children"),
    Input("t2-run",    "n_clicks"),
    State("t2-signal", "value"),
    State("t2-topn",   "value"),
    prevent_initial_call = False,
)
def update_tab2(n_clicks, signal, top_n):
    """
    Build the screener table and bar chart.
    Computes all signals on the latest month's data from the clean universe.
    """
    prices     = get_prices()
    top_n      = top_n or 50
    screener   = build_screener(prices, signal or "MOM_12_1")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig = fig_screener_bars(screener, signal, top_n=min(top_n, 30))

    # ── Table with conditional row colouring ──────────────────────────────────
    display_df = screener.head(top_n).copy()
    display_df["Signal Score"] = display_df["Signal Score"].map("{:.4f}".format)
    display_df["1M Return"]    = display_df["1M Return"].map("{:+.2f}%".format)
    display_df["3M Return"]    = display_df["3M Return"].map("{:+.2f}%".format)
    display_df["12M Return"]   = display_df["12M Return"].map("{:+.2f}%".format)
    display_df["RSI-14"]       = display_df["RSI-14"].map("{:.1f}".format)
    display_df["MA Signal"]    = display_df["MA Signal"].map(
        lambda v: "▲ Bull" if v > 0 else ("▼ Bear" if v < 0 else "—"))

    header_row = html.Tr([
        html.Th(c, style={
            "fontSize": "10px", "color": "#888", "fontFamily": "'IBM Plex Mono', monospace",
            "padding": "6px 10px", "borderBottom": "1px solid #2A2A4A",
            "letterSpacing": "0.08em",
        }) for c in display_df.columns
    ])

    def row_bg(decile):
        if decile <= 1:
            return "rgba(76,175,80,0.12)"
        elif decile >= 10:
            return "rgba(244,67,54,0.12)"
        return "transparent"

    body_rows = [
        html.Tr([
            html.Td(str(val), style={
                "fontSize": "11px", "fontFamily": "'IBM Plex Mono', monospace",
                "padding": "5px 10px", "color": "#CCC",
                "borderBottom": "1px solid #1A1A2E",
            }) for val in row
        ], style={"backgroundColor": row_bg(int(display_df.iloc[i]["Decile"]))})
        for i, row in enumerate(display_df.values.tolist())
    ]

    table = html.Table(
        [html.Thead(header_row), html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse",
               "fontFamily": "'IBM Plex Mono', monospace"},
    )

    return fig, table


# ── Tab 3: Signal comparison callback ────────────────────────────────────────

@app.callback(
    Output("t3-fig-cumret", "figure"),
    Output("t3-fig-bars",   "figure"),
    Output("t3-table",      "children"),
    Input("t3-run",      "n_clicks"),
    State("t3-lookback", "value"),
    State("t3-cost",     "value"),
    State("t3-ls",       "value"),
    prevent_initial_call = False,
)
def update_tab3(n_clicks, lookback, cost, ls_mode):
    """
    Run all 4 momentum signals with identical parameters and compare them.
    A loading spinner covers the output area while all 4 backtests run.
    """
    prices     = get_prices()
    long_short = (ls_mode == "ls")
    lookback   = lookback or 12
    cost       = cost or 10

    # ── Run all signals ───────────────────────────────────────────────────────
    results  = {}
    metrics  = {}
    for sig in SIGNAL_COLS + ["Composite"]:
        bt = run_backtest(prices, sig, lookback, "ME", cost, long_short)
        results[sig] = bt
        metrics[sig] = compute_metrics(bt)

    # ── Cumulative return figure ──────────────────────────────────────────────
    fig_cum  = fig_signal_comparison(results)

    # ── Metrics bar chart ─────────────────────────────────────────────────────
    fig_bars = fig_metrics_bars(metrics)

    # ── Metrics summary table ─────────────────────────────────────────────────
    rows = []
    metric_keys   = ["ann_return", "sharpe", "max_drawdown", "calmar"]
    metric_labels = ["Ann. Return", "Sharpe", "Max Drawdown", "Calmar"]

    # Compute best/worst per column for highlighting
    col_vals = {mk: [metrics[s].get(mk, np.nan) for s in results.keys()]
                for mk in metric_keys}
    col_best = {
        "ann_return"   : max,
        "sharpe"       : max,
        "max_drawdown" : max,  # least negative is best
        "calmar"       : max,
    }

    header_row = html.Tr([
        html.Th("Signal", style={
            "fontSize": "10px", "color": "#888", "padding": "6px 10px",
            "fontFamily": "'IBM Plex Mono', monospace", "letterSpacing": "0.08em",
            "borderBottom": "1px solid #2A2A4A",
        })] + [
        html.Th(lbl, style={
            "fontSize": "10px", "color": "#888", "padding": "6px 10px",
            "fontFamily": "'IBM Plex Mono', monospace", "letterSpacing": "0.08em",
            "borderBottom": "1px solid #2A2A4A", "textAlign": "right",
        }) for lbl in metric_labels
    ])

    for sig_key in results.keys():
        mets  = metrics[sig_key]
        cells = [html.Td(
            SIGNAL_LABELS.get(sig_key, sig_key),
            style={"fontSize": "11px", "fontFamily": "'IBM Plex Mono', monospace",
                   "padding": "5px 10px", "color": COLOURS.get(sig_key, "#CCC"),
                   "borderBottom": "1px solid #1A1A2E"},
        )]
        for mk in metric_keys:
            v     = mets.get(mk, np.nan)
            best  = col_best[mk](v for v in col_vals[mk] if not np.isnan(v))
            worst = min(v for v in col_vals[mk] if not np.isnan(v))
            colour = "#4CAF50" if abs(v - best) < 1e-9 else (
                     "#F44336" if abs(v - worst) < 1e-9 else "#CCC")
            fmt_v = f"{v*100:+.1f}%" if mk in ("ann_return", "max_drawdown") else f"{v:.2f}"
            cells.append(html.Td(
                fmt_v if not np.isnan(v) else "—",
                style={"fontSize": "11px", "fontFamily": "'IBM Plex Mono', monospace",
                       "padding": "5px 10px", "color": colour,
                       "borderBottom": "1px solid #1A1A2E", "textAlign": "right"},
            ))
        rows.append(html.Tr(cells))

    table = html.Table(
        [html.Thead(header_row), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    return fig_cum, fig_bars, table


# ── Tab 4: Fama-French callback ───────────────────────────────────────────────

@app.callback(
    Output("t4-fig-bars",    "figure"),
    Output("t4-fig-rolling", "figure"),
    Output("t4-kpi-alpha",   "children"),
    Output("t4-kpi-tstat",   "children"),
    Output("t4-kpi-r2",      "children"),
    Output("t4-kpi-alpha",   "style"),
    Output("t4-kpi-tstat",   "style"),
    Output("t4-kpi-r2",      "style"),
    Input("t4-run",      "n_clicks"),
    State("t4-signal",   "value"),
    State("t4-lookback", "value"),
    prevent_initial_call = False,
)
def update_tab4(n_clicks, signal, lookback):
    """
    Run FF3 regression on the chosen strategy's backtest returns.
    Fetches FF3 factors (cached to CSV after first run).
    """
    prices   = get_prices()
    lookback = lookback or 12

    bt = run_backtest(prices, signal or "MOM_12_1", lookback, "ME", 10, True)
    ff = fetch_ff3_factors()

    reg = run_ff3_regression(bt, ff)

    if not reg:
        empty_fig = go.Figure()
        _apply_layout(empty_fig)
        style_na  = dict(VALUE_STYLE, color="#888")
        return empty_fig, empty_fig, "—", "—", "—", style_na, style_na, style_na

    alpha   = reg["alpha_annual"]
    t_stat  = reg["alpha_t"]
    r2      = reg["r2"]
    coefs   = reg["coefs"]
    ci95    = reg["conf_ints_95"]

    alpha_str = f"{alpha*100:+.2f}%"
    tstat_str = f"{t_stat:.2f}"
    r2_str    = f"{r2:.3f}"

    # KPI colour: alpha green if positive, t-stat green if >2
    def kpi_s(val, threshold=0, positive_good=True):
        s = dict(VALUE_STYLE)
        s["color"] = "#4CAF50" if (val > threshold) == positive_good else "#F44336"
        return s

    return (
        fig_ff_factor_bars(alpha, coefs, ci95),
        fig_rolling_alpha(bt, ff),
        alpha_str, tstat_str, r2_str,
        kpi_s(alpha),
        kpi_s(t_stat, threshold=2.0),
        dict(VALUE_STYLE),   # R² is neutral
    )


# =============================================================================
# 15. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PYTHIA  —  Momentum Strategy Dashboard")
    print("=" * 60)
    print(f"  DB path      : {DB_PATH.resolve()}")
    print(f"  FF3 cache    : {FF_CACHE.resolve()}")
    print(f"  Backtest from: {BACKTEST_START}")
    print("  Dashboard    : http://127.0.0.1:8050")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"\n  ⚠  WARNING: {DB_PATH} not found.")
        print("  Run download_sp500_to_duckdb.py first to build the database.\n")

    app.run(
        debug = True,
        port  = 8050,
        host  = "127.0.0.1",
    )
