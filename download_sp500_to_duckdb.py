"""
download_sp500_to_duckdb.py
────────────────────────────────────────────────────────────────────────────────
Downloads 10 years of daily OHLCV data for all S&P 500 constituents from
Yahoo Finance (yfinance) and stores results in a DuckDB database.

ANTI-BLOCKING SAFEGUARDS:
  1. Randomised inter-batch delays (jitter) to mimic human browsing cadence
  2. Rotating User-Agent headers on every batch
  3. Exponential back-off with full jitter on errors / empty responses
  4. Conservative batch size (≤25 tickers) to stay under rate-limit radar
  5. Single-threaded yfinance downloads to minimise concurrent connections
  6. Resume capability — already-downloaded tickers skipped on re-run
  7. JSON checkpoint file survives crashes; just re-run to continue
  8. Per-ticker retry pass for any batches that permanently failed
  9. Optional HTTPS proxy support via HTTPS_PROXY environment variable

Requirements:
    pip install yfinance duckdb pandas requests fake-useragent

    fake-useragent is optional — falls back to a built-in UA rotation list.

Usage:
    python download_sp500_to_duckdb.py

Output:
    sp500_prices.duckdb              DuckDB file with table `daily_prices`
    sp500_download_progress.json     Checkpoint file (delete after full run)

Schema of `daily_prices`:
    date    DATE     — Trading date
    ticker  VARCHAR  — Ticker symbol (e.g. 'AAPL')
    open    DOUBLE
    high    DOUBLE
    low     DOUBLE
    close   DOUBLE   — Adjusted close price
    volume  BIGINT

    Primary key: (date, ticker)
────────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yfinance as yf

# ── Optional: rotating user-agents via fake-useragent ────────────────────────
try:
    from fake_useragent import UserAgent as _FUA
    _UA = _FUA()
    def random_ua() -> str:
        return _UA.random
except ImportError:
    _STATIC_UA = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.2365.92",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    ]
    def random_ua() -> str:
        return random.choice(_STATIC_UA)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH         = "sp500_prices.duckdb"
CHECKPOINT_PATH = "sp500_download_progress.json"
YEARS_BACK      = 10
END_DATE        = datetime.today().strftime("%Y-%m-%d")
START_DATE      = (datetime.today() - timedelta(days=365 * YEARS_BACK)).strftime("%Y-%m-%d")

BATCH_SIZE      = 25      # tickers per yfinance call — keep <=30
MIN_SLEEP       = 3.0     # min seconds between batches
MAX_SLEEP       = 9.0     # max seconds between batches
MAX_RETRIES     = 5       # attempts per batch before giving up
BACKOFF_BASE    = 2.5     # exponential back-off base (seconds)
BACKOFF_CAP     = 120.0   # max back-off seconds


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TICKER LIST
# ═══════════════════════════════════════════════════════════════════════════════

def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    log.info("Fetching S&P 500 tickers from Wikipedia …")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": random_ua()}
    html = requests.get(url, headers=headers, timeout=15).text
    df = pd.read_html(html, header=0)[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    log.info(f"  → {len(tickers)} tickers retrieved.")
    return tickers


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> set[str]:
    if Path(CHECKPOINT_PATH).exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        done = set(data.get("completed", []))
        log.info(f"Resuming — {len(done)} tickers already done, skipping.")
        return done
    return set()


def save_checkpoint(completed: set[str]) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(
            {"completed": sorted(completed), "saved_at": datetime.now().isoformat()},
            f,
            indent=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANTI-BLOCKING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def human_sleep(label: str = "") -> None:
    """Random pause to mimic human inter-request gap."""
    delay = random.uniform(MIN_SLEEP, MAX_SLEEP)
    log.info(f"  💤 Sleeping {delay:.1f}s {label}")
    time.sleep(delay)


def backoff_sleep(attempt: int) -> None:
    """Full-jitter exponential back-off."""
    ceiling = min(BACKOFF_BASE ** attempt, BACKOFF_CAP)
    delay   = random.uniform(0.5, ceiling)
    log.warning(f"  ⏳ Back-off sleep {delay:.1f}s (attempt {attempt}) …")
    time.sleep(delay)


def rotate_yf_useragent() -> None:
    """
    Patch yfinance's internal requests.Session with a fresh User-Agent.
    Works across multiple yfinance versions by trying known internal paths.
    """
    ua = random_ua()
    patched = False
    for attr_path in [
        ("yfinance.base", "_session"),
        ("yfinance.utils", "_session"),
    ]:
        try:
            mod = __import__(attr_path[0], fromlist=[""])
            sess = getattr(mod, attr_path[1], None)
            if sess and hasattr(sess, "headers"):
                sess.headers.update({"User-Agent": ua})
                patched = True
                break
        except Exception:
            pass
    if not patched:
        # Last resort — monkey-patch requests.Session globally for this process
        try:
            import requests
            requests.utils.default_user_agent = lambda: ua
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Reshape yfinance wide output → long format."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ["field", "ticker"]
        # future_stack silences the FutureWarning in pandas >= 2.1
        try:
            df = raw.stack(level="ticker", future_stack=True).reset_index()
        except TypeError:
            df = raw.stack(level="ticker").reset_index()
    else:
        df = raw.reset_index()
        df["ticker"] = tickers[0]

    # Normalise column names
    df.columns = [str(c).lower().strip() for c in df.columns]
    for alias in ("datetime", "level_0", "index"):
        if alias in df.columns and "date" not in df.columns:
            df.rename(columns={alias: "date"}, inplace=True)

    # Keep relevant columns
    want = ["date", "ticker", "open", "high", "low", "close", "volume"]
    df = df[[c for c in want if c in df.columns]].copy()

    # Type coercion
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    return df.dropna(subset=["close"])


def download_batch(tickers: list[str]) -> pd.DataFrame:
    """
    Download OHLCV data for a list of tickers.
    Retries with exponential back-off on failure or empty response.
    Rotates User-Agent before every attempt.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        rotate_yf_useragent()
        try:
            raw = yf.download(
                tickers=tickers,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,   # adjusted prices
                progress=False,
                threads=False,      # fewer concurrent connections = safer
                timeout=30,
            )
        except Exception as exc:
            log.warning(f"  ⚠ Exception (attempt {attempt}/{MAX_RETRIES}): {exc}")
            backoff_sleep(attempt)
            continue

        if raw is None or raw.empty:
            log.warning(f"  ⚠ Empty response (attempt {attempt}/{MAX_RETRIES})")
            backoff_sleep(attempt)
            continue

        try:
            return _to_long(raw, tickers)
        except Exception as exc:
            log.warning(f"  ⚠ Reshape error (attempt {attempt}/{MAX_RETRIES}): {exc}")
            backoff_sleep(attempt)

    log.error(f"  ✗ Batch permanently failed after {MAX_RETRIES} attempts.")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DUCKDB
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            date    DATE    NOT NULL,
            ticker  VARCHAR NOT NULL,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE  NOT NULL,
            volume  BIGINT,
            PRIMARY KEY (date, ticker)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS run_metadata (
            key   VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
    return con


def insert_df(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Upsert-safe insert — silently skips duplicate (date, ticker) rows."""
    if df.empty:
        return 0
    con.register("_batch", df)
    con.execute("""
        INSERT OR IGNORE INTO daily_prices
        SELECT date, ticker, open, high, low, close, volume
        FROM _batch
    """)
    con.unregister("_batch")
    return len(df)


def write_metadata(con: duckdb.DuckDBPyConnection, n_ok: int, n_fail: int) -> None:
    rows = [
        ("start_date",   START_DATE),
        ("end_date",     END_DATE),
        ("tickers_ok",   str(n_ok)),
        ("tickers_fail", str(n_fail)),
        ("created_at",   datetime.now().isoformat()),
        ("source",       "Yahoo Finance via yfinance"),
    ]
    for key, val in rows:
        con.execute("INSERT OR REPLACE INTO run_metadata VALUES (?, ?)", [key, val])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 68)
    log.info("  S&P 500  →  DuckDB  |  10-year daily OHLCV")
    log.info(f"  Date range : {START_DATE}  →  {END_DATE}")
    log.info(f"  Batch size : {BATCH_SIZE}  |  Sleep : {MIN_SLEEP}–{MAX_SLEEP}s")
    log.info(f"  Output     : {DB_PATH}")
    log.info("=" * 68)

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        log.info(f"  Proxy detected: {proxy}")

    all_tickers = get_sp500_tickers()
    completed   = load_checkpoint()
    todo        = [t for t in all_tickers if t not in completed]
    log.info(f"  Tickers to download: {len(todo)} / {len(all_tickers)}")

    con         = init_db(DB_PATH)
    total_rows  = 0
    failed      = []  # batches that produced no data at all

    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        log.info(f"\n── Batch {idx}/{len(batches)}: {batch[0]} … {batch[-1]} ({len(batch)} tickers) ──")
        df = download_batch(batch)

        if df.empty:
            log.warning("  ✗ No data — queued for single-ticker retry.")
            failed.extend(batch)
        else:
            rows = insert_df(con, df)
            total_rows += rows
            completed.update(batch)
            save_checkpoint(completed)
            log.info(f"  ✓ {rows:,} rows inserted  |  cumulative: {total_rows:,}")

        if idx < len(batches):
            human_sleep(f"after batch {idx}/{len(batches)}")

    # ── Per-ticker retry pass ─────────────────────────────────────────────────
    if failed:
        log.info(f"\n── Retrying {len(failed)} failed tickers one-by-one ──")
        time.sleep(random.uniform(20, 40))  # longer cooldown before retry pass

        for ticker in failed:
            if ticker in completed:
                continue
            log.info(f"  → Retrying: {ticker}")
            df = download_batch([ticker])
            if not df.empty:
                rows = insert_df(con, df)
                total_rows += rows
                completed.add(ticker)
                save_checkpoint(completed)
                log.info(f"    ✓ {ticker}: {rows} rows")
            else:
                log.warning(f"    ✗ {ticker}: still failed — skipping.")
            human_sleep()

    # ── Finalise ──────────────────────────────────────────────────────────────
    still_failed = set(all_tickers) - completed
    write_metadata(con, len(completed), len(still_failed))
    con.close()

    log.info("\n" + "=" * 68)
    log.info(f"  ✅  Complete!")
    log.info(f"  Total rows   : {total_rows:,}")
    log.info(f"  Tickers OK   : {len(completed)}")
    log.info(f"  Tickers failed: {len(still_failed)}")
    if still_failed:
        log.info(f"  Failed list  : {sorted(still_failed)}")
    log.info(f"  File         : {Path(DB_PATH).resolve()}")
    log.info("=" * 68)

    print("""
── How to use the DuckDB file ─────────────────────────────────────────────────

import duckdb
con = duckdb.connect("sp500_prices.duckdb")

# Most recent 5 rows for Apple
print(con.execute(\"\"\"
    SELECT * FROM daily_prices
    WHERE ticker = 'AAPL'
    ORDER BY date DESC LIMIT 5
\"\"\").df())

# Coverage summary per ticker
print(con.execute(\"\"\"
    SELECT ticker,
           COUNT(*)   AS trading_days,
           MIN(date)  AS first_date,
           MAX(date)  AS last_date
    FROM daily_prices
    GROUP BY ticker
    ORDER BY ticker
\"\"\").df())
──────────────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
