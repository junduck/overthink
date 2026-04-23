"""Ingest per-date market parquet files into per-code SQLite databases.

Reads {market_dir}/{date}-ohlcv-1m.pq, computes temporal features,
splits by code, and appends to {db_dir}/{code}.db SQLite databases.

SQLite databases are append-only and idempotent: re-running for the same
date safely skips already-ingested data.

Usage:
    uv run python scripts/ingest_to_sqlite.py \\
        --market-dir ~/Data/cn_1m_ohlcv --db-dir ~/Data/cn_1m_sqlite --date 2018-01-02

    uv run python scripts/ingest_to_sqlite.py \\
        --market-dir ~/Data/cn_1m_ohlcv --db-dir ~/Data/cn_1m_sqlite --all

    uv run python scripts/ingest_to_sqlite.py \\
        --market-dir ~/Data/cn_1m_ohlcv --db-dir ~/Data/cn_1m_sqlite \\
        --start-date 2018-01-01 --end-date 2018-12-31 \\
        --codes 000001.SZ,600000.SH
"""

import argparse
import sqlite3
import time
from pathlib import Path

import polars as pl

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS bars (
    time_idx      INTEGER PRIMARY KEY,
    date_epoch    INTEGER NOT NULL,
    time          TEXT NOT NULL,
    minute_of_day INTEGER NOT NULL,
    hour          INTEGER NOT NULL,
    dow           INTEGER NOT NULL,
    dom           INTEGER NOT NULL,
    month         INTEGER NOT NULL,
    open          REAL NOT NULL,
    high          REAL NOT NULL,
    low           REAL NOT NULL,
    close         REAL NOT NULL,
    volume        REAL NOT NULL,
    turnover      REAL NOT NULL
)"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_date_epoch ON bars(date_epoch)"

_INSERT = "INSERT OR IGNORE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

_SELECT_COLS = [
    "date_epoch",
    "time",
    "minute_of_day",
    "hour",
    "dow",
    "dom",
    "month",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Turnover",
]


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open or create a per-code SQLite database with tuned settings."""
    is_new = not db_path.exists()
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    if is_new:
        conn.execute("PRAGMA page_size = 65536")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_INDEX)
    return conn


def _add_temporal(df: pl.DataFrame) -> pl.DataFrame:
    """Add date_epoch, time string, and decomposed temporal columns."""
    ti = pl.col("TimeInterval")
    return df.with_columns(
        ti.cast(pl.Date).to_physical().cast(pl.Int32).alias("date_epoch"),
        ti.cast(pl.Utf8).alias("time"),
        (ti.dt.hour() * 60 + ti.dt.minute()).cast(pl.Int32).alias("minute_of_day"),
        ti.dt.hour().cast(pl.Int32).alias("hour"),
        (ti.dt.weekday().cast(pl.Int32) - 1).alias("dow"),
        (ti.dt.day().cast(pl.Int32) - 1).alias("dom"),
        (ti.dt.month().cast(pl.Int32) - 1).alias("month"),
    )


def _ingest_partition(conn: sqlite3.Connection, part: pl.DataFrame) -> int:
    """Insert one code's bars for one date. Returns rows inserted."""
    date_epoch = int(part["date_epoch"][0])

    cur = conn.execute("SELECT COUNT(*) FROM bars WHERE date_epoch = ?", (date_epoch,))
    if cur.fetchone()[0] > 0:
        return 0

    cur = conn.execute("SELECT COALESCE(MAX(time_idx), -1) FROM bars")
    start_idx = cur.fetchone()[0] + 1

    rows = part.select(_SELECT_COLS).iter_rows(named=False)
    data = [(start_idx + i, *row) for i, row in enumerate(rows)]

    conn.execute("BEGIN")
    try:
        conn.executemany(_INSERT, data)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return len(data)


def ingest_date(
    market_dir: Path,
    db_dir: Path,
    date_str: str,
    code_filter: set[str] | None = None,
) -> int:
    """Ingest one date's market data. Returns total rows inserted."""
    market_file = market_dir / f"{date_str}-ohlcv-1m.pq"
    if not market_file.exists():
        return 0

    df = pl.read_parquet(market_file)
    df = _add_temporal(df)

    if code_filter:
        df = df.filter(pl.col("Code").is_in(code_filter))

    parts = df.partition_by("Code")
    total = 0
    for part in parts:
        code = str(part["Code"][0])
        db_path = db_dir / f"{code}.db"
        conn = _open_db(db_path)
        try:
            total += _ingest_partition(conn, part)
        finally:
            conn.close()

    return total


def list_dates(market_dir: Path) -> list[str]:
    """List all available dates from market parquet filenames."""
    return sorted(
        p.stem.replace("-ohlcv-1m", "")
        for p in market_dir.glob("*-ohlcv-1m.pq")
        if not p.name.startswith(".")
    )


def _resolve_dates(args: argparse.Namespace) -> list[str]:
    """Determine which dates to process from CLI arguments."""
    market_dir = Path(args.market_dir)
    if args.date:
        return [args.date]
    all_dates = list_dates(market_dir)
    if args.start_date or args.end_date:
        dates = all_dates
        if args.start_date:
            dates = [d for d in dates if d >= args.start_date]
        if args.end_date:
            dates = [d for d in dates if d <= args.end_date]
        return dates
    if args.all:
        return all_dates
    return []


def _run_ingest(args: argparse.Namespace) -> None:
    """Execute ingestion for resolved dates."""
    market_dir = Path(args.market_dir)
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    code_filter = set(args.codes.split(",")) if args.codes else None
    dates = _resolve_dates(args)
    if not dates:
        print("[ingest] No dates to process")
        return

    t0 = time.time()
    grand_total = 0
    for i, date_str in enumerate(dates):
        t1 = time.time()
        n = ingest_date(market_dir, db_dir, date_str, code_filter)
        elapsed = time.time() - t1
        grand_total += n
        print(
            f"[ingest] {date_str}: {n:,} rows, {elapsed:.1f}s  "
            f"({i + 1}/{len(dates)}, cumulative {time.time() - t0:.0f}s)"
        )

    total_elapsed = time.time() - t0
    rate = grand_total / max(total_elapsed, 0.001)
    print(
        f"\n[ingest] Done: {len(dates)} dates, {grand_total:,} total rows, "
        f"{total_elapsed:.1f}s ({rate:,.0f} rows/s)"
    )


def main() -> None:
    """CLI entry point for market data ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest market data into per-code SQLite DBs"
    )
    parser.add_argument(
        "--market-dir",
        required=True,
        help="Directory with {date}-ohlcv-1m.pq files",
    )
    parser.add_argument(
        "--db-dir",
        required=True,
        help="Output directory for {code}.db SQLite files",
    )
    parser.add_argument("--date", help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date for range (inclusive)")
    parser.add_argument("--end-date", help="End date for range (inclusive)")
    parser.add_argument("--all", action="store_true", help="Ingest all available dates")
    parser.add_argument("--codes", help="Comma-separated code filter (for testing)")
    args = parser.parse_args()
    if not (args.date or args.start_date or args.end_date or args.all):
        parser.error("Specify --date, --start-date/--end-date, or --all")
    _run_ingest(args)


if __name__ == "__main__":
    main()
