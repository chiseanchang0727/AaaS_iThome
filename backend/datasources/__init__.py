"""Datasources the agent can read from.

    rows = await query_database("SELECT channel, SUM(views) FROM videos ...")
"""

import asyncio

from config import cfg

from .postgres import PostgresSource, TooManyRows

__all__ = ["PostgresSource", "TooManyRows", "query_database", "estimate_query", "count_query", "close_database"]

_source: PostgresSource | None = None
_lock = asyncio.Lock()


async def _get_source() -> PostgresSource:
    """One pool per process, built on first use."""
    global _source
    async with _lock:
        if _source is None:
            _source = await PostgresSource.connect(
                cfg.database.dsn, cfg.database.max_rows, cfg.database.timeout
            )
    return _source


async def query_database(query: str, max_rows: int | None = None, *, raw: bool = False) -> list[dict]:
    """Run a read-only SQL query. The agent supplies only the query.

    `max_rows` overrides `cfg.database.max_rows` for callers that need more,
    such as exporting a result to a file instead of into the agent's context.
    `raw` keeps the database's own value types instead of JSON-safe ones.
    """
    source = await _get_source()
    return await source.query(query, max_rows, raw=raw)


async def estimate_query(query: str) -> dict[str, int]:
    """The planner's estimate for a read-only query, without running it: rows, width, bytes."""
    source = await _get_source()
    return await source.estimate(query)


async def count_query(query: str) -> int:
    """How many rows a read-only query really returns, counted in the database: one number comes back.

    The database still does the query's work; only the count crosses the wire. Used when
    EXPLAIN's estimate is over a limit, because the estimate can be far off (e.g. GROUP BY on
    an expression the planner has no statistics for).
    """
    rows = await query_database(f"SELECT count(*) AS n FROM ({query.strip().rstrip(';')}) AS counted", raw=True)
    return int(rows[0]["n"])


async def close_database() -> None:
    """Close the pool on shutdown."""
    global _source
    if _source is not None:
        await _source.close()
        _source = None
