"""Read-only SQL access for the agent.

The agent supplies a query and nothing else; the DSN, pooling, timeouts and
result limits live here.

Safety is the database's job, not this module's: connect as a role that only
holds SELECT, so a query that tries to write fails in the server. Filtering SQL
strings for dangerous keywords does not hold up and is not attempted.
"""

import datetime as dt
import json
import uuid
from decimal import Decimal
from typing import Any

import asyncpg


class TooManyRows(Exception):
    """Raised instead of silently truncating, so the agent can add a LIMIT."""


def _jsonable(value: Any) -> Any:
    """Coerce pg types that do not survive JSON encoding."""
    if isinstance(value, Decimal):
        # SUM() over a bigint comes back as numeric; float would lose precision
        # past 2**53, so keep whole numbers as ints.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (uuid.UUID, dt.timedelta)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return value


class PostgresSource:
    def __init__(self, pool: asyncpg.Pool, max_rows: int, timeout: float):
        self._pool = pool
        self._max_rows = max_rows
        self._timeout = timeout

    @classmethod
    async def connect(cls, dsn: str, max_rows: int, timeout: float) -> "PostgresSource":
        """Open the pool once; reuse it for every query."""
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        return cls(pool, max_rows, timeout)

    async def close(self) -> None:
        await self._pool.close()

    async def estimate(self, sql: str) -> dict[str, int]:
        """The planner's guess for `sql`, without running it: rows, bytes per row, and bytes.

        EXPLAIN (no ANALYZE) only plans the query, in a few milliseconds, so
        it can be asked before deciding whether to move the rows at all.
        """
        async with self._pool.acquire() as conn:
            plan = json.loads(await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", timeout=self._timeout))[0]["Plan"]
        rows, width = int(plan["Plan Rows"]), int(plan["Plan Width"])
        return {"rows": rows, "width": width, "bytes": rows * width}

    async def query(self, sql: str, max_rows: int | None = None, *, raw: bool = False) -> list[dict]:
        """Run `sql`; refuse results over `max_rows` (default: the source's cap).

        Values are made JSON-safe unless `raw`, which keeps asyncpg's own types
        (Decimal, date, datetime...) for callers that write typed formats.
        """
        max_rows = self._max_rows if max_rows is None else max_rows
        async with self._pool.acquire() as conn:
            # A cursor stops a LIMIT-less query from pulling the whole table
            # into memory; one row past the cap tells us it was too big.
            async with conn.transaction():
                cursor = await conn.cursor(sql, timeout=self._timeout)
                rows = await cursor.fetch(max_rows + 1, timeout=self._timeout)

        if len(rows) > max_rows:
            raise TooManyRows(
                f"query returned more than {max_rows} rows; add a LIMIT "
                f"or aggregate the results"
            )
        if raw:
            return [dict(row) for row in rows]
        return [{k: _jsonable(v) for k, v in row.items()} for row in rows]
