"""Why EXPLAIN overestimates GROUP BY on an expression: estimate vs real rows, side by side.

    uv run --env-file ../.env python scripts/explain_estimate_demo.py                     # events_enterprise
    uv run --env-file ../.env python scripts/explain_estimate_demo.py --table events

For each query it prints what EXPLAIN estimates (no rows are read) and what
`SELECT count(*) FROM (<query>)` really returns (the query runs in the database,
one number comes back). This is the same pair the fit check uses: estimate
first, count when the estimate is over the limit.

The planner keeps statistics per column (pg_stats), not per expression. For
GROUP BY on a plain column it uses that column's number of distinct values.
For GROUP BY EXTRACT(HOUR FROM event_time) it has no statistics for the
expression, so it falls back to the column inside it, event_time, which is
almost unique per row. So it expects about one group per row, not 24.

Needs the tables from scripts/make_events_demo.py. Read-only.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg  # noqa: E402

WHERE = "WHERE event_time >= NOW() - INTERVAL '14 days'"

QUERIES = {
    "v1: raw events (14 days)": f"SELECT event_type, event_time FROM {{t}} {WHERE}",
    "GROUP BY event_type (a column)": f"SELECT event_type, COUNT(*) FROM {{t}} {WHERE} GROUP BY 1",
    "GROUP BY device_type (a column)": f"SELECT device_type, COUNT(*) FROM {{t}} {WHERE} GROUP BY 1",
    "GROUP BY event_time (the column inside)": f"SELECT event_time, COUNT(*) FROM {{t}} {WHERE} GROUP BY 1",
    "GROUP BY EXTRACT(HOUR ...) (an expression)":
        f"SELECT EXTRACT(HOUR FROM event_time)::int AS hour, COUNT(*) FROM {{t}} {WHERE} GROUP BY 1",
    "the rewrite: hour x event_type":
        f"SELECT EXTRACT(HOUR FROM event_time)::int AS hour, event_type, COUNT(*) AS count "
        f"FROM {{t}} {WHERE} GROUP BY 1, 2 ORDER BY 1, 2",
}


async def main(table: str, timeout: float) -> None:
    conn = await asyncpg.connect(cfg.database.dsn)
    try:
        print(f"Table {table}: ~{await conn.fetchval('SELECT reltuples::bigint FROM pg_class WHERE relname = $1', table):,} rows\n")

        # What the planner knows: one row per column, nothing for EXTRACT(HOUR FROM event_time).
        # n_distinct > 0 is a count; < 0 is a fraction of the rows (-1 = every row different).
        print("pg_stats (what the planner knows):")
        for name, n in await conn.fetch(
                "SELECT attname, n_distinct FROM pg_stats WHERE tablename = $1 "
                "AND attname IN ('event_type', 'device_type', 'event_time') ORDER BY attname", table):
            meaning = f"{n:,.0f} distinct values" if n > 0 else f"{-n:.0%} of the rows are distinct"
            print(f"  {name:12} n_distinct = {n:>10}   ({meaning})")
        print()

        print(f"{'query':44} {'EXPLAIN estimate':>17} {'real rows':>12} {'count took':>11}")
        for label, sql in QUERIES.items():
            sql = sql.format(t=table)
            plan = json.loads(await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}"))[0]["Plan"]
            started = time.monotonic()
            real = await conn.fetchval(f"SELECT count(*) FROM ({sql}) AS counted", timeout=timeout)
            print(f"{label:44} {plan['Plan Rows']:>17,} {real:>12,} {time.monotonic() - started:>10.1f}s")

        print("\nThe rewrite's plan (look at rows= on the Aggregate line):")
        for (line,) in await conn.fetch(f"EXPLAIN {QUERIES['the rewrite: hour x event_type'].format(t=table)}"):
            print("  " + line)
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="events_enterprise")
    parser.add_argument("--timeout", type=float, default=120, help="seconds allowed for each count")
    args = parser.parse_args()
    asyncio.run(main(args.table, args.timeout))
