"""Demo data for adaptive saved-analysis runs: two event logs with the same columns, ~400x apart.

    uv run --env-file ../.env python scripts/make_events_demo.py                   # both tables
    uv run --env-file ../.env python scripts/make_events_demo.py --save-analysis   # + the naive saved analysis

    events             ~50K rows   (5,000 sessions)
    events_enterprise  ~20M rows   (2,000,000 sessions)

Columns: event_id, user_id, session_id, device_type, event_type, event_time.
Sessions are spread over the last 60 days; a session's events are a few
minutes apart, longest on desktop and shortest on mobile, so the median
session length differs by device. The rows are generated inside PostgreSQL
(generate_series), as the ingest role, then made readable by the agent's
role, analysed (so the planner's row estimates are right) and registered as
table datasets, like an upload.

--save-analysis saves "Median session length by device" in its first, naive
form: export the raw events of the last 30 days, sessionise in Python. Fine
on `events` (~25K rows exported); on `events_enterprise` it would move ~10M
rows into the sandbox, which is what adaptive runs detect and optimize.
"""

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses import Analysis, AnalysisStore, Output, QueryInput  # noqa: E402
from config import cfg  # noqa: E402
from datasets.registry import Column, Dataset, Registry  # noqa: E402

TABLES = {"events": 5_000, "events_enterprise": 2_000_000}
"""Table -> sessions; about 10 events each."""

COLUMNS = [("event_id", "bigint"), ("user_id", "integer"), ("session_id", "bigint"),
           ("device_type", "text"), ("event_type", "text"), ("event_time", "timestamp with time zone")]

CREATE = """
DROP TABLE IF EXISTS {t};
CREATE TABLE {t} (
    event_id    bigint GENERATED ALWAYS AS IDENTITY,
    user_id     integer NOT NULL,
    session_id  bigint NOT NULL,
    device_type text NOT NULL,
    event_type  text NOT NULL,
    event_time  timestamptz NOT NULL
);
INSERT INTO {t} (user_id, session_id, device_type, event_type, event_time)
SELECT s.user_id, s.session_id, s.device_type,
       (ARRAY['view', 'click', 'scroll', 'add_to_cart', 'purchase'])[1 + floor(random() * 5)::int],
       s.start_time + make_interval(secs => e.i * (30 + random() * 240) * s.pace)
FROM (
    SELECT g AS session_id,
           1 + floor(random() * {users})::int AS user_id,
           d.device_type, d.pace,
           now() - random() * interval '60 days' AS start_time,
           1 + floor(random() * 19)::int AS n_events
    FROM generate_series(1, {sessions}) AS g
    CROSS JOIN LATERAL (
        SELECT * FROM (VALUES ('desktop', 1.5), ('mobile', 0.6), ('tablet', 1.0)) AS v(device_type, pace)
        ORDER BY random() + g * 0 LIMIT 1
    ) AS d
) AS s
CROSS JOIN LATERAL generate_series(0, s.n_events - 1) AS e(i);
GRANT SELECT ON {t} TO "{reader}";
ANALYZE {t};
"""

NAIVE_SQL = """SELECT *
FROM events
WHERE event_time >= CURRENT_DATE - INTERVAL '30 days'"""

NAIVE_SCRIPT = '''import os

import plotly.express as px
import polars as pl

events = pl.read_parquet(os.path.join(os.environ["DATA_DIR"], "recent_events.parquet"))

# One row per session: how long from its first to its last event.
sessions = events.group_by("device_type", "user_id", "session_id").agg(
    ((pl.col("event_time").max() - pl.col("event_time").min()).dt.total_seconds() / 60).alias("session_minutes")
)
medians = (
    sessions.group_by("device_type")
    .agg(pl.col("session_minutes").median().alias("median_session_minutes"))
    .sort("device_type")
)
print(medians)

fig = px.bar(medians, x="device_type", y="median_session_minutes",
             title="Median session length by device (last 30 days)",
             labels={"device_type": "Device", "median_session_minutes": "Median session (minutes)"})
fig.write_html(os.path.join(os.environ["OUTPUT_DIR"], "session_length_by_device.html"), include_plotlyjs="cdn")
'''


async def make_tables() -> None:
    reader = urlsplit(cfg.database.dsn).username
    conn = await asyncpg.connect(cfg.database.ingest_dsn)
    registry = Registry(cfg.server.uploads_dir / "registry.json")
    try:
        for table, sessions in TABLES.items():
            started = time.monotonic()
            await conn.execute(CREATE.format(t=table, sessions=sessions, users=max(500, sessions // 4), reader=reader),
                               timeout=3600)
            rows = await conn.fetchval(f"SELECT reltuples::bigint FROM pg_class WHERE relname = '{table}'")
            size = await conn.fetchval(f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))")
            registry.remove(table)
            registry.add(Dataset(name=table, kind="table", rows=rows, columns=[Column(name=n, type=t) for n, t in COLUMNS],
                                 source="scripts/make_events_demo.py", created_at=datetime.now(UTC)))
            print(f"{table:18} ~{rows:,} rows, {size}, {time.monotonic() - started:.0f}s", flush=True)
    finally:
        await conn.close()


def save_naive_analysis() -> None:
    store = AnalysisStore(cfg.server.analyses_dir)
    analysis = Analysis(
        id="sessiondemo1",
        title="Median session length by device",
        description="Median minutes from a session's first to its last event, per device type, over the last 30 days.",
        question="What is the median session length by device over the last 30 days?",
        inputs=[QueryInput(sql=NAIVE_SQL, file="recent_events.parquet")],
        script=NAIVE_SCRIPT,
        outputs=[Output(file="session_length_by_device.html")],
    )
    problems = analysis.problems()
    if problems:
        raise SystemExit(f"the demo recipe does not pass the save checks: {problems}")
    if store.get(analysis.id) is not None:
        store.delete(analysis.id)
    store.save(analysis)
    print(f"saved analysis {analysis.id}: {analysis.title} (version 1, the naive raw-export strategy)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-analysis", action="store_true", help="also save the naive demo analysis")
    parser.add_argument("--skip-tables", action="store_true", help="only (re)save the demo analysis")
    args = parser.parse_args()
    if not args.skip_tables:
        asyncio.run(make_tables())
    if args.save_analysis:
        save_naive_analysis()


if __name__ == "__main__":
    main()
