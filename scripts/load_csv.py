"""Load the CSV in `cfg.data.path` into the `videos` table.

    uv run python scripts/load_csv.py

Full reload: TRUNCATE then COPY, in one transaction, so a failure leaves the
old contents in place. Connects as the owner ($DATABASE_URL), not as the
read-only role the agent uses.
"""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataType, cfg  # noqa: E402

SCHEMA = Path(__file__).resolve().parent.parent / "datasources" / "schema.sql"

COLUMNS = [
    "video_id", "title", "channel_title", "category_id", "category_name",
    "country", "publish_time", "trending_date", "publish_hour", "publish_day",
    "days_to_trend", "views", "likes", "dislikes", "comment_count",
    "like_ratio", "dislike_ratio", "like_dislike_ratio", "title_length",
    "title_caps_ratio", "tag_count", "comments_disabled", "ratings_disabled",
]


async def main() -> None:
    path = cfg.data.path
    if cfg.data.format is not DataType.CSV:
        raise SystemExit(f"{path} is not a CSV")
    if not path.exists():
        raise SystemExit(f"{path} does not exist (run from the repo root)")

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set (owner connection, not the agent's)")

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(SCHEMA.read_text())
        async with conn.transaction():
            await conn.execute("TRUNCATE videos RESTART IDENTITY")
            result = await conn.copy_to_table(
                "videos", source=path, columns=COLUMNS, format="csv", header=True
            )
        count = await conn.fetchval("SELECT count(*) FROM videos")
        print(f"{result} -> videos now holds {count} rows")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
