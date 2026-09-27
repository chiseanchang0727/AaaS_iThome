"""Synthetic `videos` data the model cannot know from training.

Fictional categories and channels, dates in 2031, deliberately odd numbers:
a right answer can only come from querying, and a made-up one stands out.
It keeps the real table's shape (one row per video per trending day) so the
same counting mistakes are possible.

    uv run python -m evals.grounding.mock_data    # writes CSV + skill into evals/grounding/out/

Loading the CSV into the `ithome_mock` database is a one-off, see run.py.
"""

import csv
import datetime as dt
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CSV_PATH = OUT / "mock_videos.csv"
SKILL_PATH = OUT / "skills" / "query_database" / "SKILL.md"

COLUMNS = [
    "video_id", "title", "channel_title", "category_id", "category_name",
    "country", "publish_time", "trending_date", "publish_hour", "publish_day",
    "days_to_trend", "views", "likes", "dislikes", "comment_count",
    "like_ratio", "dislike_ratio", "like_dislike_ratio", "title_length",
    "title_caps_ratio", "tag_count", "comments_disabled", "ratings_disabled",
]

# name: (category_id, number of videos, median starting views, like ratio)
CATEGORIES = {
    "Pottery": (101, 70, 41_000, 0.061),
    "Beekeeping": (102, 55, 67_000, 0.048),
    "Speedcubing": (103, 48, 129_000, 0.035),
    "Birdwatching": (104, 40, 23_000, 0.072),
    "Origami": (105, 30, 88_000, 0.053),
    "Kayaking": (106, 5, 310_000, 0.029),  # tiny: a ranking trap
}
CHANNELS = [
    "Clay & Kiln", "The Wheel Room", "Glaze Theory", "Hive Minded", "Smoker & Veil",
    "Queen Cell", "Cubehead", "Sub-10 Club", "F2L Daily", "Warbler Watch", "Owl Hours",
    "Fieldmark", "Crease Lab", "Valley Fold", "Paper Crane Co", "Rapid Class IV",
    "Eddy Line", "Portage", "Kiln Gods", "Nectar Flow",
]
WORDS = ["how", "to", "the", "best", "first", "ultimate", "guide", "why", "never",
         "secret", "beginner", "pro", "mistakes", "tutorial", "vs", "challenge", "day"]
START = dt.date(2031, 1, 5)


def generate(seed: int = 7) -> pd.DataFrame:
    """One row per video per trending day, deterministic for a given seed."""
    rng = random.Random(seed)
    channels_by_category = {
        name: CHANNELS[i::len(CATEGORIES)] for i, name in enumerate(CATEGORIES)
    }
    rows, video_number = [], 0
    for category, (category_id, n_videos, median_views, like_ratio) in CATEGORIES.items():
        for _ in range(n_videos):
            video_number += 1
            video_id = f"mv{video_number:04d}{rng.choice('abcdefghjk')}"
            title = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 9)))
            if rng.random() < 0.3:
                title = title.upper()
            publish = dt.datetime.combine(
                START + dt.timedelta(days=rng.randint(0, 80)),
                dt.time(rng.randint(0, 23), rng.choice([0, 15, 30, 45])),
                tzinfo=dt.UTC,
            )
            days_to_trend = rng.randint(1, 6)
            views = median_views * rng.lognormvariate(0, 0.9)
            ratio = like_ratio * rng.uniform(0.8, 1.2)
            channel = rng.choice(channels_by_category[category])
            tags = rng.randint(0, 30)
            caps = sum(c.isupper() for c in title) / max(1, sum(c.isalpha() for c in title))
            comments_off, ratings_off = rng.random() < 0.03, rng.random() < 0.02
            for day in range(rng.randint(1, 9)):
                views *= rng.uniform(1.08, 1.6) if day else 1
                likes = 0 if ratings_off else round(views * ratio)
                dislikes = 0 if ratings_off else round(likes * rng.uniform(0.02, 0.09))
                trending = publish.date() + dt.timedelta(days=days_to_trend + day)
                rows.append({
                    "video_id": video_id, "title": title, "channel_title": channel,
                    "category_id": category_id, "category_name": category, "country": "US",
                    "publish_time": publish.isoformat(), "trending_date": trending.isoformat(),
                    "publish_hour": publish.hour, "publish_day": publish.strftime("%A"),
                    "days_to_trend": float(days_to_trend), "views": round(views),
                    "likes": likes, "dislikes": dislikes,
                    "comment_count": 0 if comments_off else round(views * rng.uniform(0.002, 0.01)),
                    "like_ratio": likes / round(views), "dislike_ratio": dislikes / round(views),
                    "like_dislike_ratio": likes / dislikes if dislikes else 0.0,
                    "title_length": len(title), "title_caps_ratio": round(caps, 4),
                    "tag_count": tags, "comments_disabled": comments_off,
                    "ratings_disabled": ratings_off,
                })

    # One viral outlier, so a mean and a median disagree.
    viral = [r for r in rows if r["category_name"] == "Pottery"][-1]["video_id"]
    for r in rows:
        if r["video_id"] == viral:
            r["views"] *= 180
    return pd.DataFrame(rows, columns=COLUMNS)


def per_video(df: pd.DataFrame) -> pd.DataFrame:
    """One row per video at its peak, the same collapse the skill teaches."""
    return df.sort_values("views").groupby("video_id", as_index=False).last()


def write_skill(df: pd.DataFrame, path: Path = SKILL_PATH) -> None:
    """The query_database skill, with facts true for this mock data."""
    videos = per_video(df)
    sizes = videos.groupby("category_name").size().sort_values()
    smallest, second = sizes.index[0], sizes.index[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""---
name: query_database
description: What you need to know about the `videos` table before querying it with query_database. Its rows are not one per video, so the obvious SUM, COUNT or ORDER BY gives wrong answers.
---

# The `videos` table

## One row is one video on one trending day

A video that trended for a week appears seven times, once per day, and its
`views`, `likes` and `comment_count` are the running totals on that day. The
table has {len(df):,} rows but only {len(videos):,} distinct videos (all US, trending
{df.trending_date.min()} to {df.trending_date.max()}).

So before writing a query, decide what the question is about:

- **Videos** (top videos, views by category, busiest channels): one row per
  video first, usually its latest (peak) numbers. Otherwise a video counts once
  per day it trended, and a "top 5" can be one video five times.
- **Trending activity over time** (videos per day, busiest day): keep the daily
  rows; a video *should* count on every day it trended.

One way to get one row per video:

```sql
WITH per_video AS (
    SELECT video_id, MAX(views) AS views, MIN(category_name) AS category_name
    FROM videos
    GROUP BY video_id
)
SELECT ... FROM per_video ...
```

## Other things worth knowing

- **Views are heavily skewed.** A few viral videos have far more views than
  the rest. Totals and averages mostly reflect those few; say which measure
  you used and why.
- **Some groups are tiny.** `{smallest}` has {sizes.iloc[0]} videos and `{second}` {sizes.iloc[1]},
  so a ranking can be led by a handful of videos. Report how many videos a
  number rests on.
""")


def write_csv(df: pd.DataFrame, path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    df = generate()
    write_csv(df)
    write_skill(df)
    print(f"{len(df):,} rows, {df.video_id.nunique():,} videos -> {CSV_PATH.relative_to(HERE.parent.parent)}")
    print(f"skill -> {SKILL_PATH.relative_to(HERE.parent.parent)}")
