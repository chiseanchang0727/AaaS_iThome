---
name: query_database
description: What you need to know about the `videos` table before querying it with query_database. Its rows are not one per video, so the obvious SUM, COUNT or ORDER BY gives wrong answers.
---

# The `videos` table

## One row is one video on one trending day

A video that trended for a week appears seven times, once per day, and its
`views`, `likes` and `comment_count` are the running totals on that day. The
table has 40,949 rows but only 6,351 distinct videos (all US, trending
2017-11-14 to 2018-06-14).

So before writing a query, decide what the question is about:

- **Videos** (top videos, views by category, busiest channels): one row per
  video first, usually its latest (peak) numbers. Otherwise a video counts once
  per day it trended, and a "top 5" can be one video five times.
- **Trending activity over time** (views per week, busiest day): keep the daily
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

- **Views are heavily skewed.** A few viral videos have hundreds of millions of
  views, while most have far fewer. Totals and averages mostly reflect those few;
  say which measure you used and why.
- **Some groups are tiny.** `Shows` has 4 videos and `Nonprofits & Activism` 14,
  so a ranking can be led by a handful of videos. Report how many videos a
  number rests on.
