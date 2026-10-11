"""Three more saved analyses for the workflow demo, each on its own question:

    uv run python scripts/make_workflow_demo.py

    bookrevenue1   Monthly bookstore revenue by genre        bookstore_sales
    publishhour1   Average views by publish hour             videos
    dailyusers1    Daily active users by device (30 days)    events

They are saved as the agent's save_analysis would save them (SQL that does
the aggregation, a polars + plotly script), without its test run: the first
run is the workflow's. Saving again replaces them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses import Analysis, AnalysisStore, Output, QueryInput  # noqa: E402
from config import cfg  # noqa: E402


def script(file: str, output: str, chart: str) -> str:
    return f'''import os

import plotly.express as px
import polars as pl

data = pl.read_parquet(os.path.join(os.environ["DATA_DIR"], "{file}"))
print(data)

{chart}
fig.write_html(os.path.join(os.environ["OUTPUT_DIR"], "{output}"), include_plotlyjs="cdn")
'''


ANALYSES = [
    Analysis(
        id="bookrevenue1",
        title="Monthly bookstore revenue by genre",
        description="Revenue (TWD) each month per genre, online and in store together.",
        question="How does bookstore revenue change by month for each genre?",
        inputs=[QueryInput(file="revenue.parquet", sql="""SELECT month, genre, SUM(revenue_twd) AS revenue_twd
FROM bookstore_sales
GROUP BY month, genre
ORDER BY month, genre""")],
        script=script("revenue.parquet", "revenue_by_genre.html", '''fig = px.line(data.sort("month"), x="month", y="revenue_twd", color="genre", markers=True,
              title="Monthly bookstore revenue by genre",
              labels={"month": "Month", "revenue_twd": "Revenue (TWD)", "genre": "Genre"})'''),
        outputs=[Output(file="revenue_by_genre.html")],
    ),
    Analysis(
        id="publishhour1",
        title="Average views by publish hour",
        description="Mean views of trending videos by the hour they were published.",
        question="Which publish hour gets the most views on average?",
        inputs=[QueryInput(file="by_hour.parquet", sql="""SELECT publish_hour, AVG(views) AS avg_views, COUNT(*) AS videos
FROM videos
GROUP BY publish_hour
ORDER BY publish_hour""")],
        script=script("by_hour.parquet", "views_by_publish_hour.html", '''fig = px.bar(data.sort("publish_hour"), x="publish_hour", y="avg_views", hover_data=["videos"],
             title="Average views by publish hour",
             labels={"publish_hour": "Publish hour", "avg_views": "Average views", "videos": "Videos"})'''),
        outputs=[Output(file="views_by_publish_hour.html")],
    ),
    Analysis(
        id="dailyusers1",
        title="Daily active users by device (last 30 days)",
        description="Distinct users with at least one event each day, per device type.",
        question="How many users are active each day on each device over the last 30 days?",
        inputs=[QueryInput(file="daily_users.parquet", sql="""SELECT date_trunc('day', event_time)::date AS day, device_type, COUNT(DISTINCT user_id) AS users
FROM events
WHERE event_time >= NOW() - INTERVAL '30 days'
GROUP BY 1, 2
ORDER BY 1, 2""")],
        script=script("daily_users.parquet", "daily_users_by_device.html", '''fig = px.line(data.sort("day"), x="day", y="users", color="device_type",
              title="Daily active users by device (last 30 days)",
              labels={"day": "Day", "users": "Active users", "device_type": "Device"})'''),
        outputs=[Output(file="daily_users_by_device.html")],
    ),
]


def main() -> None:
    store = AnalysisStore(cfg.server.analyses_dir)
    for analysis in ANALYSES:
        problems = analysis.problems()
        if problems:
            raise SystemExit(f"{analysis.id} does not pass the save checks: {problems}")
        if store.get(analysis.id) is not None:
            store.delete(analysis.id)
        store.save(analysis)
        print(f"saved {analysis.id}: {analysis.title}")


if __name__ == "__main__":
    main()
