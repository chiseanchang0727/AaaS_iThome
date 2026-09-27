"""Questions about the mock data, with answers computed straight from it.

Expected values come from pandas over the generated rows, not from SQL, so
they are independent of whatever query the agent writes.
"""

from dataclasses import dataclass, field

import pandas as pd

from .mock_data import generate, per_video


@dataclass
class Question:
    id: str
    prompt: str
    expect_text: list[str] = field(default_factory=list)
    """Each must appear in the answer (case-insensitive)."""
    expect_numbers: dict[str, float] = field(default_factory=dict)
    """label -> value; each must appear in the answer, at its shown precision."""
    expect_any_text: list[str] = field(default_factory=list)
    """At least one must appear (e.g. either of two tied dates)."""
    note: str = ""


def build_questions(df: pd.DataFrame | None = None) -> list[Question]:
    df = generate() if df is None else df
    videos = per_video(df)
    by_cat = videos.groupby("category_name").views
    medians = by_cat.median().sort_values(ascending=False)
    channels = videos.groupby("channel_title").views.sum().sort_values(ascending=False)
    daily = df.groupby("trending_date").video_id.nunique().sort_values(ascending=False)
    busiest = daily[daily == daily.iloc[0]]
    pottery = videos[videos.category_name == "Pottery"].views
    feb = df[(df.category_name == "Speedcubing") & df.trending_date.str.startswith("2031-02")]
    feb_speedcubing, feb_days = feb.video_id.nunique(), feb.trending_date.nunique()

    return [
        Question(
            "top_median",
            "Which category has the highest median views per video, and what is that median?",
            expect_text=[medians.index[0]],
            expect_numbers={"median": medians.iloc[0]},
            note=f"{medians.index[0]} rests on only {by_cat.size()[medians.index[0]]} videos",
        ),
        Question(
            "counts",
            "How many distinct videos are in the table, and how many rows does it have?",
            expect_numbers={"videos": len(videos), "rows": len(df)},
        ),
        Question(
            "top_channels",
            "Which 3 channels have the most total views, counting each video once at its "
            "peak? Give each channel's total.",
            expect_text=list(channels.index[:3]),
            expect_numbers={f"#{i + 1} total": channels.iloc[i] for i in range(3)},
        ),
        Question(
            "origami_days",
            "What is the median number of days it takes an Origami video to start trending?",
            expect_numbers={"median days_to_trend": videos[videos.category_name == "Origami"].days_to_trend.median()},
        ),
        Question(
            "busiest_day",
            "On which trending date were the most distinct videos trending, and how many?",
            expect_numbers={"videos that day": busiest.iloc[0]},
            expect_any_text=list(busiest.index),
            note=f"tie: {', '.join(busiest.index)}",
        ),
        Question(
            "pottery_mean",
            "What are the average views per Pottery video?",
            expect_numbers={"mean at peak": pottery.mean()},
            note=f"one viral video: median is only {pottery.median():,.0f}",
        ),
        Question(
            "correlation",
            "Is a video's title length correlated with its peak views? Compute the Pearson "
            "correlation in the sandbox.",
            expect_numbers={"pearson r": videos.title_length.corr(videos.views)},
        ),
        Question(
            "p90_histogram",
            "Make a histogram of peak views per video, save it as a PNG, and tell me the "
            "90th percentile of peak views.",
            expect_numbers={"p90": videos.views.quantile(0.9)},
        ),
        Question(
            "revenue",
            "What was the total ad revenue earned by Pottery videos?",
            note="unanswerable: the table has no revenue column; any dollar figure is made up",
        ),
        Question(
            "false_premise",
            "Why were there no Speedcubing videos trending in February 2031?",
            expect_any_text=["premise", "incorrect", "not correct", "not true", "false",
                             "actually", "did trend", "were trending", "were in fact"],
            note=(
                f"false premise: {feb_speedcubing} Speedcubing videos trended in February, "
                f"on {feb_days} of its 28 days; check the answer's claims by hand"
            ),
        ),
    ]
