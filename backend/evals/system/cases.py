"""Eval cases for the whole agent system, over the grounding eval's mock data.

A case is a question, plus whatever we can say in advance about a good run.
Every field but `id` and `question` is optional:

    expected_values    label -> value that must appear in the final answer.
                       A number matches at the precision the answer shows it
                       (evals/grounding/scoring.py); a string must appear as is
                       (case-insensitive).
    required_tools     tool names the run must call (e.g. "execute")
    forbidden_tools    tool names it must not call
    required_skills    skill names it must read (/skills/<name>/SKILL.md)
    required_outputs   "chart", "table", "report", "python": checked from the
                       trace (see checks.required_outputs)
    expected_behavior  what a good run does, in words, for the judges
    setup              earlier messages sent first in the same conversation;
                       only the last turn (`question`) is evaluated
    sandbox            needs code execution; skipped by --no-sandbox

Expected values come from pandas over the generated rows, not from SQL, so
they do not depend on the query the agent writes.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from evals.grounding.mock_data import generate, per_video


@dataclass
class EvalCase:
    id: str
    question: str
    expected_values: dict[str, float | str] = field(default_factory=dict)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)
    required_outputs: list[str] = field(default_factory=list)
    expected_behavior: str = ""
    setup: list[str] = field(default_factory=list)
    sandbox: bool = False
    note: str = ""
    """For people reading the results, not for the judges."""

    @property
    def turns(self) -> list[str]:
        return [*self.setup, self.question]

    def to_dict(self) -> dict[str, Any]:
        return {k: _plain(v) for k, v in asdict(self).items()}


def _plain(value: Any) -> Any:
    """numpy numbers as Python ones, so a case can be saved as JSON."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value.item() if hasattr(value, "item") else value


def build_cases(df: pd.DataFrame | None = None) -> list[EvalCase]:
    df = generate() if df is None else df
    videos = per_video(df)
    sizes = videos.groupby("category_name").size()
    medians = videos.groupby("category_name").views.median().sort_values(ascending=False)
    top = medians.index[0]
    in_top = videos[videos.category_name == top].groupby("channel_title").views.sum().sort_values(ascending=False)
    channels = videos.groupby("channel_title").views.sum().sort_values(ascending=False)
    channel_videos = videos.groupby("channel_title").size()
    biggest = sizes.idxmax()

    return [
        EvalCase(
            "lookup_counts",
            "How many distinct videos are in the table, and how many rows does it have?",
            expected_values={"distinct videos": len(videos), "rows": len(df)},
            required_tools=["query_database"],
            required_skills=["query_database"],
            expected_behavior=(
                "Counts distinct video_id for videos and all rows for rows, and reports both. "
                "A video appears on many trending days, so the two numbers differ."
            ),
        ),
        EvalCase(
            "aggregate_origami_days",
            "What is the median number of days it takes an Origami video to start trending?",
            expected_values={
                "median days_to_trend": videos[videos.category_name == "Origami"].days_to_trend.median()
            },
            required_tools=["query_database"],
            expected_behavior=(
                "Takes each Origami video once (not once per trending day) and computes the "
                "median of days_to_trend, e.g. with PERCENTILE_CONT(0.5)."
            ),
        ),
        EvalCase(
            "multi_step_top_category_channel",
            "Which category has the highest median views per video? Within that category, which "
            "channel has the most total views, counting each video once at its peak?",
            expected_values={
                "category": top, "median": medians[top],
                "channel": in_top.index[0], "channel total": in_top.iloc[0],
            },
            required_tools=["query_database"],
            expected_behavior=(
                f"First finds the category by median peak views per video ({top}), then ranks its "
                "channels by the sum of each video's peak views. Should mention that the top "
                f"category rests on only {sizes[top]} videos if it says anything about reliability."
            ),
            note=f"{top} has only {sizes[top]} videos: a ranking trap",
        ),
        EvalCase(
            "chart_p90_histogram",
            "Use Python to make a histogram of peak views per video, and tell me the 90th "
            "percentile of peak views.",
            expected_values={"p90": videos.views.quantile(0.9)},
            required_tools=["execute"],
            required_outputs=["chart", "python"],
            expected_behavior=(
                "Exports one row per video (its peak views) with export_query, plots a histogram "
                "with plotly saved as HTML in the outputs folder, and prints the 90th percentile "
                "before reporting it."
            ),
            sandbox=True,
        ),
        EvalCase(
            "follow_up_second_channel",
            "How many videos does the second one have?",
            setup=[
                "Which 3 channels have the most total views, counting each video once at its "
                "peak? Give each channel's total.",
            ],
            expected_values={"channel": channels.index[1], "videos": channel_videos[channels.index[1]]},
            expected_behavior=(
                f"'The second one' is the second channel from the previous answer ({channels.index[1]}). "
                "Counts that channel's distinct videos."
            ),
        ),
        EvalCase(
            "recover_wrong_column",
            "Using the `category` column, which category has the most distinct videos, and how many?",
            expected_values={"category": biggest, "videos": sizes[biggest]},
            required_tools=["query_database"],
            expected_behavior=(
                "There is no `category` column; the category name is in `category_name`. A good "
                "run checks the columns or reads the SQL error, fixes the query and answers. It "
                "does not repeat the failing query unchanged."
            ),
            note="tests recovery only if the agent actually trips over the column name",
        ),
        EvalCase(
            "oom_ward_clustering",
            "In the sandbox, generate 16,000 random 2-D points and run Ward hierarchical clustering "
            "on all of them (scipy). Cut the tree into 5 clusters and tell me how many points are "
            "in each.",
            required_tools=["execute"],
            required_outputs=["python"],
            expected_behavior=(
                "Ward linkage on 16,000 points needs about 1 GB for the distance matrix, so it is "
                "likely killed for running out of memory in a 1 GB sandbox. After that, a good run "
                "reads the error and changes approach (a leaner method, or asks for a bigger "
                "sandbox) instead of re-running the same code. The cluster sizes it reports must "
                "come from printed output and add up to 16,000."
            ),
            sandbox=True,
            note="the question from a real conversation that ran out of memory",
        ),
        EvalCase(
            "unanswerable_revenue",
            "What was the total ad revenue earned by Pottery videos?",
            forbidden_tools=["request_bigger_sandbox"],
            expected_behavior=(
                "The table has no revenue column. A good run checks the columns, says revenue is "
                "not in the data, and does not estimate a dollar figure."
            ),
        ),
    ]
