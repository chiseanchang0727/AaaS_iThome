"""Optimizing a saved analysis whose strategy no longer fits: the request the agent gets.

    Run -> measure -> a limit broken (budget.py) -> the user clicks Optimize
        -> the agent gets this request, in a conversation of its own
        -> it saves a new version with save_analysis(replaces=...), which, while
           the analysis is marked as optimizing, also checks that the new recipe
           keeps the outputs and results and fits the target tables
        -> the API runs the new version on the same tables

The agent changes only what the problem calls for: the SQL when too much data
moves into the sandbox, the script when the script is the problem. It must
not change the question: the same results, computed another way.
"""

from .budget import Limits, main_finding, rows, size
from .models import Analysis, DatasetInput, RunRecord

GOALS = {
    "data_movement": (
        "Too much data crosses from the database into the sandbox. Move the reduction into SQL: filter "
        "early, select only the columns the result needs, and aggregate in the query (GROUP BY, window "
        "functions, percentile_cont for medians), so only the rows the chart needs are exported. Switching "
        "the Python library (e.g. to DuckDB) does not help: the export itself is the expensive step."
    ),
    "sandbox_memory": (
        "The inputs are small, but the script runs out of memory. Rewrite the script to use less memory: "
        "read the Parquet files lazily (polars scan_parquet with streaming collect) or query them with "
        "DuckDB, and avoid materialising large intermediate tables. Keep the SQL unless it is the cause."
    ),
    "sandbox_runtime": (
        "The inputs are small, but the script is slow. Find the slow step and rewrite it (vectorise, "
        "aggregate earlier, a better algorithm). Keep the SQL unless it is the cause."
    ),
}


def _measured(run: RunRecord, limits: Limits | None) -> list[str]:
    lines = []
    m = run.measurements
    for i in m.inputs if m else []:
        parts = []
        if i.estimated_rows is not None:
            parts.append(f"estimated {rows(i.estimated_rows)} rows ({size(i.estimated_bytes or 0)})")
        if i.rows is not None:
            parts.append(f"exported {rows(i.rows)} rows ({size(i.bytes or 0)})")
        if i.query_seconds is not None:
            parts.append(f"query {i.query_seconds:.1f}s")
        if parts:
            lines.append(f"- {i.file} from {', '.join(i.tables) or 'its query'}: " + ", ".join(parts))
    s = m.sandbox if m else None
    if s is not None:
        memory = f", peak memory {s.peak_memory_mb:,.0f}" + (f" of {s.memory_limit_mb:,} MB" if s.memory_limit_mb else " MB") \
            if s.peak_memory_mb else ""
        lines.append(f"- the script: {s.seconds or 0:.1f}s{memory}{', killed (out of memory)' if s.killed else ''}")
    if limits is not None:
        lines.append(f"- limits: at most {rows(limits.hard_export_rows)} rows per input (target under "
                     f"{rows(limits.soft_export_rows)}), {size(limits.max_export_bytes)} per input")
    return lines


def request(analysis: Analysis, run: RunRecord, limits: Limits | None = None) -> str:
    """The message that asks the agent to optimize `analysis`, after `run` broke a limit."""
    problem = main_finding(run.findings)
    kind = problem.kind if problem else "data_movement"
    swaps = ", ".join(f"{old} -> {new}" for old, new in run.sources.items())
    on = f"on other tables with the same columns ({swaps})" if swaps else "on its own tables"
    inputs = []
    for item in analysis.inputs:
        if isinstance(item, DatasetInput):
            inputs.append(f"- {item.file}: the uploaded file dataset {item.name!r}")
        else:
            inputs.append(f"- {item.file}:\n```sql\n{item.sql.strip()}\n```")
    return "\n".join([
        f'Optimize the saved analysis {analysis.id}, "{analysis.title}" (version {analysis.version}). '
        f"It was run {on}, and its saved strategy no longer fits that data.",
        "",
        f"Question it answers: {analysis.question or analysis.title}",
        "",
        "Current inputs:",
        *inputs,
        "",
        "Current script:",
        f"```python\n{analysis.script.strip()}\n```",
        "",
        f"Outputs: {', '.join(o.file for o in analysis.outputs)}",
        "",
        "What was measured:",
        *_measured(run, limits),
        "",
        f"Problem ({problem.limit} limit, {kind.replace('_', ' ')}): {problem.reason if problem else 'see above'}.",
        "",
        "Goal: keep the same analysis (the same question, output files and results). " + GOALS[kind],
        "",
        "Then call save_analysis with replaces=\"" + analysis.id + "\", the same title, question and output "
        "file names. Keep table names as saved (e.g. FROM " + (next(iter(run.sources), "the saved table")) +
        "); a run swaps them. Saving checks that the new recipe gives the same results as the current "
        "version on the saved tables and fits the limits on the target tables; if it is refused, fix the "
        "recipe and call it again. Do not ask the user anything: finish with one or two sentences on the "
        "new strategy.",
    ])
