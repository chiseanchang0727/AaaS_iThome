"""The workflow advisor: an agent that suggests how to rearrange a workflow, from its past runs.

    summary (code) -> the model -> suggestions -> checked (code) -> the user applies or dismisses each

The scheduler (api/workflows.py) balances one run as it happens; the advisor
looks at the last few runs and suggests changes to the workflow itself. It
never changes anything: the user applies each suggestion, and only these kinds
can be applied:

    set_lanes          how many sandboxes a run of this workflow uses (its setting)
    run_first          give these analysis nodes priority in the queue
    optimize_analysis  start Optimize for an analysis whose last run broke a limit (Day 24)

The others are advice the user acts on by hand: share_data (analyses that read
the same data), split_workflow, other.

What the model gets is a summary, not the raw runs: the workflow, each
analysis's recipe and its last runs' phases, sizes, memory and findings, the
runs' totals, the limits, table sizes, which analyses read the same tables,
and the goal (fastest or cheapest). Every suggestion is checked before it is
shown: unknown nodes, impossible values and kinds that do not apply are
dropped or shown as advice only.
"""

import json
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from analyses.models import Analysis, QueryInput, RunRecord, now
from analyses.sources import swap_tables, tables

from .models import Workflow, WorkflowRun

Kind = Literal["set_lanes", "run_first", "optimize_analysis", "share_data", "split_workflow", "other"]
APPLICABLE = {"set_lanes", "run_first", "optimize_analysis"}
Goal = Literal["fastest", "cheapest"]

RUNS = 3
"""How many recent runs the summary covers."""
SQL_CHARS = 600


class Suggestion(BaseModel):
    """One change the advisor suggests, as the model writes it."""

    kind: Kind
    node_ids: list[str] = Field(default=[], description="The analysis node ids it concerns (from the summary)")
    value: int | None = Field(default=None, description="For set_lanes: the number of sandboxes")
    reason: str = Field(description="Why, from the measured numbers in the summary")
    expected_gain: str = Field(description="What it should change, e.g. 'about 5 s less per run', or 'one sandbox less'")


class Advice(BaseModel):
    """The model's answer."""

    summary: str = Field(description="One or two sentences: where this workflow's time and cost go")
    suggestions: list[Suggestion] = Field(default=[], description="At most five, most useful first; none if nothing helps")


class CheckedSuggestion(Suggestion):
    applicable: bool = False
    """It can be applied with one click; otherwise it is advice."""
    note: str = ""
    """Why it cannot be applied, or what applying it does."""
    status: Literal["pending", "applied", "dismissed", "failed"] = "pending"
    result: str = ""


class AdviceRecord(BaseModel):
    """One time the advisor was asked, as stored under the workflow."""

    id: str
    workflow_id: str
    created_at: str = Field(default_factory=now)
    goal: Goal = "fastest"
    runs: list[str] = []
    """The runs it looked at, newest first."""
    model: str = ""
    summary: str = ""
    suggestions: list[CheckedSuggestion] = []


Advise = Callable[[dict[str, Any], Goal], Awaitable[Advice]]
"""(summary, goal) -> the model's advice."""


# --- the summary ------------------------------------------------------------------------------------

def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()  # noqa: E731
    return round(parse(end) - parse(start), 1)


def _phases(record: RunRecord) -> dict[str, Any]:
    """One analysis run, measured: its phases in seconds, what it moved, its memory, CPU and findings."""
    sandbox = record.measurements.sandbox if record.measurements else None
    inputs = record.measurements.inputs if record.measurements else []
    samples = [m for _, m, _ in (sandbox.samples if sandbox else []) if m is not None]
    script_end = None
    if sandbox and sandbox.started_at and sandbox.seconds is not None:
        script_end = datetime.fromisoformat(sandbox.started_at.replace("Z", "+00:00")).timestamp() + sandbox.seconds
    after = None
    if script_end is not None and record.finished_at:
        after = round(datetime.fromisoformat(record.finished_at.replace("Z", "+00:00")).timestamp() - script_end, 1)
    return {
        "status": record.status,
        "before_sandbox_s": _seconds(record.queued_at, record.sandbox_at),
        "waiting_for_data_s": _seconds(record.sandbox_at, sandbox.started_at if sandbox else None),
        "script_s": round(sandbox.seconds, 1) if sandbox and sandbox.seconds is not None else None,
        "after_script_s": after,
        "rows_in": sum(i.rows or 0 for i in inputs) if inputs else None,
        "mb_in": round(sum(i.bytes or 0 for i in inputs) / 1e6, 2) if inputs else None,
        "query_s": round(sum(i.query_seconds or 0 for i in inputs), 1) if inputs else None,
        "peak_memory_mb": sandbox.peak_memory_mb if sandbox else None,
        "average_memory_mb": round(sum(samples) / len(samples)) if samples else None,
        "memory_limit_mb": sandbox.memory_limit_mb if sandbox else None,
        "cpu_s": sandbox.cpu_seconds if sandbox else None,
        "reused_sandbox": record.reused,
        "findings": [f"{f.limit} {f.kind}: {f.reason}" for f in record.findings],
        "error": record.error,
    }


_TIME_WINDOW = re.compile(r"\b(now\s*\(\)|current_date|current_timestamp)", re.IGNORECASE)


def summarize(workflow: Workflow, runs: list[WorkflowRun], analyses: dict[str, Analysis],
              records: dict[str, RunRecord], settings: dict[str, Any], table_rows: dict[str, int],
              goal: Goal) -> dict[str, Any]:
    """What the advisor is told: plain data, a few kilobytes. `records`: analysis run id -> its record."""
    batches = workflow.batches()
    in_batches = {nid: [workflow.node(b).title for b, ids in batches.items() if nid in ids] for nid in
                  {i for ids in batches.values() for i in ids}}
    nodes, reads = [], {}
    for node in workflow.nodes:
        if node.kind != "analysis":
            continue
        analysis = analyses.get(node.analysis_id)
        sql = [swap for item in (analysis.inputs if analysis else []) if isinstance(item, QueryInput)
               for swap in [_swapped(item.sql, node.sources)]]
        tables = sorted({t for s in sql for t in _tables(s)})
        reads[node.id] = tuple(tables)
        nodes.append({
            "node_id": node.id,
            "analysis": analysis.title if analysis else f"{node.analysis_id} (deleted)",
            "version": analysis.version if analysis else None,
            "batches": in_batches.get(node.id, []),
            "tables": tables,
            "priority": node.priority,
            "sql": [s[:SQL_CHARS] for s in sql],
            "uses_a_time_window": any(_TIME_WINDOW.search(s) for s in sql),
            "outputs": [o.file for o in analysis.outputs] if analysis else [],
            "last_runs": [_phases(records[n.run_id]) for run in runs for n in run.nodes
                          if n.node_id == node.id and n.run_id in records],
        })
    shared: dict[tuple, list[str]] = {}
    for node_id, tables in reads.items():
        if tables:
            shared.setdefault(tables, []).append(node_id)
    return {
        "goal": goal,
        "workflow": {"title": workflow.title, "lanes_setting": workflow.settings.lanes},
        "settings": settings,
        "runs": [{
            "run": run.id, "total_s": run.seconds, "status": run.status, "lanes": run.lanes,
            # Runs from before sandboxes were passed on (lanes None) rented one per analysis.
            "shared_sandboxes": run.lanes is not None,
            "sandboxes_started": len({n.sandbox or n.node_id for n in run.nodes if n.sandbox_at}),
            "analyses": len(run.nodes),
            "failed_or_stopped": [n.node_id for n in run.nodes if n.status in ("failed", "needs_optimization")],
        } for run in runs],
        "analyses": nodes,
        "table_rows": table_rows,
        "read_the_same_tables": [{"tables": list(t), "nodes": ids} for t, ids in shared.items() if len(ids) > 1],
    }


def _swapped(sql: str, sources: dict[str, str]) -> str:
    try:
        return swap_tables(sql, sources)
    except Exception:
        return sql


def _tables(sql: str) -> list[str]:
    try:
        return tables(sql)
    except ValueError:
        return []


# --- the model ------------------------------------------------------------------------------------------

INSTRUCTIONS = """You advise on how to arrange a data-analysis workflow so its runs are {goal}.

A workflow runs saved analyses (each: SQL queries whose rows are copied into a sandbox, then a Python
script that makes charts). A run uses a few sandboxes ("lanes"), each running analyses one after another;
starting a sandbox takes 10-15 seconds (settings.warm_sandboxes are kept started), a call to it about one.
Analyses are queued by priority, then longest first. Each analysis's phases: before_sandbox_s (size check,
waiting for a free sandbox, start-up), waiting_for_data_s (its database query was still running), script_s,
after_script_s (outputs back).

Facts to respect:
- settings.sandbox_limit is per ACCOUNT, shared by every workflow and the chat: splitting a workflow, or
  running two at once, does not add sandboxes. By default a run uses one fewer (default_lanes), so the chat
  keeps one; using all of them blocks the chat while the workflow runs: say so if you suggest it.
- Runs with shared_sandboxes false are from before sandboxes were passed between analyses (one rented per
  analysis): they show the old design's cost. Base suggestions on runs with shared_sandboxes true when there
  are any; use older ones only for facts that did not change (query times, memory, rows).

Suggest changes to the workflow, only where the numbers below show a gain. Kinds:
- set_lanes (value): sandboxes a run of it may use; at most settings.sandbox_limit. More lanes help when
  analyses wait for a sandbox; fewer save cost when sandboxes sit idle.
- run_first (node_ids): put analyses at the front of the queue, e.g. one whose slow query would otherwise
  hold up the end of the run.
- optimize_analysis (one node id): its last run broke a limit (it has findings), or its query or script is
  clearly the slow part; Optimize rewrites its recipe (SQL or script) and checks the results stay the same.
- share_data (node_ids): analyses that read the same tables and filters, so one query could serve them.
- split_workflow (node_ids): analyses that would be better in a workflow of their own.
- other: anything else worth saying, with evidence.

Use node ids exactly as given. At most five suggestions, most useful first. Each reason must cite the
measured numbers. Do not suggest what the numbers do not support; if nothing would help, say so in the
summary and return no suggestions.

The workflow, its last runs and its settings:
{summary}"""


def advisor(model: Any) -> Advise:
    """Ask `model` (a model name such as 'anthropic:claude-sonnet-4-6', or a chat model) for advice."""
    chat = init_chat_model(model, temperature=0) if isinstance(model, str) else model

    async def advise(summary: dict[str, Any], goal: Goal) -> Advice:
        text = INSTRUCTIONS.format(goal="as fast as possible" if goal == "fastest" else "as cheap as possible "
                                   "(fewer sandboxes and less time in them)",
                                   summary=json.dumps(summary, indent=1, ensure_ascii=False))
        answer = await chat.with_structured_output(Advice).ainvoke(text)
        if answer is None:
            raise ValueError("the model returned no advice")
        return answer if isinstance(answer, Advice) else Advice.model_validate(answer)

    return advise


# --- checking -------------------------------------------------------------------------------------------

def check(advice: Advice, workflow: Workflow, sandbox_limit: int | None,
          findings: dict[str, bool]) -> list[CheckedSuggestion]:
    """Each suggestion checked against the workflow: dropped if it names no real analysis node or is a
    duplicate; applicable only when it can be applied as it stands. `findings`: node id -> whether its
    last run broke a limit (only then can Optimize start)."""
    analysis_nodes = {n.id for n in workflow.nodes if n.kind == "analysis"}
    checked, seen = [], set()
    for s in advice.suggestions[:5]:
        ids = [i for i in dict.fromkeys(s.node_ids) if i in analysis_nodes]
        if s.node_ids and not ids:
            continue  # it names nodes that are not there
        key = (s.kind, tuple(sorted(ids)), s.value)
        if key in seen:
            continue
        seen.add(key)
        c = CheckedSuggestion(**{**s.model_dump(), "node_ids": ids})
        if s.kind == "set_lanes":
            if s.value is None or s.value < 1:
                continue
            if sandbox_limit and s.value > sandbox_limit:
                c.value = sandbox_limit
                c.note = f"An account may hold at most {sandbox_limit} sandboxes: set to {sandbox_limit}. "
            c.applicable = c.value != workflow.settings.lanes
            c.note += (f"Runs of this workflow will use at most {c.value} sandboxes." if c.applicable
                       else "It is already set to that.")
        elif s.kind == "run_first":
            c.applicable = bool(ids)
            c.note = "These analyses go to the front of the queue on the next run." if ids else "Names no analysis."
        elif s.kind == "optimize_analysis":
            if len(ids) != 1:
                c.note = "Optimize works on one analysis at a time."
            elif findings.get(ids[0]):
                c.applicable = True
                c.note = "Starts Optimize for its last run: the agent rewrites the recipe, and it is checked before it is saved."
            else:
                c.note = ("Its last run broke no limit, so Optimize cannot start from it. Open the analysis to see "
                          "its run, or ask the agent in the chat to make it faster.")
        else:
            c.note = "Advice: change the workflow by hand if you agree."
        checked.append(c)
    return checked
