"""Workflows from the chat: the agent groups saved analyses into a workflow, and runs it.

    list_workflows   the workflows there are, with their sections and last run
    save_workflow    a new workflow, or a change to one: sections (batches) of saved analyses, one report
    run_workflow     run it and wait: how long it took, what finished, what needs attention, the links

The agent describes sections, not nodes and edges: code builds the canvas
(an analysis node per analysis, a batch node per section, one report node
over all sections) and lays it out. Saving and running go through the same
functions as the Workflows page (api/workflows.py), with the same checks
and the same scheduler; a run takes the workflow's sandboxes, not the
chat's.

Agents are built before the workflow routes exist, so the routes are
connected here when the app starts (`connect`); without them the tools say
workflows are not available.
"""

import json
from typing import Any

from fastapi import HTTPException
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

_api: Any = None
"""api.workflows.WorkflowsApi, once the app connected it."""

RUN_WAIT_SECONDS = 300
ROW, SECTION_GAP, COLUMN = 130, 50, 420


def connect(api: Any) -> None:
    """Give the tools the workflow routes (api.main does, when the app starts)."""
    global _api
    _api = api


def connected() -> bool:
    return _api is not None


class Item(BaseModel):
    analysis_id: str = Field(description="A saved analysis's id (from list_saved_analyses or save_analysis)")
    tables: dict[str, str] = Field(default={}, description='Tables to read instead of its own, e.g. '
                                                            '{"events": "events_enterprise"}; same columns needed')
    run_first: bool = Field(default=False, description="Start it before the others, e.g. a slow one")


class Section(BaseModel):
    title: str = Field(description="The section's heading in the report, e.g. 'Users'")
    analyses: list[Item]


def layout(title: str, sections: list[Section], report_title: str | None, sandboxes: int | None) -> dict:
    """The workflow body: an analysis node per analysis (once, even in two sections), a batch per section,
    one report over all sections, placed in three columns."""
    nodes, edges, ids = [], [], {}
    row = 0
    batch_rows = []
    for s_index, section in enumerate(sections):
        first = row
        batch_id = f"b{s_index + 1}"
        for item in section.analyses:
            key = (item.analysis_id, tuple(sorted(item.tables.items())))
            if key not in ids:
                ids[key] = f"a{len(ids) + 1}"
                nodes.append({"kind": "analysis", "id": ids[key], "analysis_id": item.analysis_id,
                              "sources": item.tables, "priority": 1 if item.run_first else 0,
                              "position": {"x": 0, "y": row * ROW + s_index * SECTION_GAP}})
                row += 1
            edge = {"source": ids[key], "target": batch_id}
            if edge not in edges:
                edges.append(edge)
        middle = (first + max(row - 1, first)) / 2
        batch_rows.append(middle * ROW + s_index * SECTION_GAP)
        nodes.append({"kind": "batch", "id": batch_id, "title": section.title,
                      "position": {"x": COLUMN, "y": batch_rows[-1]}})
    nodes.append({"kind": "report", "id": "r1", "title": report_title or title,
                  "position": {"x": 2 * COLUMN, "y": sum(batch_rows) / max(len(batch_rows), 1)}})
    edges += [{"source": f"b{k + 1}", "target": "r1"} for k in range(len(sections))]
    return {"title": title, "nodes": nodes, "edges": edges, "settings": {"lanes": sandboxes}}


def _summary(run: dict, workflow_id: str) -> str:
    """A finished (or still going) run in a few lines, first the one the chat recognises."""
    nodes = run["nodes"]
    done = sum(1 for n in nodes if n["status"] == "done")
    sandboxes = len({n.get("sandbox") or n["node_id"] for n in nodes if n.get("sandbox_at")})
    took = f"in {run['seconds']:.0f} s" if run.get("seconds") is not None else "still running"
    lines = [f"Ran workflow {workflow_id}: {run['status'].replace('_', ' ')} {took} "
             f"({done} of {len(nodes)} analyses done, {sandboxes} sandboxes)."]
    for n in nodes:
        if n["status"] == "needs_optimization":
            lines.append(f"- {n['title']} needs optimization: {n['error']}. The user can optimize it on its "
                         f"page (/analyses/{n['analysis_id']}?run={n['run_id']}).")
        elif n["status"] in ("failed", "waiting", "running"):
            lines.append(f"- {n['title']}: {n['status']}{': ' + n['error'] if n.get('error') else ''}.")
    lines.append(f"Workflow: /workflows/{workflow_id}")
    for report in run.get("reports", {}):
        lines.append(f"Report: /api/workflows/{workflow_id}/runs/{run['id']}/reports/{report}")
    return "\n".join(lines)


def make_workflow_tools() -> list[BaseTool]:
    def unavailable() -> str | None:
        return None if _api is not None else "ERROR: workflows are not available here."

    @tool
    async def list_workflows() -> str:
        """List the workflows users made, newest first: id, title, sections with their analyses, and the last
        run. Use it when the user refers to a workflow or report ("my weekly report"), before changing it."""
        if (why := unavailable()) is not None:
            return why
        found = []
        for summary in await _api.list():
            workflow = await _api.get(summary["id"])
            titles = {aid: (a or {}).get("title", aid) for aid, a in workflow["analyses"].items()}
            node = {n["id"]: n for n in workflow["nodes"]}
            sections = [{"title": b["title"], "analyses": [
                {"analysis_id": node[e["source"]]["analysis_id"], "title": titles.get(node[e["source"]]["analysis_id"]),
                 "tables": node[e["source"]].get("sources", {})}
                for e in workflow["edges"] if e["target"] == b["id"]]}
                for b in workflow["nodes"] if b["kind"] == "batch"]
            last = summary.get("last_run")
            found.append({"id": summary["id"], "title": summary["title"], "sections": sections,
                          "last_run": {"status": last["status"], "seconds": last["seconds"]} if last else None})
        return json.dumps(found, ensure_ascii=False) if found else "No workflows yet."

    @tool
    async def save_workflow(title: str, sections: list[Section], report_title: str | None = None,
                            sandboxes: int | None = None, replaces: str | None = None) -> str:
        """Save a workflow: saved analyses grouped into sections, all on one report page, that run together.

        Each analysis must be saved first (save_analysis); reuse a saved one (list_saved_analyses) when it
        already answers that part. A section is a heading in the report: group related charts. An analysis may
        read other tables with the same columns (`tables`). `sandboxes`: leave it out (the default keeps one
        for the chat). `replaces`: a workflow's id, to change it instead of making a new one.

        Saving does not run it: run_workflow does. The user sees it on the Workflows page."""
        if (why := unavailable()) is not None:
            return why
        from api.workflows import WorkflowBody  # here: the api package imports this one
        if not sections or not any(s.analyses for s in sections):
            return "NOT SAVED: give at least one section with at least one analysis."
        body = WorkflowBody.model_validate(layout(title, sections, report_title, sandboxes))
        try:
            saved = await (_api.replace(replaces, body) if replaces else _api.create(body))
        except HTTPException as e:
            return f"NOT SAVED: {e.detail}. Fix it and call save_workflow again."
        count = sum(1 for n in saved["nodes"] if n["kind"] == "analysis")
        verb = "Updated" if replaces else "Saved"
        return (f"{verb} workflow {saved['id']}: \"{saved['title']}\": {count} analyses in {len(sections)} "
                f"sections, one report. The user can open it at /workflows/{saved['id']}. "
                f"Run it with run_workflow when the user wants the report.")

    @tool
    async def run_workflow(workflow_id: str) -> str:
        """Run a saved workflow and wait for it (usually under a minute): every analysis on its tables, in the
        workflow's own sandboxes. Returns how long it took, what finished, what needs attention, and the links
        to the workflow and its report. Tell the user honestly what did not finish."""
        if (why := unavailable()) is not None:
            return why
        try:
            started = await _api.start(workflow_id.strip())
        except HTTPException as e:
            return f"NOT RUN: {e.detail}."
        run = await _api.wait(workflow_id.strip(), started["id"], RUN_WAIT_SECONDS)
        return _summary(run, workflow_id.strip())

    return [list_workflows, save_workflow, run_workflow]


PROMPT = """
Workflows: when the user wants several charts together (a report, a dashboard, "every week"), build a
workflow. Each chart must be a saved analysis: make and check it as usual, then save_analysis (or reuse
one from list_saved_analyses that already answers that part). Then save_workflow with sections that group
related charts, and run_workflow if the user asked for the report itself. Say what run_workflow reports,
including analyses that failed or need optimization (the user optimizes those on the analysis's page).
To change a workflow, find it with list_workflows and call save_workflow with `replaces`.
"""
