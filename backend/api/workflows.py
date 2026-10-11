"""Workflows: saved analyses run in parallel, each feeding a batch node that collects their results.

    GET    /api/workflows                           every workflow, newest first, with its last run
    POST   /api/workflows                           create one: {"title", "nodes", "edges"} (201)
    GET    /api/workflows/{id}                      the workflow, its runs, and whether one is running
    PUT    /api/workflows/{id}                      replace its title, nodes and edges
    DELETE /api/workflows/{id}                      delete it and its runs (the analysis runs stay)
    POST   /api/workflows/{id}/runs                 run it, in the background (202)
    GET    /api/workflows/{id}/runs/{run_id}        one run: each node's status and outputs
    GET    /api/workflows/{id}/runs/{run_id}/reports/{node}   a report node's page for that run: the results (HTML)
    GET    /api/workflows/{id}/runs/{run_id}/load                 that run's load: every batch and analysis over time (HTML)

A run starts every analysis node connected to a batch, each as an ordinary
run of that analysis (trigger "workflow", on the node's tables), so each takes
its own sandbox. At most `sandbox_limit` (what one account may hold) run at
once; the rest wait in the workflow, "waiting", and start as others end. The
sandbox manager would only wait 30 seconds for a free sandbox, made for a
chat; a workflow of many analyses needs longer. The run ends when every
node's run has ended. A node that stops with
"needs_optimization" is only marked: the user optimizes it on its analysis's
page, as for any run. The outputs stay with each analysis run; a report
node's page (workflows/report.py) is built from them when it is asked for.
"""

import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from analyses import AnalysisStore
from analyses.models import now, now_ms
from analyses.sources import QueryRows, check_mapping
from analyses.store import new_id, new_run_id
from workflows import Workflow, WorkflowRun, WorkflowStore
from workflows.models import Edge, Node, NodeRun
from workflows.report import render, render_load

from .analyses import AnalysesApi
from .app import ARTIFACT_HEADERS


class WorkflowBody(BaseModel):
    title: str
    nodes: list[Node] = []
    edges: list[Edge] = []


def _seconds(start: str, end: str) -> float:
    return (datetime.fromisoformat(end.replace("Z", "+00:00")) -
            datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()


def workflows_router(store: WorkflowStore, analyses: AnalysisStore, api: AnalysesApi, *,
                     query_rows: QueryRows, sandbox_limit: int | None = None) -> APIRouter:
    """`api`: starts analysis runs as the Run button does; `query_rows`: to check each node's table swaps;
    `sandbox_limit`: how many sandboxes an account may hold at once, shown on the report's load chart."""
    router = APIRouter(prefix="/api/workflows")
    running: dict[str, asyncio.Task] = {}
    """workflow id -> its run in progress."""

    def get(workflow_id: str) -> Workflow:
        workflow = store.get(workflow_id)
        if workflow is None:
            raise HTTPException(404, f"no workflow '{workflow_id}'")
        return workflow

    async def problems(workflow: Workflow) -> list[str]:
        found = workflow.problems()
        for node in workflow.nodes:
            if node.kind != "analysis":
                continue
            analysis = analyses.get(node.analysis_id)
            if analysis is None:
                found.append(f"{node.id}: there is no saved analysis '{node.analysis_id}'")
                continue
            swaps = {old: new for old, new in node.sources.items() if old != new}
            if swaps:
                found += [f"{analysis.title}: {p}"
                          for p in await check_mapping(analysis.source_tables(), swaps, query_rows)]
        return found

    def refresh(run: WorkflowRun) -> WorkflowRun:
        """Each node's status and outputs, from its analysis run."""
        for node in run.nodes:
            record = analyses.get_run(node.analysis_id, node.run_id) if node.run_id else None
            if record is not None:
                node.status, node.error, node.seconds = record.status, record.error, record.seconds
                node.outputs = list(record.outputs)
                # Queued when the workflow started, though its analysis run began when a place freed up.
                node.queued_at = node.queued_at or record.queued_at
                node.sandbox_at, node.finished_at = record.sandbox_at, record.finished_at
                measured = record.measurements
                moved = [i for i in measured.inputs if i.rows is not None] if measured else []
                node.rows = sum(i.rows for i in moved) if moved else None
                node.bytes = sum(i.bytes or 0 for i in moved) if moved else None
                sandbox = measured.sandbox if measured else None
                if sandbox is not None:
                    node.cpu_seconds, node.peak_memory_mb = sandbox.cpu_seconds, sandbox.peak_memory_mb
                    node.memory_limit_mb = sandbox.memory_limit_mb
                    node.script_at, node.script_seconds, node.samples = sandbox.started_at, sandbox.seconds, sandbox.samples
        run.settle()
        return run

    async def finish(run: WorkflowRun, tasks: list[asyncio.Task]) -> None:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            refresh(run)
            # A run's task ends after its record is saved, so nothing is still running here;
            # a record that says so lost its task (an error): it did not finish.
            for node in run.nodes:
                if node.status in ("running", "waiting"):
                    node.status, node.error = "failed", node.error or "the run ended without a result"
            run.settle()
            run.finished_at = now()
            run.seconds = _seconds(run.started_at, run.finished_at)
            store.save_run(run)
            running.pop(run.workflow_id, None)

    def summary(workflow: Workflow) -> dict:
        runs = store.runs(workflow.id)
        return {
            "id": workflow.id, "title": workflow.title, "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
            "analyses": sum(1 for n in workflow.nodes if n.kind == "analysis"),
            "batches": sum(1 for n in workflow.nodes if n.kind == "batch"),
            "reports": sum(1 for n in workflow.nodes if n.kind == "report"),
            "last_run": refresh(runs[0]).model_dump() if runs else None,
        }

    def detail(workflow: Workflow) -> dict:
        titles = {}
        for node in workflow.nodes:
            if node.kind == "analysis":
                analysis = analyses.get(node.analysis_id)
                titles[node.analysis_id] = (
                    {"title": analysis.title, "version": analysis.version, "tables": analysis.source_tables()}
                    if analysis is not None else None)
        return {**workflow.model_dump(), "analyses": titles, "running": workflow.id in running,
                "runs": [refresh(r).model_dump() for r in store.runs(workflow.id)]}

    @router.get("")
    async def list_workflows():
        return [summary(w) for w in store.all()]

    @router.post("", status_code=201)
    async def create(body: WorkflowBody):
        workflow = Workflow(id=new_id(), title=body.title.strip(), nodes=body.nodes, edges=body.edges)
        found = await problems(workflow)
        if found:
            raise HTTPException(400, "; ".join(found))
        store.save(workflow)
        return detail(workflow)

    @router.get("/{workflow_id}")
    async def get_workflow(workflow_id: str):
        return detail(get(workflow_id))

    @router.put("/{workflow_id}")
    async def replace(workflow_id: str, body: WorkflowBody):
        old = get(workflow_id)
        if workflow_id in running:
            raise HTTPException(409, "this workflow is running; change it when the run ends")
        workflow = Workflow(id=old.id, title=body.title.strip(), nodes=body.nodes, edges=body.edges,
                            created_at=old.created_at, updated_at=now())
        found = await problems(workflow)
        if found:
            raise HTTPException(400, "; ".join(found))
        store.save(workflow)
        return detail(workflow)

    @router.delete("/{workflow_id}")
    async def delete(workflow_id: str):
        if workflow_id in running:
            raise HTTPException(409, "this workflow is running; delete it when the run ends")
        if not store.delete(workflow_id):
            raise HTTPException(404, f"no workflow '{workflow_id}'")
        return {"deleted": workflow_id}

    @router.post("/{workflow_id}/runs", status_code=202)
    async def start_run(workflow_id: str):
        workflow = get(workflow_id)
        if workflow_id in running:
            raise HTTPException(409, "this workflow is already running")
        nodes = workflow.connected()
        if not nodes:
            raise HTTPException(400, "connect at least one analysis to a batch")
        found = await problems(workflow)
        if found:
            raise HTTPException(400, "can't run it: " + "; ".join(found))
        targets = [(node, analyses.get(node.analysis_id)) for node in nodes]
        busy = sorted({a.title for _, a in targets if api.optimizing(a.id)})
        if busy:
            raise HTTPException(409, f"{', '.join(busy)} is being optimized; run the workflow when that ends")

        run = WorkflowRun(id=new_run_id(), workflow_id=workflow.id, started_at=now(), batches=workflow.batches(),
                          reports=workflow.reports(), workflow_title=workflow.title, sandbox_limit=sandbox_limit,
                          titles={n.id: n.title for n in workflow.nodes if n.kind != "analysis"})
        places = asyncio.Semaphore(sandbox_limit) if sandbox_limit else None

        async def go(node: NodeRun, analysis, swaps: dict[str, str]) -> None:
            async def launch() -> None:
                record, task = api.start(analysis, swaps, run.id)
                node.run_id, node.status = record.id, "running"
                store.save_run(run)
                await task
            if places is None:
                return await launch()
            async with places:
                await launch()

        tasks = []
        for node, analysis in targets:
            swaps = {old: new for old, new in node.sources.items() if old != new}
            entry = NodeRun(node_id=node.id, analysis_id=analysis.id, title=analysis.title, sources=swaps,
                            version=analysis.version, status="waiting", queued_at=now_ms())
            run.nodes.append(entry)
            tasks.append((entry, analysis, swaps))
        store.save_run(run)
        started = [asyncio.create_task(go(*t)) for t in tasks]
        running[workflow_id] = asyncio.create_task(finish(run, started))
        return run.model_dump()

    @router.get("/{workflow_id}/runs/{run_id}")
    async def get_run(workflow_id: str, run_id: str):
        get(workflow_id)
        run = store.get_run(workflow_id, run_id)
        if run is None:
            raise HTTPException(404, f"no run '{run_id}'")
        return refresh(run).model_dump()

    def read_output(analysis_id: str, run_id: str, name: str) -> bytes | None:
        path = analyses.output_path(analysis_id, run_id, name)
        return path.read_bytes() if path is not None else None

    @router.get("/{workflow_id}/runs/{run_id}/reports/{node_id}")
    async def report(workflow_id: str, run_id: str, node_id: str):
        get(workflow_id)
        run = store.get_run(workflow_id, run_id)
        if run is None or node_id not in run.reports:
            raise HTTPException(404, f"no report '{node_id}' in run '{run_id}'")
        # It embeds pages the agent's scripts wrote: served like them, as an untrusted sandboxed page.
        return HTMLResponse(render(refresh(run), node_id, read_output), headers=ARTIFACT_HEADERS)

    @router.get("/{workflow_id}/runs/{run_id}/load")
    async def load(workflow_id: str, run_id: str):
        get(workflow_id)
        run = store.get_run(workflow_id, run_id)
        if run is None:
            raise HTTPException(404, f"no run '{run_id}'")
        # Only our own markup and numbers, but sandboxed like the report next to it.
        return HTMLResponse(render_load(refresh(run)), headers=ARTIFACT_HEADERS)

    return router
