"""Workflows: saved analyses run in parallel, each feeding a batch node that collects their results.

    GET    /api/workflows                           every workflow, newest first, with its last run
    POST   /api/workflows                           create one: {"title", "nodes", "edges"} (201)
    GET    /api/workflows/{id}                      the workflow, its runs, and whether one is running
    PUT    /api/workflows/{id}                      replace its title, nodes and edges
    DELETE /api/workflows/{id}                      delete it and its runs (the analysis runs stay)
    POST   /api/workflows/{id}/runs                 run it, in the background (202)
    GET    /api/workflows/{id}/runs/{run_id}        one run: each node's status and outputs
    GET    /api/workflows/{id}/runs/{run_id}/reports/{node}   a report node's page for that run: the results (HTML)
    POST   /api/workflows/{id}/advice                      ask the advisor how to rearrange it: {"goal": "fastest" |
                                                           "cheapest"} -> its suggestions, checked (201)
    GET    /api/workflows/{id}/advice                      every time it was asked, newest first
    POST   /api/workflows/{id}/advice/{advice}/suggestions/{n}/apply     apply one (set_lanes, run_first, optimize)
    POST   /api/workflows/{id}/advice/{advice}/suggestions/{n}/dismiss   set one aside
    GET    /api/workflows/{id}/runs/{run_id}/performance          that run's performance: time and resources of every
                                                                   batch and analysis (HTML); /load is the old address

A run starts every analysis node connected to a batch, each as an ordinary
run of that analysis (trigger "workflow", on the node's tables). How:

    1. The size check of every node first (database only): a node it stops is
       recorded as "needs_optimization" at once and never waits for a sandbox.
       The lanes' sandboxes start meanwhile.
    2. The rest queue by priority (a node's "run first"), then longest first,
       by their analysis's past runs on the same tables (one never run counts
       as longest: it goes first).
    3. `lanes` sandboxes (the workflow's own setting, or all but one an account
       may hold, so chat keeps one; never more than an account may hold)
       take the queue: each runs analyses one after another, cleaning up in
       between (api/analyses.py, Lane), and is given back when the queue is
       empty. Renting a sandbox takes 10-15 seconds and a script about one:
       passing sandboxes on is most of the time a workflow saves.
    4. While a lane runs one analysis, the inputs of the next ones (as many
       as there are lanes) are fetched from the database, so a lane does not
       wait for a query between analyses.

Queued nodes wait in the workflow, "waiting", for as long as it takes (the
sandbox manager would wait only 30 seconds, made for a chat). The run ends
when every node's run has ended. A node that stops with
"needs_optimization" is only marked: the user optimizes it on its analysis's
page, as for any run. The outputs stay with each analysis run; a report
node's page (workflows/report.py) is built from them when it is asked for.
"""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from analyses import AnalysisStore
from analyses.models import now, now_ms
from analyses.sources import QueryRows, check_mapping
from analyses.store import new_id, new_run_id
from workflows import Workflow, WorkflowRun, WorkflowStore
from workflows.advisor import RUNS, Advise, AdviceRecord, Goal, check, summarize
from workflows.models import Edge, Node, NodeRun, WorkflowSettings
from workflows.report import render, render_load

from .analyses import AnalysesApi, Lane
from .app import ARTIFACT_HEADERS

log = logging.getLogger(__name__)


class AdviceRequest(BaseModel):
    goal: Goal = "fastest"


class WorkflowBody(BaseModel):
    title: str
    nodes: list[Node] = []
    edges: list[Edge] = []
    settings: WorkflowSettings = WorkflowSettings()


def _seconds(start: str, end: str) -> float:
    return (datetime.fromisoformat(end.replace("Z", "+00:00")) -
            datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()


def workflows_router(store: WorkflowStore, analyses: AnalysisStore, api: AnalysesApi, *,
                     query_rows: QueryRows, sandbox_limit: int | None = None, lanes: int | None = None,
                     advise: Advise | None = None, advisor_name: str = "", settings: dict | None = None) -> APIRouter:
    """`api`: starts analysis runs as the Run button does; `query_rows`: to check each node's table swaps;
    `sandbox_limit`: how many sandboxes an account may hold at once; `lanes`: how many of them a workflow run
    uses (default: all but one, for chat); `advise`: the advisor's model call (workflows/advisor.py), named
    `advisor_name`; `settings`: what the advisor is told about the limits (warm sandboxes, export limits...)."""
    default_lanes = lanes or max(1, (sandbox_limit or 2) - 1)

    def lanes_for(workflow: Workflow) -> int:
        """Its own setting, or the default; at least one, and never more than an account may hold."""
        wanted = workflow.settings.lanes or default_lanes
        return max(1, min(wanted, sandbox_limit) if sandbox_limit else wanted)
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
                node.sandbox, node.reused, node.cleanup_seconds = record.sandbox, record.reused, record.cleanup_seconds
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

    def expected_seconds(analysis_id: str, swaps: dict[str, str]) -> float:
        """How long it took last time on these tables, in its sandbox; infinite if it never ran there."""
        for record in analyses.runs(analysis_id):
            if record.status == "done" and record.sources == swaps and record.seconds is not None:
                return record.seconds
        return float("inf")

    async def schedule(run: WorkflowRun, jobs: list, priorities: dict[str, int]) -> None:
        """Check every job's size, start the stopped ones (they need no sandbox), run the rest in lanes."""
        # Start the sandboxes while the sizes are checked: the check is database work, a start takes 10-15 s.
        lanes = run.lanes or 1
        opened = [Lane(name=f"workflow-{run.id}-{k + 1}") for k in range(min(lanes, len(jobs)))]
        for lane in opened:
            await api.open_lane(lane)

        async def check(job):
            _, analysis, swaps = job
            try:
                return await api.precheck(analysis, swaps)
            except Exception:
                log.warning("size check of %s failed; its run will say why", analysis.id, exc_info=True)
                return None, []
        checks = await asyncio.gather(*(check(job) for job in jobs))
        stopped, queue = [], []
        for job, (estimates, stop) in zip(jobs, checks):
            (stopped if stop else queue).append((*job, estimates))

        fetching: dict[str, asyncio.Task] = {}

        def prefetch(job) -> None:
            entry, analysis, swaps, estimates = job
            if entry.node_id not in fetching:
                fetching[entry.node_id] = asyncio.create_task(api.fetch(analysis, swaps, estimates))

        def launch(entry: NodeRun, analysis, swaps, estimates, lane=None) -> asyncio.Task:
            record, task = api.start(analysis, swaps, run.id, lane=lane, estimates=estimates,
                                     inputs=fetching.get(entry.node_id))
            entry.run_id, entry.status = record.id, "running"
            store.save_run(run)
            return task

        tasks = [launch(*job) for job in stopped]
        # "Run first" before the rest; then longest first: short ones fill the gaps at the end.
        # The sort keeps node order among equals.
        queue.sort(key=lambda job: (-priorities.get(job[0].node_id, 0), -expected_seconds(job[1].id, job[2])))

        async def work(lane: Lane) -> None:
            try:
                while queue:
                    job = queue.pop(0)
                    prefetch(job)
                    for ahead in queue[:lanes]:  # the next ones' data, while this one runs
                        prefetch(ahead)
                    await launch(*job, lane=lane)
            finally:
                await api.close_lane(lane)  # nothing left for it: give its sandbox back now

        busy = opened[:len(queue)]
        for spare in opened[len(queue):]:  # more sandboxes started than analyses can run
            await api.close_lane(spare)
        try:
            await asyncio.gather(*tasks, *(work(lane) for lane in busy), return_exceptions=True)
        finally:
            for task in fetching.values():  # one fetched for a run that never came (an error)
                task.cancel()

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
        workflow = Workflow(id=new_id(), title=body.title.strip(), nodes=body.nodes, edges=body.edges,
                            settings=body.settings)
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
        workflow = Workflow(id=old.id, title=body.title.strip(), nodes=body.nodes, edges=body.edges, settings=body.settings,
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
                          lanes=lanes_for(workflow), titles={n.id: n.title for n in workflow.nodes if n.kind != "analysis"})
        jobs = []
        for node, analysis in targets:
            swaps = {old: new for old, new in node.sources.items() if old != new}
            entry = NodeRun(node_id=node.id, analysis_id=analysis.id, title=analysis.title, sources=swaps,
                            version=analysis.version, status="waiting", queued_at=now_ms())
            run.nodes.append(entry)
            jobs.append((entry, analysis, swaps))
        store.save_run(run)
        priorities = {n.id: n.priority for n in workflow.nodes if n.kind == "analysis"}
        running[workflow_id] = asyncio.create_task(finish(run, [asyncio.create_task(schedule(run, jobs, priorities))]))
        return run.model_dump()

    @router.get("/{workflow_id}/runs/{run_id}")
    async def get_run(workflow_id: str, run_id: str):
        get(workflow_id)
        run = store.get_run(workflow_id, run_id)
        if run is None:
            raise HTTPException(404, f"no run '{run_id}'")
        return refresh(run).model_dump()

    # --- the advisor --------------------------------------------------------------------------

    async def table_rows(names: set[str]) -> dict[str, int]:
        safe = sorted(n for n in names if n.replace("_", "").isalnum())
        if not safe:
            return {}
        listed = ", ".join(f"'{n}'" for n in safe)
        try:
            rows = await query_rows(f"SELECT relname AS name, reltuples::bigint AS rows FROM pg_class "
                                    f"WHERE relname IN ({listed}) AND relkind = 'r'")
        except Exception:
            return {}  # sizes help the advisor; without them it still answers
        return {r["name"]: int(r["rows"]) for r in rows if "name" in r and "rows" in r}

    @router.post("/{workflow_id}/advice", status_code=201)
    async def ask_advice(workflow_id: str, body: AdviceRequest | None = None):
        workflow = get(workflow_id)
        if advise is None:
            raise HTTPException(503, "no advisor is configured")
        goal = body.goal if body else "fastest"
        runs = [refresh(r) for r in store.runs(workflow_id) if r.status != "running"][:RUNS]
        if not runs:
            raise HTTPException(400, "run the workflow first: the advisor works from its runs")
        recipes = {n.analysis_id: a for n in workflow.nodes if n.kind == "analysis"
                   for a in [analyses.get(n.analysis_id)] if a is not None}
        records = {n.run_id: r for run in runs for n in run.nodes if n.run_id
                   for r in [analyses.get_run(n.analysis_id, n.run_id)] if r is not None}
        tables = {t for run in runs for n in run.nodes if n.run_id in records
                  for m in (records[n.run_id].measurements.inputs if records[n.run_id].measurements else [])
                  for t in m.tables}
        told = {**(settings or {}), "sandbox_limit": sandbox_limit, "default_lanes": default_lanes,
                "lanes_now": lanes_for(workflow)}
        summary = summarize(workflow, runs, recipes, records, told, await table_rows(tables), goal)
        try:
            advice = await advise(summary, goal)
        except Exception as e:
            log.warning("the advisor failed for workflow %s", workflow_id, exc_info=True)
            raise HTTPException(502, f"the advisor could not answer: {e}") from None
        # Optimize can start only from a run that broke a limit: the node's newest one.
        findings = {}
        for run in runs:
            for n in run.nodes:
                if n.node_id not in findings and n.run_id in records:
                    findings[n.node_id] = bool(records[n.run_id].findings)
        record = AdviceRecord(id=new_run_id(), workflow_id=workflow_id, goal=goal, runs=[r.id for r in runs],
                              model=advisor_name, summary=advice.summary,
                              suggestions=check(advice, workflow, sandbox_limit, findings))
        store.save_advice(record)
        return record.model_dump()

    @router.get("/{workflow_id}/advice")
    async def list_advice(workflow_id: str):
        get(workflow_id)
        return [a.model_dump() for a in store.advice(workflow_id)]

    def suggestion(workflow_id: str, advice_id: str, index: int):
        advice = store.get_advice(workflow_id, advice_id)
        if advice is None or not 0 <= index < len(advice.suggestions):
            raise HTTPException(404, "no such suggestion")
        return advice, advice.suggestions[index]

    @router.post("/{workflow_id}/advice/{advice_id}/suggestions/{index}/apply")
    async def apply_suggestion(workflow_id: str, advice_id: str, index: int):
        workflow = get(workflow_id)
        advice, s = suggestion(workflow_id, advice_id, index)
        if not s.applicable:
            raise HTTPException(400, f"this suggestion is advice: {s.note}")
        if s.status == "applied":
            raise HTTPException(409, "it was applied already")
        if workflow_id in running:
            raise HTTPException(409, "this workflow is running; apply it when the run ends")
        if s.kind == "set_lanes":
            workflow.settings.lanes = s.value
            s.result = f"Runs of this workflow now use at most {s.value} sandboxes."
        elif s.kind == "run_first":
            top = max((n.priority for n in workflow.nodes if n.kind == "analysis"), default=0)
            moved = [n for n in workflow.nodes if n.kind == "analysis" and n.id in s.node_ids]
            for n in moved:
                n.priority = top + 1
            s.result = f"{', '.join(n.id for n in moved)} now run first."
        elif s.kind == "optimize_analysis":
            node = workflow.node(s.node_ids[0])
            latest = next((n for run in store.runs(workflow_id) for n in run.nodes
                           if n.node_id == node.id and n.run_id), None)
            if latest is None:
                raise HTTPException(409, "that analysis has no run in this workflow to optimize from")
            try:
                await api.optimize(node.analysis_id, latest.run_id)
            except HTTPException as e:
                s.status, s.result = "failed", f"Optimize could not start: {e.detail}"
                store.save_advice(advice)
                raise
            s.result = (f"Optimize started for {latest.title}. Follow it on the analysis's page; the next run of "
                        f"this workflow uses the new version.")
        if s.kind in ("set_lanes", "run_first"):
            workflow.updated_at = now()
            store.save(workflow)
        s.status = "applied"
        store.save_advice(advice)
        return {"advice": advice.model_dump(), "workflow": detail(workflow)}

    @router.post("/{workflow_id}/advice/{advice_id}/suggestions/{index}/dismiss")
    async def dismiss_suggestion(workflow_id: str, advice_id: str, index: int):
        get(workflow_id)
        advice, s = suggestion(workflow_id, advice_id, index)
        if s.status == "applied":
            raise HTTPException(409, "it was applied already")
        s.status = "dismissed"
        store.save_advice(advice)
        return advice.model_dump()

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

    @router.get("/{workflow_id}/runs/{run_id}/performance")
    @router.get("/{workflow_id}/runs/{run_id}/load")  # the old address, kept for links made before
    async def load(workflow_id: str, run_id: str):
        get(workflow_id)
        run = store.get_run(workflow_id, run_id)
        if run is None:
            raise HTTPException(404, f"no run '{run_id}'")
        # Only our own markup and numbers, but sandboxed like the report next to it.
        return HTMLResponse(render_load(refresh(run)), headers=ARTIFACT_HEADERS)

    return router
