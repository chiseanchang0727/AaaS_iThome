"""Saved analyses: list them, run one again, optimize one that no longer fits, serve what a run made.

    GET    /api/analyses                                     every saved analysis, newest first, with its last run
    GET    /api/analyses/{id}                                the recipe, the tables it reads, and its runs
    GET    /api/analyses/{id}/versions                       every version's recipe, oldest first (the last is current)
    GET    /api/analyses/{id}/sources                        for each table it reads: the tables that could stand in
    POST   /api/analyses/{id}/runs                           run it again, in the background (202); body
                                                             {"sources": {"videos": "videos_ca"}} reads other tables
    GET    /api/analyses/{id}/runs/{run_id}                  one run: status, measurements, findings, output files
    POST   /api/analyses/{id}/runs/{run_id}/optimize         ask the agent for a version that fits (202)
    GET    /api/analyses/{id}/optimizations/{conversation}   what the agent was asked and did in one optimization
    GET    /api/analyses/{id}/runs/{run_id}/files/{name}     a file that run made
    DELETE /api/analyses/{id}                                delete it and its runs

A run first asks the database for an estimate of each query input; if that
breaks a hard limit (analyses/budget.py), it stops there with status
"needs_optimization", before any rows move or a sandbox starts. Otherwise it
takes a sandbox like a conversation does, runs the recipe measured
(analyses/runner.py), judges the cost, and gives the sandbox back. No model is
involved. A version already rewritten for memory that still runs out of memory
moves to the bigger sandbox once and tries again.

Optimize is the user's choice, after a run that broke a limit: the agent gets
the request (analyses/optimize.py) in a conversation of its own, saves a new
version through save_analysis (which checks it more strictly while the
analysis is marked as optimizing), and the new version runs on the same tables.
That conversation is part of the analysis's history, not the user's chats:
it is kept in <analysis>/optimizations/<conversation>.jsonl.
"""

import asyncio
import dataclasses
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from analyses import Analysis, AnalysisStore, RunRecord
from analyses.budget import Limits, check_estimates, main_finding
from analyses.models import Finding, InputMeasure, Measurements, now, now_ms
from analyses.optimize import request as optimization_request
from analyses.runner import (Count, Estimate, Fetched, Query, ReadDataset, RunResult, clean_first, estimate_inputs,
                             fetch_inputs, run)
from analyses.sources import QueryRows, check_mapping, source_options
from analyses.store import new_run_id

from .app import ARTIFACT_HEADERS
from .conversations import ConversationManager
from .events import run_turn
from .history import HistoryStore
from .vendor import local_plotly

log = logging.getLogger(__name__)

RUN_FOLDER = ".analysis-run"
"""Where runs keep their scratch folders (one per run), inside the sandbox's work folder."""

AskAgent = Callable[[str, str, HistoryStore], Awaitable[str]]
"""(conversation id, message, where to log it) -> the agent's final answer."""

_OPTIMIZATION = re.compile(r"^optimize-[A-Za-z0-9_-]{1,60}$")


@dataclasses.dataclass
class Lane:
    """A sandbox a workflow run keeps for several of its analyses, one after another.

    Renting a sandbox takes 10-15 seconds and a saved analysis's script about one, so a
    workflow passes its sandboxes on instead of renting one per analysis. Before the next
    analysis, the last one's processes are stopped and its folder deleted. A sandbox that
    ran out of memory, or does not clean up, is closed; the lane rents a new one.
    """

    name: str
    generation: int = 1
    """Which sandbox of the lane: a new one after a retired one."""
    runs: int = 0
    """Analyses run in the current sandbox."""

    @property
    def conversation_id(self) -> str:
        return f"{self.name}-{self.generation}"


StartRun = Callable[..., tuple[RunRecord, asyncio.Task]]
"""(analysis, table swaps, workflow run id, lane=None, estimates=None) -> its run record and the task that runs it."""
Precheck = Callable[[Analysis, dict[str, str]], Awaitable[tuple[list[InputMeasure], list[Finding]]]]
"""(analysis, table swaps) -> its estimates, and the hard limits they break (empty: it may run)."""


@dataclasses.dataclass
class AnalysesApi:
    router: APIRouter
    start: StartRun
    """Run a saved analysis in the background, as the Run button does; for workflows."""
    optimizing: Callable[[str], bool]
    """Whether an analysis is being optimized now (it may get a new version any moment)."""
    precheck: Precheck
    """The size check a run starts with, on its own: a workflow checks every analysis before queueing."""
    close_lane: Callable[[Lane], Awaitable[None]]
    """Give a lane's sandbox back."""
    open_lane: Callable[[Lane], Awaitable[None]]
    """Start a lane's sandbox now, before its first analysis needs it."""
    fetch: Callable[[Analysis, dict[str, str], list[InputMeasure] | None], Awaitable[Fetched]]
    """Fetch an analysis's inputs ahead of its run: (analysis, table swaps, estimates)."""


class RunRequest(BaseModel):
    sources: dict[str, str] = {}
    """Tables to read instead of the recipe's own, e.g. {"videos": "videos_ca"}."""


def analyses_router(*args, **kwargs) -> APIRouter:
    return analyses_api(*args, **kwargs).router


def analyses_api(
    store: AnalysisStore,
    manager: ConversationManager,
    *,
    query: Query,
    query_rows: QueryRows,
    read_dataset: ReadDataset,
    work_dir: PurePosixPath,
    account: str,
    estimate: Estimate | None = None,
    count: Count | None = None,
    limits: Limits | None = None,
    ask_agent: AskAgent | None = None,
) -> AnalysesApi:
    """`query`: SQL -> Parquet bytes for an input; `query_rows`: SQL -> rows, to look up tables' columns;
    `estimate` and `limits`: measure and judge runs; `ask_agent`: how optimization talks to the agent
    (default: a conversation through `manager`, logged with the analysis)."""
    router = APIRouter(prefix="/api/analyses")
    running: dict[str, dict[str, asyncio.Task]] = {}
    """analysis id -> its runs and optimization in progress, by run id ("optimize" for an optimization).
    A workflow may run one analysis on several tables at once."""

    def track(analysis_id: str, key: str, task: asyncio.Task) -> None:
        running.setdefault(analysis_id, {})[key] = task

    def untrack(analysis_id: str, key: str) -> None:
        tasks = running.get(analysis_id, {})
        tasks.pop(key, None)
        if not tasks:
            running.pop(analysis_id, None)

    def get(analysis_id: str) -> Analysis:
        analysis = store.get(analysis_id)
        if analysis is None:
            raise HTTPException(404, f"no saved analysis '{analysis_id}'")
        return analysis

    async def run_once(sandbox, analysis: Analysis, record: RunRecord, estimates: list[InputMeasure],
                       fetched: Fetched, prepare: str = "") -> RunResult:
        # The folder is left: the sandbox is closed right after, or cleaned before its next run.
        return await run(sandbox, analysis, query=query, read_dataset=read_dataset, scratch=work_dir / RUN_FOLDER / record.id,
                         sources=record.sources, limits=limits, estimates=estimates, fetched=fetched,
                         prepare=prepare, clean_up=False)

    async def precheck(analysis: Analysis, swaps: dict[str, str]) -> tuple[list[InputMeasure], list[Finding]]:
        estimates = await estimate_inputs(analysis, estimate, swaps, count, limits) if estimate is not None else []
        stop = check_estimates(Measurements(inputs=estimates), limits) if limits is not None else []
        return estimates, stop

    async def sandbox_for(conversation_id: str):
        """The conversation's sandbox, started if need be: (conversation, None), or (None, why not)."""
        conversation = await manager.get_or_create(conversation_id, account)
        # Why a start failed is a sandbox event; keep it, so the run can say so.
        unavailable: list[str] = []
        conversation.on_event = lambda event, fields: unavailable.append(str(fields.get("reason", ""))) \
            if event == "sandbox_unavailable" else None
        if conversation.starting is not None:
            await conversation.starting
        if conversation.sandbox is None:
            await manager.close(conversation_id)
            why = f": {unavailable[-1]}" if unavailable and unavailable[-1] else " (all may be busy)"
            return None, f"no sandbox could be started{why}; try again"
        return conversation, None

    async def retire(lane: Lane) -> None:
        """Close the lane's sandbox; its next analysis gets a new one."""
        await manager.close(lane.conversation_id)
        lane.generation, lane.runs = lane.generation + 1, 0

    async def close_lane(lane: Lane) -> None:
        await manager.close(lane.conversation_id)

    async def open_lane(lane: Lane) -> None:
        await manager.get_or_create(lane.conversation_id, account)  # its sandbox starts in the background

    async def fetch(analysis: Analysis, swaps: dict[str, str], estimates: list[InputMeasure] | None) -> Fetched:
        return await fetch_inputs(analysis, query, read_dataset, swaps, estimates)

    async def execute(analysis: Analysis, record: RunRecord, lane: Lane | None = None,
                      estimates: list[InputMeasure] | None = None, inputs: asyncio.Task | None = None) -> None:
        """Run it. With a `lane`, in the lane's sandbox, cleaned first if an earlier run used it, and left
        open for the next; without, in a sandbox of its own, closed after. `estimates`: its size check,
        if a workflow already made it; `inputs`: its inputs, being fetched already."""
        conversation_id = lane.conversation_id if lane is not None else f"analysis-{record.id}"
        result = RunResult(ok=False, error="the run did not start")
        took_sandbox = False
        fetching: asyncio.Task | None = inputs
        try:
            # Measure first: if the estimate already breaks a hard limit, no rows move and no sandbox starts.
            if estimates is None:
                estimates, stop = await precheck(analysis, record.sources)
            else:
                stop = check_estimates(Measurements(inputs=estimates), limits) if limits is not None else []
            if stop:
                result = RunResult(ok=False, error=stop[0].reason, findings=stop,
                                   measurements=Measurements(inputs=estimates))
                return
            took_sandbox = True
            # The data comes from the database, not the sandbox: fetch it while the sandbox starts.
            if fetching is None:
                fetching = asyncio.create_task(fetch_inputs(analysis, query, read_dataset, record.sources, estimates))
            conversation, why = await sandbox_for(conversation_id)
            if conversation is None:
                result = RunResult(ok=False, error=why)
                return
            reused = lane is not None and lane.runs > 0
            record.sandbox, record.sandbox_at, record.reused = conversation_id, now_ms(), reused
            store.save_run(analysis.id, record)
            # Ready, but the data may still be coming from the database: that wait counts as moving data.
            fetched = await fetching
            # A reused sandbox is cleaned in the same call that makes this run's folder.
            prepare = clean_first(work_dir / RUN_FOLDER) if reused else ""
            result = await run_once(conversation.sandbox, analysis, record, estimates, fetched, prepare)
            if reused and not result.ok and (result.error or "").startswith("could not make the scratch folder"):
                await retire(lane)  # it would not clean up: a new sandbox, once
                conversation_id = lane.conversation_id
                conversation, why = await sandbox_for(conversation_id)
                if conversation is None:
                    result = RunResult(ok=False, error=why)
                    return
                record.sandbox, record.sandbox_at, record.reused = conversation_id, now_ms(), False
                result = await run_once(conversation.sandbox, analysis, record, estimates, fetched)
            if lane is not None:
                lane.runs += 1
            rewritten_for_memory = analysis.optimization is not None and analysis.optimization.kind == "sandbox_memory"
            upgraded = False
            if result.killed and rewritten_for_memory:
                # The script was already rewritten to use less memory: now a bigger sandbox is the answer.
                upgrade = await manager.upgrade(conversation, "a saved analysis ran out of memory")
                record.notes.append(upgrade.note)
                if upgrade.sandbox is not None:
                    upgraded = True
                    result = await run_once(upgrade.sandbox, analysis, record, estimates, fetched)
            if lane is not None and result.killed and not upgraded:
                await retire(lane)  # what ran out of memory may have left the sandbox unwell: not passed on
        except Exception as e:
            log.exception("running saved analysis %s failed", analysis.id)
            result = RunResult(ok=False, error=str(e))
            if lane is not None and took_sandbox:
                await retire(lane)
        finally:
            if fetching is not None and not fetching.done():
                fetching.cancel()
            if took_sandbox and lane is None:
                await manager.close(conversation_id)
            record.finished_at = now_ms()
            record.status = "done" if result.ok else "needs_optimization" if result.needs_optimization else "failed"
            record.seconds, record.error, record.log = result.seconds, result.error, result.log
            record.outputs = list(result.outputs)
            record.measurements, record.findings = result.measurements, result.findings
            store.save_run(analysis.id, record, result.outputs)
            untrack(analysis.id, record.id)

    def transcripts(analysis_id: str) -> HistoryStore:
        """Where an analysis's optimization conversations are kept: with the analysis, not the chats."""
        return HistoryStore(store.root / analysis_id / "optimizations")

    async def talk_to_agent(conversation_id: str, message: str, history: HistoryStore) -> str:
        """One turn in a conversation of its own, logged to `history`; the agent's final answer."""
        conversation = await manager.get_or_create(conversation_id, account)
        turn = history.start_turn(conversation_id, message, account) if history is not None else None
        conversation.on_event = turn.record_event if turn is not None else None
        if conversation.stand_in is not None and turn is not None:
            conversation.stand_in.on_step = turn.record_step
        answer = ""
        try:
            async with conversation.lock:
                async for event in run_turn(conversation.agent, conversation_id, message,
                                            turn.record if turn is not None else None):
                    if event["type"] == "answer":
                        answer = event["text"]
                    elif event["type"] == "error":
                        answer = f"The agent stopped: {event['message']}"
        finally:
            await manager.close(conversation_id)
        return answer

    agent = ask_agent or talk_to_agent

    async def optimize_job(analysis: Analysis, record: RunRecord) -> None:
        conversation_id = record.optimization["conversation"]
        follow_up: tuple[Analysis, RunRecord] | None = None
        try:
            answer = await agent(conversation_id, optimization_request(analysis, record, limits), transcripts(analysis.id))
            updated = store.get(analysis.id)
            if updated is not None and updated.version > analysis.version and updated.optimization is not None:
                record.optimization.update(status="done", new_version=updated.version, message=answer[:800])
                again = RunRecord(id=new_run_id(), started_at=now(), status="running", trigger="run",
                                  sources=record.sources, version=updated.version, optimized_from=analysis.version,
                                  queued_at=now_ms())
                store.save_run(analysis.id, again)
                follow_up = (updated, again)
                track(analysis.id, again.id, asyncio.current_task())
            else:
                record.optimization.update(
                    status="failed", message=(answer or "The agent did not save a new version.")[:800])
        except Exception as e:
            log.exception("optimizing saved analysis %s failed", analysis.id)
            record.optimization.update(status="failed", message=str(e))
        finally:
            store.stop_optimizing(analysis.id)
            store.save_run(analysis.id, record)
            untrack(analysis.id, "optimize")
        if follow_up is not None:
            await execute(*follow_up)

    def start(analysis: Analysis, swaps: dict[str, str], workflow_run: str | None = None, *,
              lane: Lane | None = None, estimates: list[InputMeasure] | None = None,
              inputs: asyncio.Task | None = None) -> tuple[RunRecord, asyncio.Task]:
        """Run `analysis` (reading `swaps` instead of its tables) in the background: its record, and the task.
        A workflow passes its `lane`, and the `estimates` of its size check."""
        record = RunRecord(id=new_run_id(), started_at=now(), status="running",
                           trigger="workflow" if workflow_run else "run", sources=swaps, version=analysis.version,
                           workflow_run=workflow_run, queued_at=now_ms())
        store.save_run(analysis.id, record)
        task = asyncio.create_task(execute(analysis, record, lane, estimates, inputs))
        track(analysis.id, record.id, task)
        return record, task

    def summary(analysis: Analysis) -> dict[str, Any]:
        runs = store.runs(analysis.id)
        return {
            "id": analysis.id, "title": analysis.title, "description": analysis.description,
            "question": analysis.question, "created_at": analysis.created_at,
            "version": analysis.version, "updated_at": analysis.updated_at,
            "outputs": [o.file for o in analysis.outputs],
            "last_run": runs[0].model_dump() if runs else None,
            "last_good_run": next((r.model_dump() for r in runs if r.status == "done"), None),
        }

    @router.get("")
    async def list_analyses():
        return [summary(a) for a in store.all()]

    @router.get("/{analysis_id}")
    async def get_analysis(analysis_id: str):
        analysis = get(analysis_id)
        return {**store.to_json(analysis), "sources": analysis.source_tables(),
                "runs": [r.model_dump() for r in store.runs(analysis_id)], "running": analysis_id in running,
                "optimizing": store.optimizing(analysis_id) is not None,
                "limits": dataclasses.asdict(limits) if limits is not None else None}

    @router.get("/{analysis_id}/versions")
    async def versions(analysis_id: str):
        get(analysis_id)
        return [{**store.to_json(v), "sources": v.source_tables()} for v in store.versions(analysis_id)]

    @router.get("/{analysis_id}/sources")
    async def sources(analysis_id: str):
        return await source_options(get(analysis_id).source_tables(), query_rows)

    @router.post("/{analysis_id}/runs", status_code=202)
    async def start_run(analysis_id: str, request: RunRequest | None = None):
        analysis = get(analysis_id)
        if analysis_id in running:
            raise HTTPException(409, "this analysis is already running")
        swaps = {old: new for old, new in (request.sources if request else {}).items() if old != new}
        if swaps:
            problems = await check_mapping(analysis.source_tables(), swaps, query_rows)
            if problems:
                raise HTTPException(400, "can't run on those tables: " + "; ".join(problems))
        record, _ = start(analysis, swaps)
        return record.model_dump()

    @router.get("/{analysis_id}/runs/{run_id}")
    async def get_run(analysis_id: str, run_id: str):
        get(analysis_id)
        record = store.get_run(analysis_id, run_id)
        if record is None:
            raise HTTPException(404, f"no run '{run_id}'")
        return record.model_dump()

    @router.post("/{analysis_id}/runs/{run_id}/optimize", status_code=202)
    async def optimize(analysis_id: str, run_id: str):
        analysis = get(analysis_id)
        record = store.get_run(analysis_id, run_id)
        if record is None:
            raise HTTPException(404, f"no run '{run_id}'")
        if analysis_id in running:
            raise HTTPException(409, "this analysis is running or being optimized")
        problem = main_finding(record.findings)
        if problem is None:
            raise HTTPException(400, "that run kept within every limit: there is nothing to optimize")
        if record.version != analysis.version:
            raise HTTPException(409, f"that run used version {record.version}, and the analysis is now version "
                                     f"{analysis.version}: run it again first")
        # Each attempt talks to the agent in a new conversation: a retry must not start from the
        # failed attempt's steps. Earlier attempts stay listed with the run.
        earlier = []
        if record.optimization:
            earlier = [*record.optimization.get("earlier", []),
                       {k: v for k, v in record.optimization.items() if k != "earlier"}]
        attempt = f"-{len(earlier) + 1}" if earlier else ""
        record.optimization = {"status": "running", "conversation": f"optimize-{record.id}{attempt}", "earlier": earlier}
        store.save_run(analysis_id, record)
        store.start_optimizing(analysis_id, {"run": record.id, "sources": record.sources,
                                             "kind": problem.kind, "reason": problem.reason})
        track(analysis_id, "optimize", asyncio.create_task(optimize_job(analysis, record)))
        return record.model_dump()

    @router.get("/{analysis_id}/optimizations/{conversation}")
    async def optimization_transcript(analysis_id: str, conversation: str):
        get(analysis_id)
        history = transcripts(analysis_id)
        if not _OPTIMIZATION.match(conversation) or not history.path(conversation).is_file():
            raise HTTPException(404, f"no optimization '{conversation}'")
        return history.read(conversation)

    @router.get("/{analysis_id}/runs/{run_id}/files/{name}")
    async def output_file(analysis_id: str, run_id: str, name: str):
        path = store.output_path(analysis_id, run_id, name)
        if path is None:
            raise HTTPException(404)
        # Written by the agent's script: served like any artifact, as an untrusted sandboxed page.
        return Response(local_plotly(path.read_bytes()), media_type="text/html", headers=ARTIFACT_HEADERS)

    @router.delete("/{analysis_id}")
    async def delete_analysis(analysis_id: str):
        if analysis_id in running:
            raise HTTPException(409, "this analysis is running; delete it when the run ends")
        if not store.delete(analysis_id):
            raise HTTPException(404, f"no saved analysis '{analysis_id}'")
        return {"deleted": analysis_id}

    return AnalysesApi(router=router, start=start, optimizing=lambda aid: "optimize" in running.get(aid, {}),
                       precheck=precheck, close_lane=close_lane, open_lane=open_lane, fetch=fetch)
