"""Saved analyses: list them, run one again, and serve what a run made.

    GET    /api/analyses                                   every saved analysis, newest first, with its last run
    GET    /api/analyses/{id}                              the recipe, the tables it reads, and its runs
    GET    /api/analyses/{id}/sources                      for each table it reads: the tables that could stand in
    POST   /api/analyses/{id}/runs                         run it again, in the background (202); body
                                                           {"sources": {"videos": "videos_ca"}} reads other tables
    GET    /api/analyses/{id}/runs/{run_id}                one run: status, error, output files
    GET    /api/analyses/{id}/runs/{run_id}/files/{name}   a file that run made
    DELETE /api/analyses/{id}                              delete it and its runs

A run takes a sandbox like a conversation does (same limits, warm pool and
start-up), runs the recipe (analyses/runner.py), and gives the sandbox back.
No model is involved. If the script is killed for running out of memory, the
run moves to the bigger sandbox once and tries again.
"""

import asyncio
import logging
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analyses import Analysis, AnalysisStore, RunRecord
from analyses.models import now
from analyses.runner import Query, ReadDataset, RunResult, run
from analyses.sources import QueryRows, check_mapping, source_options
from analyses.store import new_run_id

from .app import ARTIFACT_HEADERS
from .conversations import ConversationManager

log = logging.getLogger(__name__)

RUN_FOLDER = ".analysis-run"
"""The scratch folder for a run, inside the sandbox's work folder."""


class RunRequest(BaseModel):
    sources: dict[str, str] = {}
    """Tables to read instead of the recipe's own, e.g. {"videos": "videos_ca"}."""


def analyses_router(
    store: AnalysisStore,
    manager: ConversationManager,
    *,
    query: Query,
    query_rows: QueryRows,
    read_dataset: ReadDataset,
    work_dir: PurePosixPath,
    account: str,
) -> APIRouter:
    """`query`: SQL -> Parquet bytes for an input; `query_rows`: SQL -> rows, to look up tables' columns."""
    router = APIRouter(prefix="/api/analyses")
    running: dict[str, asyncio.Task] = {}
    """analysis id -> its run in progress."""

    def get(analysis_id: str) -> Analysis:
        analysis = store.get(analysis_id)
        if analysis is None:
            raise HTTPException(404, f"no saved analysis '{analysis_id}'")
        return analysis

    async def run_once(sandbox, analysis: Analysis, sources: dict[str, str]) -> RunResult:
        return await run(sandbox, analysis, query=query, read_dataset=read_dataset,
                         scratch=work_dir / RUN_FOLDER, sources=sources)

    async def execute(analysis: Analysis, record: RunRecord) -> None:
        conversation_id = f"analysis-{record.id}"
        result = RunResult(ok=False, error="the run did not start")
        try:
            conversation = await manager.get_or_create(conversation_id, account)
            if conversation.starting is not None:
                await conversation.starting
            if conversation.sandbox is None:
                result = RunResult(ok=False, error="no sandbox could be started (all may be busy); try again")
            else:
                result = await run_once(conversation.sandbox, analysis, record.sources)
                if result.killed:
                    upgrade = await manager.upgrade(conversation, "a saved analysis ran out of memory")
                    record.notes.append(upgrade.note)
                    if upgrade.sandbox is not None:
                        result = await run_once(upgrade.sandbox, analysis, record.sources)
        except Exception as e:
            log.exception("running saved analysis %s failed", analysis.id)
            result = RunResult(ok=False, error=str(e))
        finally:
            await manager.close(conversation_id)
            record.status = "done" if result.ok else "failed"
            record.seconds, record.error, record.log = result.seconds, result.error, result.log
            record.outputs = list(result.outputs)
            store.save_run(analysis.id, record, result.outputs)
            running.pop(analysis.id, None)

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
                "runs": [r.model_dump() for r in store.runs(analysis_id)], "running": analysis_id in running}

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
        record = RunRecord(id=new_run_id(), started_at=now(), status="running", trigger="run", sources=swaps,
                           version=analysis.version)
        store.save_run(analysis_id, record)
        running[analysis_id] = asyncio.create_task(execute(analysis, record))
        return record.model_dump()

    @router.get("/{analysis_id}/runs/{run_id}")
    async def get_run(analysis_id: str, run_id: str):
        get(analysis_id)
        record = store.get_run(analysis_id, run_id)
        if record is None:
            raise HTTPException(404, f"no run '{run_id}'")
        return record.model_dump()

    @router.get("/{analysis_id}/runs/{run_id}/files/{name}")
    async def output_file(analysis_id: str, run_id: str, name: str):
        path = store.output_path(analysis_id, run_id, name)
        if path is None:
            raise HTTPException(404)
        # Written by the agent's script: served like any artifact, as an untrusted sandboxed page.
        return FileResponse(path, media_type="text/html", headers=ARTIFACT_HEADERS)

    @router.delete("/{analysis_id}")
    async def delete_analysis(analysis_id: str):
        if analysis_id in running:
            raise HTTPException(409, "this analysis is running; delete it when the run ends")
        if not store.delete(analysis_id):
            raise HTTPException(404, f"no saved analysis '{analysis_id}'")
        return {"deleted": analysis_id}

    return router
