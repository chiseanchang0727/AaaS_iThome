"""Read-only routes over saved eval runs, for the app's Evals page.

    GET /api/evals/context/runs                          every run, newest first, with a summary per arm
    GET /api/evals/context/runs/{run_id}                 one run: settings, conversations, every turn's result
    GET /api/evals/context/runs/{run_id}/history/{conversation}/{arm}?repeat=1
                                                         what that conversation's agent saw and did (JSONL lines)

    GET /api/evals/system/runs                           every whole-system run, newest first, with totals
    GET /api/evals/system/runs/{run_id}                  one run: settings, cases, every case's result
    GET /api/evals/system/runs/{run_id}/history/{case}   that case's conversation history (JSONL lines)
    GET  /api/evals/system/history-runs                  the folder of conversations to evaluate, and how many turns are in it
    POST /api/evals/system/history-runs                  check and judge every saved turn, in the background

Runs are written by evals/memory/run.py (context) and evals/system/run.py
(system), one folder each: <runs_dir>/<run id>/run.json and
<runs_dir>/<run id>/history/*.jsonl.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .history import HistoryStore

log = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^\d{8}-\d{6}$")
_NAME = re.compile(r"^[a-z0-9_]{1,64}$")
_CASE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
"""A system eval case id: a case name, or `<conversation id>_t<turn>` for a saved turn."""


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Totals per arm, as in evals/memory/run.py's printed table."""
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)
    summary = {}
    for arm, rs in by_arm.items():
        checked = [r for r in rs if r["correct"] is not None]
        n = len(rs)
        summary[arm] = {
            "turns": n,
            "checked": len(checked),
            "correct": sum(r["correct"] for r in checked),
            "grounded": sum(not r["ungrounded"] for r in rs),
            "steps": sum(r["steps"] for r in rs),
            "queries": sum(r["queries"] for r in rs),
            "skill_reads": sum(r["skill_reads"] for r in rs),
            "tokens_per_turn": sum(r["input_tokens"] for r in rs) // n,
            "first_call_per_turn": sum(r.get("first_call_tokens", 0) for r in rs) // n,
        }
    return summary


def evals_router(runs_dir: Path) -> APIRouter:
    router = APIRouter(prefix="/api/evals/context")
    runs_dir = Path(runs_dir)

    def load(run_id: str) -> dict[str, Any]:
        path = runs_dir / run_id / "run.json"
        if not _RUN_ID.match(run_id) or not path.is_file():
            raise HTTPException(404, f"no eval run '{run_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.get("/runs")
    async def list_runs():
        runs = []
        for path in sorted(runs_dir.glob("*/run.json"), reverse=True):
            if not _RUN_ID.match(path.parent.name):
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            runs.append({
                "id": run["id"],
                "created_at": run["created_at"],
                "model": run["model"],
                "arms": run["arms"],
                "repeats": run["repeats"],
                "conversations": [c["id"] for c in run["conversations"]],
                "summary": summarize(run["results"]),
            })
        return runs

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str):
        run = load(run_id)
        return {**run, "summary": summarize(run["results"])}

    @router.get("/runs/{run_id}/history/{conversation}/{arm}")
    async def get_history(run_id: str, conversation: str, arm: str, repeat: int = Query(1, ge=1)):
        load(run_id)
        if not (_NAME.match(conversation) and _NAME.match(arm)):
            raise HTTPException(404)
        thread_id = f"{conversation}-{arm}-run{repeat}"
        store = HistoryStore(runs_dir / run_id / "history")
        if not store.path(thread_id).is_file():
            raise HTTPException(404, f"no history for {conversation} / {arm} / repeat {repeat}")
        return store.read(thread_id)

    return router


DIMENSIONS = ("correct", "grounded", "instruction_following", "execution_strategy")


def system_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass counts per judged dimension, and what the run cost, as in evals/system/run.py."""
    effort = [r["efficiency"] for r in results]
    return {
        "cases": len(results),
        **{d: sum(r[d]["status"] == "pass" for r in results) for d in DIMENSIONS},
        "judge_errors": sum(r[d]["status"] == "judge_error" for r in results for d in DIMENSIONS),
        "total_tokens": sum(e["total_tokens"] for e in effort),
        "runtime_seconds": round(sum(e["runtime_seconds"] for e in effort), 2),
    }


def system_router(
    runs_dir: Path,
    history: HistoryStore | None = None,
    artifacts_dir: Path | None = None,
    make_judges: Callable[[], Any] | None = None,
) -> APIRouter:
    """The system eval's routes. With `history`, also evaluating the saved conversations in it.

    `make_judges` returns evals.system.judges.Judges (or None for code checks
    only); it is called per run, so no model client exists until one starts.
    """
    router = APIRouter(prefix="/api/evals/system")
    runs_dir = Path(runs_dir)
    job: dict[str, Any] = {"id": None, "task": None}

    def running() -> str | None:
        task = job["task"]
        return job["id"] if task is not None and not task.done() else None

    def load(run_id: str) -> dict[str, Any]:
        path = runs_dir / run_id / "run.json"
        if not _RUN_ID.match(run_id) or not path.is_file():
            raise HTTPException(404, f"no eval run '{run_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.get("/runs")
    async def list_runs():
        runs = []
        for path in sorted(runs_dir.glob("*/run.json"), reverse=True):
            if not _RUN_ID.match(path.parent.name):
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            runs.append({
                "id": run["id"],
                "created_at": run["created_at"],
                "model": run["model"],
                "judge_model": run.get("judge_model"),
                "rejudged_from": run.get("rejudged_from"),
                "cases": [r["case_id"] for r in run["results"]],
                "source": run.get("source", "cases"),
                "status": run.get("status", "done"),
                "total": run.get("total", len(run["results"])),
                "summary": system_summary(run["results"]),
            })
        return runs

    @router.get("/history-runs")
    async def history_preview():
        if history is None:
            return {"available": False, "folder": None, "conversations": 0, "turns": 0, "running": None}
        from evals.system.history_run import history_cases

        cases = history_cases(history)
        return {
            "available": True,
            "folder": str(history.root),
            "conversations": len({thread for _, thread in cases}),
            "turns": len(cases),
            "running": running(),
        }

    @router.post("/history-runs", status_code=202)
    async def start_history_run():
        if history is None:
            raise HTTPException(404, "this server has no conversation history to evaluate")
        if running():
            raise HTTPException(409, f"an evaluation of the history is already running ({running()})")
        from evals.system.history_run import evaluate_history, history_cases, new_run_id

        run_id = new_run_id()
        run_dir = runs_dir / run_id
        if run_dir.exists():
            raise HTTPException(409, "a run started this second; try again")
        judges = make_judges() if make_judges is not None else None
        meta = {"model": "saved conversations", "judge_model": getattr(judges, "name", None), "sandbox": False, "rejudged_from": None}

        async def go():
            try:
                await evaluate_history(history, artifacts_dir or runs_dir / "_none", run_dir, judges, meta)
            except Exception:
                log.exception("evaluating the conversation history failed")

        job["id"], job["task"] = run_id, asyncio.create_task(go())
        await asyncio.sleep(0)  # let it write run.json before the page asks for it
        return {"id": run_id, "total": len(history_cases(history))}

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str):
        run = load(run_id)
        return {**run, "summary": system_summary(run["results"])}

    @router.get("/runs/{run_id}/history/{case_id}")
    async def get_history(run_id: str, case_id: str):
        load(run_id)
        if not _CASE.match(case_id):
            raise HTTPException(404)
        store = HistoryStore(runs_dir / run_id / "history")
        if not store.path(case_id).is_file():
            raise HTTPException(404, f"no history for {case_id}")
        return store.read(case_id)

    return router
