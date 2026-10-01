"""Read-only routes over saved eval runs, for the app's Evals page.

    GET /api/evals/context/runs                          every run, newest first, with a summary per arm
    GET /api/evals/context/runs/{run_id}                 one run: settings, conversations, every turn's result
    GET /api/evals/context/runs/{run_id}/history/{conversation}/{arm}?repeat=1
                                                         what that conversation's agent saw and did (JSONL lines)

Runs are written by evals/memory/run.py, one folder each:
<runs_dir>/<run id>/run.json and <runs_dir>/<run id>/history/*.jsonl.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .history import HistoryStore

_RUN_ID = re.compile(r"^\d{8}-\d{6}$")
_NAME = re.compile(r"^[a-z0-9_]{1,64}$")


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
