"""Routes for the Jev vs full page: pairs of saved conversations with the same questions.

    GET  /api/evals/compare/pairs                          how many pairs and unpaired conversations there are
    POST /api/evals/compare/runs                           check and judge every pair, in the background
    GET  /api/evals/compare/runs                           every comparison run, newest first
    GET  /api/evals/compare/runs/{run_id}                  one run: a row per pair, each turn's result per side
    GET  /api/evals/compare/runs/{run_id}/history/{side}/{thread}   that side's conversation history

Runs are written by evals/system/compare.py: <runs_dir>/<run id>/run.json
and history/<side>/<thread>.jsonl.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from .history import HistoryStore

log = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^\d{8}-\d{6}$")
_THREAD = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SIDES = ("full", "jev")


def side_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """One side of a pair: verdict counts per dimension, and what its turns cost in all."""
    count = lambda dim, status: sum(r[dim]["status"] == status for r in results)  # noqa: E731
    return {
        "turns": len(results),
        **{
            dim: {"pass": count(dim, "pass"), "fail": count(dim, "fail"),
                  "unknown": count(dim, "unknown") + count(dim, "judge_error")}
            for dim in ("correct", "grounded", "instruction_following", "execution_strategy")
        },
        "steps": sum(r["efficiency"]["steps"] for r in results),
        "total_tokens": sum(r["efficiency"]["total_tokens"] for r in results),
        "runtime_seconds": round(sum(r["efficiency"]["runtime_seconds"] for r in results), 2),
        "errors": sum(len(r["recovery"]["errors"]) for r in results),
    }


def with_summaries(run: dict[str, Any]) -> dict[str, Any]:
    for row in run.get("pairs", []):
        for side in SIDES:
            row[side]["summary"] = side_summary(row[side]["results"])
    return run


def compare_router(
    runs_dir: Path,
    full: tuple[HistoryStore, Path] | None = None,
    jev: tuple[HistoryStore, Path] | None = None,
    make_judges: Callable[[], Any] | None = None,
) -> APIRouter:
    """`full` and `jev`: each side's saved conversations and their artifacts folder."""
    router = APIRouter(prefix="/api/evals/compare")
    runs_dir = Path(runs_dir)
    job: dict[str, Any] = {"id": None, "task": None}

    def running() -> str | None:
        task = job["task"]
        return job["id"] if task is not None and not task.done() else None

    def load(run_id: str) -> dict[str, Any]:
        path = runs_dir / run_id / "run.json"
        if not _RUN_ID.match(run_id) or not path.is_file():
            raise HTTPException(404, f"no comparison run '{run_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.get("/pairs")
    async def preview():
        if full is None or jev is None:
            return {"available": False, "pairs": 0, "turns": 0, "only_full": 0, "only_jev": 0, "running": None}
        from evals.system.compare import pair_histories

        found = pair_histories(full[0], jev[0])
        return {
            "available": True,
            "pairs": len(found["pairs"]),
            "turns": sum(len(p["questions"]) for p in found["pairs"]),
            "only_full": len(found["only_full"]),
            "only_jev": len(found["only_jev"]),
            "running": running(),
        }

    @router.post("/runs", status_code=202)
    async def start():
        if full is None or jev is None:
            raise HTTPException(404, "this server has no conversations to compare")
        if running():
            raise HTTPException(409, f"a comparison is already running ({running()})")
        from evals.system.compare import evaluate_pairs, new_run_id

        run_id = new_run_id()
        run_dir = runs_dir / run_id
        if run_dir.exists():
            raise HTTPException(409, "a run started this second; try again")
        judges = make_judges() if make_judges is not None else None
        meta = {"judge_model": getattr(judges, "name", None)}

        async def go():
            try:
                await evaluate_pairs({"full": full, "jev": jev}, run_dir, judges, meta)
            except Exception:
                log.exception("comparing Jev with the full conversation failed")

        job["id"], job["task"] = run_id, asyncio.create_task(go())
        await asyncio.sleep(0)  # let it write run.json before the page asks for it
        return {"id": run_id}

    @router.get("/runs")
    async def list_runs():
        runs = []
        for path in sorted(runs_dir.glob("*/run.json"), reverse=True):
            if not _RUN_ID.match(path.parent.name):
                continue
            run = with_summaries(json.loads(path.read_text(encoding="utf-8")))
            runs.append({
                "id": run["id"], "created_at": run["created_at"], "judge_model": run.get("judge_model"),
                "status": run.get("status", "done"), "total": run.get("total", len(run["pairs"])),
                "pairs": len(run["pairs"]),
            })
        return runs

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str):
        return with_summaries(load(run_id))

    @router.get("/runs/{run_id}/history/{side}/{thread}")
    async def history(run_id: str, side: str, thread: str):
        load(run_id)
        if side not in SIDES or not _THREAD.match(thread):
            raise HTTPException(404)
        store = HistoryStore(runs_dir / run_id / "history" / side)
        if not store.path(thread).is_file():
            raise HTTPException(404, f"no {side} history for {thread}")
        return store.read(thread)

    return router
