"""How much the agent's code has cost in the sandboxes: read-only, for the Load page.

    GET /api/load?account=alice&limit=200

Every code step is a `sandbox_step` line in its conversation's history
(sandboxes/metering.py writes them; api/history.py stores them). This reads
them all, newest first, with totals overall and per account.
"""

from typing import Any

from fastapi import APIRouter, Query

from .history import HistoryStore

UNKNOWN_ACCOUNT = "unknown"


def collect_steps(history: HistoryStore) -> list[dict[str, Any]]:
    """Every recorded code step, oldest first, with its account and the question it served."""
    steps = []
    for thread_id in history.thread_ids():
        records = history.read(thread_id)
        prompts = {r["turn"]: r["content"] for r in records if r.get("role") == "user"}
        account = next((r["account"] for r in records if r.get("role") == "user" and r.get("account")), UNKNOWN_ACCOUNT)
        for r in records:
            if r.get("role") != "sandbox_step":
                continue
            limit, peak = r.get("memory_limit_mb"), r.get("peak_memory_mb")
            steps.append({
                "conversation": thread_id,
                "account": account,
                "turn": r.get("turn"),
                "prompt": (prompts.get(r.get("turn")) or "")[:200],
                "ts": r.get("ts", ""),
                "command": r.get("command", ""),
                "exit_code": r.get("exit_code"),
                "seconds": r.get("seconds"),
                "run_seconds": r.get("run_seconds"),
                "cpu_seconds": r.get("cpu_seconds"),
                "peak_memory_mb": peak,
                "memory_limit_mb": limit,
                "out_of_memory": bool(r.get("signal") == 9 and limit and peak and peak >= 0.8 * limit),
            })
    steps.sort(key=lambda s: s["ts"])
    return steps


def summarize(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals for a set of steps. Missing measurements (no python3) are skipped, not zeros."""
    def total(key):
        return round(sum(s[key] or 0 for s in steps), 2)

    peaks = [s["peak_memory_mb"] for s in steps if s["peak_memory_mb"] is not None]
    run = total("run_seconds")
    return {
        "steps": len(steps),
        "conversations": len({s["conversation"] for s in steps}),
        "run_seconds": run,
        "cpu_seconds": total("cpu_seconds"),
        "overhead_seconds": round(total("seconds") - run, 2),
        "peak_memory_mb": max(peaks, default=None),
        "average_peak_memory_mb": round(sum(peaks) / len(peaks), 1) if peaks else None,
        "out_of_memory": sum(s["out_of_memory"] for s in steps),
        "failed": sum(s["exit_code"] not in (0, None) for s in steps),
    }


def load_router(history: HistoryStore) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/load")
    async def load(account: str | None = None, limit: int = Query(200, ge=1, le=2000)):
        all_steps = collect_steps(history)
        accounts = sorted({s["account"] for s in all_steps})
        steps = [s for s in all_steps if account is None or s["account"] == account]
        limits = [s["memory_limit_mb"] for s in steps if s["memory_limit_mb"]]
        return {
            "account": account,
            "accounts": accounts,
            "memory_limit_mb": limits[-1] if limits else None,
            "summary": summarize(steps),
            "per_account": {a: summarize([s for s in all_steps if s["account"] == a]) for a in accounts},
            "steps": list(reversed(steps))[:limit],
        }

    return router
