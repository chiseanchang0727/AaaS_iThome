"""How much the agent's code has cost in the sandboxes: read-only, for the Load page.

    GET /api/load?account=alice&limit=200          steps, totals, and one row per conversation
    GET /api/load/conversations/{id}               that conversation's timeline

Every code step is a `sandbox_step` line in its conversation's history
(sandboxes/metering.py writes them; api/history.py stores them), and what
happened to its sandbox a `sandbox_event` line (api/conversations.py). This
reads them back.
"""

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .history import HistoryStore

UNKNOWN_ACCOUNT = "unknown"
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _account(records: list[dict[str, Any]]) -> str:
    return next((r["account"] for r in records if r.get("role") == "user" and r.get("account")), UNKNOWN_ACCOUNT)


def conversation_rows(history: HistoryStore) -> list[dict[str, Any]]:
    """One row per conversation that ran code: what it cost and what happened to its sandbox."""
    rows = []
    for thread_id in history.thread_ids():
        records = history.read(thread_id)
        steps = [r for r in records if r.get("role") == "sandbox_step"]
        events = [r for r in records if r.get("role") == "sandbox_event"]
        if not steps and not events:
            continue
        peaks = [r["peak_memory_mb"] for r in steps if r.get("peak_memory_mb") is not None]
        sizes = [r["memory_gb"] for r in events if r.get("event") in ("sandbox_ready", "switched") and r.get("memory_gb")]
        rows.append({
            "conversation": thread_id,
            "account": _account(records),
            "question": next((r["content"][:200] for r in records if r.get("role") == "user"), ""),
            "turns": len({r.get("turn") for r in records if r.get("role") == "user"}),
            "steps": len(steps),
            "out_of_memory": sum(r.get("signal") == 9 and _near_limit(r) for r in steps),
            "upgrades": sum(r.get("event") == "switched" for r in events),
            "upgrade_failures": sum(r.get("event") == "upgrade_failed" for r in events),
            "peak_memory_mb": max(peaks, default=None),
            "sandbox_gb": sizes[-1] if sizes else None,
            "last": records[-1].get("ts", "") if records else "",
        })
    rows.sort(key=lambda r: r["last"], reverse=True)
    return rows


def _near_limit(r: dict[str, Any]) -> bool:
    limit, peak = r.get("memory_limit_mb"), r.get("peak_memory_mb")
    return bool(limit and peak and peak >= 0.8 * limit)


def _call_summary(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    for key in ("command", "file_path", "sql", "filename"):
        if args.get(key):
            return str(args[key])[:300]
    return ""


def timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as it happened: questions, the agent's actions, code steps,
    sandbox events and answers, in order. Tool results are left out (see the chat)."""
    items = []
    for r in records:
        role, base = r.get("role"), {"ts": r.get("ts", ""), "turn": r.get("turn")}
        if role == "user":
            items.append({**base, "kind": "question", "text": r.get("content", "")})
        elif role == "assistant" and r.get("tool_calls"):
            for call in r["tool_calls"]:
                items.append({**base, "kind": "action", "tool": call.get("name"), "detail": _call_summary(call)})
        elif role == "assistant" and r.get("content"):
            items.append({**base, "kind": "answer", "text": r["content"][:600]})
        elif role == "sandbox_step":
            items.append({**base, "kind": "step", **{k: r.get(k) for k in (
                "command", "exit_code", "seconds", "run_seconds", "cpu_seconds", "peak_memory_mb",
                "memory_limit_mb", "signal")}, "out_of_memory": bool(r.get("signal") == 9 and _near_limit(r))})
        elif role == "sandbox_event":
            fields = {k: v for k, v in r.items() if k not in ("id", "previous", "turn", "role", "ts", "event")}
            items.append({**base, "kind": "event", "event": r.get("event"), **fields})
    return items


def collect_steps(history: HistoryStore) -> list[dict[str, Any]]:
    """Every recorded code step, oldest first, with its account and the question it served."""
    steps = []
    for thread_id in history.thread_ids():
        records = history.read(thread_id)
        prompts = {r["turn"]: r["content"] for r in records if r.get("role") == "user"}
        account = _account(records)
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
            "conversations": [c for c in conversation_rows(history) if account is None or c["account"] == account],
        }

    @router.get("/load/conversations/{conversation_id}")
    async def conversation(conversation_id: str):
        records = history.read(conversation_id) if _ID.match(conversation_id) else []
        if not records:
            raise HTTPException(404, f"no conversation '{conversation_id}'")
        return {"conversation": conversation_id, "account": _account(records), "timeline": timeline(records)}

    return router
