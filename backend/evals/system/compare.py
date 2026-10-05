"""Jev vs the full conversation, on the same questions: one row per conversation.

Two folders of saved conversations (api/history.py files):

    full  server.eval_history_dir      (evals/system/chat_history/)  every earlier turn sent
    jev   server.eval_jev_history_dir  (evals/system/jev_history/)   Jev picks the earlier turns

A pair is one conversation from each that asked the same messages in the
same order. Each turn of both is checked and judged like a System history
run (evals/system/history_run.py); the Jev vs full page shows the pairs.

    uv run --env-file ../.env python -m evals.system.compare --record   # ask each unpaired conversation's
                                                                        # questions again in the other mode
    uv run --env-file ../.env python -m evals.system.compare            # check and judge every pair
    uv run --env-file ../.env python -m evals.system.compare --no-judge

--record runs the real agent through the app (api/wiring.py), with or
without the context filter, so a pair differs only in what each turn was sent.
"""

import argparse
import asyncio
import json
import shutil
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from api.history import HistoryStore

from .cases import EvalCase
from .history_run import copy_artifacts, evaluate_saved_turn, new_run_id
from .judges import Judges

SIDES = ("full", "jev")
_SERVER = uuid.uuid4().hex[:12]
"""This process's sandbox label. Shared by every record() call, so a later
call's start-up clean-up does not take an earlier call's sandboxes for a
crashed run's leftovers."""
OUT = Path(__file__).resolve().parent / "out"
RUNS = OUT / "compare_runs"


def questions(history: HistoryStore, thread_id: str) -> list[str]:
    """The user's messages, in turn order."""
    by_turn = {r["turn"]: r["content"] for r in history.read(thread_id) if r.get("role") == "user" and r.get("turn")}
    return [by_turn[t] for t in sorted(by_turn)]


def _oldest_first(history: HistoryStore) -> list[str]:
    return sorted(history.thread_ids(), key=lambda t: history.path(t).stat().st_mtime)


def pair_histories(full: HistoryStore, jev: HistoryStore) -> dict[str, Any]:
    """Conversations with the same questions on both sides, and those without a twin."""
    unused: dict[tuple[str, ...], list[str]] = {}
    for thread in _oldest_first(jev):
        unused.setdefault(tuple(questions(jev, thread)), []).append(thread)
    pairs, only_full = [], []
    for thread in _oldest_first(full):
        asked = questions(full, thread)
        twins = unused.get(tuple(asked))
        if twins:
            pairs.append({"full": thread, "jev": twins.pop(0), "questions": asked})
        else:
            only_full.append({"thread": thread, "questions": asked})
    only_jev = [{"thread": t, "questions": list(q)} for q, threads in unused.items() for t in threads]
    return {"pairs": pairs, "only_full": only_full, "only_jev": only_jev}


def _cases(asked: list[str], thread: str) -> list[EvalCase]:
    return [EvalCase(f"{thread}_t{i + 1}", q, setup=asked[:i]) for i, q in enumerate(asked)]


async def evaluate_pairs(
    sides: dict[str, tuple[HistoryStore, Path]],
    run_dir: Path,
    judges: Judges | None,
    meta: dict[str, Any],
    on_pair: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Check and judge both conversations of every pair. `sides`: side -> (its history, its artifacts folder)."""
    found = pair_histories(sides["full"][0], sides["jev"][0])
    meta = {
        **meta,
        "id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "total": len(found["pairs"]),
        "unpaired": {"full": found["only_full"], "jev": found["only_jev"]},
    }
    rows: list[dict[str, Any]] = []
    def save() -> None:  # after every pair, so the page can show progress
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(
            json.dumps({**meta, "pairs": rows}, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    save()
    try:
        for pair in found["pairs"]:
            row: dict[str, Any] = {"id": pair["full"], "questions": pair["questions"]}
            for side in SIDES:
                history, artifacts_dir = sides[side]
                thread = pair[side]
                copied = run_dir / "history" / side / f"{thread}.jsonl"
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(history.path(thread), copied)
                results = []
                for case in _cases(pair["questions"], thread):
                    result = await evaluate_saved_turn(history, thread, case, artifacts_dir, judges)
                    copy_artifacts(artifacts_dir, thread, result["artifacts"], run_dir / "artifacts" / side / case.id)
                    results.append(result)
                row[side] = {"thread": thread, "results": results}
            rows.append(row)
            if on_pair is not None:
                on_pair(row)
            save()
        meta["status"] = "done"
    except Exception:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc(limit=5)
        raise
    finally:
        save()
    return rows


async def record(conversations: list[list[str]], history_dir: Path, jev: bool) -> list[str]:
    """Ask each conversation's messages through the app, with Jev as the context manager or not.

    Saves the histories (and files made) to `history_dir`. Their thread ids.
    """
    import httpx

    from api.app import create_app
    from api.wiring import make_context_filter, make_manager
    from config import cfg
    from datasources import close_database
    from sandboxes import get_provider

    from .run import parse_sse

    cfg.server.context_filter = jev
    context = make_context_filter()
    history = HistoryStore(history_dir)
    provider = get_provider(cfg.sandbox, labels={"role": "compare", "server": _SERVER})
    manager = make_manager(provider, memory=context is None)
    app = create_app(manager, artifacts_dir=history_dir / "artifacts", output_dir=cfg.sandbox.output_dir,
                     history=history, context=context)
    made = []
    manager.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://compare", timeout=None) as client:
            for asked in conversations:
                thread = uuid.uuid4().hex
                for message in asked:
                    response = await client.post("/api/chat", json={"message": message, "thread_id": thread})
                    errors = [e["message"] for e in parse_sse(response.text) if e["type"] == "error"]
                    print(f"[{'jev' if jev else 'full'} {thread[:8]}] {message[:70]}"
                          + (f"  ERRORS: {errors}" if errors else ""), flush=True)
                await client.delete(f"/api/conversations/{thread}")
                made.append(thread)
    finally:
        await manager.close_all()
    await close_database()
    return made


async def main() -> None:
    from config import cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true",
                        help="ask each unpaired conversation's questions in the other mode, to make its twin")
    parser.add_argument("--no-judge", action="store_true", help="code checks only, no LLM judges")
    parser.add_argument("--judge-model", default=cfg.agent.model, help="default: the agent's model")
    args = parser.parse_args()

    full_dir, jev_dir = cfg.server.eval_history_dir, cfg.server.eval_jev_history_dir
    found = pair_histories(HistoryStore(full_dir), HistoryStore(jev_dir))
    print(f"{len(found['pairs'])} pairs, {len(found['only_full'])} full-only, {len(found['only_jev'])} Jev-only")

    if args.record:
        if found["only_full"]:
            await record([c["questions"] for c in found["only_full"]], jev_dir, jev=True)
        if found["only_jev"]:
            await record([c["questions"] for c in found["only_jev"]], full_dir, jev=False)
        return

    run_dir = RUNS / new_run_id()
    judges = None if args.no_judge else Judges(args.judge_model)
    meta = {"judge_model": None if args.no_judge else args.judge_model}
    rows = await evaluate_pairs(
        {"full": (HistoryStore(full_dir), full_dir / "artifacts"), "jev": (HistoryStore(jev_dir), jev_dir / "artifacts")},
        run_dir, judges, meta,
        on_pair=lambda row: print(f"{row['questions'][0][:60]:60} ({len(row['questions'])} turns) judged", flush=True),
    )
    print(f"\n{len(rows)} pairs -> {run_dir.relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
