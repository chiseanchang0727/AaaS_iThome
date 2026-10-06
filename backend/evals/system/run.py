"""Evaluate the whole agent system: run each case through the real app, then check and judge it.

    uv run --env-file ../.env python -m evals.system.run                      # every case
    uv run --env-file ../.env python -m evals.system.run --only lookup_counts,oom_ward_clustering
    uv run --env-file ../.env python -m evals.system.run --no-sandbox         # skip cases that need one
    uv run --env-file ../.env python -m evals.system.run --no-judge           # code checks only
    uv run --env-file ../.env python -m evals.system.run --context-filter     # Jev picks the earlier turns sent
    uv run --env-file ../.env python -m evals.system.run --rejudge 20261005-140000   # check and judge a saved run again
    uv run --env-file ../.env python -m evals.system.run --history            # judge the conversations in server.eval_history_dir
    uv run python -m evals.system.run --list

Each case is sent as chat messages to the app's own POST /api/chat, served
in-process with the production wiring (api/wiring.py): the same agent,
skills, sandbox stand-in, out-of-memory upgrades and history log as a user
gets. The trace is read back from the history file the app writes; nothing
is logged twice.

Needs the grounding eval's mock database (see evals/grounding/run.py): the
expected values are computed from it.

Each invocation is saved as out/runs/<run id>/: run.json (settings, cases,
results), history/ (the app's JSONL per case) and artifacts/ (files the agent
made). The System page under Evals in the app reads them (api/evals.py).
"""

import argparse
import asyncio
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from api.app import create_app
from api.history import HistoryStore
from api.wiring import make_context_filter, make_manager
from config import cfg
from datasources import close_database
from evals.grounding.mock_data import SKILL_PATH
from evals.grounding.run import use_mock_database
from sandboxes import get_provider

from .cases import EvalCase, build_cases
from .evaluate import evaluate
from .history_run import evaluate_history
from .judges import Judges
from .trace import build_trace

OUT = Path(__file__).resolve().parent / "out"
RUNS = OUT / "runs"
ACCOUNT = "eval"


def parse_sse(body: str) -> list[dict[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        data = [line[len("data: "):] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


async def run_case(client: httpx.AsyncClient, case: EvalCase) -> tuple[list[str], list[str]]:
    """Send the case's messages in one conversation. Errors the app streamed, and the last turn's files."""
    headers = {"X-Account": ACCOUNT}
    errors: list[str] = []
    artifacts: list[str] = []
    try:
        for message in case.turns:
            response = await client.post(
                "/api/chat", json={"message": message, "thread_id": case.id}, headers=headers
            )
            if response.status_code != 200:
                errors.append(f"HTTP {response.status_code}: {response.text[:500]}")
                break
            events = parse_sse(response.text)
            errors += [e["message"] for e in events if e["type"] == "error"]
            artifacts = [e["name"] for e in events if e["type"] == "artifact"]
    finally:
        await client.delete(f"/api/conversations/{case.id}", headers=headers)
    return errors, artifacts


def print_line(result: dict[str, Any]) -> None:
    e = result["efficiency"]
    marks = {"pass": "pass", "fail": "FAIL", "unknown": "?", "judge_error": "ERR"}
    cols = [marks.get(result[k]["status"], result[k]["status"])
            for k in ("correct", "grounded", "instruction_following", "execution_strategy")]
    print(f"{result['case_id']:34} correct={cols[0]:5} grounded={cols[1]:5} instr={cols[2]:5} strategy={cols[3]:5} "
          f"steps={e['steps']:<3} tokens={e['total_tokens']:<8,} {e['runtime_seconds']}s", flush=True)


def save(run_dir: Path, meta: dict[str, Any], results: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({**meta, "results": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


async def run(cases: list[EvalCase], run_dir: Path, meta: dict[str, Any], judges: Judges | None) -> list[dict]:
    history = HistoryStore(run_dir / "history")
    provider = get_provider(cfg.sandbox, labels={"role": "eval", "server": uuid.uuid4().hex[:12]})
    context = make_context_filter()
    manager = make_manager(provider, memory=context is None)
    app = create_app(manager, artifacts_dir=run_dir / "artifacts", output_dir=cfg.sandbox.output_dir,
                     history=history, default_account=ACCOUNT, context=context)
    results: list[dict[str, Any]] = []
    manager.start()  # what the app's startup does: clean up, fill the warm pool
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=None) as client:
            for case in cases:
                errors, artifacts = await run_case(client, case)
                trace = build_trace(history.read(case.id), turn=len(case.turns), artifacts=artifacts)
                result = await evaluate(case, trace, judges, errors)
                results.append(result)
                print_line(result)
                save(run_dir, meta, results)  # after every case, so a stopped run keeps what ran
    finally:
        await manager.close_all()
        await close_database()
    return results


async def rejudge(old_id: str, cases: list[EvalCase], run_dir: Path, meta: dict[str, Any],
                  judges: Judges | None) -> list[dict]:
    """Check and judge a saved run's histories again, with today's cases, checks and prompts."""
    old_dir = RUNS / old_id
    old = json.loads((old_dir / "run.json").read_text(encoding="utf-8"))
    shutil.copytree(old_dir / "history", run_dir / "history")
    if (old_dir / "artifacts").is_dir():
        shutil.copytree(old_dir / "artifacts", run_dir / "artifacts")
    history = HistoryStore(run_dir / "history")
    by_id = {c.id: c for c in cases}
    results = []
    for previous in old["results"]:
        case = by_id.get(previous["case_id"])
        if case is None:
            continue
        trace = build_trace(history.read(case.id), turn=len(case.turns), artifacts=previous.get("artifacts", []))
        result = await evaluate(case, trace, judges, previous.get("stream_errors", []))
        results.append(result)
        print_line(result)
        save(run_dir, meta, results)
    return results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--no-sandbox", action="store_true", help="no code execution; skips cases that need it")
    parser.add_argument("--no-judge", action="store_true", help="code checks only, no LLM judges")
    parser.add_argument("--judge-model", default=cfg.agent.model, help="default: the agent's model")
    parser.add_argument("--rejudge", metavar="RUN_ID", help="check and judge a saved run again, without running the agent")
    parser.add_argument("--context-filter", action="store_true",
                        help="send each turn only the earlier turns Jev picks (server.context_filter)")
    parser.add_argument("--history", action="store_true",
                        help="check and judge the saved conversations in server.eval_history_dir instead of the cases")
    parser.add_argument("--list", action="store_true", help="print the cases and stop")
    args = parser.parse_args()

    if args.history:
        run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
        judges = None if args.no_judge else Judges(args.judge_model)
        meta = {"model": "saved conversations", "judge_model": None if args.no_judge else args.judge_model,
                "sandbox": False, "rejudged_from": None}
        target = cfg.server.eval_history_dir
        results = await evaluate_history(HistoryStore(target), target / "artifacts",
                                         run_dir, judges, meta, on_result=print_line)
        print(f"\n{len(results)} turns -> {run_dir.relative_to(OUT.parent.parent.parent)}")
        return

    cases = build_cases()
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.id for c in cases}
        if unknown:
            parser.error(f"no such case: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in wanted]
    if args.list:
        for c in cases:
            print(f"{c.id:34} {'sandbox ' if c.sandbox else ''}{c.question}")
        return

    if args.context_filter:
        cfg.server.context_filter = True
    if not args.rejudge:
        use_mock_database()
        cfg.agent.skills_dir = SKILL_PATH.parent.parent
        if args.no_sandbox:
            cfg.sandbox.provider = "none"
            cfg.sandbox.packages = []
        if cfg.sandbox.provider == "none":
            skipped = [c.id for c in cases if c.sandbox]
            cases = [c for c in cases if not c.sandbox]
            if skipped:
                print(f"no sandbox: skipping {', '.join(skipped)}")

    old = json.loads((RUNS / args.rejudge / "run.json").read_text(encoding="utf-8")) if args.rejudge else None
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / run_id
    meta = {
        "id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": old["model"] if old else cfg.agent.model,
        "judge_model": None if args.no_judge else args.judge_model,
        "sandbox": old["sandbox"] if old else cfg.sandbox.provider != "none",
        "rejudged_from": args.rejudge,
        "context_filter": old.get("context_filter", False) if old else cfg.server.context_filter,
        "cases": [c.to_dict() for c in cases],
    }
    judges = None if args.no_judge else Judges(args.judge_model)
    if args.rejudge:
        results = await rejudge(args.rejudge, cases, run_dir, meta, judges)
    else:
        results = await run(cases, run_dir, meta, judges)
    save(run_dir, meta, results)

    count = lambda key: sum(r[key]["status"] == "pass" for r in results)  # noqa: E731
    print(f"\n{count('correct')}/{len(results)} correct, {count('grounded')}/{len(results)} grounded, "
          f"{count('instruction_following')}/{len(results)} followed instructions, "
          f"{count('execution_strategy')}/{len(results)} sound strategy -> {run_dir.relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
