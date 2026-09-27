"""Does the agent answer from the data? Run it on mock data and check.

    uv run --env-file ../.env python -m evals.grounding.run            # all, with sandbox
    uv run --env-file ../.env python -m evals.grounding.run --only counts,revenue
    uv run --env-file ../.env python -m evals.grounding.run --no-sandbox
    uv run --env-file ../.env python -m evals.grounding.run --runs 2   # repeat the set

Needs the mock data loaded into an `ithome_mock` database that `agent_ro` can
read (same server as $AGENT_DATABASE_URL). One-off, from backend/:

    uv run python -m evals.grounding.mock_data
    # then, as the database owner: CREATE DATABASE ithome_mock; run
    # datasources/schema.sql in it; \\copy videos (...) FROM the CSV;
    # GRANT CONNECT/USAGE/SELECT to agent_ro.

Each question gets a fresh agent (and sandbox). Scores:
  correct   the expected values appear in the answer
  grounded  every number in the answer appeared in a tool result first
Transcripts go to evals/grounding/out/transcripts/, sandbox files to out/outputs/.
"""

import argparse
import asyncio
import contextlib
import json
import os
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent import build_agent
from config import cfg
from datasources import close_database
from config.database import DSN_ENV
from sandboxes import download_outputs, get_provider

from .mock_data import OUT, SKILL_PATH
from .questions import Question, build_questions
from .scoring import contains_number, ungrounded

MOCK_DB = "ithome_mock"
MAX_STEPS = 40


def use_mock_database() -> None:
    """Point the agent's connection at the mock database, same server and role."""
    parts = urlsplit(os.environ[DSN_ENV])
    os.environ[DSN_ENV] = urlunsplit(parts._replace(path=f"/{MOCK_DB}"))


def text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if b.get("type") == "text")


def score(q: Question, answer: str, observations: list[str]) -> dict:
    low = answer.lower()
    missing = [t for t in q.expect_text if t.lower() not in low]
    if q.expect_any_text and not any(t.lower() in low for t in q.expect_any_text):
        missing.append(" or ".join(q.expect_any_text))
    missing += [
        f"{label}={value:,.4g}" for label, value in q.expect_numbers.items()
        if not contains_number(answer, value)
    ]
    checked = bool(q.expect_text or q.expect_numbers or q.expect_any_text)
    return {
        "correct": (not missing) if checked else None,
        "missing": missing,
        "ungrounded": ungrounded(answer, observations, q.prompt),
    }


async def ask(q: Question, sandbox, run: int) -> dict:
    agent = build_agent(sandbox=sandbox)
    state, stopped = {"messages": []}, False
    try:
        async for state in agent.astream(
            {"messages": [{"role": "user", "content": q.prompt}]},
            {"recursion_limit": MAX_STEPS},
            stream_mode="values",
        ):
            pass
    except GraphRecursionError:
        stopped = True

    messages = state["messages"]
    finals = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
    answer = text_of(finals[-1]).strip() if finals and not stopped else ""
    observations = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
    tools = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    tokens = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in messages if isinstance(m, AIMessage))

    transcript = OUT / "transcripts" / f"{q.id}-run{run}.md"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {q.id}\n\n**Q:** {q.prompt}\n"]
    for m in messages:
        if isinstance(m, AIMessage):
            if text := text_of(m).strip():
                lines.append(f"**AI:** {text}\n")
            lines += [f"**ACT** `{c['name']}` {json.dumps(c['args'], ensure_ascii=False)}\n" for c in m.tool_calls]
        elif isinstance(m, ToolMessage):
            lines.append(f"**OBSERVE**\n```\n{m.content}\n```\n")
    transcript.write_text("\n".join(lines))

    return {
        "id": q.id, "run": run, "answer": answer, "stopped": stopped, "tools": tools,
        "input_tokens": tokens, "note": q.note, **score(q, answer, observations),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--no-sandbox", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole set this many times")
    args = parser.parse_args()

    use_mock_database()
    cfg.agent.skills_dir = SKILL_PATH.parent.parent
    cfg.sandbox.provider = "none" if args.no_sandbox else "daytona"
    if args.no_sandbox:
        cfg.sandbox.packages = []
    provider = get_provider(cfg.sandbox)

    questions = build_questions()
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.id in wanted]

    results = []
    try:
        for run in range(1, args.runs + 1):
            for q in questions:
                with provider.session() if provider else contextlib.nullcontext() as sandbox:
                    result = await ask(q, sandbox, run)
                    if sandbox is not None:
                        download_outputs(sandbox, cfg.sandbox.output_dir, OUT / "outputs" / f"{q.id}-run{run}")
                results.append(result)
                verdict = {True: "correct", False: "WRONG", None: "check by hand"}[result["correct"]]
                print(f"run {run}  {q.id:15} {verdict:13} ungrounded={result['ungrounded'] or '-'}", flush=True)
    finally:
        await close_database()

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n{sum(r['correct'] is True for r in results)}/{sum(r['correct'] is not None for r in results)} correct, "
          f"{sum(not r['ungrounded'] for r in results)}/{len(results)} fully grounded "
          f"-> {(OUT / 'results.json').relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
