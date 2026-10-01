"""Does the agent get worse when Jev picks which earlier turns it sees?

    uv run --env-file ../.env python -m evals.memory.run                   # both arms, with sandbox
    uv run --env-file ../.env python -m evals.memory.run --only drilldown --runs 2
    uv run --env-file ../.env python -m evals.memory.run --no-sandbox
    uv run python -m evals.memory.conversations                            # print the conversations

Two arms, same conversations:
  checkpointer  today: LangGraph's checkpointer carries the whole conversation
  jev           no checkpointer: each turn gets only the earlier turns Jev
                picks from the JSONL history (agent/context_filter.py), plus
                the new message. Needs $TYPESAFE_API_KEY.

Needs the grounding eval's mock database (see evals/grounding/run.py). Each
conversation gets one agent and one sandbox for all its turns, like the API.
Scores per turn, as in the grounding eval, plus what the turn cost:
  correct     the expected values appear in the answer
  grounded    every number in the answer appeared in a tool result so far
  steps       tool calls; queries and skill reads counted apart
  tokens      input tokens across the turn's model calls
  1st call    input tokens of the turn's first model call: the context the
              turn starts with (prompt, tools, history sent, new message),
              before any step the agent takes adds to it
Each invocation is saved as out/runs/<run id>/: run.json (settings, the
conversations, every turn's result) and history/ (one JSONL per conversation,
arm and repeat). The Evals page in the app reads them through api/evals.py.
"""

import argparse
import asyncio
import contextlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from agent import build_agent
from agent.context_filter import ContextFilter
from api.events import text_of
from api.history import HistoryStore
from config import cfg
from datasources import close_database
from evals.grounding.mock_data import SKILL_PATH
from evals.grounding.run import MAX_STEPS, score, use_mock_database
from sandboxes import get_provider

from .conversations import Conversation, build_conversations, expectation

OUT = Path(__file__).resolve().parent / "out"
RUNS = OUT / "runs"
ARMS = ("checkpointer", "jev")


async def run_conversation(
    conv: Conversation, arm: str, sandbox, run: int, history: HistoryStore, context: ContextFilter | None
) -> list[dict]:
    agent = build_agent(sandbox=sandbox, checkpointer=InMemorySaver() if arm == "checkpointer" else None)
    thread_id = f"{conv.id}-{arm}-run{run}"
    history.path(thread_id).unlink(missing_ok=True)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_STEPS}

    results = []
    for q in conv.turns:
        start_with, sent = [HumanMessage(q.prompt)], None
        if arm == "jev":
            start_with, sent = await context.build(history.read(thread_id), q.prompt)
        log = history.start_turn(thread_id, q.prompt)
        state, stopped = None, False
        try:
            async for state in agent.astream(
                {"messages": start_with}, config, stream_mode="values"
            ):
                pass
        except GraphRecursionError:
            stopped = True

        messages = state["messages"] if state else []
        if arm == "jev":  # no checkpointer: the state is what we sent, then the turn
            new = messages[len(start_with):]
        else:
            new = messages[max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage)) + 1:]
        for m in new:
            log.record(m)

        finals = [m for m in new if isinstance(m, AIMessage) and not m.tool_calls]
        answer = text_of(finals[-1]).strip() if finals and not stopped else ""
        # Anything a tool returned earlier in the conversation counts as seen,
        # whichever arm: both had it in front of them.
        observations = [r["content"] for r in history.read(thread_id) if r["role"] == "tool"]
        calls = [c for m in new if isinstance(m, AIMessage) for c in m.tool_calls]
        results.append({
            "conversation": conv.id, "turn": q.id, "arm": arm, "run": run, "prompt": q.prompt,
            "answer": answer, "stopped": stopped, "sent_turns": sent,
            "steps": len(calls),
            "queries": sum(c["name"] == "query_database" for c in calls),
            "skill_reads": sum(c["name"] == "read_file" and str(c["args"].get("file_path", "")).endswith("SKILL.md") for c in calls),
            "input_tokens": sum(_input_tokens(m) for m in new if isinstance(m, AIMessage)),
            "first_call_tokens": next((_input_tokens(m) for m in new if isinstance(m, AIMessage)), 0),
            **score(q, answer, observations),
        })
    return results


def _input_tokens(message: AIMessage) -> int:
    return (message.usage_metadata or {}).get("input_tokens", 0)


def summarize(results: list[dict]) -> str:
    by_arm = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)
    header = (f"{'arm':13} {'correct':>9} {'grounded':>9} {'steps':>6} {'queries':>8} {'skill reads':>12} "
              f"{'tokens/turn':>12} {'1st call/turn':>14}")
    lines = [header]
    for arm, rs in by_arm.items():
        checked = [r for r in rs if r["correct"] is not None]
        lines.append(
            f"{arm:13} {sum(r['correct'] for r in checked):>4}/{len(checked):<4} "
            f"{sum(not r['ungrounded'] for r in rs):>4}/{len(rs):<4} "
            f"{sum(r['steps'] for r in rs):>6} {sum(r['queries'] for r in rs):>8} "
            f"{sum(r['skill_reads'] for r in rs):>12} {sum(r['input_tokens'] for r in rs) // max(1, len(rs)):>12,} "
            f"{sum(r['first_call_tokens'] for r in rs) // max(1, len(rs)):>14,}"
        )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated conversation ids")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated: " + ", ".join(ARMS))
    parser.add_argument("--no-sandbox", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="repeat everything this many times")
    args = parser.parse_args()

    use_mock_database()
    cfg.agent.skills_dir = SKILL_PATH.parent.parent
    cfg.sandbox.provider = "none" if args.no_sandbox else "daytona"
    if args.no_sandbox:
        cfg.sandbox.packages = []
    provider = get_provider(cfg.sandbox)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / run_id
    history = HistoryStore(run_dir / "history")

    conversations = build_conversations()
    if args.only:
        wanted = set(args.only.split(","))
        conversations = [c for c in conversations if c.id in wanted]
    arms = args.arms.split(",")
    settings = cfg.agent.context_filter
    context = None
    if "jev" in arms:
        if settings is None:
            parser.error("the jev arm needs agent.context_filter in config.yml")
        context = ContextFilter(model=settings.model, threshold=settings.threshold,
                                keep_last=settings.keep_last, max_rounds=settings.max_rounds)

    meta = {
        "id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": cfg.agent.model,
        "sandbox": not args.no_sandbox,
        "repeats": args.runs,
        "arms": arms,
        "context_filter": settings.model_dump() if settings is not None and "jev" in arms else None,
        "conversations": [
            {"id": c.id, "about": c.about,
             "turns": [{"id": q.id, "prompt": q.prompt, "expect": expectation(q), "note": q.note} for q in c.turns]}
            for c in conversations
        ],
    }
    results = []
    try:
        for run in range(1, args.runs + 1):
            for conv in conversations:
                for arm in arms:
                    with provider.session() if provider else contextlib.nullcontext() as sandbox:
                        turns = await run_conversation(conv, arm, sandbox, run, history, context)
                    results += turns
                    for r in turns:
                        verdict = {True: "correct", False: "WRONG", None: "check by hand"}[r["correct"]]
                        sent = "" if r["sent_turns"] is None else f"sent={r['sent_turns']} "
                        print(f"run {run}  {r['turn']:18} {arm:13} {verdict:8} steps={r['steps']:<2} "
                              f"1st={r['first_call_tokens']:<6,} {sent}ungrounded={r['ungrounded'] or '-'}", flush=True)
    finally:
        await close_database()
        # Saved even when a run stops early, so the turns that ran can be seen.
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps({**meta, "results": results}, indent=2, ensure_ascii=False))

    print("\n" + summarize(results))
    print(f"-> {run_dir.relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
