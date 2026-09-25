"""Run one article stage against the same question and print its trace.

    uv run --env-file .env python scripts/stages.py bare|tools|skills|forced ["question"]

bare    model + system prompt only
tools   + query_database
skills  + /skills/ mounted; the agent decides whether to read them
forced  + SkillEnforcerMiddleware: no SQL until query_database/SKILL.md has been read
"""

import asyncio
import json
import sys
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import SYSTEM_PROMPT, build_agent  # noqa: E402
from agent.tools import query_database  # noqa: E402
from config import cfg  # noqa: E402
from datasources import close_database  # noqa: E402

QUESTION = "Which video category gets the most views?"
MAX_STEPS = 25
"""LangGraph's recursion limit: graph steps before a run is stopped."""

STAGES = {
    "bare": lambda: create_deep_agent(model=cfg.agent.model, system_prompt=SYSTEM_PROMPT),
    "tools": lambda: create_deep_agent(
        model=cfg.agent.model, tools=[query_database], system_prompt=SYSTEM_PROMPT
    ),
    "skills": lambda: build_agent(require_sql_skill=False),
    "forced": lambda: build_agent(require_sql_skill=True),
}


def clip(text: str, limit: int = 500) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " …"


def text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if b.get("type") == "text")


async def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: stages.py {'|'.join(STAGES)} [question]")
    stage, question = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else QUESTION)

    # Stream rather than invoke, so a run stopped at the step limit still
    # leaves the messages it produced up to that point.
    state, stopped = {"messages": []}, False
    try:
        async for state in STAGES[stage]().astream(
            {"messages": [{"role": "user", "content": question}]},
            {"recursion_limit": MAX_STEPS},
            stream_mode="values",
        ):
            pass
    except GraphRecursionError:
        stopped = True
    finally:
        await close_database()
    result = state

    print(f"Q: {question}\n")
    turns, tokens_in, tokens_out = 0, 0, 0
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            turns += 1
            usage = message.usage_metadata or {}
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
            text = text_of(message).strip()
            if not message.tool_calls:
                print(f"\n=== ANSWER ===\n{text}")
                continue
            if text:
                print(f"[{turns}] THINK   {clip(text)}")
            for call in message.tool_calls:
                args = json.dumps(call["args"], ensure_ascii=False)
                print(f"[{turns}] ACT     {call['name']}({clip(args, 900)})")
        elif isinstance(message, ToolMessage):
            print(f"    OBSERVE {clip(message.content)}")
    if stopped:
        print(f"\n=== STOPPED: hit the {MAX_STEPS}-step limit without answering ===")
    print(f"\n--- model turns: {turns}, tokens in/out: {tokens_in}/{tokens_out}")


if __name__ == "__main__":
    asyncio.run(main())
