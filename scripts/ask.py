"""Ask the agent one question and print every step it takes.

    uv run --env-file .env python scripts/ask.py "Which category gets the most views?"
    uv run --env-file .env python scripts/ask.py --no-require-skill "..."

Each model turn prints as THINK (text it wrote) and ACT (tool calls); each
tool result prints as OBSERVE. That loop is the ReAct cycle.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import build_agent  # noqa: E402
from datasources import close_database  # noqa: E402


def clip(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " …"


def text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if b.get("type") == "text")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-require-skill", action="store_true")
    args = parser.parse_args()

    agent = build_agent(require_sql_skill=not args.no_require_skill)
    step = 0
    answer = ""
    try:
        async for update in agent.astream(
            {"messages": [{"role": "user", "content": args.question}]},
            stream_mode="updates",
        ):
            for node in update.values():
                for message in (node or {}).get("messages", []) if isinstance(node, dict) else []:
                    if isinstance(message, AIMessage):
                        step += 1
                        if text := text_of(message).strip():
                            answer = text
                            if message.tool_calls:
                                print(f"[{step}] THINK   {clip(text)}")
                        for call in message.tool_calls:
                            print(f"[{step}] ACT     {call['name']}({clip(call['args'], 600)})")
                    elif isinstance(message, ToolMessage):
                        print(f"    OBSERVE {clip(message.content)}")
    finally:
        await close_database()

    print("\n=== ANSWER ===\n" + answer)


if __name__ == "__main__":
    asyncio.run(main())
