"""One agent turn as a stream of events for the UI.

    {"type": "thinking",    "text": ...}                        text the model wrote before acting
    {"type": "tool_call",   "id", "name", "args"}               a tool it called
    {"type": "tool_result", "id", "name", "content", "error"}   what the tool returned (clipped)
    {"type": "answer",      "text": ...}                        the final answer, markdown
    {"type": "error",       "message": ...}                     the turn failed
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

MAX_STEPS = 40
"""LangGraph's recursion limit for one turn."""

RESULT_CHARS = 2000
"""Tool results are clipped to this for display; the agent saw them in full."""


def text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if isinstance(b, dict) and b.get("type") == "text")


def _messages(payload: Any) -> list:
    if not isinstance(payload, dict):
        return []
    messages = payload.get("messages") or []
    return list(messages) if isinstance(messages, (list, tuple)) else [messages]


def _clip(text: str) -> str:
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + f"\n… ({len(text):,} characters)"


async def run_turn(agent, thread_id: str, message: str) -> AsyncIterator[dict]:
    """Run one user message through `agent` in conversation `thread_id`."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_STEPS}
    answer = ""
    try:
        async for update in agent.astream(
            {"messages": [{"role": "user", "content": message}]}, config, stream_mode="updates"
        ):
            for payload in update.values():
                for m in _messages(payload):
                    if isinstance(m, AIMessage):
                        text = text_of(m).strip()
                        if not m.tool_calls:
                            answer = text or answer
                            continue
                        if text:
                            yield {"type": "thinking", "text": text}
                        for call in m.tool_calls:
                            yield {"type": "tool_call", "id": call["id"], "name": call["name"], "args": call["args"]}
                    elif isinstance(m, ToolMessage):
                        content = m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
                        yield {
                            "type": "tool_result", "id": m.tool_call_id, "name": m.name,
                            "content": _clip(content), "error": m.status == "error",
                        }
    except GraphRecursionError:
        yield {"type": "error", "message": f"stopped after {MAX_STEPS} steps without an answer"}
        return
    yield {"type": "answer", "text": answer}


def sse(event: dict) -> str:
    """One Server-Sent Event: the event type as `event:`, the rest as JSON `data:`."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
