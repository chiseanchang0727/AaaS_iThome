"""What the agent did in one turn, read back from its conversation's history.

The history file (api/history.py) already holds every model call, tool call
and result, code step (`sandbox_step`) and sandbox event, written by the app
itself. A trace is that file cut down to one turn, with nothing logged twice.
It needs nothing else, so a saved run can be checked and judged again later.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolStep:
    id: str
    name: str
    args: dict[str, Any]
    result: str | None = None
    """None when the turn stopped before the call returned."""
    error: bool = False
    """The tool said so: an error status, or query_database's "ERROR:" result."""


@dataclass
class Trace:
    question: str
    turn: int
    answer: str
    """The last assistant message without tool calls; empty if the turn stopped."""
    tools: list[ToolStep] = field(default_factory=list)
    code_steps: list[dict[str, Any]] = field(default_factory=list)
    """This turn's `sandbox_step` lines: command, exit_code, peak_memory_mb, signal, ..."""
    events: list[dict[str, Any]] = field(default_factory=list)
    """This turn's `sandbox_event` lines: sandbox_ready, upgrade_started, switched, ..."""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    runtime_seconds: float = 0.0
    earlier_observations: list[str] = field(default_factory=list)
    """Tool results from earlier turns: the agent had seen them too."""
    earlier_turns: list[dict[str, Any]] = field(default_factory=list)
    """Earlier questions and answers: {"turn", "question", "answer", "sent"}, oldest first.
    `sent`: whether the agent was given that turn (always, without a context filter)."""
    context: dict[str, Any] | None = None
    """The turn's `context` line when Jev picked the earlier turns: sent_turns, earlier_turns, seconds."""
    artifacts: list[str] = field(default_factory=list)
    """Files the turn handed to the user, relative to the conversation's artifact folder."""
    records: list[dict[str, Any]] = field(default_factory=list)
    """This turn's history lines, in order."""

    @property
    def observations(self) -> list[str]:
        """Everything a tool returned that the agent saw by the end of the turn."""
        return self.earlier_observations + [t.result for t in self.tools if t.result is not None]


def _seconds(start: str, end: str) -> float:
    parse = lambda ts: datetime.fromisoformat(ts.replace("Z", "+00:00"))  # noqa: E731
    return round((parse(end) - parse(start)).total_seconds(), 2)


def is_error(name: str, content: str, flagged: bool) -> bool:
    return flagged or (name == "query_database" and content.startswith("ERROR:"))


def build_trace(records: list[dict[str, Any]], turn: int, artifacts: list[str] | None = None) -> Trace:
    """The trace of turn `turn` (1-based) in a conversation's history lines."""
    mine = [r for r in records if r.get("turn") == turn]
    question = next((r["content"] for r in mine if r.get("role") == "user"), "")
    trace = Trace(question=question, turn=turn, answer="", artifacts=list(artifacts or []), records=mine)
    trace.context = next(
        ({k: v for k, v in r.items() if k not in ("id", "previous", "turn", "role", "ts")}
         for r in mine if r.get("role") == "context"),
        None,
    )
    # With a context filter, the agent saw only the earlier turns it was sent.
    sent = set(trace.context["sent_turns"]) if trace.context else None
    seen = lambda t: 0 < t < turn and (sent is None or t in sent)  # noqa: E731
    trace.earlier_observations = [
        r["content"] for r in records if r.get("role") == "tool" and seen(r.get("turn") or 0)
    ]

    for t in sorted({r.get("turn") or 0 for r in records if 0 < (r.get("turn") or 0) < turn}):
        earlier = [r for r in records if r.get("turn") == t]
        trace.earlier_turns.append({
            "turn": t,
            "sent": seen(t),
            "question": next((r["content"] for r in earlier if r.get("role") == "user"), ""),
            "answer": next((r.get("content", "") for r in reversed(earlier)
                            if r.get("role") == "assistant" and not r.get("tool_calls")), ""),
        })

    by_id: dict[str, ToolStep] = {}
    for r in mine:
        role = r.get("role")
        if role == "assistant":
            trace.model_calls += 1
            usage = r.get("usage") or {}
            trace.input_tokens += usage.get("input_tokens", 0)
            trace.output_tokens += usage.get("output_tokens", 0)
            calls = r.get("tool_calls") or []
            for c in calls:
                step = ToolStep(c["id"], c["name"], c.get("args") or {})
                by_id[step.id] = step
                trace.tools.append(step)
            if not calls:
                trace.answer = r.get("content", "").strip()
        elif role == "tool":
            step = by_id.get(r.get("tool_call_id", ""))
            if step is not None:
                step.result = r.get("content", "")
                step.error = is_error(step.name, step.result, bool(r.get("error")))
        elif role == "sandbox_step":
            trace.code_steps.append(r)
        elif role == "sandbox_event":
            trace.events.append(r)

    # A turn that ends on a tool result never answered, even if an earlier
    # message without calls exists (it would be thinking out loud, then acting).
    last = next((r for r in reversed(mine) if r.get("role") in ("assistant", "tool")), None)
    if last is None or last.get("role") != "assistant" or last.get("tool_calls"):
        trace.answer = ""

    stamps = [r["ts"] for r in mine if r.get("ts")]
    if len(stamps) >= 2:
        trace.runtime_seconds = _seconds(stamps[0], stamps[-1])
    return trace
