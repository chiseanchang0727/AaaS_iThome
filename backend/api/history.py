"""Conversation history on disk: one JSONL file per conversation, one line per message.

    {"id": "9f2c…", "previous": null,    "turn": 1, "role": "user", "content": "Which category has the most views?", "ts": "..."}
    {"id": "41ab…", "previous": "9f2c…", "turn": 1, "role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "query_database", "args": {...}}], "ts": "..."}
    {"id": "c07e…", "previous": "41ab…", "turn": 1, "role": "tool", "tool_call_id": "c1", "name": "query_database", "content": "...", "error": false, "ts": "..."}
    {"id": "5d19…", "previous": "c07e…", "turn": 1, "role": "assistant", "content": "Music has the most views: 4.2M.", "ts": "..."}

An assistant line also carries the model call's token counts, when the model
reported them: `"usage": {"input_tokens": 5120, "output_tokens": 88}`.

A code step run in the sandbox also gets a line with what it cost (role
`sandbox_step`: command, seconds, cpu_seconds, peak_memory_mb, ...), and what
happens to the sandbox gets one too (role `sandbox_event`: ready, moved to a
bigger one, ...); readers that rebuild messages skip both.

Every line has a unique `id` and the `id` of the line before it, `previous`
(null on the first line): the order survives without the file, e.g. as rows
in a database.

Each line is appended and flushed to disk as soon as the message exists, so a
crash loses at most the line being written. Tool results are kept in full; the
UI's clipping (events.RESULT_CHARS) does not apply here.

This is a record, not the agent's memory: the agent still remembers through
its checkpointer. `to_messages` turns a record back into messages an agent can
be given instead. One line maps to one row when this moves to a database.
"""

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .events import text_of

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class HistoryStore:
    """Reads and appends `<root>/<thread_id>.jsonl`.

    Thread ids are checked by the API (letters, digits, `_` and `-`), so they
    are safe as file names.
    """

    def __init__(
        self,
        root: Path,
        clock: Callable[[], str] = _now,
        new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.root = Path(root)
        self._clock = clock
        self._new_id = new_id
        self._lock = threading.Lock()
        """Lines can come from the event loop and from worker threads (sandbox
        steps): one at a time, so lines and the `previous` chain stay whole."""

    def path(self, thread_id: str) -> Path:
        return self.root / f"{thread_id}.jsonl"

    def thread_ids(self) -> list[str]:
        """Every conversation with a history file."""
        return sorted(p.stem for p in self.root.glob("*.jsonl")) if self.root.exists() else []

    def delete(self, thread_id: str) -> bool:
        """Remove a conversation's history file. Whether there was one."""
        with self._lock:
            path = self.path(thread_id)
            if not path.exists():
                return False
            path.unlink()
            return True

    def read(self, thread_id: str) -> list[dict[str, Any]]:
        """Every line of a conversation, oldest first. [] if it has none.

        A line that is not valid JSON (the one being written when the process
        died) is skipped.
        """
        path = self.path(thread_id)
        if not path.exists():
            return []
        records = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping unreadable line %d of %s", number, path)
        return records

    def start_turn(self, thread_id: str, message: str, account: str | None = None) -> "TurnLog":
        """Write the user's message as the first line of a new turn, with whose it is."""
        records = self.read(thread_id)
        turn = max((r.get("turn", 0) for r in records), default=0) + 1
        log = TurnLog(self, thread_id, turn, previous=records[-1].get("id") if records else None)
        log.write({"role": "user", "content": message, **({"account": account} if account else {})})
        return log

    def append(self, thread_id: str, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        path = self.path(thread_id)
        if path.exists() and path.stat().st_size and not _ends_with_newline(path):
            line = "\n" + line  # a crash cut the last line short; don't glue onto it
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


class TurnLog:
    """Appends one turn's messages. `record` takes what the agent streams."""

    def __init__(self, store: HistoryStore, thread_id: str, turn: int, previous: str | None = None) -> None:
        self.store, self.thread_id, self.turn = store, thread_id, turn
        self.previous = previous
        """The id of the last line written, which the next line points back to."""

    def write(self, fields: dict[str, Any]) -> None:
        with self.store._lock:
            line_id = self.store._new_id()
            self.store.append(self.thread_id, {
                "id": line_id, "previous": self.previous, "turn": self.turn, **fields, "ts": self.store._clock(),
            })
            self.previous = line_id

    def record_step(self, measure: Any) -> None:
        """What a code step in the sandbox cost (sandboxes/metering.StepMeasure)."""
        self.write({"role": "sandbox_step", **measure.to_dict()})

    def record_event(self, event: str, fields: dict[str, Any]) -> None:
        """Something that happened to the conversation's sandbox (api/conversations.py)."""
        self.write({"role": "sandbox_event", "event": event, **fields})

    def record(self, message: BaseMessage) -> None:
        if isinstance(message, AIMessage):
            fields: dict[str, Any] = {"role": "assistant", "content": text_of(message)}
            if message.tool_calls:
                fields["tool_calls"] = [
                    {"id": c["id"], "name": c["name"], "args": c["args"]} for c in message.tool_calls
                ]
            if usage := message.usage_metadata:
                fields["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }
            self.write(fields)
        elif isinstance(message, ToolMessage):
            content = message.content
            self.write({
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "name": message.name,
                "content": content if isinstance(content, str) else json.dumps(content, default=str),
                "error": message.status == "error",
            })


UNANSWERED = "Not run: the turn stopped before this call returned."


def to_messages(records: list[dict[str, Any]]) -> list[BaseMessage]:
    """`HistoryStore.read` output as LangChain messages, oldest first.

    A tool call with no result line (the turn stopped or crashed before it
    returned) gets an error result, because a model API rejects a history
    with an unanswered tool call.
    """
    answered = {r["tool_call_id"] for r in records if r.get("role") == "tool"}
    messages: list[BaseMessage] = []
    for r in records:
        role = r.get("role")
        if role == "user":
            messages.append(HumanMessage(r["content"]))
        elif role == "assistant":
            calls = [
                {"id": c["id"], "name": c["name"], "args": c["args"], "type": "tool_call"}
                for c in r.get("tool_calls", [])
            ]
            messages.append(AIMessage(r["content"], tool_calls=calls))
            messages += [
                ToolMessage(UNANSWERED, tool_call_id=c["id"], name=c["name"], status="error")
                for c in calls
                if c["id"] not in answered
            ]
        elif role == "tool":
            messages.append(ToolMessage(
                r["content"],
                tool_call_id=r["tool_call_id"],
                name=r.get("name"),
                status="error" if r.get("error") else "success",
            ))
    return messages
