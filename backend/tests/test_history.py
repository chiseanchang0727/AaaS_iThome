"""Conversation history as JSONL: what a turn writes, and reading it back."""

import json

from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage

from api import HistoryStore
from test_api import TOOL_TURN, Harness, chat


def store(tmp_path) -> HistoryStore:
    return HistoryStore(tmp_path / "conversations", clock=lambda: "2026-09-30T10:00:00Z")


def test_a_turn_writes_one_line_per_message(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, [TOOL_TURN], history=history).app) as client:
        thread_id = chat(client, "How many rows?")[0]["thread_id"]

    ts = "2026-09-30T10:00:00Z"
    assert history.read(thread_id) == [
        {"turn": 1, "role": "user", "content": "How many rows?", "ts": ts},
        {"turn": 1, "role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "c1", "name": "query_database", "args": {"sql": "SELECT 1"}}], "ts": ts},
        {"turn": 1, "role": "tool", "tool_call_id": "c1", "name": "query_database",
         "content": '[{"n": 1}]', "error": False, "ts": ts},
        {"turn": 1, "role": "assistant", "content": "There is **1** row.", "ts": ts},
    ]


def test_the_file_is_plain_jsonl(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, [TOOL_TURN], history=history).app) as client:
        thread_id = chat(client, "你好")[0]["thread_id"]

    lines = history.path(thread_id).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(isinstance(json.loads(line), dict) for line in lines)
    assert "你好" in lines[0]  # readable as-is, not \\u-escaped


def test_follow_ups_are_numbered_turns_in_the_same_file(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, history=history).app) as client:
        thread_id = chat(client, "first")[0]["thread_id"]
        chat(client, "second", thread_id)
        other = chat(client, "elsewhere")[0]["thread_id"]

    users = [(r["turn"], r["content"]) for r in history.read(thread_id) if r["role"] == "user"]
    assert users == [(1, "first"), (2, "second")]
    assert [r["turn"] for r in history.read(other)] == [1, 1]


def test_tool_results_are_kept_in_full(tmp_path):
    history = store(tmp_path)
    turn = [{"tools": {"messages": [ToolMessage("x" * 10_000, tool_call_id="c1", name="execute")]}}]
    with TestClient(Harness(tmp_path, [turn], history=history).app) as client:
        thread_id = chat(client)[0]["thread_id"]

    tool = next(r for r in history.read(thread_id) if r["role"] == "tool")
    assert tool["content"] == "x" * 10_000


def test_a_half_written_line_is_skipped_and_numbering_goes_on(tmp_path):
    history = store(tmp_path)
    history.start_turn("t1", "first")
    with history.path("t1").open("a") as f:
        f.write('{"turn": 1, "role": "assist')  # the process died here

    assert [r["content"] for r in history.read("t1")] == ["first"]
    history.start_turn("t1", "second")
    assert [r["turn"] for r in history.read("t1")] == [1, 2]


def test_an_unknown_conversation_has_no_history(tmp_path):
    assert store(tmp_path).read("nope") == []
