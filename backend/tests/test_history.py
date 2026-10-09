"""Conversation history as JSONL: what a turn writes, and reading it back."""

import itertools
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from api import HistoryStore
from api.history import UNANSWERED, to_messages
from test_api import TOOL_TURN, Harness, chat


def store(tmp_path) -> HistoryStore:
    ids = (f"id{n}" for n in itertools.count(1))
    return HistoryStore(tmp_path / "conversations", clock=lambda: "2026-09-30T10:00:00Z", new_id=lambda: next(ids))


def test_a_turn_writes_one_line_per_message(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, [TOOL_TURN], history=history).app) as client:
        thread_id = chat(client, "How many rows?")[0]["thread_id"]

    ts = "2026-09-30T10:00:00Z"
    assert history.read(thread_id) == [
        {"id": "id1", "previous": None, "turn": 1, "role": "user", "content": "How many rows?",
         "account": "test_user", "ts": ts},
        {"id": "id2", "previous": "id1", "turn": 1, "role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "c1", "name": "query_database", "args": {"sql": "SELECT 1"}}], "ts": ts},
        {"id": "id3", "previous": "id2", "turn": 1, "role": "tool", "tool_call_id": "c1", "name": "query_database",
         "content": '[{"n": 1}]', "error": False, "ts": ts},
        {"id": "id4", "previous": "id3", "turn": 1, "role": "assistant", "content": "There is **1** row.", "ts": ts},
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


def test_each_line_points_back_to_the_one_before_across_turns(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, history=history).app) as client:
        thread_id = chat(client, "first")[0]["thread_id"]
        chat(client, "second", thread_id)

    records = history.read(thread_id)
    assert [r["previous"] for r in records] == [None, *(r["id"] for r in records[:-1])]
    assert len({r["id"] for r in records}) == len(records)


def test_after_a_half_written_line_the_next_points_to_the_last_whole_one(tmp_path):
    history = store(tmp_path)
    history.start_turn("t1", "first")
    with history.path("t1").open("a") as f:
        f.write('{"id": "lost", "previous": "id1", "tur')
    history.start_turn("t1", "second")
    assert [(r["id"], r["previous"]) for r in history.read("t1")] == [("id1", None), ("id2", "id1")]


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


def test_to_messages_rebuilds_what_the_agent_saw(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, [TOOL_TURN], history=history).app) as client:
        thread_id = chat(client, "How many rows?")[0]["thread_id"]

    user, call, result, answer = to_messages(history.read(thread_id))
    assert isinstance(user, HumanMessage) and user.content == "How many rows?"
    assert isinstance(call, AIMessage)
    assert call.tool_calls == [{"id": "c1", "name": "query_database", "args": {"sql": "SELECT 1"}, "type": "tool_call"}]
    assert isinstance(result, ToolMessage)
    assert (result.tool_call_id, result.name, result.status) == ("c1", "query_database", "success")
    assert isinstance(answer, AIMessage) and answer.content == "There is **1** row." and not answer.tool_calls


def test_to_messages_answers_a_call_left_without_a_result(tmp_path):
    history = store(tmp_path)
    log = history.start_turn("t1", "go")
    log.record(AIMessage("", tool_calls=[{"id": "c9", "name": "execute", "args": {}, "type": "tool_call"}]))
    # the process died before the tool returned

    *_, filler = to_messages(history.read("t1"))
    assert isinstance(filler, ToolMessage)
    assert (filler.tool_call_id, filler.status, filler.content) == ("c9", "error", UNANSWERED)


def test_assistant_lines_keep_the_model_calls_token_counts(tmp_path):
    history = store(tmp_path)
    log = history.start_turn("t1", "go")
    log.record(AIMessage("done", usage_metadata={"input_tokens": 5120, "output_tokens": 88, "total_tokens": 5208}))
    log.record(AIMessage("no usage reported"))

    _, with_usage, without = history.read("t1")
    assert with_usage["usage"] == {"input_tokens": 5120, "output_tokens": 88}
    assert "usage" not in without


def test_past_conversations_are_listed_and_read_by_their_account(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, [TOOL_TURN], history=history).app) as client:
        thread_id = chat(client, "How many rows?")[0]["thread_id"]
        [listed] = client.get("/api/conversations").json()
        assert (listed["id"], listed["title"], listed["turns"]) == (thread_id, "How many rows?", 1)
        one = client.get(f"/api/conversations/{thread_id}").json()
        chat_lines = [line["role"] for line in one["lines"] if not line["role"].startswith("sandbox_")]
        assert chat_lines == ["user", "assistant", "tool", "assistant"]
        assert one["files"] == []
        assert client.get("/api/conversations", headers={"X-Account": "alice"}).json() == []
        assert client.get(f"/api/conversations/{thread_id}", headers={"X-Account": "alice"}).status_code == 404
        assert client.get("/api/conversations/nope").status_code == 404


def test_a_past_conversation_can_be_deleted_for_good_by_its_account(tmp_path):
    history = store(tmp_path)
    harness = Harness(tmp_path, [TOOL_TURN], history=history)
    with TestClient(harness.app) as client:
        thread_id = chat(client, "How many rows?")[0]["thread_id"]
        (harness.artifacts / thread_id).mkdir(parents=True, exist_ok=True)
        (harness.artifacts / thread_id / "chart.html").write_text("<p>chart</p>")

        assert client.delete(f"/api/conversations/{thread_id}/history", headers={"X-Account": "alice"}).status_code == 404
        assert client.delete(f"/api/conversations/{thread_id}/history").json() == {"deleted": thread_id}
        assert not history.path(thread_id).exists() and not (harness.artifacts / thread_id).exists()
        assert client.get("/api/conversations").json() == []
        assert client.delete(f"/api/conversations/{thread_id}/history").status_code == 404


def test_ending_a_conversation_keeps_its_history(tmp_path):
    history = store(tmp_path)
    with TestClient(Harness(tmp_path, [TOOL_TURN], history=history).app) as client:
        thread_id = chat(client, "How many rows?")[0]["thread_id"]
        client.delete(f"/api/conversations/{thread_id}")
        assert history.path(thread_id).exists()
