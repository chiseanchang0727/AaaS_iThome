"""The context filter, with a scripted TypeSafe client."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.context_filter import CRITERIA, QUESTION, ContextFilter, build_state, group_turns, skill_reads
from api import HistoryStore


class FakeClient:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    async def system_one(self, state, questions, model=None):
        self.requests.append({"state": state, "questions": questions, "model": model})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def nouls(**yes):
    return SimpleNamespace(nouls={name: SimpleNamespace(noul=p) for name, p in yes.items()})


def run(coro):
    return asyncio.run(coro)


def history(tmp_path) -> list[dict]:
    """Two turns: one that queried, one that answered from memory."""
    store = HistoryStore(tmp_path, clock=lambda: "t")
    first = store.start_turn("c", "Which category has the highest median?")
    first.record(AIMessage("", tool_calls=[{"id": "c1", "name": "query_database", "args": {"sql": "..."}, "type": "tool_call"}]))
    first.record(ToolMessage('[{"category": "Kayaking"}]', tool_call_id="c1", name="query_database"))
    first.record(AIMessage("Kayaking, 824,606."))
    store.start_turn("c", "Thanks!").record(AIMessage("You're welcome."))
    return store.read("c")


def test_turns_are_grouped_by_number(tmp_path):
    turns = group_turns(history(tmp_path))
    assert list(turns) == [1, 2]
    assert [r["role"] for r in turns[1]] == ["user", "assistant", "tool", "assistant"]


def test_jev_reads_each_turns_question_and_final_answer_only(tmp_path):
    state = build_state("How many videos?", group_turns(history(tmp_path)))
    assert state == {
        "current_request": "How many videos?",
        "earlier_turns": [
            {"turn": 1, "user": "Which category has the highest median?", "assistant": "Kayaking, 824,606."},
            {"turn": 2, "user": "Thanks!", "assistant": "You're welcome."},
        ],
    }


def test_one_request_asks_about_every_turn(tmp_path):
    client = FakeClient(nouls(turn_1=0.9, turn_2=0.1))
    run(ContextFilter(client=client, keep_last=0).pick(history(tmp_path), "How many videos?"))

    [request] = client.requests
    assert set(request["questions"]) == {"turn_1", "turn_2"}
    q = request["questions"]["turn_1"]
    assert q.instructions == QUESTION.format(name="turn_1", description="`earlier_turns[0]`")
    assert q.criteria == CRITERIA


def test_only_picked_turns_are_sent_whole(tmp_path):
    client = FakeClient(nouls(turn_1=0.9, turn_2=0.1))
    messages, picked = run(ContextFilter(client=client, keep_last=0).build(history(tmp_path), "How many videos?"))

    assert picked == [1]
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage, HumanMessage]
    assert messages[0].content == "Which category has the highest median?"
    assert messages[-1].content == "How many videos?"


def test_threshold_decides(tmp_path):
    client = FakeClient(nouls(turn_1=0.6, turn_2=0.4))
    assert run(ContextFilter(client=client, threshold=0.5, keep_last=0).pick(history(tmp_path), "x")) == [1]


def test_no_history_asks_nothing():
    client = FakeClient()
    assert run(ContextFilter(client=client).pick([], "hi")) == []
    assert client.requests == []


def test_a_failed_request_sends_every_turn(tmp_path):
    client = FakeClient(TimeoutError())
    assert run(ContextFilter(client=client).pick(history(tmp_path), "x")) == [1, 2]


def test_the_latest_turn_is_sent_even_when_jev_says_no(tmp_path):
    client = FakeClient(nouls(turn_1=0.1, turn_2=0.1))
    assert run(ContextFilter(client=client, max_rounds=1).pick(history(tmp_path), "x")) == [2]
    assert set(client.requests[0]["questions"]) == {"turn_1", "turn_2"}


def test_keep_last_covering_every_turn_asks_nothing(tmp_path):
    client = FakeClient()
    assert run(ContextFilter(client=client, keep_last=2).pick(history(tmp_path), "x")) == [1, 2]
    assert client.requests == []


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        ContextFilter(client=FakeClient(), keep_last=-1)
    with pytest.raises(ValueError):
        ContextFilter(client=FakeClient(), max_rounds=0)


# --- skill reads from turns that are not sent -----------------------------------


def read(id, path="/skills/query_database/SKILL.md"):
    return {"id": id, "name": "read_file", "args": {"file_path": path}, "type": "tool_call"}


def with_skill_read(tmp_path) -> list[dict]:
    """Turn 1 reads the skill alongside another call; turn 2 does not read it."""
    store = HistoryStore(tmp_path, clock=lambda: "t")
    first = store.start_turn("c", "How many rows?")
    first.record(AIMessage("", tool_calls=[read("r1"), {"id": "q1", "name": "list_datasets", "args": {}, "type": "tool_call"}]))
    first.record(ToolMessage("# The videos table ...", tool_call_id="r1", name="read_file"))
    first.record(ToolMessage("[videos]", tool_call_id="q1", name="list_datasets"))
    first.record(AIMessage("1,236 rows."))
    store.start_turn("c", "Thanks!").record(AIMessage("You're welcome."))
    return store.read("c")


def test_a_skill_read_in_a_dropped_turn_follows_the_new_message(tmp_path):
    client = FakeClient(nouls(turn_1=0.1, turn_2=0.1))
    messages, picked = run(ContextFilter(client=client, max_rounds=1).build(with_skill_read(tmp_path), "Top channel?"))

    assert picked == [2]
    *_, request, call, result = messages
    assert request.content == "Top channel?"
    assert call.tool_calls == [read("r1")]  # only the read, not the call made with it
    assert (result.tool_call_id, result.content) == ("r1", "# The videos table ...")


def test_a_skill_read_in_a_sent_turn_is_not_repeated(tmp_path):
    client = FakeClient(nouls(turn_1=0.9, turn_2=0.1))
    messages, _ = run(ContextFilter(client=client, max_rounds=1).build(with_skill_read(tmp_path), "Top channel?"))
    assert isinstance(messages[-1], HumanMessage)
    assert sum(isinstance(m, ToolMessage) and m.tool_call_id == "r1" for m in messages) == 1


def test_only_the_latest_successful_read_of_each_skill_is_kept():
    def assistant(*calls):
        return {"turn": 1, "role": "assistant", "content": "", "tool_calls": list(calls)}

    def result(id, error=False):
        return {"turn": 1, "role": "tool", "tool_call_id": id, "name": "read_file", "content": id, "error": error}

    records = [
        assistant(read("old")), result("old"),
        assistant(read("new")), result("new"),
        assistant(read("failed")), result("failed", error=True),
        assistant(read("other", "/skills/trend-report/SKILL.md")), result("other"),
        assistant({"id": "f", "name": "read_file", "args": {"file_path": "/data/notes.md"}, "type": "tool_call"}), result("f"),
    ]
    assert [r["content"] for r in skill_reads(records) if r["role"] == "tool"] == ["new", "other"]


# --- following references back --------------------------------------------------


class ByRequest:
    """Says yes to the turns listed for each request; no to everything else."""

    def __init__(self, needs: dict[str, set[int]]):
        self.needs, self.requests = needs, []

    async def system_one(self, state, questions, model=None):
        self.requests.append(state)
        wanted = self.needs.get(state["current_request"], set())
        return nouls(**{name: 0.9 if int(name.split("_")[1]) in wanted else 0.1 for name in questions})


def chain(tmp_path) -> list[dict]:
    """Turn 3 names nothing itself: it leans on 2, which leans on 1."""
    store = HistoryStore(tmp_path, clock=lambda: "t")
    for question, answer in [
        ("Which channel has the most views?", "Paper Crane Co."),
        ("What category is that channel in?", "Speedcubing."),
        ("How long does that category take to trend?", "3.5 days."),
        ("thanks!", "You're welcome."),
    ]:
        store.start_turn("c", question).record(AIMessage(answer))
    return store.read("c")


NEEDS = {
    "Median days for the category we talked about?": {3},
    "How long does that category take to trend?": {2},
    "What category is that channel in?": {1},
}


def test_a_chain_is_followed_back_to_its_start(tmp_path):
    client = ByRequest(NEEDS)
    picked = run(ContextFilter(client=client).pick(chain(tmp_path), "Median days for the category we talked about?"))
    assert picked == [1, 2, 3, 4]


def test_max_rounds_limits_how_far_back(tmp_path):
    records = chain(tmp_path)
    request = "Median days for the category we talked about?"
    assert run(ContextFilter(client=ByRequest(NEEDS), max_rounds=1).pick(records, request)) == [3, 4]
    assert run(ContextFilter(client=ByRequest(NEEDS), max_rounds=2).pick(records, request)) == [2, 3, 4]


def test_the_latest_turn_is_followed_back_only_when_needed(tmp_path):
    records = [r for r in chain(tmp_path) if r["turn"] <= 3]  # turn 3 is the latest

    small_talk = ByRequest(NEEDS)
    assert run(ContextFilter(client=small_talk).pick(records, "thanks!")) == [3]
    assert len(small_talk.requests) == 1  # nothing followed

    follow_up = ByRequest({**NEEDS, "And in hours?": {3}})
    assert run(ContextFilter(client=follow_up).pick(records, "And in hours?")) == [1, 2, 3]


def test_a_followed_turn_is_judged_with_its_answer_and_only_the_turns_before_it(tmp_path):
    client = ByRequest(NEEDS)
    run(ContextFilter(client=client, max_rounds=2).pick(chain(tmp_path), "Median days for the category we talked about?"))
    followed = next(s for s in client.requests if s["current_request"] == "How long does that category take to trend?")
    assert followed["current_answer"] == "3.5 days."
    assert [t["turn"] for t in followed["earlier_turns"]] == [1, 2]


def test_a_failed_later_round_keeps_what_was_found(tmp_path):
    class FailsAfterFirst(ByRequest):
        async def system_one(self, state, questions, model=None):
            if self.requests:
                raise TimeoutError
            return await super().system_one(state, questions, model)

    client = FailsAfterFirst(NEEDS)
    assert run(ContextFilter(client=client).pick(chain(tmp_path), "Median days for the category we talked about?")) == [3, 4]
