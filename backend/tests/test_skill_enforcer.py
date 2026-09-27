"""SkillEnforcerMiddleware, method by method, then wired into a real graph.

No model and no database: the graph test drives a scripted fake model against
a stub `query_database`, so it runs anywhere.
"""

import asyncio
from pathlib import Path

import pytest
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent.middleware import SkillEnforcerMiddleware

TOOL = "query_database"
SKILL = f"/skills/{TOOL}/SKILL.md"


# --- helpers ---------------------------------------------------------------


def call(name: str, args: dict | None = None, id: str = "c1") -> dict:
    return {"name": name, "args": args or {}, "id": id, "type": "tool_call"}


def query(id: str = "q1") -> dict:
    return call(TOOL, {"sql": "SELECT 1"}, id)


def read(path: str = SKILL, id: str = "r1", **extra) -> dict:
    return call("read_file", {"file_path": path, **extra}, id)


def ai(*tool_calls: dict, content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(tool_calls))


def result(id: str, content: str = "...", status: str = "success") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=id, status=status)


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """A skills dir with a SKILL.md for `query_database` and none for `ls`."""
    (tmp_path / TOOL).mkdir()
    (tmp_path / TOOL / "SKILL.md").write_text("---\nname: q\ndescription: d\n---\nrules\n")
    return tmp_path


@pytest.fixture
def enforcer(skills_dir: Path) -> SkillEnforcerMiddleware:
    return SkillEnforcerMiddleware(skills_dir, target_skills=TOOL)


def check(enforcer: SkillEnforcerMiddleware, *messages) -> dict | None:
    return enforcer._check_skills({"messages": [HumanMessage("q"), *messages]})


# --- __init__ ----------------------------------------------------------------


@pytest.mark.parametrize(
    "target_skills, expected",
    [
        (None, None),
        (TOOL, {TOOL}),
        ([TOOL, "save_analysis"], {TOOL, "save_analysis"}),
        ((TOOL,), {TOOL}),
    ],
)
def test_init_normalises_targets(skills_dir, target_skills, expected):
    assert SkillEnforcerMiddleware(skills_dir, target_skills)._targets == expected


def test_init_single_string_is_one_target_not_characters(skills_dir):
    """set("query_database") would be a set of letters."""
    assert SkillEnforcerMiddleware(skills_dir, "ab")._targets == {"ab"}


def test_init_accepts_a_str_dir(skills_dir):
    assert SkillEnforcerMiddleware(str(skills_dir)).skills_dir == skills_dir


def test_init_keeps_the_middleware_name(enforcer):
    """create_agent keys middleware by name; two with one name is an error."""
    assert enforcer.name == "SkillEnforcerMiddleware"


# --- _skill_path / _skill_virtual_path -----------------------------------------


def test_skill_path_is_on_local_disk(enforcer, skills_dir):
    assert enforcer._skill_path(TOOL) == skills_dir / TOOL / "SKILL.md"


def test_skill_virtual_path_is_the_backend_route():
    """Independent of skills_dir: it is what the agent sees, not where it lives."""
    enforcer = SkillEnforcerMiddleware(Path("/somewhere/else"), TOOL)
    assert enforcer._skill_virtual_path(TOOL) == SKILL


# --- _is_enforced ----------------------------------------------------------


def test_is_enforced_with_targets_uses_only_the_targets(skills_dir):
    enforcer = SkillEnforcerMiddleware(skills_dir, target_skills="ls")
    assert enforcer._is_enforced("ls")  # no SKILL.md on disk, still enforced
    assert not enforcer._is_enforced(TOOL)  # SKILL.md on disk, not a target


def test_is_enforced_without_targets_follows_the_disk(skills_dir):
    enforcer = SkillEnforcerMiddleware(skills_dir)
    assert enforcer._is_enforced(TOOL)
    assert not enforcer._is_enforced("ls")


def test_is_enforced_without_targets_needs_the_file_not_just_the_folder(skills_dir):
    (skills_dir / "ls").mkdir()
    assert not SkillEnforcerMiddleware(skills_dir)._is_enforced("ls")


# --- _get_path_arg ---------------------------------------------------------


@pytest.mark.parametrize(
    "args, expected",
    [
        ({"file_path": SKILL}, SKILL),
        ({"path": SKILL}, SKILL),
        ({"path": "", "file_path": SKILL}, SKILL),
        ({"path": SKILL, "file_path": "/other"}, SKILL),
        ({"limit": 100}, ""),
        ({}, ""),
    ],
)
def test_get_path_arg(enforcer, args, expected):
    assert enforcer._get_path_arg(args) == expected


# --- _skill_was_read ---------------------------------------------------------


def test_skill_was_read_false_on_empty_history(enforcer):
    assert not enforcer._skill_was_read(TOOL, [])


def test_skill_was_read_true_after_a_read(enforcer):
    assert enforcer._skill_was_read(TOOL, [ai(read()), result("r1")])


def test_skill_was_read_accepts_the_path_key(enforcer):
    messages = [ai(call("read_file", {"path": SKILL}, "r1")), result("r1")]
    assert enforcer._skill_was_read(TOOL, messages)


def test_skill_was_read_needs_the_result_not_just_the_call(enforcer):
    """The read issued in the turn being checked has no result yet."""
    assert not enforcer._skill_was_read(TOOL, [ai(read())])


def test_skill_was_read_ignores_a_result_for_another_call(enforcer):
    assert not enforcer._skill_was_read(TOOL, [ai(read(id="r1")), result("other")])


def test_skill_was_read_ignores_a_truncated_read(enforcer):
    partial = "@@ lines 1-3 of 55 | next offset 3 @@\n---\nname: q\n"
    assert not enforcer._skill_was_read(TOOL, [ai(read(limit=3)), result("r1", partial)])


def test_skill_was_read_counts_a_read_to_the_end(enforcer):
    full = "@@ lines 1-55 of 55 @@\n---\nname: q\n"
    assert enforcer._skill_was_read(TOOL, [ai(read()), result("r1", full)])


def test_skill_was_read_counts_a_full_read_after_a_truncated_one(enforcer):
    messages = [
        ai(read(id="r1", limit=3)),
        result("r1", "@@ lines 1-3 of 55 | next offset 3 @@\nx"),
        ai(read(id="r2")),
        result("r2", "@@ lines 1-55 of 55 @@\nx"),
    ]
    assert enforcer._skill_was_read(TOOL, messages)


def test_skill_was_read_is_not_fooled_by_the_phrase_in_the_skill_body(enforcer):
    """Only a status header marks truncation, not the words anywhere."""
    full = "@@ lines 1-5 of 5 @@\nSee next offset 3 in the docs."
    assert enforcer._skill_was_read(TOOL, [ai(read()), result("r1", full)])


def test_skill_was_read_ignores_other_skills(enforcer):
    messages = [ai(read("/skills/trend-report/SKILL.md")), result("r1")]
    assert not enforcer._skill_was_read(TOOL, messages)


def test_skill_was_read_ignores_other_tools_on_the_path(enforcer):
    messages = [ai(call("grep", {"path": SKILL}, "g1")), result("g1")]
    assert not enforcer._skill_was_read(TOOL, messages)


def test_skill_was_read_ignores_messages_without_tool_calls(enforcer):
    messages = [HumanMessage(SKILL), AIMessage(content=f"I should read {SKILL}"), result("x", SKILL)]
    assert not enforcer._skill_was_read(TOOL, messages)


def test_skill_was_read_finds_a_read_further_back(enforcer):
    messages = [ai(read()), result("r1"), ai(call("ls", id="l1")), result("l1")]
    assert enforcer._skill_was_read(TOOL, messages)


def test_skill_was_read_needs_the_exact_path(enforcer):
    assert not enforcer._skill_was_read(TOOL, [ai(read(SKILL + ".bak")), result("r1")])


def test_skill_was_read_ignores_a_read_that_failed(enforcer):
    messages = [ai(read()), result("r1", "Error: file not found", status="error")]
    assert not enforcer._skill_was_read(TOOL, messages)


# --- _block_message / _skipped_message ---------------------------------------


def test_block_message(enforcer):
    msg = enforcer._block_message(TOOL, "q1")
    assert isinstance(msg, ToolMessage)
    assert (msg.tool_call_id, msg.name, msg.status) == ("q1", TOOL, "error")
    assert f"path='{SKILL}'" in msg.content  # the exact path the model must pass
    assert "whole file" in msg.content


def test_skipped_message(enforcer):
    msg = enforcer._skipped_message("ls", "l1")
    assert (msg.tool_call_id, msg.name, msg.status) == ("l1", "ls", "error")
    assert "'ls'" in msg.content
    assert SKILL not in msg.content  # it was not this call's fault


# --- _check_skills -----------------------------------------------------------


def test_check_passes_when_there_is_no_ai_message(enforcer):
    assert check(enforcer) is None


def test_check_passes_a_final_answer(enforcer):
    assert check(enforcer, AIMessage(content="done")) is None


def test_check_passes_unenforced_tools(enforcer):
    assert check(enforcer, ai(call("ls", {"path": "/"}, "l1"))) is None


def test_check_blocks_an_unread_enforced_tool(enforcer):
    update = check(enforcer, ai(query()))
    assert update["jump_to"] == "model"
    [reply] = update["messages"]
    assert (reply.tool_call_id, reply.status) == ("q1", "error")
    assert SKILL in reply.content


def test_check_passes_after_the_skill_was_read(enforcer):
    assert check(enforcer, ai(read()), result("r1"), ai(query())) is None


def test_check_still_blocks_after_reading_a_different_skill(enforcer):
    update = check(enforcer, ai(read("/skills/trend-report/SKILL.md")), result("r1"), ai(query()))
    assert update["jump_to"] == "model"


def test_check_answers_every_call_in_a_blocked_turn(enforcer):
    """jump_to skips the tools node for the whole turn; a call left without a
    result makes the next model request fail."""
    update = check(enforcer, ai(call("ls", id="l1"), query("q1"), query("q2")))
    replies = {m.tool_call_id: m for m in update["messages"]}
    assert set(replies) == {"l1", "q1", "q2"}
    assert SKILL in replies["q1"].content and SKILL in replies["q2"].content
    assert "Not run" in replies["l1"].content


def test_check_replies_in_tool_call_order(enforcer):
    update = check(enforcer, ai(call("ls", id="l1"), query("q1"), call("glob", id="g1")))
    assert [m.tool_call_id for m in update["messages"]] == ["l1", "q1", "g1"]


def test_check_only_looks_at_the_latest_turn(enforcer):
    """An earlier blocked turn has already been answered."""
    history = [ai(query("q0")), enforcer._block_message(TOOL, "q0"), ai(call("ls", id="l1"))]
    assert check(enforcer, *history) is None


def test_check_does_not_mutate_state(enforcer):
    messages = [HumanMessage("q"), ai(query())]
    enforcer._check_skills({"messages": messages})
    assert len(messages) == 2


def test_check_tolerates_state_without_messages(enforcer):
    assert enforcer._check_skills({}) is None


def test_check_blocks_a_read_issued_alongside_the_query(enforcer):
    """The SQL was written before the skill was read, so it can't follow it."""
    update = check(enforcer, ai(read(), query()))
    assert update is not None and update["jump_to"] == "model"


# --- after_model / aafter_model --------------------------------------------


@pytest.mark.parametrize("hook", ["after_model", "aafter_model"])
def test_hooks_declare_the_jump_to_model(hook):
    assert getattr(SkillEnforcerMiddleware, hook).__can_jump_to__ == ["model"]


def test_after_model_delegates_to_check(enforcer):
    state = {"messages": [HumanMessage("q"), ai(query())]}
    assert enforcer.after_model(state, runtime=None) == enforcer._check_skills(state)


def test_aafter_model_delegates_to_check(enforcer):
    state = {"messages": [HumanMessage("q"), ai(query())]}
    assert asyncio.run(enforcer.aafter_model(state, runtime=None)) == enforcer._check_skills(state)


# --- wired into a graph ------------------------------------------------------


class ScriptedModel(FakeMessagesListChatModel):
    """Replays a fixed list of AIMessages; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def run_graph(skills_dir: Path, script: list[AIMessage]) -> tuple[list, list[str]]:
    """Run the agent to completion; return its messages and the SQL that ran."""
    executed: list[str] = []

    @tool(TOOL)
    async def stub_query(sql: str) -> str:
        """Stub for the database tool."""
        executed.append(sql)
        return '[{"n": 1}]'

    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/skills/": FilesystemBackend(root_dir=skills_dir, virtual_mode=True)},
    )
    agent = create_deep_agent(
        model=ScriptedModel(responses=script),
        tools=[stub_query],
        backend=backend,
        middleware=[SkillEnforcerMiddleware(skills_dir, target_skills=TOOL)],
    )
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("how many?")]}))
    return out["messages"], executed


def test_graph_blocks_then_lets_the_retry_through(skills_dir):
    messages, executed = run_graph(
        skills_dir,
        [
            ai(call("ls", {"path": "/"}, "l1"), query("q1")),
            ai(read(id="r1")),
            ai(call(TOOL, {"sql": "SELECT 2"}, "q2")),
            AIMessage(content="1"),
        ],
    )
    assert executed == ["SELECT 2"]  # the blocked call never reached the tool

    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    assert set(results) == {"l1", "q1", "r1", "q2"}  # nothing left dangling
    assert "Not run" in results["l1"].content
    assert SKILL in results["q1"].content
    assert "rules" in results["r1"].content  # read through /skills/ to disk
    assert results["q2"].content == '[{"n": 1}]'
    assert messages[-1].content == "1"


def test_graph_blocks_a_query_sent_alongside_the_read(skills_dir):
    """The SQL was written before the skill came back, so it must be re-sent."""
    messages, executed = run_graph(
        skills_dir,
        [
            ai(read(id="r1"), query("q1")),
            ai(read(id="r2")),
            ai(call(TOOL, {"sql": "SELECT 2"}, "q2")),
            AIMessage(content="1"),
        ],
    )
    assert executed == ["SELECT 2"]
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    assert "Not run" in results["r1"].content  # skipped with the rest of the turn
    assert SKILL in results["q1"].content


def test_graph_does_not_accept_a_truncated_read(skills_dir):
    (skills_dir / TOOL / "SKILL.md").write_text("---\nname: q\ndescription: d\n---\n" + "rule\n" * 20)
    messages, executed = run_graph(
        skills_dir,
        [
            ai(read(id="r1", limit=2)),
            ai(query("q1")),
            ai(read(id="r2")),
            ai(call(TOOL, {"sql": "SELECT 2"}, "q2")),
            AIMessage(content="1"),
        ],
    )
    assert executed == ["SELECT 2"]
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    assert "next offset" in results["r1"].content  # the real read_file header
    assert SKILL in results["q1"].content


def test_graph_runs_unenforced_tools_normally(skills_dir):
    messages, executed = run_graph(
        skills_dir,
        [ai(call("ls", {"path": "/skills/"}, "l1")), AIMessage(content="done")],
    )
    [ls_result] = [m for m in messages if isinstance(m, ToolMessage)]
    assert "Not run" not in ls_result.content and TOOL in ls_result.content
    assert executed == []
