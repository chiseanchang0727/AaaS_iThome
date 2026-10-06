"""The whole-system eval: trace from history, code checks, judges, and the run through the API.

No model, no database: judges get a fake chat model, and the API run uses
test_api's scripted agent and temp-dir sandboxes.
"""

import asyncio

import httpx
from langchain_core.messages import AIMessage, ToolMessage

from api import HistoryStore
from evals.system import checks
from evals.system.cases import EvalCase, build_cases
from evals.system.evaluate import evaluate
from evals.system.judges import Judges, render_evidence
from evals.system.run import parse_sse, run_case
from evals.system.trace import build_trace
from test_api import OUTPUT_DIR, Harness, call, write_output


def line(turn, role, ts="2026-10-05T10:00:00Z", **fields):
    return {"id": f"{role}-{turn}-{ts}", "turn": turn, "role": role, "ts": ts, **fields}


def assistant(turn, content="", calls=(), usage=(100, 10), ts="2026-10-05T10:00:01Z"):
    fields = {"content": content, "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}
    if calls:
        fields["tool_calls"] = [{"id": c[0], "name": c[1], "args": c[2]} for c in calls]
    return line(turn, "assistant", ts, **fields)


def tool(turn, id, name, content, error=False):
    return line(turn, "tool", tool_call_id=id, name=name, content=content, error=error)


EARLIER = [
    line(1, "user", content="Top channels?"),
    assistant(1, calls=[("q0", "query_database", {"sql": "SELECT 1"})]),
    tool(1, "q0", "query_database", '[{"channel": "Kiln", "total": 4321987}]'),
    assistant(1, "Kiln has 4,321,987 views."),
]

TURN = [
    line(2, "user", content="How many videos does it have?", ts="2026-10-05T10:01:00Z"),
    assistant(2, calls=[("s1", "read_file", {"file_path": "/skills/query_database/SKILL.md"})]),
    tool(2, "s1", "read_file", "skill text"),
    assistant(2, calls=[("q1", "query_database", {"sql": "SELECT count(*) FROM video"})]),
    tool(2, "q1", "query_database", 'ERROR: relation "video" does not exist'),
    assistant(2, calls=[("q2", "query_database", {"sql": "SELECT count(*) FROM video"})]),
    tool(2, "q2", "query_database", 'ERROR: relation "video" does not exist'),
    assistant(2, calls=[("q3", "query_database", {"sql": "SELECT count(*) FROM videos"})]),
    tool(2, "q3", "query_database", '[{"count": 17}]'),
    assistant(2, "Kiln has 17 videos, about 254,234 views each.", usage=(300, 30), ts="2026-10-05T10:01:12.5Z"),
]


def trace():
    return build_trace(EARLIER + TURN, turn=2, artifacts=["hist.html"])


def case(**fields):
    return EvalCase("c", "How many videos does it have?", **fields)


# --- trace ------------------------------------------------------------------------


def test_trace_reads_one_turn_from_the_history():
    t = trace()
    assert t.question == "How many videos does it have?"
    assert t.answer == "Kiln has 17 videos, about 254,234 views each."
    assert [s.name for s in t.tools] == ["read_file", "query_database", "query_database", "query_database"]
    assert [s.error for s in t.tools] == [False, True, True, False]
    assert (t.model_calls, t.input_tokens, t.output_tokens) == (5, 700, 70)
    assert t.runtime_seconds == 12.5
    assert t.earlier_turns == [{"turn": 1, "sent": True, "question": "Top channels?", "answer": "Kiln has 4,321,987 views."}]
    assert t.context is None
    assert '[{"channel": "Kiln", "total": 4321987}]' in t.observations


def test_a_turn_that_stops_on_a_tool_result_has_no_answer():
    t = build_trace(TURN[:5], turn=2)
    assert t.answer == ""


# --- code checks ------------------------------------------------------------------


def test_tool_usage_counts_and_checks_expectations():
    usage = checks.tool_usage(trace(), case(
        required_tools=["query_database", "execute"], forbidden_tools=["export_query"],
        required_skills=["query_database"],
    ))
    assert usage["tools"] == {"read_file": 1, "query_database": 3}
    assert (usage["queries"], usage["skill_reads"], usage["code_executions"]) == (3, 1, 0)
    assert [c["pass"] for c in usage["checks"]] == [True, False, True, True]
    assert usage["status"] == "fail"


def test_efficiency_is_measured_not_judged():
    e = checks.efficiency(trace())
    assert e == {"steps": 4, "queries": 3, "skill_reads": 1, "code_executions": 0, "model_calls": 5,
                 "input_tokens": 700, "output_tokens": 70, "total_tokens": 770, "runtime_seconds": 12.5}


def test_expected_values_match_numbers_at_shown_precision_and_text():
    answer = "Pottery leads with a median of 1.32M; 17 videos."
    ok = checks.expected_values(answer, case(expected_values={"cat": "pottery", "median": 1_316_002, "n": 17}))
    assert ok["status"] == "pass"
    bad = checks.expected_values(answer, case(expected_values={"median": 1_416_002}))
    assert bad["status"] == "fail" and bad["missing"] == ["median=1,416,002"]
    assert checks.expected_values(answer, case())["status"] is None


def test_required_outputs_from_files_answer_and_code():
    t = trace()
    t.answer += "\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    t.tools.append(checks.ToolStep("e1", "execute", {"command": "python3 /home/x.py"},
                                   "done\n[Command succeeded with exit code 0]"))
    found = checks.required_outputs(t, case(required_outputs=["chart", "table", "python", "report", "podcast"]))
    assert [(c["check"], c["pass"]) for c in found] == [
        ("makes a chart", True), ("makes a table", True), ("makes a python", True),
        ("makes a report", True), ("makes a podcast", None),
    ]


def test_recovery_tells_identical_retries_from_changed_ones():
    r = checks.recovery(trace())
    assert [e["next"] for e in r["errors"]] == ["identical", "changed"]
    assert (r["retries"], r["identical_retries"], r["oom_events"], r["sandbox_moves"]) == (2, 1, 0, 0)


def test_recovery_counts_out_of_memory_and_moves_and_a_rewrite_counts_as_changed():
    records = [
        line(1, "user", content="cluster"),
        assistant(1, calls=[("e1", "execute", {"command": "python3 a.py"})]),
        line(1, "sandbox_step", command="python3 a.py", exit_code=137, signal=9, peak_memory_mb=990, memory_limit_mb=1024),
        tool(1, "e1", "execute", "Killed\n[Command failed with exit code 137]"),
        assistant(1, calls=[("w1", "write_file", {"file_path": "a.py", "content": "lean"})]),
        tool(1, "w1", "write_file", "ok"),
        assistant(1, calls=[("e2", "execute", {"command": "python3 a.py"})]),
        line(1, "sandbox_event", event="switched", memory_gb=4),
        line(1, "sandbox_step", command="python3 a.py", exit_code=0, peak_memory_mb=2100, memory_limit_mb=4096),
        tool(1, "e2", "execute", "6763\n[Command succeeded with exit code 0]"),
        assistant(1, "Clusters: 6763 ..."),
    ]
    r = checks.recovery(build_trace(records, turn=1))
    assert [e["next"] for e in r["errors"]] == ["changed"]
    assert (r["oom_events"], r["sandbox_moves"], r["oom_resolved_by"]) == (1, 1, "bigger sandbox")
    assert [s["out_of_memory"] for s in r["code_steps"]] == [True, False]


def test_numeric_grounding_flags_values_no_tool_showed():
    g = checks.numeric_grounding(trace(), "How many videos does it have?")
    assert g["answer_values"] == ["17", "254,234"]
    assert g["ungrounded_values"] == ["254,234"]


# --- judges and the combined result ---------------------------------------------


class FakeModel:
    """Stands in for a chat model: `with_structured_output(schema).ainvoke(text)`."""

    def __init__(self, answers):
        self.answers, self.prompts = answers, []

    def with_structured_output(self, schema):
        model = self

        class Bound:
            async def ainvoke(self, text):
                model.prompts.append(text)
                name = next(n for n in model.answers if n in text)
                answer = model.answers[name]
                if isinstance(answer, Exception):
                    raise answer
                return schema.model_validate(answer)

        return Bound()


PASS = {"status": "pass", "reason": "fine"}
ANSWERS = {
    "answered the user's question correctly": PASS,
    "final answer is grounded": {"status": "fail", "reason": "one made-up average",
                                 "unsupported_claims": [{"claim": "about 254,234 views each", "reason": "never computed"}]},
    "did what the user asked": PASS,
    "how a data-analysis agent went about a task": RuntimeError("rate limited"),
}


def test_a_failed_code_check_wins_and_a_failed_judge_is_an_error_not_a_pass():
    model = FakeModel(ANSWERS)
    result = asyncio.run(evaluate(case(expected_values={"videos": 18}, required_tools=["query_database"]),
                                  trace(), Judges(model)))
    assert result["correct"]["status"] == "fail"
    assert result["correct"]["reason"].startswith("Missing from the answer: videos=18.")
    assert result["grounded"]["status"] == "fail"
    assert result["grounded"]["unsupported_claims"][0]["claim"] == "about 254,234 views each"
    assert result["grounded"]["ungrounded_values"] == ["254,234"]
    assert result["instruction_following"]["status"] == "pass"
    assert result["execution_strategy"]["status"] == "judge_error"
    assert "rate limited" in result["execution_strategy"]["reason"]
    # the judges saw the trace, with the failed queries numbered
    assert all('ERROR: relation "video"' in p for p in model.prompts[1:])


def test_without_judges_only_code_decides():
    result = asyncio.run(evaluate(case(expected_values={"videos": 17}), trace(), None))
    assert result["correct"]["status"] == "pass"
    assert result["grounded"]["status"] == "unknown"
    assert result["instruction_following"]["status"] == "unknown"
    assert result["execution_strategy"]["status"] == "unknown"


def test_no_answer_is_a_correctness_fail():
    result = asyncio.run(evaluate(case(), build_trace(TURN[:5], turn=2), None))
    assert (result["correct"]["status"], result["stopped"]) == ("fail", True)


def test_evidence_lists_steps_in_order():
    text = render_evidence(trace())
    assert text.index("[1] call read_file") < text.index("[2] call query_database") < text.index("[4] result")


def test_the_cases_are_plain_json():
    import json

    cases = build_cases()
    assert len({c.id for c in cases}) == len(cases)
    json.dumps([c.to_dict() for c in cases])


# --- the run, through the real API ------------------------------------------------


def test_run_case_goes_through_the_api_and_the_history_is_the_trace(tmp_path):
    turns = [
        [{"model": {"messages": [AIMessage("Top 3: A, B, C.")]}}],
        [
            {"model": {"messages": [AIMessage("", tool_calls=[call("query_database", {"sql": "SELECT 7"}, "c1")],
                                              usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55})]}},
            {"tools": {"messages": [ToolMessage('[{"n": 7}]', tool_call_id="c1", name="query_database")]}},
            write_output("chart.html", b"<html></html>"),
            {"model": {"messages": [AIMessage("B has 7 videos.")]}},
        ],
    ]
    history = HistoryStore(tmp_path / "history")
    harness = Harness(tmp_path, turns, history=history)
    the_case = EvalCase("follow", "How many videos does the second one have?", setup=["Top 3 channels?"],
                        expected_values={"videos": 7}, required_outputs=["chart"])

    async def go():
        transport = httpx.ASGITransport(app=harness.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval") as client:
            return await run_case(client, the_case)

    errors, artifacts = asyncio.run(go())
    assert errors == [] and artifacts == ["chart.html"]
    t = build_trace(history.read("follow"), turn=2, artifacts=artifacts)
    assert t.answer == "B has 7 videos." and t.input_tokens == 50
    assert t.earlier_turns == [{"turn": 1, "sent": True, "question": "Top 3 channels?", "answer": "Top 3: A, B, C."}]
    result = asyncio.run(evaluate(the_case, t, None))
    assert result["correct"]["status"] == "pass"
    assert result["instruction_following"]["status"] == "pass"
    assert harness.provider.destroyed  # the conversation was ended, its sandbox deleted


def test_parse_sse():
    body = 'event: thread\ndata: {"type": "thread"}\n\nevent: done\ndata: {"type": "done"}\n\n'
    assert [e["type"] for e in parse_sse(body)] == ["thread", "done"]


# --- evaluating saved conversations -------------------------------------------------


def saved_history(tmp_path) -> HistoryStore:
    history = HistoryStore(tmp_path / "conversations")
    for r in EARLIER + TURN:
        history.append("Conv-1", r)
    (tmp_path / "artifacts" / "Conv-1").mkdir(parents=True)
    (tmp_path / "artifacts" / "Conv-1" / "hist.html").write_text("<html></html>")
    return history


def test_every_saved_turn_becomes_a_case_with_the_turns_before_it():
    from evals.system.history_run import history_cases, turn_artifacts

    records = EARLIER + [assistant(1, calls=[("w", "write_file", {"file_path": "/out/hist.html"})])]
    assert turn_artifacts(records, 1, ["hist.html", "other.png"]) == ["hist.html"]
    assert turn_artifacts(records, 2, ["hist.html"]) == []


def test_evaluate_history_saves_a_run_the_page_can_show(tmp_path):
    from evals.system.history_run import evaluate_history, history_cases

    history = saved_history(tmp_path)
    [(first, _), (second, thread)] = history_cases(history)
    assert (first.id, second.id, thread) == ("Conv-1_t1", "Conv-1_t2", "Conv-1")
    assert second.setup == ["Top channels?"]

    run_dir = tmp_path / "runs" / "20261005-130000"
    seen = []
    results = asyncio.run(evaluate_history(history, tmp_path / "artifacts", run_dir, None, {"model": "m"}, seen.append))
    assert [r["case_id"] for r in results] == ["Conv-1_t1", "Conv-1_t2"] == [r["case_id"] for r in seen]
    assert results[1]["recovery"]["identical_retries"] == 1
    saved = __import__("json").loads((run_dir / "run.json").read_text())
    assert (saved["status"], saved["total"], saved["source"]) == ("done", 2, "history")
    assert (run_dir / "history" / "Conv-1_t2.jsonl").is_file()


def test_the_history_run_routes_start_one_run_at_a_time(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.evals import system_router

    history = saved_history(tmp_path)
    app = FastAPI()
    app.include_router(system_router(tmp_path / "runs", history=history, artifacts_dir=tmp_path / "artifacts"))
    with TestClient(app) as client:
        assert client.get("/api/evals/system/history-runs").json() == {
            "available": True, "folder": str(tmp_path / "conversations"), "conversations": 1, "turns": 2, "running": None,
        }
        started = client.post("/api/evals/system/history-runs")
        assert started.status_code == 202 and started.json()["total"] == 2
        run_id = started.json()["id"]
        for _ in range(100):
            run = client.get(f"/api/evals/system/runs/{run_id}").json()
            if run["status"] == "done":
                break
        assert run["status"] == "done" and len(run["results"]) == 2
        assert client.get(f"/api/evals/system/runs/{run_id}/history/Conv-1_t2").status_code == 200
        [listed] = client.get("/api/evals/system/runs").json()
        assert (listed["source"], listed["status"], listed["total"]) == ("history", "done", 2)


def test_without_history_there_is_nothing_to_start(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.evals import system_router

    app = FastAPI()
    app.include_router(system_router(tmp_path / "runs"))
    client = TestClient(app)
    assert client.get("/api/evals/system/history-runs").json()["available"] is False
    assert client.post("/api/evals/system/history-runs").status_code == 404


# --- Jev as the context manager -----------------------------------------------------


def test_with_a_context_filter_only_sent_turns_count_as_seen():
    records = EARLIER + [TURN[0], line(2, "context", sent_turns=[], earlier_turns=1, seconds=0.4)] + TURN[1:]
    t = build_trace(records, turn=2)
    assert t.context == {"sent_turns": [], "earlier_turns": 1, "seconds": 0.4}
    assert t.earlier_turns[0]["sent"] is False
    assert '[{"channel": "Kiln", "total": 4321987}]' not in t.observations  # turn 1 was not sent
    from evals.system.judges import render_conversation

    assert "NOT sent to the agent" in render_conversation(t)


class FakeContext:
    """Stands in for agent.context_filter.ContextFilter: sends no earlier turn."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    async def build(self, records, request):
        from langchain_core.messages import HumanMessage

        self.calls.append((sum(r["role"] == "user" for r in records), request))
        if self.fail:
            raise RuntimeError("jev down")
        return [HumanMessage(request)], []


def test_the_app_sends_only_what_the_context_filter_picks_and_logs_it(tmp_path):
    from fastapi.testclient import TestClient

    from test_api import chat

    history = HistoryStore(tmp_path / "history")
    harness = Harness(tmp_path, history=history)
    context = FakeContext()
    from api import create_app

    app = create_app(harness.manager, artifacts_dir=tmp_path / "a", output_dir=OUTPUT_DIR,
                     history=history, context=context)
    with TestClient(app) as client:
        thread_id = chat(client, "first")[0]["thread_id"]
        chat(client, "second", thread_id)

    assert context.calls == [(0, "first"), (1, "second")]  # earlier turns in the history it was given
    second_input = harness.agents[0].calls[1][0]["messages"]
    assert [m.content for m in second_input] == ["second"]  # turn 1 was not sent
    lines = [r for r in history.read(thread_id) if r["role"] == "context"]
    assert [(r["turn"], r["sent_turns"], r["earlier_turns"]) for r in lines] == [(1, [], 0), (2, [], 1)]


def test_if_jev_fails_the_whole_conversation_is_sent(tmp_path):
    from fastapi.testclient import TestClient

    from api import create_app
    from test_api import chat

    history = HistoryStore(tmp_path / "history")
    harness = Harness(tmp_path, history=history)
    app = create_app(harness.manager, artifacts_dir=tmp_path / "a", output_dir=OUTPUT_DIR,
                     history=history, context=FakeContext(fail=True))
    with TestClient(app) as client:
        thread_id = chat(client, "first")[0]["thread_id"]
        chat(client, "second", thread_id)

    second_input = harness.agents[0].calls[1][0]["messages"]
    assert [m.content for m in second_input] == ["first", "ok", "second"]
    [_, fallback] = [r for r in history.read(thread_id) if r["role"] == "context"]
    assert fallback["sent_turns"] == [1] and fallback["error"] == "jev down"
