"""The workflow advisor: what it is told, how its suggestions are checked, and applying them.

The model is a stand-in that returns fixed advice and keeps what it was told.
"""

import asyncio
from pathlib import PurePosixPath

from fastapi import FastAPI
from fastapi.testclient import TestClient

from analyses import AnalysisStore
from api import ConversationManager
from api.analyses import analyses_api
from api.workflows import workflows_router
from test_adaptive_runs import LIMITS, fake_estimate, fake_query, fake_rows, naive
from test_analyses import ShellProvider, no_datasets
from test_workflows import lanes_body, wait_for
from workflows import Workflow, WorkflowStore
from workflows.advisor import Advice, Suggestion, check


class StandIn:
    """The advisor's model: answers with `advice`, keeps each summary it was given."""

    def __init__(self, advice: Advice | Exception):
        self.advice, self.told = advice, []

    async def __call__(self, summary, goal):
        self.told.append((summary, goal))
        if isinstance(self.advice, Exception):
            raise self.advice
        return self.advice


def build(tmp_path, advise=None, sandbox_limit=3):
    analyses = AnalysisStore(tmp_path / "saved")
    analyses.save(naive())

    async def no_agent(conversation_id, message, history):
        return "I could not find a better strategy."

    manager = ConversationManager(lambda sandbox, checkpointer: object(), ShellProvider(tmp_path), idle_seconds=900)
    api = analyses_api(analyses, manager, query=fake_query, query_rows=fake_rows, read_dataset=no_datasets,
                       work_dir=PurePosixPath(tmp_path) / "work", account="test_user",
                       estimate=fake_estimate, limits=LIMITS, ask_agent=no_agent)
    app = FastAPI()
    app.include_router(api.router)
    app.include_router(workflows_router(WorkflowStore(tmp_path / "workflows"), analyses, api, query_rows=fake_rows,
                                        sandbox_limit=sandbox_limit, advise=advise, advisor_name="stand-in",
                                        settings={"warm_sandboxes": 2}))
    return app


TWO_AND_A_BIG_ONE = lanes_body(("a", "sessions1", {}), ("b", "sessions1", {}),
                               ("big", "sessions1", {"events": "events_big"}))


def workflow_with_a_run(client) -> tuple[str, dict]:
    workflow = client.post("/api/workflows", json=TWO_AND_A_BIG_ONE).json()
    started = client.post(f"/api/workflows/{workflow['id']}/runs").json()
    return workflow["id"], wait_for(client, workflow["id"], started["id"])


def test_the_advisor_is_told_the_workflow_its_runs_and_its_limits(tmp_path):
    advisor = StandIn(Advice(summary="Mostly waiting for sandboxes.", suggestions=[]))
    with TestClient(build(tmp_path, advisor)) as client:
        workflow_id, run = workflow_with_a_run(client)
        answer = client.post(f"/api/workflows/{workflow_id}/advice", json={"goal": "cheapest"})
    assert answer.status_code == 201, answer.text
    assert (answer.json()["summary"], answer.json()["suggestions"], answer.json()["runs"]) == (
        "Mostly waiting for sandboxes.", [], [run["id"]])
    [(summary, goal)] = advisor.told
    assert goal == "cheapest" and summary["goal"] == "cheapest"
    assert summary["settings"] == {"warm_sandboxes": 2, "sandbox_limit": 3, "default_lanes": 2, "lanes_now": 2}
    [told_run] = summary["runs"]
    assert (told_run["total_s"], told_run["analyses"], told_run["failed_or_stopped"]) == (run["seconds"], 3, ["big"])
    nodes = {n["node_id"]: n for n in summary["analyses"]}
    assert nodes["a"]["tables"] == ["events"] and nodes["big"]["tables"] == ["events_big"]
    assert nodes["a"]["sql"] == ["SELECT * FROM events WHERE minute >= 0"] and not nodes["a"]["uses_a_time_window"]
    [a_run] = nodes["a"]["last_runs"]
    assert a_run["status"] == "done" and a_run["script_s"] is not None and a_run["rows_in"] == 11
    [big_run] = nodes["big"]["last_runs"]
    assert big_run["status"] == "needs_optimization" and "data_movement" in big_run["findings"][0]
    assert summary["read_the_same_tables"] == [{"tables": ["events"], "nodes": ["a", "b"]}]


def test_suggestions_are_checked_before_they_are_shown():
    workflow = Workflow.model_validate({**TWO_AND_A_BIG_ONE, "id": "w1", "settings": {"lanes": 2}})
    advice = Advice(summary="s", suggestions=[
        Suggestion(kind="set_lanes", value=9, reason="r", expected_gain="g"),
        Suggestion(kind="set_lanes", value=2, reason="already so", expected_gain="g"),
        Suggestion(kind="run_first", node_ids=["nope"], reason="names no real node", expected_gain="g"),
        Suggestion(kind="run_first", node_ids=["big", "nope", "big"], reason="r", expected_gain="g"),
        Suggestion(kind="optimize_analysis", node_ids=["a"], reason="no limit broken", expected_gain="g"),
        Suggestion(kind="optimize_analysis", node_ids=["big"], reason="r", expected_gain="g"),
        Suggestion(kind="share_data", node_ids=["a", "b"], reason="r", expected_gain="g"),
    ])
    checked = check(advice, workflow, sandbox_limit=3, findings={"a": False, "big": True})
    # At most five; the one naming no real node is dropped.
    assert [(c.kind, c.node_ids, c.value, c.applicable) for c in checked] == [
        ("set_lanes", [], 3, True),  # 9 is more than an account may hold
        ("set_lanes", [], 2, False),  # already set
        ("run_first", ["big"], None, True),
        ("optimize_analysis", ["a"], None, False),
    ]
    assert "at most 3 sandboxes" in checked[0].note and "broke no limit" in checked[3].note


def test_applying_lanes_and_run_first_changes_the_next_run(tmp_path):
    advisor = StandIn(Advice(summary="s", suggestions=[
        Suggestion(kind="set_lanes", value=1, reason="sandboxes sat idle", expected_gain="one sandbox less"),
        Suggestion(kind="run_first", node_ids=["b"], reason="b's query is slow", expected_gain="2 s"),
        Suggestion(kind="share_data", node_ids=["a", "b"], reason="same table", expected_gain="one query less"),
    ]))
    with TestClient(build(tmp_path, advisor)) as client:
        workflow_id, _ = workflow_with_a_run(client)
        advice = client.post(f"/api/workflows/{workflow_id}/advice").json()
        base = f"/api/workflows/{workflow_id}/advice/{advice['id']}/suggestions"
        lanes = client.post(f"{base}/0/apply").json()
        assert lanes["workflow"]["settings"]["lanes"] == 1
        first = client.post(f"{base}/1/apply").json()
        assert {n["id"]: n.get("priority") for n in first["workflow"]["nodes"] if n["kind"] == "analysis"} == \
            {"a": 0, "b": 1, "big": 0}
        assert client.post(f"{base}/1/apply").status_code == 409  # applied already
        advice_only = client.post(f"{base}/2/apply")
        assert advice_only.status_code == 400 and "advice" in advice_only.json()["detail"]
        dismissed = client.post(f"{base}/2/dismiss").json()
        assert [s["status"] for s in dismissed["suggestions"]] == ["applied", "applied", "dismissed"]
        assert [a["id"] for a in client.get(f"/api/workflows/{workflow_id}/advice").json()] == [advice["id"]]

        started = client.post(f"/api/workflows/{workflow_id}/runs").json()
        run = wait_for(client, workflow_id, started["id"])
    assert run["lanes"] == 1
    ran = [n["node_id"] for n in sorted(run["nodes"], key=lambda n: n["sandbox_at"] or "~") if n["sandbox_at"]]
    assert ran == ["b", "a"]  # b first, by priority


def test_applying_optimize_starts_it_for_the_analysis(tmp_path):
    advisor = StandIn(Advice(summary="s", suggestions=[
        Suggestion(kind="optimize_analysis", node_ids=["big"], reason="it stopped before exporting", expected_gain="g"),
    ]))
    with TestClient(build(tmp_path, advisor)) as client:
        workflow_id, run = workflow_with_a_run(client)
        advice = client.post(f"/api/workflows/{workflow_id}/advice").json()
        assert advice["suggestions"][0]["applicable"] is True
        applied = client.post(f"/api/workflows/{workflow_id}/advice/{advice['id']}/suggestions/0/apply")
        assert applied.status_code == 200, applied.text
        assert "Optimize started" in applied.json()["advice"]["suggestions"][0]["result"]
        big = next(n for n in run["nodes"] if n["node_id"] == "big")
        for _ in range(100):
            record = client.get(f"/api/analyses/sessions1/runs/{big['run_id']}").json()
            if record["optimization"]["status"] != "running":
                break
            asyncio.run(asyncio.sleep(0.02))
    assert record["optimization"]["status"] == "failed"  # the stand-in agent saved nothing: version unchanged


def test_without_runs_an_advisor_or_an_answer(tmp_path):
    with TestClient(build(tmp_path, StandIn(Advice(summary="s")))) as client:
        workflow = client.post("/api/workflows", json=TWO_AND_A_BIG_ONE).json()
        no_runs = client.post(f"/api/workflows/{workflow['id']}/advice")
    assert no_runs.status_code == 400 and "run the workflow first" in no_runs.json()["detail"]
    with TestClient(build(tmp_path / "none", advise=None)) as client:
        workflow = client.post("/api/workflows", json=TWO_AND_A_BIG_ONE).json()
        assert client.post(f"/api/workflows/{workflow['id']}/advice").status_code == 503
    with TestClient(build(tmp_path / "broken", StandIn(RuntimeError("rate limited")))) as client:
        workflow_id, _ = workflow_with_a_run(client)
        broken = client.post(f"/api/workflows/{workflow_id}/advice")
    assert broken.status_code == 502 and "rate limited" in broken.json()["detail"]


def test_a_workflow_keeps_its_settings_and_priorities_when_saved(tmp_path):
    with TestClient(build(tmp_path)) as client:
        body = {**TWO_AND_A_BIG_ONE, "settings": {"lanes": 1}}
        body["nodes"][0]["priority"] = 2
        saved = client.post("/api/workflows", json=body).json()
        assert saved["settings"] == {"lanes": 1} and saved["nodes"][0]["priority"] == 2
        refused = client.put(f"/api/workflows/{saved['id']}", json={**body, "settings": {"lanes": 0}})
    assert refused.status_code == 400 and "at least one sandbox" in refused.json()["detail"]
