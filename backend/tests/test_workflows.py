"""Workflows: analyses run in parallel, each feeding a batch node that collects them.

Uses the adaptive-runs fixtures: `events` is small, `events_big` has the same
columns and is too big for the saved recipe, so a node on it stops with
"needs_optimization" while the node on `events` finishes.
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
from workflows import Workflow, WorkflowRun, WorkflowStore


def build(tmp_path, sandbox_limit=None):
    analyses = AnalysisStore(tmp_path / "saved")
    analyses.save(naive())
    manager = ConversationManager(lambda sandbox, checkpointer: object(), ShellProvider(tmp_path), idle_seconds=900)
    api = analyses_api(analyses, manager, query=fake_query, query_rows=fake_rows, read_dataset=no_datasets,
                       work_dir=PurePosixPath(tmp_path) / "work", account="test_user",
                       estimate=fake_estimate, limits=LIMITS)
    app = FastAPI()
    app.include_router(api.router)
    app.include_router(workflows_router(WorkflowStore(tmp_path / "workflows"), analyses, api, query_rows=fake_rows,
                                        sandbox_limit=sandbox_limit))
    return app, analyses


def body(**fields) -> dict:
    """Median sessions on the small table and on the big one, both into one batch, into one report."""
    return {
        "title": "Sessions on both tables",
        "nodes": [
            {"kind": "analysis", "id": "small", "analysis_id": "sessions1", "position": {"x": 0, "y": 0}},
            {"kind": "analysis", "id": "big", "analysis_id": "sessions1", "sources": {"events": "events_big"}},
            {"kind": "analysis", "id": "loose", "analysis_id": "sessions1"},
            {"kind": "batch", "id": "batch", "title": "Both tables"},
            {"kind": "report", "id": "report", "title": "Sessions report"},
        ],
        "edges": [{"source": "small", "target": "batch"}, {"source": "big", "target": "batch"},
                  {"source": "batch", "target": "report"}],
        **fields,
    }


def wait_for(client, workflow_id, run_id):
    for _ in range(300):
        run = client.get(f"/api/workflows/{workflow_id}/runs/{run_id}").json()
        if run["status"] != "running" and not client.get(f"/api/workflows/{workflow_id}").json()["running"]:
            return run
        asyncio.run(asyncio.sleep(0.02))
    raise AssertionError("the workflow did not finish")


def test_a_workflow_runs_its_connected_analyses_in_parallel_and_the_batch_collects_them(tmp_path):
    app, analyses = build(tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/workflows", json=body())
        assert created.status_code == 201, created.text
        workflow = created.json()
        assert workflow["analyses"]["sessions1"]["title"] == "Median session length by device"

        started = client.post(f"/api/workflows/{workflow['id']}/runs")
        assert started.status_code == 202, started.text
        # Both nodes started at once, as two runs of the same analysis; the unconnected one did not.
        assert [n["node_id"] for n in started.json()["nodes"]] == ["small", "big"]
        assert client.post(f"/api/workflows/{workflow['id']}/runs").status_code == 409
        run = wait_for(client, workflow["id"], started.json()["id"])

    assert run["status"] == "partly_done"
    assert run["batches"] == {"batch": ["small", "big"]} and run["reports"] == {"report": ["batch"]}
    assert run["titles"] == {"batch": "Both tables", "report": "Sessions report"}
    small, big = run["nodes"]
    assert (small["status"], small["outputs"], small["sources"]) == ("done", ["sessions.html"], {})
    assert (big["status"], big["sources"]) == ("needs_optimization", {"events": "events_big"})
    assert "100,000" in big["error"]

    records = {r.id: r for r in analyses.runs("sessions1")}
    for node in (small, big):
        record = records[node["run_id"]]
        assert (record.trigger, record.workflow_run) == ("workflow", run["id"])
    # When each was asked for, got its sandbox, and ended: the big one never got one (the size check stopped it).
    assert small["queued_at"] <= small["sandbox_at"] <= small["finished_at"]
    assert big["queued_at"] <= big["finished_at"] and big["sandbox_at"] is None
    assert small["rows"] == 11 and small["peak_memory_mb"] is not None and big["rows"] is None
    assert analyses.output_path("sessions1", small["run_id"], "sessions.html") is not None

    with TestClient(app) as client:
        page = client.get(f"/api/workflows/{workflow['id']}/runs/{run['id']}/reports/report")
        missing = client.get(f"/api/workflows/{workflow['id']}/runs/{run['id']}/reports/batch")
        load = client.get(f"/api/workflows/{workflow['id']}/runs/{run['id']}/load")
        no_run = client.get(f"/api/workflows/{workflow['id']}/runs/20990101-000000000-zzzz/load")
    assert page.status_code == 200 and page.headers["content-security-policy"] == "sandbox allow-scripts"
    text = page.text
    assert "<h1>Sessions report</h1>" in text and "<h2>Both tables</h2>" in text
    assert "Sessions on both tables" in text and "1 of 2 analyses done" in text
    assert "on events → events_big" in text and "needs optimization" in text
    assert "<h2>Load</h2>" not in text and "Load of this batch" not in text  # the results only
    # The chart is embedded, escaped, in a sandboxed frame: the page stands on its own.
    assert '<iframe sandbox="allow-scripts" title="sessions.html" srcdoc="' in text and "Plotly.newPlot(&quot;c&quot;" in text
    assert missing.status_code == 404

    # The load is a page of its own: the whole run, each batch, each analysis.
    assert load.status_code == 200 and load.headers["content-security-policy"] == "sandbox allow-scripts"
    text = load.text
    assert "<h1>Load: Sessions on both tables</h1>" in text and "<h2>Load</h2>" in text
    assert "sandboxes busy at the peak" in text and "11 rows" in text
    assert 'aria-label="Analyses in a sandbox and waiting for one, over time; at most 1 in sandboxes at once"' in text
    assert "no sandbox" in text  # the big one's bar
    assert "<h2>Both tables</h2>" in text and "Load of this batch" in text
    assert "<iframe" not in text  # no results here
    assert no_run.status_code == 404


def test_no_more_run_at_once_than_the_account_may_hold_sandboxes(tmp_path):
    """The rest wait in the workflow, for as long as it takes, instead of the manager's short wait."""
    app, analyses = build(tmp_path, sandbox_limit=1)
    with TestClient(app) as client:
        workflow = client.post("/api/workflows", json=body()).json()
        started = client.post(f"/api/workflows/{workflow['id']}/runs").json()
        assert [(n["status"], n["run_id"]) for n in started["nodes"]] == [("waiting", None), ("waiting", None)]
        run = wait_for(client, workflow["id"], started["id"])
    first, second = sorted((analyses.get_run("sessions1", n["run_id"]) for n in run["nodes"]), key=lambda r: r.queued_at)
    assert second.queued_at >= first.finished_at  # one after the other
    # Both count as queued from when the workflow started: the wait shows on the load chart.
    assert run["nodes"][0]["queued_at"][:19] == run["nodes"][1]["queued_at"][:19]
    assert {n["status"] for n in run["nodes"]} == {"done", "needs_optimization"}


def test_a_workflow_must_have_the_right_shape(tmp_path):
    app, _ = build(tmp_path)
    with TestClient(app) as client:
        backwards = client.post("/api/workflows", json=body(edges=[{"source": "batch", "target": "small"}]))
        missing = client.post("/api/workflows", json=body(nodes=[
            {"kind": "analysis", "id": "a", "analysis_id": "gone"}, {"kind": "batch", "id": "b"}]))
        bad_table = client.post("/api/workflows", json=body(nodes=[
            {"kind": "analysis", "id": "small", "analysis_id": "sessions1", "sources": {"events": "nope"}},
            {"kind": "batch", "id": "batch"}], edges=[{"source": "small", "target": "batch"}]))
        untitled = client.post("/api/workflows", json=body(title="  "))
    assert backwards.status_code == 400 and "from an analysis to a batch" in backwards.json()["detail"]
    assert "not batch -> analysis" in backwards.json()["detail"]
    assert missing.status_code == 400 and "no saved analysis 'gone'" in missing.json()["detail"]
    assert bad_table.status_code == 400 and "no table nope" in bad_table.json()["detail"]
    assert untitled.status_code == 400 and "title" in untitled.json()["detail"]


def test_nothing_connected_to_a_batch_means_nothing_to_run(tmp_path):
    app, _ = build(tmp_path)
    with TestClient(app) as client:
        workflow = client.post("/api/workflows", json=body(edges=[])).json()
        refused = client.post(f"/api/workflows/{workflow['id']}/runs")
    assert refused.status_code == 400 and "connect at least one analysis" in refused.json()["detail"]


def test_a_workflow_can_be_changed_listed_and_deleted(tmp_path):
    app, _ = build(tmp_path)
    with TestClient(app) as client:
        workflow = client.post("/api/workflows", json=body()).json()
        changed = client.put(f"/api/workflows/{workflow['id']}", json=body(title="Renamed", edges=[]))
        assert changed.status_code == 200 and changed.json()["updated_at"] is not None
        [listed] = client.get("/api/workflows").json()
        assert (listed["title"], listed["analyses"], listed["batches"], listed["reports"], listed["last_run"]) == \
            ("Renamed", 3, 1, 1, None)
        assert client.delete(f"/api/workflows/{workflow['id']}").json() == {"deleted": workflow["id"]}
        assert client.get(f"/api/workflows/{workflow['id']}").status_code == 404


def test_batches_and_connected_nodes():
    workflow = Workflow.model_validate(body(id="w1", nodes=[*body()["nodes"], {"kind": "batch", "id": "second"}],
                                           edges=[*body()["edges"], {"source": "small", "target": "second"}]))
    assert workflow.problems() == []
    assert workflow.batches() == {"batch": ["small", "big"], "second": ["small"]}
    assert [n.id for n in workflow.connected()] == ["small", "big"]  # small runs once, though in two batches
    assert workflow.reports() == {"report": ["batch"]}
    skipping = Workflow.model_validate(body(id="w3", edges=[{"source": "small", "target": "report"}]))
    assert skipping.problems() == ["an edge must go from an analysis to a batch, or from a batch to a report, "
                                   "not analysis -> report"]
    twice = Workflow.model_validate(body(id="w2", edges=[*body()["edges"], {"source": "small", "target": "batch"}]))
    assert twice.problems() == ["small -> batch is connected twice"]


def node_at(name, queued, ready, finished, script=None, samples=(), **fields):
    from workflows.models import NodeRun

    stamp = lambda sec: None if sec is None else f"2026-10-11T10:00:{sec:06.3f}Z"  # noqa: E731
    return NodeRun(node_id=name, analysis_id=name, title=name, status="done", queued_at=stamp(queued),
                   sandbox_at=stamp(ready), finished_at=stamp(finished), script_at=stamp(script),
                   script_seconds=(samples[-1][0] if samples else None), samples=list(samples), **fields)


def test_load_counts_who_runs_and_who_waits():
    from workflows.load import counts, facts, phases, run_load

    # Three start at once with room for two: c waits until a ends; d is stopped by the size check.
    nodes = [node_at("a", 0, 1, 5, rows=10, bytes=2000, peak_memory_mb=200, memory_limit_mb=1024),
             node_at("b", 0, 1, 8, peak_memory_mb=300, memory_limit_mb=1024), node_at("c", 0, 6, 9), node_at("d", 0, None, 0.5)]
    found = phases(nodes)
    assert counts(found) == [(0, 0, 4), (0.5, 0, 3), (1, 2, 1), (5, 1, 1), (6, 2, 0), (8, 1, 0), (9, 0, 0)]
    numbers = dict((label, value) for value, label in facts(nodes, found, 2))
    assert numbers["sandboxes busy at the peak (1s in)"] == "2 of 2"
    assert numbers["before sandboxes were ready, all together"] == "8s"  # 1 + 1 + 6 + 0.5, rounded
    assert numbers["highest peak memory: b"] == "300 MB of 1,024 MB"
    page = run_load(WorkflowRun(id="r", workflow_id="w", started_at="t", sandbox_limit=2), nodes, [("Batch", nodes)])
    assert "sandbox limit: 2" in page and '<text class="group" x="8" y="17">Batch</text>' in page
    old = run_load(WorkflowRun(id="r", workflow_id="w", started_at="t"), [node_at("x", 0, 1, 2)][:0], [])
    assert "only runs from now on have load charts" in old


def test_memory_and_cpu_over_time():
    from workflows.load import analysis_load, batch_load, memory_total, phases

    # a's script runs 2s..6s, b's 3s..5s; samples are [seconds since its script started, MB, CPU seconds so far].
    a = node_at("a", 0, 1, 7, script=2, samples=[[0, 100, 10.0], [2, 300, 12.0], [4, 200, 16.0]], memory_limit_mb=1024)
    b = node_at("b", 0, 1, 6, script=3, samples=[[0, 50, 1.0], [2, 50, 1.5]])
    found = {p.node.node_id: p for p in phases([a, b])}
    assert found["a"].memory() == [(2, 100), (4, 300), (6, 200)]
    assert found["a"].cores() == [(4, 1.0), (6, 2.0)]  # 2 CPU seconds in 2s, then 4 in 2s
    total = dict(memory_total(list(found.values()), 8, steps=8))
    assert (total[1], total[2], total[3], total[4], total[5], total[7]) == (0, 100, 150, 350, 350, 0)

    chart = analysis_load(a)
    assert "Memory while its script ran (its sandbox has 1,024 MB)" in chart
    assert "sandbox memory: 1,024 MB" not in chart  # 300 MB is far from it: no line flattening the curve
    near = analysis_load(node_at("n", 0, 1, 7, script=2, samples=[[0, 100, 1.0], [1, 900, 2.0]], memory_limit_mb=1024))
    assert "sandbox memory: 1,024 MB" in near and ">0.2s<" in near  # a short run: tenths of a second
    assert "CPU while its script ran" in chart and "highest 2 cores" in chart
    assert analysis_load(node_at("c", 0, 1, 2)) == ""  # no samples, no charts

    block = batch_load(WorkflowRun(id="r", workflow_id="w", started_at="t", sandbox_limit=3), [a], [a, b])
    assert "Load of this batch" in block and "300 MB" in block and "highest 300 MB" in block
