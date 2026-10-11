"""Adaptive runs of a saved analysis: measure, find what no longer fits, optimize, validate, run again.

The scenarios from the design: a small table runs as saved; a much larger
table with the same columns stops before exporting and, after Optimize, runs
as version 2 with the work in SQL; a memory problem with small inputs asks
for a script rewrite, not SQL; a version already rewritten for memory gets
the bigger sandbox; a failed optimization leaves the version as it was.

The sandbox is a local shell; scripts read real Parquet (pyarrow) and write
Plotly-style HTML, so the results check compares real numbers. The "agent"
calls the real save_analysis tool.
"""

import asyncio
from pathlib import PurePosixPath
from statistics import median

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.save_analysis import make_save_analysis
from agent.tools import to_parquet
from analyses import Analysis, AnalysisStore, Output, QueryInput
from analyses.budget import Limits, check_estimates, check_run
from analyses.models import Finding, InputMeasure, Measurements, OptimizationNote, RunRecord, SandboxMeasure
from analyses.optimize import request
from analyses.same_results import chart_values, differences
from api import ConversationManager
from api.analyses import analyses_router
from test_analyses import ShellProvider, no_datasets, shell, wait_for

LIMITS = Limits(hard_export_rows=100_000, max_export_bytes=200_000_000, soft_export_rows=50_000,
                soft_export_bytes=20_000_000, soft_run_seconds=120, memory_warning_ratio=0.8)

# Raw events: device, session, minute of each event.
RAW = [{"device_type": d, "session_id": s, "minute": float(m)}
       for d, sessions in {"desktop": {1: [0, 30], 2: [0, 10, 50]}, "mobile": {3: [0, 5], 4: [0, 2], 5: [0, 9]}}.items()
       for s, minutes in sessions.items() for m in minutes]
MEDIANS = {d: median(max(e["minute"] for e in RAW if e["session_id"] == s) for s in {e["session_id"] for e in RAW if e["device_type"] == d})
           for d in ("desktop", "mobile")}

NAIVE_SQL = "SELECT * FROM events WHERE minute >= 0"
PUSHED_SQL = ("SELECT device_type, percentile_cont(0.5) WITHIN GROUP (ORDER BY minutes) AS median_minutes "
              "FROM (SELECT device_type, session_id, MAX(minute) - MIN(minute) AS minutes FROM events "
              "WHERE minute >= 0 GROUP BY device_type, session_id) AS s GROUP BY device_type")

CHART = '''
def write_chart(medians):
    devices = sorted(medians)
    payload = json.dumps([{"type": "bar", "x": devices, "y": [medians[d] for d in devices]}])
    with open(os.path.join(os.environ["OUTPUT_DIR"], "sessions.html"), "w") as f:
        f.write(f'<div id="c"></div><script>Plotly.newPlot("c", {payload}, {{}})</script>')
'''

NAIVE_SCRIPT = '''import io, json, os, statistics
import pyarrow.parquet as pq
''' + CHART + '''
rows = pq.ParquetFile(os.path.join(os.environ["DATA_DIR"], "recent_events.parquet")).read().to_pylist()
sessions = {}
for r in rows:
    sessions.setdefault((r["device_type"], r["session_id"]), []).append(r["minute"])
by_device = {}
for (device, _), minutes in sessions.items():
    by_device.setdefault(device, []).append(max(minutes) - min(minutes))
write_chart({d: statistics.median(v) for d, v in by_device.items()})
'''

PUSHED_SCRIPT = '''import io, json, os
import pyarrow.parquet as pq
''' + CHART + '''
rows = pq.ParquetFile(os.path.join(os.environ["DATA_DIR"], "recent_events.parquet")).read().to_pylist()
write_chart({r["device_type"]: r["median_minutes"] for r in rows})
'''


EVENTS = {"device_type": "text", "session_id": "bigint", "minute": "double precision"}


async def fake_rows(sql: str) -> list[dict]:
    """information_schema.columns: `events` and `events_big` have the same columns."""
    return [{"table_name": t, "column_name": c, "data_type": d} for t in ("events", "events_big") for c, d in EVENTS.items()]


async def fake_query(sql: str) -> bytes:
    if "percentile_cont" in sql.lower():
        return to_parquet([{"device_type": d, "median_minutes": m} for d, m in sorted(MEDIANS.items())])
    return to_parquet(RAW)


async def fake_estimate(sql: str) -> dict:
    rows = 2 if "percentile_cont" in sql.lower() else 10_000_000 if "events_big" in sql else 25_000
    return {"rows": rows, "width": 42, "bytes": rows * 42}


def naive(**fields) -> Analysis:
    return Analysis(**{"id": "sessions1", "title": "Median session length by device",
                       "question": "Median session length by device?",
                       "inputs": [QueryInput(sql=NAIVE_SQL, file="recent_events.parquet")],
                       "script": NAIVE_SCRIPT, "outputs": [Output(file="sessions.html")], **fields})


def build(tmp_path, analysis: Analysis, agent=None, estimate=None, count=None):
    store = AnalysisStore(tmp_path / "saved")
    store.save(analysis)
    provider = ShellProvider(tmp_path)
    manager = ConversationManager(lambda sandbox, checkpointer: object(), provider, idle_seconds=900)
    app = FastAPI()
    app.include_router(analyses_router(
        store, manager, query=fake_query, query_rows=fake_rows, read_dataset=no_datasets,
        work_dir=PurePosixPath(tmp_path) / "work", account="test_user",
        estimate=estimate or fake_estimate, count=count, limits=LIMITS, ask_agent=agent,
    ))
    return app, store, provider


def save_tool(tmp_path, store, estimate=None, count=None):
    return make_save_analysis(shell(tmp_path), store, query=fake_query, read_dataset=no_datasets,
                              work_dir=PurePosixPath(tmp_path) / "check", estimate=estimate or fake_estimate,
                              limits=LIMITS, count=count)


def agent_saving(tmp_path, store, sql: str, script: str, asked: list[str]):
    """An agent that answers the optimization request by saving one recipe, as the real one would."""
    tool = save_tool(tmp_path, store)

    async def ask(conversation_id: str, message: str, history) -> str:
        asked.append(message)
        log = history.start_turn(conversation_id, message)
        answer = await tool.ainvoke({
            "title": "Median session length by device", "question": "Median session length by device?",
            "inputs": [{"kind": "query", "sql": sql, "file": "recent_events.parquet"}],
            "script": script, "outputs": ["sessions.html"], "replaces": "sessions1",
        }, config={"configurable": {"thread_id": conversation_id}})
        log.write({"role": "assistant", "content": answer})
        return answer

    return ask


def client_get_after(app, path):
    with TestClient(app) as client:
        return client.get(path).json()


def run_and_wait(client, sources=None):
    started = client.post("/api/analyses/sessions1/runs", json={"sources": sources or {}})
    assert started.status_code == 202, started.text
    return wait_for(client, "sessions1", started.json()["id"])


def wait_until_idle(client):
    for _ in range(300):
        if not client.get("/api/analyses/sessions1").json()["running"]:
            return client.get("/api/analyses/sessions1").json()
        asyncio.run(asyncio.sleep(0.02))
    raise AssertionError("still running")


# --- the scenarios -------------------------------------------------------------------------


def test_small_table_runs_as_saved(tmp_path):
    app, store, _ = build(tmp_path, naive())
    with TestClient(app) as client:
        record = run_and_wait(client)
    assert (record["status"], record["version"], record["findings"]) == ("done", 1, [])
    [measured] = record["measurements"]["inputs"]
    assert (measured["estimated_rows"], measured["rows"]) == (25_000, len(RAW))
    assert measured["bytes"] > 0 and measured["query_seconds"] is not None
    assert record["measurements"]["sandbox"]["seconds"] is not None
    assert store.get("sessions1").version == 1
    with TestClient(app) as client:
        limits = client.get("/api/analyses/sessions1").json()["limits"]
    assert (limits["hard_export_rows"], limits["soft_export_rows"]) == (100_000, 50_000)
    assert "soft_query_seconds" not in limits


def test_large_table_stops_before_exporting_then_optimizes_into_sql(tmp_path):
    asked: list[str] = []
    agent = agent_saving(tmp_path, AnalysisStore(tmp_path / "saved"), PUSHED_SQL, PUSHED_SCRIPT, asked)
    app, store, provider = build(tmp_path, naive(), agent)
    with TestClient(app) as client:
        stopped = run_and_wait(client, {"events": "events_big"})
        assert stopped["status"] == "needs_optimization"
        assert stopped["findings"] == [{"limit": "hard", "kind": "data_movement", "reason": (
            "recent_events.parquet (from events_big) would export about 10.0M rows into the sandbox "
            "(estimated); the limit is 100,000")}]
        assert stopped["measurements"]["inputs"][0]["rows"] is None  # nothing was exported
        assert provider.created == []  # and no sandbox was started

        optimizing = client.post(f"/api/analyses/sessions1/runs/{stopped['id']}/optimize")
        assert optimizing.status_code == 202
        state = wait_until_idle(client)

    assert "Move the reduction into SQL" in asked[0] and "events -> events_big" in asked[0]
    v2 = store.get("sessions1")
    assert v2.version == 2 and v2.inputs[0].sql == PUSHED_SQL
    assert (v2.optimization.from_version, v2.optimization.kind, v2.optimization.sources) == (1, "data_movement", {"events": "events_big"})
    assert v2.optimization.results_check == "Its results match version 1 on the saved tables."
    assert v2.conversation is None and v2.optimization.conversation == f"optimize-{stopped['id']}"  # v1 had no chat
    assert (tmp_path / "saved" / "sessions1" / "versions" / "1.json").is_file()
    assert not state["optimizing"]

    versions = client_get_after(app, "/api/analyses/sessions1/versions")
    assert [(v["version"], v["inputs"][0]["sql"]) for v in versions] == [(1, NAIVE_SQL), (2, PUSHED_SQL)]
    assert versions[0]["optimization"] is None and versions[1]["optimization"]["from_version"] == 1
    transcript = client_get_after(app, f"/api/analyses/sessions1/optimizations/optimize-{stopped['id']}")
    assert transcript[0]["role"] == "user" and transcript[0]["content"].startswith("Optimize the saved analysis sessions1")
    assert (tmp_path / "saved" / "sessions1" / "optimizations" / f"optimize-{stopped['id']}.jsonl").is_file()

    runs = {r["id"]: r for r in state["runs"]}
    assert runs[stopped["id"]]["optimization"]["status"] == "done"
    assert runs[stopped["id"]]["optimization"]["new_version"] == 2
    [again] = [r for r in state["runs"] if r.get("optimized_from") == 1]
    assert (again["status"], again["version"], again["sources"]) == ("done", 2, {"events": "events_big"})
    assert again["measurements"]["inputs"][0]["estimated_rows"] == 2
    assert again["measurements"]["inputs"][0]["rows"] == len(MEDIANS)  # 2 rows crossed, not 10M


def test_memory_problem_with_small_inputs_asks_for_a_script_rewrite_not_sql():
    run = RunRecord(id="20261007-100000-aaaa", started_at="t", status="needs_optimization", trigger="run",
                    measurements=Measurements(inputs=[InputMeasure(file="recent_events.parquet", rows=2_000)],
                                              sandbox=SandboxMeasure(seconds=4, peak_memory_mb=1010, memory_limit_mb=1024, killed=True)))
    run.findings = check_run(run.measurements, LIMITS)
    assert [(f.limit, f.kind) for f in run.findings] == [("hard", "sandbox_memory")]
    message = request(naive(), run, LIMITS)
    assert "Keep the SQL unless it is the cause" in message and "Move the reduction into SQL" not in message


def test_a_version_rewritten_for_memory_that_still_runs_out_gets_the_bigger_sandbox(tmp_path):
    killed = NAIVE_SCRIPT + "import signal; os.kill(os.getpid(), signal.SIGKILL)\n"
    rewritten = naive(script=killed, version=2, optimization=OptimizationNote(
        from_version=1, kind="sandbox_memory", reason="the script ran out of memory"))
    app, _, _ = build(tmp_path, rewritten)
    with TestClient(app) as client:
        record = run_and_wait(client)
    assert record["notes"] and record["notes"][0].startswith("No bigger sandbox is available")  # the existing path
    assert record["status"] == "needs_optimization"


def test_a_failed_optimization_leaves_the_version_unchanged(tmp_path):
    asked: list[str] = []
    store = AnalysisStore(tmp_path / "saved")
    wrong = PUSHED_SCRIPT.replace('r["median_minutes"]', 'r["median_minutes"] * 2')  # different results
    app, store, _ = build(tmp_path, naive(), agent_saving(tmp_path, store, PUSHED_SQL, wrong, asked))
    with TestClient(app) as client:
        stopped = run_and_wait(client, {"events": "events_big"})
        client.post(f"/api/analyses/sessions1/runs/{stopped['id']}/optimize")
        state = wait_until_idle(client)
    record = next(r for r in state["runs"] if r["id"] == stopped["id"])
    assert record["optimization"]["status"] == "failed"
    assert "gives different results than version 1" in record["optimization"]["message"]
    assert store.get("sessions1").version == 1 and not state["optimizing"]
    assert not (tmp_path / "saved" / "sessions1" / "versions").exists()


def test_a_failed_optimization_can_be_tried_again_in_a_new_conversation(tmp_path):
    asked: list[str] = []
    wrong = PUSHED_SCRIPT.replace('r["median_minutes"]', 'r["median_minutes"] * 2')
    agent = agent_saving(tmp_path, AnalysisStore(tmp_path / "saved"), PUSHED_SQL, wrong, asked)
    app, store, _ = build(tmp_path, naive(), agent)
    with TestClient(app) as client:
        stopped = run_and_wait(client, {"events": "events_big"})
        client.post(f"/api/analyses/sessions1/runs/{stopped['id']}/optimize")
        wait_until_idle(client)
        again = client.post(f"/api/analyses/sessions1/runs/{stopped['id']}/optimize")
        assert again.status_code == 202
        state = wait_until_idle(client)
    o = next(r for r in state["runs"] if r["id"] == stopped["id"])["optimization"]
    assert o["conversation"] == f"optimize-{stopped['id']}-2"
    assert [e["conversation"] for e in o["earlier"]] == [f"optimize-{stopped['id']}"]
    assert [e["status"] for e in o["earlier"]] == ["failed"]
    assert len(asked) == 2


def test_an_optimization_that_still_moves_too_much_is_refused(tmp_path):
    store = AnalysisStore(tmp_path / "saved")
    store.save(naive())
    store.start_optimizing("sessions1", {"run": "r", "sources": {"events": "events_big"}, "kind": "data_movement",
                                         "reason": "too many rows"})
    answer = asyncio.run(save_tool(tmp_path, store).ainvoke({
        "title": "Median session length by device", "question": "q",
        "inputs": [{"kind": "query", "sql": "SELECT device_type, session_id, minute FROM events", "file": "recent_events.parquet"}],
        "script": NAIVE_SCRIPT, "outputs": ["sessions.html"], "replaces": "sessions1",
    }, config={"configurable": {"thread_id": "t"}}))
    assert answer.startswith("NOT SAVED. This is an optimization of version 1")
    assert "would export about 10.0M rows" in answer
    assert store.get("sessions1").version == 1


def test_optimize_needs_a_run_that_broke_a_limit(tmp_path):
    app, _, _ = build(tmp_path, naive())
    with TestClient(app) as client:
        record = run_and_wait(client)
        refused = client.post(f"/api/analyses/sessions1/runs/{record['id']}/optimize")
    assert refused.status_code == 400 and "nothing to optimize" in refused.json()["detail"]


# --- the pieces ------------------------------------------------------------------------------


def test_where_the_bottleneck_is():
    big = InputMeasure(file="a.parquet", rows=80_000, bytes=3_000_000, query_seconds=1)
    small = InputMeasure(file="a.parquet", rows=1_000, bytes=30_000, query_seconds=0.1)
    oom = SandboxMeasure(seconds=3, peak_memory_mb=1000, memory_limit_mb=1024, killed=True)
    slow = SandboxMeasure(seconds=400, peak_memory_mb=100, memory_limit_mb=1024)
    kinds = lambda m: [(f.limit, f.kind) for f in check_run(m, LIMITS)]  # noqa: E731
    assert kinds(Measurements(inputs=[big], sandbox=oom)) == [("hard", "data_movement"), ("soft", "data_movement")]
    assert kinds(Measurements(inputs=[small], sandbox=oom)) == [("hard", "sandbox_memory")]
    assert kinds(Measurements(inputs=[small], sandbox=slow)) == [("soft", "sandbox_runtime")]
    assert kinds(Measurements(inputs=[big], sandbox=slow)) == [("soft", "data_movement")]  # not the script's fault
    assert kinds(Measurements(inputs=[small], sandbox=SandboxMeasure(seconds=1))) == []
    over = InputMeasure(file="a.parquet", estimated_rows=500, estimated_bytes=300_000_000)
    assert check_estimates(Measurements(inputs=[over]), LIMITS)[0].reason.endswith("the limit is 200.0 MB")


def test_chart_data_is_read_from_plotly_json_and_base64_arrays():
    import base64
    from array import array

    plain = b'<script>Plotly.newPlot("c", [{"x": ["a", "b"], "y": [1.5, 2.0]}], {})</script>'
    packed = array("d", [2.0, 1.5]).tobytes()
    binary = ('<script>Plotly.newPlot( "c" , [{"x": ["b", "a"], "y": {"dtype": "f8", "bdata": "'
              + base64.b64encode(packed).decode() + '"}}], {})</script>').encode()
    assert chart_values(plain) == chart_values(binary) == ([1.5, 2.0], ["a", "b"])
    assert differences({"c.html": plain}, {"c.html": binary}) == ([], [])
    changed, _ = differences({"c.html": plain}, {"c.html": plain.replace(b"2.0", b"2.5")})
    assert changed == ["c.html: 1 of 2 values changed by more than 1%: 2 -> 2.5"]
    assert differences({"c.html": b"<p>no chart</p>"}, {"c.html": plain}) == ([], ["c.html has no chart data to compare"])


def test_the_request_carries_the_recipe_measurements_and_goal():
    run = RunRecord(id="20261007-100000-aaaa", started_at="t", status="needs_optimization", trigger="run",
                    sources={"events": "events_big"}, findings=[Finding(limit="hard", kind="data_movement", reason="too big")],
                    measurements=Measurements(inputs=[InputMeasure(file="recent_events.parquet", tables=["events_big"],
                                                                   estimated_rows=10_048_594, estimated_bytes=422_000_000)]))
    message = request(naive(), run, LIMITS)
    for part in ('Optimize the saved analysis sessions1, "Median session length by device" (version 1)',
                 "events -> events_big", NAIVE_SQL, "statistics.median",
                 "recent_events.parquet from events_big: estimated 10.0M rows (422.0 MB)",
                 "Problem (hard limit, data movement): too big.", 'replaces="sessions1"'):
        assert part in message, part


def test_the_agent_reads_why_a_version_was_optimized(tmp_path):
    import json

    from agent.save_analysis import make_analysis_readers

    store = AnalysisStore(tmp_path / "saved")
    store.save(naive(version=2, optimization=OptimizationNote(from_version=1, kind="data_movement", reason="too big",
                                                              sources={"events": "events_big"})))
    _, read = make_analysis_readers(store)
    optimized = json.loads(read.invoke({"analysis_id": "sessions1"}))["optimized"]
    assert (optimized["from_version"], optimized["sources"]) == (1, {"events": "events_big"})
    assert "a user clicked Optimize" in optimized["how"]


def test_a_slow_query_with_a_small_export_is_not_flagged():
    """Query time is measured but has no target: slowness there is the database's work, not data movement."""
    slow = InputMeasure(file="a.parquet", rows=3, bytes=1_000, query_seconds=6.7)
    assert check_run(Measurements(inputs=[slow], sandbox=SandboxMeasure(seconds=1)), LIMITS) == []


# --- when EXPLAIN is far off: count before refusing ------------------------------------


async def blind_estimate(sql: str) -> dict:
    """A planner without statistics for a GROUP BY expression: it expects as many groups as rows."""
    rows = 10_000_000 if "events_big" in sql else 25_000
    return {"rows": rows, "width": 42, "bytes": rows * 42}


async def real_count(sql: str) -> int:
    return len(MEDIANS) if "percentile_cont" in sql.lower() else (10_000_000 if "events_big" in sql else 25_000)


def test_a_correct_rewrite_is_not_refused_because_the_estimate_is_far_off(tmp_path):
    """The bug behind 'stopped after 40 steps': a GROUP BY the planner overestimated was refused every time."""
    store = AnalysisStore(tmp_path / "saved")
    store.save(naive())
    store.start_optimizing("sessions1", {"run": "r", "sources": {"events": "events_big"}, "kind": "data_movement",
                                         "reason": "too many rows"})
    answer = asyncio.run(save_tool(tmp_path, store, estimate=blind_estimate, count=real_count).ainvoke({
        "title": "Median session length by device", "question": "q",
        "inputs": [{"kind": "query", "sql": PUSHED_SQL, "file": "recent_events.parquet"}],
        "script": PUSHED_SCRIPT, "outputs": ["sessions.html"], "replaces": "sessions1",
    }, config={"configurable": {"thread_id": "t"}}))
    assert answer.startswith("Updated analysis sessions1 to version 2 (optimized)"), answer
    # without a count, the same estimate refuses it, which is what happened
    store2 = AnalysisStore(tmp_path / "saved2")
    store2.save(naive())
    store2.start_optimizing("sessions1", {"run": "r", "sources": {"events": "events_big"}, "kind": "data_movement",
                                          "reason": "too many rows"})
    refused = asyncio.run(save_tool(tmp_path, store2, estimate=blind_estimate).ainvoke({
        "title": "Median session length by device", "question": "q",
        "inputs": [{"kind": "query", "sql": PUSHED_SQL, "file": "recent_events.parquet"}],
        "script": PUSHED_SCRIPT, "outputs": ["sessions.html"], "replaces": "sessions1",
    }, config={"configurable": {"thread_id": "t"}}))
    assert refused.startswith("NOT SAVED")


def test_run_counts_before_stopping_and_says_so(tmp_path):
    pushed = naive(inputs=[QueryInput(sql=PUSHED_SQL, file="recent_events.parquet")], script=PUSHED_SCRIPT)
    app, _, provider = build(tmp_path, pushed, estimate=blind_estimate, count=real_count)
    with TestClient(app) as client:
        record = run_and_wait(client, {"events": "events_big"})
    assert (record["status"], record["findings"]) == ("done", [])  # the estimate said 10M; the count said 2
    i = record["measurements"]["inputs"][0]
    assert (i["estimated_rows"], i["counted_rows"], i["rows"]) == (10_000_000, 2, 2)

    app, _, provider = build(tmp_path / "raw", naive(), estimate=blind_estimate, count=real_count)
    with TestClient(app) as client:
        stopped = run_and_wait(client, {"events": "events_big"})
    assert stopped["status"] == "needs_optimization" and provider.created == []
    assert "would export 10.0M rows into the sandbox (counted in the database)" in stopped["findings"][0]["reason"]
