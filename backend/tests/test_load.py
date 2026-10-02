"""GET /api/load: code steps read back from conversation histories."""

import itertools

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.history import HistoryStore
from api.load import load_router
from sandboxes.metering import StepMeasure


def store(tmp_path):
    clock = (f"2026-10-02T10:{m:02d}:00Z" for m in itertools.count())
    return HistoryStore(tmp_path, clock=lambda: next(clock))


def step(command, run, cpu, peak, exit_code=0, signal=None, seconds=None):
    return StepMeasure(command, exit_code, seconds if seconds is not None else run + 1.5, run, cpu, peak,
                       signal, 1024)


def histories(tmp_path):
    history = store(tmp_path)
    alice = history.start_turn("c-alice", "chart the views", "alice")
    alice.record_step(step("python3 chart.py", 1.0, 0.9, 210))
    alice.record_step(step("python3 big.py", 0.7, 0.6, 973, exit_code=137, signal=9))
    bob = history.start_turn("c-bob", "median by device", "bob")
    bob.record_step(step("python3 duck.py", 0.5, 0.4, 144))
    history.start_turn("c-old", "from before accounts")  # no account, no steps
    return history


def client(history):
    app = FastAPI()
    app.include_router(load_router(history))
    return TestClient(app)


def test_every_step_newest_first_with_its_account_and_question(tmp_path):
    body = client(histories(tmp_path)).get("/api/load").json()
    assert [s["command"] for s in body["steps"]] == ["python3 duck.py", "python3 big.py", "python3 chart.py"]
    first = body["steps"][-1]
    assert (first["account"], first["conversation"], first["turn"], first["prompt"]) == (
        "alice", "c-alice", 1, "chart the views")
    assert body["accounts"] == ["alice", "bob"] and body["memory_limit_mb"] == 1024


def test_totals_overall_and_per_account(tmp_path):
    body = client(histories(tmp_path)).get("/api/load").json()
    assert body["summary"] == {
        "steps": 3, "conversations": 2, "run_seconds": 2.2, "cpu_seconds": 1.9, "overhead_seconds": 4.5,
        "peak_memory_mb": 973, "average_peak_memory_mb": 442.3, "out_of_memory": 1, "failed": 1,
    }
    assert body["per_account"]["bob"]["steps"] == 1 and body["per_account"]["alice"]["out_of_memory"] == 1


def test_one_account_and_a_limit(tmp_path):
    c = client(histories(tmp_path))
    alice = c.get("/api/load", params={"account": "alice"}).json()
    assert {s["account"] for s in alice["steps"]} == {"alice"} and alice["summary"]["steps"] == 2
    assert alice["accounts"] == ["alice", "bob"]  # still every account, for the picker
    assert len(c.get("/api/load", params={"limit": 1}).json()["steps"]) == 1


def test_out_of_memory_needs_a_kill_near_the_limit(tmp_path):
    history = store(tmp_path)
    log = history.start_turn("c", "q", "alice")
    log.record_step(step("killed small", 0.1, 0.1, 50, exit_code=137, signal=9))
    [s] = client(history).get("/api/load").json()["steps"]
    assert s["out_of_memory"] is False and s["exit_code"] == 137


def test_unmeasured_steps_count_but_add_nothing(tmp_path):
    history = store(tmp_path)
    history.start_turn("c", "q", "alice").record_step(StepMeasure("echo hi", 0, 1.6))
    summary = client(history).get("/api/load").json()["summary"]
    assert summary["steps"] == 1 and summary["run_seconds"] == 0 and summary["peak_memory_mb"] is None


def test_no_history_yet(tmp_path):
    body = client(HistoryStore(tmp_path / "none")).get("/api/load").json()
    assert body["steps"] == [] and body["summary"]["steps"] == 0 and body["accounts"] == []


def test_conversation_rows_count_moves_to_a_bigger_sandbox(tmp_path):
    history = store(tmp_path)
    log = history.start_turn("c-big", "the 3.2 GB matrix", "load-test")
    log.record_event("sandbox_ready", {"how": "warm", "memory_gb": 1})
    log.record_step(step("python3 m.py", 9.6, 9.4, 973, exit_code=137, signal=9))
    for event, fields in [("upgrade_started", {"from_gb": 1, "to_gb": 4}), ("bigger_created", {"seconds": 14}),
                          ("switched", {"memory_gb": 4}), ("old_deleted", {"seconds": 0.5})]:
        log.record_event(event, fields)
    log.record_step(StepMeasure("python3 m.py", 0, 36, 34.6, 34.8, 3084, None, 4096))
    history.start_turn("c-sql", "a plain SQL question", "load-test")  # no code: not listed

    [row] = client(history).get("/api/load").json()["conversations"]
    assert row["conversation"] == "c-big" and row["question"] == "the 3.2 GB matrix"
    assert (row["steps"], row["out_of_memory"], row["upgrades"], row["peak_memory_mb"], row["sandbox_gb"]) == (
        2, 1, 1, 3084, 4)
    assert (row["run_seconds"], row["cpu_seconds"]) == (44.2, 44.2)


def test_a_conversations_timeline_mixes_actions_steps_and_events(tmp_path):
    from langchain_core.messages import AIMessage

    history = store(tmp_path)
    log = history.start_turn("c", "the matrix", "load-test")
    log.record(AIMessage("", tool_calls=[{"id": "1", "name": "execute", "args": {"command": "python3 m.py"}, "type": "tool_call"}]))
    log.record_step(step("python3 m.py", 9.6, 9.4, 973, exit_code=137, signal=9))
    log.record_event("upgrade_started", {"from_gb": 1, "to_gb": 4})
    log.record(AIMessage("The mean is 0.5."))

    body = client(history).get("/api/load/conversations/c").json()
    assert body["account"] == "load-test"
    kinds = [(i["kind"], i.get("tool") or i.get("event")) for i in body["timeline"]]
    assert kinds == [("question", None), ("action", "execute"), ("step", None), ("event", "upgrade_started"), ("answer", None)]
    assert body["timeline"][1]["detail"] == "python3 m.py"
    assert body["timeline"][2]["out_of_memory"] is True
    assert body["timeline"][3]["to_gb"] == 4
    assert client(history).get("/api/load/conversations/nope").status_code == 404
    assert client(history).get("/api/load/conversations/..%2Fx").status_code == 404


def test_steps_are_marked_first_rewrite_same_code_or_bigger(tmp_path):
    from langchain_core.messages import AIMessage

    from api.load import resolution, step_roles

    def write(log, path):
        log.record(AIMessage("", tool_calls=[{"id": path, "name": "write_file", "args": {"file_path": path}, "type": "tool_call"}]))

    killed = dict(exit_code=137, signal=9)
    history = store(tmp_path)
    log = history.start_turn("c", "median with polars only", "alice")
    log.record_step(step("python3 check.py", 0.2, 0.2, 50))                    # first
    log.record_step(step("python3 median.py", 2, 2, 965, **killed))            # first, killed
    log.record_step(step("python3 median.py", 2, 2, 965, **killed))            # same code
    write(log, "/w/median.py")
    log.record_step(step("python3 median.py", 2, 2, 970, **killed))            # rewrite 1
    log.record_step(step("python3 counts.py", 1, 1, 217))                      # rewrite 2

    records = history.read("c")
    roles = [step_roles(records)[r["id"]] for r in records if r["role"] == "sandbox_step"]
    assert roles == [{"strategy": "first"}, {"strategy": "first"}, {"strategy": "same code"},
                     {"strategy": "rewrite", "rewrite": 1}, {"strategy": "rewrite", "rewrite": 2}]
    assert resolution(records, step_roles(records)) == "rewrite"

    [row] = client(history).get("/api/load").json()["conversations"]
    assert (row["rewrites"], row["resolved_by"]) == (2, "rewrite")
    timeline = client(history).get("/api/load/conversations/c").json()["timeline"]
    assert [i.get("rewrite") for i in timeline if i["kind"] == "step"] == [None, None, None, 1, 2]


def test_a_move_marks_later_steps_and_resolves_by_bigger_sandbox(tmp_path):
    from api.load import resolution, step_roles

    history = store(tmp_path)
    log = history.start_turn("c", "the matrix", "alice")
    log.record_step(step("python3 m.py", 2, 2, 980, exit_code=137, signal=9))
    log.record_event("switched", {"memory_gb": 4})
    log.record_step(StepMeasure("python3 m.py", 0, 6, 5, 5, 3085, None, 4096))
    records = history.read("c")
    roles = step_roles(records)
    assert [roles[r["id"]]["strategy"] for r in records if r["role"] == "sandbox_step"] == ["first", "bigger sandbox"]
    assert resolution(records, roles) == "bigger sandbox"


def test_no_kill_means_nothing_to_resolve_and_a_failed_end_is_not_resolved(tmp_path):
    from api.load import resolution, step_roles

    history = store(tmp_path)
    history.start_turn("ok", "q", "a").record_step(step("python3 x.py", 1, 1, 100))
    log = history.start_turn("bad", "q", "a")
    log.record_step(step("python3 x.py", 1, 1, 980, exit_code=137, signal=9))
    log.record_step(step("python3 y.py", 1, 1, 980, exit_code=137, signal=9))
    for thread, expected in (("ok", None), ("bad", "not resolved")):
        records = history.read(thread)
        assert resolution(records, step_roles(records)) == expected



def test_a_conversation_is_named_by_the_question_that_ran_code(tmp_path):
    history = store(tmp_path)
    history.start_turn("c", "hi, a big file is coming", "alice")          # small talk, no code
    history.start_turn("c", "median per device with polars", "alice").record_step(step("python3 m.py", 1, 1, 200))
    [row] = client(history).get("/api/load").json()["conversations"]
    assert row["question"] == "median per device with polars"
