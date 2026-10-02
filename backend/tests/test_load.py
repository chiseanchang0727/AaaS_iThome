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
