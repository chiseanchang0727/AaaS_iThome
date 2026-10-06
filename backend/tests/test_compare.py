"""Jev vs full: pairing saved conversations by their questions, judging pairs, and the routes."""

import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import HistoryStore
from api.compare import compare_router, side_summary
from evals.system.compare import evaluate_pairs, pair_histories


def conversation(history: HistoryStore, thread: str, turns: list[tuple[str, str]], jev: bool = False) -> None:
    for n, (question, answer) in enumerate(turns, start=1):
        log = history.start_turn(thread, question)
        if jev:
            log.write({"role": "context", "sent_turns": list(range(1, n)), "earlier_turns": n - 1, "seconds": 0.2})
        log.write({"role": "assistant", "content": "", "usage": {"input_tokens": 1000 * n, "output_tokens": 10},
                   "tool_calls": [{"id": f"q{n}", "name": "query_database", "args": {"sql": "SELECT 1"}}]})
        log.write({"role": "tool", "tool_call_id": f"q{n}", "name": "query_database", "content": '[{"n": 7}]', "error": False})
        log.write({"role": "assistant", "content": answer, "usage": {"input_tokens": 500, "output_tokens": 20}})


def folders(tmp_path):
    full, jev = HistoryStore(tmp_path / "full"), HistoryStore(tmp_path / "jev")
    conversation(full, "f1", [("Top channel?", "A, 7 views."), ("How many videos does it have?", "7.")])
    conversation(jev, "j1", [("Top channel?", "A, 7 views."), ("How many videos does it have?", "7.")], jev=True)
    conversation(full, "f2", [("Only asked here?", "7.")])
    conversation(jev, "j2", [("Only asked with Jev?", "7.")], jev=True)
    return full, jev


def test_conversations_pair_by_the_same_questions_in_the_same_order(tmp_path):
    full, jev = folders(tmp_path)
    found = pair_histories(full, jev)
    assert found["pairs"] == [{"full": "f1", "jev": "j1", "questions": ["Top channel?", "How many videos does it have?"]}]
    assert [c["thread"] for c in found["only_full"]] == ["f2"]
    assert [c["thread"] for c in found["only_jev"]] == ["j2"]


def test_each_pair_is_judged_turn_by_turn_on_both_sides(tmp_path):
    full, jev = folders(tmp_path)
    run_dir = tmp_path / "runs" / "20261005-150000"
    rows = asyncio.run(evaluate_pairs(
        {"full": (full, tmp_path / "full" / "artifacts"), "jev": (jev, tmp_path / "jev" / "artifacts")},
        run_dir, None, {"judge_model": None},
    ))
    [row] = rows
    assert row["id"] == "f1" and row["full"]["thread"] == "f1" and row["jev"]["thread"] == "j1"
    assert [r["context"] for r in row["full"]["results"]] == [None, None]
    assert [r["context"]["sent_turns"] for r in row["jev"]["results"]] == [[], [1]]
    saved = json.loads((run_dir / "run.json").read_text())
    assert (saved["status"], saved["total"], len(saved["unpaired"]["full"])) == ("done", 1, 1)
    assert (run_dir / "history" / "jev" / "j1.jsonl").is_file()


def test_a_side_adds_up_its_turns():
    results = [
        {"correct": {"status": s}, "grounded": {"status": "pass"}, "instruction_following": {"status": "pass"},
         "execution_strategy": {"status": "judge_error"}, "efficiency": {"steps": 2, "total_tokens": 1000, "runtime_seconds": 1.5},
         "recovery": {"errors": []}}
        for s in ("pass", "fail", "unknown")
    ]
    s = side_summary(results)
    assert s["correct"] == {"pass": 1, "fail": 1, "unknown": 1}
    assert s["execution_strategy"] == {"pass": 0, "fail": 0, "unknown": 3}
    assert (s["turns"], s["steps"], s["total_tokens"], s["runtime_seconds"]) == (3, 6, 3000, 4.5)


def test_the_routes_start_a_comparison_and_show_it(tmp_path):
    full, jev = folders(tmp_path)
    app = FastAPI()
    app.include_router(compare_router(tmp_path / "runs", full=(full, tmp_path / "fa"), jev=(jev, tmp_path / "ja")))
    with TestClient(app) as client:
        assert client.get("/api/evals/compare/pairs").json() == {
            "available": True, "pairs": 1, "turns": 2, "only_full": 1, "only_jev": 1, "running": None,
        }
        run_id = client.post("/api/evals/compare/runs").json()["id"]
        for _ in range(100):
            run = client.get(f"/api/evals/compare/runs/{run_id}").json()
            if run["status"] == "done":
                break
        assert run["pairs"][0]["jev"]["summary"]["turns"] == 2
        assert client.get(f"/api/evals/compare/runs/{run_id}/history/jev/j1").status_code == 200
        assert client.get(f"/api/evals/compare/runs/{run_id}/history/other/j1").status_code == 404
        [listed] = client.get("/api/evals/compare/runs").json()
        assert (listed["pairs"], listed["status"]) == (1, "done")
