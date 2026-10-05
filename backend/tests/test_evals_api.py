"""The read-only eval routes, over a run written into a temp dir."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.evals import evals_router, summarize, system_router


def result(arm, turn, correct=True, **extra):
    return {
        "conversation": "chat", "turn": turn, "arm": arm, "run": 1, "prompt": "q", "answer": "a",
        "stopped": False, "sent_turns": None if arm == "checkpointer" else [], "steps": 1, "queries": 1,
        "skill_reads": 0, "input_tokens": 1000, "first_call_tokens": 600, "correct": correct,
        "missing": [], "ungrounded": [], **extra,
    }


RESULTS = [
    result("checkpointer", "chat.1"), result("checkpointer", "chat.2", correct=None),
    result("jev", "chat.1", correct=False, ungrounded=["12"]), result("jev", "chat.2", correct=None, first_call_tokens=400),
]


def write_run(root, run_id="20261001-120000"):
    run_dir = root / run_id
    (run_dir / "history").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id, "created_at": "2026-10-01T12:00:00+08:00", "model": "anthropic:claude-sonnet-4-6",
        "sandbox": False, "repeats": 1, "arms": ["checkpointer", "jev"], "context_filter": {"keep_last": 1},
        "conversations": [{"id": "chat", "about": "", "turns": [{"id": "chat.1", "prompt": "q", "expect": [], "note": ""}]}],
        "results": RESULTS,
    }))
    (run_dir / "history" / "chat-jev-run1.jsonl").write_text(
        '{"id": "a", "previous": null, "turn": 1, "role": "user", "content": "q"}\n'
    )
    return run_dir


def client(root) -> TestClient:
    app = FastAPI()
    app.include_router(evals_router(root))
    return TestClient(app)


def test_summary_counts_per_arm():
    s = summarize(RESULTS)
    assert s["checkpointer"] == {
        "turns": 2, "checked": 1, "correct": 1, "grounded": 2, "steps": 2, "queries": 2,
        "skill_reads": 0, "tokens_per_turn": 1000, "first_call_per_turn": 600,
    }
    assert (s["jev"]["correct"], s["jev"]["grounded"], s["jev"]["first_call_per_turn"]) == (0, 1, 500)


def test_runs_are_listed_newest_first_with_a_summary(tmp_path):
    write_run(tmp_path, "20261001-120000")
    write_run(tmp_path, "20261002-090000")
    runs = client(tmp_path).get("/api/evals/context/runs").json()
    assert [r["id"] for r in runs] == ["20261002-090000", "20261001-120000"]
    assert runs[0]["conversations"] == ["chat"]
    assert runs[0]["summary"]["jev"]["turns"] == 2


def test_no_runs_is_an_empty_list(tmp_path):
    assert client(tmp_path / "missing").get("/api/evals/context/runs").json() == []


def test_one_run_has_its_results_and_summary(tmp_path):
    write_run(tmp_path)
    run = client(tmp_path).get("/api/evals/context/runs/20261001-120000").json()
    assert len(run["results"]) == 4
    assert run["summary"]["checkpointer"]["correct"] == 1


def test_a_conversations_history_is_its_jsonl_lines(tmp_path):
    write_run(tmp_path)
    lines = client(tmp_path).get("/api/evals/context/runs/20261001-120000/history/chat/jev").json()
    assert lines == [{"id": "a", "previous": None, "turn": 1, "role": "user", "content": "q"}]


def test_unknown_or_malformed_ids_are_404(tmp_path):
    write_run(tmp_path)
    c = client(tmp_path)
    assert c.get("/api/evals/context/runs/20990101-000000").status_code == 404
    assert c.get("/api/evals/context/runs/..%2F..%2Fetc").status_code == 404
    assert c.get("/api/evals/context/runs/20261001-120000/history/chat/checkpointer").status_code == 404
    assert c.get("/api/evals/context/runs/20261001-120000/history/Chat/jev").status_code == 404


# --- the whole-system eval ----------------------------------------------------


def system_result(case_id, correct="pass", grounded="pass", strategy="pass"):
    return {
        "case_id": case_id, "final_answer": "a",
        "correct": {"status": correct}, "grounded": {"status": grounded},
        "instruction_following": {"status": "pass"}, "execution_strategy": {"status": strategy},
        "efficiency": {"total_tokens": 1000, "runtime_seconds": 2.5},
    }


def write_system_run(root, run_id="20261005-120000"):
    run_dir = root / run_id
    (run_dir / "history").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id, "created_at": "2026-10-05T12:00:00+08:00", "model": "m", "judge_model": "j",
        "sandbox": True, "rejudged_from": None, "cases": [],
        "results": [system_result("a"), system_result("b", correct="fail", strategy="judge_error")],
    }))
    (run_dir / "history" / "a.jsonl").write_text('{"id": "x", "previous": null, "turn": 1, "role": "user", "content": "q"}\n')


def system_client(root) -> TestClient:
    app = FastAPI()
    app.include_router(system_router(root))
    return TestClient(app)


def test_system_runs_are_listed_with_pass_counts(tmp_path):
    write_system_run(tmp_path)
    [run] = system_client(tmp_path).get("/api/evals/system/runs").json()
    assert run["cases"] == ["a", "b"]
    assert run["summary"] == {
        "cases": 2, "correct": 1, "grounded": 2, "instruction_following": 2, "execution_strategy": 1,
        "judge_errors": 1, "total_tokens": 2000, "runtime_seconds": 5.0,
    }


def test_system_run_and_case_history(tmp_path):
    write_system_run(tmp_path)
    client = system_client(tmp_path)
    assert client.get("/api/evals/system/runs/20261005-120000").json()["summary"]["cases"] == 2
    assert client.get("/api/evals/system/runs/20261005-120000/history/a").json()[0]["content"] == "q"
    assert client.get("/api/evals/system/runs/20261005-120000/history/b").status_code == 404
    assert client.get("/api/evals/system/runs/nope").status_code == 404
    assert client.get("/api/evals/system/runs/20261005-120000/history/..%2Fx").status_code == 404
