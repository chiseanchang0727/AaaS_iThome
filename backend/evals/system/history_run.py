"""Evaluate conversations people already had, from saved history files.

The files to judge live in `server.eval_history_dir` (default
evals/system/chat_history/), apart from the app's own `history_dir`: copy the
conversations you want judged there.

Nothing is run again: each saved turn is checked and judged as it happened.
A turn becomes a case with no expectations (its question, and the turns
before it as setup), so only what needs no answer key applies: measurements,
recovery, numeric grounding, and the judges.

The run is saved like evals/system/run.py's (run.json, history/, artifacts/),
so the System page shows it the same way. run.json carries `status`
("running", then "done" or "failed") and `total`, and is rewritten after
every turn, so the page can show progress.
"""

import json
import shutil
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from api.history import HistoryStore

from .cases import EvalCase
from .evaluate import evaluate
from .judges import Judges
from .trace import build_trace


def history_cases(history: HistoryStore) -> list[tuple[EvalCase, str]]:
    """One case per saved turn, with the conversation it came from. Oldest conversation first."""
    found = []
    for thread_id in sorted(history.thread_ids(), key=lambda t: history.path(t).stat().st_mtime):
        records = history.read(thread_id)
        questions = {r["turn"]: r["content"] for r in records if r.get("role") == "user" and r.get("turn")}
        ordered = [questions[t] for t in sorted(questions)]
        for i, turn in enumerate(sorted(questions)):
            case = EvalCase(f"{thread_id}_t{turn}", questions[turn], setup=ordered[:i])
            found.append((case, thread_id))
    return found


def turn_artifacts(records: list[dict[str, Any]], turn: int, files: list[str]) -> list[str]:
    """The conversation's files this turn made: the ones its tool calls name.

    The history does not say which turn made a file, so this is a guess from
    the turn's tool calls (the script that wrote it names the path).
    """
    calls = json.dumps(
        [c for r in records if r.get("turn") == turn and r.get("role") == "assistant" for c in r.get("tool_calls") or []],
        ensure_ascii=False,
    )
    return [f for f in files if Path(f).name in calls]


def _files(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def _save(run_dir: Path, meta: dict[str, Any], results: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({**meta, "results": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


async def evaluate_saved_turn(
    history: HistoryStore, thread_id: str, case: EvalCase, artifacts_dir: Path, judges: Judges | None
) -> dict[str, Any]:
    """Check and judge one saved turn (the case's last). Its result names the files it made in `artifacts`."""
    records = history.read(thread_id)
    turn = len(case.turns)
    artifacts = turn_artifacts(records, turn, _files(artifacts_dir / thread_id))
    result = await evaluate(case, build_trace(records, turn, artifacts), judges)
    result["conversation"] = thread_id
    return result


def copy_artifacts(artifacts_dir: Path, thread_id: str, names: list[str], target_dir: Path) -> None:
    for name in names:
        target = target_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifacts_dir / thread_id / name, target)


async def evaluate_history(
    history: HistoryStore,
    artifacts_dir: Path,
    run_dir: Path,
    judges: Judges | None,
    meta: dict[str, Any],
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Check and judge every saved turn in `history`; save the run to `run_dir`."""
    cases = history_cases(history)
    meta = {
        **meta,
        "id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "history",
        "status": "running",
        "total": len(cases),
        "cases": [c.to_dict() for c, _ in cases],
    }
    copied = HistoryStore(run_dir / "history")
    results: list[dict[str, Any]] = []
    _save(run_dir, meta, results)
    try:
        for case, thread_id in cases:
            # One copy per case, named after it: the page finds a case's history by its id.
            copied.root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(history.path(thread_id), copied.path(case.id))
            result = await evaluate_saved_turn(history, thread_id, case, artifacts_dir, judges)
            copy_artifacts(artifacts_dir, thread_id, result["artifacts"], run_dir / "artifacts" / case.id)
            results.append(result)
            if on_result is not None:
                on_result(result)
            _save(run_dir, meta, results)
        meta["status"] = "done"
    except Exception:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc(limit=5)
        raise
    finally:
        _save(run_dir, meta, results)
    return results
