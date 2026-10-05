"""Everything about a run that code can check: what happened, not whether it was good.

Measurements (steps, tokens, runtime) are recorded, never marked pass or fail.
Checks against a case's expectations are pass / fail, or None when the case
sets no expectation. Recovery is recorded as facts (what failed, what came
next) for the execution-strategy judge to weigh.
"""

import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from api.load import resolution, step_roles
from evals.grounding.scoring import contains_number, numbers_in, ungrounded

from .cases import EvalCase
from .trace import ToolStep, Trace

_EXIT = re.compile(r"\[Command (?:succeeded|failed) with exit code (-?\d+)\]\s*$")
_TABLE = re.compile(r"^\s*\|.*\|\s*$\n^\s*\|?\s*:?-{3,}", re.MULTILINE)
_CODE = re.compile(r"`[^`]*`")
CHART_FILES = {".html", ".png", ".svg", ".jpg", ".jpeg", ".webp"}
REPORT_FILES = {".html", ".md", ".pdf", ".docx"}
TABLE_FILES = {".csv", ".xlsx"}
FILE_CHANGES = ("write_file", "edit_file")


def skill_name(step: ToolStep) -> str | None:
    """The skill a read_file call read (/skills/<name>/SKILL.md), else None."""
    if step.name != "read_file":
        return None
    path = PurePosixPath(str(step.args.get("file_path", "")))
    return path.parent.name if path.name == "SKILL.md" else None


def tool_usage(trace: Trace, case: EvalCase) -> dict[str, Any]:
    names = [t.name for t in trace.tools]
    skills = [s for t in trace.tools if (s := skill_name(t))]
    checks = (
        [{"check": f"uses {t}", "pass": t in names} for t in case.required_tools]
        + [{"check": f"does not use {t}", "pass": t not in names} for t in case.forbidden_tools]
        + [{"check": f"reads the {s} skill", "pass": s in skills} for s in case.required_skills]
    )
    return {
        "tools": dict(Counter(names)),
        "tool_calls": len(names),
        "queries": names.count("query_database"),
        "exports": names.count("export_query"),
        "skill_reads": len(skills),
        "skills_read": sorted(set(skills)),
        "code_executions": names.count("execute"),
        "checks": checks,
        "status": _status(checks),
    }


def efficiency(trace: Trace) -> dict[str, Any]:
    names = [t.name for t in trace.tools]
    return {
        "steps": len(names),
        "queries": names.count("query_database"),
        "skill_reads": sum(skill_name(t) is not None for t in trace.tools),
        "code_executions": names.count("execute"),
        "model_calls": trace.model_calls,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "total_tokens": trace.input_tokens + trace.output_tokens,
        "runtime_seconds": trace.runtime_seconds,
    }


def expected_values(answer: str, case: EvalCase) -> dict[str, Any]:
    """Each expected value in the answer? Numbers at the answer's shown precision."""
    found, missing = [], []
    for label, value in case.expected_values.items():
        if isinstance(value, str):
            ok = value.lower() in answer.lower()
            shown = value
        else:
            ok = contains_number(answer, float(value))
            shown = f"{int(value):,}" if float(value).is_integer() else f"{float(value):,.6g}"
        (found if ok else missing).append(f"{label}={shown}")
    status = None if not case.expected_values else "pass" if not missing else "fail"
    return {"status": status, "found": found, "missing": missing}


def exit_code(step: ToolStep) -> int | None:
    """An execute result's exit code, from the line the tool ends it with."""
    match = _EXIT.search(step.result or "")
    return int(match.group(1)) if match else None


def ran_python(trace: Trace) -> list[str]:
    """Commands that ran Python and succeeded."""
    return [
        str(t.args.get("command", ""))[:200] for t in trace.tools
        if t.name == "execute" and "python" in str(t.args.get("command", "")) and exit_code(t) == 0
    ]


def required_outputs(trace: Trace, case: EvalCase) -> list[dict[str, Any]]:
    """One check per required output. Files are what the turn handed to the user."""
    suffixes = {a: PurePosixPath(a).suffix.lower() for a in trace.artifacts}
    checks = []
    for output in case.required_outputs:
        if output == "chart":
            evidence = [a for a, s in suffixes.items() if s in CHART_FILES]
        elif output == "report":
            evidence = [a for a, s in suffixes.items() if s in REPORT_FILES]
        elif output == "table":
            evidence = [a for a, s in suffixes.items() if s in TABLE_FILES]
            if _TABLE.search(trace.answer):
                evidence.append("markdown table in the answer")
        elif output == "python":
            evidence = ran_python(trace)
        else:
            checks.append({"check": f"makes a {output}", "pass": None, "evidence": ["no code check for this output"]})
            continue
        checks.append({"check": f"makes a {output}", "pass": bool(evidence), "evidence": evidence})
    return checks


def failed(step: ToolStep) -> bool:
    if step.name == "execute":
        code = exit_code(step)
        return step.error or (code is not None and code != 0)
    return step.error


def recovery(trace: Trace) -> dict[str, Any]:
    """What failed, and what the agent did next with the same tool.

    After a failure, the next call to the same tool is a retry:
        changed     other arguments, or a file was written/edited in between
        identical   the same arguments, nothing changed in between
    No later call to that tool: none (it gave up on it, or moved on).
    """
    errors = []
    for i, step in enumerate(trace.tools):
        if not failed(step):
            continue
        later = trace.tools[i + 1:]
        retry = next((j for j, t in enumerate(later) if t.name == step.name), None)
        if retry is None:
            next_move = "none"
        else:
            edited = any(t.name in FILE_CHANGES for t in later[:retry])
            next_move = "changed" if edited or later[retry].args != step.args else "identical"
        errors.append({
            "tool": step.name,
            "args": _clip(_describe(step), 300),
            "result": _clip(step.result or "(no result: the turn stopped)", 600),
            "next": next_move,
        })

    roles = step_roles(trace.records)
    ooms = [s for s in trace.code_steps if s.get("signal") == 9 and _near_limit(s)]
    return {
        "errors": errors,
        "retries": sum(e["next"] != "none" for e in errors),
        "identical_retries": sum(e["next"] == "identical" for e in errors),
        "oom_events": len(ooms),
        "oom_resolved_by": resolution(trace.records, roles),
        "sandbox_moves": sum(e.get("event") == "switched" for e in trace.events),
        "sandbox_events": [e.get("event") for e in trace.events],
        "code_steps": [
            {
                "command": _clip(s.get("command", ""), 300),
                "exit_code": s.get("exit_code"),
                "seconds": s.get("seconds"),
                "peak_memory_mb": s.get("peak_memory_mb"),
                "memory_limit_mb": s.get("memory_limit_mb"),
                "out_of_memory": s in ooms,
                "strategy": roles.get(s.get("id", ""), {}).get("strategy"),
            }
            for s in trace.code_steps
        ],
    }


def numeric_grounding(trace: Trace, asked: str) -> dict[str, Any]:
    """Numbers in the answer that no tool result showed: a first pass, not proof.

    A number the agent derived itself (a difference, a share) also lands in
    `ungrounded_values`; the groundedness judge decides which are fine.
    """
    answer_values = list(dict.fromkeys(n.text for n in numbers_in(_CODE.sub(" ", trace.answer))))
    observed = list(dict.fromkeys(n.text for o in trace.observations for n in numbers_in(o)))
    return {
        "answer_values": answer_values,
        "observed_values": observed[:300],
        "observed_count": len(observed),
        "ungrounded_values": ungrounded(trace.answer, trace.observations, asked),
    }


def _near_limit(step: dict[str, Any]) -> bool:
    limit, peak = step.get("memory_limit_mb"), step.get("peak_memory_mb")
    return bool(limit) and peak is not None and peak >= 0.8 * limit


def _status(checks: list[dict[str, Any]]) -> str | None:
    if not checks:
        return None
    return "pass" if all(c["pass"] for c in checks) else "fail"


def _describe(step: ToolStep) -> str:
    for key in ("sql", "command", "file_path"):
        if key in step.args:
            return str(step.args[key])
    return str(step.args)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …"
