"""One case's trace -> its result: code checks first, then the judges.

Where code and a judge look at the same dimension, a failed code check wins:
an expected value missing from the answer is a fail whatever the judge says.
A passed code check does not make a pass by itself; the judge still looks at
what code cannot see. Without judges (--no-judge) only code decides, and what
code cannot decide is "unknown".
"""

import asyncio
from typing import Any

from . import checks
from .cases import EvalCase
from .judges import Judges
from .trace import Trace

NOT_JUDGED = {"status": "unknown", "reason": "not judged (run with judges to get a verdict)"}


def _combine(code_status: str | None, code_reason: str, judge: dict[str, Any] | None) -> dict[str, Any]:
    if code_status == "fail":
        extra = f" Judge ({judge['status']}): {judge['reason']}" if judge else ""
        return {"status": "fail", "reason": code_reason + extra}
    if judge is None:
        return {"status": code_status or "unknown", "reason": code_reason or NOT_JUDGED["reason"]}
    return {"status": judge["status"], "reason": judge["reason"]}


async def _skip() -> None:
    return None


async def evaluate(
    case: EvalCase, trace: Trace, judges: Judges | None, stream_errors: list[str] | None = None
) -> dict[str, Any]:
    usage = checks.tool_usage(trace, case)
    values = checks.expected_values(trace.answer, case)
    outputs = checks.required_outputs(trace, case)
    recovery = checks.recovery(trace)
    efficiency = checks.efficiency(trace)
    grounding = checks.numeric_grounding(trace, " ".join(case.turns))

    instruction_checks = usage["checks"] + outputs
    if judges is None:
        judged = (None, None, None, None)
    else:
        judged = await asyncio.gather(
            judges.correctness(trace, case, values) if trace.answer else _skip(),
            judges.groundedness(trace, grounding) if trace.answer else _skip(),
            judges.instruction_following(trace, instruction_checks),
            judges.execution_strategy(trace, case, recovery, efficiency),
        )
    correct_j, grounded_j, instruction_j, strategy_j = judged

    if not trace.answer:
        correct = {"status": "fail", "reason": "No final answer: the turn stopped before answering."}
        grounded = {"status": "unknown", "reason": "No final answer to check."}
    else:
        correct = _combine(values["status"], f"Missing from the answer: {', '.join(values['missing'])}."
                           if values["missing"] else "Every expected value is in the answer.", correct_j)
        grounded = (
            {"status": grounded_j["status"], "reason": grounded_j["reason"]} if grounded_j else NOT_JUDGED
        )

    failed_checks = [c["check"] for c in instruction_checks if c["pass"] is False]
    passed = instruction_checks and all(c["pass"] for c in instruction_checks)
    instruction = _combine(
        "fail" if failed_checks else "pass" if passed else None,
        f"Failed: {', '.join(failed_checks)}." if failed_checks else "",
        instruction_j,
    )

    return {
        "case_id": case.id,
        "question": trace.question,
        "final_answer": trace.answer,
        "stopped": not trace.answer,
        "stream_errors": stream_errors or [],
        "artifacts": trace.artifacts,
        "correct": {**correct, "expected_values": values, "judge": correct_j},
        "grounded": {
            **grounded,
            **grounding,
            "unsupported_claims": (grounded_j or {}).get("unsupported_claims", []),
            "judge": grounded_j,
        },
        "instruction_following": {**instruction, "checks": instruction_checks, "judge": instruction_j},
        "execution_strategy": {**(strategy_j or NOT_JUDGED), "judge": strategy_j},
        "tool_usage": usage,
        "recovery": recovery,
        "efficiency": efficiency,
        "context": trace.context,
    }
