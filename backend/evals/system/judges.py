"""LLM judges for what code cannot check, one question per dimension.

Each judge reads its prompt from prompts/<name>.md (string.Template: $question,
$evidence, ...) and answers in a fixed structure. It sees only observable
behavior: messages, tool calls, results, code steps, sandbox events.

Jev (classify/jev_judge.py) answers with probabilities over labels and gives
no reasons or claim lists, so the judges use a chat model with structured
output instead.

A judge call that fails is recorded as status "judge_error" with the error as
the reason: never as a pass.
"""

import json
from pathlib import Path
from string import Template
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .cases import EvalCase
from .trace import Trace

PROMPTS = Path(__file__).resolve().parent / "prompts"
RESULT_CHARS = 2000
"""A tool result is clipped to this in what a judge sees."""
GROUNDING_RESULT_CHARS = 6000
"""More for the groundedness judge: the number it looks for may be far down."""
EVIDENCE_CHARS = 80_000

Status = Literal["pass", "fail", "unknown"]


class Verdict(BaseModel):
    status: Status = Field(description="pass, fail, or unknown when the evidence is not enough to tell")
    reason: str = Field(description="One to three sentences, pointing at the specific step or claim")


class UnsupportedClaim(BaseModel):
    claim: str = Field(description="The claim, quoted from the final answer")
    reason: str = Field(description="Why nothing the agent observed supports it")


class GroundednessVerdict(Verdict):
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)


def prompt(name: str, **values: str) -> str:
    return Template((PROMPTS / f"{name}.md").read_text(encoding="utf-8")).safe_substitute(values)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… ({len(text):,} characters in all)"


def render_conversation(trace: Trace) -> str:
    if not trace.earlier_turns:
        return "(none: this is the first message)"
    parts = [
        f"Turn {t['turn']}{'' if t['sent'] else ' (NOT sent to the agent this turn: it did not see it)'}\n"
        f"User: {t['question']}\nAgent: {t['answer'] or '(no answer)'}"
        for t in trace.earlier_turns
    ]
    if trace.context is not None:
        parts.insert(0, f"A context filter chose which earlier turns to send: {trace.context.get('sent_turns')}.")
    return "\n\n".join(parts)


def render_evidence(trace: Trace, result_chars: int = RESULT_CHARS, earlier: bool = False) -> str:
    """The turn as numbered steps, in the order they happened."""
    lines: list[str] = []
    if earlier and trace.earlier_observations:
        lines.append("Earlier tool results (from previous turns):")
        lines += [f"- {_clip(o, result_chars)}" for o in trace.earlier_observations]
        lines.append("")
    number: dict[str, int] = {}
    for r in trace.records:
        role = r.get("role")
        if role == "assistant":
            text = (r.get("content") or "").strip()
            if r.get("tool_calls"):
                if text:
                    lines.append(f"Agent said: {text}")
                for c in r["tool_calls"]:
                    number[c["id"]] = len(number) + 1
                    args = json.dumps(c.get("args") or {}, ensure_ascii=False)
                    lines.append(f"[{number[c['id']]}] call {c['name']} {_clip(args, 3000)}")
        elif role == "tool":
            n = number.get(r.get("tool_call_id", ""), "?")
            flag = " (error)" if r.get("error") else ""
            lines.append(f"[{n}] result{flag}:\n{_clip(r.get('content', ''), result_chars)}")
        elif role == "sandbox_step":
            peak, limit = r.get("peak_memory_mb"), r.get("memory_limit_mb")
            memory = f", peak memory {peak} MB" + (f" of {limit} MB" if limit else "") if peak is not None else ""
            killed = f", killed by signal {r['signal']}" if r.get("signal") else ""
            lines.append(f"    (code step: exit code {r.get('exit_code')}, {r.get('seconds')}s{memory}{killed})")
        elif role == "sandbox_event":
            extra = {k: v for k, v in r.items() if k not in ("id", "previous", "turn", "role", "event", "ts")}
            lines.append(f"    (sandbox event: {r.get('event')} {json.dumps(extra, ensure_ascii=False)})")
    if not number:
        lines.append("(no tool calls)")
    return _clip("\n".join(lines), EVIDENCE_CHARS)


def _bullets(items: list[str], empty: str = "(none)") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


class Judges:
    def __init__(self, model: str | BaseChatModel) -> None:
        self.name = model if isinstance(model, str) else type(model).__name__
        self.model = init_chat_model(model, temperature=0) if isinstance(model, str) else model

    async def _ask(self, schema: type[BaseModel], text: str) -> dict[str, Any]:
        try:
            verdict = await self.model.with_structured_output(schema).ainvoke(text)
            if verdict is None:
                raise ValueError("the model returned no structured answer")
            return verdict.model_dump() if isinstance(verdict, BaseModel) else schema.model_validate(verdict).model_dump()
        except Exception as e:  # a judge that fails is reported, never passed
            return {"status": "judge_error", "reason": f"{type(e).__name__}: {e}"}

    async def correctness(self, trace: Trace, case: EvalCase, values: dict[str, Any]) -> dict[str, Any]:
        if case.expected_values:
            expected = _bullets([f"found: {v}" for v in values["found"]] + [f"MISSING: {v}" for v in values["missing"]])
        else:
            expected = "(none given)"
        return await self._ask(Verdict, prompt(
            "correctness",
            conversation=render_conversation(trace), question=trace.question,
            expected_values=expected, expected_behavior=case.expected_behavior or "(none given)",
            evidence=render_evidence(trace), answer=trace.answer or "(no answer: the turn stopped)",
        ))

    async def groundedness(self, trace: Trace, grounding: dict[str, Any]) -> dict[str, Any]:
        return await self._ask(GroundednessVerdict, prompt(
            "groundedness",
            conversation=render_conversation(trace), question=trace.question,
            evidence=render_evidence(trace, GROUNDING_RESULT_CHARS, earlier=True),
            answer=trace.answer or "(no answer: the turn stopped)",
            ungrounded=_bullets(grounding["ungrounded_values"]),
        ))

    async def instruction_following(self, trace: Trace, checks: list[dict[str, Any]]) -> dict[str, Any]:
        rendered = [
            f"{c['check']}: {'pass' if c['pass'] else 'unknown' if c['pass'] is None else 'FAIL'}"
            + (f" ({'; '.join(c['evidence'])})" if c.get("evidence") else "")
            for c in checks
        ]
        return await self._ask(Verdict, prompt(
            "instruction_following",
            conversation=render_conversation(trace), question=trace.question,
            code_checks=_bullets(rendered), evidence=render_evidence(trace),
            answer=trace.answer or "(no answer: the turn stopped)",
        ))

    async def execution_strategy(
        self, trace: Trace, case: EvalCase, recovery: dict[str, Any], efficiency: dict[str, Any]
    ) -> dict[str, Any]:
        facts = [f"{e['tool']} failed: {e['args']} -> next call to it: {e['next']}" for e in recovery["errors"]]
        facts += [
            f"out-of-memory kills: {recovery['oom_events']}"
            + (f", resolved by: {recovery['oom_resolved_by']}" if recovery["oom_resolved_by"] else ""),
            f"moves to a bigger sandbox: {recovery['sandbox_moves']}",
        ]
        return await self._ask(Verdict, prompt(
            "execution_strategy",
            conversation=render_conversation(trace), question=trace.question,
            expected_behavior=case.expected_behavior or "(none given)",
            evidence=render_evidence(trace), recovery=_bullets(facts),
            metrics=_bullets([f"{k}: {v}" for k, v in efficiency.items()]),
        ))
