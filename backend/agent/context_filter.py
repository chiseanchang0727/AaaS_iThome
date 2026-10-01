"""Picks the earlier turns of a conversation that the new message needs.

The history (api/history.py) keeps every turn. The last `keep_last` turns are
always sent: follow-ups mostly point one turn back. For each older turn, Jev
answers one yes/no question, all in one request, and only the turns it picks
are sent to the model, in full. A turn it skips is not sent this time but
stays in the history, so a later message can still pick it.

References chain: "how long does that category take?" needs the turn that
named the category, which itself asked "what category is that channel in?"
and needs the turn that named the channel. So each turn Jev picks is asked
about in turn: which turns before it did it need? This repeats until a round
finds nothing new, at most `max_rounds` rounds. Each round is one request per
turn to follow, run together; Jev takes about 0.3s. A turn sent only because
it is among the last `keep_last` is not followed: after "thanks!", its chain
is not needed.

Skills read in a turn that is not sent are sent anyway, as if the agent had
just read them for the new message: the agent keeps the rules, and the skill
enforcer (agent/middleware.py) does not make it read them again.

What Jev reads is small: each earlier turn's question and final answer, not
its tool calls. The wording below kept 24 of 25 needed turns and no unneeded
ones in an experiment (10 turns, 18 follow-ups, jev-1.13).
"""

from __future__ import annotations

import asyncio
import logging
from itertools import groupby
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from typesafe_sdk import AsyncTypeSafeClient

from api.history import to_messages
from classify import JevJudge

log = logging.getLogger(__name__)

QUESTION = (
    "Does replying to `current_request` need anything from {description} ({name}): "
    "a result, a definition, or something the request refers back to?"
)
CRITERIA = {
    "true": "The reply would be wrong, incomplete or ambiguous without this turn: the request "
            "refers to it ('that', 'those', 'before'), builds on its result, or asks to summarize it.",
    "false": "The reply can be written without this turn: it is about a different subject, "
             "or the request is a new question that stands on its own.",
}


def group_turns(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """History lines by turn number, oldest first."""
    return {turn: list(lines) for turn, lines in groupby(records, key=lambda r: r["turn"])}


def build_state(
    request: str, turns: dict[int, list[dict[str, Any]]], answer: str | None = None
) -> dict[str, Any]:
    """What Jev reads: the message, and each earlier turn's question and answer.

    `answer` is the message's own answer, when it is an earlier turn being
    followed back rather than the new message.
    """
    state: dict[str, Any] = {"current_request": request}
    if answer is not None:
        state["current_answer"] = answer
    state["earlier_turns"] = [
        {"turn": n, "user": _user(lines), "assistant": _answer(lines)} for n, lines in turns.items()
    ]
    return state


def _user(lines: list[dict[str, Any]]) -> str:
    return next((r["content"] for r in lines if r["role"] == "user"), "")


def _answer(lines: list[dict[str, Any]]) -> str:
    """The turn's final answer: its last assistant line that called no tool."""
    finals = [r["content"] for r in lines if r["role"] == "assistant" and not r.get("tool_calls")]
    return finals[-1] if finals else ""


class ContextFilter:
    """Chooses which earlier turns to send with a new message.

    Args:
        client: Shared TypeSafe client. By default one is made, reading
            $TYPESAFE_API_KEY.
        model: A pinned Jev model, so `threshold` keeps its meaning.
        threshold: An older turn is sent when Jev's probability of yes is at
            least this.
        keep_last: Always send this many of the most recent turns, without
            asking Jev. 0 asks about every turn.
        max_rounds: How far to follow references back. 1 asks only about the
            new message; each more round asks, for every turn just added, which
            earlier turns it needed.
    """

    def __init__(
        self,
        *,
        client: AsyncTypeSafeClient | None = None,
        model: str = "jev-1.13.0",
        threshold: float = 0.5,
        keep_last: int = 1,
        max_rounds: int = 3,
    ) -> None:
        if keep_last < 0:
            raise ValueError("keep_last must be 0 or more")
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._client = client or AsyncTypeSafeClient()
        self.model = model
        self.threshold = threshold
        self.keep_last = keep_last
        self.max_rounds = max_rounds

    async def pick(self, records: list[dict[str, Any]], request: str) -> list[int]:
        """The earlier turns `request` needs, directly or through other turns, oldest first.

        Round 1 asks about the new message and every earlier turn. The last
        `keep_last` are sent whatever Jev says, but only the turns it says yes
        to are followed back. Later rounds follow each turn just added back to
        the turns before it. If round 1 fails, every turn is picked: sending too
        much is safer than losing what the user refers to. A later round that
        fails just adds nothing.
        """
        turns = group_turns(records)
        numbers = list(turns)
        split = max(0, len(numbers) - self.keep_last)
        older, recent = numbers[:split], numbers[split:]
        if not older:
            return recent  # all sent anyway; their chains only lead to each other
        try:
            needed = await self._ask(request, None, turns, numbers)
        except Exception:
            log.warning("context filter failed; sending every earlier turn", exc_info=True)
            return numbers
        picked = set(needed) | set(recent)

        follow = needed
        for _ in range(self.max_rounds - 1):
            asks = [(t, [n for n in numbers if n < t and n not in picked]) for t in follow]
            asks = [(t, candidates) for t, candidates in asks if candidates]
            if not asks:
                break
            answers = await asyncio.gather(
                *(
                    self._ask(_user(turns[t]), _answer(turns[t]), {n: turns[n] for n in numbers if n < t}, candidates)
                    for t, candidates in asks
                ),
                return_exceptions=True,
            )
            found = set()
            for (t, _), answer in zip(asks, answers):
                if isinstance(answer, BaseException):
                    log.warning("context filter could not follow turn %d back", t, exc_info=answer)
                else:
                    found |= set(answer)
            follow = sorted(found - picked)
            if not follow:
                break
            picked |= set(follow)
        return sorted(picked)

    async def _ask(
        self,
        request: str,
        answer: str | None,
        turns: dict[int, list[dict[str, Any]]],
        candidates: list[int],
    ) -> list[int]:
        """Which of `candidates` does `request` need? One request; Jev reads all of `turns`."""
        index = {n: i for i, n in enumerate(turns)}
        labels = {f"turn_{n}": f"`earlier_turns[{index[n]}]`" for n in candidates}
        judge = JevJudge(labels, client=self._client, model=self.model)
        yes = await judge.noul(build_state(request, turns, answer), question=QUESTION, criteria=CRITERIA)
        return [n for n in candidates if yes[f"turn_{n}"] >= self.threshold]

    async def build(self, records: list[dict[str, Any]], request: str) -> tuple[list[BaseMessage], list[int]]:
        """The messages to start the new turn with, and the picked turn numbers.

        The picked turns, whole (tool calls with their results); then the new
        message; then any skill read only in a turn that was not picked, as a
        read of this turn.
        """
        picked = await self.pick(records, request)
        kept = [r for r in records if r["turn"] in picked]
        reads = skill_reads([r for r in records if r["turn"] not in picked], already=skill_reads(kept))
        return [*to_messages(kept), HumanMessage(request), *to_messages(reads)], picked


def _skill_path(call: dict[str, Any]) -> str | None:
    path = str(call.get("args", {}).get("file_path", ""))
    return path if call.get("name") == "read_file" and path.endswith("/SKILL.md") else None


def skill_reads(records: list[dict[str, Any]], *, already: list[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    """The latest successful read of each SKILL.md in `records`, as history lines.

    Each read is one assistant line holding only that `read_file` call, then
    its result. Skills read in `already` are left out.
    """
    have = {_skill_path(c) for r in already if r["role"] == "assistant" for c in r.get("tool_calls", [])}
    results = {r["tool_call_id"]: r for r in records if r["role"] == "tool" and not r.get("error")}
    latest: dict[str, tuple[dict, dict]] = {}
    for r in records:
        for call in r.get("tool_calls", []) if r["role"] == "assistant" else []:
            path = _skill_path(call)
            if path and path not in have and call["id"] in results:
                latest.pop(path, None)
                latest[path] = ({**r, "content": "", "tool_calls": [call]}, results[call["id"]])
    return [line for pair in latest.values() for line in pair]
