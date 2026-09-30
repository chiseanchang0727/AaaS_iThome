"""Judge any input against a set of labels with Jev (TypeSafe), one method per answer type.

This module only asks and reads answers. What goes in (the state, the
questions, the label descriptions) is the caller's job: see
agent/skill_router.py for one caller.

    jev = JevJudge({"billing": "Charges, refunds", "shipping": "Delivery"})
    state = {"ticket": "I was charged twice and the parcel is late."}

    # Choice: which ONE label fits? The labels compete; probabilities sum to 1.
    await jev.choice(state, question="Which team should handle `ticket`?")

    # Noul: does EACH label apply? Judged one by one; any number can be yes.
    await jev.noul(state, question="Does `ticket` need the {name} team ({description})?")

    # Score: how much does EACH label apply, on levels you describe?
    await jev.score(state, question="How urgent is `ticket` for {name}?",
                    levels=["can wait", "this week", "today"])

`noul` and `score` send one question per label, all in one request:
Jev reads the state once and answers them in parallel.

Measured with jev-1.13: `choice` put the right skill first on 54 of 55
requests (50 skills); one `noul` request kept 24 of 25 needed
conversation turns and no unneeded ones (10 turns, 18 follow-ups).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from typesafe_sdk import AsyncTypeSafeClient, Choice, JSONContent, Noul, Score

__all__ = ["Classification", "JevJudge", "Level", "NONE"]

NONE = "none"
"""`choice`'s option that means no label fits. Never a label of its own."""


@dataclass(frozen=True)
class Classification:
    probabilities: dict[str, float]
    """Every label (and `NONE`, if offered), most likely first. Sums to about 1."""

    confidence: float
    """How concentrated `probabilities` is, 0 to 1. Low when labels compete."""

    selected: list[str] = field(default_factory=list)
    """The labels that apply: `choice`'s top ones, or `select`'s top one plus
    the runners-up it confirmed."""

    checks: dict[str, float] = field(default_factory=dict)
    """`select` only: the probability each runner-up applies, by label."""

    @property
    def top(self) -> str | None:
        """The most likely label, or None when `NONE` came first."""
        best = next(iter(self.probabilities))
        return None if best == NONE else best


@dataclass(frozen=True)
class Level:
    """One label's `score` answer."""

    score: float
    """Position on the levels, 0 to len(levels) - 1; can fall between two."""

    confidence: float
    """How concentrated `probabilities` is, 0 to 1."""

    probabilities: dict[int, float]
    """Probability of each level, by its index in `levels`."""


class JevJudge:
    """Judges inputs against `labels`, a {name: description} map.

    Args:
        labels: Label name -> what it covers. The model reads the description,
            so write ones that tell neighbouring labels apart. For items that
            live in the state (e.g. conversation turns), the description can
            point there: {"turn_1": "`earlier_turns[0]`"}.
        client: Shared client. By default one is made per instance, reading
            $TYPESAFE_API_KEY. Share one when making instances per request.
        model: A pinned model id, so thresholds tuned on it keep their meaning.
    """

    def __init__(
        self,
        labels: Mapping[str, str],
        *,
        client: AsyncTypeSafeClient | None = None,
        model: str = "jev-1.13.0",
    ) -> None:
        if not labels:
            raise ValueError("a classifier needs at least one label")
        if NONE in labels:
            raise ValueError(f"'{NONE}' is reserved for the nothing-fits option")
        self.labels = dict(labels)
        self.model = model
        self._client = client or AsyncTypeSafeClient()

    # --- Choice ------------------------------------------------------------

    async def choice(
        self,
        state: JSONContent,
        *,
        question: str,
        none: str | None = None,
        top_n: int = 1,
    ) -> Classification:
        """Which label fits best? One Choice over every label.

        Args:
            question: What to decide, e.g. "Which skill should handle `request`?".
                Refer to parts of the state by backticked name.
            none: What the "nothing fits" option means, or None to force a
                label on every input.
            top_n: `selected` holds up to this many labels, most likely first,
                but never one ranked below `NONE`.
        """
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        criteria = dict(self.labels)
        if none is not None:
            criteria[NONE] = none
        response = await self._client.system_one(
            state=state,
            questions={"which": Choice(instructions=question, criteria=criteria)},
            model=self.model,
        )
        answer = response.choices["which"]
        ranked = dict(sorted(answer.probabilities.items(), key=lambda kv: -kv[1]))
        above_none = []
        for name in ranked:
            if name == NONE or len(above_none) == top_n:
                break
            above_none.append(name)
        return Classification(ranked, answer.confidence, above_none)

    # --- Noul --------------------------------------------------------------

    async def noul(
        self,
        state: JSONContent,
        *,
        question: str,
        criteria: Mapping[str, str] | None = None,
        only: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Does each label apply? One yes/no question per label, one request.

        Args:
            question: Formatted per label with `name` and `description`, e.g.
                "Does `request` need {description}?".
            criteria: {"true": ..., "false": ...}: what yes and no mean.
            only: Ask about these labels only (default: all).

        Returns:
            {label: probability of yes}, in label order. Unlike `choice`,
            these do not compete: all can be high, or all low.
        """
        names = list(only) if only is not None else list(self.labels)
        questions = {
            name: Noul(
                instructions=question.format(name=name, description=self.labels[name]),
                criteria=dict(criteria) if criteria else None,
            )
            for name in names
        }
        response = await self._client.system_one(
            state=state, questions=questions, model=self.model
        )
        return {name: response.nouls[name].noul for name in names}

    # --- Score -------------------------------------------------------------

    async def score(
        self,
        state: JSONContent,
        *,
        question: str,
        levels: Sequence[str],
        only: Sequence[str] | None = None,
    ) -> dict[str, Level]:
        """How much does each label apply? One Score per label, one request.

        Args:
            question: Formatted per label with `name` and `description`.
            levels: 2 to 10 descriptions, lowest first. Each must stand on its
                own: the model reads them, not their position.
            only: Ask about these labels only (default: all).
        """
        if not 2 <= len(levels) <= 10:
            raise ValueError("a Score needs 2 to 10 levels")
        names = list(only) if only is not None else list(self.labels)
        questions = {
            name: Score(
                instructions=question.format(name=name, description=self.labels[name]),
                criteria=list(levels),
            )
            for name in names
        }
        response = await self._client.system_one(
            state=state, questions=questions, model=self.model
        )
        return {
            name: Level(
                score=answer.score,
                confidence=answer.confidence,
                probabilities={int(k): v for k, v in answer.probabilities.items()},
            )
            for name in names
            for answer in [response.scores[name]]
        }

    # --- Choice, then Noul -------------------------------------------------

    async def select(
        self,
        state: JSONContent,
        *,
        question: str,
        verify: str,
        verify_criteria: Mapping[str, str] | None = None,
        none: str | None = None,
        shortlist: int = 3,
        threshold: float = 0.9,
    ) -> Classification:
        """`choice`, then keep runners-up that also apply on their own.

        The top label is always kept (unless it is `NONE`: then nothing is).
        Places 2..`shortlist` go through `noul` with `verify`, and those
        at or above `threshold` join it. Two requests.
        """
        first = await self.choice(state, question=question, none=none)
        if first.top is None:
            return first
        runners_up = [n for n in first.probabilities if n not in (NONE, first.top)]
        runners_up = runners_up[: shortlist - 1]
        if not runners_up:
            return first
        checks = await self.noul(
            state, question=verify, criteria=verify_criteria, only=runners_up
        )
        extra = [name for name in runners_up if checks[name] >= threshold]
        return Classification(
            first.probabilities, first.confidence, [first.top, *extra], checks
        )
