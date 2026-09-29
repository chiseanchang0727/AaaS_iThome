"""Sort any input into a fixed set of labels with Jev (TypeSafe).

This module only asks and reads answers. What goes in (the state, the
question, the label descriptions) is the caller's job: see
agent/skill_router.py for one caller.

    from classify import JevClassifier

    teams = JevClassifier(
        {"billing": "Charges, invoices, refunds", "shipping": "Delivery, delays"},
        question="Which team should handle `ticket`?",
        none="Neither team handles this.",
    )
    result = await teams.classify({"ticket": "I was charged twice."})
    result.top            # "billing", or None when nothing fits
    result.probabilities  # every label, most likely first

One Choice question ranks every label at once (up to 255). With `none` set,
the model can also say nothing fits. `select` adds a second request for inputs
that need several labels: it keeps the top label and asks, label by label,
whether each runner-up applies too.

On a 50-skill roster and 65 labelled requests (jev-1.13), `classify` put the
right skill first on 54 of 55 covered requests and said "none" on 9 of 10 that
no skill covered, in one ~0.3 s request.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from typesafe_sdk import AsyncTypeSafeClient, Choice, JSONContent, Noul

__all__ = ["Classification", "JevClassifier", "NONE"]

NONE = "none"
"""The option that means no label fits. Never a label of its own."""


@dataclass(frozen=True)
class Classification:
    probabilities: dict[str, float]
    """Every label (and `NONE`, if offered), most likely first. Sums to about 1."""

    confidence: float
    """How concentrated `probabilities` is, 0 to 1. Low when labels compete."""

    selected: list[str] = field(default_factory=list)
    """The labels that apply: the top one, plus any `select` confirmed."""

    checks: dict[str, float] = field(default_factory=dict)
    """`select` only: the probability each runner-up applies, by label."""

    @property
    def top(self) -> str | None:
        """The most likely label, or None when `NONE` came first."""
        best = next(iter(self.probabilities))
        return None if best == NONE else best


class JevClassifier:
    """Sorts inputs into `labels`, a {name: description} map.

    Args:
        labels: Option name -> what it covers. The model reads both, so write
            descriptions that tell neighbouring labels apart.
        question: What to decide, e.g. "Which skill should handle `request`?".
            Refer to parts of the state by backticked name.
        none: What the "nothing fits" option means, or None to force a label
            on every input.
        verify: `select`'s per-label yes/no question, formatted with `name`
            and `description`. Needed only for `select`.
        verify_criteria: {"true": ..., "false": ...}: what yes and no mean.
        client: Shared client. By default one is made per classifier, reading
            $TYPESAFE_API_KEY.
        model: A pinned model id, so thresholds tuned on it keep their meaning.
    """

    def __init__(
        self,
        labels: Mapping[str, str],
        *,
        question: str,
        none: str | None = None,
        verify: str | None = None,
        verify_criteria: Mapping[str, str] | None = None,
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
        self._verify = verify
        self._verify_criteria = dict(verify_criteria) if verify_criteria else None
        criteria = dict(self.labels)
        if none is not None:
            criteria[NONE] = none
        self._choice = Choice(instructions=question, criteria=criteria)

    async def classify(self, state: JSONContent) -> Classification:
        """Rank every label against `state` (a string or a JSON object)."""
        response = await self._client.system_one(
            state=state, questions={"which": self._choice}, model=self.model
        )
        answer = response.choices["which"]
        ranked = dict(sorted(answer.probabilities.items(), key=lambda kv: -kv[1]))
        best = next(iter(ranked))
        return Classification(ranked, answer.confidence, [] if best == NONE else [best])

    async def select(
        self, state: JSONContent, *, shortlist: int = 3, threshold: float = 0.9
    ) -> Classification:
        """`classify`, then also keep runners-up that apply on their own.

        The top label is always kept (unless it is `NONE`: then nothing is).
        The next `shortlist - 1` labels each get the `verify` question, and
        those at or above `threshold` join it. A runner-up often gets little of
        the Choice's probability even when it applies, because the Choice
        splits probability between labels; the yes/no question judges it alone.
        """
        if self._verify is None:
            raise ValueError("select needs a `verify` question")
        first = await self.classify(state)
        if first.top is None:
            return first
        runners_up = [n for n in first.probabilities if n not in (NONE, first.top)]
        runners_up = runners_up[: shortlist - 1]
        if not runners_up:
            return first
        questions = {
            name: Noul(
                instructions=self._verify.format(name=name, description=self.labels[name]),
                criteria=self._verify_criteria,
            )
            for name in runners_up
        }
        response = await self._client.system_one(
            state=state, questions=questions, model=self.model
        )
        checks = {name: response.nouls[name].noul for name in runners_up}
        extra = [name for name in runners_up if checks[name] >= threshold]
        return Classification(
            first.probabilities, first.confidence, [first.top, *extra], checks
        )
