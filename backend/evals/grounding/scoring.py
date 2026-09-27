"""Does an answer contain the right numbers, and did it see them first?

A number written as "824.6K" is taken to mean 824,600 +- 50: it matches any
value within half of its last shown digit. "1,316,002", "1.32M", "~1.3M" and
"1.3 million" all match 1,316,002; "1.4M" does not.
"""

import re
from dataclasses import dataclass

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:19|20)\d{2}\b")
_CODE = re.compile(r"`[^`]*`")
_NUMBER = re.compile(
    r"(?<![\w.])(-?(?:\d{1,3}(?:,\d{3})+|\d+))(?:\.(\d+))?"
    r"(?!\d)(?:\s*(%|thousand\b|million\b|billion\b|[KkMB](?![a-z])))?"
    r"(?!st\b|nd\b|rd\b|th\b)",
    re.IGNORECASE,
)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


@dataclass(frozen=True)
class Number:
    value: float
    tolerance: float
    text: str

    def matches(self, target: float) -> bool:
        return abs(self.value - target) <= self.tolerance + 1e-9 * max(1.0, abs(target))


def numbers_in(text: str) -> list[Number]:
    """Every number in `text`, with the tolerance its written precision implies.

    Dates, years and ordinals ("90th") are skipped: they are labels, not
    measurements. A percentage yields two readings, 12% as both 12 and 0.12.
    """
    text = _DATE.sub(" ", text)
    found = []
    for m in _NUMBER.finditer(text):
        whole, decimals, suffix = m.group(1), m.group(2) or "", m.group(3)
        value = float(whole.replace(",", "") + ("." + decimals if decimals else ""))
        step = 10.0 ** -len(decimals)
        scale = _SCALE.get((suffix or "").lower(), 1.0)
        found.append(Number(value * scale, step * scale / 2, m.group(0).strip()))
        if suffix == "%":
            found.append(Number(value / 100, step / 200, m.group(0).strip()))
    return found


def contains_number(text: str, target: float) -> bool:
    return any(n.matches(target) for n in numbers_in(text))


def ungrounded(answer: str, observations: list[str], question: str) -> list[str]:
    """Numbers in `answer` that appear in no observation.

    Ignores small whole numbers (ranks, "top 3", list markers), anything
    already in the question, and anything in `backticks`: that is the agent
    quoting its method (`PERCENTILE_CONT(0.5)`), not stating a result. A number
    the agent derived itself (a ratio, a difference) also lands here: that is a
    flag to review, not proof of a made-up figure.
    """
    seen = [n for text in observations for n in numbers_in(text)]
    asked = numbers_in(question)

    # A percentage has two readings; it is grounded if either one is.
    readings: dict[str, list[Number]] = {}
    for n in numbers_in(_CODE.sub(" ", answer)):
        readings.setdefault(n.text, []).append(n)

    flagged = []
    for text, options in readings.items():
        first = options[0]
        if first.tolerance <= 0.5 and abs(first.value) <= 10 and first.value.is_integer():
            continue
        if any(n.matches(a.value) for n in options for a in asked):
            continue
        if not any(n.matches(s.value) for n in options for s in seen):
            flagged.append(text)
    return flagged
