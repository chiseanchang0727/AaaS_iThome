"""Do two versions of an analysis show the same results? Compared through their charts' data.

An optimization may change how the numbers are computed (SQL instead of
Python), never which numbers. So before an optimized version is saved, the
old and the new version both run on the same data and their outputs are
compared: the values plotted in each chart (Plotly HTML embeds them as JSON,
or base64 arrays in newer Plotly).

Same results: the same category labels, and the same numbers in any order
within REL_TOLERANCE. Two correct strategies can differ in the last digits
(e.g. a duration in exact seconds in SQL, whole seconds in Python), not more.
Outputs without chart data cannot be compared this way; that is reported,
not treated as a match.
"""

import base64
import json
import re
from array import array

REL_TOLERANCE = 0.01
_NEW_PLOT = re.compile(r"Plotly\.newPlot\(\s*")
_FIELDS = ("x", "y", "z", "values", "labels")
_TYPECODES = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "i2": "h", "i1": "b",
              "u8": "Q", "u4": "I", "u2": "H", "u1": "B"}


def _decode(value) -> list:
    """A trace field as a plain list: JSON lists as they are, Plotly's {"dtype", "bdata"} arrays decoded."""
    if isinstance(value, dict) and "bdata" in value:
        code = _TYPECODES.get(value.get("dtype", "f8"))
        if code is None:
            return []
        numbers = array(code)
        numbers.frombytes(base64.b64decode(value["bdata"]))
        return numbers.tolist()
    if isinstance(value, list):
        flat = []
        for v in value:
            flat.extend(_decode(v) if isinstance(v, (list, dict)) else [v])
        return flat
    return []


def chart_values(html: bytes) -> tuple[list[float], list[str]] | None:
    """The numbers and labels plotted in an HTML output, sorted; None if it has no Plotly chart."""
    text = html.decode("utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    numbers: list[float] = []
    labels: list[str] = []
    found = False
    for match in _NEW_PLOT.finditer(text):
        position = match.end()
        try:
            _, position = decoder.raw_decode(text, position)  # the element id
            position = text.index(",", position) + 1
            while text[position].isspace():
                position += 1
            traces, _ = decoder.raw_decode(text, position)
        except (ValueError, IndexError):
            continue
        found = True
        for trace in traces if isinstance(traces, list) else []:
            for name in _FIELDS:
                for v in _decode(trace.get(name)):
                    if isinstance(v, bool) or v is None:
                        continue
                    if isinstance(v, (int, float)):
                        numbers.append(float(v))
                    elif isinstance(v, str):
                        labels.append(v)
    return (sorted(numbers), sorted(set(labels))) if found else None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOLERANCE * max(abs(a), abs(b)) + 1e-9


def differences(old: dict[str, bytes], new: dict[str, bytes]) -> tuple[list[str], list[str]]:
    """(differences, notes) between two runs' outputs, file by file.

    Differences mean the results changed; notes say what could not be compared.
    """
    problems, notes = [], []
    for name in sorted(set(old) & set(new)):
        a, b = chart_values(old[name]), chart_values(new[name])
        if a is None or b is None:
            notes.append(f"{name} has no chart data to compare")
            continue
        (a_numbers, a_labels), (b_numbers, b_labels) = a, b
        if a_labels != b_labels:
            problems.append(f"{name}: the labels changed: {a_labels[:8]} -> {b_labels[:8]}")
        if len(a_numbers) != len(b_numbers):
            problems.append(f"{name}: {len(a_numbers)} values before, {len(b_numbers)} now")
            continue
        wrong = [(x, y) for x, y in zip(a_numbers, b_numbers) if not _close(x, y)]
        if wrong:
            shown = ", ".join(f"{x:,.6g} -> {y:,.6g}" for x, y in wrong[:5])
            problems.append(f"{name}: {len(wrong)} of {len(a_numbers)} values changed by more than "
                            f"{REL_TOLERANCE:.0%}: {shown}")
    for name in sorted(set(old) - set(new)):
        problems.append(f"{name} is no longer made")
    return problems, notes
