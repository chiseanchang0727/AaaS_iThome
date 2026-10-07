"""Finds results typed into a script's fixed text, against the recipe's own query results.

An agent that saw "Music leads with 6.0M views" tends to write that sentence
into the report's HTML. On other data, or the same table next month, the
chart changes and the sentence does not. So before a recipe is saved, its
script's string literals (and the fixed parts of f-strings) are checked
against the values its inputs actually returned:

- a text value from the results ("Music", a channel name), as a whole word;
- a number that matches a value in the results at the precision it is
  written ("6.0M" matches 6,012,345; "6,351" matches 6351).

Dates in fixed text are refused whatever the inputs returned ("November
2017", "2018-06", "2017-11-14"): a report's date range is read from the data,
and a date the user really wants fixed belongs in the query's WHERE.

Small numbers are skipped (counts like "top 10" and settings), as are CSS
sizes and colours. It only knows what the inputs returned: a figure from the
raw table that no input selected (e.g. its row count) goes unnoticed, and so
do short values such as a country code ("US").
"""

import ast
import io
import re
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from evals.grounding.scoring import numbers_in

MIN_TEXT = 3
"""Shorter text values ("US", "M") are too likely to appear by chance."""
MIN_NUMBER = 100
"""Whole numbers below this ("top 20", "5 categories") are not taken for results."""
MAX_VALUES = 20_000
"""Values read per column, at most."""

_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
           "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_DATE = re.compile(rf"\b(?:(?:{_MONTHS})\.?\s+(?:\d{{1,2}},?\s+)?(?:19|20)\d{{2}}|(?:19|20)\d{{2}}-(?:0[1-9]|1[0-2])(?:-\d{{2}})?)\b")
_CSS = re.compile(r"#[0-9A-Fa-f]{3,8}\b|\b\d+(?:\.\d+)?(?:px|em|rem|vh|vw|pt|ms|s)\b")


@dataclass(frozen=True)
class Found:
    line: int
    text: str
    """What the script wrote: "Music", "6.0M", "November 2017"."""
    file: str = ""
    column: str = ""
    """Where in the results it came from; empty for a date."""

    def problem(self) -> str:
        if not self.file:
            return (f"line {self.line} of the script writes the date \u201c{self.text}\u201d: read dates and "
                    "date ranges from the data instead. If it is a fixed choice the user asked for, put it "
                    "in the query's WHERE")
        return (f"line {self.line} of the script writes \u201c{self.text}\u201d, a result from {self.file} "
                f"({self.column}): compute it from the data instead. If it is a fixed choice the user asked "
                "for (e.g. only one category), put it in the query's WHERE")


def literals(script: str) -> list[tuple[int, str]]:
    """(line, text) of every string literal, including the fixed parts of f-strings."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []
    return [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _values(content: bytes) -> tuple[dict[str, set[str]], dict[str, list[float]]]:
    """A Parquet file's text values and numbers, by column."""
    try:
        # ParquetFile, single-threaded: read_table's dataset reader can deadlock
        # in its thread pool on in-memory bytes (seen with polars in the process).
        table = pq.ParquetFile(io.BytesIO(content)).read(use_threads=False)
    except Exception:
        return {}, {}
    texts: dict[str, set[str]] = {}
    numbers: dict[str, list[float]] = {}
    for name in table.column_names:
        column = table.column(name).slice(0, MAX_VALUES)
        kind = column.type
        if pa.types.is_string(kind) or pa.types.is_large_string(kind):
            texts[name] = {v for v in column.to_pylist() if isinstance(v, str) and len(v.strip()) >= MIN_TEXT}
        elif pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind):
            numbers[name] = [float(v) for v in column.to_pylist() if v is not None]
    return texts, numbers


def _whole_word(value: str, text: str) -> bool:
    start = text.find(value)
    while start != -1:
        end = start + len(value)
        if (start == 0 or not text[start - 1].isalnum()) and (end == len(text) or not text[end].isalnum()):
            return True
        start = text.find(value, start + 1)
    return False


def typed_results(script: str, inputs: dict[str, bytes]) -> list[Found]:
    """Results from `inputs` (file name -> Parquet bytes) written into the script's fixed text."""
    found: list[Found] = []
    seen: set[tuple[int, str]] = set()
    pieces = [(line, _CSS.sub(" ", text)) for line, text in literals(script)]
    for line, text in pieces:
        for match in _DATE.finditer(text):
            if (line, match.group()) not in seen:
                seen.add((line, match.group()))
                found.append(Found(line, match.group()))
    for file, content in inputs.items():
        texts, numbers = _values(content)
        for line, text in pieces:
            for column, values in texts.items():
                for value in values:
                    if value in text and _whole_word(value, text) and (line, value) not in seen:
                        seen.add((line, value))
                        found.append(Found(line, value, file, column))
            for n in numbers_in(text):
                if abs(n.value) < MIN_NUMBER and n.value.is_integer() and n.tolerance <= 0.5:
                    continue
                if (line, n.text) in seen:
                    continue
                for column, values in numbers.items():
                    if any(n.matches(v) for v in values):
                        seen.add((line, n.text))
                        found.append(Found(line, n.text, file, column))
                        break
    return sorted(found, key=lambda f: (f.line, f.text))
