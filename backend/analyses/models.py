"""The saved-analysis format: what is stored as JSON, and checks a recipe must pass before it runs."""

import ast
import re
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .sources import tables

_PARQUET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.parquet$")
_HTML = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.html$")
TYPED_NUMBERS = 6
"""A list of at least this many number literals in a script is taken for data
copied from a result (e.g. monthly counts), not settings."""


def typed_number_lists(script: str) -> list[tuple[int, int]]:
    """(line, how many numbers) for each list or tuple of TYPED_NUMBERS+ number literals in `script`."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []  # the test run reports it
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            numbers = [e for e in node.elts if isinstance(e, ast.Constant) and type(e.value) in (int, float)]
            if len(numbers) >= TYPED_NUMBERS:
                found.append((node.lineno, len(numbers)))
    return found


class QueryInput(BaseModel):
    """Rows from the database, saved as `file` in $DATA_DIR before the script runs."""

    kind: Literal["query"] = "query"
    sql: str = Field(description="One read-only PostgreSQL query")
    file: str = Field(description="Plain name ending in .parquet, e.g. monthly.parquet")


class DatasetInput(BaseModel):
    """An uploaded file dataset, copied to $DATA_DIR/<name>.parquet before the script runs."""

    kind: Literal["dataset"] = "dataset"
    name: str = Field(description="The file dataset's name, as list_datasets shows it")

    @property
    def file(self) -> str:
        return f"{self.name}.parquet"


Input = Annotated[QueryInput | DatasetInput, Field(discriminator="kind")]


class Output(BaseModel):
    file: str = Field(description="Plain name of a file the script writes to $OUTPUT_DIR")
    format: Literal["html"] = "html"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class RunRecord(BaseModel):
    """One run of a saved analysis: how it went, and which output files it made."""

    id: str
    started_at: str
    status: Literal["running", "done", "failed"]
    trigger: Literal["save", "run"]
    """"save": the test run when it was saved; "run": someone clicked Run."""
    seconds: float | None = None
    outputs: list[str] = []
    error: str | None = None
    log: str = ""
    """The script's output, clipped."""
    notes: list[str] = []
    """E.g. it moved to a bigger sandbox after running out of memory."""
    sources: dict[str, str] = {}
    """Tables this run read instead of the recipe's: {"videos": "videos_ca"}. Empty: the recipe's own."""
    version: int = 1
    """The recipe version it ran."""


class Analysis(BaseModel):
    id: str
    title: str
    description: str = ""
    question: str = ""
    """The user's message that led to it."""
    conversation: str | None = None
    created_at: str = Field(default_factory=now)
    version: int = 1
    """1 when first saved; each change saved over it (save_analysis `replaces`) adds one."""
    updated_at: str | None = None
    """When the current version was saved, if it is not the first."""
    inputs: list[Input]
    script: str
    outputs: list[Output]

    def source_tables(self) -> list[str]:
        """The tables its queries read, sorted; a query that does not parse adds nothing."""
        found: list[str] = []
        for item in self.inputs:
            if isinstance(item, QueryInput):
                try:
                    found += tables(item.sql)
                except ValueError:
                    pass
        return sorted(set(found))

    def problems(self) -> list[str]:
        """What is wrong with the recipe, before running anything. Empty when it may run."""
        found = []
        if not self.title.strip():
            found.append("give it a title")
        if not self.inputs:
            found.append("name at least one input: the script must read its data, not contain it")
        if not self.outputs:
            found.append("name at least one output file")
        files = [i.file for i in self.inputs]
        for f in files:
            if not _PARQUET.match(f):
                found.append(f"input file {f!r} must be a plain name ending in .parquet")
        if len(set(files)) != len(files):
            found.append("input files must have different names")
        for o in self.outputs:
            if not _HTML.match(o.file):
                found.append(f"output {o.file!r} must be a plain name ending in .html")
        if self.inputs and "DATA_DIR" not in self.script:
            found.append("the script must read its inputs from os.environ['DATA_DIR']")
        if self.outputs and "OUTPUT_DIR" not in self.script:
            found.append("the script must write its outputs to os.environ['OUTPUT_DIR']")
        # A script that never names an input file is likely carrying copied numbers instead.
        for f in files:
            if f not in self.script:
                found.append(f"the script never reads {f}: load every input from $DATA_DIR")
        for line, count in typed_number_lists(self.script):
            found.append(
                f"line {line} of the script types in {count} numbers: if they came from a result, compute "
                "them from the inputs instead; if they are settings (e.g. bins), build them with range()"
            )
        for item in self.inputs:
            if isinstance(item, QueryInput):
                try:
                    tables(item.sql)
                except ValueError as e:
                    found.append(f"the query for {item.file} could not be read: {e}")
        return found
