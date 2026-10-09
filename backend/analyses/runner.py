"""Run a saved analysis in a sandbox, in an empty scratch folder, and measure what it cost.

    <scratch>/data/        the inputs: each query's rows as Parquet, each file dataset
    <scratch>/out/         where the script writes its outputs
    <scratch>/analysis.py  the script, run as: DATA_DIR=<scratch>/data OUTPUT_DIR=<scratch>/out python3 analysis.py

The folder starts empty and is deleted after, so the script can only use
what the recipe names: not a file left over from the conversation.

With `estimate` and `limits`, a run measures as it goes (budget.py): the
planner's estimate of each query input first, and if that already breaks a
hard limit it stops there, before moving any rows. Then each input's real
rows, size and query time, and the script's time, CPU and peak memory.
"""

import asyncio
import io
import shlex
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import pyarrow.parquet as pq
from deepagents.backends.protocol import SandboxBackendProtocol

from datasources import TooManyRows
from sandboxes.metering import run_measured

from .budget import Limits, check_estimates, check_run, needs_optimization, over_estimate
from .models import Analysis, Finding, InputMeasure, Measurements, QueryInput, SandboxMeasure
from .sources import swap_tables, tables

LOG_CHARS = 4000
KILLED = 137
"""Exit code of a command killed by SIGKILL: almost always out of memory."""

Query = Callable[[str], Awaitable[bytes]]
"""SQL -> the rows as a Parquet file."""
Estimate = Callable[[str], Awaitable[dict[str, int]]]
"""SQL -> the planner's estimate: {"rows", "width", "bytes"} (datasources.estimate_query)."""
Count = Callable[[str], Awaitable[int]]
"""SQL -> how many rows it really returns, counted in the database (datasources.count_query)."""
ReadDataset = Callable[[str], bytes | None]
"""File dataset name -> its Parquet bytes, or None if there is no such file dataset."""


@dataclass
class RunResult:
    ok: bool
    error: str | None = None
    log: str = ""
    seconds: float = 0.0
    outputs: dict[str, bytes] = field(default_factory=dict)
    killed: bool = False
    """Killed by SIGKILL, usually for running out of memory."""
    measurements: Measurements = field(default_factory=Measurements)
    findings: list[Finding] = field(default_factory=list)

    @property
    def needs_optimization(self) -> bool:
        return needs_optimization(self.findings)


@dataclass
class Fetched:
    files: dict[str, bytes]
    measures: list[InputMeasure]
    problem: str | None = None
    too_large: Finding | None = None
    """An input over the hard row limit: the estimate was too low, the export stopped."""


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= LOG_CHARS else "…" + text[-LOG_CHARS:]


def _tables(sql: str) -> list[str]:
    try:
        return tables(sql)
    except ValueError:
        return []


def _parquet_rows(content: bytes) -> int | None:
    try:
        return pq.ParquetFile(io.BytesIO(content)).metadata.num_rows
    except Exception:
        return None


async def estimate_inputs(
    analysis: Analysis, estimate: Estimate, sources: dict[str, str] | None = None,
    count: Count | None = None, limits: Limits | None = None,
) -> list[InputMeasure]:
    """The planner's estimate for each query input (after swapping tables); nothing is exported.

    With `count` and `limits`: an estimate over a hard limit is checked by counting the rows in
    the database, since the planner can be far off; only the count comes back.
    """
    found = []
    for item in analysis.inputs:
        if not isinstance(item, QueryInput):
            continue
        sql = swap_tables(item.sql, sources or {})
        measure = InputMeasure(file=item.file, tables=_tables(sql))
        try:
            guess = await estimate(sql)
            measure.estimated_rows, measure.estimated_bytes = guess["rows"], guess["bytes"]
        except Exception:
            pass  # no estimate: the export itself still has its hard row limit
        if count is not None and limits is not None and over_estimate(measure, limits):
            try:
                measure.counted_rows = await count(sql)
            except Exception:
                pass  # e.g. the count timed out: the estimate stands
        found.append(measure)
    return found


async def fetch_inputs(
    analysis: Analysis, query: Query, read_dataset: ReadDataset, sources: dict[str, str] | None = None,
    estimates: list[InputMeasure] | None = None,
) -> Fetched:
    """Every input's bytes by file name, and what each cost. Stops at the first problem."""
    known = {m.file: m for m in estimates or []}
    fetched = Fetched(files={}, measures=[])
    for item in analysis.inputs:
        if isinstance(item, QueryInput):
            sql = swap_tables(item.sql, sources or {})
            measure = known.get(item.file) or InputMeasure(file=item.file, tables=_tables(sql))
            fetched.measures.append(measure)
            started = time.monotonic()
            try:
                content = await query(sql)
            except TooManyRows as e:
                measure.query_seconds = round(time.monotonic() - started, 2)
                fetched.too_large = Finding(limit="hard", kind="data_movement",
                                            reason=f"{item.file} returned too many rows to export: {e}")
                fetched.problem = f"the query for {item.file} returned too many rows"
                return fetched
            except Exception as e:
                fetched.problem = f"the query for {item.file} failed: {e}"
                return fetched
            measure.query_seconds = round(time.monotonic() - started, 2)
            measure.rows, measure.bytes = _parquet_rows(content), len(content)
            fetched.files[item.file] = content
        else:
            content = read_dataset(item.name)
            if content is None:
                fetched.problem = f"there is no uploaded file dataset named {item.name!r}"
                return fetched
            fetched.measures.append(InputMeasure(file=item.file, rows=_parquet_rows(content), bytes=len(content)))
            fetched.files[item.file] = content
    return fetched


def _real(sandbox: SandboxBackendProtocol) -> SandboxBackendProtocol:
    """The sandbox behind a conversation's stand-in, so the script is metered once, plainly.

    The stand-in (sandboxes/lazy.py) adds out-of-memory advice meant for the
    agent; a recipe run handles that itself.
    """
    real = getattr(sandbox, "real", None)
    return real() if callable(real) else sandbox


def run_in_sandbox(
    sandbox: SandboxBackendProtocol, analysis: Analysis, inputs: dict[str, bytes], scratch: PurePosixPath
) -> RunResult:
    """Put the inputs and script in `scratch`, run it (metered), and collect the outputs. Blocking."""
    started = time.monotonic()
    q = shlex.quote(str(scratch))

    def done(**fields) -> RunResult:
        return RunResult(seconds=round(time.monotonic() - started, 2), **fields)

    try:
        made = sandbox.execute(f"rm -rf {q} && mkdir -p {q}/data {q}/out")
        if made.exit_code != 0:
            return done(ok=False, error=f"could not make the scratch folder: {made.output.strip()[-300:]}")
        uploads = [(str(scratch / "data" / name), content) for name, content in inputs.items()]
        uploads.append((str(scratch / "analysis.py"), analysis.script.encode()))
        for response in sandbox.upload_files(uploads):
            if response.error:
                return done(ok=False, error=f"could not write {response.path}: {response.error}")

        result, step = run_measured(_real(sandbox), f"cd {q} && DATA_DIR={q}/data OUTPUT_DIR={q}/out python3 analysis.py")
        killed = result.exit_code == KILLED
        cost = Measurements(sandbox=SandboxMeasure(
            seconds=step.run_seconds or step.seconds, cpu_seconds=step.cpu_seconds,
            peak_memory_mb=step.peak_memory_mb, memory_limit_mb=step.memory_limit_mb, killed=killed,
        ))
        log = _clip(result.output)
        if result.exit_code != 0:
            why = "it ran out of memory (killed)" if killed else f"the script failed (exit code {result.exit_code})"
            return done(ok=False, error=why, log=log, killed=killed, measurements=cost)

        paths = [str(scratch / "out" / o.file) for o in analysis.outputs]
        outputs, missing = {}, []
        for o, response in zip(analysis.outputs, sandbox.download_files(paths)):
            if response.error or response.content is None:
                missing.append(o.file)
            else:
                outputs[o.file] = response.content
        if missing:
            return done(ok=False, error=f"the script did not write {', '.join(missing)} to $OUTPUT_DIR",
                        log=log, measurements=cost)
        return done(ok=True, log=log, outputs=outputs, measurements=cost)
    finally:
        try:
            sandbox.execute(f"rm -rf {q}")
        except Exception:
            pass


async def run(
    sandbox: SandboxBackendProtocol,
    analysis: Analysis,
    *,
    query: Query,
    read_dataset: ReadDataset,
    scratch: PurePosixPath,
    sources: dict[str, str] | None = None,
    estimate: Estimate | None = None,
    limits: Limits | None = None,
    estimates: list[InputMeasure] | None = None,
    count: Count | None = None,
) -> RunResult:
    """Fetch the inputs (reading `sources` instead of their tables), then run the script in `sandbox`.

    With `estimate` (or `estimates` already made) and `limits`: stop before exporting
    when an estimate breaks a hard limit, and judge what the run cost (RunResult.findings).
    """
    started = time.monotonic()
    if estimates is None:
        estimates = await estimate_inputs(analysis, estimate, sources, count, limits) if estimate is not None else []
    if limits is not None:
        stop = check_estimates(Measurements(inputs=estimates), limits)
        if stop:
            return RunResult(ok=False, error=stop[0].reason, findings=stop, measurements=Measurements(inputs=estimates),
                             seconds=round(time.monotonic() - started, 2))

    fetched = await fetch_inputs(analysis, query, read_dataset, sources, estimates)
    if fetched.problem:
        return RunResult(ok=False, error=fetched.problem, seconds=round(time.monotonic() - started, 2),
                         measurements=Measurements(inputs=fetched.measures),
                         findings=[fetched.too_large] if fetched.too_large else [])

    result = await asyncio.to_thread(run_in_sandbox, sandbox, analysis, fetched.files, scratch)
    result.seconds = round(time.monotonic() - started, 2)
    result.measurements = Measurements(inputs=fetched.measures, sandbox=result.measurements.sandbox)
    if limits is not None:
        result.findings = check_run(result.measurements, limits)
    return result
