"""Run a saved analysis in a sandbox, in an empty scratch folder.

    <scratch>/data/        the inputs: each query's rows as Parquet, each file dataset
    <scratch>/out/         where the script writes its outputs
    <scratch>/analysis.py  the script, run as: DATA_DIR=<scratch>/data OUTPUT_DIR=<scratch>/out python3 analysis.py

The folder starts empty and is deleted after, so the script can only use
what the recipe names: not a file left over from the conversation.
"""

import asyncio
import shlex
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from deepagents.backends.protocol import SandboxBackendProtocol

from .models import Analysis, QueryInput
from .sources import swap_tables

LOG_CHARS = 4000
KILLED = 137
"""Exit code of a command killed by SIGKILL: almost always out of memory."""

Query = Callable[[str], Awaitable[bytes]]
"""SQL -> the rows as a Parquet file."""
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


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= LOG_CHARS else "…" + text[-LOG_CHARS:]


async def fetch_inputs(
    analysis: Analysis, query: Query, read_dataset: ReadDataset, sources: dict[str, str] | None = None
) -> tuple[dict[str, bytes], str | None]:
    """Every input's bytes by file name, or the first problem getting one. `sources` swaps tables."""
    files: dict[str, bytes] = {}
    for item in analysis.inputs:
        if isinstance(item, QueryInput):
            try:
                files[item.file] = await query(swap_tables(item.sql, sources or {}))
            except Exception as e:
                return files, f"the query for {item.file} failed: {e}"
        else:
            content = read_dataset(item.name)
            if content is None:
                return files, f"there is no uploaded file dataset named {item.name!r}"
            files[item.file] = content
    return files, None


def run_in_sandbox(
    sandbox: SandboxBackendProtocol, analysis: Analysis, inputs: dict[str, bytes], scratch: PurePosixPath
) -> RunResult:
    """Put the inputs and script in `scratch`, run it, and collect the outputs. Blocking."""
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

        result = sandbox.execute(f"cd {q} && DATA_DIR={q}/data OUTPUT_DIR={q}/out python3 analysis.py")
        log = _clip(result.output)
        if result.exit_code != 0:
            killed = result.exit_code == KILLED
            why = "it ran out of memory (killed)" if killed else f"the script failed (exit code {result.exit_code})"
            return done(ok=False, error=why, log=log, killed=killed)

        paths = [str(scratch / "out" / o.file) for o in analysis.outputs]
        outputs, missing = {}, []
        for o, response in zip(analysis.outputs, sandbox.download_files(paths)):
            if response.error or response.content is None:
                missing.append(o.file)
            else:
                outputs[o.file] = response.content
        if missing:
            return done(ok=False, error=f"the script did not write {', '.join(missing)} to $OUTPUT_DIR", log=log)
        return done(ok=True, log=log, outputs=outputs)
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
) -> RunResult:
    """Fetch the inputs (reading `sources` instead of their tables), then run the script in `sandbox`."""
    started = time.monotonic()
    inputs, problem = await fetch_inputs(analysis, query, read_dataset, sources)
    if problem:
        return RunResult(ok=False, error=problem, seconds=round(time.monotonic() - started, 2))
    result = await asyncio.to_thread(run_in_sandbox, sandbox, analysis, inputs, scratch)
    result.seconds = round(time.monotonic() - started, 2)
    return result
