"""Tools the agent can call.

A tool's docstring is its prompt: the model reads it to decide when to call the
tool and what to pass. Errors come back as text rather than exceptions, so the
model can read them and fix its next attempt instead of the run crashing.
"""

import asyncio
import io
import json
import math
import re
import uuid
from decimal import Decimal
from pathlib import PurePosixPath

import asyncpg
import pyarrow as pa
import pyarrow.parquet as pq
from deepagents.backends.protocol import BackendProtocol
from langchain_core.tools import BaseTool, tool

from config import cfg
from datasets import DatasetStore, Registry
from datasources import TooManyRows
from datasources import query_database as _query_database

CHARS_PER_TOKEN = 4
"""The estimate deepagents uses too. Undercounts non-Latin text, e.g. CJK titles."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def format_result(rows: list[dict], max_tokens: int) -> str:
    """Rows as JSON, or a refusal saying how big they were.

    The limit is on size, not on what the query does: the model can select
    fewer columns, fewer rows, or aggregate, whichever suits the question.
    Refusing beats truncating: a cut-off result looks complete, and the model
    would answer from part of the data without knowing.
    """
    content = json.dumps(rows, ensure_ascii=False)
    tokens = estimate_tokens(content)
    if tokens <= max_tokens:
        return content
    columns = len(rows[0])
    return (
        f"ERROR: result is ~{tokens:,} tokens ({len(rows):,} rows x {columns} columns), "
        f"over the limit of {max_tokens:,}. Nothing was returned; narrow the query "
        f"to what the answer needs."
    )


@tool
async def query_database(sql: str) -> str:
    """Run one read-only PostgreSQL query and return the rows as JSON.

    The database holds the built-in `videos` table and any tables users
    uploaded; list_datasets names them all. A result too large for the context
    is refused with its size, so the query can be narrowed.
    """
    try:
        rows = await _query_database(sql)
    except (TooManyRows, asyncpg.PostgresError) as e:
        return f"ERROR: {e}"
    return format_result(rows, cfg.agent.max_result_tokens)


_PARQUET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.parquet$")


def _arrow_value(value):
    """Keep types Parquet can store; convert the two it stores awkwardly."""
    if isinstance(value, Decimal):
        # SUM(bigint) comes back as numeric. As a Parquet decimal it would load
        # as a Decimal column, not as numbers ready for arithmetic and charts.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def to_parquet(rows: list[dict]) -> bytes:
    """Rows as a Parquet file, keeping dates, timestamps, ints and booleans typed."""
    table = pa.Table.from_pylist([{k: _arrow_value(v) for k, v in row.items()} for row in rows])
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def make_export_query(backend: BackendProtocol, data_dir: PurePosixPath, max_rows: int) -> BaseTool:
    """An export_query tool that writes into `backend` under `data_dir`.

    The rows travel from Postgres to the sandbox without passing through the
    model: it gets back the file's path, size and columns, nothing else.
    """

    @tool
    async def export_query(sql: str, filename: str) -> str:
        """Run one read-only PostgreSQL query and save the rows as a Parquet file
        in the sandbox, for analysis with Python (polars, statistics, charts).

        Use this instead of query_database when you need to process many rows
        in code. Returns the file's path, size and columns, not the rows. Load
        it with polars, `pl.read_parquet(path)`; column types are kept.
        `filename` is a plain name ending in .parquet, e.g. "per_video.parquet".
        """
        if not _PARQUET_NAME.match(filename):
            return f"ERROR: filename must be a plain name ending in .parquet, got {filename!r}"
        try:
            rows = await _query_database(sql, max_rows=max_rows, raw=True)
        except (TooManyRows, asyncpg.PostgresError) as e:
            return f"ERROR: {e}"

        path = str(data_dir / filename)
        content = to_parquet(rows)
        [response] = await asyncio.to_thread(backend.upload_files, [(path, content)])
        if response.error:
            return f"ERROR: could not write {path}: {response.error}"

        columns = ", ".join(rows[0]) if rows else "(none)"
        return f"Wrote {len(rows):,} rows to {path} ({len(content):,} bytes). Columns: {columns}"

    return export_query


VIDEOS = {
    "name": "videos",
    "kind": "table",
    "description": "Built in: US YouTube trending videos, one row per video per "
    "trending day. Read /skills/query_database/SKILL.md before querying it.",
}


def make_list_datasets(registry: Registry, data_dir: PurePosixPath | None) -> BaseTool:
    """A list_datasets tool over `registry`. Without a sandbox (`data_dir`
    None), file datasets are listed as unavailable."""

    @tool
    def list_datasets() -> str:
        """List the data you can analyse: the built-in `videos` table plus every
        dataset users uploaded, with row counts, columns and types.

        A "table" is queried with query_database or export_query by its name.
        A "file" is a Parquet file already in the sandbox at its "path".
        """
        entries = [VIDEOS]
        for d in registry.list():
            entry = {
                "name": d.name, "kind": d.kind, "rows": d.rows,
                "columns": {c.name: c.type for c in d.columns}, "uploaded_from": d.source,
            }
            if d.kind == "file":
                entry["path"] = (
                    str(DatasetStore.sandbox_path(data_dir, d.name)) if data_dir
                    else "unavailable: no sandbox in this session"
                )
            entries.append(entry)
        return json.dumps(entries, ensure_ascii=False)

    return list_datasets


def make_request_bigger_sandbox(sandbox) -> BaseTool:
    """A request_bigger_sandbox tool for `sandbox` (a sandboxes.LazySandbox).

    Sync on purpose: LangChain runs sync tools in a worker thread, and the
    move itself runs on the event loop, which an async tool would block.
    """

    @tool
    def request_bigger_sandbox(reason: str) -> str:
        """Move this conversation to a sandbox with more memory (e.g. 1 GB -> 4 GB).

        Use only after a command was killed for running out of memory, and only
        when the work truly needs that much memory at once; otherwise change the
        approach (fewer columns, aggregate earlier, DuckDB, chunks, sampling),
        which is cheaper. `reason`: one sentence on why the memory is needed.
        Files in the work folder are copied over. Run the command again after.
        """
        return sandbox.request_bigger(reason)

    return request_bigger_sandbox
