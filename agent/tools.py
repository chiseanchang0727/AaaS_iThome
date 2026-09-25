"""Tools the agent can call.

A tool's docstring is its prompt: the model reads it to decide when to call the
tool and what to pass. Errors come back as text rather than exceptions, so the
model can read them and fix its next attempt instead of the run crashing.
"""

import json
import math

import asyncpg
from langchain_core.tools import tool

from config import cfg
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

    The data lives in a single table, `videos`. A result too large for the
    context is refused with its size, so the query can be narrowed.
    """
    try:
        rows = await _query_database(sql)
    except (TooManyRows, asyncpg.PostgresError) as e:
        return f"ERROR: {e}"
    return format_result(rows, cfg.agent.max_result_tokens)
