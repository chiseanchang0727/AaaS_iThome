"""Which tables a recipe reads, and running it on other tables with the same columns.

A saved query names its tables plainly (`FROM videos`). A run may swap a
table for another: `{"videos": "videos_ca"}`. The SQL is parsed (sqlglot), so
only real table references change, not a column alias or a CTE that happens
to have the same name, and the new table keeps the old name as its alias, so
`videos.views` still works.

A swap is allowed only to a table that has every column of the original
(except an `id` key), with the same kind of type: any integer for an
integer, any text for a text, and so on.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

import sqlglot
from sqlglot import exp

QueryRows = Callable[[str], Awaitable[list[dict[str, Any]]]]
"""Read-only SQL -> rows, as the agent's database user."""

_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
IGNORED_COLUMNS = {"id"}
_FAMILIES = {
    "smallint": "integer", "integer": "integer", "bigint": "integer",
    "real": "number", "double precision": "number", "numeric": "number",
    "text": "text", "character varying": "text", "character": "text",
    "timestamp with time zone": "timestamptz", "timestamp without time zone": "timestamp",
}


def _parse(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read="postgres")


def _base_tables(tree: exp.Expression) -> list[exp.Table]:
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return [t for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes]


def tables(sql: str) -> list[str]:
    """The tables a query reads, lower case and sorted. Raises ValueError if it does not parse."""
    try:
        tree = _parse(sql)
    except sqlglot.errors.ParseError as e:
        raise ValueError(str(e).splitlines()[0]) from None
    return sorted({t.name.lower() for t in _base_tables(tree)})


def swap_tables(sql: str, mapping: dict[str, str]) -> str:
    """`sql` reading mapping[old] wherever it read old. Unchanged when nothing applies."""
    mapping = {k.lower(): v for k, v in mapping.items() if k.lower() != v.lower()}
    if not mapping:
        return sql
    tree = _parse(sql)
    for table in _base_tables(tree):
        new = mapping.get(table.name.lower())
        if new is None:
            continue
        if not table.alias:
            table.set("alias", exp.TableAlias(this=exp.to_identifier(table.name)))
        table.set("this", exp.to_identifier(new))
    return tree.sql(dialect="postgres")


def family(data_type: str) -> str:
    return _FAMILIES.get(data_type, data_type)


async def table_columns(query_rows: QueryRows, names: list[str] | None = None) -> dict[str, dict[str, str]]:
    """table -> column -> type, for the tables the agent's database user can read (or just `names`)."""
    where = ""
    if names is not None:
        safe = [n for n in names if _NAME.match(n)]
        if not safe:
            return {}
        where = " AND table_name IN (" + ", ".join(f"'{n}'" for n in safe) + ")"
    rows = await query_rows(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema = 'public'{where} ORDER BY table_name, ordinal_position"
    )
    found: dict[str, dict[str, str]] = {}
    for r in rows:
        found.setdefault(r["table_name"], {})[r["column_name"]] = r["data_type"]
    return found


def differences(original: dict[str, str], other: dict[str, str]) -> list[str]:
    """Why `other` cannot stand in for `original`: missing columns, other kinds of type. Empty if it can."""
    problems = []
    for column, data_type in original.items():
        if column in IGNORED_COLUMNS:
            continue
        if column not in other:
            problems.append(f"no column {column}")
        elif family(other[column]) != family(data_type):
            problems.append(f"{column} is {other[column]}, not {data_type}")
    return problems


async def source_options(source_tables: list[str], query_rows: QueryRows) -> list[dict[str, Any]]:
    """For each table a recipe reads: every readable table, and whether it can stand in."""
    columns = await table_columns(query_rows)
    options = []
    for table in source_tables:
        original = columns.get(table)
        candidates = []
        for name, cols in sorted(columns.items()):
            if name == table:
                continue
            problems = differences(original, cols) if original else [f"{table} is not readable"]
            candidates.append({"name": name, "ok": not problems, "problems": problems[:5]})
        options.append({"table": table, "candidates": candidates})
    return options


async def check_mapping(source_tables: list[str], mapping: dict[str, str], query_rows: QueryRows) -> list[str]:
    """Problems with running on `mapping` instead; empty when every swap is allowed."""
    problems = [f"the analysis does not read {old}" for old in mapping if old not in source_tables]
    problems += [f"{new!r} is not a table name" for new in mapping.values() if not _NAME.match(new)]
    if problems:
        return problems
    columns = await table_columns(query_rows, [*mapping, *mapping.values()])
    for old, new in mapping.items():
        if new not in columns:
            problems.append(f"there is no table {new} you can read")
        elif old in columns:
            problems += [f"{new}: {p}" for p in differences(columns[old], columns[new])]
    return problems
