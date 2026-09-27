"""Turning an uploaded CSV or Parquet file into a table or a stored file."""

import json
import re
from decimal import Decimal
from pathlib import Path

import asyncpg
import polars as pl

from .registry import Column

SUPPORTED = {".csv", ".parquet"}
PREVIEW_ROWS = 20


class UploadError(ValueError):
    """The upload can't be used; the message says why, for the user."""


def read_upload(path: Path) -> pl.DataFrame:
    """Read a CSV or Parquet file, inferring column types."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise UploadError(f"unsupported file type {suffix or '(none)'}; upload .csv or .parquet")
    try:
        if suffix == ".csv":
            df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=10_000)
        else:
            df = pl.read_parquet(path)
    except Exception as e:
        raise UploadError(f"could not read {path.name}: {e}") from None
    if df.width == 0:
        raise UploadError(f"{path.name} has no columns")
    return df.rename(dict(zip(df.columns, clean_columns(df.columns))))


def clean_columns(names: list[str]) -> list[str]:
    """Column names that are safe, readable Postgres identifiers, and unique.

    "Views (M)" -> "views_m", "2024" -> "c_2024", duplicates get _2, _3...
    """
    cleaned, seen = [], set()
    for raw in names:
        name = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_") or "column"
        if name[0].isdigit():
            name = f"c_{name}"
        name = name[:60]
        candidate, n = name, 2
        while candidate in seen:
            candidate, n = f"{name}_{n}", n + 1
        seen.add(candidate)
        cleaned.append(candidate)
    return cleaned


def pg_type(dtype: pl.DataType) -> str:
    """The Postgres column type for a polars dtype; text when nothing fits."""
    if dtype in (pl.Int8, pl.Int16, pl.UInt8):
        return "smallint"
    if dtype in (pl.Int32, pl.UInt16):
        return "integer"
    if dtype in (pl.Int64, pl.UInt32):
        return "bigint"
    if dtype == pl.UInt64 or isinstance(dtype, pl.Decimal):
        return "numeric"
    if dtype == pl.Float32:
        return "real"
    if dtype == pl.Float64:
        return "double precision"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "timestamptz" if dtype.time_zone else "timestamp"
    if dtype == pl.Time:
        return "time"
    if isinstance(dtype, pl.Duration):
        return "interval"
    return "text"


def _cell_converter(dtype: pl.DataType):
    """How to turn a polars value into what asyncpg expects for its column."""
    target = pg_type(dtype)
    if target == "numeric":
        return lambda v: None if v is None else Decimal(str(v))
    if target == "text" and dtype not in (pl.String, pl.Categorical, pl.Enum, pl.Null):
        # Lists, structs and the like: store as JSON text rather than lose them.
        return lambda v: None if v is None else json.dumps(v, default=str)
    return None


def table_columns(df: pl.DataFrame) -> list[Column]:
    return [Column(name=n, type=pg_type(t)) for n, t in df.schema.items()]


def file_columns(df: pl.DataFrame) -> list[Column]:
    return [Column(name=n, type=str(t)) for n, t in df.schema.items()]


def preview(df: pl.DataFrame) -> list[dict]:
    """The first rows as JSON-safe dicts, for the user to check the parse."""
    return json.loads(df.head(PREVIEW_ROWS).write_json())


async def create_table(dsn: str, name: str, df: pl.DataFrame, reader_role: str) -> None:
    """Create table `name` from `df` and let `reader_role` SELECT from it.

    One transaction: on any failure nothing is left behind. Fails if the table
    already exists rather than replacing it.
    """
    columns = table_columns(df)
    converters = [_cell_converter(t) for t in df.schema.values()]
    ddl = ", ".join(f'"{c.name}" {c.type}' for c in columns)

    def records():
        for row in df.iter_rows():
            yield tuple(conv(v) if conv else v for conv, v in zip(converters, row))

    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await conn.execute(f'CREATE TABLE "{name}" ({ddl})')
            await conn.copy_records_to_table(name, records=records(), columns=[c.name for c in columns])
            await conn.execute(f'GRANT SELECT ON "{name}" TO "{reader_role}"')
    except asyncpg.DuplicateTableError:
        raise UploadError(f"a table named {name!r} already exists") from None
    finally:
        await conn.close()


async def drop_table(dsn: str, name: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    finally:
        await conn.close()
