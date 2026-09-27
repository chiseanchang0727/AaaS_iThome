"""export_query, download_outputs and build_agent with a sandbox, without Daytona.

The database is stubbed too, so these run anywhere.
"""

import asyncio
import datetime as dt
import io
import uuid
from decimal import Decimal
from pathlib import Path, PurePosixPath

import asyncpg
import pandas as pd
import pyarrow.parquet as pq
import pytest
from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

import agent.tools as tools
from agent import build_agent
from agent.middleware import SkillEnforcerMiddleware
from agent.tools import make_export_query, to_parquet
from config import SandboxConfig, cfg
from datasources import TooManyRows
from sandboxes import download_outputs

DATA_DIR = PurePosixPath("/data")
OUTPUT_DIR = PurePosixPath("/outputs")
ROWS = [
    {"category_name": "Music", "views": 4821071901, "n": 799},
    {"category_name": "Film & Animation", "views": 825343729, "n": 321},
]


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalShellBackend:
    return LocalShellBackend(root_dir=tmp_path, virtual_mode=True)


@pytest.fixture
def stub_query(monkeypatch):
    """Replace the database call; records the arguments it was given."""
    state = {"out": ROWS}

    async def fake(sql: str, max_rows: int | None = None, *, raw: bool = False):
        state["sql"], state["max_rows"], state["raw"] = sql, max_rows, raw
        if isinstance(state["out"], Exception):
            raise state["out"]
        return state["out"]

    monkeypatch.setattr(tools, "_query_database", fake)
    return state


def export(backend, sql="SELECT 1", filename="out.parquet", max_rows=100_000) -> str:
    tool = make_export_query(backend, DATA_DIR, max_rows)
    return asyncio.run(tool.ainvoke({"sql": sql, "filename": filename}))


# --- to_parquet ----------------------------------------------------------------


def read(content: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(content))


def test_to_parquet_round_trips_the_rows():
    assert read(to_parquet(ROWS)).to_dict("records") == ROWS


def test_to_parquet_keeps_column_types():
    """The point of Parquet over CSV: no parsing, dates are dates, ints are ints."""
    rows = [
        {
            "views": 225211923,
            "like_ratio": 0.042,
            "trending_date": dt.date(2018, 6, 2),
            "publish_time": dt.datetime(2018, 5, 6, 4, 0, tzinfo=dt.UTC),
            "comments_disabled": False,
            "title": "This Is America",
        }
    ]
    dtypes = read(to_parquet(rows)).dtypes
    assert str(dtypes["views"]) == "int64"
    assert str(dtypes["like_ratio"]) == "float64"
    assert str(dtypes["publish_time"]).startswith("datetime64")
    assert str(dtypes["comments_disabled"]) == "bool"
    table = pq.read_table(io.BytesIO(to_parquet(rows)))
    assert str(table.schema.field("trending_date").type) == "date32[day]"


def test_to_parquet_turns_decimals_into_numbers():
    """SUM(bigint) is numeric in Postgres; as a Parquet decimal pandas would
    hand back Python objects instead of numbers."""
    df = read(to_parquet([{"total": Decimal("4821071901")}, {"total": Decimal("10.5")}]))
    assert str(df["total"].dtype) == "float64"
    assert df["total"].tolist() == [4821071901.0, 10.5]


def test_to_parquet_keeps_whole_decimals_as_ints():
    df = read(to_parquet([{"total": Decimal("4821071901")}]))
    assert str(df["total"].dtype) == "int64"


def test_to_parquet_stores_uuids_as_text():
    assert read(to_parquet([{"id": uuid.UUID(int=0)}]))["id"][0] == "00000000-0000-0000-0000-000000000000"


def test_to_parquet_keeps_nulls():
    df = read(to_parquet([{"views": 1}, {"views": None}]))
    assert df["views"].isna().tolist() == [False, True]


def test_to_parquet_keeps_non_ascii():
    assert read(to_parquet([{"title": "방탄소년단"}]))["title"][0] == "방탄소년단"


def test_to_parquet_of_nothing_is_a_readable_empty_table():
    assert len(read(to_parquet([]))) == 0


# --- export_query ------------------------------------------------------------


def test_export_writes_the_file_into_the_sandbox(sandbox, stub_query, tmp_path):
    export(sandbox, filename="by_category.parquet")
    written = tmp_path / "data" / "by_category.parquet"
    assert read(written.read_bytes()).to_dict("records") == ROWS


def test_export_asks_for_raw_values(sandbox, stub_query):
    """JSON-safe values would already have turned dates into strings."""
    export(sandbox)
    assert stub_query["raw"] is True


def test_export_returns_a_summary_not_the_rows(sandbox, stub_query):
    summary = export(sandbox, filename="by_category.parquet")
    assert summary.startswith("Wrote 2 rows to /data/by_category.parquet")
    assert "Columns: category_name, views, n" in summary
    assert "Music" not in summary and "4821071901" not in summary


def test_export_uses_its_own_row_cap(sandbox, stub_query):
    export(sandbox, max_rows=50_000)
    assert stub_query["max_rows"] == 50_000


def test_export_of_an_empty_result(sandbox, stub_query, tmp_path):
    stub_query["out"] = []
    summary = export(sandbox, filename="empty.parquet")
    assert summary.startswith("Wrote 0 rows")
    assert len(read((tmp_path / "data" / "empty.parquet").read_bytes())) == 0


@pytest.mark.parametrize(
    "filename",
    ["../escape.parquet", "sub/dir.parquet", "/abs.parquet", "data.csv", ".hidden.parquet", "", "a b.parquet"],
)
def test_export_rejects_anything_but_a_plain_parquet_name(sandbox, stub_query, filename):
    assert export(sandbox, filename=filename).startswith("ERROR: filename")
    assert "sql" not in stub_query  # rejected before touching the database


def test_export_reports_too_many_rows(sandbox, stub_query):
    stub_query["out"] = TooManyRows("query returned more than 100000 rows")
    assert export(sandbox) == "ERROR: query returned more than 100000 rows"


def test_export_reports_sql_errors(sandbox, stub_query):
    stub_query["out"] = asyncpg.exceptions.UndefinedColumnError('column "view" does not exist')
    assert export(sandbox) == 'ERROR: column "view" does not exist'


def test_export_through_the_agents_composite_backend(sandbox, stub_query, tmp_path):
    """Paths outside /skills/ route to the sandbox; the skills dir is untouched."""
    skills = tmp_path / "skills"
    skills.mkdir()
    backend = CompositeBackend(
        default=sandbox, routes={"/skills/": FilesystemBackend(root_dir=skills, virtual_mode=True)}
    )
    export(backend, filename="x.parquet")
    assert (tmp_path / "data" / "x.parquet").exists()
    assert list(skills.iterdir()) == []


# --- download_outputs --------------------------------------------------------


def test_download_copies_outputs_keeping_their_layout(sandbox, tmp_path):
    sandbox.upload_files([("/outputs/chart.png", b"PNG"), ("/outputs/report/summary.md", b"# hi")])
    local = tmp_path / "local"
    written = download_outputs(sandbox, OUTPUT_DIR, local)
    assert sorted(p.relative_to(local).as_posix() for p in written) == ["chart.png", "report/summary.md"]
    assert (local / "chart.png").read_bytes() == b"PNG"
    assert (local / "report" / "summary.md").read_bytes() == b"# hi"


def test_download_ignores_files_outside_the_output_dir(sandbox, tmp_path):
    sandbox.upload_files([("/data/raw.csv", b"a,b"), ("/outputs/chart.png", b"PNG")])
    written = download_outputs(sandbox, OUTPUT_DIR, tmp_path / "local")
    assert [p.name for p in written] == ["chart.png"]


def test_download_with_no_outputs_writes_nothing(sandbox, tmp_path):
    assert download_outputs(sandbox, OUTPUT_DIR, tmp_path / "local") == []
    assert not (tmp_path / "local").exists()


# --- the enforcer covers export_query too ------------------------------------


def test_export_query_needs_the_query_database_skill():
    enforcer = SkillEnforcerMiddleware(
        cfg.agent.skills_dir, target_skills="query_database",
        shared_skills={"export_query": "query_database"},
    )
    call = {"name": "export_query", "args": {"sql": "SELECT 1", "filename": "a.parquet"}, "id": "e1"}
    blocked = enforcer._check_skills({"messages": [AIMessage(content="", tool_calls=[call])]})
    assert "/skills/query_database/SKILL.md" in blocked["messages"][0].content

    read = {"name": "read_file", "args": {"file_path": "/skills/query_database/SKILL.md"}, "id": "r1"}
    history = [
        AIMessage(content="", tool_calls=[read]),
        ToolMessage(content="@@ lines 1-41 of 41 @@\n...", tool_call_id="r1"),
        AIMessage(content="", tool_calls=[call]),
    ]
    assert enforcer._check_skills({"messages": history}) is None


# --- build_agent -------------------------------------------------------------


def tool_names(graph) -> set[str]:
    return set(graph.nodes["tools"].bound._tools_by_name)


def test_agent_without_a_sandbox_has_no_export(monkeypatch):
    assert "export_query" not in tool_names(build_agent())


def test_agent_with_a_sandbox_gets_export_and_execute(sandbox):
    names = tool_names(build_agent(sandbox=sandbox))
    assert {"query_database", "export_query", "execute"} <= names


# --- config ------------------------------------------------------------------


def test_sandbox_dirs_must_be_absolute():
    with pytest.raises(ValidationError, match="absolute"):
        SandboxConfig(provider="none", data_dir="data", output_dir="/o", export_max_rows=1)


def test_options_default_to_empty():
    config = SandboxConfig(provider="none", data_dir="/d", output_dir="/o", export_max_rows=1)
    assert config.options == {}
