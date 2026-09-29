"""Uploads: parsing, naming, the registry, the store and its routes.

Postgres is stubbed: `create_table` / `drop_table` are replaced by recorders.
The real table path is exercised by hand against the database.
"""

import datetime as dt
import io
import json
from pathlib import Path, PurePosixPath

import polars as pl
import pytest
from deepagents.backends.local_shell import LocalShellBackend
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import datasets as datasets_module
from agent import touches_videos
from agent.middleware import SkillEnforcerMiddleware
from agent.tools import make_list_datasets
from api.datasets import datasets_router, upload_files_hook
from datasets import DatasetStore, Registry, UploadError
from datasets.ingest import clean_columns, pg_type, read_upload
from datasets.registry import Column, Dataset, check_name
from datasets import suggest_name

# Run the async tests on asyncio via anyio's plugin (installed with httpx).
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


DATA_DIR = PurePosixPath("/data")
CSV = b"Category,Views (M),Published,Is Live\nMusic,1.5,2031-01-02,true\nGaming,2.25,2031-01-03,false\n"


@pytest.fixture
def db(monkeypatch):
    """Record table creates and drops instead of touching Postgres."""
    calls = {"create": [], "drop": []}

    async def create_table(dsn, name, df, reader_role):
        calls["create"].append((dsn, name, df, reader_role))

    async def drop_table(dsn, name):
        calls["drop"].append(name)

    monkeypatch.setattr(datasets_module, "create_table", create_table)
    monkeypatch.setattr(datasets_module, "drop_table", drop_table)
    return calls


@pytest.fixture
def store(tmp_path, db) -> DatasetStore:
    return DatasetStore(tmp_path / "uploads", max_bytes=1 << 20,
                        ingest_dsn=lambda: "postgresql://ingest@db", reader_role=lambda: "agent_ro")


def stage(store, content=CSV, filename="Sales 2031.csv"):
    return store.stage(filename, io.BytesIO(content))


# --- parsing and naming ----------------------------------------------------------


def test_clean_columns():
    assert clean_columns(["Views (M)", "2024", "  Title ", "title", "Title!", "", "é"]) == [
        "views_m", "c_2024", "title", "title_2", "title_3", "column", "column_2",
    ]


def test_column_names_fit_postgres():
    assert len(clean_columns(["x" * 100])[0]) <= 63


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (pl.Int64, "bigint"), (pl.Int32, "integer"), (pl.Int16, "smallint"), (pl.UInt64, "numeric"),
        (pl.Float64, "double precision"), (pl.Float32, "real"), (pl.Boolean, "boolean"),
        (pl.Date, "date"), (pl.Datetime("us"), "timestamp"), (pl.Datetime("us", "UTC"), "timestamptz"),
        (pl.String, "text"), (pl.List(pl.Int64), "text"),
    ],
)
def test_pg_type(dtype, expected):
    assert pg_type(dtype) == expected


def test_csv_types_are_inferred(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(CSV)
    df = read_upload(path)
    assert df.columns == ["category", "views_m", "published", "is_live"]
    assert df.schema["views_m"] == pl.Float64
    assert df.schema["published"] == pl.Date
    assert df.schema["is_live"] == pl.Boolean


def test_parquet_is_read(tmp_path):
    path = tmp_path / "a.parquet"
    pl.DataFrame({"When": [dt.date(2031, 1, 1)]}).write_parquet(path)
    assert read_upload(path).schema == {"when": pl.Date}


def test_unreadable_file_is_a_clear_error(tmp_path):
    path = tmp_path / "a.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(UploadError, match="could not read a.parquet"):
        read_upload(path)


@pytest.mark.parametrize("name, ok", [
    ("sales_2031", True), ("s", True), ("Sales", False), ("2031_sales", False), ("sales-2031", False),
    ("videos", False), ("pg_stats", False), ("x" * 64, False), ("", False),
])
def test_check_name(name, ok):
    assert (check_name(name) is None) is ok


@pytest.mark.parametrize("filename, expected", [
    ("Sales 2031 (Q1).csv", "sales_2031_q1"), ("2031.csv", "d_2031"), ("videos.csv", "videos_data"),
    ("???.parquet", "dataset"),
])
def test_suggest_name_is_always_valid(filename, expected):
    assert suggest_name(filename) == expected
    assert check_name(suggest_name(filename)) is None


# --- registry ---------------------------------------------------------------------


def dataset(name="sales", kind="table") -> Dataset:
    return Dataset(name=name, kind=kind, rows=2, columns=[Column(name="a", type="bigint")],
                   source="s.csv", created_at=dt.datetime(2031, 1, 1, tzinfo=dt.UTC))


def test_registry_round_trip(tmp_path):
    registry = Registry(tmp_path / "r.json")
    assert registry.list() == []
    registry.add(dataset("a"))
    registry.add(dataset("b", "file"))
    assert [d.name for d in Registry(tmp_path / "r.json").list()] == ["a", "b"]  # persisted
    assert registry.remove("a") and not registry.remove("a")
    assert [d.name for d in registry.list()] == ["b"]


def test_registry_leaves_no_temp_file(tmp_path):
    Registry(tmp_path / "r.json").add(dataset())
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]


# --- the store --------------------------------------------------------------------


def test_stage_parses_and_previews(store):
    staged = stage(store)
    assert staged.rows == 2 and staged.suggested_name == "sales_2031"
    assert [(c.name, c.type) for c in staged.columns] == [
        ("category", "text"), ("views_m", "double precision"), ("published", "date"), ("is_live", "boolean"),
    ]
    assert staged.preview[0] == {"category": "Music", "views_m": 1.5, "published": "2031-01-02", "is_live": True}


def test_stage_refuses_big_files_and_keeps_nothing(store):
    store.max_bytes = 10
    with pytest.raises(UploadError, match="larger than"):
        stage(store)
    assert not any(store.staging.iterdir())


def test_stage_refuses_other_file_types(store):
    with pytest.raises(UploadError, match="unsupported file type .xlsx"):
        stage(store, filename="a.xlsx")


def test_stage_of_a_bad_file_keeps_nothing(store):
    with pytest.raises(UploadError):
        stage(store, content=b"\x00\x01", filename="a.parquet")
    assert not any(store.staging.iterdir())


async def test_commit_as_table(store, db):
    staged = stage(store)
    result = await store.commit(staged.upload_id, "table", "sales_2031")
    [(dsn, name, df, reader)] = db["create"]
    assert (dsn, name, reader) == ("postgresql://ingest@db", "sales_2031", "agent_ro")
    assert df.height == 2
    assert result.kind == "table" and result.source == "Sales 2031.csv"
    assert store.registry.get("sales_2031") is not None
    assert not any(store.staging.iterdir())  # staging cleared


async def test_commit_as_file_stores_typed_parquet(store, db):
    staged = stage(store)
    result = await store.commit(staged.upload_id, "file", "sales")
    assert db["create"] == []
    df = pl.read_parquet(store.file_path("sales"))
    assert df.schema["published"] == pl.Date and df.height == 2
    assert {c.name: c.type for c in result.columns}["published"] == "Date"


@pytest.mark.parametrize("name", ["Bad Name", "videos"])
async def test_commit_rejects_bad_names(store, name):
    with pytest.raises(UploadError):
        await store.commit(stage(store).upload_id, "table", name)


async def test_commit_rejects_a_taken_name(store):
    await store.commit(stage(store).upload_id, "file", "sales")
    with pytest.raises(UploadError, match="already exists"):
        await store.commit(stage(store).upload_id, "table", "sales")


@pytest.mark.parametrize("upload_id", ["../../etc", "0" * 32])
async def test_commit_rejects_unknown_uploads(store, upload_id):
    with pytest.raises(UploadError, match="unknown"):
        await store.commit(upload_id, "file", "sales")


async def test_delete(store, db):
    await store.commit(stage(store).upload_id, "table", "t")
    await store.commit(stage(store).upload_id, "file", "f")
    assert await store.delete("t") and await store.delete("f")
    assert db["drop"] == ["t"]
    assert not store.file_path("f").exists()
    assert store.list() == [] and not await store.delete("t")


async def test_sandbox_files_lists_only_files(store):
    await store.commit(stage(store).upload_id, "table", "t")
    await store.commit(stage(store).upload_id, "file", "f")
    [(path, content)] = store.sandbox_files(DATA_DIR)
    assert path == "/data/uploads/f.parquet" and content == store.file_path("f").read_bytes()


# --- routes ------------------------------------------------------------------------


@pytest.fixture
def api(store, tmp_path):
    live = [LocalShellBackend(root_dir=tmp_path / "live", virtual_mode=True)]
    (tmp_path / "live").mkdir()
    app = FastAPI()
    app.include_router(datasets_router(store, DATA_DIR, lambda: live))
    with TestClient(app) as client:
        yield client, live[0], tmp_path / "live"


def upload(client, content=CSV, filename="Sales 2031.csv"):
    return client.post("/api/uploads", files={"file": (filename, content, "text/csv")})


def test_upload_route_returns_the_preview(api):
    client, _, _ = api
    body = upload(client).json()
    assert body["rows"] == 2 and body["suggested_name"] == "sales_2031" and len(body["upload_id"]) == 32


def test_upload_route_errors(api, store):
    client, _, _ = api
    assert upload(client, filename="a.xlsx").status_code == 400
    store.max_bytes = 10
    assert upload(client).status_code == 413


def test_commit_and_list_routes(api):
    client, _, _ = api
    upload_id = upload(client).json()["upload_id"]
    response = client.post("/api/datasets", json={"upload_id": upload_id, "kind": "table", "name": "sales"})
    assert response.status_code == 201
    names = [(d["name"], d["built_in"]) for d in client.get("/api/datasets").json()]
    assert names == [("videos", True), ("sales", False)]


def test_commit_route_rejects_bad_names(api):
    client, _, _ = api
    upload_id = upload(client).json()["upload_id"]
    response = client.post("/api/datasets", json={"upload_id": upload_id, "kind": "table", "name": "DROP TABLE"})
    assert response.status_code == 400


def test_a_new_file_reaches_running_sandboxes(api):
    client, _, root = api
    upload_id = upload(client).json()["upload_id"]
    client.post("/api/datasets", json={"upload_id": upload_id, "kind": "file", "name": "sales"})
    assert pl.read_parquet(root / "data" / "uploads" / "sales.parquet").height == 2


def test_delete_route(api):
    client, _, _ = api
    upload_id = upload(client).json()["upload_id"]
    client.post("/api/datasets", json={"upload_id": upload_id, "kind": "file", "name": "sales"})
    assert client.delete("/api/datasets/sales").status_code == 200
    assert client.delete("/api/datasets/sales").status_code == 404


async def test_new_sandboxes_get_every_file_dataset(store, tmp_path):
    await store.commit(stage(store).upload_id, "file", "sales")
    (tmp_path / "sb").mkdir()
    sandbox = LocalShellBackend(root_dir=tmp_path / "sb", virtual_mode=True)
    await upload_files_hook(store, DATA_DIR)(sandbox)
    assert (tmp_path / "sb" / "data" / "uploads" / "sales.parquet").exists()


# --- what the agent sees ------------------------------------------------------------


async def test_list_datasets_tool(store):
    await store.commit(stage(store).upload_id, "table", "sales")
    await store.commit(stage(store).upload_id, "file", "survey")
    entries = json.loads(make_list_datasets(store.registry, DATA_DIR).invoke({}))
    assert [e["name"] for e in entries] == ["videos", "sales", "survey"]
    assert "SKILL.md" in entries[0]["description"]
    assert entries[1]["columns"]["published"] == "date" and "path" not in entries[1]
    assert entries[2]["path"] == "/data/uploads/survey.parquet"


async def test_list_datasets_without_a_sandbox_marks_files_unavailable(store):
    await store.commit(stage(store).upload_id, "file", "survey")
    entries = json.loads(make_list_datasets(store.registry, None).invoke({}))
    assert entries[1]["path"].startswith("unavailable")


@pytest.mark.parametrize("sql, expected", [
    ("SELECT * FROM videos", True), ('SELECT * FROM "videos"', True), ("select count(*) from public.VIDEOS", True),
    ("SELECT * FROM sales_2031", False), ("SELECT * FROM videos_backup", False), ("", False),
])
def test_touches_videos(sql, expected):
    assert touches_videos({"name": "query_database", "args": {"sql": sql}}) is expected


def test_enforcer_only_blocks_sql_on_videos():
    from config import cfg

    enforcer = SkillEnforcerMiddleware(cfg.agent.skills_dir, target_skills="query_database", applies_to=touches_videos)

    def check(sql):
        call = {"name": "query_database", "args": {"sql": sql}, "id": "q1", "type": "tool_call"}
        return enforcer._check_skills({"messages": [AIMessage(content="", tool_calls=[call])]})

    assert check("SELECT * FROM videos") is not None
    assert check("SELECT * FROM sales_2031") is None
