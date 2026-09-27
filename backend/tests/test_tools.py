"""The query_database tool's size limit. No database: queries are stubbed."""

import asyncio
import json

import asyncpg
import pytest

import agent.tools as tools
from agent.tools import estimate_tokens, format_result, query_database
from config import cfg
from datasources import TooManyRows


def rows(n: int, columns: int = 3) -> list[dict]:
    return [{f"col{c}": f"value-{i}-{c}" for c in range(columns)} for i in range(n)]


def size(result: list[dict]) -> int:
    return estimate_tokens(json.dumps(result, ensure_ascii=False))


# --- estimate_tokens -----------------------------------------------------------


@pytest.mark.parametrize("text, expected", [("", 0), ("abc", 1), ("abcd", 1), ("abcde", 2)])
def test_estimate_tokens_rounds_up(text, expected):
    assert estimate_tokens(text) == expected


# --- format_result -------------------------------------------------------------


def test_small_result_is_returned_as_json():
    result = rows(3)
    assert json.loads(format_result(result, max_tokens=5000)) == result


def test_empty_result_is_returned():
    assert format_result([], max_tokens=5000) == "[]"


def test_result_exactly_at_the_limit_is_returned():
    result = rows(10)
    assert json.loads(format_result(result, max_tokens=size(result))) == result


def test_result_one_token_over_the_limit_is_refused():
    result = rows(10)
    assert format_result(result, max_tokens=size(result) - 1).startswith("ERROR:")


def test_refusal_reports_the_size_so_the_model_can_adapt():
    result = rows(1000, columns=5)
    message = format_result(result, max_tokens=5000)
    assert f"~{size(result):,} tokens" in message
    assert "1,000 rows x 5 columns" in message
    assert "limit of 5,000" in message


def test_refusal_contains_none_of_the_data():
    """Refuse, don't truncate: no partial result the model could mistake for all of it."""
    message = format_result(rows(1000), max_tokens=5000)
    assert "value-" not in message
    assert len(message) < 300


def test_limit_is_on_size_not_on_row_count():
    """Same number of rows, different width: only the wide one is refused."""
    narrow, wide = rows(200, columns=1), rows(200, columns=20)
    limit = size(narrow)
    assert not format_result(narrow, max_tokens=limit).startswith("ERROR:")
    assert format_result(wide, max_tokens=limit).startswith("ERROR:")


def test_non_ascii_is_kept_readable():
    """Titles stay as written, not \\uXXXX escapes (which would also inflate the size)."""
    result = [{"title": "BTS (방탄소년단) 'FAKE LOVE'"}]
    assert "방탄소년단" in format_result(result, max_tokens=5000)


# --- the tool --------------------------------------------------------------------


@pytest.fixture
def stub_query(monkeypatch):
    """Replace the database call; returns a setter for what it yields or raises."""
    state = {}

    async def fake(sql: str):
        state["sql"] = sql
        if isinstance(state.get("out"), Exception):
            raise state["out"]
        return state["out"]

    monkeypatch.setattr(tools, "_query_database", fake)
    return state


def run(sql: str = "SELECT 1") -> str:
    return asyncio.run(query_database.ainvoke({"sql": sql}))


def test_tool_passes_the_sql_through(stub_query):
    stub_query["out"] = rows(1)
    run("SELECT category_name FROM videos")
    assert stub_query["sql"] == "SELECT category_name FROM videos"


def test_tool_returns_a_small_result(stub_query):
    stub_query["out"] = rows(2)
    assert json.loads(run()) == rows(2)


def test_tool_uses_the_configured_limit(stub_query, monkeypatch):
    stub_query["out"] = rows(10)
    monkeypatch.setattr(cfg.agent, "max_result_tokens", size(rows(10)) - 1)
    assert run().startswith("ERROR: result is")
    monkeypatch.setattr(cfg.agent, "max_result_tokens", size(rows(10)))
    assert not run().startswith("ERROR:")


def test_tool_reports_too_many_rows(stub_query):
    stub_query["out"] = TooManyRows("query returned more than 1000 rows")
    assert run() == "ERROR: query returned more than 1000 rows"


def test_tool_reports_sql_errors(stub_query):
    stub_query["out"] = asyncpg.exceptions.UndefinedColumnError('column "view" does not exist')
    assert run() == 'ERROR: column "view" does not exist'


def test_tool_lets_other_errors_raise(stub_query):
    """A lost connection is not something the model can fix by rewriting SQL."""
    stub_query["out"] = ConnectionError("database unreachable")
    with pytest.raises(ConnectionError):
        run()


# --- config ----------------------------------------------------------------------


def test_shipped_config_sets_a_result_limit():
    assert cfg.agent.max_result_tokens > 0
