"""Saved analyses: the recipe checks, the store, running one in a sandbox, the tool, and the routes.

The sandbox is a local shell in a temp dir, so scripts really run (stdlib
only, with fake Parquet bytes: the runner never looks inside an input).
"""

import asyncio
from pathlib import PurePosixPath

import pytest
from deepagents.backends.local_shell import LocalShellBackend
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.save_analysis import make_save_analysis
from analyses import Analysis, AnalysisStore, DatasetInput, Output, QueryInput, RunRecord
from analyses.models import typed_number_lists
from analyses.runner import run
from analyses.sources import differences, swap_tables, tables
from api import ConversationManager
from api.analyses import analyses_router
from test_api import TempDirProvider

SCRIPT = """\
import os
data = open(os.path.join(os.environ["DATA_DIR"], "monthly.parquet"), "rb").read()
with open(os.path.join(os.environ["OUTPUT_DIR"], "chart.html"), "w") as f:
    f.write(f"<p>{len(data)} bytes</p>")
print("drew it")
"""


def recipe(**fields) -> Analysis:
    return Analysis(**{
        "id": "abc123", "title": "Monthly", "inputs": [QueryInput(sql="SELECT 1", file="monthly.parquet")],
        "script": SCRIPT, "outputs": [Output(file="chart.html")], **fields,
    })


QUERIES: list[str] = []


async def fake_query(sql: str) -> bytes:
    QUERIES.append(sql)
    if "boom" in sql:
        raise RuntimeError('column "boom" does not exist')
    return b"PAR1-rows-for:" + sql.encode()


COLUMNS = {
    "videos": {"id": "bigint", "video_id": "text", "views": "bigint", "publish_hour": "smallint"},
    "videos_ca": {"video_id": "text", "views": "bigint", "publish_hour": "bigint"},  # no id: fine
    "sales": {"month": "date", "amount": "numeric"},
    "videos_bad": {"video_id": "text", "views": "text"},
}


async def fake_rows(sql: str) -> list[dict]:
    """information_schema.columns, as the agent's database user sees it."""
    import re

    wanted = re.search(r"table_name IN \((.*)\)", sql)
    names = re.findall(r"'([^']+)'", wanted.group(1)) if wanted else list(COLUMNS)
    return [{"table_name": t, "column_name": c, "data_type": d}
            for t in names if t in COLUMNS for c, d in COLUMNS[t].items()]


def no_datasets(name: str) -> bytes | None:
    return b"PAR1-sales" if name == "sales" else None


def shell(tmp_path) -> LocalShellBackend:
    return LocalShellBackend(root_dir=tmp_path, virtual_mode=False, inherit_env=True)


# --- the recipe --------------------------------------------------------------------


def test_a_complete_recipe_has_no_problems():
    assert recipe().problems() == []


def test_a_script_that_never_reads_its_input_is_refused():
    copied = recipe(script='import os\nopen(os.environ["OUTPUT_DIR"] + "/chart.html", "w").write(str([904, 1365]))\n'
                           'print(os.environ["DATA_DIR"])')
    assert copied.problems() == ["the script never reads monthly.parquet: load every input from $DATA_DIR"]


@pytest.mark.parametrize("fields, problem", [
    ({"inputs": []}, "name at least one input"),
    ({"outputs": [Output(file="../x.html")]}, "must be a plain name ending in .html"),
    ({"inputs": [QueryInput(sql="SELECT 1", file="a.csv")]}, "must be a plain name ending in .parquet"),
    ({"script": 'open("/home/data/monthly.parquet")'}, "os.environ['DATA_DIR']"),
])
def test_recipe_problems(fields, problem):
    assert any(problem in p for p in recipe(**fields).problems())


def test_a_dataset_input_is_its_name_as_a_parquet_file():
    assert DatasetInput(name="sales").file == "sales.parquet"


# --- the store ---------------------------------------------------------------------


def test_the_store_keeps_recipes_and_runs(tmp_path):
    store = AnalysisStore(tmp_path)
    store.save(recipe())
    assert store.get("abc123").inputs[0].sql == "SELECT 1"
    assert [a.id for a in store.all()] == ["abc123"]
    store.save_run("abc123", RunRecord(id="20261006-100000-aaaa", started_at="t", status="done", trigger="save"),
                   {"chart.html": b"<p>old</p>"})
    store.save_run("abc123", RunRecord(id="20261006-110000-bbbb", started_at="t", status="failed", trigger="run"))
    assert [r.id for r in store.runs("abc123")] == ["20261006-110000-bbbb", "20261006-100000-aaaa"]
    assert store.output_path("abc123", "20261006-100000-aaaa", "chart.html").read_bytes() == b"<p>old</p>"
    assert store.output_path("abc123", "20261006-100000-aaaa", "../analysis.json") is None
    assert store.get("../etc") is None
    assert store.delete("abc123") and store.get("abc123") is None


# --- running -------------------------------------------------------------------------


def go(sandbox, analysis, scratch):
    return asyncio.run(run(sandbox, analysis, query=fake_query, read_dataset=no_datasets, scratch=scratch))


def test_a_run_fetches_the_inputs_runs_the_script_and_collects_the_outputs(tmp_path):
    scratch = PurePosixPath(tmp_path) / "scratch"
    result = go(shell(tmp_path), recipe(), scratch)
    assert result.ok, result.error
    assert result.outputs == {"chart.html": b"<p>22 bytes</p>"}  # len(b"PAR1-rows-for:SELECT 1")
    assert "drew it" in result.log
    assert not (tmp_path / "scratch").exists()  # the scratch folder is cleaned up


def test_a_run_starts_in_an_empty_folder(tmp_path):
    """A file left in the work folder by the conversation is not there for the script."""
    scratch = PurePosixPath(tmp_path) / "scratch"
    (tmp_path / "scratch" / "data").mkdir(parents=True)
    (tmp_path / "scratch" / "data" / "leftover.parquet").write_bytes(b"x")
    script = SCRIPT.replace('"monthly.parquet"', '"leftover.parquet"') + "\n# monthly.parquet"
    result = go(shell(tmp_path), recipe(script=script), scratch)
    assert not result.ok and "failed" in result.error and "leftover.parquet" in result.log


def test_failures_say_why(tmp_path):
    scratch = PurePosixPath(tmp_path) / "scratch"
    bad_sql = go(shell(tmp_path), recipe(inputs=[QueryInput(sql="SELECT boom", file="monthly.parquet")]), scratch)
    assert bad_sql.error == 'the query for monthly.parquet failed: column "boom" does not exist'
    no_file = go(shell(tmp_path), recipe(outputs=[Output(file="other.html")]), scratch)
    assert no_file.error == "the script did not write other.html to $OUTPUT_DIR"
    missing = go(shell(tmp_path), recipe(inputs=[DatasetInput(name="gone")], script=SCRIPT.replace("monthly", "gone")), scratch)
    assert missing.error == "there is no uploaded file dataset named 'gone'"
    killed = go(shell(tmp_path), recipe(script=SCRIPT + "import signal; os.kill(os.getpid(), signal.SIGKILL)"), scratch)
    assert killed.killed and killed.error == "it ran out of memory (killed)"


# --- the tool ------------------------------------------------------------------------


def save_tool(tmp_path):
    store = AnalysisStore(tmp_path / "saved")
    tool = make_save_analysis(shell(tmp_path), store, query=fake_query, read_dataset=no_datasets,
                              work_dir=PurePosixPath(tmp_path) / "work")
    return tool, store


def call(tool, **args):
    base = {"title": "Monthly", "question": "Chart videos per month",
            "inputs": [{"kind": "query", "sql": "SELECT 1", "file": "monthly.parquet"}],
            "script": SCRIPT, "outputs": ["chart.html"]}
    return asyncio.run(tool.ainvoke({**base, **args}, config={"configurable": {"thread_id": "conv-1"}}))


def test_the_tool_test_runs_then_saves_with_the_first_run(tmp_path):
    tool, store = save_tool(tmp_path)
    answer = call(tool)
    [saved] = store.all()
    assert answer.startswith(f'Saved analysis {saved.id}: "Monthly"')
    assert (saved.conversation, saved.question) == ("conv-1", "Chart videos per month")
    [first] = store.runs(saved.id)
    assert (first.trigger, first.status, first.outputs) == ("save", "done", ["chart.html"])


def test_the_tool_saves_nothing_when_the_recipe_fails(tmp_path):
    tool, store = save_tool(tmp_path)
    assert call(tool, script="print('no env vars')").startswith("NOT SAVED. Fix the recipe")
    failed = call(tool, script=SCRIPT + "raise SystemExit(3)")
    assert failed.startswith("NOT SAVED: the test run in an empty folder failed: the script failed (exit code 3)")
    assert "drew it" in failed  # its output, so the agent can fix it
    assert store.all() == []


# --- the routes ------------------------------------------------------------------------


class ShellProvider(TempDirProvider):
    """Each sandbox a local shell with real paths, so the run's absolute scratch folder works."""

    def create(self):
        sandbox = super().create()
        return LocalShellBackend(root_dir=sandbox.cwd, virtual_mode=False, inherit_env=True)


def app_for(tmp_path, analysis: Analysis):
    store = AnalysisStore(tmp_path / "saved")
    store.save(analysis)
    provider = ShellProvider(tmp_path)
    manager = ConversationManager(lambda sandbox, checkpointer: object(), provider, idle_seconds=900)
    app = FastAPI()
    app.include_router(analyses_router(store, manager, query=fake_query, query_rows=fake_rows, read_dataset=no_datasets,
                                       work_dir=PurePosixPath(tmp_path) / "work", account="test_user"))
    return app, store, provider


def wait_for(client, analysis_id, run_id):
    for _ in range(200):
        record = client.get(f"/api/analyses/{analysis_id}/runs/{run_id}").json()
        if record["status"] != "running":
            return record
        asyncio.run(asyncio.sleep(0.02))
    raise AssertionError("the run did not finish")


def test_run_takes_a_sandbox_runs_it_and_gives_the_sandbox_back(tmp_path):
    app, store, provider = app_for(tmp_path, recipe())
    with TestClient(app) as client:
        [listed] = client.get("/api/analyses").json()
        assert (listed["id"], listed["last_run"]) == ("abc123", None)
        started = client.post("/api/analyses/abc123/runs")
        assert started.status_code == 202
        record = wait_for(client, "abc123", started.json()["id"])
        assert (record["status"], record["outputs"], record["trigger"]) == ("done", ["chart.html"], "run")
        page = client.get(f"/api/analyses/abc123/runs/{record['id']}/files/chart.html")
        assert page.text == "<p>22 bytes</p>"
        assert page.headers["content-security-policy"] == "sandbox allow-scripts"
        assert client.get("/api/analyses").json()[0]["last_good_run"]["id"] == record["id"]
    assert provider.destroyed  # the run's sandbox was deleted after


def test_a_run_killed_for_memory_is_not_moved_to_a_bigger_sandbox_first(tmp_path):
    """Rewriting the script comes first (test_adaptive_runs.py); the bigger sandbox only after that."""
    killed = SCRIPT + "import signal; os.kill(os.getpid(), signal.SIGKILL)"
    app, store, _ = app_for(tmp_path, recipe(script=killed))
    with TestClient(app) as client:
        record = wait_for(client, "abc123", client.post("/api/analyses/abc123/runs").json()["id"])
    assert record["status"] == "failed" and record["error"] == "it ran out of memory (killed)"
    assert record["notes"] == [] and record["measurements"]["sandbox"]["killed"] is True


def test_missing_things_are_404_and_delete_works(tmp_path):
    app, store, _ = app_for(tmp_path, recipe())
    with TestClient(app) as client:
        assert client.get("/api/analyses/nope").status_code == 404
        assert client.post("/api/analyses/nope/runs").status_code == 404
        assert client.get("/api/analyses/abc123/runs/20261006-100000-aaaa/files/x.html").status_code == 404
        assert client.delete("/api/analyses/abc123").json() == {"deleted": "abc123"}
        assert client.get("/api/analyses").json() == []


# --- no typed-in data ------------------------------------------------------------------


def test_a_script_that_types_in_a_list_of_numbers_is_refused():
    typed = SCRIPT + "counts = [904, 1365, 1433, 1210, 878, 718, 730, 368]\n"
    assert typed_number_lists(typed) == [(6, 8)]
    assert any("line 6 of the script types in 8 numbers" in p for p in recipe(script=typed).problems())
    # a few fixed settings are fine
    assert typed_number_lists(SCRIPT + "bins = [0, 10, 100]\nsize = (8, 4)\n") == []


# --- other tables --------------------------------------------------------------------


def test_tables_are_the_real_table_references():
    sql = ("WITH m AS (SELECT video_id, MAX(views) AS videos FROM videos v GROUP BY 1) "
           "SELECT COUNT(*) AS videos FROM m JOIN public.channels USING (video_id)")
    assert tables(sql) == ["channels", "videos"]
    with pytest.raises(ValueError):
        tables("SELEC nope FROM")


def test_a_swapped_table_keeps_the_old_name_as_its_alias():
    swapped = swap_tables("SELECT videos.views AS videos FROM videos WHERE videos.views > 0", {"videos": "videos_ca"})
    assert swapped == "SELECT videos.views AS videos FROM videos_ca AS videos WHERE videos.views > 0"
    aliased = swap_tables("SELECT v.views FROM videos AS v", {"videos": "videos_ca"})
    assert aliased == "SELECT v.views FROM videos_ca AS v"
    assert swap_tables("SELECT 1 FROM videos", {}) == "SELECT 1 FROM videos"  # untouched text


def test_a_stand_in_needs_every_column_with_the_same_kind_of_type():
    assert differences(COLUMNS["videos"], COLUMNS["videos_ca"]) == []  # id ignored; smallint ~ bigint
    assert differences(COLUMNS["videos"], COLUMNS["videos_bad"]) == ["views is text, not bigint", "no column publish_hour"]


def test_the_recipe_knows_its_tables():
    assert recipe(inputs=[QueryInput(sql="SELECT views FROM videos", file="monthly.parquet")]).source_tables() == ["videos"]


def test_sources_lists_which_tables_could_stand_in(tmp_path):
    app, _, _ = app_for(tmp_path, recipe(inputs=[QueryInput(sql="SELECT views FROM videos", file="monthly.parquet")]))
    with TestClient(app) as client:
        assert client.get("/api/analyses/abc123").json()["sources"] == ["videos"]
        [option] = client.get("/api/analyses/abc123/sources").json()
    assert option["table"] == "videos"
    assert {c["name"]: c["ok"] for c in option["candidates"]} == {"sales": False, "videos_bad": False, "videos_ca": True}


def test_run_on_another_table(tmp_path):
    QUERIES.clear()
    app, store, _ = app_for(tmp_path, recipe(inputs=[QueryInput(sql="SELECT views FROM videos", file="monthly.parquet")]))
    with TestClient(app) as client:
        refused = client.post("/api/analyses/abc123/runs", json={"sources": {"videos": "videos_bad"}})
        assert refused.status_code == 400
        assert refused.json()["detail"] == ("can't run on those tables: videos_bad: views is text, not bigint; "
                                            "videos_bad: no column publish_hour")
        assert client.post("/api/analyses/abc123/runs", json={"sources": {"sales": "videos"}}).status_code == 400

        started = client.post("/api/analyses/abc123/runs", json={"sources": {"videos": "videos_ca"}})
        record = wait_for(client, "abc123", started.json()["id"])
    assert (record["status"], record["sources"]) == ("done", {"videos": "videos_ca"})
    assert QUERIES == ["SELECT views FROM videos_ca AS videos"]


# --- the agent sees what is saved --------------------------------------------------


def readers(tmp_path):
    from agent.save_analysis import make_analysis_readers

    store = AnalysisStore(tmp_path / "saved")
    list_tool, read_tool = make_analysis_readers(store)
    return store, list_tool, read_tool


def test_the_agent_lists_saved_analyses(tmp_path):
    import json

    store, list_tool, _ = readers(tmp_path)
    assert list_tool.invoke({}).startswith("No analyses are saved yet")
    store.save(recipe(inputs=[QueryInput(sql="SELECT views FROM videos", file="monthly.parquet"), DatasetInput(name="sales")],
                      question="Chart videos per month", script=SCRIPT + "# sales.parquet"))
    store.save_run("abc123", RunRecord(id="20261006-120000-aaaa", started_at="t", status="failed", trigger="run",
                                       error="it ran out of memory (killed)", sources={"videos": "videos_ca"}))
    [entry] = json.loads(list_tool.invoke({}))
    assert entry["id"] == "abc123" and entry["question"] == "Chart videos per month"
    assert (entry["tables"], entry["uploaded_files"], entry["outputs"]) == (["videos"], ["sales"], ["chart.html"])
    assert entry["last_run"] == {"at": "t", "status": "failed", "how": "run", "version": 1,
                                 "read_instead": {"videos": "videos_ca"}, "error": "it ran out of memory (killed)"}


def test_the_agent_reads_one_recipe(tmp_path):
    import json

    store, _, read_tool = readers(tmp_path)
    store.save(recipe())
    found = json.loads(read_tool.invoke({"analysis_id": " abc123 "}))
    assert found["script"] == SCRIPT
    assert found["inputs"] == [{"kind": "query", "sql": "SELECT 1", "file": "monthly.parquet"}]
    assert found["recent_runs"] == []
    assert read_tool.invoke({"analysis_id": "nope"}).startswith("ERROR: no saved analysis with id 'nope'")


def test_the_agent_always_has_the_readers_and_save_only_with_a_sandbox():
    from agent import build_agent

    def names(agent):
        return set(agent.nodes["tools"].bound.tools_by_name)

    without = names(build_agent(require_sql_skill=False))
    assert {"list_saved_analyses", "read_saved_analysis"} <= without and "save_analysis" not in without


# --- results typed into the script ------------------------------------------------------


def parquet(rows):
    from agent.tools import to_parquet

    return to_parquet(rows)


RESULTS = {"by_category.parquet": parquet([
    {"category": "Music", "avg_views": 6_012_345.0, "videos": 799},
    {"category": "Entertainment", "avg_views": 1_745_000.0, "videos": 1621},
])}


def test_results_written_into_the_report_text_are_found():
    from analyses.typed_data import typed_results

    script = (
        'import os\n'
        'top = df["category"][0]\n'
        'html = f"""<style>.box {{ max-width: 900px; color: #222; }}</style>\n'
        '<p>Music leads all categories with an average of 6.0M views per video.</p>\n'
        '<p>{top} is first; the top 20 are shown; data covers 1,621 videos.</p>"""\n'
    )
    found = [(f.line, f.text, f.column) for f in typed_results(script, RESULTS)]
    assert found == [(3, "6.0M", "avg_views"), (3, "Music", "category"), (5, "1,621", "videos")]
    assert "put it in the query's WHERE" in typed_results(script, RESULTS)[0].problem()


def test_computed_text_passes():
    from analyses.typed_data import typed_results

    script = ('import os\nimport polars as pl\n'
              'df = pl.read_parquet(os.environ["DATA_DIR"] + "/by_category.parquet")\n'
              'top = df.row(0, named=True)\n'
              'html = f"<p>{top[\'category\']} leads with {top[\'avg_views\'] / 1e6:.1f}M views; top 10 shown</p>"\n')
    assert typed_results(script, RESULTS) == []


def test_the_tool_refuses_results_typed_into_the_script(tmp_path):
    store = AnalysisStore(tmp_path / "saved")

    async def results(sql):
        return RESULTS["by_category.parquet"]

    tool = make_save_analysis(shell(tmp_path), store, query=results, read_dataset=no_datasets,
                              work_dir=PurePosixPath(tmp_path) / "work")
    typed = SCRIPT.replace("monthly.parquet", "by_category.parquet").replace("bytes</p>", "bytes; Music leads</p>")
    answer = call(tool, inputs=[{"kind": "query", "sql": "SELECT 1", "file": "by_category.parquet"}], script=typed)
    assert answer.startswith("NOT SAVED. The script has results typed into it")
    assert "\u201cMusic\u201d, a result from by_category.parquet (category)" in answer
    assert store.all() == []


# --- changing a saved analysis ----------------------------------------------------------


def test_changing_a_saved_analysis_makes_a_new_version_of_it(tmp_path):
    tool, store = save_tool(tmp_path)
    call(tool)
    [first] = store.all()
    answer = call(tool, title="Monthly, no table", replaces=first.id, script=SCRIPT.replace("drew it", "drew it again"))
    assert answer.startswith(f'Updated analysis {first.id} to version 2: "Monthly, no table"')
    [current] = store.all()
    assert (current.id, current.version, current.title, current.created_at) == (first.id, 2, "Monthly, no table", first.created_at)
    assert current.updated_at is not None
    assert (tmp_path / "saved" / first.id / "versions" / "1.json").is_file()
    assert [r.version for r in store.runs(first.id)] == [2, 1]


def test_changing_an_unknown_analysis_saves_nothing(tmp_path):
    tool, store = save_tool(tmp_path)
    assert call(tool, replaces="nope").startswith("NOT SAVED: there is no saved analysis 'nope' to change")
    assert store.all() == []


def test_dates_written_into_the_text_are_found_whatever_the_inputs():
    from analyses.typed_data import typed_results

    script = ('html = f"<h2>US Trending Videos - November 2017 to June 2018</h2>"\n'
              'note = "from 2017-11-14, monthly since 2018-01; Sept. 3, 2019"\n'
              'ok = "<p>{first} to {last}, 12 months, in 2 columns</p>"\n')
    found = [(f.line, f.text) for f in typed_results(script, {})]
    assert found == [(1, "June 2018"), (1, "November 2017"), (2, "2017-11-14"), (2, "2018-01"), (2, "Sept. 3, 2019")]
    assert "read dates and date ranges from the data" in typed_results(script, {})[0].problem()
