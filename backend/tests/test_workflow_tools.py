"""The agent's workflow tools: save_workflow lays out sections, run_workflow runs and reports back.

They call the same workflow functions as the page (api/workflows.py), with the
test sandboxes (local shells) and the adaptive-runs fixtures: `events` is small,
`events_big` too big for the saved recipe.
"""

import asyncio
import json
from pathlib import PurePosixPath

import pytest

from agent import workflows as tools_module
from agent.workflows import Item, Section, layout, make_workflow_tools
from analyses import AnalysisStore
from api import ConversationManager
from api.analyses import analyses_api
from api.workflows import workflows_api
from test_adaptive_runs import LIMITS, fake_estimate, fake_query, fake_rows, naive
from test_analyses import ShellProvider, no_datasets
from workflows import WorkflowStore


@pytest.fixture
def tools(tmp_path):
    analyses = AnalysisStore(tmp_path / "saved")
    analyses.save(naive())
    analyses.save(naive(id="other1", title="Another chart"))
    manager = ConversationManager(lambda sandbox, checkpointer: object(), ShellProvider(tmp_path), idle_seconds=900)
    api = analyses_api(analyses, manager, query=fake_query, query_rows=fake_rows, read_dataset=no_datasets,
                       work_dir=PurePosixPath(tmp_path) / "work", account="test_user",
                       estimate=fake_estimate, limits=LIMITS)
    tools_module.connect(workflows_api(WorkflowStore(tmp_path / "workflows"), analyses, api, query_rows=fake_rows,
                                       sandbox_limit=3))
    yield {t.name: t for t in make_workflow_tools()}
    tools_module.connect(None)


def call(tool, **args) -> str:
    return asyncio.run(tool.ainvoke(args))


def test_sections_become_a_canvas():
    body = layout("Weekly", [
        Section(title="Users", analyses=[Item(analysis_id="sessions1"), Item(analysis_id="other1", run_first=True)]),
        Section(title="Again", analyses=[Item(analysis_id="sessions1"),
                                         Item(analysis_id="sessions1", tables={"events": "events_big"})]),
    ], None, None)
    kinds = [(n["kind"], n["id"]) for n in body["nodes"]]
    # sessions1 on its own tables is one node in both sections; on other tables it is another.
    assert kinds == [("analysis", "a1"), ("analysis", "a2"), ("batch", "b1"), ("analysis", "a3"), ("batch", "b2"),
                     ("report", "r1")]
    assert body["edges"] == [{"source": "a1", "target": "b1"}, {"source": "a2", "target": "b1"},
                             {"source": "a1", "target": "b2"}, {"source": "a3", "target": "b2"},
                             {"source": "b1", "target": "r1"}, {"source": "b2", "target": "r1"}]
    nodes = {n["id"]: n for n in body["nodes"]}
    assert (nodes["a2"]["priority"], nodes["a3"]["sources"], nodes["r1"]["title"]) == (1, {"events": "events_big"}, "Weekly")
    assert nodes["a1"]["position"]["x"] < nodes["b1"]["position"]["x"] < nodes["r1"]["position"]["x"]
    assert body["settings"] == {"lanes": None}


def test_the_agent_saves_runs_lists_and_changes_a_workflow(tools):
    sections = [{"title": "Sessions", "analyses": [{"analysis_id": "sessions1"},
                                                   {"analysis_id": "sessions1", "tables": {"events": "events_big"}}]},
                {"title": "More", "analyses": [{"analysis_id": "other1"}]}]
    saved = call(tools["save_workflow"], title="Weekly", sections=sections, report_title="Weekly report")
    assert saved.startswith('Saved workflow ') and '"Weekly": 3 analyses in 2 sections' in saved
    workflow_id = saved.split()[2].rstrip(":")

    ran = call(tools["run_workflow"], workflow_id=workflow_id)
    first, *rest = ran.splitlines()
    assert first.startswith(f"Ran workflow {workflow_id}: partly done in ") and "(2 of 3 analyses done, 2 sandboxes)" in first
    assert any("needs optimization" in line and "/analyses/sessions1?run=" in line for line in rest)
    assert f"Workflow: /workflows/{workflow_id}" in rest
    assert any(line.startswith(f"Report: /api/workflows/{workflow_id}/runs/") and line.endswith("/reports/r1") for line in rest)

    [listed] = json.loads(call(tools["list_workflows"]))
    assert (listed["id"], listed["title"], listed["last_run"]["status"]) == (workflow_id, "Weekly", "partly_done")
    assert [s["title"] for s in listed["sections"]] == ["Sessions", "More"]
    assert listed["sections"][0]["analyses"][1]["tables"] == {"events": "events_big"}

    changed = call(tools["save_workflow"], title="Weekly", replaces=workflow_id,
                   sections=[{"title": "Only", "analyses": [{"analysis_id": "other1"}]}])
    assert changed.startswith(f"Updated workflow {workflow_id}:") and "1 analyses in 1 sections" in changed


def test_what_cannot_be_saved_or_run_says_why(tools):
    unknown = call(tools["save_workflow"], title="W", sections=[{"title": "S", "analyses": [{"analysis_id": "gone"}]}])
    assert unknown.startswith("NOT SAVED:") and "no saved analysis 'gone'" in unknown
    empty = call(tools["save_workflow"], title="W", sections=[])
    assert empty.startswith("NOT SAVED: give at least one section")
    assert call(tools["run_workflow"], workflow_id="nope").startswith("NOT RUN: no workflow 'nope'")


def test_without_the_app_the_tools_say_so():
    tools_module.connect(None)
    [list_workflows, *_] = make_workflow_tools()
    assert call(list_workflows) == "ERROR: workflows are not available here."
