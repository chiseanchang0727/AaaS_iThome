"""Tools for saved analyses: save one, and see the ones saved.

save_analysis turns the conversation's analysis into a recipe that can be run
again. The recipe is checked, then test-run in an empty scratch folder of the
conversation's sandbox (analyses/runner.py), and saved only if that run made
every output. So an analysis that saved also runs from the Analyses page.

list_saved_analyses and read_saved_analysis only read; they need no sandbox.
"""

import asyncio
import json
import time

from pathlib import Path, PurePosixPath

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from analyses import Analysis, AnalysisStore, DatasetInput, Output, QueryInput, RunRecord
from analyses.models import now
from analyses.runner import Query, ReadDataset, fetch_inputs, run_in_sandbox
from analyses.store import new_id, new_run_id
from analyses.typed_data import typed_results
from datasets import Registry

CHECK_FOLDER = ".analysis-check"
"""The scratch folder for the test run, inside the sandbox's work folder."""


def dataset_reader(registry: Registry, files_dir: Path) -> ReadDataset:
    """Reads an uploaded file dataset's Parquet file from the server's uploads."""

    def read(name: str) -> bytes | None:
        dataset = registry.get(name)
        path = files_dir / f"{name}.parquet"
        return path.read_bytes() if dataset is not None and dataset.kind == "file" and path.is_file() else None

    return read


def make_save_analysis(
    sandbox: SandboxBackendProtocol,
    store: AnalysisStore,
    *,
    query: Query,
    read_dataset: ReadDataset,
    work_dir: PurePosixPath,
) -> BaseTool:
    @tool
    async def save_analysis(
        title: str,
        question: str,
        inputs: list[QueryInput | DatasetInput],
        script: str,
        outputs: list[str],
        config: RunnableConfig,
        description: str = "",
        replaces: str = "",
    ) -> str:
        """Save the analysis from this conversation so it can be run again later,
        on the data as it is then. Call it only when the user asks to save an
        analysis, or to change one that is saved.

        To change a saved analysis, pass its id as `replaces` (read it first
        with read_saved_analysis): the new recipe becomes its next version,
        keeping its id and run history. Leave `replaces` empty only for a new
        analysis; never save a changed copy as a new one.

        It saves a recipe, not your results: a later run gets fresh data and
        may read other tables with the same columns, so nothing in it may
        depend on what you saw in this conversation.
        - `inputs`: where the data comes from, each loaded into $DATA_DIR before
          the script runs. A query: {"kind": "query", "sql": "...", "file":
          "monthly.parquet"}, saved like export_query. Name tables plainly
          (`FROM videos`). An uploaded file dataset: {"kind": "dataset",
          "name": "sales"}, at $DATA_DIR/sales.parquet.
        - `script`: complete Python that reads every input with
          `os.environ["DATA_DIR"]` and writes every output with
          `os.environ["OUTPUT_DIR"]`. Hard-code nothing that came from the data:
          no numbers, labels, top-N names, ids or date ranges you saw in a
          result, not even in the report's text ("Music leads with 6.0M
          views"): compute them from the inputs, in SQL or in the script, or a
          later run will show stale or wrong data. Polars and plotly, as in
          the sandbox. A script whose fixed text contains values from its own
          inputs' results, or that types in a list of numbers, is refused. When
          changing a saved analysis, fix any of this in its old script too.
        - `outputs`: the file names the script writes, ending in .html
          (e.g. "trending_by_month.html").
        - `title`: a few words; `question`: the user's question this answers;
          `description`: one or two sentences on what it shows. Like the
          script, title and description must stay true on other data: say
          what is measured ("distinct trending videos per month"), not the
          results, the country or the date range you saw.

        It runs the recipe in an empty folder before saving. If that fails, it
        says why and nothing is saved: fix the recipe and call it again.
        """
        replaces = replaces.strip()
        if replaces and store.get(replaces) is None:
            return (f"NOT SAVED: there is no saved analysis {replaces!r} to change. "
                    "list_saved_analyses shows the ids; leave `replaces` empty for a new analysis.")
        analysis = Analysis(
            id=replaces or new_id(), title=title.strip(), description=description.strip(), question=question.strip(),
            conversation=(config.get("configurable") or {}).get("thread_id"),
            inputs=inputs, script=script, outputs=[Output(file=f) for f in outputs],
        )
        problems = analysis.problems()
        if problems:
            return "NOT SAVED. Fix the recipe and call save_analysis again:\n- " + "\n- ".join(problems)

        started_at, started = now(), time.monotonic()
        data, problem = await fetch_inputs(analysis, query, read_dataset)
        if problem:
            return f"NOT SAVED: {problem}. Fix the recipe and call save_analysis again."
        typed = typed_results(analysis.script, data)
        if typed:
            return ("NOT SAVED. The script has results typed into it, which would be wrong on other data:\n- "
                    + "\n- ".join(f.problem() for f in typed) + "\nFix the script and call save_analysis again.")

        result = await asyncio.to_thread(run_in_sandbox, sandbox, analysis, data, work_dir / CHECK_FOLDER / analysis.id)
        seconds = round(time.monotonic() - started, 2)
        if not result.ok:
            log = f"\nIts output:\n{result.log}" if result.log else ""
            return (f"NOT SAVED: the test run in an empty folder failed: {result.error}.{log}\n"
                    "Fix the recipe and call save_analysis again.")

        if replaces:
            analysis = store.replace(analysis)
        else:
            store.save(analysis)
        store.save_run(analysis.id, RunRecord(
            id=new_run_id(), started_at=started_at, status="done", trigger="save", version=analysis.version,
            seconds=seconds, outputs=list(result.outputs), log=result.log,
        ), result.outputs)
        files = ", ".join(result.outputs)
        what = (f"Updated analysis {analysis.id} to version {analysis.version}" if replaces
                else f"Saved analysis {analysis.id}")
        return (f"{what}: \"{analysis.title}\". Its test run made {files} in {seconds:.0f}s. "
                "The user can run it again from the Analyses page.")

    return save_analysis


RECENT_RUNS = 5


def _run_summary(run: RunRecord) -> dict:
    summary = {"at": run.started_at, "status": run.status, "how": run.trigger, "version": run.version}
    if run.sources:
        summary["read_instead"] = run.sources
    if run.error:
        summary["error"] = run.error
    return summary


def make_analysis_readers(store: AnalysisStore) -> list[BaseTool]:
    """list_saved_analyses and read_saved_analysis over `store`."""

    @tool
    def list_saved_analyses() -> str:
        """List the analyses users saved (with save_analysis), newest first: id,
        title, description, the question it answers, the tables it reads, the
        files it makes, and its last run.

        Use it when the user asks what is saved, or refers to a saved analysis
        ("my monthly chart"). Users run them from the Analyses page, on the
        tables they read or other tables with the same columns. Read one with
        read_saved_analysis to see its SQL and script.
        """
        entries = []
        for a in store.all():
            runs = store.runs(a.id)
            entries.append({
                "id": a.id, "title": a.title, "description": a.description, "question": a.question,
                "saved_at": a.created_at, "version": a.version, "changed_at": a.updated_at,
                "tables": a.source_tables(),
                "uploaded_files": [i.name for i in a.inputs if isinstance(i, DatasetInput)],
                "outputs": [o.file for o in a.outputs],
                "last_run": _run_summary(runs[0]) if runs else None,
            })
        if not entries:
            return "No analyses are saved yet. Users save one by asking you to save an analysis."
        return json.dumps(entries, ensure_ascii=False)

    @tool
    def read_saved_analysis(analysis_id: str) -> str:
        """Read one saved analysis by its id (from list_saved_analyses): its
        inputs (the SQL of each query, or the uploaded file it reads), its
        script, its output files, and its most recent runs.

        Use it to explain what a saved analysis does, or to start from its
        recipe: adapt its SQL and script for a new question, then run them in
        the sandbox as usual (and save_analysis again if the user wants).
        """
        a = store.get(analysis_id.strip())
        if a is None:
            return f"ERROR: no saved analysis with id {analysis_id!r}. list_saved_analyses shows the ids."
        return json.dumps({
            "id": a.id, "title": a.title, "description": a.description, "question": a.question,
            "saved_at": a.created_at, "version": a.version, "changed_at": a.updated_at,
            "conversation": a.conversation,
            "inputs": [i.model_dump() for i in a.inputs], "tables": a.source_tables(),
            "script": a.script, "outputs": [o.file for o in a.outputs],
            "recent_runs": [_run_summary(r) for r in store.runs(a.id)[:RECENT_RUNS]],
        }, ensure_ascii=False)

    return [list_saved_analyses, read_saved_analysis]
