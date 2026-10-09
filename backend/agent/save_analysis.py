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
from analyses.budget import Limits, check_estimates, rows as fmt_rows
from analyses.models import Measurements, OptimizationNote, now
from analyses.runner import Count, Estimate, Query, ReadDataset, estimate_inputs, fetch_inputs, run_in_sandbox
from analyses.same_results import differences
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
    estimate: Estimate | None = None,
    limits: Limits | None = None,
    count: Count | None = None,
) -> BaseTool:
    """`estimate` and `limits` are used when saving an optimized version (analyses/optimize.py)."""

    async def optimization_problems(analysis: Analysis, current: Analysis, pending: dict) -> list[str]:
        """Before running anything: an optimized version keeps its outputs and must fit the target tables."""
        problems = []
        if {o.file for o in analysis.outputs} != {o.file for o in current.outputs}:
            problems.append("an optimized version makes the same output files as the current one: "
                            + ", ".join(o.file for o in current.outputs))
        if estimate is None or limits is None:
            return problems
        sources = pending.get("sources") or {}
        estimates = await estimate_inputs(analysis, estimate, sources, count, limits)
        problems += [f"on {', '.join(sources.values()) or 'its tables'} it still would not fit: {f.reason}"
                     for f in check_estimates(Measurements(inputs=estimates), limits)]
        if not problems and pending.get("kind") == "data_movement":
            for m in estimates:
                n = m.counted_rows if m.counted_rows is not None else m.estimated_rows
                if n is not None and n > limits.soft_export_rows:
                    problems.append(f"{m.file} would still export about {fmt_rows(n)} rows from "
                                    f"{', '.join(m.tables)}; reduce it below {fmt_rows(limits.soft_export_rows)} "
                                    "rows in SQL (filter, select only needed columns, aggregate)")
        return problems

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
        pending = store.optimizing(replaces) if replaces else None
        current = store.get(replaces) if pending else None
        if pending and current:
            problems = await optimization_problems(analysis, current, pending)
            if problems:
                return ("NOT SAVED. This is an optimization of version "
                        f"{current.version}, and the new recipe does not meet it yet:\n- " + "\n- ".join(problems)
                        + "\nFix the recipe and call save_analysis again.")

        started_at, started = now(), time.monotonic()
        fetched = await fetch_inputs(analysis, query, read_dataset)
        if fetched.problem:
            return f"NOT SAVED: {fetched.problem}. Fix the recipe and call save_analysis again."
        data = fetched.files
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

        compared = ""
        if pending and current:
            before = await fetch_inputs(current, query, read_dataset)
            old_run = None if before.problem else await asyncio.to_thread(
                run_in_sandbox, sandbox, current, before.files, work_dir / CHECK_FOLDER / f"{analysis.id}-before")
            if old_run is not None and old_run.ok:
                changed, notes = differences(old_run.outputs, result.outputs)
                if changed:
                    return (f"NOT SAVED. On the saved tables, the new recipe gives different results than version "
                            f"{current.version}; an optimization must keep the results:\n- " + "\n- ".join(changed)
                            + "\nFix the recipe and call save_analysis again.")
                compared = (" Its results match version {v} on the saved tables." if not notes
                            else " " + "; ".join(notes) + ".").format(v=current.version)
            else:
                compared = f" Version {current.version} could not run on the saved tables, so results were not compared."
            # The analysis stays linked to the chat it was saved from; this conversation goes in the note.
            optimizer, analysis.conversation = analysis.conversation, current.conversation
            analysis.optimization = OptimizationNote(
                from_version=current.version, kind=pending.get("kind", "data_movement"),
                reason=pending.get("reason", ""), sources=pending.get("sources") or {},
                run=pending.get("run"), conversation=optimizer, results_check=compared.strip() or None,
            )

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
        if pending:
            what += " (optimized)"
        return (f"{what}: \"{analysis.title}\". Its test run made {files} in {seconds:.0f}s.{compared} "
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
        script, its output files, its most recent runs, and, for a version an
        optimization saved, why (`optimized`: from which version, the problem,
        the tables). Earlier versions are not shown.

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
            "optimized": {**a.optimization.model_dump(),
                          "how": "a run broke a limit, a user clicked Optimize, and you rewrote the recipe; "
                                 "it was saved after the save checks passed (nothing changes on its own)"}
                         if a.optimization else None,
            "conversation": a.conversation,
            "inputs": [i.model_dump() for i in a.inputs], "tables": a.source_tables(),
            "script": a.script, "outputs": [o.file for o in a.outputs],
            "recent_runs": [_run_summary(r) for r in store.runs(a.id)[:RECENT_RUNS]],
        }, ensure_ascii=False)

    return [list_saved_analyses, read_saved_analysis]
