"""A report node's page: the batches connected to it, as one HTML page.

Built from a workflow run when it is asked for, not stored: the run and its
analysis runs' output files do not change once it ends. Each batch is a
section; each analysis in it shows its status and its charts. The charts are
embedded (iframe srcdoc), so the page works on its own when downloaded. They
were written by the agent's scripts, so each stays in a sandboxed frame that
may run its chart script and nothing more.

The load of a run is a page of its own (`render_load`), not part of a
report: a report is the results. The load page covers the whole run: every
batch, and each analysis's memory and CPU while its script ran.
"""

from collections.abc import Callable
from html import escape

from api.vendor import local_plotly

from .load import STYLE as LOAD_STYLE
from .load import analysis_load, batch_load, run_load
from .models import NodeRun, WorkflowRun

ReadOutput = Callable[[str, str, str], bytes | None]
"""(analysis id, run id, file name) -> the file that run made, or None."""

STATUS = {
    "waiting": "waiting for a sandbox…",
    "running": "running…",
    "done": "done",
    "failed": "failed",
    "needs_optimization": "needs optimization: it stopped before exporting",
}

STYLE = """
body { margin: 0; font: 15px/1.5 system-ui, -apple-system, 'Segoe UI', 'Noto Sans TC', sans-serif;
       color: #1c1c1a; background: #f7f7f5; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { margin: 0 0 4px; font-size: 1.6em; }
h2 { margin: 32px 0 8px; font-size: 1.25em; border-bottom: 1px solid #deded8; padding-bottom: 6px; }
h3 { margin: 0; font-size: 1.05em; }
.muted { color: #6b6b66; }
.item { background: #fff; border: 1px solid #deded8; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
.item header { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
.status { font-size: 0.85em; padding: 1px 8px; border-radius: 99px; border: 1px solid currentColor; white-space: nowrap; }
.done { color: #1f8a4c; } .failed { color: #c2372b; } .needs_optimization, .running { color: #9a6a00; }
.error { color: #c2372b; }
iframe { display: block; width: 100%; height: 520px; border: 1px solid #deded8; border-radius: 8px;
         margin-top: 10px; background: #fff; }
"""


def _tables(sources: dict[str, str]) -> str:
    return ", ".join(f"{old} → {new}" for old, new in sources.items()) or "its saved tables"


def _item(node: NodeRun, read_output: ReadOutput) -> str:
    details = [f"on {escape(_tables(node.sources))}"]
    if node.version is not None:
        details.append(f"version {node.version}")
    if node.seconds is not None:
        details.append(f"{node.seconds:.0f}s")
    parts = [
        f'<section class="item"><header><h3>{escape(node.title)}</h3>'
        f'<span class="status {node.status}">{escape(STATUS[node.status])}</span></header>',
        f'<p class="muted">{" · ".join(details)}</p>',
    ]
    if node.error:
        parts.append(f'<p class="error">{escape(node.error)}</p>')
    for name in node.outputs:
        content = read_output(node.analysis_id, node.run_id, name) if node.run_id else None
        if content is None:
            parts.append(f'<p class="muted">{escape(name)} is no longer there.</p>')
            continue
        page = local_plotly(content).decode("utf-8", errors="replace")
        parts.append(f'<iframe sandbox="allow-scripts" title="{escape(name)}" srcdoc="{escape(page, quote=True)}"></iframe>')
    parts.append("</section>")
    return "\n".join(parts)


def render(run: WorkflowRun, report_id: str, read_output: ReadOutput) -> str:
    """The page for report node `report_id` in `run`. Raises KeyError if the run has no such report."""
    batch_ids = run.reports[report_id]
    by_id = {n.node_id: n for n in run.nodes}
    groups = [(run.titles.get(b, "Batch"), [by_id[i] for i in run.batches.get(b, []) if i in by_id]) for b in batch_ids]
    # An analysis in two of these batches ran once: count it once.
    nodes = list({n.node_id: n for _, members in groups for n in members}.values())
    done = sum(1 for n in nodes if n.status == "done")
    title = run.titles.get(report_id, "Report")
    took = f" · took {run.seconds:.0f}s" if run.seconds is not None else ""
    sections = []
    for name, members in groups:
        body = "\n".join(_item(n, read_output) for n in members) or '<p class="muted">Nothing was connected to it.</p>'
        sections.append(f"<h2>{escape(name)}</h2>\n{body}")
    if not sections:
        sections.append('<p class="muted">No batch was connected to this report.</p>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>{STYLE}</style></head>
<body><main>
<h1>{escape(title)}</h1>
<p class="muted">{escape(run.workflow_title or "Workflow")} · run {escape(run.started_at)}{took} · {done} of {len(nodes)} analyses done</p>
{chr(10).join(sections)}
</main></body></html>
"""


def render_load(run: WorkflowRun) -> str:
    """The load page of `run`: every batch in it, how much it used and when, down to each analysis."""
    by_id = {n.node_id: n for n in run.nodes}
    groups = [(run.titles.get(b, "Batch"), [by_id[i] for i in ids if i in by_id]) for b, ids in run.batches.items()]
    nodes = list({n.node_id: n for _, members in groups for n in members}.values())
    title = f"Load: {run.workflow_title or 'Workflow'}"
    took = f" · took {run.seconds:.0f}s" if run.seconds is not None else ""
    sections = []
    for name, members in groups:
        analyses = []
        for n in members:
            chart = analysis_load(n) or '<p class="muted">No samples for this analysis.</p>'
            analyses.append(f'<section class="item"><header><h3>{escape(n.title)}</h3>'
                            f'<span class="status {n.status}">{escape(STATUS[n.status])}</span></header>{chart}</section>')
        sections.append(f"<h2>{escape(name)}</h2>\n{batch_load(run, members, nodes)}\n" + "\n".join(analyses))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>{STYLE}{LOAD_STYLE}</style></head>
<body><main>
<h1>{escape(title)}</h1>
<p class="muted">run {escape(run.started_at)}{took} · {len(nodes)} analyses</p>
{run_load(run, nodes, groups)}
{chr(10).join(sections)}
</main></body></html>
"""
