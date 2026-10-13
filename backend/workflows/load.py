"""The load of a workflow run over time, drawn as SVG for its report page.

Each analysis node goes through three phases, from its analysis run's times:

    queued_at ── waiting ──> sandbox_at ── moving data ──> script_at ── script ──> script end ── moving data ──> finished_at
    (size check, a free sandbox, start-up)   (inputs in)                                  (outputs out, sandbox back)

While the script runs, the sandbox samples it every 0.1 s (sandboxes/metering.py):
the memory of its processes, and the CPU time used so far. Each run has a
sandbox of its own, so those are that analysis's.

A workflow run passes a few sandboxes from one analysis to the next (api/analyses.py, Lane): a
sandbox may run several of its analyses, cleaned up in between.

Drawn for the whole run, for each batch, and for each analysis:
    - each sandbox over time: its start-up, the analyses it ran (by batch), the clean-ups (whole run)
    - how many analyses were in a sandbox, and how many not yet, over time (whole run and batches)
    - memory over time, added up over the analyses (whole run and batches), or its own (analysis)
    - CPU over time, in cores (analysis)
    - one bar per analysis, in its phases
No scripts and no fonts to fetch: the page works on its own when saved.
"""

from dataclasses import dataclass
from datetime import datetime
from html import escape

from .models import NodeRun, WorkflowRun

WAITING = "#f2c14e"
MOVING = "#9cbcf0"
SCRIPT = "#2a6fdb"
MEMORY = "#7b4fd6"
CPU = "#1f8a4c"
LIMIT = "#c2372b"
STARTUP = "#c9c9c2"
CLEANUP = "#e6e6e0"
BATCHES = ["#2a6fdb", "#eb6834", "#1f8a4c", "#7b4fd6", "#b5179e", "#0b8a8a"]
"""A colour per batch, in order; text on them is white."""

STYLE = """
.load { background: #fff; border: 1px solid #deded8; border-radius: 10px; padding: 14px 16px; margin: 16px 0 8px; }
.load h2, .load h3 { margin: 0 0 6px; border: 0; padding: 0; }
.load h4 { margin: 14px 0 2px; font-size: 0.95em; }
.batch-load { margin: 4px 0 12px; padding: 10px 14px; background: #fbfbf9; border: 1px solid #ecece6; border-radius: 10px; }
.facts { display: flex; flex-wrap: wrap; gap: 8px 24px; margin: 8px 0 12px; padding: 0; list-style: none; }
.facts b { display: block; font-size: 1.15em; }
.facts span { color: #6b6b66; font-size: 0.85em; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 0.85em; color: #6b6b66; margin: 4px 0; }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 5px; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 4px 16px; }
svg text { font: 12px system-ui, -apple-system, 'Segoe UI', sans-serif; fill: #6b6b66; }
svg .label { fill: #1c1c1a; }
svg .group { fill: #1c1c1a; font-weight: 600; }
"""


def _time(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def size(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1000:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:,.1f} GB"


@dataclass
class Phases:
    """One analysis's times, in seconds from the start of the run (the first analysis asked for)."""

    node: NodeRun
    queued: float
    ready: float | None
    """Its sandbox was ready; None if it never got one."""
    script: float | None
    script_end: float | None
    finished: float

    def memory(self) -> list[tuple[float, float]]:
        """(time in the run, MB) while the script ran."""
        if self.script is None:
            return []
        return [(self.script + t, m) for t, m, _ in self.node.samples if m is not None]

    def cores(self) -> list[tuple[float, float]]:
        """(time in the run, CPU cores in use) between samples."""
        if self.script is None:
            return []
        cpu = [(t, u) for t, _, u in self.node.samples if u is not None]
        return [(self.script + t1, max(u1 - u0, 0) / (t1 - t0)) for (t0, u0), (t1, u1) in zip(cpu, cpu[1:]) if t1 > t0]


def phases(nodes: list[NodeRun]) -> list[Phases]:
    timed = [n for n in nodes if n.queued_at and n.finished_at]
    if not timed:
        return []
    start = min(_time(n.queued_at) for n in timed)
    found = []
    for n in timed:
        script = _time(n.script_at) - start if n.script_at else None
        end = script + n.script_seconds if script is not None and n.script_seconds is not None else None
        found.append(Phases(node=n, queued=_time(n.queued_at) - start, ready=_time(n.sandbox_at) - start if n.sandbox_at else None,
                            script=script, script_end=end, finished=_time(n.finished_at) - start))
    return found


def counts(found: list[Phases]) -> list[tuple[float, int, int]]:
    """(time, analyses in a sandbox, analyses not yet in one) at each moment that changes."""
    moments = sorted({t for p in found for t in (p.queued, p.ready, p.finished) if t is not None})
    return [
        (t,
         sum(1 for p in found if p.ready is not None and p.ready <= t < p.finished),
         sum(1 for p in found if p.queued <= t < (p.ready if p.ready is not None else p.finished)))
        for t in moments
    ]


def memory_total(found: list[Phases], end: float, steps: int = 240) -> list[tuple[float, float]]:
    """(time, MB) summed over the analyses: each at its latest sample, while its script runs."""
    curves = [(p, p.memory()) for p in found]
    curves = [(p, c) for p, c in curves if c]
    if not curves or end <= 0:
        return []
    total = []
    for i in range(steps + 1):
        t = end * i / steps
        mb = 0.0
        for p, curve in curves:
            if p.script_end is not None and not (curve[0][0] <= t <= p.script_end):
                continue
            mb += next((m for at, m in reversed(curve) if at <= t), 0.0)
        total.append((t, mb))
    return total


# --- drawing -----------------------------------------------------------------------------------------

WIDTH = 1000


def _ticks(maximum: float) -> list[float]:
    """Up to about five round steps from 0 to at least `maximum`."""
    if maximum <= 0:
        return [0, 1]
    raw = maximum / 4
    step = next(s * 10 ** e for e in range(-2, 8) for s in (1, 2, 2.5, 5) if s * 10 ** e >= raw)
    ticks = [0.0]
    while ticks[-1] < maximum:
        ticks.append(ticks[-1] + step)
    return ticks


def _number(v: float) -> str:
    return f"{v:,.0f}" if v >= 10 or v == int(v) else f"{v:,.1f}"


def _seconds(t: float, span: float) -> str:
    """A time-axis label: tenths of a second when the whole axis is short."""
    return f"{t:.1f}s" if span < 10 else f"{t:.0f}s"


def line_chart(points: list[tuple[float, float]], *, start: float, end: float, color: str, unit: str, label: str,
               limit: float | None = None, limit_label: str = "", height: int = 150, width: int = WIDTH) -> str:
    """A filled step line over [start, end]; `limit` dashed. Time on the x axis in seconds."""
    left, bottom, top, right = 52, 24, 22, 10
    plot_w, plot_h = width - left - right, height - bottom - top
    peak = max((v for _, v in points), default=0)
    ticks = _ticks(max(peak, limit or 0) * 1.05)
    y_max = ticks[-1] or 1
    span = (end - start) or 1
    x = lambda t: left + (t - start) / span * plot_w  # noqa: E731
    y = lambda v: top + plot_h - v / y_max * plot_h  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{escape(label)}; '
             f'highest {_number(peak)} {escape(unit)}">']
    for v in ticks:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#eeeeea"/>'
                     f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{_number(v)}</text>')
    if points:
        steps = []
        for i, (t, v) in enumerate(points):
            nxt = points[i + 1][0] if i + 1 < len(points) else t
            steps += [(x(t), y(v)), (x(nxt), y(v))]
        line = " ".join(f"{a:.1f},{b:.1f}" for a, b in steps)
        area = f"{steps[0][0]:.1f},{y(0):.1f} {line} {steps[-1][0]:.1f},{y(0):.1f}"
        parts.append(f'<polygon points="{area}" fill="{color}" fill-opacity="0.18"/>'
                     f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
    if limit:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(limit):.1f}" y2="{y(limit):.1f}" stroke="{LIMIT}" '
                     f'stroke-dasharray="6 4" stroke-width="1.5"/><text x="{width - right - 2}" y="{y(limit) - 5:.1f}" '
                     f'text-anchor="end">{escape(limit_label)}</text>')
    parts.append(f'<text x="{left - 6}" y="12" text-anchor="end" class="label">{escape(unit)}</text>')
    for i in range(6):
        t = start + span * i / 5
        parts.append(f'<text x="{x(t):.1f}" y="{height - 6}" text-anchor="middle">{_seconds(t, span)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def sandbox_of(p: Phases) -> str | None:
    """Which sandbox it ran in; a run from before sandboxes were shared had one of its own."""
    if p.node.sandbox:
        return p.node.sandbox
    return f"own-{p.node.node_id}" if p.ready is not None else None


def limit_of(run: WorkflowRun) -> tuple[int | None, str]:
    """The line the sandboxes in use are measured against, and what to call it."""
    if run.lanes:
        return run.lanes, f"this workflow's sandboxes: {run.lanes}"
    return run.sandbox_limit, f"sandbox limit: {run.sandbox_limit}"


def sandboxes_chart(found: list[Phases], end: float, batch_of: dict[str, int]) -> str:
    """One row per sandbox: its start-up, then each analysis it ran (coloured by batch), clean-ups between."""
    rows: dict[str, list[Phases]] = {}
    for p in sorted((p for p in found if p.ready is not None), key=lambda p: p.ready):
        rows.setdefault(sandbox_of(p), []).append(p)
    if not rows:
        return ""
    row, label_w = 30, 110
    plot_w = WIDTH - label_w - 20
    height = row * len(rows) + 24
    x = lambda t: label_w + (t / end if end else 0) * plot_w  # noqa: E731
    ran = sum(len(r) for r in rows.values())
    parts = [f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" role="img" aria-label="Each sandbox over time: '
             f'{len(rows)} sandbox{"" if len(rows) == 1 else "es"} for {ran} analys{"is" if ran == 1 else "es"}">']

    def block(start: float, stop: float, top: float, color: str, title: str, text: str = "") -> str:
        width = max(x(stop) - x(start), 1)
        label = ""
        if text and width > 40:
            fits = max(int(width / 7) - 1, 1)
            short = text if len(text) <= fits else text[: max(fits - 1, 1)] + "…"
            label = f'<text x="{x(start) + 5:.1f}" y="{top + 15}" style="fill:#fff">{escape(short)}</text>'
        return (f'<g><title>{escape(title)}</title><rect x="{x(start):.1f}" y="{top}" width="{width:.1f}" height="21" '
                f'rx="3" fill="{color}"/>{label}</g>')

    for i, (_, runs) in enumerate(rows.items()):
        top = i * row + 4
        parts.append(f'<text class="label" x="{label_w - 10}" y="{top + 15}" text-anchor="end">Sandbox {i + 1}</text>')
        first = runs[0]
        parts.append(block(first.queued, first.ready, top, STARTUP,
                           f"{first.ready - first.queued:.1f}s before it was ready (size check, start-up)"))
        for before, p in zip([None, *runs], runs):
            if before is not None:
                parts.append(block(before.finished, p.ready, top, CLEANUP,
                                   f"{p.ready - before.finished:.1f}s between analyses (clean-up)"))
            color = BATCHES[batch_of.get(p.node.node_id, 0) % len(BATCHES)]
            reused = ", reused" if p.node.reused else ", new sandbox"
            parts.append(block(p.ready, p.finished, top, color,
                               f"{p.node.title}: {p.finished - p.ready:.1f}s{reused}", p.node.title))
    for k in range(6):
        t = end * k / 5
        parts.append(f'<text x="{x(t):.1f}" y="{height - 6}" text-anchor="middle">{_seconds(t, end)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def counts_chart(found: list[Phases], end: float, limit: int | None, height: int = 170, width: int = WIDTH,
                 limit_label: str = "") -> str:
    """Analyses in a sandbox (blue) and not yet in one (yellow), stacked; the sandbox limit dashed."""
    moments = counts(found)
    left, bottom, top, right = 52, 24, 10, 10
    plot_w, plot_h = width - left - right, height - bottom - top
    peak = max((r for _, r, _ in moments), default=0)
    y_max = max(max((r + w for _, r, w in moments), default=0), limit or 0, 1)
    x = lambda t: left + (t / end if end else 0) * plot_w  # noqa: E731
    y = lambda v: top + plot_h - v / y_max * plot_h  # noqa: E731

    def steps(level) -> str:
        points = [(x(0), y(0))]
        for i, (t, r, w) in enumerate(moments):
            nxt = moments[i + 1][0] if i + 1 < len(moments) else end
            points += [(x(t), y(level(r, w))), (x(nxt), y(level(r, w)))]
        points.append((x(end), y(0)))
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in points)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Analyses in a sandbox and waiting for one, over time; at most {peak} in sandboxes at once">']
    for v in range(y_max + 1):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#eeeeea"/>'
                     f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    parts.append(f'<polygon points="{steps(lambda r, w: r + w)}" fill="{WAITING}"/>')
    parts.append(f'<polygon points="{steps(lambda r, w: r)}" fill="{SCRIPT}"/>')
    if limit:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(limit):.1f}" y2="{y(limit):.1f}" stroke="{LIMIT}" '
                     f'stroke-dasharray="6 4" stroke-width="1.5"/><text x="{width - right - 2}" y="{y(limit) - 5:.1f}" '
                     f'text-anchor="end">{escape(limit_label or f"sandbox limit: {limit}")}</text>')
    for i in range(6):
        t = end * i / 5
        parts.append(f'<text x="{x(t):.1f}" y="{height - 6}" text-anchor="middle">{_seconds(t, end)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bars(groups: list[tuple[str | None, list[Phases]]], end: float) -> str:
    """One row per analysis in its phases, under its group's name (a batch) when there is one."""
    row, label_w = 26, 300
    plot_w = WIDTH - label_w - 70
    rows = sum(len(g) + (1 if name else 0) for name, g in groups)
    height = row * rows + 22
    x = lambda t: label_w + (t / end if end else 0) * plot_w  # noqa: E731
    parts = [f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" role="img" aria-label="When each analysis waited, moved data and ran">']

    def bar(start: float, stop: float, top: float, color: str, title: str) -> str:
        return (f'<rect x="{x(start):.1f}" y="{top}" width="{max(x(stop) - x(start), 1):.1f}" height="17" rx="3" '
                f'fill="{color}"><title>{escape(title)}</title></rect>')

    i = 0
    for name, members in groups:
        if name:
            parts.append(f'<text class="group" x="8" y="{i * row + 17}">{escape(name)}</text>')
            i += 1
        for p in sorted(members, key=lambda p: p.queued):
            top = i * row + 4
            title = p.node.title if len(p.node.title) <= 42 else p.node.title[:41] + "…"
            parts.append(f'<text class="label" x="{label_w - 10}" y="{top + 13}" text-anchor="end">'
                         f'<title>{escape(p.node.title)}</title>{escape(title)}</text>')
            ready = p.ready if p.ready is not None else p.finished
            parts.append(bar(p.queued, ready, top, WAITING, f"{ready - p.queued:.1f}s before a sandbox was ready"))
            if p.ready is not None:
                parts.append(bar(p.ready, p.finished, top, MOVING, f"{p.finished - p.ready:.1f}s in its sandbox"))
            if p.script is not None and p.script_end is not None:
                memory = f", peak {p.node.peak_memory_mb:,.0f} MB" if p.node.peak_memory_mb else ""
                parts.append(bar(p.script, p.script_end, top, SCRIPT, f"{p.script_end - p.script:.1f}s running its script{memory}"))
            note = f"{p.node.peak_memory_mb:,.0f} MB" if p.node.peak_memory_mb else ("no sandbox" if p.ready is None else "")
            if p.node.reused:
                note += " · reused"
            parts.append(f'<text x="{x(p.finished) + 6:.1f}" y="{top + 13}">{note}</text>')
            i += 1
    for k in range(6):
        t = end * k / 5
        parts.append(f'<text x="{x(t):.1f}" y="{height - 6}" text-anchor="middle">{t:.0f}s</text>')
    parts.append("</svg>")
    return "".join(parts)


def legend(*items: tuple[str, str]) -> str:
    spans = "".join(f'<span><span class="swatch" style="background:{color}"></span>{escape(text)}</span>' for color, text in items)
    return f'<div class="legend">{spans}</div>'


PHASE_LEGEND = legend(
    (WAITING, "before its sandbox was ready (size check, waiting for a free one, start-up)"),
    (MOVING, "in its sandbox, moving data (inputs in, outputs out)"),
    (SCRIPT, "running its script"),
)


def facts(nodes: list[NodeRun], found: list[Phases], limit: int | None) -> list[tuple[str, str]]:
    """The numbers at the top of a load block."""
    found_facts = []
    if found:
        moments = counts(found)
        peak_at, peak, _ = max(moments, key=lambda c: (c[1], -c[0]))
        of = f" of {limit}" if limit else ""
        found_facts.append((f"{peak}{of}", f"sandboxes busy at the peak ({peak_at:.0f}s in)"))
        used = {sandbox_of(p) for p in found} - {None}
        ran = sum(1 for p in found if p.ready is not None)
        found_facts.append((f"{len(used)}", f"sandboxes started, for {ran} analyses"))
        waited = sum((p.ready if p.ready is not None else p.finished) - p.queued for p in found)
        found_facts.append((f"{waited:.0f}s", "before sandboxes were ready, all together"))
        span = max(p.finished for p in found) - min(p.queued for p in found)
        found_facts.append((f"{span:.0f}s", "from the first start to the last end"))
        total = memory_total(found, max(p.finished for p in found))
        if total:
            at, mb = max(total, key=lambda tv: tv[1])
            found_facts.append((f"{mb:,.0f} MB", f"memory in use at the peak, all together ({at:.0f}s in)"))
    moved = [n for n in nodes if n.rows is not None]
    if moved:
        found_facts.append((f"{sum(n.rows for n in moved):,} rows", f"{size(sum(n.bytes or 0 for n in moved))} into the sandboxes"))
    if any(n.cpu_seconds for n in nodes):
        found_facts.append((f"{sum(n.cpu_seconds or 0 for n in nodes):.0f}s", "CPU time of the scripts"))
    heaviest = max((n for n in nodes if n.peak_memory_mb), key=lambda n: n.peak_memory_mb, default=None)
    if heaviest is not None:
        of = f" of {heaviest.memory_limit_mb:,} MB" if heaviest.memory_limit_mb else ""
        found_facts.append((f"{heaviest.peak_memory_mb:,.0f} MB{of}", f"highest peak memory: {heaviest.title}"))
    return found_facts


def _facts_list(items: list[tuple[str, str]]) -> str:
    return '<ul class="facts">' + "".join(f"<li><b>{escape(v)}</b><span>{escape(k)}</span></li>" for v, k in items) + "</ul>"


def run_load(run: WorkflowRun, nodes: list[NodeRun], groups: list[tuple[str | None, list[NodeRun]]]) -> str:
    """The whole run's region of the performance page: numbers, the charts, the bars by batch."""
    found = phases(nodes)
    limit, limit_label = limit_of(run)
    parts = ['<section class="load"><h2>Whole run</h2>', _facts_list(facts(nodes, found, limit))]
    if not found:
        parts.append('<p class="muted">The timing of each analysis was not recorded for this run: '
                     "only runs from now on have load charts.</p></section>")
        return "\n".join(parts)
    end = max(p.finished for p in found)
    by_node = {p.node.node_id: p for p in found}
    parts += ["<h4>Analyses in sandboxes over time</h4>",
              legend((SCRIPT, "in a sandbox"), (WAITING, "before its sandbox was ready")),
              counts_chart(found, end, limit, limit_label=limit_label)]
    batch_of = {}
    for k, (_, members) in enumerate(groups):
        for n in members:
            batch_of.setdefault(n.node_id, k)
    chart = sandboxes_chart(found, end, batch_of)
    if chart:
        parts += ["<h4>Sandboxes over time</h4>",
                  legend((STARTUP, "starting up"), (CLEANUP, "clean-up between analyses"),
                         *((BATCHES[k % len(BATCHES)], name or "Batch") for k, (name, _) in enumerate(groups))),
                  chart]
    total = memory_total(found, end)
    if total:
        parts += ["<h4>Memory in use over time, all sandboxes together</h4>",
                  line_chart(total, start=0, end=end, color=MEMORY, unit="MB", label="Memory in use over time")]
    parts += ["<h4>Each analysis</h4>", PHASE_LEGEND,
              bars([(name, [by_node[n.node_id] for n in members if n.node_id in by_node]) for name, members in groups], end)]
    parts.append("</section>")
    return "\n".join(parts)


def batch_load(run: WorkflowRun, members: list[NodeRun], all_nodes: list[NodeRun]) -> str:
    """A batch's own load: its numbers, its analyses in sandboxes and its memory over time, its bars.
    On the run's time axis, so batches can be compared."""
    everyone = phases(all_nodes)
    if not everyone:
        return ""
    end = max(p.finished for p in everyone)
    ids = {n.node_id for n in members}
    found = [p for p in everyone if p.node.node_id in ids]
    if not found:
        return ""
    limit, limit_label = limit_of(run)
    parts = ['<div class="batch-load"><h3>Performance of this batch</h3>', _facts_list(facts(members, found, limit)),
             '<div class="pair"><div>', "<h4>Analyses in sandboxes</h4>",
             counts_chart(found, end, limit, height=170, width=600, limit_label=limit_label),
             "</div><div>"]
    total = memory_total(found, end)
    if total:
        parts += ["<h4>Memory in use</h4>",
                  line_chart(total, start=0, end=end, color=MEMORY, unit="MB", label="Memory in use over time",
                             height=170, width=600)]
    else:
        parts.append('<h4>Memory in use</h4><p class="muted">No samples for this batch.</p>')
    parts += ["</div></div>", bars([(None, found)], end), "</div>"]
    return "\n".join(parts)


def analysis_load(node: NodeRun) -> str:
    """One analysis's memory and CPU while its script ran, on the script's own time."""
    if not node.samples:
        return ""
    memory = [(t, m) for t, m, _ in node.samples if m is not None]
    cpu = [(t, u) for t, _, u in node.samples if u is not None]
    cores = [(t1, max(u1 - u0, 0) / (t1 - t0)) for (t0, u0), (t1, u1) in zip(cpu, cpu[1:]) if t1 > t0]
    end = node.script_seconds or max((t for t, *_ in node.samples), default=0)
    parts = ['<div class="pair">']
    if memory:
        limit = node.memory_limit_mb
        # The limit is drawn only when memory comes near it: otherwise it flattens the curve. The heading says it.
        near = limit if limit and max(m for _, m in memory) >= limit / 2 else None
        of = f" (its sandbox has {limit:,} MB)" if limit else ""
        parts.append(f"<div><h4>Memory while its script ran{of}</h4>" + line_chart(
            memory, start=0, end=end, color=MEMORY, unit="MB", label="Memory while its script ran",
            limit=near, limit_label=f"sandbox memory: {limit:,} MB" if near else "", height=150, width=600) + "</div>")
    if cores:
        parts.append("<div><h4>CPU while its script ran</h4>" + line_chart(
            cores, start=0, end=end, color=CPU, unit="cores", label="CPU cores in use while its script ran",
            height=150, width=600) + "</div>")
    parts.append("</div>")
    return "".join(parts)
