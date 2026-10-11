"""A workflow: saved analyses run in parallel, each feeding a batch node that collects them.

    analysis node --edge--> batch node --edge--> report node
    analysis node --edge--/

An analysis node is one saved analysis with its own table swaps, so the same
analysis can appear twice, on different tables. A batch node waits for the
analyses connected to it and collects their status and outputs into one
report. A workflow may have several batch nodes; an analysis node connected
to two of them still runs once. An analysis node connected to none does not run.
A report node turns the batches connected to it into one page, a section each.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from analyses.models import now


class Position(BaseModel):
    """Where the node sits on the canvas."""

    x: float = 0
    y: float = 0


class AnalysisNode(BaseModel):
    kind: Literal["analysis"] = "analysis"
    id: str
    analysis_id: str
    sources: dict[str, str] = {}
    """Tables to read instead of the analysis's own, e.g. {"events": "events_enterprise"}."""
    position: Position = Position()


class BatchNode(BaseModel):
    kind: Literal["batch"] = "batch"
    id: str
    title: str = "Batch"
    position: Position = Position()


class ReportNode(BaseModel):
    kind: Literal["report"] = "report"
    id: str
    title: str = "Report"
    position: Position = Position()


Node = Annotated[AnalysisNode | BatchNode | ReportNode, Field(discriminator="kind")]

EDGES = {("analysis", "batch"), ("batch", "report")}
"""The connections a workflow allows: (source kind, target kind)."""


class Edge(BaseModel):
    source: str
    """An analysis node's id, or a batch node's."""
    target: str
    """A batch node's id, or a report node's."""


class Workflow(BaseModel):
    id: str
    title: str
    nodes: list[Node] = []
    edges: list[Edge] = []
    created_at: str = Field(default_factory=now)
    updated_at: str | None = None

    def node(self, node_id: str) -> AnalysisNode | BatchNode | ReportNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def problems(self) -> list[str]:
        """What is wrong with its shape; empty when it can be saved."""
        problems = []
        if not self.title.strip():
            problems.append("give it a title")
        ids = [n.id for n in self.nodes]
        problems += [f"two nodes have the id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
        seen = set()
        for edge in self.edges:
            source, target = self.node(edge.source), self.node(edge.target)
            if source is None or target is None:
                problems.append(f"an edge {edge.source} -> {edge.target} names a node that is not there")
            elif (source.kind, target.kind) not in EDGES:
                problems.append(f"an edge must go from an analysis to a batch, or from a batch to a report, "
                                f"not {source.kind} -> {target.kind}")
            elif (edge.source, edge.target) in seen:
                problems.append(f"{edge.source} -> {edge.target} is connected twice")
            seen.add((edge.source, edge.target))
        return problems

    def batches(self) -> dict[str, list[str]]:
        """Batch node id -> the analysis node ids connected to it, in node order."""
        order = [n.id for n in self.nodes]
        return {
            b.id: sorted({e.source for e in self.edges if e.target == b.id}, key=order.index)
            for b in self.nodes if isinstance(b, BatchNode)
        }

    def reports(self) -> dict[str, list[str]]:
        """Report node id -> the batch node ids connected to it, in node order."""
        order = [n.id for n in self.nodes]
        return {
            r.id: sorted({e.source for e in self.edges if e.target == r.id}, key=order.index)
            for r in self.nodes if isinstance(r, ReportNode)
        }

    def connected(self) -> list[AnalysisNode]:
        """The analysis nodes a run starts: those connected to a batch."""
        batches = {n.id for n in self.nodes if isinstance(n, BatchNode)}
        used = {e.source for e in self.edges if e.target in batches}
        return [n for n in self.nodes if isinstance(n, AnalysisNode) and n.id in used]


class NodeRun(BaseModel):
    """One analysis node in a workflow run: a run of that analysis."""

    node_id: str
    analysis_id: str
    title: str
    sources: dict[str, str] = {}
    version: int | None = None
    run_id: str | None = None
    """The analysis run (under saved_analyses/<analysis>/runs/); None while it waits for a place."""
    status: Literal["waiting", "running", "done", "failed", "needs_optimization"] = "running"
    """waiting: in the workflow, for one of the account's sandboxes to be free."""
    error: str | None = None
    seconds: float | None = None
    outputs: list[str] = []
    queued_at: str | None = None
    sandbox_at: str | None = None
    finished_at: str | None = None
    """When it was asked for, when its sandbox was ready, and when it ended (ms; from its analysis run)."""
    rows: int | None = None
    bytes: int | None = None
    """Moved from the database into its sandbox, all inputs together."""
    cpu_seconds: float | None = None
    peak_memory_mb: float | None = None
    memory_limit_mb: int | None = None
    script_at: str | None = None
    """When its script started (ms), after its inputs were copied into the sandbox."""
    script_seconds: float | None = None
    samples: list[list[float | None]] = []
    """While the script ran: [seconds since it started, MB resident, CPU seconds so far]."""


class WorkflowRun(BaseModel):
    id: str
    workflow_id: str
    started_at: str
    finished_at: str | None = None
    seconds: float | None = None
    status: Literal["running", "done", "partly_done", "failed"] = "running"
    """done: every node done; failed: none done; partly_done: some of each."""
    nodes: list[NodeRun] = []
    batches: dict[str, list[str]] = {}
    """Batch node id -> its analysis node ids, as the workflow was when it ran."""
    reports: dict[str, list[str]] = {}
    """Report node id -> its batch node ids, as the workflow was when it ran."""
    titles: dict[str, str] = {}
    """Batch and report node id -> its title, as the workflow was when it ran."""
    workflow_title: str = ""
    sandbox_limit: int | None = None
    """How many sandboxes one account could hold at once when it ran: the line its load is measured against."""

    def settle(self) -> None:
        """The overall status, from the nodes'."""
        if any(n.status in ("running", "waiting") for n in self.nodes):
            self.status = "running"
        elif all(n.status == "done" for n in self.nodes):
            self.status = "done"
        elif any(n.status == "done" for n in self.nodes):
            self.status = "partly_done"
        else:
            self.status = "failed"
