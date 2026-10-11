"""Does a saved analysis's strategy still fit its data? Limits, and where the bottleneck is.

Schema compatibility (sources.py) says a recipe *can* run on a table. This
says whether it *should*, as written: the same `SELECT *` that exports 25K
rows from one table would move 10M from a bigger one with the same columns.

Measure first, then judge:
- before anything moves, the planner's estimate of each query input
  (check_estimates): over a hard limit, the run stops there;
- after a run, what actually happened (check_run).

Each finding names the kind of problem, so the fix fits it:
    data_movement    too many rows / bytes into the sandbox -> rewrite the SQL
    sandbox_memory   killed, or peak memory near the limit, with small inputs -> rewrite the script
    sandbox_runtime  the script is slow on small inputs -> rewrite the script
A sandbox problem caused by a big export is data movement: a bigger
sandbox would only hide it.
"""

from dataclasses import dataclass

from .models import Finding, Measurements

MB = 1_000_000


@dataclass(frozen=True)
class Limits:
    hard_export_rows: int
    max_export_bytes: int
    soft_export_rows: int
    soft_export_bytes: int
    soft_run_seconds: float
    memory_warning_ratio: float
    hard_query_seconds: float | None = None
    """The database's timeout: a query stops there. Shown with the limits; not judged here.
    There is no query-time target: a slow query with a small export is the database working
    (an index or a pre-computed table helps), which a rewrite of the recipe cannot fix."""

    @classmethod
    def from_config(cls, cfg) -> "Limits":
        e = cfg.analysis_execution
        return cls(
            hard_export_rows=cfg.sandbox.export_max_rows, max_export_bytes=int(e.max_export_mb * MB),
            soft_export_rows=e.soft_export_rows, soft_export_bytes=int(e.soft_export_mb * MB),
            soft_run_seconds=e.soft_run_seconds,
            memory_warning_ratio=e.memory_warning_ratio, hard_query_seconds=cfg.database.timeout,
        )


def rows(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1_000_000 else f"{n:,}"


def size(n: int) -> str:
    return f"{n / MB:,.1f} MB" if n >= MB else f"{n / 1000:,.0f} KB"


def over_estimate(i, limits: Limits) -> bool:
    """Whether the estimate alone is over a hard limit (then the rows are counted to be sure)."""
    return ((i.estimated_rows or 0) > limits.hard_export_rows) or ((i.estimated_bytes or 0) > limits.max_export_bytes)


def check_estimates(m: Measurements, limits: Limits) -> list[Finding]:
    """Hard limits the query would break before it runs: the run should not start moving data.

    The real count, when there is one, wins over the planner's estimate: the estimate only
    decides whether to count.
    """
    found = []
    for i in m.inputs:
        where = f"{i.file} (from {', '.join(i.tables) or 'its query'})"
        width = (i.estimated_bytes / i.estimated_rows) if i.estimated_rows and i.estimated_bytes else None
        if i.counted_rows is not None:
            n, b, how = i.counted_rows, (i.counted_rows * width if width else None), "counted in the database"
        else:
            n, b, how = i.estimated_rows, i.estimated_bytes, "estimated"
        if n is not None and n > limits.hard_export_rows:
            found.append(Finding(limit="hard", kind="data_movement", reason=(
                f"{where} would export {'' if i.counted_rows is not None else 'about '}{rows(n)} rows into the sandbox "
                f"({how}); the limit is {rows(limits.hard_export_rows)}")))
        elif b is not None and b > limits.max_export_bytes:
            found.append(Finding(limit="hard", kind="data_movement", reason=(
                f"{where} would export about {size(int(b))} into the sandbox ({how}); "
                f"the limit is {size(limits.max_export_bytes)}")))
    return found


def check_run(m: Measurements, limits: Limits) -> list[Finding]:
    """Limits a finished (or killed) run exceeded, hard first."""
    found: list[Finding] = []
    big_export = False
    for i in m.inputs:
        if i.rows is not None and i.rows > limits.soft_export_rows:
            big_export = True
            found.append(Finding(limit="soft", kind="data_movement", reason=(
                f"{i.file} exported {rows(i.rows)} rows; the target is under {rows(limits.soft_export_rows)}")))
        elif i.bytes is not None and i.bytes > limits.soft_export_bytes:
            big_export = True
            found.append(Finding(limit="soft", kind="data_movement", reason=(
                f"{i.file} exported {size(i.bytes)}; the target is under {size(limits.soft_export_bytes)}")))

    s = m.sandbox
    if s is not None:
        if s.killed:
            kind = "data_movement" if big_export else "sandbox_memory"
            why = " after a large export" if big_export else ""
            found.insert(0, Finding(limit="hard", kind=kind, reason=f"the script ran out of memory{why}"))
        elif s.peak_memory_mb and s.memory_limit_mb and s.peak_memory_mb >= limits.memory_warning_ratio * s.memory_limit_mb:
            found.append(Finding(limit="soft", kind="data_movement" if big_export else "sandbox_memory", reason=(
                f"the script peaked at {s.peak_memory_mb:,.0f} of {s.memory_limit_mb:,} MB")))
        if s.seconds is not None and s.seconds > limits.soft_run_seconds and not big_export:
            found.append(Finding(limit="soft", kind="sandbox_runtime", reason=(
                f"the script took {s.seconds:.0f}s; the target is under {limits.soft_run_seconds:.0f}s")))
    return found


def needs_optimization(findings: list[Finding]) -> bool:
    return any(f.limit == "hard" for f in findings)


def main_finding(findings: list[Finding]) -> Finding | None:
    """The one to act on: the first hard finding, else the first soft one."""
    return next((f for f in findings if f.limit == "hard"), findings[0] if findings else None)
