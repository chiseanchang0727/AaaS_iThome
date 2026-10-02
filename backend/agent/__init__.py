"""The deep agent, wired up from config.

    from agent import build_agent

    agent = build_agent()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
"""

import logging
import os
import re

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from langgraph.checkpoint.base import BaseCheckpointSaver

from config import cfg
from datasets import Registry

from .middleware import SkillEnforcerMiddleware
from .skill_router import SkillRouterMiddleware, build_classifier
from .tools import make_export_query, make_list_datasets, make_request_bigger_sandbox, query_database

__all__ = ["build_agent", "SYSTEM_PROMPT", "SANDBOX_PROMPT"]

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a data analyst. The data is the built-in `videos` table (US YouTube
trending videos) in PostgreSQL, plus datasets users upload as tables or files.
Call list_datasets to see what exists, with columns and types, before
assuming a table or column. Answer questions by querying the data. Keep the
final answer short: the result, then one line on how it was computed.

Every claim you make about the data must rest on a result you have seen: the
numbers, and also descriptions such as "every day", "most videos" or "none".
If you have not seen a result that supports a statement, query for it or
leave the statement out.

When a question asks for the top or highest item, check whether others tie
with it before answering, and name every tied item.
"""

SANDBOX_PROMPT = """
You also have a sandbox for analysis that SQL does poorly (statistics, reshaping)
and for charts and report files. Get data into it with export_query, which
saves a query's rows as a Parquet file in {data_dir}; then work on it in
Python. execute runs shell commands, not Python: write your code to a .py file
with write_file, then run it with execute (`python3 /path/to/script.py`). Use
polars, not pandas.
Save every chart or report you make to {output_dir}: files there are handed to
the user after the run.

The sandbox has little memory (about 1 GB), so keep the data you load small:
- Do the heavy work in SQL: filter, join and aggregate in the query, and export
  only the rows and columns the analysis needs, not whole tables.
- For a file you have not checked, look at its size before loading it:
  `pl.scan_parquet(path).select(pl.len()).collect()` and `pl.read_parquet_schema(path)`.
- Compute on large files with DuckDB, which reads only the columns it needs and
  works through files bigger than memory, including medians, percentiles and
  distinct counts that polars cannot stream:
  `import duckdb; con = duckdb.connect(); con.execute("SET memory_limit='600MB'")`, then
  `con.sql("SELECT device, median(x) FROM '/path/file.parquet' GROUP BY device").pl()`.
- Use polars for small data and for shaping results for charts. Load a large
  file with polars only lazily and only the columns you need:
  `pl.scan_parquet(path).select(...).collect(engine="streaming")`.
- If a command is killed for running out of memory, do not run the same code
  again: try a lighter approach first (fewer columns, aggregate earlier,
  DuckDB, chunks, sampling). Only if that also runs out of memory and the work
  truly needs that much memory at once, call request_bigger_sandbox with the
  reason, then run the command again.

Make charts with plotly (plotly.express accepts polars DataFrames) and save
each as interactive HTML, which the user sees rendered in the chat:
`fig.write_html("{output_dir}/<name>.html", include_plotlyjs="cdn")`.

export_query does not show you the rows, and you cannot see charts. When you
compute something in the sandbox, print the numbers and facts you will report.
"""


_VIDEOS = re.compile(r"\bvideos\b", re.IGNORECASE)


def touches_videos(tool_call: dict) -> bool:
    """Whether a SQL tool call names the `videos` table.

    A plain word match: it also fires on "videos" in a string literal or
    comment, which only means the skill gets read when it wasn't needed.
    """
    return bool(_VIDEOS.search(str(tool_call.get("args", {}).get("sql", ""))))


def skill_router() -> SkillRouterMiddleware | None:
    """The router from `cfg.agent.skill_router`, or None when it is off or has no key."""
    settings = cfg.agent.skill_router
    if settings is None:
        return None
    if not os.environ.get("TYPESAFE_API_KEY"):
        log.warning("agent.skill_router is set but $TYPESAFE_API_KEY is not; routing is off")
        return None
    classifier = build_classifier(cfg.agent.skills_dir, model=settings.model)
    return SkillRouterMiddleware(classifier, top_n=settings.top_n, multi=settings.multi)


def build_agent(
    *,
    require_sql_skill: bool = True,
    sandbox: SandboxBackendProtocol | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """A deep agent with the database tool and the skills in `cfg.agent.skills_dir`.

    Without `sandbox`, files the agent writes go to in-memory state and vanish
    with the run, and `execute` is unavailable. With one, they live in the
    sandbox, `execute` runs there, and export_query can put query results into
    it. Either way /skills/ is the real directory and is read-only to the agent.

    With `checkpointer`, each run with the same `thread_id` in its config
    continues the same conversation.
    """
    backend = CompositeBackend(
        default=sandbox or StateBackend(),
        routes={
            "/skills/": FilesystemBackend(root_dir=cfg.agent.skills_dir, virtual_mode=True),
        },
    )
    registry = Registry(cfg.server.uploads_dir / "registry.json")
    data_dir = cfg.sandbox.data_dir if sandbox is not None else None
    tools = [query_database, make_list_datasets(registry, data_dir)]
    system_prompt = SYSTEM_PROMPT
    if sandbox is not None:
        tools.append(
            make_export_query(backend, cfg.sandbox.data_dir, cfg.sandbox.export_max_rows)
        )
        system_prompt += SANDBOX_PROMPT.format(
            data_dir=cfg.sandbox.data_dir, output_dir=cfg.sandbox.output_dir
        )
        if hasattr(sandbox, "request_bigger"):  # the API's stand-in, which can move sandboxes
            tools.append(make_request_bigger_sandbox(sandbox))

    middleware = []
    if router := skill_router():
        middleware.append(router)
    # Appended last: after_model hooks run in reverse order, so this checks
    # the model's calls before any middleware added ahead of it.
    if require_sql_skill:
        middleware.append(
            SkillEnforcerMiddleware(
                cfg.agent.skills_dir,
                target_skills="query_database",
                shared_skills={"export_query": "query_database"},
                applies_to=touches_videos,
            )
        )

    return create_deep_agent(
        model=cfg.agent.model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        skills=["/skills/"],
        permissions=[FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")],
        middleware=middleware,
        checkpointer=checkpointer,
    )
