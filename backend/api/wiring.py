"""The production ConversationManager and dataset store, built from config.

api/main.py serves them; evals/system/run.py builds its own manager here, so
an eval runs through exactly what the app runs.
"""

import asyncio
from urllib.parse import urlsplit

from agent import build_agent
from config import cfg
from datasets import DatasetStore
from sandboxes import SandboxProvider
from sandboxes.base import make_dirs

from .conversations import ConversationManager
from .datasets import upload_files_hook

store = DatasetStore(
    cfg.server.uploads_dir,
    max_bytes=int(cfg.server.max_upload_mb * (1 << 20)),
    ingest_dsn=lambda: cfg.database.ingest_dsn,
    reader_role=lambda: urlsplit(cfg.database.dsn).username,
)

copy_uploads = upload_files_hook(store, cfg.sandbox.data_dir)


async def ready_for_conversation(sandbox) -> None:
    """Create the data and outputs folders the agent is told to use, then copy uploaded files in."""
    await asyncio.to_thread(make_dirs, sandbox, cfg.sandbox.data_dir, cfg.sandbox.output_dir)
    await copy_uploads(sandbox)


def make_context_filter():
    """agent.context_filter's Jev filter when server.context_filter is on, else None."""
    if not cfg.server.context_filter:
        return None
    settings = cfg.agent.context_filter
    if settings is None:
        raise ValueError("server.context_filter needs agent.context_filter in config.yml")
    from agent.context_filter import ContextFilter

    return ContextFilter(model=settings.model, threshold=settings.threshold,
                         keep_last=settings.keep_last, max_rounds=settings.max_rounds)


def make_manager(provider: SandboxProvider | None, memory: bool = True) -> ConversationManager:
    """Conversations as the app runs them: the real agent, sandbox limits and upgrades from config.

    `memory=False` builds agents without a checkpointer: each turn is sent
    its earlier turns by the context filter instead (create_app's `context`).
    """
    return ConversationManager(
        build_agent=lambda sandbox, checkpointer: build_agent(
            sandbox=sandbox, checkpointer=checkpointer if memory else None
        ),
        provider=provider,
        idle_seconds=cfg.server.sandbox_idle_minutes * 60,
        on_sandbox_ready=ready_for_conversation,
        max_sandboxes=cfg.server.max_sandboxes,
        wait_seconds=cfg.server.sandbox_wait_seconds,
        warm_sandboxes=cfg.server.warm_sandboxes,
        max_per_account=cfg.server.max_sandboxes_per_account,
        max_memory_gb=cfg.server.max_memory_gb,
        bigger_sandbox=(cfg.server.bigger_sandbox.memory_gb, cfg.server.bigger_sandbox.cpu)
        if cfg.server.bigger_sandbox else None,
        work_dir=cfg.sandbox.data_dir.parent,
        retries_first=cfg.server.bigger_sandbox.retries_first if cfg.server.bigger_sandbox else 1,
    )
