"""The API wired to the real agent, sandbox provider, datasets and config.

    uv run --env-file ../.env uvicorn api.main:app --reload     # from backend/
"""

import uuid
from pathlib import Path
import asyncio
from urllib.parse import urlsplit

from agent import build_agent
from config import cfg
from datasets import DatasetStore
from datasources import close_database
from sandboxes import get_provider
from sandboxes.base import make_dirs

from .app import create_app
from .conversations import ConversationManager
from .datasets import datasets_router, upload_files_hook
from .evals import evals_router
from .load import load_router
from .history import HistoryStore

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


manager = ConversationManager(
    build_agent=lambda sandbox, checkpointer: build_agent(sandbox=sandbox, checkpointer=checkpointer),
    # role and server mark this run's sandboxes, so the next run can delete
    # any it leaves behind (see ConversationManager.clean_up_leftovers).
    provider=get_provider(cfg.sandbox, labels={"role": "api", "server": uuid.uuid4().hex[:12]}),
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

history = HistoryStore(cfg.server.history_dir)

app = create_app(
    manager,
    artifacts_dir=cfg.server.artifacts_dir,
    output_dir=cfg.sandbox.output_dir,
    on_shutdown=close_database,
    routers=[
        datasets_router(store, cfg.sandbox.data_dir, manager.live_sandboxes),
        evals_router(Path("evals/memory/out/runs")),
        load_router(history),
    ],
    history=history,
    default_account=cfg.server.default_account,
)
