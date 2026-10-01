"""The API wired to the real agent, sandbox provider, datasets and config.

    uv run --env-file ../.env uvicorn api.main:app --reload     # from backend/
"""

from pathlib import Path
from urllib.parse import urlsplit

from agent import build_agent
from config import cfg
from datasets import DatasetStore
from datasources import close_database
from sandboxes import get_provider

from .app import create_app
from .conversations import ConversationManager
from .datasets import datasets_router, upload_files_hook
from .evals import evals_router
from .history import HistoryStore

store = DatasetStore(
    cfg.server.uploads_dir,
    max_bytes=int(cfg.server.max_upload_mb * (1 << 20)),
    ingest_dsn=lambda: cfg.database.ingest_dsn,
    reader_role=lambda: urlsplit(cfg.database.dsn).username,
)

manager = ConversationManager(
    build_agent=lambda sandbox, checkpointer: build_agent(sandbox=sandbox, checkpointer=checkpointer),
    provider=get_provider(cfg.sandbox),
    idle_seconds=cfg.server.sandbox_idle_minutes * 60,
    on_sandbox_ready=upload_files_hook(store, cfg.sandbox.data_dir),
    max_sandboxes=cfg.server.max_sandboxes,
    wait_seconds=cfg.server.sandbox_wait_seconds,
)

app = create_app(
    manager,
    artifacts_dir=cfg.server.artifacts_dir,
    output_dir=cfg.sandbox.output_dir,
    on_shutdown=close_database,
    routers=[
        datasets_router(store, cfg.sandbox.data_dir, manager.live_sandboxes),
        evals_router(Path("evals/memory/out/runs")),
    ],
    history=HistoryStore(cfg.server.history_dir),
)
