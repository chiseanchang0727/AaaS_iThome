"""The API wired to the real agent, sandbox provider, datasets and config.

    uv run --env-file ../.env uvicorn api.main:app --reload     # from backend/
"""

import uuid
from pathlib import Path

from config import cfg
from datasources import close_database
from evals.system.judges import Judges
from sandboxes import get_provider

from .app import create_app
from .compare import compare_router
from .datasets import datasets_router
from .evals import evals_router, system_router
from .history import HistoryStore
from .load import load_router
from .wiring import make_context_filter, make_manager, store

# role and server mark this run's sandboxes, so the next run can delete
# any it leaves behind (see ConversationManager.clean_up_leftovers).
context = make_context_filter()  # None unless server.context_filter is on
manager = make_manager(
    get_provider(cfg.sandbox, labels={"role": "api", "server": uuid.uuid4().hex[:12]}), memory=context is None
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
        system_router(
            Path("evals/system/out/runs"),
            history=HistoryStore(cfg.server.eval_history_dir),
            artifacts_dir=cfg.server.eval_history_dir / "artifacts",
            make_judges=lambda: Judges(cfg.agent.model),
        ),
        compare_router(
            Path("evals/system/out/compare_runs"),
            full=(HistoryStore(cfg.server.eval_history_dir), cfg.server.eval_history_dir / "artifacts"),
            jev=(HistoryStore(cfg.server.eval_jev_history_dir), cfg.server.eval_jev_history_dir / "artifacts"),
            make_judges=lambda: Judges(cfg.agent.model),
        ),
        load_router(history),
    ],
    history=history,
    default_account=cfg.server.default_account,
    context=context,
)
