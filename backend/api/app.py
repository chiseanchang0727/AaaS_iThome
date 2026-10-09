"""The HTTP API the frontend talks to.

    uv run --env-file ../.env uvicorn api.main:app --reload     # from backend/

    POST   /api/chat                         stream one turn (Server-Sent Events)
    GET    /api/conversations                saved conversations of this account, latest first
    GET    /api/conversations/{id}           one saved conversation's history and the files it made
    DELETE /api/conversations/{id}           end a conversation, delete its sandbox (its history stays)
    DELETE /api/conversations/{id}/history   delete a saved conversation for good: history, files, sandbox
    GET    /api/artifacts/{id}/{path}        a file the agent made
    GET    /api/health
    GET    /api/sandboxes                    how many sandboxes exist, and what they are doing
    GET    /api/load                         what the agent's code steps cost (api/load.py)
    ...plus the upload routes in api/datasets.py and the eval routes in api/evals.py
"""

import asyncio
import contextlib
import logging
import mimetypes
import re
import shutil
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from sandboxes import download_outputs

from .conversations import DEFAULT_ACCOUNT, ConversationManager, NotYourConversation
from .events import run_turn, sse
from .history import HistoryStore, to_messages

logger = logging.getLogger(__name__)

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ACCOUNT = re.compile(r"^[a-z0-9_-]{1,64}$")

ARTIFACT_HEADERS = {
    # The agent writes these files, so treat them as untrusted: served as their
    # own sandboxed origin, they can run their chart scripts but cannot reach
    # the app, its storage or the API, even when opened directly.
    "Content-Security-Policy": "sandbox allow-scripts",
    "X-Content-Type-Options": "nosniff",
}


SANDBOX_REPLACED = (
    "The sandbox for this conversation had stopped, so a new one was started. "
    "Files from earlier messages are gone; the conversation itself is kept."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = Field(default=None, pattern=_ID.pattern)


def create_app(
    manager: ConversationManager,
    artifacts_dir: Path,
    output_dir: PurePosixPath,
    reap_every_seconds: float = 60,
    on_shutdown=None,
    routers: Sequence[APIRouter] = (),
    history: HistoryStore | None = None,
    default_account: str = DEFAULT_ACCOUNT,
    context: Any = None,
) -> FastAPI:
    """The app. With `context` (an agent.context_filter.ContextFilter) and
    `history`, each turn is sent only the earlier turns Jev picks from the
    history, and the manager's agents must be built without a checkpointer.
    Without it, the agent's checkpointer carries the whole conversation."""
    def account(x_account: str | None = Header(default=None)) -> str:
        """Who the request is for: the X-Account header (no login yet), else the default."""
        if x_account is None:
            return default_account
        if not _ACCOUNT.match(x_account):
            raise HTTPException(400, "X-Account must be 1-64 lowercase letters, digits, _ or -")
        return x_account

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async def reap_forever():
            while True:
                await asyncio.sleep(reap_every_seconds)
                await manager.reap_idle()

        manager.start()  # begin filling the warm sandbox pool
        reaper = asyncio.create_task(reap_forever())
        try:
            yield
        finally:
            reaper.cancel()
            await manager.close_all()
            if on_shutdown is not None:
                await on_shutdown()

    app = FastAPI(title="AaaS iThome", lifespan=lifespan)
    for router in routers:
        app.include_router(router)

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/sandboxes")
    async def sandboxes(account: str = Depends(account)):
        """How many sandboxes exist and what each is doing (see ConversationManager.status)."""
        return {**manager.status(), "account": account}

    @app.post("/api/chat")
    async def chat(request: ChatRequest, account: str = Depends(account)):
        thread_id = request.thread_id or manager.new_id()
        existing = manager.get(thread_id)
        if existing is not None and existing.account != account:
            raise HTTPException(404, "no such conversation")
        if existing is not None and existing.lock.locked():
            raise HTTPException(409, "this conversation is still answering a message")
        return StreamingResponse(
            _turn(thread_id, request.message, account),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _turn(thread_id: str, message: str, account: str) -> AsyncIterator[str]:
        yield sse({"type": "thread", "thread_id": thread_id})
        try:
            conversation = await manager.get_or_create(thread_id, account)
        except NotYourConversation:
            yield sse({"type": "error", "message": "no such conversation"})
            yield sse({"type": "done"})
            return
        except Exception as e:
            yield sse({"type": "error", "message": f"could not start the conversation: {e}"})
            yield sse({"type": "done"})
            return

        async with conversation.lock:
            try:
                earlier = history.read(thread_id) if context is not None and history is not None else None
                log = history.start_turn(thread_id, message, account) if history is not None else None
                start_with = await _pick_context(earlier, message, log) if earlier is not None else None
                on_message = log.record if log is not None else None
                # This turn's log hears sandbox events from here on, including a
                # replacement below; set the step hook after it, on the stand-in
                # the agent will actually use.
                conversation.on_event = log.record_event if log is not None else None
                if await manager.ensure_sandbox(conversation):
                    yield sse({"type": "notice", "message": SANDBOX_REPLACED})
                if conversation.stand_in is not None:
                    conversation.stand_in.on_step = log.record_step if log is not None else None
                async for event in run_turn(conversation.agent, thread_id, message, on_message, start_with):
                    yield sse(event)
                if conversation.sandbox is not None:
                    for event in await _collect_artifacts(conversation, thread_id):
                        yield sse(event)
                manager.touch(conversation, sandbox_ok=True)
            except Exception as e:
                yield sse({"type": "error", "message": str(e)})
            finally:
                manager.touch(conversation)
        yield sse({"type": "done"})

    async def _pick_context(earlier: list[dict], message: str, log) -> list:
        """The earlier turns Jev picks, then the message; recorded as a `context` line.

        If Jev fails, every earlier turn is sent, as with a checkpointer.
        """
        started = time.monotonic()
        turns = len({r.get("turn") for r in earlier if r.get("role") == "user"})
        try:
            start_with, picked = await context.build(earlier, message)
            fields: dict[str, Any] = {"sent_turns": picked}
        except Exception as e:
            logger.warning("context filter failed; sending the whole conversation", exc_info=True)
            start_with = [*to_messages(earlier), HumanMessage(message)]
            fields = {"sent_turns": sorted({r["turn"] for r in earlier if r.get("role") == "user"}), "error": str(e)}
        if log is not None:
            log.write({"role": "context", **fields, "earlier_turns": turns,
                       "seconds": round(time.monotonic() - started, 3)})
        return start_with

    async def _collect_artifacts(conversation, thread_id: str) -> list[dict]:
        local = artifacts_dir / thread_id
        written = await asyncio.to_thread(download_outputs, conversation.sandbox, output_dir, local)
        events = []
        for path in sorted(written):
            relative = path.relative_to(local).as_posix()
            if relative in conversation.announced:
                continue
            conversation.announced.add(relative)
            events.append({
                "type": "artifact",
                "name": relative,
                "url": f"/api/artifacts/{thread_id}/{relative}",
                "kind": path.suffix.lstrip(".").lower(),
            })
        return events

    @app.delete("/api/conversations/{thread_id}")
    async def end_conversation(thread_id: str, account: str = Depends(account)):
        if not _ID.match(thread_id):
            raise HTTPException(404)
        conversation = manager.get(thread_id)
        if conversation is not None and conversation.account != account:
            raise HTTPException(404)
        return {"closed": await manager.close(thread_id)}

    def owner(records: list[dict]) -> str:
        """Whose a saved conversation is: the account on its first message (older lines have none)."""
        return next((r["account"] for r in records if r.get("role") == "user" and r.get("account")), default_account)

    def files_made(thread_id: str) -> list[dict]:
        folder = artifacts_dir / thread_id
        if not folder.is_dir():
            return []
        return [
            {"name": p.relative_to(folder).as_posix(), "url": f"/api/artifacts/{thread_id}/{p.relative_to(folder).as_posix()}",
             "kind": p.suffix.lstrip(".").lower()}
            for p in sorted(folder.rglob("*")) if p.is_file()
        ]

    @app.get("/api/conversations")
    async def past_conversations(account: str = Depends(account)):
        """This account's saved conversations, latest first: what each began with, how long it went."""
        if history is None:
            return []
        found = []
        for thread_id in history.thread_ids():
            records = history.read(thread_id)
            questions = [r for r in records if r.get("role") == "user"]
            if not questions or owner(records) != account:
                continue
            found.append({
                "id": thread_id,
                "title": questions[0]["content"][:160],
                "turns": len(questions),
                "started_at": records[0].get("ts"),
                "last_at": records[-1].get("ts"),
                "files": len(files_made(thread_id)),
            })
        return sorted(found, key=lambda c: c["last_at"] or "", reverse=True)

    @app.get("/api/conversations/{thread_id}")
    async def past_conversation(thread_id: str, account: str = Depends(account)):
        """One saved conversation: every history line, and the files it made."""
        records = history.read(thread_id) if history is not None and _ID.match(thread_id) else []
        if not records or owner(records) != account:
            raise HTTPException(404, "no such conversation")
        return {"id": thread_id, "lines": records, "files": files_made(thread_id)}

    @app.delete("/api/conversations/{thread_id}/history")
    async def delete_past_conversation(thread_id: str, account: str = Depends(account)):
        """Delete a saved conversation for good: its history, the files it made, and its sandbox if any."""
        records = history.read(thread_id) if history is not None and _ID.match(thread_id) else []
        if not records or owner(records) != account:
            raise HTTPException(404, "no such conversation")
        live = manager.get(thread_id)
        if live is not None and live.lock.locked():
            raise HTTPException(409, "this conversation is still answering a message")
        await manager.close(thread_id)
        history.delete(thread_id)
        files = artifacts_dir / thread_id
        if files.is_dir():
            shutil.rmtree(files)
        return {"deleted": thread_id}

    @app.get("/api/artifacts/{thread_id}/{path:path}")
    async def artifact(thread_id: str, path: str):
        if not _ID.match(thread_id):
            raise HTTPException(404)
        root = (artifacts_dir / thread_id).resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404)
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type, headers=ARTIFACT_HEADERS)

    return app
