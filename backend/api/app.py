"""The HTTP API the frontend talks to.

    uv run --env-file ../.env uvicorn api.main:app --reload     # from backend/

    POST   /api/chat                         stream one turn (Server-Sent Events)
    DELETE /api/conversations/{id}           end a conversation, delete its sandbox
    GET    /api/artifacts/{id}/{path}        a file the agent made
    GET    /api/health
    ...plus the upload routes in api/datasets.py
"""

import asyncio
import contextlib
import mimetypes
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from sandboxes import download_outputs

from .conversations import ConversationManager
from .events import run_turn, sse

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

ARTIFACT_HEADERS = {
    # The agent writes these files, so treat them as untrusted: served as their
    # own sandboxed origin, they can run their chart scripts but cannot reach
    # the app, its storage or the API, even when opened directly.
    "Content-Security-Policy": "sandbox allow-scripts",
    "X-Content-Type-Options": "nosniff",
}


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
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async def reap_forever():
            while True:
                await asyncio.sleep(reap_every_seconds)
                await manager.reap_idle()

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

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        thread_id = request.thread_id or manager.new_id()
        existing = manager.get(thread_id)
        if existing is not None and existing.lock.locked():
            raise HTTPException(409, "this conversation is still answering a message")
        return StreamingResponse(
            _turn(thread_id, request.message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _turn(thread_id: str, message: str) -> AsyncIterator[str]:
        yield sse({"type": "thread", "thread_id": thread_id})
        try:
            conversation = await manager.get_or_create(thread_id)
        except Exception as e:
            yield sse({"type": "error", "message": f"could not start a sandbox: {e}"})
            yield sse({"type": "done"})
            return

        async with conversation.lock:
            try:
                async for event in run_turn(conversation.agent, thread_id, message):
                    yield sse(event)
                if conversation.sandbox is not None:
                    for event in await _collect_artifacts(conversation, thread_id):
                        yield sse(event)
            except Exception as e:
                yield sse({"type": "error", "message": str(e)})
            finally:
                manager.touch(conversation)
        yield sse({"type": "done"})

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
    async def end_conversation(thread_id: str):
        if not _ID.match(thread_id):
            raise HTTPException(404)
        return {"closed": await manager.close(thread_id)}

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
