"""Upload routes: stage a file, preview it, keep it as a table or a file.

    POST   /api/uploads            multipart `file` -> parsed preview + upload_id
    POST   /api/datasets           {upload_id, kind, name} -> the new dataset
    GET    /api/datasets           every dataset (the built-in videos table first)
    DELETE /api/datasets/{name}    drop the table or delete the file
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Literal

from deepagents.backends.protocol import SandboxBackendProtocol
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from datasets import DatasetStore, UploadError

BUILT_IN = {
    "name": "videos", "kind": "table", "built_in": True,
    "description": "US YouTube trending videos, one row per video per trending day",
}


class CommitRequest(BaseModel):
    upload_id: str
    kind: Literal["table", "file"]
    name: str


def datasets_router(
    store: DatasetStore,
    data_dir: PurePosixPath,
    live_sandboxes: Callable[[], list[SandboxBackendProtocol]],
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/uploads")
    async def upload(file: UploadFile):
        try:
            staged = await asyncio.to_thread(store.stage, file.filename or "upload", file.file)
        except UploadError as e:
            status = 413 if "larger than" in str(e) else 400
            raise HTTPException(status, str(e)) from None
        return asdict(staged)

    @router.post("/datasets", status_code=201)
    async def commit(request: CommitRequest):
        try:
            dataset = await store.commit(request.upload_id, request.kind, request.name)
        except UploadError as e:
            raise HTTPException(400, str(e)) from None
        if dataset.kind == "file":
            # Conversations already running get the file too, not just new ones.
            files = store.sandbox_files(data_dir, [dataset.name])
            for sandbox in live_sandboxes():
                await asyncio.to_thread(sandbox.upload_files, files)
        return dataset

    @router.get("/datasets")
    async def list_datasets():
        return [BUILT_IN, *(d.model_dump(mode="json") | {"built_in": False} for d in store.list())]

    @router.delete("/datasets/{name}")
    async def delete(name: str):
        if not await store.delete(name):
            raise HTTPException(404, f"no dataset named {name!r}")
        return {"deleted": name}

    return router


def upload_files_hook(store: DatasetStore, data_dir: PurePosixPath):
    """A ConversationManager on_sandbox_ready hook: copy file datasets in."""

    async def hook(sandbox: SandboxBackendProtocol) -> None:
        files = store.sandbox_files(data_dir)
        if files:
            await asyncio.to_thread(sandbox.upload_files, files)

    return hook
