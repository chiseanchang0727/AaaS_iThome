"""A stand-in for a sandbox that is still starting.

The agent is built on this, so it can start answering at once: SQL questions
never touch the sandbox. The first call that does (running code, reading or
writing a file in it) waits until the real sandbox is ready, then everything
is passed through.

If the sandbox never arrives (every slot stayed busy, the provider failed, the
conversation ended), each call returns that operation's error result, the way
a sandbox reports a failed command or a missing file. The agent sees the
reason as a tool result and can still answer from what it already has.

Thread-safe: the agent calls sandbox methods from worker threads.
"""

import asyncio
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout

from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
    _method_accepts_max_count,
    execute_accepts_timeout,
)

START_TIMEOUT = 180.0
"""Seconds a call waits for the sandbox: a slot (up to 30s), then the start (~12s)."""


class SandboxUnavailable(RuntimeError):
    """The sandbox for this conversation could not be started."""


class LazySandbox(SandboxBackendProtocol):
    """Waits for the real sandbox on first use. Settled once, by `ready` or `failed`."""

    def __init__(self, start_timeout: float = START_TIMEOUT) -> None:
        self._future: Future[SandboxBackendProtocol] = Future()
        self._start_timeout = start_timeout

    def ready(self, sandbox: SandboxBackendProtocol) -> None:
        self._future.set_result(sandbox)

    def failed(self, reason: BaseException) -> None:
        self._future.set_exception(reason)

    @property
    def settled(self) -> bool:
        return self._future.done()

    def _real(self) -> SandboxBackendProtocol:
        """The real sandbox, waiting for it if needed; SandboxUnavailable if it won't come."""
        try:
            return self._future.result(timeout=self._start_timeout)
        except FutureTimeout:
            raise SandboxUnavailable("the sandbox is taking too long to start") from None
        except Exception as e:
            raise SandboxUnavailable(f"no sandbox: {e}") from e

    @staticmethod
    def _why(e: SandboxUnavailable) -> str:
        return f"Sandbox unavailable ({e}). Code and files can't be used for this; answer from the data you already have."

    @property
    def id(self) -> str:
        if self._future.done() and self._future.exception() is None:
            return self._future.result().id
        return "starting"

    # --- passed through once the sandbox is ready ---------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            sandbox = self._real()
        except SandboxUnavailable as e:
            return ExecuteResponse(output=self._why(e), exit_code=1)
        if timeout is not None and execute_accepts_timeout(type(sandbox)):
            return sandbox.execute(command, timeout=timeout)
        return sandbox.execute(command)

    def ls(self, path: str) -> LsResult:
        try:
            return self._real().ls(path)
        except SandboxUnavailable as e:
            return LsResult(error=self._why(e))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            return self._real().read(file_path, offset, limit)
        except SandboxUnavailable as e:
            return ReadResult(error=self._why(e))

    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None, *, max_count: int | None = None
    ) -> GrepResult:
        try:
            sandbox = self._real()
        except SandboxUnavailable as e:
            return GrepResult(error=self._why(e))
        if max_count is not None and _method_accepts_max_count(type(sandbox), "grep"):
            return sandbox.grep(pattern, path, glob, max_count=max_count)
        return sandbox.grep(pattern, path, glob)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        try:
            return self._real().glob(pattern, path)
        except SandboxUnavailable as e:
            return GlobResult(error=self._why(e))

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            return self._real().write(file_path, content)
        except SandboxUnavailable as e:
            return WriteResult(error=self._why(e), path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        try:
            return self._real().edit(file_path, old_string, new_string, replace_all)
        except SandboxUnavailable as e:
            return EditResult(error=self._why(e), path=file_path)

    def delete(self, file_path: str) -> DeleteResult:
        try:
            return self._real().delete(file_path)
        except SandboxUnavailable as e:
            return DeleteResult(error=self._why(e), path=file_path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        try:
            return self._real().upload_files(files)
        except SandboxUnavailable as e:
            return [FileUploadResponse(path=path, error=self._why(e)) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        try:
            return self._real().download_files(paths)
        except SandboxUnavailable as e:
            return [FileDownloadResponse(path=path, error=self._why(e)) for path in paths]

    # Async versions run the sync ones in a thread (the protocol's default), so
    # waiting for the sandbox never blocks the event loop.
    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        return await asyncio.to_thread(self.execute, command, timeout=timeout)
