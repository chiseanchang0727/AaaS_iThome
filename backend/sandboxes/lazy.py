"""A stand-in for a sandbox that is still starting.

The agent is built on this, so it can start answering at once: SQL questions
never touch the sandbox. The first call that does (running code, reading or
writing a file in it) waits until the real sandbox is ready, then everything
is passed through.

If the sandbox never arrives (every slot stayed busy, the provider failed, the
conversation ended), each call returns that operation's error result, the way
a sandbox reports a failed command or a missing file. The agent sees the
reason as a tool result and can still answer from what it already has.

A command killed for running out of memory goes back to the agent: rewriting
is cheaper than a bigger machine, and only the agent knows whether the work
can be split. With `retries_first` = 1 (the default):

    1st kill            try a lighter approach first; no bigger sandbox yet
    same code again     doesn't count: change the approach first
    a changed attempt
      also killed       now the agent may call request_bigger_sandbox(reason)
    one more kill       safety net: moved and run again without asking

An attempt is a command plus the files as they were: running a different
command, or the same one after write_file / edit_file, is a new attempt.
`request_bigger` (the tool) gets the bigger sandbox through
`on_out_of_memory`: the server starts it and copies the work over, and later
commands run there. `retries_first` = 0 allows asking right after a kill.

Thread-safe: the agent calls sandbox methods from worker threads.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
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
)

from .metering import StepMeasure, run_measured


@dataclass(frozen=True)
class Upgrade:
    """The answer to "this ran out of memory": a bigger sandbox, or why not."""

    sandbox: SandboxBackendProtocol | None
    note: str
    """What the agent is told, e.g. "moved to a 4 GB sandbox"."""

log = logging.getLogger(__name__)

LIGHTER = "select fewer columns, aggregate earlier, query files with DuckDB, process in chunks or sample"
TRY_LIGHTER = (
    f"Out of memory in this sandbox. Try a lighter approach first: {LIGHTER}. "
    "A bigger sandbox becomes available only if a changed attempt still runs out of memory."
)
SAME_AGAIN = (
    "This is the same code that already ran out of memory. Change the approach before running it again: "
    f"{LIGHTER}."
)
MAY_ASK = (
    "Still out of memory. If the work truly needs this much memory at once, call "
    "request_bigger_sandbox with the reason, then run the command again. Otherwise keep reducing memory: "
    f"{LIGHTER}."
)
ASK_FIRST = (
    f"Out of memory in this sandbox. If the work can be done with less memory, change the approach: {LIGHTER}. "
    "If it truly needs this much memory at once, call request_bigger_sandbox with the reason, "
    "then run the command again."
)
NOT_YET = (
    f"Not yet: try a lighter approach first ({LIGHTER}). A bigger sandbox is available after a changed "
    "attempt also runs out of memory."
)

START_TIMEOUT = 180.0
"""Seconds a call waits for the sandbox: a slot (up to 30s), then the start (~12s)."""


class SandboxUnavailable(RuntimeError):
    """The sandbox for this conversation could not be started."""


class LazySandbox(SandboxBackendProtocol):
    """Waits for the real sandbox on first use. Settled once, by `ready` or `failed`."""

    def __init__(self, start_timeout: float = START_TIMEOUT, retries_first: int = 1) -> None:
        self._future: Future[SandboxBackendProtocol] = Future()
        self._start_timeout = start_timeout
        self.on_step: Callable[[StepMeasure], None] | None = None
        """Called with what each command cost (sandboxes/metering.py), from
        the worker thread that ran it. Set per turn by the API."""
        self.on_out_of_memory: Callable[[str], Upgrade] | None = None
        """Asks for a bigger sandbox, with the reason; called from a worker
        thread. Set by the conversation manager once the sandbox is ready."""
        self._replaced: SandboxBackendProtocol | None = None
        self.retries_first = retries_first
        """Lighter attempts that must also run out of memory before a bigger sandbox."""
        self._changes = 0
        """Files written or edited so far: part of what makes an attempt new."""
        self._killed_attempts: set[tuple[str, int]] = set()
        self._kills = 0

    def ready(self, sandbox: SandboxBackendProtocol) -> None:
        self._future.set_result(sandbox)

    def failed(self, reason: BaseException) -> None:
        self._future.set_exception(reason)

    @property
    def settled(self) -> bool:
        return self._future.done()

    def _real(self) -> SandboxBackendProtocol:
        """The real sandbox, waiting for it if needed; SandboxUnavailable if it won't come."""
        if self._replaced is not None:
            return self._replaced
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
        if self._replaced is not None:
            return self._replaced.id
        if self._future.done() and self._future.exception() is None:
            return self._future.result().id
        return "starting"

    # --- passed through once the sandbox is ready ---------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            sandbox = self._real()
        except SandboxUnavailable as e:
            return ExecuteResponse(output=self._why(e), exit_code=1)
        result, measure = run_measured(sandbox, command, timeout)
        self._record(measure)
        if not measure.out_of_memory or self.on_out_of_memory is None:
            return result
        attempt = (command, self._changes)
        repeated = attempt in self._killed_attempts
        self._killed_attempts.add(attempt)
        self._kills += 1
        if self._kills < self.retries_first + 2:
            return ExecuteResponse(output=f"{result.output}\n[{self._advice(repeated)}]", exit_code=result.exit_code)
        # Safety net: still killed after the agent could have asked.
        upgrade = self._ask(f"ran out of memory {self._kills} times; the last: {command[:200]}")
        if upgrade.sandbox is None:
            return ExecuteResponse(output=f"{result.output}\n[{upgrade.note}]", exit_code=result.exit_code)
        retried, measure = run_measured(upgrade.sandbox, command, timeout)
        self._record(measure)
        return ExecuteResponse(
            output=(f"[It ran out of memory again, so: {upgrade.note} The command was run again there.]\n"
                    f"{retried.output}"),
            exit_code=retried.exit_code,
            truncated=retried.truncated,
        )

    @property
    def may_ask(self) -> bool:
        """Whether enough different attempts ran out of memory to allow a bigger sandbox."""
        return len(self._killed_attempts) >= self.retries_first + 1

    def _advice(self, repeated: bool) -> str:
        if self.may_ask:
            return ASK_FIRST if self.retries_first == 0 and self._kills == 1 else MAY_ASK
        return SAME_AGAIN if repeated else TRY_LIGHTER

    def request_bigger(self, reason: str) -> str:
        """Move to a bigger sandbox for work that needs more memory. What happened, for the agent."""
        if self.on_out_of_memory is None:
            return "No bigger sandbox is available here."
        if not self.may_ask:
            return NOT_YET
        upgrade = self._ask(reason)
        if upgrade.sandbox is None:
            return upgrade.note
        return f"{upgrade.note} Run the command again; it now runs there."

    def _ask(self, reason: str) -> Upgrade:
        try:
            upgrade = self.on_out_of_memory(reason)
        except Exception as e:
            log.warning("could not get a bigger sandbox", exc_info=True)
            return Upgrade(None, f"A bigger sandbox could not be started ({e}).")
        if upgrade.sandbox is not None:
            self._replaced = upgrade.sandbox
        return upgrade

    def _record(self, measure: StepMeasure) -> None:
        if self.on_step is not None:
            try:
                self.on_step(measure)
            except Exception:
                log.warning("could not record a sandbox step", exc_info=True)

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
        self._changes += 1
        try:
            return self._real().write(file_path, content)
        except SandboxUnavailable as e:
            return WriteResult(error=self._why(e), path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        self._changes += 1
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
