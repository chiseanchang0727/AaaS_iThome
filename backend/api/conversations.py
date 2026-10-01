"""Conversations: an agent, a sandbox and a memory per thread, reused across turns.

A conversation's sandbox is created on its first message and kept for its
follow-ups, so exported files and loaded data survive between turns. It is
deleted after `idle_seconds` without a message: sandboxes are billed while
they exist. The conversation's memory (the checkpointer) outlives the sandbox;
a later message gets a fresh sandbox, and files from the old one are gone.

At most `max_sandboxes` exist at once. A new conversation waits up to
`wait_seconds` for one to be deleted, then is refused (`SandboxesBusy`).

A sandbox can die on its own (the provider stops it, a crash). Before a turn,
one unused for `check_after_seconds` is checked, and replaced if it does not
answer: same slot, same memory, new empty sandbox.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from sandboxes import SandboxProvider

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 15.0
"""Seconds a sandbox gets to answer a health check before it counts as dead."""


class SandboxesBusy(RuntimeError):
    """Every sandbox slot stayed taken for the whole wait."""

AgentBuilder = Callable[[SandboxBackendProtocol | None, BaseCheckpointSaver], Any]
SandboxHook = Callable[[SandboxBackendProtocol], Awaitable[None]]


@dataclass
class Conversation:
    id: str
    agent: Any
    sandbox: SandboxBackendProtocol | None
    last_used: float
    sandbox_ok_at: float = 0.0
    """When the sandbox last worked: started, passed a check, or ran a turn."""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    announced: set[str] = field(default_factory=set)
    """Artifact paths already sent to the client, so each is sent once."""


class ConversationManager:
    def __init__(
        self,
        build_agent: AgentBuilder,
        provider: SandboxProvider | None,
        idle_seconds: float,
        on_sandbox_ready: SandboxHook | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_sandboxes: int | None = None,
        wait_seconds: float = 30.0,
        check_after_seconds: float = 60.0,
    ) -> None:
        self._build_agent = build_agent
        self._provider = provider
        self._idle_seconds = idle_seconds
        self._on_sandbox_ready = on_sandbox_ready
        self._clock = clock
        self._max_sandboxes = max_sandboxes
        self._slots = asyncio.Semaphore(max_sandboxes) if max_sandboxes else None
        self._wait_seconds = wait_seconds
        self._check_after_seconds = check_after_seconds
        self._conversations: dict[str, Conversation] = {}
        self._starting: dict[str, asyncio.Task[Conversation]] = {}
        """Conversations whose sandbox is being started, so a second request
        for the same new id waits for that start instead of making another."""
        self.checkpointer = InMemorySaver()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def live_sandboxes(self) -> list[SandboxBackendProtocol]:
        return [c.sandbox for c in self._conversations.values() if c.sandbox is not None]

    async def get_or_create(self, conversation_id: str) -> Conversation:
        """The live conversation with this id, or a new one (sandbox included).

        A live conversation is returned at once, never waiting on another
        conversation's sandbox. New conversations start their sandboxes side
        by side; two requests for the same new id share one start.
        """
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            starting = self._starting.get(conversation_id)
            if starting is None:
                starting = asyncio.create_task(self._create(conversation_id))
                self._starting[conversation_id] = starting
                starting.add_done_callback(lambda _: self._starting.pop(conversation_id, None))
            # Shielded: a client that disconnects mid-start must not cancel the
            # start for others waiting on it, or leave a half-made sandbox.
            conversation = await asyncio.shield(starting)
        self.touch(conversation)
        return conversation

    async def _create(self, conversation_id: str) -> Conversation:
        await self._take_slot()
        try:
            sandbox = await self._start_sandbox()
        except BaseException:
            self._free_slot()
            raise
        agent = self._build_agent(sandbox, self.checkpointer)
        now = self._clock()
        conversation = Conversation(conversation_id, agent, sandbox, now, sandbox_ok_at=now)
        self._conversations[conversation_id] = conversation
        return conversation

    def touch(self, conversation: Conversation, sandbox_ok: bool = False) -> None:
        """Mark a conversation as used now, restarting its idle countdown.

        `sandbox_ok`: a turn just ran in its sandbox, so it worked just now.
        """
        conversation.last_used = self._clock()
        if sandbox_ok:
            conversation.sandbox_ok_at = conversation.last_used

    async def ensure_sandbox(self, conversation: Conversation) -> bool:
        """Replace the conversation's sandbox if it stopped answering. True if replaced.

        Only a sandbox unused for `check_after_seconds` is checked: one that
        just ran a turn is taken to be alive, so busy conversations pay
        nothing. Call it holding the conversation's lock. If no new sandbox
        can be started, the conversation is closed (memory kept) and the
        error raised; its next message starts over.
        """
        if conversation.sandbox is None or self._provider is None:
            return False
        if self._clock() - conversation.sandbox_ok_at < self._check_after_seconds:
            return False
        if await self._alive(conversation.sandbox):
            conversation.sandbox_ok_at = self._clock()
            return False

        logger.warning("the sandbox of conversation %s stopped answering; starting a new one", conversation.id)
        await self._destroy(conversation.sandbox)
        conversation.sandbox = None
        try:
            sandbox = await self._start_sandbox()  # reuses the dead sandbox's slot
        except BaseException:
            self._conversations.pop(conversation.id, None)
            self._free_slot()
            raise
        conversation.sandbox = sandbox
        conversation.agent = self._build_agent(sandbox, self.checkpointer)
        conversation.sandbox_ok_at = self._clock()
        return True

    async def _alive(self, sandbox: SandboxBackendProtocol) -> bool:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._provider.is_alive, sandbox), HEALTH_CHECK_TIMEOUT
            )
        except TimeoutError:
            return False

    async def close(self, conversation_id: str) -> bool:
        """Delete a conversation's sandbox. Its memory stays in the checkpointer."""
        conversation = self._conversations.pop(conversation_id, None)
        if conversation is None:
            return False
        await self._stop_sandbox(conversation)
        return True

    async def reap_idle(self) -> list[str]:
        """Close conversations idle for longer than `idle_seconds`, unless busy."""
        now = self._clock()
        idle = [
            c.id for c in self._conversations.values()
            if now - c.last_used > self._idle_seconds and not c.lock.locked()
        ]
        for conversation_id in idle:
            await self.close(conversation_id)
        return idle

    async def close_all(self) -> None:
        # Let sandboxes still starting finish first, so they are deleted too.
        await asyncio.gather(*self._starting.values(), return_exceptions=True)
        for conversation_id in list(self._conversations):
            await self.close(conversation_id)

    async def _start_sandbox(self) -> SandboxBackendProtocol | None:
        if self._provider is None:
            return None
        sandbox = await asyncio.to_thread(self._provider.create)
        try:
            await asyncio.to_thread(self._provider.prepare, sandbox)
            if self._on_sandbox_ready is not None:
                await self._on_sandbox_ready(sandbox)
        except BaseException:
            await asyncio.to_thread(self._provider.destroy, sandbox)
            raise
        return sandbox

    async def _stop_sandbox(self, conversation: Conversation) -> None:
        if conversation.sandbox is None or self._provider is None:
            return
        try:
            await self._destroy(conversation.sandbox)
        finally:
            conversation.sandbox = None
            self._free_slot()

    async def _destroy(self, sandbox: SandboxBackendProtocol) -> None:
        try:
            await asyncio.to_thread(self._provider.destroy, sandbox)
        except Exception:
            logger.exception("could not delete sandbox %s", getattr(sandbox, "id", sandbox))

    async def _take_slot(self) -> None:
        if self._slots is None or self._provider is None:
            return
        try:
            await asyncio.wait_for(self._slots.acquire(), self._wait_seconds)
        except TimeoutError:
            raise SandboxesBusy(
                f"all {self._max_sandboxes} sandboxes are in use; try again in a minute"
            ) from None

    def _free_slot(self) -> None:
        if self._slots is not None and self._provider is not None:
            self._slots.release()
