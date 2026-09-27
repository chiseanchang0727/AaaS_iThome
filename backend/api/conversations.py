"""Conversations: an agent, a sandbox and a memory per thread, reused across turns.

A conversation's sandbox is created on its first message and kept for its
follow-ups, so exported files and loaded data survive between turns. It is
deleted after `idle_seconds` without a message: sandboxes are billed while
they exist. The conversation's memory (the checkpointer) outlives the sandbox;
a later message gets a fresh sandbox, and files from the old one are gone.
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

AgentBuilder = Callable[[SandboxBackendProtocol | None, BaseCheckpointSaver], Any]
SandboxHook = Callable[[SandboxBackendProtocol], Awaitable[None]]


@dataclass
class Conversation:
    id: str
    agent: Any
    sandbox: SandboxBackendProtocol | None
    last_used: float
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
    ) -> None:
        self._build_agent = build_agent
        self._provider = provider
        self._idle_seconds = idle_seconds
        self._on_sandbox_ready = on_sandbox_ready
        self._clock = clock
        self._conversations: dict[str, Conversation] = {}
        self._create_lock = asyncio.Lock()
        self.checkpointer = InMemorySaver()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def live_sandboxes(self) -> list[SandboxBackendProtocol]:
        return [c.sandbox for c in self._conversations.values() if c.sandbox is not None]

    async def get_or_create(self, conversation_id: str) -> Conversation:
        """The live conversation with this id, or a new one (sandbox included)."""
        async with self._create_lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                sandbox = await self._start_sandbox()
                agent = self._build_agent(sandbox, self.checkpointer)
                conversation = Conversation(conversation_id, agent, sandbox, self._clock())
                self._conversations[conversation_id] = conversation
            self.touch(conversation)
            return conversation

    def touch(self, conversation: Conversation) -> None:
        """Mark a conversation as used now, restarting its idle countdown."""
        conversation.last_used = self._clock()

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
            await asyncio.to_thread(self._provider.destroy, conversation.sandbox)
        except Exception:
            logger.exception("could not delete the sandbox of conversation %s", conversation.id)
