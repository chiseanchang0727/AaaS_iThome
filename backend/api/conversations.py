"""Conversations: an agent, a sandbox and a memory per thread, reused across turns.

A conversation's sandbox is started on its first message and kept for its
follow-ups, so exported files and loaded data survive between turns. It is
deleted after `idle_seconds` without a message: sandboxes are billed while
they exist. The conversation's memory (the checkpointer) outlives the sandbox;
a later message gets a fresh sandbox, and files from the old one are gone.

The sandbox starts in the background: the agent is built on a stand-in
(sandboxes.LazySandbox) and starts answering at once. SQL never touches the
sandbox; the first code or file step waits for it. If it never comes, those
steps get an error result saying so, and the next message tries again.

At most `max_sandboxes` exist at once. A start waits up to `wait_seconds` for
one to be deleted, then gives up (`SandboxesBusy`).

A sandbox can die on its own (the provider stops it, a crash). Before a turn,
one unused for `check_after_seconds` is checked, and replaced if it does not
answer: same slot, same memory, new empty sandbox.

If the provider's sandboxes carry a `server` label (api/main.py gives each
run its own), `start` also deletes sandboxes an earlier run left behind (it
crashed before deleting them): same labels, another `server`.

Starting a sandbox takes ~12s (create, then pip install). `warm_sandboxes`
keeps that many started and prepared ahead of time; a new conversation takes
one and only waits for its uploaded files to be copied in. Warm sandboxes hold
slots like any other, and the pool refills only from slots nobody is waiting
for.
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
from sandboxes.lazy import LazySandbox

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
    """The real sandbox once started; the agent holds a stand-in until then."""
    last_used: float
    sandbox_ok_at: float = 0.0
    """When the sandbox last worked: started, passed a check, or ran a turn."""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    starting: asyncio.Task[None] | None = None
    """The sandbox start running in the background, if any."""
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
        warm_sandboxes: int = 0,
    ) -> None:
        if max_sandboxes is not None and warm_sandboxes > max_sandboxes:
            raise ValueError("warm_sandboxes cannot be more than max_sandboxes")
        self._build_agent = build_agent
        self._provider = provider
        self._idle_seconds = idle_seconds
        self._on_sandbox_ready = on_sandbox_ready
        self._clock = clock
        self._max_sandboxes = max_sandboxes
        self._in_use = 0
        """Slots taken: sandboxes that exist or are starting, warm ones included."""
        self._waiting = 0
        """New conversations waiting for a slot or a warm sandbox."""
        self._changed = asyncio.Event()
        """Set when a slot frees up or a warm sandbox is ready."""
        self._wait_seconds = wait_seconds
        self._check_after_seconds = check_after_seconds
        self._warm_target = warm_sandboxes if provider is not None else 0
        self._warm: list[tuple[SandboxBackendProtocol, float]] = []
        """Ready sandboxes, oldest first, with when each became ready."""
        self._warming: set[asyncio.Task[None]] = set()
        self._closing = False
        self._cleanup: asyncio.Task[list[str]] | None = None
        self._conversations: dict[str, Conversation] = {}
        self._starts: set[asyncio.Task[None]] = set()
        self.checkpointer = InMemorySaver()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def live_sandboxes(self) -> list[SandboxBackendProtocol]:
        """Sandboxes in use by conversations (not warm ones: they get files when handed out)."""
        return [c.sandbox for c in self._conversations.values() if c.sandbox is not None]

    def status(self) -> dict[str, Any]:
        """Where every sandbox slot is, for the app's status bar.

        busy: a conversation is answering in it. idle: its conversation is
        waiting for the next message (deleted after `idle_seconds`).
        starting: being started for a conversation. warm / warming: ready for
        the next new conversation / being prepared to be. waiting: starts
        waiting for a slot.
        """
        with_sandbox = [c for c in self._conversations.values() if c.sandbox is not None]
        busy = sum(c.lock.locked() for c in with_sandbox)
        return {
            "enabled": self._provider is not None,
            "provider": getattr(self._provider, "name", None),
            "max": self._max_sandboxes,
            "in_use": self._in_use,
            "busy": busy,
            "idle": len(with_sandbox) - busy,
            "starting": sum(
                c.sandbox is None and c.starting is not None and not c.starting.done()
                for c in self._conversations.values()
            ),
            "warm": len(self._warm),
            "warming": self._warming_count(),
            "waiting": self._waiting,
            "idle_minutes": self._idle_seconds / 60,
        }

    def start(self) -> None:
        """Delete what a crashed run left behind, and begin filling the warm pool.

        Call once the event loop runs (app startup). Both run in the background.
        """
        if self._provider is not None and self._provider.labels.get("server"):
            self._cleanup = asyncio.create_task(self.clean_up_leftovers())
        self._refill()

    async def clean_up_leftovers(self) -> list[str]:
        """Delete sandboxes with this run's labels but another run's `server`. Their ids.

        Another run that is still alive with the same labels would lose its
        sandboxes too: each deployment needs its own labels (e.g. env).
        """
        own = self._provider.labels.get("server")
        if not own:
            return []
        query = {k: v for k, v in self._provider.labels.items() if k != "server"}
        try:
            found = await asyncio.to_thread(self._provider.find, query)
        except Exception:
            logger.warning("could not list sandboxes to clean up", exc_info=True)
            return []
        deleted = []
        for leftover in found:
            if leftover.labels.get("server") == own:
                continue
            try:
                await asyncio.to_thread(self._provider.delete_found, leftover)
                deleted.append(leftover.id)
            except Exception:
                logger.warning("could not delete leftover sandbox %s", leftover.id, exc_info=True)
        if deleted:
            logger.warning("deleted %d sandbox(es) an earlier run left behind: %s", len(deleted), deleted)
        return deleted

    async def get_or_create(self, conversation_id: str) -> Conversation:
        """The live conversation with this id, or a new one. Never waits for a sandbox.

        A new conversation's sandbox starts in the background (`starting`).
        """
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            conversation = Conversation(conversation_id, None, None, self._clock())
            self._conversations[conversation_id] = conversation
            self._launch(conversation, have_slot=False)
        self.touch(conversation)
        return conversation

    def _launch(self, conversation: Conversation, *, have_slot: bool) -> None:
        """Build the agent on a stand-in and start the real sandbox behind it.

        `have_slot`: the conversation already holds a slot (replacing a dead
        sandbox), so the start does not take another.
        """
        if self._provider is None:
            conversation.agent = self._build_agent(None, self.checkpointer)
            return
        lazy = LazySandbox()
        conversation.agent = self._build_agent(lazy, self.checkpointer)
        task = asyncio.create_task(self._start_for(conversation, lazy, have_slot))
        conversation.starting = task
        self._starts.add(task)
        task.add_done_callback(self._starts.discard)

    async def _start_for(self, conversation: Conversation, lazy: LazySandbox, have_slot: bool) -> None:
        try:
            if have_slot:
                try:
                    sandbox = await self._start_sandbox()
                except BaseException:
                    self._free_slot()
                    raise
            else:
                sandbox = await self._start_in_new_slot()
        except BaseException as e:
            logger.warning("no sandbox for conversation %s: %s", conversation.id, e)
            lazy.failed(e if isinstance(e, Exception) else RuntimeError("the start was cancelled"))
            if not isinstance(e, Exception):
                raise
            return
        if self._closing or self._conversations.get(conversation.id) is not conversation:
            await self._destroy(sandbox)  # the conversation ended while it started
            self._free_slot()
            lazy.failed(RuntimeError("the conversation ended"))
            return
        conversation.sandbox = sandbox
        conversation.sandbox_ok_at = self._clock()
        lazy.ready(sandbox)

    async def _start_in_new_slot(self) -> SandboxBackendProtocol:
        """A warm sandbox, or a new one in a freed slot. SandboxesBusy if neither comes."""
        warm = await self._obtain()
        try:
            if warm is None:
                return await self._start_sandbox()
            await self._ready_for_conversation(warm)
            return warm
        except BaseException:
            self._free_slot()
            raise
        finally:
            self._refill()

    def touch(self, conversation: Conversation, sandbox_ok: bool = False) -> None:
        """Mark a conversation as used now, restarting its idle countdown.

        `sandbox_ok`: a turn just ran in its sandbox, so it worked just now.
        """
        conversation.last_used = self._clock()
        if sandbox_ok:
            conversation.sandbox_ok_at = conversation.last_used

    async def ensure_sandbox(self, conversation: Conversation) -> bool:
        """Before a turn: replace a dead sandbox, retry a failed start. True if replaced.

        Only a sandbox unused for `check_after_seconds` is checked: one that
        just ran a turn is taken to be alive, so busy conversations pay
        nothing. A replacement starts in the background, like a first start.
        Call it holding the conversation's lock.
        """
        if self._provider is None:
            return False
        if conversation.sandbox is None:
            if conversation.starting is not None and conversation.starting.done():
                self._launch(conversation, have_slot=False)  # the last start failed: try again
            return False
        if self._clock() - conversation.sandbox_ok_at < self._check_after_seconds:
            return False
        if await self._alive(conversation.sandbox):
            conversation.sandbox_ok_at = self._clock()
            return False

        logger.warning("the sandbox of conversation %s stopped answering; starting a new one", conversation.id)
        await self._destroy(conversation.sandbox)
        conversation.sandbox = None
        self._launch(conversation, have_slot=True)  # reuses the dead sandbox's slot
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
        self._refill()  # also retries a pool that failed to fill earlier
        return idle

    async def close_all(self) -> None:
        self._closing = True
        if self._cleanup is not None:
            await asyncio.gather(self._cleanup, return_exceptions=True)
        # Let sandboxes still starting finish first, so they are deleted too.
        await asyncio.gather(*self._starts, *self._warming, return_exceptions=True)
        for conversation_id in list(self._conversations):
            await self.close(conversation_id)
        while self._warm:
            sandbox, _ = self._warm.pop()
            await self._destroy(sandbox)
            self._free_slot()

    async def _start_sandbox(self) -> SandboxBackendProtocol | None:
        """A new sandbox, prepared and with this conversation's files."""
        sandbox = await self._start_prepared()
        if sandbox is not None:
            await self._ready_for_conversation(sandbox)
        return sandbox

    async def _start_prepared(self) -> SandboxBackendProtocol | None:
        """A new sandbox with its packages installed: the slow part (~12s)."""
        if self._provider is None:
            return None
        sandbox = await asyncio.to_thread(self._provider.create)
        try:
            await asyncio.to_thread(self._provider.prepare, sandbox)
        except BaseException:
            await self._destroy(sandbox)
            raise
        return sandbox

    async def _ready_for_conversation(self, sandbox: SandboxBackendProtocol) -> None:
        """Copy the uploaded files in. Done at hand-out, so a warm sandbox gets
        files uploaded after it was started."""
        if self._on_sandbox_ready is None:
            return
        try:
            await self._on_sandbox_ready(sandbox)
        except BaseException:
            await self._destroy(sandbox)
            raise

    # --- slots and the warm pool ------------------------------------------------

    async def _obtain(self) -> SandboxBackendProtocol | None:
        """A warm sandbox, or None after taking a slot to start one in.

        Waits up to `wait_seconds` for either, then raises `SandboxesBusy`.
        """
        if self._provider is None:
            return None
        deadline = asyncio.get_running_loop().time() + self._wait_seconds
        self._waiting += 1
        try:
            while True:
                warm = await self._take_warm()
                if warm is not None:
                    return warm
                if self._try_take_slot():
                    return None
                remaining = deadline - asyncio.get_running_loop().time()
                self._changed.clear()
                try:
                    await asyncio.wait_for(self._changed.wait(), max(remaining, 0))
                except TimeoutError:
                    raise SandboxesBusy(
                        f"all {self._max_sandboxes} sandboxes are in use; try again in a minute"
                    ) from None
        finally:
            self._waiting -= 1

    async def _take_warm(self) -> SandboxBackendProtocol | None:
        """The oldest warm sandbox that still answers; dead ones are dropped."""
        while self._warm:
            sandbox, ready_at = self._warm.pop(0)
            if self._clock() - ready_at < self._check_after_seconds or await self._alive(sandbox):
                return sandbox
            logger.warning("a warm sandbox stopped answering; dropping it")
            await self._destroy(sandbox)
            self._free_slot()
        return None

    def _refill(self) -> None:
        """Start warm sandboxes until the pool (ready plus starting) is full.

        Only from free slots, and never while a new conversation waits: it
        should get the slot (or the next warm sandbox) first.
        """
        if self._closing:
            return
        while len(self._warm) + self._warming_count() < self._warm_target and self._waiting == 0:
            if not self._try_take_slot():
                return
            task = asyncio.create_task(self._warm_one())
            self._warming.add(task)
            task.add_done_callback(self._warming.discard)

    def _warming_count(self) -> int:
        # A finished task stays in the set until its done-callback runs.
        return sum(not task.done() for task in self._warming)

    async def _warm_one(self) -> None:
        try:
            sandbox = await self._start_prepared()
        except Exception:
            logger.warning("could not start a warm sandbox; will retry later", exc_info=True)
            self._free_slot(refill=False)  # retried by the reaper, not in a tight loop
            return
        if self._closing:
            await self._destroy(sandbox)
            self._free_slot()
            return
        self._warm.append((sandbox, self._clock()))
        self._changed.set()

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

    def _try_take_slot(self) -> bool:
        if self._max_sandboxes is not None and self._in_use >= self._max_sandboxes:
            return False
        self._in_use += 1
        return True

    def _free_slot(self, refill: bool = True) -> None:
        if self._provider is None:
            return
        self._in_use -= 1
        self._changed.set()
        if refill:
            self._refill()
