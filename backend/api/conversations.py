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

A command killed for running out of memory gets a bigger sandbox
(`bigger_sandbox`, e.g. 4 GB): it is started, the work folder copied over,
the conversation switched to it, the old one deleted, and the command run
again. Memory is budgeted in GB (`max_memory_gb`, the provider account's
limit), so a bigger sandbox is only started when it fits.

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
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from pathlib import PurePosixPath

from sandboxes import SandboxProvider
from sandboxes.base import copy_work_dir
from sandboxes.lazy import LazySandbox, Upgrade

logger = logging.getLogger(__name__)

DEFAULT_ACCOUNT = "test_user"
"""Whose a conversation is when nobody says (there is no login yet)."""

HEALTH_CHECK_TIMEOUT = 15.0
"""Seconds a sandbox gets to answer a health check before it counts as dead."""


class SandboxesBusy(RuntimeError):
    """Every sandbox slot stayed taken for the whole wait."""


class NotYourConversation(LookupError):
    """The conversation belongs to another account."""

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
    stand_in: LazySandbox | None = None
    """What the agent was built on; its `on_step` hears what each code step cost."""
    announced: set[str] = field(default_factory=set)
    """Artifact paths already sent to the client, so each is sent once."""
    account: str = DEFAULT_ACCOUNT
    """Who it belongs to. Only that account can use it."""
    memory_gb: int = 1
    """Memory of its sandbox: the provider's default, or more after an upgrade."""
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    """Hears what happens to its sandbox (ready, moved, ...). Set per turn by the API."""


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
        max_per_account: int | None = None,
        max_memory_gb: int | None = None,
        bigger_sandbox: tuple[int, int] | None = None,
        work_dir: PurePosixPath | None = None,
        copy_files: Callable[[SandboxBackendProtocol, SandboxBackendProtocol], None] | None = None,
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
        self._max_per_account = max_per_account
        self._default_gb = provider.default_memory_gb if provider is not None else 0
        self._max_memory_gb = max_memory_gb
        self._memory_used = 0
        """GB of memory in sandboxes that exist or are starting, warm ones included."""
        self._bigger = bigger_sandbox
        """(memory GB, vCPU) for a sandbox that ran out of memory; None: no upgrades."""
        if copy_files is None and work_dir is not None:
            copy_files = lambda old, new: copy_work_dir(old, new, work_dir)  # noqa: E731
        self._copy_files = copy_files
        self._held: Counter[str] = Counter()
        """Slots each account holds (its conversations' sandboxes, started or starting)."""
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
            "max_per_account": self._max_per_account,
            "accounts": self._account_status(),
            "memory_used_gb": self._memory_used,
            "max_memory_gb": self._max_memory_gb,
            "bigger": sum(c.sandbox is not None and c.memory_gb > self._default_gb for c in self._conversations.values()),
        }

    def _account_status(self) -> dict[str, dict[str, int]]:
        accounts: dict[str, dict[str, int]] = {}
        for c in self._conversations.values():
            a = accounts.setdefault(c.account, {"sandboxes": self._held[c.account], "busy": 0, "idle": 0, "starting": 0})
            if c.sandbox is not None:
                a["busy" if c.lock.locked() else "idle"] += 1
            elif c.starting is not None and not c.starting.done():
                a["starting"] += 1
        return accounts

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

    async def get_or_create(self, conversation_id: str, account: str = DEFAULT_ACCOUNT) -> Conversation:
        """The live conversation with this id, or a new one for `account`. Never waits for a sandbox.

        A new conversation's sandbox starts in the background (`starting`).
        Raises NotYourConversation if the id is another account's.
        """
        conversation = self._conversations.get(conversation_id)
        if conversation is not None and conversation.account != account:
            raise NotYourConversation(conversation_id)
        if conversation is None:
            conversation = Conversation(
                conversation_id, None, None, self._clock(), account=account, memory_gb=self._default_gb
            )
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
        conversation.stand_in = lazy
        conversation.agent = self._build_agent(lazy, self.checkpointer)
        task = asyncio.create_task(self._start_for(conversation, lazy, have_slot))
        conversation.starting = task
        self._starts.add(task)
        task.add_done_callback(self._starts.discard)

    def _event(self, conversation: Conversation, event: str, **fields: Any) -> None:
        if conversation.on_event is None:
            return
        try:
            conversation.on_event(event, fields)
        except Exception:
            logger.warning("could not record sandbox event %s", event, exc_info=True)

    async def _start_for(self, conversation: Conversation, lazy: LazySandbox, have_slot: bool) -> None:
        started = time.monotonic()
        try:
            if have_slot:
                try:
                    sandbox, how = await self._start_sandbox(), "replacement"
                except BaseException:
                    self._free_slot(conversation.account)
                    raise
            else:
                sandbox, how = await self._start_in_new_slot(conversation.account)
        except BaseException as e:
            logger.warning("no sandbox for conversation %s: %s", conversation.id, e)
            self._event(conversation, "sandbox_unavailable", reason=str(e), seconds=round(time.monotonic() - started, 2))
            lazy.failed(e if isinstance(e, Exception) else RuntimeError("the start was cancelled"))
            if not isinstance(e, Exception):
                raise
            return
        if self._closing or self._conversations.get(conversation.id) is not conversation:
            await self._destroy(sandbox)  # the conversation ended while it started
            self._free_slot(conversation.account)
            lazy.failed(RuntimeError("the conversation ended"))
            return
        conversation.sandbox = sandbox
        conversation.sandbox_ok_at = self._clock()
        lazy.on_out_of_memory = self._upgrader(conversation)
        lazy.ready(sandbox)
        self._event(conversation, "sandbox_ready", how=how, sandbox=getattr(sandbox, "id", None),
                    memory_gb=conversation.memory_gb, seconds=round(time.monotonic() - started, 2))

    def _upgrader(self, conversation: Conversation) -> Callable[[str], Upgrade]:
        """`upgrade`, callable from the worker thread that ran out of memory."""
        loop = asyncio.get_running_loop()

        def ask(reason: str = "") -> Upgrade:
            return asyncio.run_coroutine_threadsafe(self.upgrade(conversation, reason), loop).result(timeout=900)

        return ask

    def _upgrade_failed(self, conversation: Conversation, note: str) -> Upgrade:
        self._event(conversation, "upgrade_failed", reason=note)
        return Upgrade(None, note)

    async def upgrade(self, conversation: Conversation, reason: str = "") -> Upgrade:
        """Move the conversation to a bigger sandbox, with its work folder. Why not, if not."""
        smaller = ("To fit, use less memory: select fewer columns, aggregate earlier, "
                   "query the file with DuckDB, or sample.")
        if self._bigger is None or self._provider is None or conversation.sandbox is None:
            return Upgrade(None, f"No bigger sandbox is available. {smaller}")
        memory_gb, cpu = self._bigger
        if conversation.memory_gb >= memory_gb:
            return self._upgrade_failed(
                conversation, f"This sandbox already has {conversation.memory_gb} GB, the most available. {smaller}")
        if self._max_memory_gb is not None and self._memory_used + memory_gb > self._max_memory_gb:
            return self._upgrade_failed(conversation, (f"No room for a {memory_gb} GB sandbox right now "
                                                       f"({self._memory_used} of {self._max_memory_gb} GB in use). {smaller}"))
        old, old_gb = conversation.sandbox, conversation.memory_gb
        self._event(conversation, "upgrade_started", from_gb=old_gb, to_gb=memory_gb, cpu=cpu, reason=reason,
                    sandbox=getattr(old, "id", None), memory_used_gb=self._memory_used + memory_gb)
        self._memory_used += memory_gb  # held while both sandboxes exist
        t = time.monotonic()
        try:
            bigger = await asyncio.to_thread(self._provider.create_bigger, memory_gb, cpu)
        except NotImplementedError:
            self._memory_used -= memory_gb
            return self._upgrade_failed(conversation, f"This sandbox provider cannot make bigger sandboxes. {smaller}")
        except Exception as e:
            self._memory_used -= memory_gb
            logger.warning("could not start a bigger sandbox", exc_info=True)
            return self._upgrade_failed(conversation, f"A bigger sandbox could not be started ({e}). {smaller}")
        self._event(conversation, "bigger_created", sandbox=getattr(bigger, "id", None), memory_gb=memory_gb,
                    cpu=cpu, seconds=round(time.monotonic() - t, 2))
        try:
            t = time.monotonic()
            await asyncio.to_thread(self._provider.prepare, bigger)
            self._event(conversation, "packages_installed", packages=list(self._provider.packages),
                        seconds=round(time.monotonic() - t, 2))
            if self._copy_files is not None:
                t = time.monotonic()
                moved = await asyncio.to_thread(self._copy_files, conversation.sandbox, bigger)
                self._event(conversation, "files_copied", bytes=moved if isinstance(moved, int) else None,
                            seconds=round(time.monotonic() - t, 2))
        except Exception as e:
            await self._destroy(bigger)
            self._memory_used -= memory_gb
            logger.warning("could not move to a bigger sandbox", exc_info=True)
            return self._upgrade_failed(conversation, f"The work could not be moved to a bigger sandbox ({e}). {smaller}")
        conversation.sandbox, conversation.memory_gb = bigger, memory_gb
        conversation.sandbox_ok_at = self._clock()
        self._event(conversation, "switched", sandbox=getattr(bigger, "id", None), memory_gb=memory_gb)
        t = time.monotonic()
        await self._destroy(old)
        self._memory_used -= old_gb
        self._changed.set()
        self._event(conversation, "old_deleted", sandbox=getattr(old, "id", None), memory_gb=old_gb,
                    seconds=round(time.monotonic() - t, 2), memory_used_gb=self._memory_used)
        logger.info("conversation %s moved to a %d GB sandbox", conversation.id, memory_gb)
        return Upgrade(bigger, f"Moved to a bigger sandbox ({memory_gb} GB memory, {cpu} vCPU) "
                               "with the files from the work folder.")

    async def _start_in_new_slot(self, account: str) -> tuple[SandboxBackendProtocol, str]:
        """A warm sandbox, or a new one in a freed slot, and which. SandboxesBusy if neither comes."""
        warm = await self._obtain(account)
        try:
            if warm is None:
                return await self._start_sandbox(), "new"
            await self._ready_for_conversation(warm)
            return warm, "warm"
        except BaseException:
            self._free_slot(account)
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
        self._event(conversation, "sandbox_dead", sandbox=getattr(conversation.sandbox, "id", None))
        await self._destroy(conversation.sandbox)
        conversation.sandbox = None
        if conversation.memory_gb != self._default_gb:  # the replacement is default-sized
            self._memory_used -= conversation.memory_gb - self._default_gb
            conversation.memory_gb = self._default_gb
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
            self._free_slot(None)

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

    async def _obtain(self, account: str) -> SandboxBackendProtocol | None:
        """A warm sandbox, or None after taking a slot to start one in, for `account`.

        An account with no sandbox is guaranteed one: when every slot is taken,
        the longest-idle sandbox of an account holding two or more is closed
        for it (that conversation keeps its memory and gets a new sandbox on
        its next message). An account at `max_per_account` waits for one of
        its own. Waits up to `wait_seconds`, then raises `SandboxesBusy`.
        """
        if self._provider is None:
            return None
        deadline = asyncio.get_running_loop().time() + self._wait_seconds
        self._waiting += 1
        try:
            while True:
                if self._max_per_account is not None and self._held[account] >= self._max_per_account:
                    reason = (f"this account already uses {self._held[account]} sandboxes, its limit; "
                              "finish or close a conversation first")
                else:
                    warm = await self._take_warm()
                    if warm is not None:
                        self._held[account] += 1
                        return warm
                    if self._try_take_slot():
                        self._held[account] += 1
                        return None
                    if self._held[account] == 0 and await self._reclaim_for(account):
                        continue
                    reason = self._full_reason()
                remaining = deadline - asyncio.get_running_loop().time()
                self._changed.clear()
                try:
                    await asyncio.wait_for(self._changed.wait(), max(remaining, 0))
                except TimeoutError:
                    raise SandboxesBusy(reason) from None
        finally:
            self._waiting -= 1

    def _full_reason(self) -> str:
        if self._max_sandboxes is not None and self._in_use >= self._max_sandboxes:
            return f"all {self._max_sandboxes} sandboxes are in use; try again in a minute"
        return (f"all {self._max_memory_gb} GB of sandbox memory is in use "
                f"({self._memory_used} GB); try again in a minute")

    async def _reclaim_for(self, account: str) -> bool:
        """Close another account's longest-idle sandbox, if one holds two or more. True if closed."""
        candidates = [
            c for c in self._conversations.values()
            if c.account != account and c.sandbox is not None and not c.lock.locked()
            and self._held[c.account] >= 2
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda c: c.last_used)
        logger.info("closing %s's idle conversation %s so %s gets a sandbox", victim.account, victim.id, account)
        await self.close(victim.id)
        return True

    async def _take_warm(self) -> SandboxBackendProtocol | None:
        """The oldest warm sandbox that still answers; dead ones are dropped."""
        while self._warm:
            sandbox, ready_at = self._warm.pop(0)
            if self._clock() - ready_at < self._check_after_seconds or await self._alive(sandbox):
                return sandbox
            logger.warning("a warm sandbox stopped answering; dropping it")
            await self._destroy(sandbox)
            self._free_slot(None)
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
            self._free_slot(None, refill=False)  # retried by the reaper, not in a tight loop
            return
        if self._closing:
            await self._destroy(sandbox)
            self._free_slot(None)
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
            self._free_slot(conversation.account, gb=conversation.memory_gb)

    async def _destroy(self, sandbox: SandboxBackendProtocol) -> None:
        try:
            await asyncio.to_thread(self._provider.destroy, sandbox)
        except Exception:
            logger.exception("could not delete sandbox %s", getattr(sandbox, "id", sandbox))

    def _try_take_slot(self) -> bool:
        """A slot for a default-sized sandbox, if one is free and its memory fits the budget."""
        if self._max_sandboxes is not None and self._in_use >= self._max_sandboxes:
            return False
        if self._max_memory_gb is not None and self._memory_used + self._default_gb > self._max_memory_gb:
            return False
        self._in_use += 1
        self._memory_used += self._default_gb
        return True

    def _free_slot(self, account: str | None, refill: bool = True, gb: int | None = None) -> None:
        """Give back a slot; `account` is whose it was (None: the warm pool's), `gb` its memory."""
        if self._provider is None:
            return
        self._in_use -= 1
        self._memory_used -= self._default_gb if gb is None else gb
        if account is not None:
            self._held[account] -= 1
            if self._held[account] <= 0:
                del self._held[account]
        self._changed.set()
        if refill:
            self._refill()
