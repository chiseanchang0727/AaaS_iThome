"""ConversationManager when sandboxes are slow to start and requests overlap.

Sandbox starts are real threads (asyncio.to_thread), held at a gate the test
opens, so "one start is still running" is a state the test controls.
"""

import asyncio
import threading

import pytest

from api import ConversationManager, SandboxesBusy
from sandboxes import SandboxProvider


class Sandbox:
    def __init__(self, n):
        self.id = f"sandbox-{n}"


class GatedProvider(SandboxProvider):
    """`create` waits until `release()`; records what it made and deleted."""

    name = "gated"

    def __init__(self, fail=False, open=False):
        super().__init__()
        self.gate = threading.Event()
        if open:
            self.gate.set()
        self.alive = True
        self.checks = 0
        self.entered = threading.Semaphore(0)
        self.fail = fail
        self.created, self.destroyed = [], []

    def create(self):
        self.entered.release()
        self.gate.wait(timeout=5)
        if self.fail:
            raise RuntimeError("quota exceeded")
        sandbox = Sandbox(len(self.created))
        self.created.append(sandbox)
        return sandbox

    def destroy(self, sandbox):
        self.destroyed.append(sandbox)

    def is_alive(self, sandbox):
        self.checks += 1
        return self.alive

    def release(self):
        self.gate.set()

    async def wait_entered(self, n=1):
        for _ in range(n):
            assert await asyncio.to_thread(self.entered.acquire, True, 5), "create was never called"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def manager(provider, **options) -> ConversationManager:
    return ConversationManager(
        lambda sandbox, checkpointer: {"sandbox": sandbox}, provider, idle_seconds=900, **options
    )


def run(coro):
    return asyncio.run(coro)


def test_a_live_conversation_does_not_wait_for_another_to_start():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        provider.release()
        await m.get_or_create("old")
        provider.gate.clear()

        starting = asyncio.create_task(m.get_or_create("new"))
        await provider.wait_entered(2)  # "old", then "new" is now held at the gate
        old = await asyncio.wait_for(m.get_or_create("old"), timeout=1)
        assert old.id == "old" and not starting.done()
        provider.release()
        await starting

    run(scenario())


def test_new_conversations_start_side_by_side():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        a = asyncio.create_task(m.get_or_create("a"))
        b = asyncio.create_task(m.get_or_create("b"))
        await provider.wait_entered(2)  # both inside create at once
        provider.release()
        assert {c.sandbox.id for c in await asyncio.gather(a, b)} == {"sandbox-0", "sandbox-1"}

    run(scenario())


def test_two_requests_for_one_new_conversation_share_one_sandbox():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        first = asyncio.create_task(m.get_or_create("same"))
        second = asyncio.create_task(m.get_or_create("same"))
        await provider.wait_entered()
        provider.release()
        one, two = await asyncio.gather(first, second)
        assert one is two and len(provider.created) == 1

    run(scenario())


def test_a_failed_start_fails_its_waiters_and_can_be_retried():
    async def scenario():
        provider = GatedProvider(fail=True)
        m = manager(provider)
        waiters = [asyncio.create_task(m.get_or_create("x")) for _ in range(2)]
        provider.release()
        for w in waiters:
            with pytest.raises(RuntimeError, match="quota"):
                await w
        assert m.get("x") is None

        provider.fail = False
        assert (await m.get_or_create("x")).sandbox.id == "sandbox-0"

    run(scenario())


def test_a_waiter_that_leaves_does_not_cancel_the_start():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        leaving = asyncio.create_task(m.get_or_create("x"))
        await provider.wait_entered()
        leaving.cancel()  # the browser closed the stream
        provider.release()
        conversation = await m.get_or_create("x")
        assert conversation.sandbox.id == "sandbox-0" and len(provider.created) == 1

    run(scenario())


def test_shutdown_also_deletes_a_sandbox_still_starting():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        asyncio.create_task(m.get_or_create("late"))
        await provider.wait_entered()
        closing = asyncio.create_task(m.close_all())
        await asyncio.sleep(0)
        provider.release()
        await closing
        assert provider.destroyed == provider.created and len(provider.created) == 1

    run(scenario())


# --- the cap on sandboxes ----------------------------------------------------------


def test_a_full_house_refuses_a_new_conversation_after_the_wait():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=1, wait_seconds=0.05)
        await m.get_or_create("first")
        with pytest.raises(SandboxesBusy, match="all 1 sandboxes are in use"):
            await m.get_or_create("second")
        assert len(provider.created) == 1 and m.get("second") is None
        assert (await m.get_or_create("first")).id == "first"  # live ones are unaffected

    run(scenario())


def test_a_waiting_conversation_gets_the_slot_a_closed_one_frees():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=1, wait_seconds=5)
        await m.get_or_create("first")
        waiting = asyncio.create_task(m.get_or_create("second"))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        await m.close("first")
        assert (await asyncio.wait_for(waiting, 1)).sandbox.id == "sandbox-1"

    run(scenario())


def test_a_failed_start_gives_its_slot_back():
    async def scenario():
        provider = GatedProvider(fail=True, open=True)
        m = manager(provider, max_sandboxes=1, wait_seconds=0.05)
        with pytest.raises(RuntimeError, match="quota"):
            await m.get_or_create("x")
        provider.fail = False
        assert (await m.get_or_create("y")).sandbox is not None

    run(scenario())


def test_no_limit_by_default():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider)
        for n in range(5):
            await m.get_or_create(f"c{n}")
        assert len(provider.created) == 5

    run(scenario())


# --- replacing a dead sandbox --------------------------------------------------------


def test_a_recently_used_sandbox_is_not_checked():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, check_after_seconds=60)
        conversation = await m.get_or_create("x")
        clock.now = 59
        assert await m.ensure_sandbox(conversation) is False
        assert provider.checks == 0

    run(scenario())


def test_an_idle_sandbox_that_answers_is_kept():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, check_after_seconds=60)
        conversation = await m.get_or_create("x")
        clock.now = 120
        assert await m.ensure_sandbox(conversation) is False
        assert provider.checks == 1 and conversation.sandbox.id == "sandbox-0"
        assert await m.ensure_sandbox(conversation) is False  # just checked: not again
        assert provider.checks == 1

    run(scenario())


def test_a_dead_sandbox_is_replaced_in_the_same_slot():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, check_after_seconds=60, max_sandboxes=1, wait_seconds=0.05)
        conversation = await m.get_or_create("x")
        old = conversation.sandbox
        clock.now, provider.alive = 120, False

        assert await m.ensure_sandbox(conversation) is True
        assert provider.destroyed == [old]
        assert conversation.sandbox.id == "sandbox-1"
        assert conversation.agent == {"sandbox": conversation.sandbox}  # rebuilt on the new one
        with pytest.raises(SandboxesBusy):  # still holds its one slot
            await m.get_or_create("other")

    run(scenario())


def test_if_no_new_sandbox_starts_the_conversation_is_closed_and_its_slot_freed():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, check_after_seconds=60, max_sandboxes=1, wait_seconds=0.05)
        conversation = await m.get_or_create("x")
        clock.now, provider.alive, provider.fail = 120, False, True

        with pytest.raises(RuntimeError, match="quota"):
            await m.ensure_sandbox(conversation)
        assert m.get("x") is None
        provider.fail = False
        assert (await m.get_or_create("y")).sandbox is not None

    run(scenario())
