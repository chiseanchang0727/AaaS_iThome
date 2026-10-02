"""ConversationManager when sandboxes are slow to start and requests overlap.

Sandbox starts are real threads (asyncio.to_thread), held at a gate the test
opens, so "one start is still running" is a state the test controls. A new
conversation is returned at once; its sandbox starts in the background
(`conversation.starting`), behind the stand-in the agent was built on.
"""

import asyncio
import threading

import pytest

from api import ConversationManager, SandboxesBusy
from sandboxes import FoundSandbox, SandboxProvider


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


async def started(m: ConversationManager, conversation_id: str):
    """The conversation, once its background sandbox start has finished."""
    conversation = await asyncio.wait_for(m.get_or_create(conversation_id), 1)
    if conversation.starting is not None:
        await asyncio.wait_for(asyncio.shield(conversation.starting), 2)
    return conversation


def stand_in(conversation):
    """What the agent was built on (the fake builder keeps it)."""
    return conversation.agent["sandbox"]


# --- starting in the background ------------------------------------------------------


def test_a_new_conversation_is_ready_before_its_sandbox():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        conversation = await asyncio.wait_for(m.get_or_create("x"), 1)  # no wait for the sandbox
        assert conversation.sandbox is None and not stand_in(conversation).settled
        await provider.wait_entered()
        provider.release()
        await conversation.starting
        assert conversation.sandbox.id == "sandbox-0"
        assert stand_in(conversation).id == "sandbox-0"  # the stand-in now passes through

    run(scenario())


def test_the_stand_in_waits_for_the_sandbox_then_passes_calls_through():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        conversation = await m.get_or_create("x")
        reading = asyncio.create_task(asyncio.to_thread(stand_in(conversation).ls, "/"))
        await provider.wait_entered()
        await asyncio.sleep(0.05)
        assert not reading.done()  # waiting for the sandbox
        provider.release()
        with pytest.raises(AttributeError):  # reached the fake sandbox, which has no ls
            await asyncio.wait_for(reading, 2)

    run(scenario())


def test_a_live_conversation_does_not_wait_for_another_to_start():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider)
        await started(m, "old")
        provider.gate.clear()
        new = await m.get_or_create("new")
        await provider.wait_entered(2)
        assert (await asyncio.wait_for(m.get_or_create("old"), 1)).sandbox.id == "sandbox-0"
        provider.release()
        await new.starting

    run(scenario())


def test_new_conversations_start_side_by_side():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        a, b = await m.get_or_create("a"), await m.get_or_create("b")
        await provider.wait_entered(2)  # both inside create at once
        provider.release()
        await asyncio.gather(a.starting, b.starting)
        assert {a.sandbox.id, b.sandbox.id} == {"sandbox-0", "sandbox-1"}

    run(scenario())


def test_two_requests_for_one_new_conversation_share_one_sandbox():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider)
        one, two = await asyncio.gather(m.get_or_create("same"), m.get_or_create("same"))
        await one.starting
        assert one is two and len(provider.created) == 1

    run(scenario())


def test_a_failed_start_is_reported_by_the_stand_in_and_retried_next_turn():
    async def scenario():
        provider = GatedProvider(fail=True, open=True)
        m = manager(provider)
        conversation = await started(m, "x")
        assert conversation.sandbox is None
        result = await asyncio.to_thread(stand_in(conversation).execute, "python plot.py")
        assert result.exit_code == 1 and "quota exceeded" in result.output

        provider.fail = False
        assert await m.ensure_sandbox(conversation) is False  # not a replacement, a retry
        await conversation.starting
        assert conversation.sandbox.id == "sandbox-0"
        assert stand_in(conversation).id == "sandbox-0"  # a new agent, on a new stand-in

    run(scenario())


def test_a_conversation_closed_while_starting_does_not_keep_its_sandbox():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider, max_sandboxes=1, wait_seconds=0.05)
        conversation = await m.get_or_create("x")
        await provider.wait_entered()
        await m.close("x")
        provider.release()
        await conversation.starting
        assert provider.destroyed == provider.created and len(provider.created) == 1
        assert (await started(m, "y")).sandbox is not None  # its slot was freed

    run(scenario())


def test_shutdown_also_deletes_a_sandbox_still_starting():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider)
        await m.get_or_create("late")
        await provider.wait_entered()
        closing = asyncio.create_task(m.close_all())
        await asyncio.sleep(0)
        provider.release()
        await closing
        assert provider.destroyed == provider.created and len(provider.created) == 1

    run(scenario())


# --- the cap on sandboxes ----------------------------------------------------------


def test_a_full_house_leaves_the_new_conversation_without_a_sandbox():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=1, wait_seconds=0.05)
        await started(m, "first")
        second = await started(m, "second")  # answers anyway; only code steps fail
        assert second.sandbox is None and len(provider.created) == 1
        result = await asyncio.to_thread(stand_in(second).execute, "ls")
        assert "all 1 sandboxes are in use" in result.output

    run(scenario())


def test_a_waiting_start_gets_the_slot_a_closed_conversation_frees():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=1, wait_seconds=5)
        await started(m, "first")
        second = await m.get_or_create("second")
        await asyncio.sleep(0.05)
        assert not second.starting.done()
        await m.close("first")
        await asyncio.wait_for(second.starting, 1)
        assert second.sandbox.id == "sandbox-1"

    run(scenario())


def test_a_failed_start_gives_its_slot_back():
    async def scenario():
        provider = GatedProvider(fail=True, open=True)
        m = manager(provider, max_sandboxes=1, wait_seconds=0.05)
        assert (await started(m, "x")).sandbox is None
        provider.fail = False
        assert (await started(m, "y")).sandbox is not None

    run(scenario())


def test_no_limit_by_default():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider)
        for n in range(5):
            await started(m, f"c{n}")
        assert len(provider.created) == 5

    run(scenario())


# --- replacing a dead sandbox --------------------------------------------------------


def test_a_recently_used_sandbox_is_not_checked():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, check_after_seconds=60)
        conversation = await started(m, "x")
        clock.now = 59
        assert await m.ensure_sandbox(conversation) is False
        assert provider.checks == 0

    run(scenario())


def test_an_idle_sandbox_that_answers_is_kept():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, check_after_seconds=60)
        conversation = await started(m, "x")
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
        conversation = await started(m, "x")
        old = conversation.sandbox
        clock.now, provider.alive = 120, False

        assert await m.ensure_sandbox(conversation) is True
        assert provider.destroyed == [old] and conversation.sandbox is None
        await conversation.starting
        assert conversation.sandbox.id == "sandbox-1"
        assert stand_in(conversation).id == "sandbox-1"  # the agent was rebuilt on it
        other = await started(m, "other")  # still holds its one slot
        assert other.sandbox is None

    run(scenario())


def test_if_no_replacement_starts_its_slot_is_freed():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, check_after_seconds=60, max_sandboxes=1, wait_seconds=0.05)
        conversation = await started(m, "x")
        clock.now, provider.alive, provider.fail = 120, False, True

        assert await m.ensure_sandbox(conversation) is True
        await conversation.starting
        assert conversation.sandbox is None
        provider.fail = False
        assert (await started(m, "y")).sandbox is not None

    run(scenario())


# --- the warm pool -----------------------------------------------------------------


async def warmed(m: ConversationManager) -> None:
    """Wait for warm sandboxes being started to be ready."""
    await asyncio.wait_for(asyncio.gather(*list(m._warming)), 2)


def test_startup_fills_the_pool():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, warm_sandboxes=2)
        m.start()
        await warmed(m)
        assert len(provider.created) == 2 and m.live_sandboxes() == []

    run(scenario())


def test_no_pool_by_default():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider)
        m.start()
        await asyncio.sleep(0.05)
        assert provider.created == []

    run(scenario())


def test_a_new_conversation_takes_a_warm_sandbox_and_the_pool_refills():
    async def scenario():
        provider = GatedProvider(open=True)
        files_for = []

        async def copy_files(sandbox):
            files_for.append(sandbox.id)

        m = manager(provider, warm_sandboxes=1, on_sandbox_ready=copy_files)
        m.start()
        await warmed(m)
        provider.gate.clear()  # from now on, starting a sandbox hangs

        conversation = await started(m, "x")  # done at once: it was warm
        assert conversation.sandbox.id == "sandbox-0"
        assert files_for == ["sandbox-0"]  # files are copied at hand-out, not before
        await provider.wait_entered(2)  # the refill started the next one
        provider.release()
        await warmed(m)
        assert len(provider.created) == 2

    run(scenario())


def test_warm_sandboxes_count_toward_the_cap():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, warm_sandboxes=1, max_sandboxes=1, wait_seconds=0.05)
        m.start()
        await warmed(m)
        await started(m, "a")  # takes the warm one; no slot left to refill
        await asyncio.sleep(0.05)
        assert len(provider.created) == 1
        assert (await started(m, "b")).sandbox is None

    run(scenario())


def test_a_waiting_start_gets_the_freed_slot_before_the_pool():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, warm_sandboxes=1, max_sandboxes=1, wait_seconds=5)
        m.start()
        await warmed(m)
        await started(m, "a")
        b = await m.get_or_create("b")
        await asyncio.sleep(0.05)
        await m.close("a")
        await asyncio.wait_for(b.starting, 1)
        assert b.sandbox.id == "sandbox-1" and len(provider.created) == 2  # no third, warm one

    run(scenario())


def test_a_warm_sandbox_that_died_is_dropped_for_a_new_one():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, warm_sandboxes=1, check_after_seconds=60)
        m.start()
        await warmed(m)
        dead = provider.created[0]
        clock.now, provider.alive = 600, False  # it sat in the pool, then stopped

        conversation = await started(m, "x")
        assert dead in provider.destroyed and conversation.sandbox is not dead

    run(scenario())


def test_a_pool_that_fails_to_fill_is_retried_by_the_reaper():
    async def scenario():
        provider = GatedProvider(open=True, fail=True)
        m = manager(provider, warm_sandboxes=1)
        m.start()
        await warmed(m)
        assert provider.created == [] and m._in_use == 0  # the slot came back

        provider.fail = False
        await m.reap_idle()
        await warmed(m)
        assert len(provider.created) == 1

    run(scenario())


def test_shutdown_deletes_warm_sandboxes_too():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, warm_sandboxes=2)
        m.start()
        await warmed(m)
        await started(m, "x")
        await m.close_all()
        assert sorted(s.id for s in provider.destroyed) == sorted(s.id for s in provider.created)
        assert len(provider.created) == 3  # two warm, one refill after the hand-out

    run(scenario())


def test_more_warm_than_allowed_is_refused():
    with pytest.raises(ValueError):
        manager(GatedProvider(), warm_sandboxes=3, max_sandboxes=2)


# --- status ------------------------------------------------------------------------


def test_status_counts_busy_idle_and_starting():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=5)
        await started(m, "idle")
        busy = await started(m, "busy")
        provider.gate.clear()
        starting = await m.get_or_create("starting")
        await provider.wait_entered(3)
        async with busy.lock:  # busy answering a message
            s = m.status()
        assert s == {
            "enabled": True, "provider": "gated", "max": 5, "in_use": 3,
            "busy": 1, "idle": 1, "starting": 1, "warm": 0, "warming": 0, "waiting": 0,
            "idle_minutes": 15.0, "max_per_account": None,
            "accounts": {"test_user": {"sandboxes": 3, "busy": 1, "idle": 1, "starting": 1}},
            "memory_used_gb": 3, "max_memory_gb": None, "bigger": 0,
        }
        provider.release()
        await starting.starting

    run(scenario())


def test_status_counts_the_warm_pool():
    async def scenario():
        provider = GatedProvider()
        m = manager(provider, warm_sandboxes=2)
        m.start()
        await provider.wait_entered(2)
        assert (m.status()["warming"], m.status()["in_use"]) == (2, 2)
        provider.release()
        await warmed(m)
        assert (m.status()["warm"], m.status()["warming"], m.status()["in_use"]) == (2, 0, 2)

    run(scenario())


def test_status_without_sandboxes():
    s = ConversationManager(lambda sandbox, checkpointer: {}, None, idle_seconds=60).status()
    assert s["enabled"] is False and s["in_use"] == 0 and s["provider"] is None



# --- cleaning up after a crashed run -------------------------------------------------


class ListingProvider(GatedProvider):
    """A provider whose account already has some labeled sandboxes."""

    def __init__(self, existing, labels, fail_find=False, fail_delete=()):
        super().__init__(open=True)
        self.labels = labels
        self.existing, self.fail_find, self.fail_delete = existing, fail_find, set(fail_delete)
        self.queries, self.deleted_found = [], []

    def find(self, labels):
        self.queries.append(labels)
        if self.fail_find:
            raise RuntimeError("API down")
        return [f for f in self.existing if all(f.labels.get(k) == v for k, v in labels.items())]

    def delete_found(self, found):
        if found.id in self.fail_delete:
            raise RuntimeError("not allowed")
        self.deleted_found.append(found.id)


LABELS = {"app": "aaas-ithome", "env": "dev", "role": "api", "server": "now"}


def found(id, **labels):
    return FoundSandbox(id=id, labels={**LABELS, **labels})


def test_startup_deletes_what_an_earlier_run_left_behind():
    async def scenario():
        provider = ListingProvider(
            [found("crashed-1", server="before"), found("crashed-2", server="before"),
             found("mine", server="now"), found("other-env", env="prod", server="x"),
             FoundSandbox(id="an-eval", labels={"app": "aaas-ithome", "env": "dev"})],
            LABELS,
        )
        m = manager(provider)
        m.start()
        assert await asyncio.wait_for(m._cleanup, 1) == ["crashed-1", "crashed-2"]
        assert provider.queries == [{"app": "aaas-ithome", "env": "dev", "role": "api"}]
        assert provider.deleted_found == ["crashed-1", "crashed-2"]

    run(scenario())


def test_no_server_label_means_no_cleanup():
    async def scenario():
        provider = ListingProvider([found("x", server="before")], {"app": "aaas-ithome"})
        m = manager(provider)
        m.start()
        assert m._cleanup is None and provider.queries == []
        assert await m.clean_up_leftovers() == []

    run(scenario())


def test_cleanup_failures_never_stop_the_server():
    async def scenario():
        down = ListingProvider([], LABELS, fail_find=True)
        assert await manager(down).clean_up_leftovers() == []

        stuck = ListingProvider([found("a", server="old"), found("b", server="old")], LABELS, fail_delete=["a"])
        assert await manager(stuck).clean_up_leftovers() == ["b"]  # one failure, the rest still go

    run(scenario())


def test_shutdown_waits_for_the_cleanup():
    async def scenario():
        provider = ListingProvider([found("crashed", server="before")], LABELS)
        m = manager(provider)
        m.start()
        await m.close_all()
        assert provider.deleted_found == ["crashed"]

    run(scenario())


# --- accounts ------------------------------------------------------------------------


def test_a_conversation_belongs_to_its_account():
    async def scenario():
        from api import NotYourConversation

        m = manager(GatedProvider(open=True))
        mine = await m.get_or_create("x", "alice")
        assert mine.account == "alice"
        with pytest.raises(NotYourConversation):
            await m.get_or_create("x", "bob")
        assert (await m.get_or_create("x", "alice")) is mine
        await mine.starting

    run(scenario())


def test_an_account_at_its_limit_waits_for_its_own_sandboxes():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_per_account=1, wait_seconds=0.05)
        first = await m.get_or_create("a1", "alice")
        await first.starting
        second = await m.get_or_create("a2", "alice")
        await second.starting
        assert first.sandbox is not None and second.sandbox is None
        result = await asyncio.to_thread(stand_in(second).execute, "ls")
        assert "this account already uses 1 sandboxes" in result.output
        bob = await m.get_or_create("b1", "bob")  # other accounts are unaffected
        await bob.starting
        assert bob.sandbox is not None

    run(scenario())


async def started_for(m, conversation_id, account):
    conversation = await m.get_or_create(conversation_id, account)
    await asyncio.wait_for(asyncio.shield(conversation.starting), 2)
    return conversation


def test_an_account_without_a_sandbox_gets_one_from_an_account_with_several():
    async def scenario():
        clock, provider = Clock(), GatedProvider(open=True)
        m = manager(provider, clock=clock, max_sandboxes=2, wait_seconds=0.05)
        older = await started_for(m, "a1", "alice")
        clock.now = 10
        newer = await started_for(m, "a2", "alice")
        clock.now = 20

        bob = await started_for(m, "b1", "bob")
        assert bob.sandbox is not None
        assert m.get("a1") is None and older.sandbox is None  # alice's longest-idle one went
        assert m.get("a2") is newer and newer.sandbox is not None
        assert m.status()["accounts"]["alice"]["sandboxes"] == 1
        assert m.status()["accounts"]["bob"]["sandboxes"] == 1

    run(scenario())


def test_nothing_is_taken_from_an_account_with_only_one():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=2, wait_seconds=0.05)
        await started_for(m, "a1", "alice")
        await started_for(m, "c1", "carol")
        bob = await started_for(m, "b1", "bob")
        assert bob.sandbox is None and m.get("a1").sandbox and m.get("c1").sandbox

    run(scenario())


def test_a_busy_conversation_is_never_taken():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=2, wait_seconds=0.05)
        a1 = await started_for(m, "a1", "alice")
        a2 = await started_for(m, "a2", "alice")
        async with a1.lock, a2.lock:  # both answering
            bob = await started_for(m, "b1", "bob")
        assert bob.sandbox is None and a1.sandbox and a2.sandbox

    run(scenario())


def test_an_account_that_already_has_one_does_not_take_from_others():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, max_sandboxes=3, wait_seconds=0.05)
        await started_for(m, "a1", "alice")
        await started_for(m, "a2", "alice")
        await started_for(m, "b1", "bob")
        second_for_bob = await started_for(m, "b2", "bob")  # bob has one: no guarantee left
        assert second_for_bob.sandbox is None and m.get("a1").sandbox and m.get("a2").sandbox

    run(scenario())


# --- memory budget and bigger sandboxes ----------------------------------------------


class SizingProvider(GatedProvider):
    """Also makes bigger sandboxes, which can be made to fail."""

    def __init__(self, fail_bigger=False):
        super().__init__(open=True)
        self.fail_bigger, self.bigger = fail_bigger, []

    def create_bigger(self, memory_gb, cpu):
        if self.fail_bigger:
            raise RuntimeError("Total memory limit exceeded")
        sandbox = Sandbox(f"big-{len(self.bigger)}")
        sandbox.memory_gb, sandbox.cpu = memory_gb, cpu
        self.bigger.append(sandbox)
        return sandbox


def test_the_memory_budget_limits_how_many_start():
    async def scenario():
        m = manager(GatedProvider(open=True), max_memory_gb=2, wait_seconds=0.05)
        await started(m, "a")
        await started(m, "b")
        third = await started(m, "c")
        assert third.sandbox is None and m.status()["memory_used_gb"] == 2
        result = await asyncio.to_thread(stand_in(third).execute, "ls")
        assert "all 2 GB of sandbox memory is in use" in result.output

    run(scenario())


def test_an_upgrade_moves_the_work_and_swaps_the_sandbox():
    async def scenario():
        copied = []
        provider = SizingProvider()
        m = manager(provider, max_memory_gb=10, bigger_sandbox=(4, 2),
                    copy_files=lambda old, new: copied.append((old.id, new.id)))
        conversation = await started(m, "x")
        small = conversation.sandbox

        upgrade = await m.upgrade(conversation)
        assert upgrade.sandbox is conversation.sandbox and conversation.sandbox.id == "sandbox-big-0"
        assert (conversation.sandbox.memory_gb, conversation.sandbox.cpu) == (4, 2)
        assert "4 GB memory, 2 vCPU" in upgrade.note
        assert copied == [("sandbox-0", "sandbox-big-0")] and provider.destroyed == [small]
        status = m.status()
        assert (status["memory_used_gb"], status["in_use"], status["bigger"]) == (4, 1, 1)

        await m.close("x")  # the whole 4 GB comes back
        assert m.status()["memory_used_gb"] == 0

    run(scenario())


def test_an_upgrade_needs_room_in_the_budget():
    async def scenario():
        provider = SizingProvider()
        m = manager(provider, max_memory_gb=4, bigger_sandbox=(4, 2), copy_files=lambda o, n: None)
        conversation = await started(m, "x")  # 1 GB in use: 1 + 4 does not fit in 4
        upgrade = await m.upgrade(conversation)
        assert upgrade.sandbox is None and "No room for a 4 GB sandbox" in upgrade.note
        assert provider.bigger == [] and m.status()["memory_used_gb"] == 1

    run(scenario())


def test_no_second_upgrade_and_none_when_turned_off():
    async def scenario():
        m = manager(SizingProvider(), bigger_sandbox=(4, 2), copy_files=lambda o, n: None)
        conversation = await started(m, "x")
        await m.upgrade(conversation)
        again = await m.upgrade(conversation)
        assert again.sandbox is None and "already has 4 GB" in again.note

        off = manager(SizingProvider())
        assert "No bigger sandbox is available" in (await off.upgrade(await started(off, "y"))).note

    run(scenario())


def test_a_failed_upgrade_keeps_the_old_sandbox_and_the_budget():
    async def scenario():
        provider = SizingProvider(fail_bigger=True)
        m = manager(provider, max_memory_gb=10, bigger_sandbox=(4, 2), copy_files=lambda o, n: None)
        conversation = await started(m, "x")
        upgrade = await m.upgrade(conversation)
        assert upgrade.sandbox is None and "Total memory limit exceeded" in upgrade.note
        assert conversation.sandbox.id == "sandbox-0" and m.status()["memory_used_gb"] == 1

        def broken_copy(old, new):
            raise RuntimeError("tar failed")

        provider.fail_bigger = False
        m._copy_files = broken_copy
        upgrade = await m.upgrade(conversation)
        assert "could not be moved" in upgrade.note and provider.destroyed == provider.bigger
        assert conversation.sandbox.id == "sandbox-0" and m.status()["memory_used_gb"] == 1

    run(scenario())


def test_the_stand_in_reaches_the_upgrade_from_its_worker_thread():
    async def scenario():
        m = manager(SizingProvider(), bigger_sandbox=(4, 2), copy_files=lambda o, n: None)
        conversation = await started(m, "x")
        heard = listen(conversation)
        upgrade = await asyncio.to_thread(stand_in(conversation).on_out_of_memory, "needs 3.2 GB at once")
        assert upgrade.sandbox is conversation.sandbox and conversation.memory_gb == 4
        assert heard[0] == ("upgrade_started", heard[0][1]) and heard[0][1]["reason"] == "needs 3.2 GB at once"

    run(scenario())


# --- sandbox events ------------------------------------------------------------------


def listen(conversation):
    heard = []
    conversation.on_event = lambda event, fields: heard.append((event, fields))
    return heard


def test_a_move_reports_each_stage_in_order():
    async def scenario():
        m = manager(SizingProvider(), max_memory_gb=10, bigger_sandbox=(4, 2), copy_files=lambda old, new: 12345)
        conversation = await started(m, "x")
        heard = listen(conversation)
        await m.upgrade(conversation)
        assert [e for e, _ in heard] == [
            "upgrade_started", "bigger_created", "packages_installed", "files_copied", "switched", "old_deleted"]
        fields = dict(heard)
        assert (fields["upgrade_started"]["from_gb"], fields["upgrade_started"]["to_gb"]) == (1, 4)
        assert fields["upgrade_started"]["memory_used_gb"] == 5  # both alive for a moment
        assert fields["files_copied"]["bytes"] == 12345
        assert fields["switched"]["sandbox"] == "sandbox-big-0"
        assert fields["old_deleted"]["memory_used_gb"] == 4
        assert all("seconds" in fields[e] for e in ("bigger_created", "packages_installed", "files_copied"))

    run(scenario())


def test_a_refused_move_reports_why():
    async def scenario():
        m = manager(SizingProvider(), max_memory_gb=4, bigger_sandbox=(4, 2), copy_files=lambda o, n: 0)
        conversation = await started(m, "x")
        heard = listen(conversation)
        await m.upgrade(conversation)
        [(event, fields)] = heard
        assert event == "upgrade_failed" and "No room for a 4 GB sandbox" in fields["reason"]

    run(scenario())


def test_ready_says_whether_the_sandbox_was_warm_or_new():
    async def scenario():
        provider = GatedProvider(open=True)
        m = manager(provider, warm_sandboxes=1)
        m.start()
        await warmed(m)
        heard = {}
        for name in ("first", "second"):
            conversation = await m.get_or_create(name)
            heard[name] = listen(conversation)
            await conversation.starting
        await warmed(m)
        assert heard["first"][0][0] == "sandbox_ready" and heard["first"][0][1]["how"] == "warm"
        assert heard["second"][0][1]["how"] in ("warm", "new")  # the refill may or may not be ready
        assert heard["first"][0][1]["memory_gb"] == 1

    run(scenario())
