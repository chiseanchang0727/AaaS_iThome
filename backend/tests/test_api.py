"""The HTTP API with a scripted agent and local temp-dir sandboxes.

No model, no Daytona, no database: the agent is a fake that replays updates
and can drop files into its sandbox as if it had made them.
"""

import asyncio
import json
import threading
from pathlib import Path, PurePosixPath

import pytest
from deepagents.backends.local_shell import LocalShellBackend
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from api import ConversationManager, HistoryStore, create_app
from api.events import RESULT_CHARS
from sandboxes import SandboxProvider

OUTPUT_DIR = PurePosixPath("/outputs")


# --- fakes ---------------------------------------------------------------------


class TempDirProvider(SandboxProvider):
    """Each sandbox is a fresh temp dir. Records creates and destroys."""

    name = "tempdir"

    def __init__(self, root: Path, fail_prepare: bool = False):
        super().__init__()
        self.root, self.fail_prepare = root, fail_prepare
        self.created, self.destroyed = [], []
        self._lock, self._next = threading.Lock(), 0

    def create(self):
        # Sandboxes start side by side (in threads), so number them under a lock.
        with self._lock:
            path = self.root / f"sandbox-{self._next}"
            self._next += 1
        path.mkdir()
        sandbox = LocalShellBackend(root_dir=path, virtual_mode=True)
        self.created.append(sandbox)
        return sandbox

    def prepare(self, sandbox):
        if self.fail_prepare:
            raise RuntimeError("pip install failed")

    def destroy(self, sandbox):
        self.destroyed.append(sandbox)


def call(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


class ScriptedAgent:
    """Replays `turns` (one list of updates per user message) and records calls.

    An update may be a callable taking the sandbox, to act on it mid-turn
    (e.g. write an output file), or an exception to raise.
    """

    def __init__(self, sandbox, turns):
        self.sandbox, self.turns, self.calls = sandbox, list(turns), []

    async def astream(self, input, config, stream_mode):
        self.calls.append((input, config))
        for update in self.turns.pop(0) if self.turns else [{"model": {"messages": [AIMessage("ok")]}}]:
            if isinstance(update, Exception):
                raise update
            if callable(update):
                # In a worker thread, like the real agent's tools: the sandbox
                # may still be starting, and waiting for it must not block the loop.
                await asyncio.to_thread(update, self.sandbox)
                continue
            yield update


def write_output(name: str, content: bytes):
    return lambda sandbox: sandbox.upload_files([(f"/outputs/{name}", content)])


class Harness:
    def __init__(self, tmp_path: Path, turns=(), provider=None, idle_seconds=900, history=None, **manager_options):
        self.now = 0.0
        self.provider = provider if provider is not None else TempDirProvider(tmp_path)
        self.agents: list[ScriptedAgent] = []
        self.turns = list(turns)
        self.shutdown_called = False

        def build(sandbox, checkpointer):
            self.checkpointer = checkpointer
            agent = ScriptedAgent(sandbox, self.turns)
            self.agents.append(agent)
            return agent

        async def on_shutdown():
            self.shutdown_called = True

        self.manager = ConversationManager(
            build, self.provider, idle_seconds=idle_seconds, clock=lambda: self.now, **manager_options
        )
        self.artifacts = tmp_path / "artifacts"
        self.app = create_app(
            self.manager, self.artifacts, OUTPUT_DIR, reap_every_seconds=3600, on_shutdown=on_shutdown,
            history=history,
        )


def events_of(response) -> list[dict]:
    """Parse a Server-Sent Events body into its data payloads."""
    events, kind = [], None
    for line in response.iter_lines():
        if line.startswith("event: "):
            kind = line.removeprefix("event: ")
        elif line.startswith("data: "):
            event = json.loads(line.removeprefix("data: "))
            assert event["type"] == kind
            events.append(event)
    return events


def chat(client, message="hi", thread_id=None) -> list[dict]:
    body = {"message": message} | ({"thread_id": thread_id} if thread_id else {})
    with client.stream("POST", "/api/chat", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return events_of(response)


def types(events) -> list[str]:
    return [e["type"] for e in events]


TOOL_TURN = [
    {"model": {"messages": [AIMessage("Let me check.", tool_calls=[call("query_database", {"sql": "SELECT 1"}, "c1")])]}},
    {"tools": {"messages": [ToolMessage('[{"n": 1}]', tool_call_id="c1", name="query_database")]}},
    {"model": {"messages": [AIMessage("There is **1** row.")]}},
]


# --- chat ----------------------------------------------------------------------


def test_health(tmp_path):
    with TestClient(Harness(tmp_path).app) as client:
        assert client.get("/api/health").json() == {"ok": True}


def test_a_turn_streams_thread_steps_answer_done(tmp_path):
    with TestClient(Harness(tmp_path, [TOOL_TURN]).app) as client:
        events = chat(client)
    assert types(events) == ["thread", "thinking", "tool_call", "tool_result", "answer", "done"]
    assert events[2] == {"type": "tool_call", "id": "c1", "name": "query_database", "args": {"sql": "SELECT 1"}}
    assert events[3]["content"] == '[{"n": 1}]' and events[3]["error"] is False
    assert events[4]["text"] == "There is **1** row."


def test_a_new_conversation_gets_an_id(tmp_path):
    with TestClient(Harness(tmp_path).app) as client:
        thread_id = chat(client)[0]["thread_id"]
    assert len(thread_id) == 32


def test_follow_ups_reuse_the_agent_and_sandbox(tmp_path):
    harness = Harness(tmp_path)
    with TestClient(harness.app) as client:
        thread_id = chat(client, "first")[0]["thread_id"]
        assert chat(client, "second", thread_id)[0]["thread_id"] == thread_id
    assert len(harness.agents) == 1 and len(harness.provider.created) == 1
    configs = [config["configurable"]["thread_id"] for _, config in harness.agents[0].calls]
    assert configs == [thread_id, thread_id]
    assert [i["messages"][0]["content"] for i, _ in harness.agents[0].calls] == ["first", "second"]


def test_separate_conversations_get_separate_sandboxes(tmp_path):
    harness = Harness(tmp_path)
    with TestClient(harness.app) as client:
        chat(client)
        chat(client)
    assert len(harness.provider.created) == 2


def test_a_tool_error_is_marked(tmp_path):
    turn = [{"tools": {"messages": [ToolMessage("ERROR: bad", tool_call_id="c1", name="query_database", status="error")]}}]
    with TestClient(Harness(tmp_path, [turn]).app) as client:
        result = next(e for e in chat(client) if e["type"] == "tool_result")
    assert result["error"] is True


def test_long_tool_results_are_clipped_for_display(tmp_path):
    turn = [{"tools": {"messages": [ToolMessage("x" * 10_000, tool_call_id="c1", name="execute")]}}]
    with TestClient(Harness(tmp_path, [turn]).app) as client:
        result = next(e for e in chat(client) if e["type"] == "tool_result")
    assert len(result["content"]) < RESULT_CHARS + 50 and "10,000 characters" in result["content"]


def test_hitting_the_step_limit_is_an_error(tmp_path):
    with TestClient(Harness(tmp_path, [[GraphRecursionError()]]).app) as client:
        events = chat(client)
    assert types(events) == ["thread", "error", "done"]
    assert "steps" in events[1]["message"]


def test_an_agent_crash_is_an_error_not_a_broken_stream(tmp_path):
    with TestClient(Harness(tmp_path, [[RuntimeError("model unavailable")]]).app) as client:
        events = chat(client)
    assert types(events) == ["thread", "error", "done"]
    assert events[1]["message"] == "model unavailable"


def test_a_sandbox_that_fails_to_start_still_answers_and_is_deleted(tmp_path):
    harness = Harness(tmp_path, provider=TempDirProvider(tmp_path, fail_prepare=True))
    with TestClient(harness.app) as client:
        events = chat(client)
    assert types(events) == ["thread", "answer", "done"]  # SQL-only turns don't need it
    assert harness.provider.destroyed == harness.provider.created


def test_a_code_step_without_a_sandbox_gets_the_reason(tmp_path):
    seen = []
    turn = [lambda sandbox: seen.append(sandbox.execute("python plot.py")),
            {"model": {"messages": [AIMessage("Here is what I found without a chart.")]}}]
    harness = Harness(tmp_path, [turn], provider=TempDirProvider(tmp_path, fail_prepare=True))
    with TestClient(harness.app) as client:
        events = chat(client)
    assert types(events) == ["thread", "answer", "done"]
    assert seen[0].exit_code == 1 and "pip install failed" in seen[0].output


def test_a_busy_conversation_refuses_a_second_message(tmp_path):
    harness = Harness(tmp_path)

    class HeldLock:
        def locked(self):
            return True

    with TestClient(harness.app) as client:
        thread_id = chat(client)[0]["thread_id"]
        harness.manager.get(thread_id).lock = HeldLock()
        response = client.post("/api/chat", json={"message": "again", "thread_id": thread_id})
    assert response.status_code == 409


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "hi", "thread_id": "../etc"}, {}])
def test_bad_requests_are_rejected(tmp_path, body):
    with TestClient(Harness(tmp_path).app) as client:
        assert client.post("/api/chat", json=body).status_code == 422


# --- artifacts -----------------------------------------------------------------


def test_files_the_agent_saves_are_announced_and_served(tmp_path):
    html = b"<html><body><script>1</script>chart</body></html>"
    harness = Harness(tmp_path, [[write_output("chart.html", html), *TOOL_TURN]])
    with TestClient(harness.app) as client:
        events = chat(client)
        [artifact] = [e for e in events if e["type"] == "artifact"]
        response = client.get(artifact["url"])
    assert types(events)[-2:] == ["artifact", "done"]
    assert artifact["name"] == "chart.html" and artifact["kind"] == "html"
    assert response.status_code == 200 and response.content == html
    assert response.headers["content-type"].startswith("text/html")


def test_artifacts_are_served_sandboxed(tmp_path):
    """Agent-written HTML must not run with the app's origin."""
    harness = Harness(tmp_path, [[write_output("chart.html", b"<p>x</p>")]])
    with TestClient(harness.app) as client:
        url = next(e["url"] for e in chat(client) if e["type"] == "artifact")
        headers = client.get(url).headers
    assert headers["content-security-policy"] == "sandbox allow-scripts"
    assert headers["x-content-type-options"] == "nosniff"


def test_each_artifact_is_announced_once(tmp_path):
    harness = Harness(tmp_path, [
        [write_output("a.html", b"1")],
        [write_output("b.png", b"2")],
    ])
    with TestClient(harness.app) as client:
        thread_id = chat(client)[0]["thread_id"]
        second = chat(client, "more", thread_id)
    assert [e["name"] for e in second if e["type"] == "artifact"] == ["b.png"]


def test_artifacts_keep_subfolders(tmp_path):
    harness = Harness(tmp_path, [[write_output("report/index.html", b"<p>r</p>")]])
    with TestClient(harness.app) as client:
        artifact = next(e for e in chat(client) if e["type"] == "artifact")
        assert client.get(artifact["url"]).status_code == 200
    assert artifact["name"] == "report/index.html"


def test_no_sandbox_means_no_artifacts(tmp_path):
    harness = Harness(tmp_path, [TOOL_TURN])
    harness.manager._provider = None
    with TestClient(harness.app) as client:
        events = chat(client)
    assert "artifact" not in types(events)
    assert harness.agents[0].sandbox is None


@pytest.mark.parametrize(
    "url",
    [
        "/api/artifacts/{t}/../../../pyproject.toml",
        "/api/artifacts/{t}/%2e%2e/%2e%2e/config.yml",
        "/api/artifacts/{t}/missing.html",
        "/api/artifacts/bad$id/chart.html",
    ],
)
def test_artifact_paths_cannot_escape(tmp_path, url):
    harness = Harness(tmp_path, [[write_output("chart.html", b"x")]])
    with TestClient(harness.app) as client:
        thread_id = chat(client)[0]["thread_id"]
        assert client.get(url.format(t=thread_id)).status_code == 404


# --- lifecycle -----------------------------------------------------------------


def test_ending_a_conversation_deletes_its_sandbox(tmp_path):
    harness = Harness(tmp_path)
    with TestClient(harness.app) as client:
        thread_id = chat(client)[0]["thread_id"]
        assert client.delete(f"/api/conversations/{thread_id}").json() == {"closed": True}
        assert client.delete(f"/api/conversations/{thread_id}").json() == {"closed": False}
    assert harness.provider.destroyed == harness.provider.created


def test_idle_conversations_lose_their_sandbox_but_keep_their_memory(tmp_path):
    harness = Harness(tmp_path, idle_seconds=900)
    with TestClient(harness.app) as client:
        thread_id = chat(client)[0]["thread_id"]
        sandbox_started(client, harness, thread_id)
        harness.now = 901
        closed = client.portal.call(harness.manager.reap_idle)
        assert closed == [thread_id]
        assert len(harness.provider.destroyed) == 1

        chat(client, "back again", thread_id)
    assert len(harness.provider.created) == 2  # a fresh sandbox
    assert harness.agents[1].calls[0][1]["configurable"]["thread_id"] == thread_id
    assert harness.checkpointer is harness.manager.checkpointer  # same memory


def test_recent_conversations_are_not_reaped(tmp_path):
    harness = Harness(tmp_path, idle_seconds=900)
    with TestClient(harness.app) as client:
        chat(client)
        harness.now = 899
        assert client.portal.call(harness.manager.reap_idle) == []


def test_shutdown_deletes_every_sandbox(tmp_path):
    harness = Harness(tmp_path)
    with TestClient(harness.app) as client:
        chat(client)
        chat(client)
    assert len(harness.provider.destroyed) == 2
    assert harness.shutdown_called


# --- sandbox limits and recovery -------------------------------------------------


def sandbox_started(client, harness, thread_id):
    """Wait, on the app's event loop, for the conversation's background sandbox start."""

    async def wait():
        await harness.manager.get(thread_id).starting

    client.portal.call(wait)


class DyingProvider(TempDirProvider):
    """A TempDirProvider whose sandboxes can be made to stop answering."""

    alive = True

    def is_alive(self, sandbox):
        return self.alive


def test_a_full_house_still_answers_and_code_steps_say_why(tmp_path):
    seen = []
    code_turn = [lambda sandbox: seen.append(sandbox.execute("echo hi")),
                 {"model": {"messages": [AIMessage("ok")]}}]
    harness = Harness(tmp_path, [code_turn], max_sandboxes=1, wait_seconds=0)
    with TestClient(harness.app) as client:
        chat(client)  # takes the only sandbox
        events = chat(client)  # a second conversation
    assert types(events) == ["thread", "answer", "done"]
    assert seen[0].exit_code == 0
    assert seen[1].exit_code == 1 and "all 1 sandboxes are in use" in seen[1].output


def test_a_dead_sandbox_is_replaced_and_the_user_told(tmp_path):
    provider = DyingProvider(tmp_path)
    harness = Harness(tmp_path, provider=provider)
    with TestClient(harness.app) as client:
        thread_id = chat(client)[0]["thread_id"]
        sandbox_started(client, harness, thread_id)
        harness.now += 120
        provider.alive = False
        events = chat(client, "again", thread_id)
    assert types(events) == ["thread", "notice", "answer", "done"]
    assert "new one was started" in events[1]["message"]
    # the dead one when replaced, then the new one at shutdown
    assert len(provider.created) == 2 and provider.destroyed == provider.created
    assert len(harness.agents) == 2  # the agent was rebuilt on the new sandbox
