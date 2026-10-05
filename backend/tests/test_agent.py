"""Agent wiring that can be checked without calling a model.

The skill enforcer has its own file: tests/test_skill_enforcer.py.
"""

from agent import build_agent


def test_agent_builds_from_config():
    """Graph construction only; no model is called."""
    assert build_agent() is not None


def test_agent_builds_without_the_enforcer():
    assert build_agent(require_sql_skill=False) is not None



def test_the_stand_in_gets_a_tool_to_ask_for_a_bigger_sandbox():
    from deepagents.backends.local_shell import LocalShellBackend

    from sandboxes.lazy import LazySandbox

    def tools(agent):
        return set(agent.nodes["tools"].bound.tools_by_name)

    assert "request_bigger_sandbox" in tools(build_agent(sandbox=LazySandbox()))
    # a plain sandbox (evals, scripts) cannot move, so no tool
    assert "request_bigger_sandbox" not in tools(build_agent(sandbox=LocalShellBackend(root_dir=".", virtual_mode=True)))


def test_request_bigger_sandbox_passes_the_reason_on():
    from agent.tools import make_request_bigger_sandbox

    class Stand:
        def request_bigger(self, reason):
            return f"moved because {reason}"

    tool = make_request_bigger_sandbox(Stand())
    assert tool.name == "request_bigger_sandbox"
    assert tool.invoke({"reason": "the matrix needs 3.2 GB"}) == "moved because the matrix needs 3.2 GB"
