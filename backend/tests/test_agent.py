"""Agent wiring that can be checked without calling a model.

The skill enforcer has its own file: tests/test_skill_enforcer.py.
"""

from agent import build_agent


def test_agent_builds_from_config():
    """Graph construction only; no model is called."""
    assert build_agent() is not None


def test_agent_builds_without_the_enforcer():
    assert build_agent(require_sql_skill=False) is not None
