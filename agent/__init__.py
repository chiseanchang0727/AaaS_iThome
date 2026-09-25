"""The deep agent, wired up from config.

    from agent import build_agent

    agent = build_agent()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
"""

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from config import cfg

from .middleware import SkillEnforcerMiddleware
from .tools import query_database

__all__ = ["build_agent", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = """\
You are a data analyst for a YouTube trending-videos dataset stored in PostgreSQL.
Answer questions by querying the database; never guess a number. Keep the final
answer short: the result, then one line on how it was computed.
"""

def build_agent(*, require_sql_skill: bool = True):
    """A deep agent with the database tool and the skills in `cfg.agent.skills_dir`.

    Files the agent writes go to in-memory state and vanish with the run;
    /skills/ is backed by the real directory and is read-only to the agent.
    """
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": FilesystemBackend(root_dir=cfg.agent.skills_dir, virtual_mode=True),
        },
    )
    # Appended last: after_model hooks run in reverse order, so this checks
    # the model's calls before any middleware added ahead of it.
    middleware = (
        [SkillEnforcerMiddleware(cfg.agent.skills_dir, target_skills="query_database")]
        if require_sql_skill
        else []
    )

    return create_deep_agent(
        model=cfg.agent.model,
        tools=[query_database],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        skills=["/skills/"],
        permissions=[FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")],
        middleware=middleware,
    )
