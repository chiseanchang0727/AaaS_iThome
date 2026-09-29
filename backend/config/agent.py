"""Config for the agent itself."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class SkillRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "jev-1.13.0"
    """TypeSafe model id. Pinned: a new version can move the probabilities."""

    multi: bool = False
    """Also name a second or third skill when a second request confirms it."""


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    """`provider:model`, as `init_chat_model` reads it."""

    skills_dir: Path
    """Directory of `<skill-name>/SKILL.md` folders, mounted at /skills/."""

    max_result_tokens: int
    """Largest tool result, in approximate tokens, that enters the context."""

    skill_router: SkillRouterConfig | None = None
    """Name the relevant skills in the system prompt, per user message. Off if
    absent. Needs $TYPESAFE_API_KEY."""
