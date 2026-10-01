"""Config for the agent itself."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SkillRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "jev-1.13.0"
    """TypeSafe model id. Pinned: a new version can move the probabilities."""

    top_n: int = Field(default=1, ge=1)
    """Name at most this many skills per user message."""

    multi: bool = False
    """How to pick past the first skill. False: the `top_n` most likely.
    True: a second request checks places 2..top_n and keeps those it confirms."""


class ContextFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "jev-1.13.0"
    """TypeSafe model id. Pinned: a new version can move the probabilities."""

    threshold: float = Field(default=0.5, ge=0, le=1)
    """An older turn is sent when Jev's probability of yes is at least this."""

    keep_last: int = Field(default=1, ge=0)
    """Always send this many of the most recent turns, without asking Jev."""

    max_rounds: int = Field(default=3, ge=1)
    """How far to follow references back ("that category" -> the turn that
    named it -> the turn that turn relied on). 1: only the new message."""


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

    context_filter: ContextFilterConfig | None = None
    """Send only the earlier turns a new message needs (agent/context_filter.py).
    Off if absent: every turn is sent. Needs $TYPESAFE_API_KEY."""
