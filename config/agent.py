"""Config for the agent itself."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    """`provider:model`, as `init_chat_model` reads it."""

    skills_dir: Path
    """Directory of `<skill-name>/SKILL.md` folders, mounted at /skills/."""

    max_result_tokens: int
    """Largest tool result, in approximate tokens, that enters the context."""
