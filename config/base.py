"""Top-level schema for the repo-root `config.yml`. One field per section."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .data import DataConfig


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig

    @classmethod
    def load(cls, path: Path | str = "config.yml") -> "Config":
        """Read `path` and validate it."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
