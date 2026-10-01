"""Config for the sandbox the agent runs code in."""

from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    """`none` for no code execution, or a name in `sandboxes.PROVIDERS`.

    Checked by `sandboxes.get_provider`, not here: config does not import the
    providers, so adding one never touches this file.
    """

    options: dict[str, Any] = {}
    """The selected provider's own settings, validated by its `Options` class."""

    packages: list[str] = []
    """pip packages installed in each new sandbox before the agent starts, for
    what its image lacks. Needs network access in the sandbox."""

    data_dir: PurePosixPath
    """Where export_query writes query results, inside the sandbox. Home
    directories differ between providers, so check this when switching."""

    output_dir: PurePosixPath
    """Where the agent saves charts and reports; downloaded after the run."""

    labels: dict[str, str] = {}
    """Put on every sandbox, so this app's sandboxes can be told apart from
    others on the account. The API adds `role` and `server` (see api/main.py)
    and, at startup, deletes sandboxes a crashed run of it left behind."""

    export_max_rows: int
    """Row cap for export_query. Higher than database.max_rows: the rows go to
    a file in the sandbox, not into the model's context."""

    @field_validator("data_dir", "output_dir")
    @classmethod
    def _absolute(cls, path: PurePosixPath) -> PurePosixPath:
        if not path.is_absolute():
            raise ValueError(f"must be an absolute path inside the sandbox: {path}")
        return path
