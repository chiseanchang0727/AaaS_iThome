"""Config for the HTTP API the frontend talks to."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SandboxSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_gb: int = Field(ge=1)
    cpu: int = Field(default=1, ge=1)


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_dir: Path
    """Where files the agent made are kept after each turn, one folder per
    conversation. Resolved against the working directory."""

    uploads_dir: Path
    """Where uploaded files and the dataset registry live."""

    history_dir: Path
    """Where each conversation's history is written, one JSONL file per
    conversation (api/history.py)."""

    sandbox_idle_minutes: float
    """A conversation's sandbox is deleted after this long without a message."""

    max_sandboxes: int | None = Field(default=10, ge=1)
    """At most this many sandboxes at once (they are billed). None: no limit."""

    sandbox_wait_seconds: float = Field(default=30, ge=0)
    """How long a new conversation waits for a free sandbox before it is refused."""

    max_sandboxes_per_account: int | None = Field(default=3, ge=1)
    """At most this many sandboxes per account. None: no per-account limit."""

    max_memory_gb: int | None = Field(default=None, ge=1)
    """Memory of all sandboxes together, in GB: the provider account's limit
    (Daytona's tier). A bigger sandbox only starts when it fits. None: no budget."""

    bigger_sandbox: "SandboxSize | None" = None
    """Size to move a conversation to when a command runs out of memory. None: never."""

    default_account: str = "test_user"
    """Whose requests without an X-Account header are (there is no login yet)."""

    warm_sandboxes: int = Field(default=0, ge=0)
    """Sandboxes kept started and prepared ahead of time, so a new conversation
    skips the ~12s start. They count toward max_sandboxes and are billed."""

    @model_validator(mode="after")
    def _warm_fits(self) -> "ServerConfig":
        if self.max_sandboxes is not None and self.warm_sandboxes > self.max_sandboxes:
            raise ValueError("warm_sandboxes cannot be more than max_sandboxes")
        return self

    max_upload_mb: float
    """Uploads larger than this are refused."""
