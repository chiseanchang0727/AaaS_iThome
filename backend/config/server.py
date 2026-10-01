"""Config for the HTTP API the frontend talks to."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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

    max_upload_mb: float
    """Uploads larger than this are refused."""
