"""Config for the HTTP API the frontend talks to."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


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

    max_upload_mb: float
    """Uploads larger than this are refused."""
