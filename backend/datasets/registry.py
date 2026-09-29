"""What has been uploaded: one record per dataset, kept in a JSON file.

The records are also what the agent reads (via list_datasets) to learn what
exists, so they carry the columns and their types.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
"""A dataset name: also its table name, so a safe Postgres identifier."""

RESERVED = {"videos", "datasets", "user", "table", "select", "order", "group"}


class Column(BaseModel):
    name: str
    type: str
    """The Postgres type for a table; the polars type for a file."""


class Dataset(BaseModel):
    name: str
    kind: Literal["table", "file"]
    rows: int
    columns: list[Column]
    source: str
    """The file the user uploaded."""
    created_at: datetime


def check_name(name: str) -> str | None:
    """Why `name` can't be a dataset name, or None if it can."""
    if not NAME.match(name):
        return "use lowercase letters, digits and _, starting with a letter (max 63)"
    if name in RESERVED or name.startswith("pg_"):
        return f"{name!r} is reserved"
    return None


class Registry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def list(self) -> list[Dataset]:
        if not self.path.exists():
            return []
        return [Dataset.model_validate(d) for d in json.loads(self.path.read_text())]

    def get(self, name: str) -> Dataset | None:
        return next((d for d in self.list() if d.name == name), None)

    def add(self, dataset: Dataset) -> None:
        with self._lock:
            datasets = [d for d in self.list() if d.name != dataset.name] + [dataset]
            self._write(datasets)

    def remove(self, name: str) -> bool:
        with self._lock:
            datasets = self.list()
            kept = [d for d in datasets if d.name != name]
            self._write(kept)
            return len(kept) != len(datasets)

    def _write(self, datasets: list[Dataset]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([d.model_dump(mode="json") for d in datasets], indent=2))
        tmp.replace(self.path)  # atomic: a crash never leaves half a registry


def now() -> datetime:
    return datetime.now(UTC)
