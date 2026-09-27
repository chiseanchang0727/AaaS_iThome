"""Datasets users upload: staged, previewed, then kept as a table or a file.

    staged = store.stage("sales.csv", fileobj)      # parse + preview, nothing kept yet
    await store.commit(staged.upload_id, "table", "sales_2031")

A table goes into Postgres through the ingest role, readable by the agent's
role. A file is stored as Parquet, typed, and copied into every sandbox at
`{data_dir}/uploads/<name>.parquet`.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

from .ingest import SUPPORTED, UploadError, create_table, drop_table, file_columns, preview, read_upload, table_columns
from .registry import Column, Dataset, Registry, check_name, now

__all__ = ["Column", "Dataset", "DatasetStore", "Registry", "Staged", "UploadError"]

_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
CHUNK = 1 << 20


@dataclass
class Staged:
    upload_id: str
    filename: str
    rows: int
    columns: list[Column]
    preview: list[dict]
    suggested_name: str


class DatasetStore:
    def __init__(
        self,
        root: Path,
        max_bytes: int,
        ingest_dsn: Callable[[], str],
        reader_role: Callable[[], str],
    ) -> None:
        self.root = root
        self.registry = Registry(root / "registry.json")
        self.staging = root / "staging"
        self.files = root / "files"
        self.max_bytes = max_bytes
        self._ingest_dsn = ingest_dsn
        self._reader_role = reader_role

    # --- upload and preview ------------------------------------------------------

    def stage(self, filename: str, fileobj: BinaryIO) -> Staged:
        """Save an upload for review and parse it. Nothing becomes a dataset yet."""
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED:
            raise UploadError(f"unsupported file type {suffix or '(none)'}; upload .csv or .parquet")
        upload_id = uuid.uuid4().hex
        self.staging.mkdir(parents=True, exist_ok=True)
        path = self.staging / f"{upload_id}{suffix}"

        size = 0
        with path.open("wb") as out:
            while chunk := fileobj.read(CHUNK):
                size += len(chunk)
                if size > self.max_bytes:
                    out.close()
                    path.unlink()
                    raise UploadError(f"file is larger than {self.max_bytes // (1 << 20)} MB")
                out.write(chunk)

        try:
            df = read_upload(path)
        except UploadError:
            path.unlink()
            raise
        (self.staging / f"{upload_id}.json").write_text(json.dumps({"filename": filename}))
        return Staged(
            upload_id=upload_id,
            filename=filename,
            rows=df.height,
            columns=table_columns(df),
            preview=preview(df),
            suggested_name=suggest_name(filename),
        )

    # --- keep, list, remove ------------------------------------------------------

    async def commit(self, upload_id: str, kind: Literal["table", "file"], name: str) -> Dataset:
        """Turn a staged upload into a dataset named `name`."""
        if problem := check_name(name):
            raise UploadError(problem)
        if self.registry.get(name) is not None:
            raise UploadError(f"a dataset named {name!r} already exists")
        path, filename = self._staged(upload_id)
        df = read_upload(path)

        if kind == "table":
            await create_table(self._ingest_dsn(), name, df, self._reader_role())
            columns = table_columns(df)
        else:
            self.files.mkdir(parents=True, exist_ok=True)
            df.write_parquet(self.file_path(name))
            columns = file_columns(df)

        dataset = Dataset(name=name, kind=kind, rows=df.height, columns=columns, source=filename, created_at=now())
        self.registry.add(dataset)
        self._discard(upload_id)
        return dataset

    def list(self) -> list[Dataset]:
        return self.registry.list()

    async def delete(self, name: str) -> bool:
        dataset = self.registry.get(name)
        if dataset is None:
            return False
        if dataset.kind == "table":
            await drop_table(self._ingest_dsn(), name)
        else:
            self.file_path(name).unlink(missing_ok=True)
        return self.registry.remove(name)

    # --- files for sandboxes -----------------------------------------------------

    def file_path(self, name: str) -> Path:
        return self.files / f"{name}.parquet"

    @staticmethod
    def sandbox_path(data_dir: PurePosixPath, name: str) -> PurePosixPath:
        return data_dir / "uploads" / f"{name}.parquet"

    def sandbox_files(self, data_dir: PurePosixPath, names: list[str] | None = None) -> list[tuple[str, bytes]]:
        """(sandbox path, content) for every file dataset, or just `names`."""
        return [
            (str(self.sandbox_path(data_dir, d.name)), self.file_path(d.name).read_bytes())
            for d in self.list()
            if d.kind == "file" and (names is None or d.name in names) and self.file_path(d.name).exists()
        ]

    # --- staging -----------------------------------------------------------------

    def _staged(self, upload_id: str) -> tuple[Path, str]:
        if not _UPLOAD_ID.match(upload_id):
            raise UploadError("unknown upload")
        matches = [p for p in self.staging.glob(f"{upload_id}.*") if p.suffix != ".json"]
        meta = self.staging / f"{upload_id}.json"
        if not matches or not meta.exists():
            raise UploadError("unknown or expired upload; upload the file again")
        return matches[0], json.loads(meta.read_text())["filename"]

    def _discard(self, upload_id: str) -> None:
        for p in self.staging.glob(f"{upload_id}.*"):
            p.unlink(missing_ok=True)

    def clear_staging(self) -> None:
        shutil.rmtree(self.staging, ignore_errors=True)


def suggest_name(filename: str) -> str:
    """A valid dataset name from a filename: "Sales 2031 (Q1).csv" -> "sales_2031_q1"."""
    stem = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_") or "dataset"
    if not stem[0].isalpha():
        stem = f"d_{stem}"
    stem = stem[:63]
    return stem if check_name(stem) is None else f"{stem[:57]}_data"
