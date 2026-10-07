"""Saved analyses on disk, one folder each:

    <root>/<id>/analysis.json                 the recipe (models.Analysis), its current version
    <root>/<id>/versions/<n>.json             earlier versions, kept when it is changed
    <root>/<id>/runs/<run id>/run.json        each run (models.RunRecord)
    <root>/<id>/runs/<run id>/outputs/...     the files that run made
"""

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .models import Analysis, RunRecord

_ID = re.compile(r"^[a-z0-9]{1,40}$")
_RUN_ID = re.compile(r"^\d{8}-\d{6}(\d{3})?-[a-z0-9]{4}$")
"""20261006-233253512-8a0d: to the millisecond, so runs in the same second still sort in order
(older ids have no milliseconds)."""
_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S%f')[:-3]}-{uuid.uuid4().hex[:4]}"


class AnalysisStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, analysis_id: str) -> Path:
        if not _ID.match(analysis_id):
            raise KeyError(analysis_id)
        return self.root / analysis_id

    def save(self, analysis: Analysis) -> None:
        folder = self._dir(analysis.id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")

    def replace(self, new: Analysis) -> Analysis:
        """Save `new` as the next version of the analysis with its id, keeping the current one in versions/.

        The id, creation time and runs stay; `new` gets the next version number.
        Raises KeyError if there is no analysis with that id.
        """
        current = self.get(new.id)
        if current is None:
            raise KeyError(new.id)
        versions = self._dir(new.id) / "versions"
        versions.mkdir(exist_ok=True)
        (versions / f"{current.version}.json").write_text(current.model_dump_json(indent=2), encoding="utf-8")
        updated = new.model_copy(update={
            "version": current.version + 1, "created_at": current.created_at, "updated_at": new.created_at,
        })
        self.save(updated)
        return updated

    def get(self, analysis_id: str) -> Analysis | None:
        try:
            path = self._dir(analysis_id) / "analysis.json"
        except KeyError:
            return None
        return Analysis.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def all(self) -> list[Analysis]:
        """Newest first."""
        found = [self.get(p.parent.name) for p in self.root.glob("*/analysis.json")] if self.root.exists() else []
        return sorted((a for a in found if a is not None), key=lambda a: a.created_at, reverse=True)

    def delete(self, analysis_id: str) -> bool:
        try:
            folder = self._dir(analysis_id)
        except KeyError:
            return False
        if not folder.is_dir():
            return False
        shutil.rmtree(folder)
        return True

    # --- runs ---------------------------------------------------------------------

    def run_dir(self, analysis_id: str, run_id: str) -> Path:
        if not _RUN_ID.match(run_id):
            raise KeyError(run_id)
        return self._dir(analysis_id) / "runs" / run_id

    def save_run(self, analysis_id: str, run: RunRecord, outputs: dict[str, bytes] | None = None) -> None:
        folder = self.run_dir(analysis_id, run.id)
        (folder / "outputs").mkdir(parents=True, exist_ok=True)
        for name, content in (outputs or {}).items():
            if _FILE.match(name):
                (folder / "outputs" / name).write_bytes(content)
        (folder / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def runs(self, analysis_id: str) -> list[RunRecord]:
        """Newest first."""
        folder = self._dir(analysis_id) / "runs"
        if not folder.is_dir():
            return []
        found = [RunRecord.model_validate_json(p.read_text(encoding="utf-8")) for p in folder.glob("*/run.json")]
        return sorted(found, key=lambda r: r.id, reverse=True)

    def get_run(self, analysis_id: str, run_id: str) -> RunRecord | None:
        try:
            path = self.run_dir(analysis_id, run_id) / "run.json"
        except KeyError:
            return None
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def output_path(self, analysis_id: str, run_id: str, name: str) -> Path | None:
        try:
            path = self.run_dir(analysis_id, run_id) / "outputs" / name
        except KeyError:
            return None
        return path if _FILE.match(name) and path.is_file() else None

    def to_json(self, analysis: Analysis) -> dict:
        return json.loads(analysis.model_dump_json())
