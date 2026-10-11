"""Workflows on disk, one folder each:

    <root>/<id>/workflow.json           the workflow (models.Workflow)
    <root>/<id>/runs/<run id>.json      each run (models.WorkflowRun)

A run's outputs are not copied: each node's files stay with its analysis run.
"""

import re
import shutil
from pathlib import Path

from .models import Workflow, WorkflowRun

_ID = re.compile(r"^[a-z0-9]{1,40}$")
_RUN_ID = re.compile(r"^\d{8}-\d{9}-[a-z0-9]{4}$")


class WorkflowStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, workflow_id: str) -> Path:
        if not _ID.match(workflow_id):
            raise KeyError(workflow_id)
        return self.root / workflow_id

    def save(self, workflow: Workflow) -> None:
        folder = self._dir(workflow.id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "workflow.json").write_text(workflow.model_dump_json(indent=2), encoding="utf-8")

    def get(self, workflow_id: str) -> Workflow | None:
        try:
            path = self._dir(workflow_id) / "workflow.json"
        except KeyError:
            return None
        return Workflow.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def all(self) -> list[Workflow]:
        """Newest first."""
        if not self.root.is_dir():
            return []
        found = [Workflow.model_validate_json(p.read_text(encoding="utf-8")) for p in self.root.glob("*/workflow.json")]
        return sorted(found, key=lambda w: w.updated_at or w.created_at, reverse=True)

    def delete(self, workflow_id: str) -> bool:
        try:
            folder = self._dir(workflow_id)
        except KeyError:
            return False
        if not folder.is_dir():
            return False
        shutil.rmtree(folder)
        return True

    def save_run(self, run: WorkflowRun) -> None:
        if not _RUN_ID.match(run.id):
            raise KeyError(run.id)
        folder = self._dir(run.workflow_id) / "runs"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{run.id}.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def runs(self, workflow_id: str) -> list[WorkflowRun]:
        """Newest first."""
        try:
            folder = self._dir(workflow_id) / "runs"
        except KeyError:
            return []
        if not folder.is_dir():
            return []
        found = [WorkflowRun.model_validate_json(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
        return sorted(found, key=lambda r: r.id, reverse=True)

    def get_run(self, workflow_id: str, run_id: str) -> WorkflowRun | None:
        if not _RUN_ID.match(run_id):
            return None
        try:
            path = self._dir(workflow_id) / "runs" / f"{run_id}.json"
        except KeyError:
            return None
        return WorkflowRun.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None
