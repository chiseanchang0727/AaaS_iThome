from pathlib import Path

import pytest
from pydantic import ValidationError

from config import Config, DataType


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(body)
    return path


def test_shipped_yml_parses_into_the_schema():
    cfg = Config.load()
    assert isinstance(cfg, Config)
    assert cfg.data.format is DataType.CSV


def test_unknown_format_is_rejected_at_load_time(tmp_path):
    """A format we cannot read fails when the config loads, not when something
    first tries to open the file."""
    path = write(tmp_path, "data: {path: data/trending.parquet}")
    with pytest.raises(ValidationError, match="unsupported data format 'parquet'"):
        Config.load(path)


def test_format_is_not_a_config_field(tmp_path):
    path = write(tmp_path, "data: {path: data/trending.csv, format: csv}")
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        Config.load(path)


def test_more_warm_sandboxes_than_the_cap_is_rejected():
    from config import ServerConfig

    base = {"artifacts_dir": "a", "uploads_dir": "u", "history_dir": "h",
            "sandbox_idle_minutes": 15, "max_upload_mb": 50}
    assert ServerConfig(**base, max_sandboxes=2, warm_sandboxes=2).warm_sandboxes == 2
    assert ServerConfig(**base, max_sandboxes=None, warm_sandboxes=5).warm_sandboxes == 5
    with pytest.raises(ValidationError, match="warm_sandboxes cannot be more than max_sandboxes"):
        ServerConfig(**base, max_sandboxes=2, warm_sandboxes=3)
