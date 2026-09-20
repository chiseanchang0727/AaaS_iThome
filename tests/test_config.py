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
