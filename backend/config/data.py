"""Config for the dataset the app reads."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class DataType(str, Enum):
    """Data formats the app can read."""

    CSV = "csv"

    @classmethod
    def from_path(cls, path: Path) -> "DataType":
        suffix = path.suffix.lower().lstrip(".")
        try:
            return cls(suffix)
        except ValueError:
            raise ValueError(f"unsupported data format {suffix!r} for {path}") from None


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path

    @model_validator(mode="after")
    def _check_format(self) -> "DataConfig":
        """Fail at load time on a format we cannot read, not at first read."""
        DataType.from_path(self.path)
        return self

    @property
    def format(self) -> DataType:
        return DataType.from_path(self.path)
