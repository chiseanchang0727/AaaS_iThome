"""Limits for running saved analyses: when a recipe's strategy no longer fits its data.

Hard limits: the run cannot go on safely, so it stops and asks for an
optimization. Soft limits: the run works, but an optimization is worth it.
Rows have one hard limit everywhere, `sandbox.export_max_rows`, and a query's
hard time limit is `database.timeout`: both are reused, not repeated here.
"""

from pydantic import BaseModel, ConfigDict, Field


class AnalysisExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soft_export_rows: int = Field(default=50_000, ge=1)
    """Rows one input may move into the sandbox before an optimization is recommended."""

    soft_export_mb: float = Field(default=20, gt=0)
    max_export_mb: float = Field(default=200, gt=0)
    """Size of one input's Parquet file: recommended above the soft limit, refused above the hard one."""

    soft_run_seconds: float = Field(default=120, gt=0)
    """The script's own time in the sandbox."""

    memory_warning_ratio: float = Field(default=0.8, gt=0, le=1)
    """Peak memory at this share of the sandbox's limit is close to running out."""
