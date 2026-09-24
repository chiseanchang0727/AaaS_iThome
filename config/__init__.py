"""Application config, loaded once at import time.

    from config import cfg

    cfg.data.path      # Path to the dataset
    cfg.data.format    # DataType.CSV
"""

from .base import Config
from .data import DataConfig, DataType
from .database import DatabaseConfig

__all__ = ["Config", "DataConfig", "DataType", "DatabaseConfig", "cfg"]

cfg = Config.load()
