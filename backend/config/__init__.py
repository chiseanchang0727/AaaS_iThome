"""Application config, loaded once at import time.

    from config import cfg

    cfg.data.path      # Path to the dataset
    cfg.data.format    # DataType.CSV
"""

from .agent import AgentConfig
from .base import Config
from .data import DataConfig, DataType
from .database import DatabaseConfig
from .sandbox import SandboxConfig

__all__ = ["AgentConfig", "Config", "DataConfig", "DataType", "DatabaseConfig", "SandboxConfig", "cfg"]

cfg = Config.load()
