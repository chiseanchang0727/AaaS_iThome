"""Application config, loaded once at import time.

    from config import cfg

    cfg.data.path      # Path to the dataset
    cfg.data.format    # DataType.CSV
"""

from .agent import AgentConfig
from .analysis_execution import AnalysisExecutionConfig
from .base import Config
from .data import DataConfig, DataType
from .database import DatabaseConfig
from .sandbox import SandboxConfig
from .server import ServerConfig

__all__ = ["AgentConfig", "AnalysisExecutionConfig", "Config", "DataConfig", "DataType", "DatabaseConfig", "SandboxConfig", "ServerConfig", "cfg"]

cfg = Config.load()
