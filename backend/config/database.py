"""Config for the database the agent queries."""

import os

from pydantic import BaseModel, ConfigDict

DSN_ENV = "AGENT_DATABASE_URL"
INGEST_DSN_ENV = "INGEST_DATABASE_URL"


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rows: int
    """Refuse a result with more rows than this rather than fetching it all.

    Protects the database and process memory. How much of a result may enter
    the agent's context is `AgentConfig.max_result_tokens`.
    """

    timeout: float
    """Seconds before a query is cancelled."""

    @property
    def dsn(self) -> str:
        """The agent's read-only connection.

        Read from the environment, not the yml: it carries a password. Kept
        separate from $DATABASE_URL (the owner, used by scripts/load_csv.py) so
        the agent cannot write even if a query tries.

        Looked up on use rather than at load, so the app runs without a
        database configured until something actually queries one.
        """
        try:
            return os.environ[DSN_ENV]
        except KeyError:
            raise RuntimeError(f"{DSN_ENV} is not set") from None

    @property
    def ingest_dsn(self) -> str:
        """The connection that creates tables from uploads.

        Its role can create tables and grant SELECT on them to the agent's
        role, and can touch nothing else: it holds no rights on `videos`.
        """
        try:
            return os.environ[INGEST_DSN_ENV]
        except KeyError:
            raise RuntimeError(f"{INGEST_DSN_ENV} is not set") from None
