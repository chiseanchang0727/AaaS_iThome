"""Daytona sandboxes, via the `langchain-daytona` backend.

Reads $DAYTONA_API_KEY. Nothing from this process's environment is passed into
the sandbox, so the database credentials never reach code the model writes.
"""

from deepagents.backends.protocol import SandboxBackendProtocol

from .base import ProviderOptions, SandboxProvider


class DaytonaOptions(ProviderOptions):
    snapshot: str | None = None
    """Snapshot (image) to start from; Daytona's default when unset."""

    auto_stop_interval: int | None = None
    """Minutes idle before Daytona stops the sandbox: a backstop if `destroy`
    never runs. Daytona's default when unset."""

    network_block_all: bool = False
    """Cut the sandbox off from the internet (no pip install at runtime)."""

    command_timeout: int = 30 * 60
    """Seconds before a command run with `execute` is abandoned."""


class DaytonaProvider(SandboxProvider):
    name = "daytona"
    Options = DaytonaOptions
    options: DaytonaOptions

    def __init__(self, options=None, packages=()) -> None:
        super().__init__(options, packages)
        self._client = None
        self._sandboxes = {}

    def _get_client(self):
        # Imported here so the SDK is only needed when Daytona is selected.
        if self._client is None:
            from daytona import Daytona

            self._client = Daytona()
        return self._client

    def create(self) -> SandboxBackendProtocol:
        from daytona import CreateSandboxFromSnapshotParams
        from langchain_daytona import DaytonaSandbox

        params = CreateSandboxFromSnapshotParams(
            **self.options.model_dump(exclude={"command_timeout"}, exclude_none=True)
        )
        sandbox = self._get_client().create(params)
        backend = DaytonaSandbox(sandbox=sandbox, timeout=self.options.command_timeout)
        self._sandboxes[backend.id] = sandbox
        return backend

    def destroy(self, sandbox: SandboxBackendProtocol) -> None:
        raw = self._sandboxes.pop(sandbox.id, None)
        if raw is not None:
            self._get_client().delete(raw)
