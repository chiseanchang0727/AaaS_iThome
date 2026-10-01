"""What every sandbox provider implements, and what works on any sandbox.

A provider owns a sandbox's lifecycle: create it from config, destroy it after
the run. What it creates is a deepagents `SandboxBackendProtocol`, so running
commands and moving files work the same whichever provider made it.
"""

import shlex
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from deepagents.backends.protocol import SandboxBackendProtocol
from pydantic import BaseModel, ConfigDict


class SandboxSetupError(RuntimeError):
    """A new sandbox could not be prepared, e.g. a package failed to install."""


@dataclass(frozen=True)
class FoundSandbox:
    """A sandbox that exists at the provider, found by its labels."""

    id: str
    labels: dict[str, str] = field(default_factory=dict)
    state: str = ""
    handle: Any = None
    """The provider's own object for it, for `delete_found`."""


class ProviderOptions(BaseModel):
    """Base for a provider's own settings, read from `sandbox.options`."""

    model_config = ConfigDict(extra="forbid")


class SandboxProvider(ABC):
    """Creates and destroys sandboxes of one kind.

    To add a provider: subclass this, set `name` and `Options`, implement
    `create` and `destroy`, and register the class in `sandboxes.PROVIDERS`.
    """

    name: ClassVar[str]
    """The value of `sandbox.provider` that selects this class."""

    Options: ClassVar[type[ProviderOptions]] = ProviderOptions
    """Schema for `sandbox.options`; unknown keys are rejected."""

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        packages: Iterable[str] = (),
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self.options = self.Options.model_validate(options or {})
        self.packages = list(packages)
        self.labels = dict(labels or {})
        """Put on every sandbox `create` makes, if the provider supports labels."""

    @abstractmethod
    def create(self) -> SandboxBackendProtocol:
        """Start a sandbox and return it as a deepagents backend."""

    @abstractmethod
    def destroy(self, sandbox: SandboxBackendProtocol) -> None:
        """Tear down a sandbox from `create`. Must not raise if already gone."""

    def find(self, labels: Mapping[str, str]) -> list[FoundSandbox]:
        """Every sandbox at the provider carrying all of `labels`, ours or not.

        Providers that cannot list return []: nothing is found, nothing deleted.
        """
        return []

    def delete_found(self, found: FoundSandbox) -> None:
        """Delete a sandbox from `find`."""
        raise NotImplementedError

    def is_alive(self, sandbox: SandboxBackendProtocol) -> bool:
        """Does the sandbox still run commands? A cheap round trip.

        Providers with a status API can override this with something cheaper.
        """
        try:
            return sandbox.execute("true").exit_code == 0
        except Exception:
            return False

    def prepare(self, sandbox: SandboxBackendProtocol) -> None:
        """Get a new sandbox ready: install `packages`. Same for every provider.

        Raises `SandboxSetupError` rather than letting the agent start without
        a package it was promised.
        """
        if not self.packages:
            return
        command = "pip install --quiet --disable-pip-version-check " + " ".join(
            shlex.quote(p) for p in self.packages
        )
        result = sandbox.execute(command)
        if result.exit_code != 0:
            raise SandboxSetupError(f"`{command}` failed ({result.exit_code}): {result.output.strip()}")

    @contextmanager
    def session(self) -> Iterator[SandboxBackendProtocol]:
        """A prepared sandbox for the length of a `with` block, destroyed even
        on error, including a failed `prepare`.

        Sandboxes are billed while they exist, so prefer this over calling
        `create` and `destroy` by hand.
        """
        sandbox = self.create()
        try:
            self.prepare(sandbox)
            yield sandbox
        finally:
            self.destroy(sandbox)


def download_outputs(
    sandbox: SandboxBackendProtocol, output_dir: PurePosixPath, local_dir: Path
) -> list[Path]:
    """Copy every file under `output_dir` in the sandbox into `local_dir`.

    Keeps the layout below `output_dir`. Returns the local paths written; none
    if the agent never created `output_dir`.
    """
    # Look before globbing: globbing a missing directory logs a warning.
    listing = sandbox.ls(str(output_dir.parent))
    if listing.error or not any(
        PurePosixPath(e["path"].rstrip("/")) == output_dir for e in listing.entries or []
    ):
        return []
    found = sandbox.glob("**/*", path=str(output_dir))
    if found.error or not found.matches:
        return []
    remote = [m["path"] for m in found.matches if not m.get("is_dir")]

    written = []
    for response in sandbox.download_files(remote):
        if response.error or response.content is None:
            continue
        target = local_dir / PurePosixPath(response.path).relative_to(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        written.append(target)
    return written
