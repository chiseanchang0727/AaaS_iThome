"""Sandboxes the agent can run code in, chosen by `sandbox.provider` in config.

    provider = get_provider(cfg.sandbox)       # None when provider is "none"
    with provider.session() as sandbox:
        agent = build_agent(sandbox=sandbox)
        ...
        download_outputs(sandbox, cfg.sandbox.output_dir, Path("reports/run-1"))
"""

from config import SandboxConfig

from .base import ProviderOptions, SandboxProvider, SandboxSetupError, download_outputs
from .daytona import DaytonaProvider

__all__ = [
    "PROVIDERS",
    "ProviderOptions",
    "SandboxProvider",
    "SandboxSetupError",
    "download_outputs",
    "get_provider",
]

PROVIDERS: dict[str, type[SandboxProvider]] = {
    cls.name: cls for cls in (DaytonaProvider,)
}
"""Every provider `sandbox.provider` can name. Add new ones here."""


def get_provider(config: SandboxConfig) -> SandboxProvider | None:
    """The provider `config` selects, with its options validated; None for "none"."""
    if config.provider == "none":
        if config.options or config.packages:
            raise ValueError("sandbox.options or packages is set but sandbox.provider is none")
        return None
    try:
        cls = PROVIDERS[config.provider]
    except KeyError:
        known = ", ".join(sorted(["none", *PROVIDERS]))
        raise ValueError(f"unknown sandbox provider {config.provider!r}; known: {known}") from None
    return cls(config.options, config.packages)
