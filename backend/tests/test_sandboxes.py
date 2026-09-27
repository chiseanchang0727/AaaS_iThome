"""Sandbox providers: the ABC, the registry, and Daytona against a fake client."""

from pathlib import Path

import daytona
import pytest
from deepagents.backends.local_shell import LocalShellBackend
from langchain_daytona import DaytonaSandbox
from pydantic import ValidationError

from config import SandboxConfig
from deepagents.backends.protocol import ExecuteResponse

from sandboxes import PROVIDERS, ProviderOptions, SandboxProvider, SandboxSetupError, get_provider
from sandboxes.daytona import DaytonaOptions, DaytonaProvider


def sandbox_config(provider: str, options: dict | None = None, packages=()) -> SandboxConfig:
    return SandboxConfig(
        provider=provider, options=options or {}, packages=list(packages),
        data_dir="/d", output_dir="/o", export_max_rows=1,
    )


# --- the ABC -------------------------------------------------------------------


class LocalOptions(ProviderOptions):
    label: str = "local"


class LocalProvider(SandboxProvider):
    """A minimal provider: temp dirs on this machine. Records its calls."""

    name = "local"
    Options = LocalOptions

    def __init__(self, options=None, root: Path | None = None):
        super().__init__(options)
        self.root = root
        self.created, self.destroyed = [], []

    def create(self):
        backend = LocalShellBackend(root_dir=self.root, virtual_mode=True)
        self.created.append(backend)
        return backend

    def destroy(self, sandbox):
        self.destroyed.append(sandbox)


def test_provider_must_implement_create_and_destroy():
    class Incomplete(SandboxProvider):
        name = "incomplete"

        def create(self):
            return None

    with pytest.raises(TypeError, match="destroy"):
        Incomplete()


def test_options_are_validated_by_the_providers_schema():
    assert LocalProvider({"label": "x"}).options.label == "x"
    assert LocalProvider().options.label == "local"


def test_unknown_options_are_rejected():
    with pytest.raises(ValidationError, match="extra"):
        LocalProvider({"lable": "typo"})


def test_session_yields_the_created_sandbox_and_destroys_it(tmp_path):
    provider = LocalProvider(root=tmp_path)
    with provider.session() as sandbox:
        assert provider.created == [sandbox]
        assert provider.destroyed == []
    assert provider.destroyed == [sandbox]


def test_session_destroys_the_sandbox_even_on_error(tmp_path):
    provider = LocalProvider(root=tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with provider.session():
            raise RuntimeError("boom")
    assert len(provider.destroyed) == 1


def test_session_gives_a_working_backend(tmp_path):
    with LocalProvider(root=tmp_path).session() as sandbox:
        sandbox.upload_files([("/x.txt", b"hi")])
        [response] = sandbox.download_files(["/x.txt"])
    assert response.content == b"hi"


# --- packages: installed by the ABC, for every provider ---------------------------


class RecordingSandbox:
    """Records commands instead of running them; answers with a set exit code."""

    def __init__(self, exit_code: int = 0, output: str = ""):
        self.commands, self.exit_code, self.output = [], exit_code, output

    def execute(self, command, timeout=None):
        self.commands.append(command)
        return ExecuteResponse(output=self.output, exit_code=self.exit_code, truncated=False)


class RecordingProvider(SandboxProvider):
    name = "recording"

    def __init__(self, options=None, packages=(), sandbox=None):
        super().__init__(options, packages)
        self.sandbox = sandbox or RecordingSandbox()
        self.destroyed = []

    def create(self):
        return self.sandbox

    def destroy(self, sandbox):
        self.destroyed.append(sandbox)


def test_no_packages_means_no_install():
    provider = RecordingProvider()
    with provider.session() as sandbox:
        pass
    assert sandbox.commands == []


def test_packages_are_pip_installed_before_the_sandbox_is_handed_over():
    provider = RecordingProvider(packages=["pyarrow", "scipy>=1.11"])
    with provider.session() as sandbox:
        assert len(sandbox.commands) == 1
    command = sandbox.commands[0]
    assert command.startswith("pip install")
    assert "pyarrow" in command and "'scipy>=1.11'" in command  # quoted for the shell


def test_package_names_cannot_inject_shell_commands():
    provider = RecordingProvider(packages=["pyarrow; rm -rf /"])
    with provider.session() as sandbox:
        pass
    assert "'pyarrow; rm -rf /'" in sandbox.commands[0]


def test_a_failed_install_raises_and_still_destroys_the_sandbox():
    provider = RecordingProvider(
        packages=["no-such-package"], sandbox=RecordingSandbox(exit_code=1, output="No matching distribution")
    )
    with pytest.raises(SandboxSetupError, match="No matching distribution"):
        with provider.session():
            pytest.fail("the agent should not get a half-prepared sandbox")
    assert len(provider.destroyed) == 1


# --- the registry --------------------------------------------------------------


def test_registry_is_keyed_by_provider_name():
    assert PROVIDERS == {"daytona": DaytonaProvider}
    assert all(name == cls.name for name, cls in PROVIDERS.items())


def test_none_means_no_sandbox():
    assert get_provider(sandbox_config("none")) is None


def test_none_with_options_is_a_mistake():
    with pytest.raises(ValueError, match="provider is none"):
        get_provider(sandbox_config("none", {"snapshot": "x"}))


def test_none_with_packages_is_a_mistake():
    with pytest.raises(ValueError, match="provider is none"):
        get_provider(sandbox_config("none", packages=["pyarrow"]))


def test_get_provider_passes_the_packages_on():
    assert get_provider(sandbox_config("daytona", packages=["pyarrow"])).packages == ["pyarrow"]


def test_unknown_provider_lists_the_known_ones():
    with pytest.raises(ValueError, match="unknown sandbox provider 'modal'; known: daytona, none"):
        get_provider(sandbox_config("modal"))


def test_get_provider_passes_the_options_on():
    provider = get_provider(sandbox_config("daytona", {"snapshot": "py-data"}))
    assert isinstance(provider, DaytonaProvider)
    assert provider.options.snapshot == "py-data"


def test_get_provider_rejects_options_for_another_provider():
    with pytest.raises(ValidationError):
        get_provider(sandbox_config("daytona", {"gpu": "A100"}))


def test_a_new_provider_needs_only_a_registry_entry(monkeypatch, tmp_path):
    monkeypatch.setitem(PROVIDERS, "local", LocalProvider)
    provider = get_provider(sandbox_config("local", {"label": "dev"}))
    assert isinstance(provider, LocalProvider) and provider.options.label == "dev"


# --- Daytona -------------------------------------------------------------------


class FakeRawSandbox:
    def __init__(self, id: str):
        self.id = id


class FakeDaytona:
    """Stands in for daytona.Daytona: records create/delete, no network."""

    instances: list["FakeDaytona"] = []

    def __init__(self):
        self.created, self.deleted = [], []
        FakeDaytona.instances.append(self)

    def create(self, params):
        self.created.append(params)
        return FakeRawSandbox(f"sbx-{len(self.created)}")

    def delete(self, sandbox):
        self.deleted.append(sandbox)


@pytest.fixture
def fake_daytona(monkeypatch):
    FakeDaytona.instances = []
    monkeypatch.setattr(daytona, "Daytona", FakeDaytona)
    return FakeDaytona


def test_daytona_defaults_leave_daytonas_own_defaults_alone(fake_daytona):
    DaytonaProvider().create()
    [params] = fake_daytona.instances[0].created
    assert params.snapshot is None and params.auto_stop_interval is None
    assert params.network_block_all is False


def test_daytona_passes_its_options_to_create(fake_daytona):
    options = {"snapshot": "py-data", "auto_stop_interval": 15, "network_block_all": True}
    DaytonaProvider(options).create()
    [params] = fake_daytona.instances[0].created
    assert (params.snapshot, params.auto_stop_interval, params.network_block_all) == ("py-data", 15, True)


def test_daytona_returns_a_deepagents_backend_with_the_command_timeout(fake_daytona):
    backend = DaytonaProvider({"command_timeout": 120}).create()
    assert isinstance(backend, DaytonaSandbox)
    assert backend.id == "sbx-1"
    assert backend._default_timeout == 120


def test_daytona_command_timeout_is_not_sent_to_create(fake_daytona):
    DaytonaProvider({"command_timeout": 120}).create()
    [params] = fake_daytona.instances[0].created
    assert "command_timeout" not in params.model_dump()


def test_daytona_passes_no_environment_into_the_sandbox(fake_daytona, monkeypatch):
    """The DB credentials live in this process's env; none of it goes in."""
    monkeypatch.setenv("AGENT_DATABASE_URL", "postgresql://secret")
    DaytonaProvider().create()
    [params] = fake_daytona.instances[0].created
    assert not params.env_vars


def test_daytona_destroy_deletes_that_sandbox(fake_daytona):
    provider = DaytonaProvider()
    first, second = provider.create(), provider.create()
    provider.destroy(first)
    assert [s.id for s in fake_daytona.instances[0].deleted] == ["sbx-1"]
    provider.destroy(second)
    assert [s.id for s in fake_daytona.instances[0].deleted] == ["sbx-1", "sbx-2"]


def test_daytona_destroy_twice_is_harmless(fake_daytona):
    provider = DaytonaProvider()
    backend = provider.create()
    provider.destroy(backend)
    provider.destroy(backend)
    assert len(fake_daytona.instances[0].deleted) == 1


def test_daytona_reuses_one_client(fake_daytona):
    provider = DaytonaProvider()
    provider.create()
    provider.create()
    assert len(fake_daytona.instances) == 1


def test_daytona_session_installs_packages_through_execute(fake_daytona, monkeypatch):
    commands = []
    monkeypatch.setattr(
        DaytonaSandbox, "execute",
        lambda self, command, timeout=None: commands.append(command) or ExecuteResponse(output="", exit_code=0),
    )
    with DaytonaProvider(packages=["pyarrow"]).session():
        pass
    assert commands and "pyarrow" in commands[0]
    assert len(fake_daytona.instances[0].deleted) == 1


def test_daytona_session_deletes_on_error(fake_daytona):
    with pytest.raises(RuntimeError):
        with DaytonaProvider().session():
            raise RuntimeError("agent crashed")
    assert len(fake_daytona.instances[0].deleted) == 1


def test_daytona_options_reject_unknown_keys():
    with pytest.raises(ValidationError):
        DaytonaOptions(image="python:3.12")
