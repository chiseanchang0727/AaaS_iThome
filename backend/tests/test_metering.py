"""Measuring sandbox commands: the wrapper on a real local shell, and its edges."""

import asyncio

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from sandboxes.lazy import LazySandbox
from sandboxes.metering import MARKER, StepMeasure, _killed_note, run_measured, split_report


def shell(tmp_path):
    return LocalShellBackend(root_dir=tmp_path, virtual_mode=True)


def test_the_agent_sees_exactly_what_the_command_printed(tmp_path):
    command = """echo "it's 'quoted'"; echo oops >&2; exit 3"""
    result, measure = run_measured(shell(tmp_path), command)
    plain = shell(tmp_path).execute(command)
    assert result.output == plain.output and result.exit_code == plain.exit_code == 3
    assert MARKER not in result.output


def test_a_step_reports_time_cpu_and_memory(tmp_path):
    command = "python3 -c 'x = bytearray(150 * 1024 * 1024); sum(range(3_000_000))'"
    result, m = run_measured(shell(tmp_path), command)
    assert result.exit_code == 0
    assert m.command == command and m.exit_code == 0
    assert m.run_seconds > 0 and m.cpu_seconds > 0 and m.seconds >= m.run_seconds
    assert 140 < m.peak_memory_mb < 400  # the 150 MB it allocated, plus the interpreter


def test_long_commands_are_clipped_in_the_record(tmp_path):
    _, m = run_measured(shell(tmp_path), "true " + "x" * 1000)
    assert len(m.command) == 500


def test_a_report_printed_by_the_command_itself_is_left_alone():
    fake = f"before\n{MARKER} {{\"run_seconds\": 9}}\nafter\n{MARKER} {{\"run_seconds\": 1.5}}\n"
    output, report = split_report(fake)
    assert report == {"run_seconds": 1.5}
    assert output == f"before\n{MARKER} {{\"run_seconds\": 9}}\nafter"


class NoPython:
    """A sandbox without python3: the wrapper fails, the plain command works."""

    def __init__(self):
        self.commands = []

    def execute(self, command, timeout=None):
        self.commands.append(command)
        if command.startswith("python3 -c"):
            return ExecuteResponse(output="bash: python3: command not found", exit_code=127)
        return ExecuteResponse(output="hello\n", exit_code=0)


def test_without_python3_the_command_still_runs_unmeasured():
    sandbox = NoPython()
    result, m = run_measured(sandbox, "echo hello")
    assert (result.output, result.exit_code) == ("hello\n", 0)
    assert sandbox.commands[-1] == "echo hello"
    assert m.run_seconds is None and m.peak_memory_mb is None and m.seconds >= 0


def test_the_stand_in_reports_each_step(tmp_path):
    lazy, steps = LazySandbox(), []
    lazy.on_step = steps.append
    lazy.ready(shell(tmp_path))
    result = asyncio.run(lazy.aexecute("echo hi"))
    assert result.output.strip() == "hi"
    assert [s.command for s in steps] == ["echo hi"] and steps[0].peak_memory_mb is not None


def test_a_failing_recorder_never_breaks_the_step(tmp_path):
    lazy = LazySandbox()

    def broken(_):
        raise OSError("disk full")

    lazy.on_step = broken
    lazy.ready(shell(tmp_path))
    assert lazy.execute("echo still").output.strip() == "still"



def test_a_killed_step_exits_like_a_shell_and_says_why(tmp_path):
    result, m = run_measured(shell(tmp_path), "echo started; kill -9 $$")
    assert result.exit_code == 137 and m.signal == 9
    assert result.output.startswith("started\n")
    assert "Killed (signal 9), most likely for using too much memory" in result.output


def test_out_of_memory_is_named_only_near_the_limit():
    near = StepMeasure("x", 137, 1.0, peak_memory_mb=972, signal=9, memory_limit_mb=1024)
    far = StepMeasure("x", 137, 1.0, peak_memory_mb=50, signal=9, memory_limit_mb=1024)
    assert near.out_of_memory and not far.out_of_memory
    assert "ran out of memory (it reached 972 MB of the sandbox's 1024 MB)" in _killed_note(near)
    assert "most likely" in _killed_note(far)
    assert _killed_note(StepMeasure("x", 143, 1.0, signal=15)) == "[Killed by signal 15.]"


# --- out of memory: a bigger sandbox, and the command again -------------------------


class Scripted:
    """A sandbox whose wrapped commands report the given memory, as metering prints it."""

    def __init__(self, id, output, peak, limit, killed=False):
        self.id, self.output, self.peak, self.limit, self.killed = id, output, peak, limit, killed
        self.commands = []

    def execute(self, command, timeout=None):
        import json

        self.commands.append(command)
        report = {"run_seconds": 0.4, "cpu_seconds": 0.4, "peak_memory_mb": self.peak,
                  "signal": 9 if self.killed else None, "memory_limit_mb": self.limit}
        out = ("" if self.killed else self.output) + f"\n{MARKER} {json.dumps(report)}\n"
        return ExecuteResponse(output=out, exit_code=137 if self.killed else 0)


def test_out_of_memory_moves_to_a_bigger_sandbox_and_runs_again():
    from sandboxes.lazy import Upgrade

    small = Scripted("small", "", 980, 1024, killed=True)
    big = Scripted("big", "mean: 0.5\n", 1545, 4096)
    lazy, steps = LazySandbox(), []
    lazy.on_step = steps.append
    lazy.on_out_of_memory = lambda: Upgrade(big, "Moved to a bigger sandbox (4 GB memory, 2 vCPU).")
    lazy.ready(small)

    result = lazy.execute("python3 matrix.py")
    assert result.exit_code == 0
    assert result.output.startswith("[Moved to a bigger sandbox (4 GB memory, 2 vCPU). The command ran out of memory")
    assert "mean: 0.5" in result.output
    assert [(s.out_of_memory, s.memory_limit_mb) for s in steps] == [(True, 1024), (False, 4096)]
    assert lazy.id == "big"  # later steps go to the bigger sandbox
    lazy.execute("ls")
    assert len(small.commands) == 1 and len(big.commands) == 2


def test_without_a_bigger_sandbox_the_agent_is_told_why():
    from sandboxes.lazy import Upgrade

    small = Scripted("small", "", 980, 1024, killed=True)
    lazy = LazySandbox()
    lazy.on_out_of_memory = lambda: Upgrade(None, "No room for a 4 GB sandbox right now.")
    lazy.ready(small)
    result = lazy.execute("python3 matrix.py")
    assert result.exit_code == 137
    assert "ran out of memory" in result.output and result.output.endswith("[No room for a 4 GB sandbox right now.]")
    assert lazy.id == "small"


def test_other_failures_do_not_ask_for_a_bigger_sandbox():
    asked = []
    lazy = LazySandbox()
    lazy.on_out_of_memory = lambda: asked.append(1)
    lazy.ready(Scripted("s", "", 50, 1024, killed=True))  # killed, but far from the limit
    lazy.execute("python3 x.py")
    assert asked == []
