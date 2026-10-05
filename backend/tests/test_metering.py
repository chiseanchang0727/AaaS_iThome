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


def test_a_program_killed_under_the_shell_counts_as_killed_too(tmp_path):
    # Not the shell itself: bash outlives the program and exits 128+9 for it,
    # as for `python3 job.py 2>&1` in the sandbox.
    result, m = run_measured(shell(tmp_path), "echo started; sh -c 'kill -9 $$'")
    assert result.exit_code == 137 and m.signal == 9
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


def oom_stand_in(upgrade_to=None, retries_first=1, note="Moved to a bigger sandbox (4 GB memory, 2 vCPU)."):
    """A stand-in on a 1 GB sandbox that is always killed; asking for more gives `upgrade_to`."""
    from sandboxes.lazy import Upgrade

    small = Scripted("small", "", 980, 1024, killed=True)
    lazy, steps, asked = LazySandbox(retries_first=retries_first), [], []
    lazy.on_step = steps.append

    def ask(reason):
        asked.append(reason)
        return Upgrade(upgrade_to, note if upgrade_to else "No room for a 4 GB sandbox right now.")

    lazy.on_out_of_memory = ask
    small.write = lambda path, content: None   # the agent rewriting its script
    small.edit = lambda *a, **k: None
    lazy.ready(small)
    return lazy, small, steps, asked


def test_the_first_kill_asks_for_a_lighter_approach_and_refuses_a_bigger_sandbox():
    from sandboxes.lazy import NOT_YET, TRY_LIGHTER

    lazy, small, steps, asked = oom_stand_in(Scripted("big", "ok\n", 1545, 4096))
    result = lazy.execute("python3 matrix.py")
    assert result.exit_code == 137 and result.output.endswith(f"[{TRY_LIGHTER}]")
    assert lazy.request_bigger("needs 3.2 GB") == NOT_YET
    assert asked == [] and lazy.id == "small"


def test_running_the_same_code_again_does_not_count_as_trying():
    from sandboxes.lazy import NOT_YET, SAME_AGAIN

    lazy, *_ = oom_stand_in(Scripted("big", "", 10, 4096))
    lazy.execute("python3 matrix.py")
    again = lazy.execute("python3 matrix.py")  # no file changed in between
    assert again.output.endswith(f"[{SAME_AGAIN}]")
    assert lazy.request_bigger("please") == NOT_YET


def test_after_a_changed_attempt_also_fails_the_agent_may_ask():
    from sandboxes.lazy import MAY_ASK

    big = Scripted("big", "mean: 0.5\n", 1545, 4096)
    lazy, small, steps, asked = oom_stand_in(big)
    lazy.execute("python3 matrix.py")
    lazy.write("/home/daytona/matrix.py", "chunked version")   # the rewrite
    second = lazy.execute("python3 matrix.py")
    assert second.output.endswith(f"[{MAY_ASK}]") and asked == []

    answer = lazy.request_bigger("the matrix must be held in memory at once")
    assert asked == ["the matrix must be held in memory at once"] and "Run the command again" in answer
    result = lazy.execute("python3 matrix.py")
    assert result.exit_code == 0 and "mean: 0.5" in result.output and lazy.id == "big"
    assert [s.memory_limit_mb for s in steps] == [1024, 1024, 4096]


def test_a_different_command_also_counts_as_a_new_attempt():
    from sandboxes.lazy import MAY_ASK

    lazy, *_ = oom_stand_in(Scripted("big", "", 10, 4096))
    lazy.execute("python3 a.py")
    assert lazy.execute("python3 b.py").output.endswith(f"[{MAY_ASK}]")


def test_one_more_kill_after_that_moves_without_asking():
    big = Scripted("big", "mean: 0.5\n", 1545, 4096)
    lazy, small, steps, asked = oom_stand_in(big)
    lazy.execute("python3 a.py")
    lazy.execute("python3 b.py")
    result = lazy.execute("python3 b.py")  # the third kill: safety net
    assert len(asked) == 1 and "ran out of memory 3 times" in asked[0]
    assert result.exit_code == 0 and result.output.startswith("[It ran out of memory again, so: Moved")


def test_with_no_retries_required_the_agent_may_ask_at_once():
    from sandboxes.lazy import ASK_FIRST

    big = Scripted("big", "ok\n", 1545, 4096)
    lazy, small, steps, asked = oom_stand_in(big, retries_first=0)
    assert lazy.execute("python3 matrix.py").output.endswith(f"[{ASK_FIRST}]")
    assert lazy.request_bigger("needs it").startswith("Moved")
    lazy2, *_ = oom_stand_in(big, retries_first=0)
    lazy2.execute("python3 x.py")
    assert lazy2.execute("python3 x.py").output.startswith("[It ran out of memory again")  # second kill moves


def test_when_no_bigger_sandbox_is_possible_the_agent_is_told_why():
    lazy, small, steps, asked = oom_stand_in(upgrade_to=None, retries_first=0)
    lazy.execute("python3 matrix.py")
    assert lazy.request_bigger("needs it") == "No room for a 4 GB sandbox right now."
    result = lazy.execute("python3 matrix.py")
    assert result.exit_code == 137 and result.output.endswith("[No room for a 4 GB sandbox right now.]")
    assert lazy.id == "small"
    assert LazySandbox().request_bigger("x") == "No bigger sandbox is available here."


def test_other_failures_do_not_ask_for_a_bigger_sandbox():
    asked = []
    lazy = LazySandbox()
    lazy.on_out_of_memory = lambda reason: asked.append(reason)
    lazy.ready(Scripted("s", "", 50, 1024, killed=True))  # killed, but far from the limit
    lazy.execute("python3 x.py")
    assert asked == []
