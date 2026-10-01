"""Measure what each command run in a sandbox costs: time, CPU, memory.

The command runs under a small Python wrapper in the sandbox, which reports
the command's wall time, CPU time and peak memory (getrusage of the processes
it started) as one marked line on stdout. That line is removed before the
output goes back, so the agent sees exactly what the command printed.

The sandbox's own memory counter (cgroup memory.peak) counts from when the
sandbox started, not per command, so it cannot tell steps apart; getrusage
can. Peak memory is that of the largest single process the command ran.

Measuring must never break a step: if the wrapper cannot run (no python3 in
the image), the command is run again as it was and comes back unmeasured.
"""

import base64
import json
import re
import shlex
import time
from dataclasses import asdict, dataclass

from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol, execute_accepts_timeout

MARKER = "@@sandbox_step@@"
_REPORT = re.compile(rf"\n?{re.escape(MARKER)} (\{{[^\n]*\}})\n?")

# Runs the command in bash, waits, then prints one report line. ru_maxrss is
# KiB on Linux (the sandboxes) but bytes on macOS. The command travels
# base64-encoded, so no quoting can break it.
# A command killed by a signal (the out-of-memory killer sends SIGKILL) exits
# 128+signal, as in a shell, and the report says which signal and what the
# sandbox's memory limit is (cgroup v2 memory.max; absent where unlimited).
_WRAPPER = (
    "import base64,json,resource,subprocess,sys,time\n"
    "c=base64.b64decode(sys.argv[1]).decode()\n"
    "t=time.monotonic()\n"
    "r=subprocess.run(['bash','-c',c])\n"
    "u=resource.getrusage(resource.RUSAGE_CHILDREN)\n"
    "sig=-r.returncode if r.returncode<0 else None\n"
    "try:\n"
    " m=open('/sys/fs/cgroup/memory.max').read().strip();lim=round(int(m)/1048576) if m.isdigit() else None\n"
    "except OSError:\n"
    " lim=None\n"
    "sys.stdout.flush()\n"
    f"print('\\n{MARKER} '+json.dumps({{'run_seconds':round(time.monotonic()-t,3),"
    "'cpu_seconds':round(u.ru_utime+u.ru_stime,3),"
    "'peak_memory_mb':round(u.ru_maxrss/(1048576 if sys.platform=='darwin' else 1024),1),"
    "'signal':sig,'memory_limit_mb':lim}),flush=True)\n"
    "sys.exit(128+sig if sig else r.returncode)\n"
)


@dataclass(frozen=True)
class StepMeasure:
    command: str
    """The command, clipped to 500 characters."""
    exit_code: int | None
    seconds: float
    """Wall time as seen from here: includes the round trip to the sandbox."""
    run_seconds: float | None = None
    """Wall time inside the sandbox."""
    cpu_seconds: float | None = None
    """User + system CPU time of everything the command ran."""
    peak_memory_mb: float | None = None
    """Peak resident memory of the largest process the command ran."""
    signal: int | None = None
    """The signal that killed it, if any: 9 is usually the out-of-memory killer."""
    memory_limit_mb: int | None = None
    """The sandbox's memory limit, when it has one."""

    @property
    def out_of_memory(self) -> bool:
        """Killed by SIGKILL with memory near the sandbox's limit: the OOM killer."""
        if self.signal != 9 or not self.memory_limit_mb or self.peak_memory_mb is None:
            return False
        return self.peak_memory_mb >= 0.8 * self.memory_limit_mb

    def to_dict(self) -> dict:
        return asdict(self)


def wrap(command: str) -> str:
    encoded = base64.b64encode(command.encode()).decode()
    return f"python3 -c {shlex.quote(_WRAPPER)} {encoded}"


def split_report(output: str) -> tuple[str, dict | None]:
    """The output without the report line, and the report (None if absent)."""
    match = None
    for match in _REPORT.finditer(output):
        pass  # the last one is ours; an earlier one would be the command's own
    if match is None:
        return output, None
    try:
        report = json.loads(match.group(1))
    except json.JSONDecodeError:
        return output, None
    return output[: match.start()] + output[match.end() :], report


def run_measured(
    sandbox: SandboxBackendProtocol, command: str, timeout: int | None = None
) -> tuple[ExecuteResponse, StepMeasure]:
    """Run `command` in `sandbox`; its result as the agent should see it, and what it cost."""

    def run(cmd: str) -> ExecuteResponse:
        if timeout is not None and execute_accepts_timeout(type(sandbox)):
            return sandbox.execute(cmd, timeout=timeout)
        return sandbox.execute(cmd)

    started = time.monotonic()
    result = run(wrap(command))
    output, report = split_report(result.output)
    if report is None and result.exit_code == 127 and "python3" in result.output:
        result = run(command)  # no python3 in this sandbox: run it unmeasured
        output = result.output
    measure = StepMeasure(
        command=command[:500],
        exit_code=result.exit_code,
        seconds=round(time.monotonic() - started, 3),
        **(report or {}),
    )
    if measure.signal is not None:
        output = output.rstrip("\n") + "\n" + _killed_note(measure)
    return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=result.truncated), measure


def _killed_note(m: StepMeasure) -> str:
    """Tells the agent why its command died, so it can try a lighter way."""
    if m.out_of_memory:
        limit = f" of the sandbox's {m.memory_limit_mb} MB" if m.memory_limit_mb else ""
        return (
            f"[Killed: the command ran out of memory (it reached {m.peak_memory_mb:.0f} MB{limit}). "
            "Use less memory: select fewer columns, filter or aggregate earlier, sample the rows, "
            "or process the data in chunks.]"
        )
    if m.signal == 9:
        return (
            "[Killed (signal 9), most likely for using too much memory. If so: select fewer "
            "columns, filter or aggregate earlier, sample the rows, or process in chunks.]"
        )
    return f"[Killed by signal {m.signal}.]"
