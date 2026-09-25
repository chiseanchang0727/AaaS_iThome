"""Middleware that turns a skill from a suggestion into a precondition."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

# read_file's status header when it stopped before the end of the file, e.g.
# "@@ lines 1-3 of 10 | next offset 3 @@".
_TRUNCATED = re.compile(r"^@@ .*\bnext offset \d+.* @@$", re.MULTILINE)


class SkillEnforcerMiddleware(AgentMiddleware):
    """Blocks enforced tool calls (in after_model) until the tool's skill is read.

    Convention: the skill for tool `<tool>` lives at `/skills/<tool>/SKILL.md`.

    Runs in after_model, so a blocked call never reaches the tools node. On
    block it injects error ToolMessages and jumps back to the model
    (jump_to='model'), skipping any later after_model hooks and the tools node.
    after_model hooks run in reverse list order, so append this LAST to the
    middleware list if a HITL middleware should only see calls that passed:
    otherwise the user is asked to approve a call that is about to be blocked.

    Flow for an enforced tool (e.g. query_database):
      1. Model emits query_database (skill unread)
      2. after_model: blocked, jump_to model (tools node skipped)
      3. Model reads /skills/query_database/SKILL.md via read_file
      4. Model re-emits query_database, now with the skill's result in its
         history -> runs

    Args:
        skills_dir: Root directory containing skill folders.
        target_skills: Skill name(s) to enforce. If None, enforces any tool
            that has a SKILL.md file. If provided, only these are enforced.
    """

    def __init__(self, skills_dir: Path, target_skills: str | list[str] | None = None) -> None:
        super().__init__()
        self.skills_dir = Path(skills_dir)
        if target_skills is None:
            self._targets: set[str] | None = None
        elif isinstance(target_skills, str):
            self._targets = {target_skills}
        else:
            self._targets = set(target_skills)

    def _skill_path(self, tool_name: str) -> Path:
        """Local FS path, used only to check the skill file exists on disk."""
        return self.skills_dir / tool_name / "SKILL.md"

    def _skill_virtual_path(self, tool_name: str) -> str:
        """Virtual path the agent reads via the /skills/ composite route.

        Emitted in block messages and matched against read_file calls in
        history. Must not be the local FS path: the composite backend routes by
        the /skills/ prefix, and any other path falls through to the default
        (in-memory) backend and finds nothing.
        """
        return f"/skills/{tool_name}/SKILL.md"

    def _is_enforced(self, tool_name: str) -> bool:
        if self._targets is not None:
            return tool_name in self._targets
        return self._skill_path(tool_name).exists()

    def _get_path_arg(self, args: dict) -> str:
        """Extract the file path from tool args regardless of key name."""
        return args.get("path", "") or args.get("file_path", "")

    def _skill_was_read(self, tool_name: str, messages: list) -> bool:
        """True if the model has already *seen* this tool's SKILL.md.

        A read_file call is not enough; it needs a result in the history that
        succeeded and was not cut short. That rules out:

        - a read issued in the same turn as the enforced call: it has no result
          yet, so the call was written without the skill;
        - a read that failed (wrong path, permission): status="error";
        - a partial read (`limit` too small): read_file's header then carries
          `next offset N`, pointing at the unread rest.
        """
        skill_path = self._skill_virtual_path(tool_name)
        read_ids = {
            tc["id"]
            for msg in messages
            if isinstance(msg, AIMessage)
            for tc in msg.tool_calls
            if tc["name"] == "read_file" and self._get_path_arg(tc["args"]) == skill_path
        }
        return any(
            isinstance(msg, ToolMessage)
            and msg.tool_call_id in read_ids
            and msg.status != "error"
            and not _TRUNCATED.search(str(msg.content))
            for msg in messages
        )

    def _block_message(self, tool_name: str, tool_call_id: str) -> ToolMessage:
        return ToolMessage(
            content=(
                f"Skill check failed: you must read the skill for '{tool_name}' "
                f"before using it.\n"
                f"Call read_file with path='{self._skill_virtual_path(tool_name)}' "
                f"and read the whole file, then retry in a later turn."
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
        )

    def _skipped_message(self, tool_name: str, tool_call_id: str) -> ToolMessage:
        return ToolMessage(
            content=(
                f"Not run: another call in this turn failed its skill check. "
                f"Re-issue '{tool_name}' if you still need it."
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
        )

    def _check_skills(self, state: AgentState) -> dict[str, Any] | None:
        """Block enforced tool calls whose skill hasn't been read; jump to model.

        Scans the last AIMessage's tool_calls. If any enforced call is blocked,
        every call in that message gets a ToolMessage: jump_to skips the tools
        node for all of them, and a tool call left without a result is rejected
        by the API on the next model call.
        """
        messages = state.get("messages", [])
        last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        blocked = {
            tc["id"]
            for tc in last_ai_msg.tool_calls
            if self._is_enforced(tc["name"]) and not self._skill_was_read(tc["name"], messages)
        }
        if not blocked:
            return None

        replies = [
            self._block_message(tc["name"], tc["id"])
            if tc["id"] in blocked
            else self._skipped_message(tc["name"], tc["id"])
            for tc in last_ai_msg.tool_calls
        ]
        return {"messages": replies, "jump_to": "model"}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._check_skills(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._check_skills(state)
