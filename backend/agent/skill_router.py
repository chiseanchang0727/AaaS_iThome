"""Names the skills relevant to each user message, before the agent starts on it.

Everything the classifier reads is built here: the labels (`load_skills`), the
state (`build_state`) and the wording (the constants below, passed in by
`pick_skills`). The classifier
itself, classify/jev_classifier.py, knows nothing about skills.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, NotRequired

import yaml
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from typesafe_sdk import AsyncTypeSafeClient

from classify import JevJudge

log = logging.getLogger(__name__)

# Wording measured on a 50-skill roster (jev-1.13). Rerun the live check in
# tests/test_skill_router.py before changing it.
QUESTION = (
    "Which of these skills is the right one to load to help with the user's `request`? "
    "Pick 'none' if no skill is needed or none of them covers what is asked."
)
NONE_DESCRIPTION = "No skill is needed: small talk, general knowledge, or a task none of the skills covers."
VERIFY = "Would the skill '{name}' help carry out the user's `request`? The skill: {description}"
VERIFY_CRITERIA = {
    "true": "The request asks for what this skill does, or for part of it.",
    "false": "The request asks for something else, or needs no skill.",
}


def load_skills(skills_dir: Path) -> dict[str, str]:
    """{folder: description} for every `<folder>/SKILL.md`, from its frontmatter.

    Keyed by folder, not frontmatter `name`: the folder is what the
    /skills/<folder>/SKILL.md path the model reads is built from.
    """
    skills = {}
    for path in sorted(Path(skills_dir).glob("*/SKILL.md")):
        _, front, _ = path.read_text().split("---", 2)
        skills[path.parent.name] = yaml.safe_load(front)["description"]
    return skills


def build_state(messages: list[AnyMessage]) -> dict[str, str] | None:
    """What the classifier reads: the latest user message, or None if there is none.

    Only the message itself: the wording above was measured on that. A
    follow-up ("now as a chart") may need the previous turn too; add it here as
    its own field and re-measure.
    """
    request = next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    return {"request": request} if request else None


def build_classifier(
    skills_dir: Path, *, model: str, client: AsyncTypeSafeClient | None = None
) -> JevJudge:
    """A classifier whose labels are every skill in `skills_dir`."""
    return JevJudge(load_skills(skills_dir), model=model, client=client)


async def pick_skills(
    classifier: JevJudge, state: dict[str, str], *, top_n: int = 1, multi: bool = False
) -> list[str]:
    """The skills for `state`, asked with this module's wording.

    multi=False: the `top_n` most likely (one request). multi=True: the most
    likely, plus places 2..`top_n` that a yes/no check confirms (two requests).
    """
    if multi:
        result = await classifier.select(
            state,
            question=QUESTION,
            none=NONE_DESCRIPTION,
            verify=VERIFY,
            verify_criteria=VERIFY_CRITERIA,
            shortlist=top_n,
        )
    else:
        result = await classifier.classify(
            state, question=QUESTION, none=NONE_DESCRIPTION, top_n=top_n
        )
    return result.selected


class SkillRouterState(AgentState):
    relevant_skills: NotRequired[list[str]]


class SkillRouterMiddleware(AgentMiddleware):
    """Classifies the latest user message once per run and tells the model the result.

    before_agent picks the skills and keeps them in state; every model call in
    the run then gets one extra block at the end of its system prompt:

        <skill_relevance>
        Relevant to the current request: trend-report. Read /skills/trend-report/SKILL.md ...
        </skill_relevance>

    The skill list the model already sees is untouched, so this only says where
    to look first. If the classifier fails, the run goes on without the block.

    Args:
        classifier: Usually `build_classifier(skills_dir, model=...)`.
        top_n: Name at most this many skills.
        multi: How to pick beyond the first. False: take the `top_n` most likely.
            True: take the first, then check places 2..`top_n` with a second
            request and keep those it confirms (`JevJudge.select`).
    """

    state_schema = SkillRouterState

    def __init__(self, classifier: JevJudge, *, top_n: int = 1, multi: bool = False) -> None:
        super().__init__()
        self.classifier = classifier
        self.top_n = top_n
        self.multi = multi

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        classifier_state = build_state(state.get("messages", []))
        if classifier_state is None:
            return None
        try:
            skills = await pick_skills(
                self.classifier, classifier_state, top_n=self.top_n, multi=self.multi
            )
        except Exception:
            log.warning("skill routing failed; running without a suggestion", exc_info=True)
            return {"relevant_skills": []}
        return {"relevant_skills": skills}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        skills = request.state.get("relevant_skills") or []
        if skills:
            request = request.override(system_message=_append(request.system_message, _block(skills)))
        return await handler(request)


def _block(skills: list[str]) -> str:
    reads = ", ".join(f"/skills/{name}/SKILL.md" for name in skills)
    return (
        "<skill_relevance>\n"
        f"Relevant to the current request: {', '.join(skills)}. Read {reads} before "
        "acting on it. Ignore this if it does not fit what the user actually asked for.\n"
        "</skill_relevance>"
    )


def _append(message: SystemMessage | None, text: str) -> SystemMessage:
    if message is None:
        return SystemMessage(content=text)
    if isinstance(message.content, str):
        return SystemMessage(content=f"{message.content}\n\n{text}")
    return SystemMessage(content=[*message.content, {"type": "text", "text": text}])
