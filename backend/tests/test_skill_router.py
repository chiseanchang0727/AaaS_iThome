"""The skill router, on a set of mock skills written into a temporary skills dir.

Most tests script the TypeSafe client, so they run anywhere. The last one asks
the real API how well the router picks among the mock skills; it is skipped
unless asked for:

    uv run --env-file ../.env pytest -m live tests/test_skill_router.py
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.skill_router import (
    NONE_DESCRIPTION,
    QUESTION,
    SkillRouterMiddleware,
    build_classifier,
    build_state,
    load_skills,
)
from classify import NONE

# Mock skills for a data-analyst agent. Several are deliberate lookalikes
# (the two charts, the two write-ups, the two SQL helpers) so the live test
# has to tell neighbours apart.
MOCK_SKILLS = {
    "query_database": "Rules for querying the built-in `videos` table: one row per video per trending day, dedupe before counting or summing.",
    "sql-explain": "Explain what an existing SQL query does, clause by clause, in plain language.",
    "data-dictionary": "Describe tables and columns: what each field means, its type, units and allowed values.",
    "clean-upload": "Fix a messy uploaded CSV: bad headers, mixed types, stray whitespace, broken dates, duplicate rows.",
    "join-datasets": "Combine an uploaded dataset with the videos table or another upload on a shared key.",
    "descriptive-stats": "Summary statistics for a column: mean, median, percentiles, spread, counts.",
    "correlation": "Measure how strongly two numeric variables move together (Pearson, Spearman) with significance.",
    "forecast": "Predict future values of a time series (e.g. next month's views) with a forecasting model.",
    "chart-interactive": "Make an interactive plotly chart saved as HTML that the user can hover and zoom in the chat.",
    "chart-static-png": "Make a static PNG or SVG image of a chart for pasting into slides, documents or email.",
    "trend-report": "House format for a written report on the trending data: headline with the number, a ranking table, one caveat.",
    "executive-summary": "Condense a long analysis into three bullet points for a busy manager, no tables or method.",
    "export-excel": "Export query results to an Excel (.xlsx) workbook, with one sheet per table.",
    "email-draft": "Draft an email that shares results with a colleague or stakeholder.",
    "translate-text": "Translate titles, descriptions or a finished answer into another language.",
}

# (request, the skill that should come first; None: no skill fits)
LABELLED = [
    ("Which channel has the most total views?", "query_database"),
    ("What does this SQL actually do? WITH per_video AS (SELECT video_id, MAX(views) ...", "sql-explain"),
    ("What's the difference between the views and likes columns, and what unit is comment_count in?", "data-dictionary"),
    ("The CSV I just uploaded has dates like '03/04/18' and '2018-4-3' mixed together", "clean-upload"),
    ("I uploaded my channel's revenue sheet, can you line it up with the trending videos by video id?", "join-datasets"),
    ("Give me the median, 25th and 75th percentile of views per video", "descriptive-stats"),
    ("Do videos with more likes also get more comments? How strong is that relationship?", "correlation"),
    ("How many views will Music videos get next month?", "forecast"),
    ("Plot daily views for the top 5 categories so I can hover over the lines", "chart-interactive"),
    ("I need a PNG of the category bar chart for my slides", "chart-static-png"),
    ("Write up a report on which categories are winning this quarter", "trend-report"),
    ("My boss has two minutes. Summarize what we found about Music vs Gaming", "executive-summary"),
    ("Can I get the top 100 videos as a spreadsheet?", "export-excel"),
    ("Write a note to Mei explaining the viral video results", "email-draft"),
    ("Please give me this answer in Traditional Chinese", "translate-text"),
    ("Hi! What can you do?", None),
    ("Thanks, that's all for today", None),
    ("What's the capital of Australia?", None),
    ("Write me a poem about cats", None),
]


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """MOCK_SKILLS as `<folder>/SKILL.md` files, the way skills/ holds them."""
    for folder, description in MOCK_SKILLS.items():
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "SKILL.md").write_text(
            f"---\nname: {folder}\ndescription: {description!r}\n---\n\n# {folder}\n"
        )
    return tmp_path


# --- a scripted TypeSafe client ------------------------------------------------


class FakeClient:
    """Answers each request from a queue; records what was asked."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def system_one(self, state, questions, model=None):
        self.requests.append({"state": state, "questions": questions, "model": model})
        return self.responses.pop(0)


def choice(**probabilities):
    return SimpleNamespace(
        choices={"which": SimpleNamespace(probabilities=probabilities, confidence=0.8)}
    )


def picks(skill: str) -> SimpleNamespace:
    """A Choice answer that puts `skill` first."""
    rest = {name: 0.0 for name in [*MOCK_SKILLS, NONE] if name != skill}
    return choice(**{skill: 0.9, **rest})


def router(skills_dir, *responses, top_n=1, multi=False) -> SkillRouterMiddleware:
    classifier = build_classifier(skills_dir, model="jev-1.13.0", client=FakeClient(*responses))
    return SkillRouterMiddleware(classifier, top_n=top_n, multi=multi)


def run(coro):
    return asyncio.run(coro)


# --- the inputs the router builds ------------------------------------------------


def test_load_skills_reads_every_description(skills_dir):
    assert load_skills(skills_dir) == MOCK_SKILLS


def test_load_skills_keys_by_folder_not_frontmatter_name(tmp_path):
    (tmp_path / "trend-report").mkdir()
    (tmp_path / "trend-report" / "SKILL.md").write_text(
        "---\nname: a-different-name\ndescription: Report format.\n---\n"
    )
    assert load_skills(tmp_path) == {"trend-report": "Report format."}


def test_the_state_is_the_latest_user_message():
    messages = [HumanMessage("old"), AIMessage("hi"), HumanMessage("Write up a report")]
    assert build_state(messages) == {"request": "Write up a report"}
    assert build_state([AIMessage("no user yet")]) is None


def test_the_classifier_asks_about_every_mock_skill(skills_dir):
    mw = router(skills_dir, picks("forecast"))
    run(mw.abefore_agent({"messages": [HumanMessage("next month?")]}, None))

    which = mw.classifier._client.requests[0]["questions"]["which"]
    assert which.instructions == QUESTION
    assert which.criteria == {**MOCK_SKILLS, NONE: NONE_DESCRIPTION}


# --- the middleware ------------------------------------------------------------


def test_before_agent_keeps_the_pick_in_state(skills_dir):
    mw = router(skills_dir, picks("trend-report"))
    state = {"messages": [HumanMessage("Write up a report")]}
    assert run(mw.abefore_agent(state, None)) == {"relevant_skills": ["trend-report"]}


def test_top_n_names_that_many_skills(skills_dir):
    ranked = choice(**{"forecast": 0.6, "chart-interactive": 0.3, "correlation": 0.08, "none": 0.02})
    mw = router(skills_dir, ranked, top_n=2)
    state = {"messages": [HumanMessage("Forecast next month and chart it")]}
    assert run(mw.abefore_agent(state, None)) == {"relevant_skills": ["forecast", "chart-interactive"]}


def test_nothing_fits_means_no_skills(skills_dir):
    mw = router(skills_dir, picks(NONE))
    assert run(mw.abefore_agent({"messages": [HumanMessage("hi")]}, None)) == {"relevant_skills": []}


def test_multi_also_keeps_a_confirmed_runner_up(skills_dir):
    both = choice(**{"forecast": 0.7, "chart-interactive": 0.25, "correlation": 0.05})
    confirmed = SimpleNamespace(nouls={"chart-interactive": SimpleNamespace(noul=0.97),
                                       "correlation": SimpleNamespace(noul=0.1)})
    mw = router(skills_dir, both, confirmed, top_n=3, multi=True)
    state = {"messages": [HumanMessage("Forecast next month and chart it")]}
    assert run(mw.abefore_agent(state, None)) == {"relevant_skills": ["forecast", "chart-interactive"]}


def test_a_failed_request_routes_nothing(skills_dir):
    class Broken:
        async def system_one(self, **_):
            raise TimeoutError

    mw = router(skills_dir)
    mw.classifier._client = Broken()
    assert run(mw.abefore_agent({"messages": [HumanMessage("x")]}, None)) == {"relevant_skills": []}


def test_model_calls_get_the_block_only_when_a_skill_was_picked(skills_dir):
    mw = router(skills_dir)
    seen = []

    async def handler(request):
        seen.append(request.system_message)
        return "ok"

    def call(skills):
        request = SimpleNamespace(
            state={"relevant_skills": skills},
            system_message=SystemMessage("You are a data analyst."),
        )
        request.override = lambda **kw: SimpleNamespace(**{**vars(request), **kw})
        return run(mw.awrap_model_call(request, handler))

    call(["trend-report"])
    call([])
    assert "<skill_relevance>" in seen[0].content
    assert "/skills/trend-report/SKILL.md" in seen[0].content
    assert seen[0].content.startswith("You are a data analyst.")
    assert seen[1].content == "You are a data analyst."


def test_the_block_reaches_the_model_inside_a_deep_agent(skills_dir):
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Recording(FakeMessagesListChatModel):
        seen: list = []

        def bind_tools(self, tools, **kwargs):
            return self

        async def _agenerate(self, messages, *args, **kwargs):
            self.seen.append(messages)
            return await super()._agenerate(messages, *args, **kwargs)

    model = Recording(responses=[AIMessage("done")])
    agent = create_deep_agent(model=model, middleware=[router(skills_dir, picks("trend-report"))])

    state = run(agent.ainvoke({"messages": [{"role": "user", "content": "Write up a report"}]}))

    assert state["relevant_skills"] == ["trend-report"]
    system = model.seen[0][0]
    assert isinstance(system, SystemMessage)
    assert "/skills/trend-report/SKILL.md" in system.text


# --- live: the real API on the mock skills ---------------------------------------


def print_result(request: str, want: str | None, result) -> None:
    """One request: what was asked, what was expected, and every skill's verdict.

    T marks what the router would load; the number is the probability the
    classifier gave that skill. `none` is the nothing-fits option.
    """
    mark = "ok  " if result.top == want else "MISS"
    print(f"\n[{mark}] {request}")
    print(f"       expected: {want or NONE}   got: {result.top or NONE}   confidence: {result.confidence:.2f}")
    for name, p in result.probabilities.items():
        verdict = "T" if name in result.selected or (name == NONE and not result.selected) else "F"
        print(f"       {verdict}  {p:.2f}  {name}")


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="needs $TYPESAFE_API_KEY")
def test_live_router_picks_the_right_mock_skill(skills_dir):
    classifier = build_classifier(skills_dir, model="jev-1.13.0")

    async def classify_all():
        return await asyncio.gather(*(classifier.classify(build_state([HumanMessage(q)]), question=QUESTION, none=NONE_DESCRIPTION) for q, _ in LABELLED))

    results = run(classify_all())
    for (request, want), result in zip(LABELLED, results):
        print_result(request, want, result)
    wrong = [(q, want, r.top) for (q, want), r in zip(LABELLED, results) if r.top != want]
    print(f"\n{len(LABELLED) - len(wrong)}/{len(LABELLED)} requests routed as expected")
    # Measured 54/55 + 9/10 on a 50-skill roster; allow one miss here.
    assert len(wrong) <= 1, wrong
