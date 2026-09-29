"""JevJudge against a scripted client: no API calls."""

import asyncio
from types import SimpleNamespace

import pytest

from classify import NONE, JevJudge, Level

LABELS = {"billing": "Charges and refunds", "shipping": "Delivery", "returns": "Exchanges"}
QUESTION = "Which team?"
NOTHING_FITS = "None of these teams handles it."
VERIFY = "Does '{name}' apply? {description}"
VERIFY_CRITERIA = {"true": "It applies.", "false": "It does not."}


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


def nouls(**values):
    return SimpleNamespace(nouls={k: SimpleNamespace(noul=v) for k, v in values.items()})


def scores(**values):
    return SimpleNamespace(scores={
        k: SimpleNamespace(score=v, confidence=0.7, probabilities={"0": 0.1, "1": 0.2, "2": 0.7})
        for k, v in values.items()
    })


def jev(*responses):
    return JevJudge(LABELS, client=FakeClient(*responses))


def run(coro):
    return asyncio.run(coro)


def test_needs_a_label():
    with pytest.raises(ValueError, match="at least one"):
        JevJudge({}, client=FakeClient())


def test_none_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        JevJudge({"none": "x"}, client=FakeClient())


# --- classify (Choice) -----------------------------------------------------------


def test_classify_ranks_every_label_and_offers_none():
    j = jev(choice(shipping=0.1, billing=0.85, returns=0.03, none=0.02))
    result = run(j.classify("charged twice", question=QUESTION, none=NOTHING_FITS))

    assert list(result.probabilities) == ["billing", "shipping", "returns", NONE]
    assert result.top == "billing" and result.selected == ["billing"]
    asked = j._client.requests[0]
    assert asked["state"] == "charged twice"
    assert asked["questions"]["which"].instructions == QUESTION
    assert asked["questions"]["which"].criteria == {**LABELS, NONE: NOTHING_FITS}
    assert asked["model"] == "jev-1.13.0"


def test_none_first_means_no_label():
    j = jev(choice(billing=0.2, shipping=0.05, returns=0.05, none=0.7))
    result = run(j.classify("hi", question=QUESTION, none=NOTHING_FITS))
    assert result.top is None and result.selected == []


def test_without_none_every_input_gets_a_label():
    j = jev(choice(billing=1.0, shipping=0.0, returns=0.0))
    run(j.classify("x", question=QUESTION))
    assert NONE not in j._client.requests[0]["questions"]["which"].criteria


def test_top_n_keeps_the_most_likely_labels():
    j = jev(choice(billing=0.5, returns=0.3, shipping=0.15, none=0.05))
    assert run(j.classify("x", question=QUESTION, none=NOTHING_FITS, top_n=2)).selected == ["billing", "returns"]


def test_top_n_stops_at_none():
    j = jev(choice(billing=0.5, none=0.3, returns=0.15, shipping=0.05))
    assert run(j.classify("x", question=QUESTION, none=NOTHING_FITS, top_n=3)).selected == ["billing"]


def test_top_n_must_be_positive():
    with pytest.raises(ValueError, match="top_n"):
        run(jev().classify("x", question=QUESTION, top_n=0))


# --- check_each (Noul) -----------------------------------------------------------


def test_check_each_asks_one_yes_no_per_label_in_one_request():
    j = jev(nouls(billing=0.9, shipping=0.8, returns=0.05))
    result = run(j.check_each("x", question=VERIFY, criteria=VERIFY_CRITERIA))

    # independent: two can be high at once
    assert result == {"billing": 0.9, "shipping": 0.8, "returns": 0.05}
    assert len(j._client.requests) == 1
    asked = j._client.requests[0]["questions"]
    assert asked["shipping"].instructions == "Does 'shipping' apply? Delivery"
    assert asked["shipping"].criteria == VERIFY_CRITERIA


def test_check_each_can_ask_about_some_labels_only():
    j = jev(nouls(returns=0.4))
    assert run(j.check_each("x", question=VERIFY, only=["returns"])) == {"returns": 0.4}
    assert set(j._client.requests[0]["questions"]) == {"returns"}


def test_labels_can_point_into_the_state():
    """Items that live in the state (conversation turns) are named by path."""
    turns = JevJudge({"turn_1": "`earlier_turns[0]`"}, client=FakeClient(nouls(turn_1=0.97)))
    question = "Does `current_request` need {description} ({name})?"
    assert run(turns.check_each({"current_request": "..."}, question=question)) == {"turn_1": 0.97}
    asked = turns._client.requests[0]["questions"]["turn_1"].instructions
    assert asked == "Does `current_request` need `earlier_turns[0]` (turn_1)?"


# --- score_each (Score) ----------------------------------------------------------


def test_score_each_rates_every_label_on_the_levels():
    j = jev(scores(billing=1.6, shipping=0.2, returns=0.0))
    levels = ["can wait", "this week", "today"]
    result = run(j.score_each("x", question="How urgent for {name}?", levels=levels))

    assert result["billing"] == Level(score=1.6, confidence=0.7, probabilities={0: 0.1, 1: 0.2, 2: 0.7})
    asked = j._client.requests[0]["questions"]["billing"]
    assert asked.instructions == "How urgent for billing?" and list(asked.criteria) == levels


def test_score_each_needs_two_to_ten_levels():
    with pytest.raises(ValueError, match="2 to 10"):
        run(jev().score_each("x", question="?", levels=["only one"]))


# --- select (Choice, then Noul) ----------------------------------------------------


def test_select_keeps_runners_up_that_pass_their_own_check():
    j = jev(
        choice(billing=0.6, returns=0.35, shipping=0.04, none=0.01),
        nouls(returns=0.95, shipping=0.2),
    )
    result = run(j.select("x", question=QUESTION, verify=VERIFY, verify_criteria=VERIFY_CRITERIA,
                          none=NOTHING_FITS, threshold=0.9))

    assert result.selected == ["billing", "returns"]
    assert result.checks == {"returns": 0.95, "shipping": 0.2}
    # the second request asks only about the runners-up, never the top label or none
    assert set(j._client.requests[1]["questions"]) == {"returns", "shipping"}


def test_select_skips_the_second_request_when_nothing_fits():
    j = jev(choice(billing=0.1, shipping=0.1, returns=0.1, none=0.7))
    result = run(j.select("x", question=QUESTION, verify=VERIFY, none=NOTHING_FITS))
    assert result.selected == [] and len(j._client.requests) == 1
