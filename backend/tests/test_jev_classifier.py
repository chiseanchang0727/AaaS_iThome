"""JevClassifier against a scripted client: no API calls."""

import asyncio
from types import SimpleNamespace

import pytest

from classify import NONE, JevClassifier

LABELS = {"billing": "Charges and refunds", "shipping": "Delivery", "returns": "Exchanges"}
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


def classifier(*responses, **kwargs):
    kwargs = {"question": "Which team?", "none": NOTHING_FITS, **kwargs}
    return JevClassifier(LABELS, client=FakeClient(*responses), **kwargs)


def run(coro):
    return asyncio.run(coro)


# --- classify ------------------------------------------------------------------


def test_classify_ranks_every_label_and_offers_none():
    clf = classifier(choice(shipping=0.1, billing=0.85, returns=0.03, none=0.02))
    result = run(clf.classify("charged twice"))

    assert list(result.probabilities) == ["billing", "shipping", "returns", NONE]
    assert result.top == "billing" and result.selected == ["billing"]
    asked = clf._client.requests[0]
    assert asked["state"] == "charged twice"
    assert asked["questions"]["which"].criteria == {**LABELS, NONE: NOTHING_FITS}
    assert asked["model"] == "jev-1.13.0"


def test_none_first_means_no_label():
    result = run(classifier(choice(billing=0.2, shipping=0.05, returns=0.05, none=0.7)).classify("hi"))
    assert result.top is None and result.selected == []


def test_without_none_every_input_gets_a_label():
    clf = classifier(choice(billing=1.0, shipping=0.0, returns=0.0), none=None)
    run(clf.classify("x"))
    assert NONE not in clf._client.requests[0]["questions"]["which"].criteria


def test_none_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        JevClassifier({"none": "x"}, question="?", client=FakeClient())


def test_needs_a_label():
    with pytest.raises(ValueError, match="at least one"):
        JevClassifier({}, question="?", client=FakeClient())


# --- select --------------------------------------------------------------------


def test_select_keeps_runners_up_that_pass_their_own_check():
    clf = classifier(
        choice(billing=0.6, returns=0.35, shipping=0.04, none=0.01),
        nouls(returns=0.95, shipping=0.2),
        verify=VERIFY,
        verify_criteria=VERIFY_CRITERIA,
    )
    result = run(clf.select("x", threshold=0.9))

    assert result.selected == ["billing", "returns"]
    assert result.checks == {"returns": 0.95, "shipping": 0.2}
    # the second request asks only about the runners-up, never the top label or none
    second = clf._client.requests[1]["questions"]
    assert set(second) == {"returns", "shipping"}
    assert second["returns"].instructions == "Does 'returns' apply? Exchanges"


def test_select_skips_the_second_request_when_nothing_fits():
    clf = classifier(choice(billing=0.1, shipping=0.1, returns=0.1, none=0.7), verify=VERIFY)
    result = run(clf.select("x"))
    assert result.selected == [] and len(clf._client.requests) == 1


def test_select_needs_a_verify_question():
    with pytest.raises(ValueError, match="verify"):
        run(classifier().select("x"))
