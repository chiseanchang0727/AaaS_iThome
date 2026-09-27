"""The grounding eval's number matching. Wrong here means a wrong verdict."""

import pytest

from evals.grounding.scoring import contains_number, numbers_in, ungrounded


@pytest.mark.parametrize(
    "text",
    ["824,606", "824606", "824.6K", "824.61k", "0.82M", "~825K", "about 0.8M views"],
)
def test_formats_that_match_824606(text):
    assert contains_number(text, 824_606)


@pytest.mark.parametrize("text", ["830K", "0.83M", "824,000", "1.3M", "82,460"])
def test_formats_that_do_not_match_824606(text):
    assert not contains_number(text, 824_606)


def test_precision_follows_the_written_digits():
    assert contains_number("r = 0.03", 0.0255)
    assert not contains_number("r = 0.030", 0.0255)
    assert contains_number("r ≈ 0.026", 0.0255)


def test_percentages_match_both_readings():
    assert contains_number("12.5%", 12.5)
    assert contains_number("12.5%", 0.125)


def test_dates_and_years_are_not_numbers():
    assert numbers_in("On 2031-01-23, in 2031") == []


def test_words_after_a_number_are_not_suffixes():
    """'103 Kayaking' is 103, not 103,000; '5 Minutes' is 5."""
    assert [n.value for n in numbers_in("103 Kayaking videos, 5 Minutes")] == [103, 5]


def test_negative_numbers():
    assert contains_number("r = -0.12", -0.12)


# --- ungrounded ------------------------------------------------------------------


def test_a_number_seen_in_a_tool_result_is_grounded():
    assert ungrounded("Kayaking leads with 824.6K", ['[{"median": 824606.0}]'], "q") == []


def test_a_number_never_seen_is_flagged():
    assert ungrounded("Kayaking leads with 1.4M", ['[{"median": 824606.0}]'], "q") == ["1.4M"]


def test_small_whole_numbers_are_ignored():
    assert ungrounded("The top 3 channels", [], "q") == []


def test_numbers_from_the_question_are_ignored():
    assert ungrounded("The 90th percentile is ...", [], "tell me the 90th percentile") == []


def test_a_percentage_is_grounded_by_either_reading():
    assert ungrounded("like ratio 6.1%", ['{"like_ratio": 0.061}'], "q") == []


def test_each_ungrounded_number_is_listed_once():
    assert ungrounded("1.4M, again 1.4M", [], "q") == ["1.4M"]


# --- number words, ordinals, code ---------------------------------------------


@pytest.mark.parametrize("text", ["1.3 million", "1.28 Million views", "1,277.7 thousand"])
def test_number_words_scale_the_value(text):
    assert contains_number(text, 1_277_728)


def test_billion_as_a_word():
    assert contains_number("4.8 billion", 4_821_071_901)


def test_ordinals_are_not_numbers():
    """'90th percentile' names the statistic; 90 is not a measurement."""
    assert numbers_in("the 90th percentile, 1st place, 22nd, 3rd") == []


def test_numbers_in_backticks_are_method_not_result():
    answer = "Median 4 days, via `PERCENTILE_CONT(0.5)` and `.quantile(0.90)`"
    assert ungrounded(answer, [], "q") == []


def test_a_number_outside_backticks_is_still_checked():
    assert ungrounded("`PERCENTILE_CONT(0.5)` gives 1.4M", [], "q") == ["1.4M"]
