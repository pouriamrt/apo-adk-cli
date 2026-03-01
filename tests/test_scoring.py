from apo.evaluation.reference import reference_score
from apo.evaluation.scorer import get_scorer


def test_reference_exact_match():
    score = reference_score("Paris", "Paris")
    assert score == 1.0


def test_reference_case_insensitive():
    score = reference_score("paris", "Paris")
    assert score >= 0.9


def test_reference_partial_match():
    score = reference_score("The capital is Paris", "Paris")
    assert 0.3 < score < 1.0


def test_reference_no_match():
    score = reference_score("London", "Paris")
    assert score < 0.3


def test_get_scorer_auto_with_reference():
    scorer = get_scorer("auto")
    task = {"input": "x", "expected_output": "y"}
    assert scorer.mode_for_task(task) == "reference"


def test_get_scorer_auto_without_reference():
    scorer = get_scorer("auto")
    task = {"input": "x"}
    assert scorer.mode_for_task(task) == "llm-judge"
