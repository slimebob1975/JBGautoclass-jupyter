"""Behavioral characterization of spot-check candidate selection (280o)."""
from types import SimpleNamespace

import numpy as np

from JBGSpotCheckSelection import consider_spot_check_candidate


def _run(pipeline=object(), score=0.7, stdev=0.1, failure="", best=0.6, best_std=0.2):
    state = SimpleNamespace(best_cv_score=best, best_stdev=best_std)
    chosen = []

    def select(score, deviation, previous_score, previous_deviation):
        return score > previous_score or (score == previous_score and deviation < previous_deviation)

    def update(**kwargs):
        chosen.append(kwargs)
        state.best_cv_score = kwargs["cv_score"]
        state.best_stdev = kwargs["cv_stdev"]

    result = consider_spot_check_candidate(
        state=state, pipeline=pipeline, preprocessor="pre", reduction="red",
        algorithm="alg", cv_score=score, cv_stdev=stdev,
        num_features=4, num_components=3, failure=failure,
        is_best_run_yet=select, on_best_candidate=update,
    )
    return result, chosen, state


def test_improving_candidate_updates_existing_state():
    result, chosen, state = _run()
    assert result == ("", True)
    assert len(chosen) == 1
    assert chosen[0]["num_features"] == 4
    assert chosen[0]["num_components"] == 3
    assert state.best_cv_score == 0.7


def test_stable_but_weaker_candidate_does_not_replace_winner():
    result, chosen, _ = _run(score=0.5)
    assert result == ("", True)
    assert not chosen


def test_tie_with_lower_deviation_wins():
    result, chosen, _ = _run(score=0.6, stdev=0.1)
    assert result == ("", True)
    assert len(chosen) == 1


def test_failed_candidate_preserves_failure_and_state():
    result, chosen, _ = _run(failure="SKIPPED")
    assert result == ("SKIPPED", False)
    assert not chosen


def test_unavailable_pipeline_is_not_successful():
    result, chosen, _ = _run(pipeline=None)
    assert result == ("", False)
    assert not chosen


def test_nonfinite_scores_are_rejected():
    for score, stdev in [(np.nan, 0.1), (np.inf, 0.1), (0.7, np.nan), (0.7, np.inf)]:
        result, chosen, _ = _run(score=score, stdev=stdev)
        assert result == ("", False)
        assert not chosen
