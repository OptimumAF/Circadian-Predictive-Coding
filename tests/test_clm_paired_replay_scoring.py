"""Scalar-only scoring guard checks; no model construction, fit or encoding."""

from copy import deepcopy

import pytest

from test_clm_paired_replay_comparison import comparison_module


def payload():
    arm = dict(
        complete=True,
        update_budget_commits=6,
        replay_budget_rows=18,
        readouts=[
            dict(phase=phase, probabilities=[[0.2, 0.3, 0.5] for _ in range(4)])
            for phase in ("before", "after")
        ],
    )
    return dict(complete=True, arms=dict(replay=arm, return_only=deepcopy(arm)))


@pytest.mark.parametrize("fault", ["complete", "updates", "rows", "readout"])
def test_should_refuse_all_incomplete_work_before_decoding_any_arm(monkeypatch, fault):
    import scripts.run_clm_temporal_sequence as study

    value = payload()
    arm = value["arms"]["return_only"]
    if fault == "complete":
        arm["complete"] = False
    elif fault == "updates":
        arm["update_budget_commits"] = 5
    elif fault == "rows":
        arm["replay_budget_rows"] = 17
    else:
        arm["readouts"].pop()

    def forbidden(*args, **kwargs):
        raise AssertionError("unfinished second arm admitted first-arm labels")

    monkeypatch.setattr(study, "_score_cached", forbidden)
    with pytest.raises(ValueError, match="both fixed paid"):
        comparison_module.score_pair(value)


def test_should_score_both_complete_fixed_arms_using_unchanged_cached_metrics():
    value = payload()
    before = comparison_module.encoded(value)
    result = comparison_module.score_pair(value)
    assert comparison_module.encoded(value) == before
    assert result["cached_typed_decodes"] == 32
    assert result["arms"]["replay"] == result["arms"]["return_only"]
    assert all(
        result["replay_minus_return_only"][key] == 0.0
        for key in ("accuracy", "nll_mean", "brier_mean", "ece_5bin")
    )
