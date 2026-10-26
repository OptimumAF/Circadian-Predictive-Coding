"""Both-value endpoint comparisons preserve mixed and negative outcomes."""

from copy import deepcopy
from typing import Any

import pytest

from src.adapters.key_door_reward_pilot import _compare_endpoints


def pair():
    rows: list[dict[str, Any]] = []
    shifts = (0.25, -0.5, 0.0, 0.75, -0.25, 0.125, 0.0, -0.125)
    for arm, native in enumerate((8, 16)):
        scores, episodes = {}, []
        for i in range(8):
            context, cue = "ABCD"[i // 2], 1 if i % 2 == 0 else -1
            key = f"{context}-{'red' if cue == 1 else 'blue'}"
            scores[key] = dict(
                ce=1.0 + i / 8 + (shifts[i] if arm else 0), accuracy=0.625 if arm else 0.5
            )
            episodes.append(
                dict(
                    context=context,
                    cue=cue,
                    checkpoint_updates=native,
                    success=i < (2 if arm else 4),
                    **{"return": 0.25 if arm else 0.5},
                )
            )
        rows.append(
            dict(
                stream_result=dict(checkpoints=[dict(updates=native, scores=scores)]),
                episodes=episodes,
                training=[None] * 8
                + [
                    dict(collection=dict(total_return=0), receipt=dict(diagnostic=dict(value=-0.0)))
                    for _ in range(8)
                ],
            )
        )
    return rows


def test_should_preserve_both_endpoint_values_and_all_signed_metric_directions():
    rows = pair()
    before = deepcopy(rows)
    result = _compare_endpoints(rows)
    assert rows == before
    assert result["endpoint_native_updates"] == dict(bc_only=8, warm_start_rl=16)
    assert result["mean_ce"] == dict(bc_only=1.4375, warm_start_rl=1.46875)
    assert result["ce_reduction_bc_minus_rl"] == -0.03125
    assert result["accuracy_gain_rl_minus_bc"] == 0.125
    assert result["success_gain_rl_minus_bc"] == -0.25
    assert result["return_gain_rl_minus_bc"] == -0.25
    assert len(result["endpoint_rows"]) == 8
    assert [p["ce_reduction_bc_minus_rl"] for p in result["endpoint_rows"]] == [
        -0.25,
        0.5,
        0.0,
        -0.75,
        0.25,
        -0.125,
        0.0,
        0.125,
    ]
    assert result["endpoint_rows"][0]["ce_bc"] == 1.0
    assert result["endpoint_rows"][0]["ce_rl"] == 1.25
    assert all(p["accuracy_gain_rl_minus_bc"] == 0.125 for p in result["endpoint_rows"])
    assert [p["success_gain_rl_minus_bc"] for p in result["endpoint_rows"]] == [
        0,
        0,
        -1,
        -1,
        0,
        0,
        0,
        0,
    ]
    assert all(p["return_gain_rl_minus_bc"] == -0.25 for p in result["endpoint_rows"])
    assert result["nonzero_reward_episodes"] == 0
    assert result["learning_benefit_established"] is False


def test_should_refuse_mismatched_endpoint_contexts_before_reporting_a_difference():
    rows = pair()
    rows[1]["episodes"][0]["cue"] = -1
    with pytest.raises(ValueError, match="context/cue"):
        _compare_endpoints(rows)


def test_should_refuse_a_partial_endpoint_without_selecting_the_remaining_contexts():
    rows = pair()
    for row in rows:
        row["episodes"].pop()
    with pytest.raises(ValueError, match="all eight"):
        _compare_endpoints(rows)


def test_should_refuse_duplicate_pairs_even_when_both_arms_have_eight_matching_rows():
    rows = pair()
    for row in rows:
        row["episodes"][-1] = deepcopy(row["episodes"][0])
    with pytest.raises(ValueError, match="all eight unique"):
        _compare_endpoints(rows)
