"""Fixed positive weighting keeps every TRAIN row and its declared order.

Scalar-only fixtures: no learner, random draw, optimizer, diagnostic or SDK work.
The recipe is declared in advance; these tests make no task-performance claim.
"""

from dataclasses import asdict, replace
import math
from typing import Any

import pytest

from scripts import push_cube_correction_weights as weights
from test_continuous_mixed_correction_fit import inputs


def groups():
    training, references = inputs()
    return tuple(
        zip(("corrections", "push", "a", "b", "train20"), (training, *references), strict=True)
    )


def test_should_assign_fixed_positive_mass_to_all148_rows_without_selection():
    original = groups()
    values, protocol = weights.build_balanced_correction_weights(original)
    assert type(values) is tuple and len(values) == 148
    assert all(type(value) is float and math.isfinite(value) and value > 0 for value in values)
    assert values[:50] == (49.0,) + (1.0,) * 49
    offset = 50
    for count in (18, 15, 50, 15):
        assert values[offset : offset + count] == (98.0 / count,) * count
        offset += count
    total = math.fsum(values)
    offset = 0
    for count in (50, 18, 15, 50, 15):
        assert math.fsum(values[offset : offset + count]) / total == pytest.approx(0.2)
        offset += count
    assert values[0] / total == pytest.approx(0.1)
    assert math.fsum(values[1:50]) / total == pytest.approx(0.1)


def test_should_record_exact_order_identity_raw_values_and_frozen_recipe():
    original = groups()
    values, protocol = weights.build_balanced_correction_weights(original)
    assert protocol == dict(
        definition="positive_train_prefix_group_weights_v1",
        group_order=["corrections", "push", "a", "b", "train20"],
        groups=[
            dict(group=name, episode_id=episode.episode_id, rows=len(episode.transitions))
            for name, episode in original
        ],
        values=list(values),
        rule="equal five-group mass; correction initial/remainder split equally",
        ideal_group_mass=0.2,
        ideal_correction_initial_mass=0.1,
    )


def test_should_leave_inputs_untouched_and_return_detached_metadata():
    original = groups()
    before = tuple(asdict(episode) for _, episode in original)
    values, first = weights.build_balanced_correction_weights(original)
    _, second = weights.build_balanced_correction_weights(original)
    first["values"][0] = 500.0
    first["groups"][0]["episode_id"] = "replacement"
    assert values[0] == second["values"][0] == 49.0
    assert second["groups"][0]["episode_id"] == original[0][1].episode_id
    assert tuple(asdict(episode) for _, episode in original) == before


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "pair_list",
        "order",
        "name",
        "count",
        "duplicate",
        "spec",
        "incomplete",
        "probe_train",
        "probe_test",
        "final_test",
        "non_episode",
    ],
)
def test_should_refuse_wrong_order_counts_identity_spec_or_role(fault):
    original = groups()
    invalid: Any = original
    if fault == "list":
        invalid = list(original)
    elif fault == "pair_list":
        invalid = (list(original[0]),) + original[1:]
    elif fault == "order":
        invalid = (original[1], original[0]) + original[2:]
    elif fault == "name":
        invalid = (("unknown", original[0][1]),) + original[1:]
    elif fault == "count":
        short = original[0][1].transitions[:-2] + (
            replace(original[0][1].transitions[-2], terminated=True),
        )
        invalid = (("corrections", replace(original[0][1], transitions=short)),) + original[1:]
    elif fault == "duplicate":
        invalid = (
            original[:1]
            + (("push", replace(original[1][1], episode_id=original[0][1].episode_id)),)
            + original[2:]
        )
    elif fault == "spec":
        spec = replace(
            original[1][1].spec, action_units=("other",) + original[1][1].spec.action_units[1:]
        )
        invalid = original[:1] + (("push", replace(original[1][1], spec=spec)),) + original[2:]
    elif fault == "incomplete":
        episode = original[0][1]
        object.__setattr__(episode.transitions[-1], "terminated", False)
        invalid = (("corrections", episode),) + original[1:]
    elif fault == "non_episode":
        invalid = (("corrections", asdict(original[0][1])),) + original[1:]
    else:
        invalid = (("corrections", replace(original[0][1], role=fault)),) + original[1:]
    with pytest.raises(ValueError):
        weights.build_balanced_correction_weights(invalid)
