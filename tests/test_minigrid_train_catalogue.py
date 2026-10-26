"""Independent catalogue composition checks without learners or game execution.

Synthetic complete public histories exercise provenance, ordered repeated row
exposure, role admission and owned transport arrays. No model is trained here.
"""

from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_train_catalogue as catalogue
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_minigrid_invariant_encoding import frame


def episode(identity="A-blue", *, context="A", actions=(0, 2, 6), cue=2, role="train"):
    initial = frame(wall_color=0, direction=1, cue_color=cue, key_color=cue, door_state=0)
    views = tuple(
        initial
        + frame(
            wall_color=index % 6,
            direction=(index + 1) % 4,
            cue_color=cue,
            key_color=cue,
            door_state=index % 3,
        )
        for index in range(len(actions) + 1)
    )
    return DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                index, views[index], action, 0.0, views[index + 1], index == len(actions) - 1, 7
            )
            for index, action in enumerate(actions)
        ),
        7,
    )


def records():
    return (episode(), episode("B-red", context="B", actions=(1, 3), cue=0))


def test_should_preserve_order_labels_spans_and_explicit_repeated_row_costs():
    sources = records()
    before = pickle.dumps(sources)
    plan = compose_catalogue_training(sources, 3)
    assert type(plan.training) is tuple and len(plan.training) == 3
    assert type(plan.source_spans) is tuple
    assert [
        (span.episode_id, span.context, span.start, span.stop) for span in plan.source_spans
    ] == [("A-blue", "A", 0, 3), ("B-red", "B", 3, 5)]
    assert plan.source_rows == 5 and plan.row_exposures == 15
    assert plan.input_array_bytes == 3 * 5 * (296 + 1) * 8
    assert plan.source_template_array_bytes == 5 * 297 * 8
    assert plan.validation_projection_rows == 10
    assert plan.validation_projected_array_equivalent_bytes == 10 * 296 * 8
    expected_features = np.array(
        [step.observation for source in sources for step in source.transitions], dtype=np.float64
    )
    expected_targets = np.array([[0], [2], [6], [1], [3]], dtype=np.float64)
    for wake, batch in enumerate(plan.training):
        assert batch.role == "train" and batch.context == "train-catalogue"
        assert batch.episode is None and batch.num_classes == 7
        assert batch.features.shape == (5, 296) and batch.targets.shape == (5, 1)
        assert batch.features.dtype == batch.targets.dtype == np.float64
        assert batch.features.flags.owndata and batch.targets.flags.owndata
        np.testing.assert_array_equal(batch.features, expected_features)
        np.testing.assert_array_equal(batch.targets, expected_targets)
        assert batch.row_ids == tuple(
            f"catalogue:{wake}:{source.episode_id}:step:{step.step}"
            for source in sources
            for step in source.transitions
        )
    row_ids = [identity for batch in plan.training for identity in batch.row_ids]
    assert len(set(row_ids)) == 15
    assert (
        sum(b.features.nbytes + b.targets.nbytes for b in plan.training) == plan.input_array_bytes
    )
    assert pickle.dumps(sources) == before


def test_should_transport_raw_current_history_without_early_projection_or_future_metadata():
    sources = records()
    first = compose_catalogue_training(sources, 1)
    last = sources[-1].transitions[-1]
    changed = (
        replace(sources[0], episode_id="renamed-A", context="other-A"),
        replace(
            sources[-1],
            context="other-B",
            transitions=(
                *sources[-1].transitions[:-1],
                replace(
                    last,
                    reward=1.0,
                    next_observation=last.next_observation[:148] + frame(cue_color=4),
                ),
            ),
        ),
    )
    second = compose_catalogue_training(changed, 1)
    np.testing.assert_array_equal(first.training[0].features, second.training[0].features)
    np.testing.assert_array_equal(first.training[0].targets, second.training[0].targets)
    # Raw wall colour and compass are intentional transport data; the existing
    # invariant learner owns its one projection at the learning boundary.
    assert first.training[0].features[0, 1] == -1.0
    assert first.training[0].features[0, 147] == pytest.approx(-1 / 3)
    assert first.source_spans[0].episode_id != second.source_spans[0].episode_id


def test_should_own_each_wake_and_keep_source_and_sibling_arrays_independent():
    sources = records()
    before = pickle.dumps(sources)
    plan = compose_catalogue_training(sources, 3)
    arrays = [value for batch in plan.training for value in (batch.features, batch.targets)]
    for index, left in enumerate(arrays):
        assert all(not np.shares_memory(left, right) for right in arrays[index + 1 :])
    second_before = plan.training[1].features.copy()
    plan.training[0].features.fill(0)
    plan.training[0].targets.fill(4)
    np.testing.assert_array_equal(plan.training[1].features, second_before)
    assert pickle.dumps(sources) == before
    object.__setattr__(sources[-1].transitions[-1], "action", 5)
    assert plan.training[1].targets[-1, 0] == 3.0


def test_should_preserve_numpy_and_python_random_state():
    numpy_before = pickle.dumps(np.random.get_state())
    python_before = random.getstate()
    compose_catalogue_training(records(), 2)
    assert pickle.dumps(np.random.get_state()) == numpy_before
    assert random.getstate() == python_before


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize("updates", [None, False, True, 0, -1, 129, 1.0, "1", np.int64(1)])
def test_should_refuse_unbounded_or_coerced_update_counts(updates):
    with pytest.raises(ValueError):
        compose_catalogue_training(records(), updates)


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "tuple_subclass",
        "empty",
        "too_many_sources",
        "wrong_record",
        "final_test",
        "probe_train",
        "probe_test",
        "extra_episode_field",
        "extra_transition_field",
        "incomplete",
        "wrong_classes",
        "wrong_width",
        "late_current",
        "late_next",
        "discontinuous",
        "changed_initial",
        "duplicate_ids",
        "too_many_rows",
    ],
)
def test_should_refuse_entire_invalid_catalogue_and_leave_evidence_untouched(fault):
    sources: Any = records()
    if fault == "list":
        sources = list(sources)
    elif fault == "tuple_subclass":
        sources = TupleSubclass(sources)
    elif fault == "empty":
        sources = ()
    elif fault == "too_many_sources":
        sources = tuple(episode(str(index), actions=(2,)) for index in range(17))
    elif fault == "wrong_record":
        sources = (sources[0], object())
    elif fault in ("final_test", "probe_train", "probe_test"):
        sources = (sources[0], replace(sources[-1], role=fault))
    elif fault in ("extra_episode_field", "extra_transition_field"):
        victim = sources[-1] if fault == "extra_episode_field" else sources[-1].transitions[-1]
        object.__setattr__(victim, "foreign", 1)
    elif fault == "incomplete":
        object.__setattr__(sources[-1].transitions[-1], "terminal", False)
    elif fault == "wrong_classes":
        object.__setattr__(sources[-1], "num_classes", 2)
    elif fault == "wrong_width":
        object.__setattr__(sources[-1].transitions[-1], "next_observation", frame())
    elif fault in ("late_current", "late_next"):
        field = "observation" if fault == "late_current" else "next_observation"
        values = list(getattr(sources[-1].transitions[-1], field))
        values[-2] += 0.01
        object.__setattr__(sources[-1].transitions[-1], field, tuple(values))
    elif fault == "discontinuous":
        object.__setattr__(sources[-1].transitions[-1], "observation", frame(cue_color=4) * 2)
    elif fault == "changed_initial":
        source = sources[-1]
        changed_first = frame(cue_color=4)
        transitions = tuple(
            replace(
                step,
                observation=changed_first + step.observation[148:],
                next_observation=changed_first + step.next_observation[148:],
            )
            for step in source.transitions
        )
        sources = (*sources[:-1], replace(source, transitions=transitions))
    elif fault == "duplicate_ids":
        sources = (sources[0], replace(sources[-1], episode_id=sources[0].episode_id))
    else:
        sources = tuple(episode(str(index), actions=(2,) * 16) for index in range(9))
    before = pickle.dumps(sources)
    with pytest.raises(ValueError):
        compose_catalogue_training(sources, 1)
    assert pickle.dumps(sources) == before


def test_should_admit_maximum_source_count_rows_and_update_count_within_storage_cap():
    sources = tuple(episode(str(index), actions=(2,) * 8) for index in range(16))
    largest_catalogue = compose_catalogue_training(sources, 1)
    assert len(largest_catalogue.source_spans) == 16 and largest_catalogue.source_rows == 128
    assert largest_catalogue.input_array_bytes == 128 * 297 * 8
    longest = compose_catalogue_training((episode(actions=(6,)),), 128)
    assert len(longest.training) == longest.row_exposures == 128


def test_should_refuse_total_retained_array_storage_before_projection(monkeypatch):
    # 128 rows * 56 complete owned wakes exceeds the independent transport cap,
    # even though both row and update limits pass individually.
    sources = tuple(episode(str(index), actions=(2,) * 16) for index in range(8))
    before = pickle.dumps(sources)
    assert 128 * 56 * 297 * 8 > 16 * 1024**2

    def forbidden(*args, **kwargs):
        pytest.fail("over-cap catalogue reached projection")

    monkeypatch.setattr(catalogue, "canonicalize_public_history", forbidden)
    with pytest.raises(ValueError):
        compose_catalogue_training(sources, 56)
    assert pickle.dumps(sources) == before


def test_should_refuse_observed_foreign_source_change_without_repair(monkeypatch):
    sources = records()
    original = catalogue.canonicalize_public_history
    changed = False

    def project(values):
        nonlocal changed
        result = original(values)
        if not changed:
            object.__setattr__(sources[-1].transitions[-1], "action", 5)
            changed = True
        return result

    monkeypatch.setattr(catalogue, "canonicalize_public_history", project)
    with pytest.raises(ValueError):
        compose_catalogue_training(sources, 1)
    assert changed and sources[-1].transitions[-1].action == 5
