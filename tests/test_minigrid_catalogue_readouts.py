"""Native joint-catalogue checkpoints use original isolated TRAIN readouts."""

from copy import deepcopy
import pickle
from time import monotonic

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_invariant_learnability import bound, train_sources


@pytest.mark.parametrize("method", ("bp", "pc", "neutral_cpc", "gated_cpc"))
def test_should_score_joint_checkpoints_with_explicit_native_rows_and_isolated_owners(method):
    source = bound(method)
    original = source.snapshot_state()
    sources = train_sources()
    before = pickle.dumps(sources)
    plan = compose_catalogue_training(tuple(pilot.shared._episode(p) for p in sources), 2)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=60), monotonic
    )
    readouts = []

    def observe(point, state, arrived):
        if point == 2:
            # An ordinary nine-row checkpoint cannot silently accept joint clocks.
            with pytest.raises(ValueError):
                pilot._invariant_snapshot(state, point, f"public_invariant_{method}")
        raw = pilot._score_checkpoint(
            source,
            state,
            method=f"public_invariant_{method}",
            updates=point,
            sources=sources,
            budget=budget,
            check_boundary=lambda: None,
            catalogue_rows=54,
        )
        assert raw["status"] == "complete", raw["reason"]
        assert raw["catalogue_rows_per_update"] == 54
        assert raw["expected_training_examples"] == 54 * point
        assert raw["source_unchanged"] is raw["state_unchanged"] is True
        assert len(raw["scores"]) == 6
        assert raw["work"]["score_forward_calls_completed"] == 6
        assert raw["work"]["score_rows_completed"] == 54
        readouts.append(raw)

    stream = AttributionStream(
        source,
        plan.training,
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=budget,
        observe_checkpoint=observe,
        training_only=True,
    )
    actor_before = stream._runtime.actor.snapshot_state().state
    stream.run_until(2)
    result = stream.finish()
    assert [row["updates"] for row in readouts] == [0, 2]
    assert result["completed_updates"] == 2 and result["training_examples"] == 108
    assert result["score_forward_calls"] == 0
    assert _same_state(original, source.snapshot_state())
    assert _same_state(actor_before, stream._runtime.actor.snapshot_state().state)
    assert pickle.dumps(sources) == before


@pytest.mark.parametrize(
    "field,value", [("catalogue_rows_per_update", 54.0), ("expected_training_examples", False)]
)
def test_should_refuse_retyped_row_evidence_without_erasing_paid_score(monkeypatch, field, value):
    source, sources = bound("bp"), train_sources()
    original = source.snapshot_state()
    published = None
    saved = pilot.probability_scores

    def corrupt(probabilities, targets, count):
        nonlocal published
        result = saved(probabilities, targets, count)
        assert published is not None
        published[field] = value
        return result

    # Why: the report is local until return. Test-only frame access injects a
    # foreign callback write without adding a production mutation/debug hook.
    import inspect

    def boundary():
        nonlocal published
        caller = inspect.currentframe()
        assert caller is not None and caller.f_back is not None
        guard = caller.f_back
        assert guard.f_back is not None
        published = guard.f_back.f_locals["report"]

    monkeypatch.setattr(pilot, "probability_scores", corrupt)
    raw = pilot._score_checkpoint(
        source,
        deepcopy(original),
        method="public_invariant_bp",
        updates=0,
        sources=sources,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0),
        check_boundary=boundary,
        catalogue_rows=54,
    )
    assert raw["status"] == "failed" and "evidence changed" in raw["reason"]
    assert type(raw["catalogue_rows_per_update"]) is int and raw["catalogue_rows_per_update"] == 54
    assert type(raw["expected_training_examples"]) is int and raw["expected_training_examples"] == 0
    assert raw["work"]["score_forward_calls_completed"] == 1
    assert raw["work"]["score_rows_completed"] == 9
    assert len(raw["scores"]) == 1
    assert _same_state(original, source.snapshot_state())
