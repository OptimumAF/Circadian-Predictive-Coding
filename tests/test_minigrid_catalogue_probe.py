"""Real owned BP probe/old-A/return from a joint54-row native checkpoint.

Synthetic fixture labels demonstrate execution/isolation, not task acquisition.
One invocation plans34BPupdates396rows and32probe/readout predictions288rows.
"""

from time import monotonic
import pickle
from typing import Any

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSampler
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import bound, train_sources


def test_should_keep_joint_initial_rows_and_separate_probe_return_charges():
    source, sources = bound("bp"), train_sources()
    initial = source.snapshot_state()
    plan = compose_catalogue_training(tuple(pilot.shared._episode(p) for p in sources), 2)
    wake = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=60), monotonic
    )
    stream = AttributionStream(
        source,
        plan.training,
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=wake,
        training_only=True,
    )
    actor = stream._runtime.actor.snapshot_state().state
    stream.run_until(2)
    checkpoint = stream._runtime.candidate_snapshot().state
    result = stream.finish()
    assert result["training_examples"] == 108
    raw_train, raw_test = raw_probe_data()
    data = (pilot._history_packets(raw_train), pilot._history_packets(raw_test))
    readouts = sources[:2]
    before_data = pickle.dumps((data, readouts))
    slot: dict[str, Any] = dict(method="public_invariant_bp", checkpoint_updates=2, report=None)
    sampler = ProcessRssSampler()
    assert sampler.sample() is not None
    completed = pilot._run_probe(
        slot,
        {"source": source, "row": {"method": "public_invariant_bp"}},
        checkpoint,
        data,
        sampler=sampler,
        deadline=monotonic() + 60,
        check_boundary=lambda: None,
        training_readouts=readouts,
        return_to_a=True,
        catalogue_rows=54,
    )
    completed()
    assert slot["status"] == "complete", slot["reason"]
    assert slot["catalogue_rows_per_update"] == 54
    assert slot["expected_training_examples"] == slot["native_training_examples_before"] == 108
    assert slot["native_updates_before"] == 2
    assert slot["charged_updates"] == 16 and slot["return_charged_updates"] == 16
    assert slot["report"]["completed_training_examples"] == 144
    assert slot["report"]["return_phase"]["completed_training_examples"] == 144
    assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert slot["budget_binding_preserved"] is slot["return_budget_binding_preserved"] is True
    assert wake.updates_completed == 2
    assert _same_state(source.snapshot_state(), initial)
    assert _same_state(stream._runtime.actor.snapshot_state().state, actor)
    assert checkpoint.completed_updates == 2 and checkpoint.training_examples == 108
    assert pickle.dumps((data, readouts)) == before_data
    # Equal-valued retyping after success cannot qualify a preceding paid probe.
    slot["expected_training_examples"] = 108.0
    with pytest.raises(ValueError, match="row binding changed"):
        completed()
    assert slot["charged_updates"] == slot["return_charged_updates"] == 16


@pytest.mark.parametrize(
    "field,value", [("catalogue_rows_per_update", 54.0), ("expected_training_examples", False)]
)
def test_should_preserve_original_initial_row_metadata_after_callback_refusal(field, value):
    source = bound("bp")
    original = source.snapshot_state()
    raw_train, raw_test = raw_probe_data()
    data = (pilot._history_packets(raw_train), pilot._history_packets(raw_test))
    before = pickle.dumps(data)
    slot: dict[str, Any] = dict(method="public_invariant_bp", checkpoint_updates=0, report=None)

    def corrupt():
        if "native_training_examples_before" in slot:
            slot[field] = value

    sampler = ProcessRssSampler()
    assert sampler.sample() is not None
    with pytest.raises(ValueError, match="row binding changed"):
        pilot._run_probe(
            slot,
            {"source": source, "row": {"method": "public_invariant_bp"}},
            original,
            data,
            sampler=sampler,
            deadline=monotonic() + 30,
            check_boundary=corrupt,
            catalogue_rows=54,
        )
    assert slot["status"] == "failed" and "row binding changed" in slot["reason"]
    assert (
        type(slot["catalogue_rows_per_update"]) is int and slot["catalogue_rows_per_update"] == 54
    )
    assert (
        type(slot["expected_training_examples"]) is int and slot["expected_training_examples"] == 0
    )
    assert slot["charged_updates"] in (None, 0)
    assert _same_state(original, source.snapshot_state())
    assert pickle.dumps(data) == before
