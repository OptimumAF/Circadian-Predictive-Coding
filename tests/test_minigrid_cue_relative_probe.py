"""Cue-relative public admission and disposable native F16/A16 integration.

Five native cases plan320 updates8640rows within300 scores3000rows. These are
synthetic engineering frames, not teacher, acquisition or advantage evidence.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_cue_relative_route as route
from src.adapters import minigrid_learnability as pilot
from src.adapters.cue_relative_torch_feature_learner import (
    CueRelativeTorchFeatureLearner,
    CueRelativeTorchFeatureSnapshot,
)
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app import toy_execution_budget as budgets
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_cue_relative_encoding import CUE_RELATIVE_HISTORY_ENCODING_ID
from src.shared.torch_runtime import require_torch
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_encoding import change_pixel
from test_minigrid_invariant_learnability import episode, train_sources
from test_minigrid_joint_catalogue_route import FLAGS as PREVIOUS_FLAGS
from test_torch_feature_learner import assert_tree, tensors
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
FLAGS: dict[str, Any] = dict(**PREVIOUS_FLAGS, joint_catalogue=True)


def raw_f_data(*, distinct_marker: bool = False):
    """The existing moving-marker fixture has two initial Balls; retain one cue."""
    cohorts = []
    for cohort in raw_probe_data():
        packets = []
        for packet in cohort:
            original = packet.episode
            assert original is not None
            transitions = list(original.transitions)
            initial = change_pixel(
                transitions[0].observation, 5, (7 if distinct_marker else 3, 0, 0)
            )
            transitions[0] = replace(transitions[0], observation=initial)
            packets.append(pilot._packet(replace(original, transitions=tuple(transitions))))
        cohorts.append(tuple(packets))
    return tuple(cohorts)


def history_f_data():
    return tuple(pilot._history_packets(cohort) for cohort in raw_f_data())


def test_should_preserve_public_f_novelty_with_one_initial_cue_and_a_noncue_marker():
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    data = raw_f_data(distinct_marker=True)
    route.validate_histories(
        tuple(
            pilot.shared._episode(packet)
            for cohort in data
            for packet in pilot._history_packets(cohort)
        )
    )
    support = pilot.summarize_probe_support(raw, data, catalogue_sources=True)
    assert support["all_canonical_inputs_unseen"] is True
    assert all(
        record["canonical"]["unmatched_input_rows"] == 9 for record in support["probe"]["packets"]
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_explicit_recipe_with_original_native_math_and_dose(monkeypatch, method):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    options = {**FLAGS, "qualification_method": method}
    old = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(**options, cue_relative_history=True)
    assert selected["methods"] == [f"public_cue_relative_{method}"]
    assert selected["cue_relative_history"]["encoding_id"] == CUE_RELATIVE_HISTORY_ENCODING_ID
    assert CUE_RELATIVE_HISTORY_ENCODING_ID in repr(selected)
    for key in (
        "seed",
        "learning_rate",
        "momentum",
        "inference_steps",
        "hidden_dim",
        "inference_learning_rate",
        "num_classes",
        "task_sequence",
        "checkpoint_updates",
    ):
        assert selected[key] == old[key]
    assert selected["planned_work"]["native_train_calls"] == 192
    for name in ("old_task_readouts", "return_phase"):
        assert set(selected[name]["input_array_bytes_by_method"]) == {
            f"public_cue_relative_{method}"
        }
    assert selected["planned_work"]["training_rows"] == 7488
    assert selected["planned_work"]["final_episodes"] == 0


@pytest.mark.parametrize("options", [{}, dict(public_history=True), PREVIOUS_FLAGS, FLAGS])
def test_should_preserve_default_protocol_byte_values(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, cue_relative_history=False
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonbool_selection_before_resources(monkeypatch, value):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**FLAGS, cue_relative_history=value)


@pytest.mark.parametrize(
    "key",
    [
        "public_history",
        "invariant_history",
        "joint_catalogue",
        "distinct_probe",
        "old_task_readouts",
        "return_to_a",
    ],
)
def test_should_require_complete_existing_isolated_route(monkeypatch, key):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**FLAGS, key: False}, cue_relative_history=True)


@pytest.mark.parametrize(
    "key",
    [
        "activation_health",
        "gradient_geometry",
        "prior_gradient_geometry",
        "probe_rollouts",
        "deferred_probe_rollouts",
        "weighted_bp",
        "pc_update_diagnostics",
    ],
)
def test_should_refuse_projection_specific_or_different_objective_options(monkeypatch, key):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    options: dict[str, Any] = {**FLAGS, "cue_relative_history": True, key: True}
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)


def test_should_admit_full_causal_next_frames_without_projecting_transport_arrays():
    sources, data = train_sources(), history_f_data()
    before = pickle.dumps((sources, data))
    pilot._invariant_fit_sources(sources, cue_relative_history=True)
    pilot._invariant_probe_inputs(data, cue_relative_history=True)
    assert pickle.dumps((sources, data)) == before
    for is_probe in (False, True):
        packets = data[0] if is_probe else sources
        last = packets[-1]
        original = last.episode
        assert original is not None
        transitions = list(original.transitions)
        # Corrupt only the final next frame, which is absent from features.
        values = list(transitions[-1].next_observation)
        values[-2] = 0.125
        transitions[-1] = replace(transitions[-1], next_observation=tuple(values))
        broken = (*packets[:-1], pilot._packet(replace(original, transitions=tuple(transitions))))
        with pytest.raises(ValueError):
            if is_probe:
                pilot._invariant_probe_inputs((broken, data[1]), cue_relative_history=True)
            else:
                pilot._invariant_fit_sources(broken, cue_relative_history=True)


@pytest.fixture(scope="module")
def native_work():
    patch = pytest.MonkeyPatch()
    train, predict = TorchFeatureLearner.train_batch, TorchFeatureLearner.predict
    work = dict(updates=0, training_rows=0, predictions=0, prediction_rows=0)

    def update(owner, features, targets):
        work["updates"] += 1
        work["training_rows"] += len(features)
        assert work["updates"] <= 400 and work["training_rows"] <= 12000
        return train(owner, features, targets)

    def score(owner, features):
        work["predictions"] += 1
        work["prediction_rows"] += len(features)
        assert work["predictions"] <= 300 and work["prediction_rows"] <= 3000
        return predict(owner, features)

    patch.setattr(TorchFeatureLearner, "train_batch", update)
    patch.setattr(TorchFeatureLearner, "predict", score)
    yield work
    patch.undo()


@pytest.fixture
def native_case(monkeypatch, native_work, record_property):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    monkeypatch.setattr(pilot, "monotonic", lambda: 1.0)
    monkeypatch.setattr(pilot.shared, "monotonic", lambda: 1.0)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    before = native_work.copy()
    yield torch
    for key, value in native_work.items():
        record_property(f"cue_probe_native_{key}", value - before[key])


@pytest.mark.parametrize("method", METHODS)
def test_should_run_native_isolated_f_and_return_for_each_unchanged_method(
    monkeypatch, native_case, method
):
    torch = native_case
    selected = f"public_cue_relative_{method}"
    source = pilot._history_learner(selected)
    assert type(source) is CueRelativeTorchFeatureLearner
    assert source.policy == route.policy(selected)
    assert source.policy.native_policy == TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
    state = source.snapshot_state()
    assert type(state) is CueRelativeTorchFeatureSnapshot
    owned = pilot._invariant_snapshot(state, 0, selected, catalogue_rows=54)
    assert_tree(torch, owned, state)
    assert all(
        a.data_ptr() != b.data_ptr() for a in tensors(torch, owned) for b in tensors(torch, state)
    )
    pilot.shared._invariant_feature_counter(state, 0, selected)
    pilot.shared._invariant_feature_counter(state, 0, selected, catalogue_rows=54)
    bad_clock = deepcopy(state)
    object.__setattr__(bad_clock.native_state, "training_examples", 1)
    with pytest.raises(ValueError):
        pilot.shared._invariant_feature_counter(bad_clock, 0, selected, catalogue_rows=54)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(state, 32, selected, catalogue_rows=54)
    old = InvariantTorchFeatureLearner(
        pilot._history_learner(f"public_invariant_{method}").policy
    ).snapshot_state()
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(old, 0, selected, catalogue_rows=54)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(state, 0, f"public_invariant_{method}", catalogue_rows=54)
    sources, data = train_sources(), history_f_data()
    old_a = sources[:2]
    inputs = pickle.dumps((sources, data))
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    benchmark = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0)
    original_binding = pilot._budget_binding(benchmark)
    row = pilot._new_row(selected)
    slot = row["probes"][0]
    budgets_seen: list[ToyBudgetSession] = []
    original_probe = pilot.run_fresh_task_probe

    def capture(owner, checkpoint, training, evaluation, **options):
        budgets_seen.extend((options["budget"], options["return_budget"]))
        assert options["budget"] is not options["return_budget"]
        assert all(b.updates_completed == 0 for b in budgets_seen)
        return original_probe(owner, checkpoint, training, evaluation, **options)

    monkeypatch.setattr(pilot, "run_fresh_task_probe", capture)
    guard = pilot._run_probe(
        slot,
        dict(source=source, row=row, budget=benchmark),
        state,
        data,
        sampler=Sampler(),
        deadline=10.0,
        check_boundary=lambda: None,
        training_readouts=old_a,
        return_to_a=True,
        catalogue_rows=54,
    )
    guard()
    assert slot["status"] == slot["report"]["status"] == "complete"
    assert slot["charged_updates"] == slot["return_charged_updates"] == 16
    assert all(b.updates_completed == 16 and b.replay_examples_completed == 0 for b in budgets_seen)
    assert all(b is not benchmark and b.budget.max_training_updates == 16 for b in budgets_seen)
    assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert slot["report"]["completed_training_examples"] == 144
    returned = slot["report"]["return_phase"]
    assert returned["status"] == "complete" and returned["completed_training_examples"] == 144
    assert returned["checkpoints"][0]["scores"] == slot["report"]["checkpoints"][-1]["scores"]
    for context in ("F", "A"):
        outcome = slot["return_fit_outcomes"][context]
        assert outcome["ce_gain"] == pytest.approx(
            outcome["curve"][0]["ce"] - outcome["curve"][-1]["ce"]
        )
    assert_tree(torch, source.snapshot_state(), state)
    assert benchmark.updates_completed == 0 and pilot._budget_binding(benchmark) == original_binding
    assert pickle.dumps((sources, data)) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    previous = slot["report"]
    slot["report"] = deepcopy(previous)
    with pytest.raises(ValueError):
        guard()
    slot["report"] = previous
    guard()


def test_should_run_native_public_joint_route_with_new_bounds_and_no_final_data(
    monkeypatch, native_case
):
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    owners, originals, arms = [], [], []
    build, train = pilot._history_learner, pilot.shared._train_arm

    def learner(method):
        owned = build(method)
        owners.append(owned)
        originals.append(deepcopy(owned.snapshot_state()))
        return owned

    def arm(*args, **kwargs):
        assert kwargs["catalogue_rows"] == 54
        result = train(*args, **kwargs)
        arms.append(result)
        return result

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(
        pilot,
        "_collect_probe_sources",
        lambda boundary, reports, **kwargs: raw_f_data(distinct_marker=True),
    )
    monkeypatch.setattr(pilot, "_history_learner", learner)
    monkeypatch.setattr(pilot.shared, "_train_arm", arm)
    report = pilot.run_pilot(**FLAGS, cue_relative_history=True, train_fit_bounds=True)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == "public_cue_relative_bp" and row["status"] == "complete"
    assert row["charged_updates"] == row["stream_result"]["completed_updates"] == 128
    assert row["native_training_examples"] == row["stream_result"]["training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert report["work"]["native_train_calls"] == 192
    assert report["work"]["training_rows"] == 7488
    assert report["work"]["total_score_forwards_completed"] == 94
    assert report["work"]["total_score_rows_completed"] == 846
    assert report["train_fit_bounds"]["encoding_id"] == CUE_RELATIVE_HISTORY_ENCODING_ID
    assert report["train_fit_bounds"]["work"]["native_updates"] == 0
    assert all(
        slot["charged_updates"] == slot["return_charged_updates"] == 16 for slot in row["probes"]
    )
    assert row["final_evaluation"]["status"] == "not_run"
    assert report["scientific_advantage_verified"] is False and report["comparisons"] is None
    assert len(owners) == 1 and _same_state(owners[0].snapshot_state(), originals[0])
    arms[0]["guard"]()
