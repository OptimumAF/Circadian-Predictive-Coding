"""Public fixed old-A replay route; bounded synthetic native engineering cases.

Two native cases total320updates8640rows/P126/1134. No physics, teachers, final
evaluation or comparative inference. Existing four-method API evidence is reused.
"""

from copy import deepcopy
import json
import pickle
import random
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app import toy_execution_budget as budgets
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode, train_sources
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS as ORIGINAL_FLAGS
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


FLAGS = {**ORIGINAL_FLAGS, "joint_catalogue": True}


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_describe_extra_costs_and_preserve_original_recipe(monkeypatch, method):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    flags = {**FLAGS, "qualification_method": method}
    original = pilot.describe_protocol(**flags)
    assert original == pilot.describe_protocol(**flags, old_a_rehearsal=False)
    selected = pilot.describe_protocol(**flags, old_a_rehearsal=True)
    assert selected["methods"] == original["methods"]
    assert selected["primary_comparisons"] == original["primary_comparisons"]
    assert selected["planned_work"]["native_train_calls"] == 256
    assert selected["planned_work"]["training_rows"] == 8064
    assert selected["planned_work"]["replay_rows"] == 576
    assert selected["planned_work"]["final_episodes"] == 0


@pytest.mark.parametrize("key", list(FLAGS))
def test_should_refuse_partial_public_option_before_allocating(monkeypatch, key):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**FLAGS, key: False}, old_a_rehearsal=True)


@pytest.fixture(scope="module")
def native_work():
    patch = pytest.MonkeyPatch()
    train, predict = TorchFeatureLearner.train_batch, TorchFeatureLearner.predict
    work = dict(updates=0, training_rows=0, predictions=0, prediction_rows=0)

    def update(owner, features, targets):
        work["updates"] += 1
        work["training_rows"] += len(features)
        assert work["updates"] <= 384 and work["training_rows"] <= 9216
        return train(owner, features, targets)

    def score(owner, features):
        work["predictions"] += 1
        work["prediction_rows"] += len(features)
        assert work["predictions"] <= 160 and work["prediction_rows"] <= 1440
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
        record_property(f"rehearsal_cli_native_{key}", value - before[key])


def test_should_run_native_probe_with_separate_replay_budget_and_protected_evidence(
    monkeypatch, native_case
):
    torch = native_case
    method = "public_invariant_bp"
    source = pilot._history_learner(method)
    state = source.snapshot_state()
    sources = train_sources()
    data = tuple(pilot._history_packets(cohort) for cohort in raw_probe_data())
    inputs = pickle.dumps((sources, data))
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    benchmark = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0)
    binding = pilot._budget_binding(benchmark)
    row = pilot._new_row(method)
    slot = row["probes"][0]
    plans = []
    build = pilot.build_minigrid_old_a_rehearsal

    def capture(*args, **kwargs):
        plan = build(*args, **kwargs)
        plans.append((plan, kwargs["replay_budget"]))
        return plan

    monkeypatch.setattr(pilot, "build_minigrid_old_a_rehearsal", capture)
    guard = pilot._run_probe(
        slot,
        dict(source=source, row=row, budget=benchmark),
        state,
        data,
        sampler=Sampler(),
        deadline=10.0,
        check_boundary=lambda: None,
        training_readouts=sources[:2],
        return_to_a=True,
        catalogue_rows=54,
        old_a_rehearsal=True,
    )
    guard()
    assert slot["status"] == "complete"
    assert slot["charged_updates"] == slot["return_charged_updates"] == 16
    work = slot["rehearsal_work"]
    assert work["charged_replay_examples"] == work["returned_training_rows"] == 288
    assert work["attempted_updates"] == work["returned_updates"] == work["committed_updates"] == 32
    assert work["bank_encoded_bytes"] == 87280
    assert work["validation_projection_rows"] == 36
    assert work["budget_binding_preserved"] is True and work["work_unknown"] is False
    assert slot["projection_accounting"]["replay_rows_returned"] == 288
    assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert _same_state(source.snapshot_state(), state)
    assert benchmark.updates_completed == 0 and pilot._budget_binding(benchmark) == binding
    assert pickle.dumps((sources, data)) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    for key in ("report", "rehearsal_work", "projection_accounting"):
        original = slot[key]
        slot[key] = deepcopy(original)
        with pytest.raises(ValueError):
            guard()
        slot[key] = original
        guard()
    replay = plans[0][1]
    replay.replay_examples_completed = 287
    with pytest.raises(ValueError):
        guard()
    replay.replay_examples_completed = 288
    guard()
    original_limits, original_clock = replay.budget, replay.clock
    replay.budget = deepcopy(original_limits)
    with pytest.raises(ValueError):
        guard()
    replay.budget = original_limits
    replay.clock = lambda: 1.0
    with pytest.raises(ValueError):
        guard()
    replay.clock = original_clock
    replay.last_clock = 0.5
    with pytest.raises(ValueError):
        guard()
    replay.last_clock = 1.0
    guard()


def test_should_run_native_public_cli_with_exact_replay_costs_and_no_final_data(
    monkeypatch, native_case, capsys
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
        result = train(*args, **kwargs)
        arms.append(result)
        return result

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(pilot, "_collect_probe_sources", lambda *args, **kwargs: raw_probe_data())
    monkeypatch.setattr(pilot, "_history_learner", learner)
    monkeypatch.setattr(pilot.shared, "_train_arm", arm)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *CLI_FLAGS, "--joint-catalogue", "--old-a-rehearsal"]
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == "public_invariant_bp" and row["status"] == "complete"
    assert row["charged_updates"] == row["stream_result"]["completed_updates"] == 128
    assert row["native_training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    work = report["work"]
    assert work["native_train_calls"] == 256 and work["training_rows"] == 8064
    assert work["replay_rows"] == 576
    assert work["total_score_forwards_completed"] == 94
    assert work["total_score_rows_completed"] == 846
    replay = work["old_a_rehearsal"]
    assert replay["native_train_calls"] == replay["returned_native_train_calls"] == 64
    assert replay["training_rows"] == replay["replay_rows"] == 576
    assert replay["logical_bank_encoded_bytes"] == 174560
    assert replay["validation_projection_rows"] == 72
    assert replay["validation_projected_array_equivalent_bytes"] == 170496
    assert replay["partial_work_unknown"] is False
    assert row["final_evaluation"]["status"] == "not_run"
    assert report["scientific_advantage_verified"] is False and report["comparisons"] is None
    assert len(owners) == 1 and _same_state(owners[0].snapshot_state(), originals[0])
    arms[0]["guard"]()
