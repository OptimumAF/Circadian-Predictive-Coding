"""Explicit H32 isolated F16/A16 composition on synthetic public histories.

One public BP case:192 updates7488rows/P94calls846rows. Four small owned-fork
cases add4updates36rows/P8calls72rows. No actual physics, teacher or final data.
These checks establish execution/isolation, not useful task acquisition.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.shared.torch_runtime import require_torch
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, METHODS
from test_minigrid_train_fit_bounds import histories
from src.core.demonstration_episode import episode_arrays
from test_torch_feature_learner import assert_tree, tensors


WIDE_FLAGS = {**FLAGS, "joint_catalogue": True, "wide_neural": True}
EXCLUDED = (
    "activation_health",
    "gradient_geometry",
    "pc_qualification",
    "acquisition_only",
    "train_a_rollout",
    "memory_control",
    "weighted_bp",
    "weighted_pc",
    "weighted_catalogue",
    "train_action_margins",
    "train_fit_bounds",
    "cue_relative_history",
    "bp_update_diagnostics",
)


def forbidden(*args, **kwargs):
    raise AssertionError("admission must precede resource construction")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize(
    "options", [{}, {"public_history": True}, FLAGS, {**FLAGS, "joint_catalogue": True}]
)
def test_should_preserve_existing_descriptors_with_explicit_false(no_resources, options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, wide_neural=False
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_original_doses_with_explicit_fixed_h32(no_resources, method):
    protocol = pilot.describe_protocol(**{**WIDE_FLAGS, "qualification_method": method})
    assert protocol["methods"] == [f"public_invariant_{method}"]
    assert protocol["hidden_dim"] == 32 and protocol["seed"] == 17
    assert protocol["learning_rate"] == 0.03 and protocol["momentum"] == 0.9
    assert protocol["inference_steps"] == 2 and protocol["inference_learning_rate"] == 0.04
    assert protocol["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert protocol["probe"]["source_stages"] == ["initial", "post_joint128"]
    assert protocol["primary_comparisons"] == []
    wide = protocol["wide_neural"]
    assert wide["hidden_dim"] == 32 and wide["parameter_elements"] == 9735
    assert wide["parameter_array_equivalent_bytes"] == 38940
    assert protocol["trainable_parameters"] == {f"public_invariant_{method}": 9735}
    assert protocol["public_history"]["parameters_by_method"] == protocol["trainable_parameters"]
    work = protocol["planned_work"]
    assert work["benchmark_native_train_calls"] == 128
    assert work["benchmark_training_rows"] == 6912
    assert work["probe_native_train_calls"] == work["return_native_train_calls"] == 32
    assert work["probe_training_rows"] == work["return_training_rows"] == 288
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["final_episodes"] == work["learner_live_episodes"] == work["replay_rows"] == 0


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_non_native_wide_flags_before_resources(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**WIDE_FLAGS, "wide_neural": flag})


@pytest.mark.parametrize(
    "changed",
    [
        {"public_history": False},
        {"invariant_history": False},
        {"joint_catalogue": False},
        {"qualification_method": None},
        {"qualification_method": "recurrent_bc"},
        {"qualification_method": np.str_("bp")},
        {"distinct_probe": False},
        {"old_task_readouts": False},
        {"return_to_a": False},
        *({key: True} for key in EXCLUDED),
    ],
)
def test_should_refuse_incomplete_or_extra_wide_routes_before_resources(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**WIDE_FLAGS, **changed})


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_should_forward_cli_flag_only_when_explicit(monkeypatch, capsys, describe, wide):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *CLI_FLAGS, "--joint-catalogue"]
        + (["--wide-neural"] if wide else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**FLAGS, "joint_catalogue": True, **({"wide_neural": True} if wide else {})}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-history-learnability"],
        CLI_FLAGS,
        [*CLI_FLAGS, "--joint-catalogue", "--train-a-rollout"],
        [*CLI_FLAGS, "--joint-catalogue", "--cue-relative-history"],
        [*CLI_FLAGS, "--joint-catalogue", "--train-action-margins"],
        [*CLI_FLAGS, "--joint-catalogue", "--weighted-bp"],
        [*CLI_FLAGS, "--joint-catalogue", "--pc-update-diagnostics"],
    ],
)
def test_should_refuse_wide_cli_in_other_scenarios_or_mixed_routes(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--wide-neural", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("method", METHODS)
def test_should_own_complete_h32_state_and_refuse_wrong_policy_counters(method):
    torch = require_torch()
    bound = f"public_invariant_{method}"
    source = pilot._history_learner(bound, wide_neural=True)
    initial = source.snapshot_state()
    policy = TorchFeaturePolicy(method, 296, 32, 17, num_classes=7)
    assert initial.policy == InvariantTorchFeaturePolicy(policy)
    owned = pilot._invariant_snapshot(initial, 0, bound, catalogue_rows=54, wide_neural=True)
    assert_tree(torch, owned, initial)
    for left, right in zip(tensors(torch, owned), tensors(torch, initial), strict=True):
        assert left.data_ptr() != right.data_ptr()
    child = source.fork()
    features, targets = episode_arrays(histories()[0])
    before = pickle.dumps((features, targets))
    child.train_batch(features, targets)
    trained = child.snapshot_state()
    assert trained.completed_updates == 1 and trained.training_examples == 9
    assert_tree(torch, source.snapshot_state(), initial)
    source.restore_state(trained)
    caller = deepcopy(trained)
    source.restore_state(caller)
    tensors(torch, caller)[0].add_(1)
    assert_tree(torch, source.snapshot_state(), trained)
    np.testing.assert_array_equal(source.predict(features), child.predict(features))
    assert pickle.dumps((features, targets)) == before
    if method == "bp":
        assert trained.native_state.optimizer_state and trained.native_state.gradients is not None
    else:
        assert trained.native_state.optimizer_state is None
    if method in ("neutral_cpc", "gated_cpc"):
        assert "split_generator_state" in trained.native_state.head_state
        assert "energy_history" in trained.native_state.head_state
    malformed = []
    for field, value in (("completed_updates", True), ("training_examples", 1)):
        wrong = deepcopy(initial)
        object.__setattr__(wrong.native_state, field, value)
        malformed.append(wrong)
    malformed.append(
        InvariantTorchFeatureSnapshot(
            InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)),
            deepcopy(initial.native_state),
        )
    )
    malformed.append(initial.native_state)
    rogue = deepcopy(initial)
    object.__setattr__(rogue, "foreign", True)
    malformed.append(rogue)
    for state in malformed:
        with pytest.raises(ValueError):
            pilot._invariant_snapshot(state, 0, bound, catalogue_rows=54, wide_neural=True)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(initial, 0, bound, wide_neural=True)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(initial, 0, bound, catalogue_rows=54)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(trained, 1, bound, catalogue_rows=54, wide_neural=True)
    assert_tree(torch, source.snapshot_state(), trained)


def test_should_execute_public_h32_joint_fit_and_both_isolated_f16_a16_cohorts(
    monkeypatch,
    tmp_path,
):
    torch = require_torch()
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    probe = raw_probe_data()
    inputs_before = pickle.dumps((raw, probe))
    rng = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
    owners, originals, arms, cohorts = [], [], [], []
    build, train, run_probe = (
        pilot._history_learner,
        pilot.shared._train_arm,
        pilot.run_fresh_task_probe,
    )

    def learner(method, **options):
        assert options == {"wide_neural": True}
        owned = build(method, **options)
        owners.append(owned)
        originals.append(deepcopy(owned.snapshot_state()))
        return owned

    def native_arm(*args, **options):
        assert options["catalogue_rows"] == 54 and options["wide_neural"] is True
        arm = train(*args, **options)
        arms.append(arm)
        return arm

    def isolated(source, state, training, evaluation, **options):
        monitors, returned = options["training_readouts"], options["return_training"]
        assert state.policy.native_policy.hidden_dim == 32
        assert len(training) == 16 and len(evaluation) == len(monitors) == len(returned) == 2
        assert all(packet.role == "probe_train" for packet in training)
        assert all(packet.role == "probe_test" for packet in evaluation)
        assert all(packet.role == "train" and packet.context == "A" for packet in monitors)
        assert {packet.episode.episode_id.rsplit("-", 1)[-1] for packet in monitors} == {
            "blue",
            "red",
        }
        before = pickle.dumps((training, evaluation, monitors, returned))
        cohorts.append((state.completed_updates, options["budget"], options["return_budget"]))
        result = run_probe(source, state, training, evaluation, **options)
        assert pickle.dumps((training, evaluation, monitors, returned)) == before
        return result

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(pilot, "_collect_probe_sources", lambda boundary, reports, **options: probe)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", learner)
    monkeypatch.setattr(pilot.shared, "_train_arm", native_arm)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", isolated)
    report = pilot.run_pilot(**WIDE_FLAGS)
    label = os.environ.get("WIDE_CHECK_LABEL")
    destination = (
        Path("artifacts/runs/wide-neural-learnability-20261018") / f"{label}-public-bp.json"
        if label
        else tmp_path / "public-bp.json"
    )
    with destination.open("x", encoding="utf8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == "public_invariant_bp" and row["status"] == "complete"
    assert row["charged_updates"] == row["stream_result"]["completed_updates"] == 128
    assert row["native_training_examples"] == row["stream_result"]["training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [value["updates"] for value in row["readouts"]] == [0, 32, 64, 96, 128]
    assert all(
        value["status"] == "complete" and len(value["scores"]) == 6 for value in row["readouts"]
    )
    assert [point for point, _, _ in cohorts] == [0, 128]
    budgets = [budget for _, fresh, returned in cohorts for budget in (fresh, returned)]
    assert len({id(budget) for budget in budgets}) == 4
    assert all(
        budget.updates_completed == 16 and budget.replay_examples_completed == 0
        for budget in budgets
    )
    for slot in row["probes"]:
        assert slot["status"] == "complete"
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["expected_training_examples"] == 54 * slot["checkpoint_updates"]
        assert slot["source_unchanged"] is slot["state_unchanged"] is True
        assert slot["return_budget_binding_preserved"] is True
        fresh = slot["report"]
        returned = fresh["return_phase"]
        assert (
            fresh["completed_training_examples"] == returned["completed_training_examples"] == 144
        )
        assert returned["checkpoints"][0]["scores"] == fresh["checkpoints"][-1]["scores"]
        for phase in (fresh, returned):
            assert [value["updates"] for value in phase["checkpoints"]] == [0, 4, 8, 16]
            monitored = phase["training_readouts"]
            assert monitored["status"] == "complete" and monitored["completed_rows"] == 72
            assert all(len(value["scores"]) == 2 for value in monitored["checkpoints"])
        assert (
            returned["training_readouts"]["checkpoints"][0]["scores"]
            == fresh["training_readouts"]["checkpoints"][-1]["scores"]
        )
    assert report["work"]["native_train_calls"] == 192
    assert report["work"]["training_rows"] == 7488
    assert report["work"]["total_score_forwards_completed"] == 94
    assert report["work"]["total_score_rows_completed"] == 846
    wide = report["work"]["wide_neural"]
    assert wide["native_calls_reported"] == wide["bp_optimizer_steps_derived"] == 192
    assert wide["bp_backward_calls_derived"] == 192
    assert wide["training_rows_reported"] == 7488
    assert wide["pc_parameter_updates_derived"] == wide["nominal_latent_row_steps"] == 0
    assert wide["failure_work_may_be_unobserved"] is False
    assert report["scientific_advantage_verified"] is False and report["comparisons"] is None
    assert row["final_evaluation"]["status"] == "not_run"
    assert len(owners) == len(arms) == 1
    assert_tree(torch, owners[0].snapshot_state(), originals[0])
    arms[0]["guard"]()
    assert pickle.dumps((raw, probe)) == inputs_before
    assert random.getstate() == rng[0] and pickle.dumps(np.random.get_state()) == rng[1]
    assert torch.equal(torch.get_rng_state(), rng[2])


def test_should_refuse_old_e_probe_in_explicit_h32_route_before_native_copy(
    monkeypatch, no_resources
):
    raw = raw_probe_data(pair="E")
    data = tuple(pilot._history_packets(packets) for packets in raw)
    monitors = pilot._history_packets(
        tuple(
            pilot._packet(episode("A", cue, "train", f"minigrid-source-A-{cue}"))
            for cue in ("blue", "red")
        )
    )
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    with pytest.raises(ValueError, match="original F pair"):
        pilot._run_probe(
            {"checkpoint_updates": 0, "method": "public_invariant_bp"},
            {"row": {"method": "public_invariant_bp"}},
            object(),
            data,
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
            training_readouts=monitors,
            return_to_a=True,
            catalogue_rows=54,
            wide_neural=True,
        )


@pytest.mark.parametrize("unknown", [False, True])
def test_should_leave_primitive_work_unknown_after_an_unobserved_failure(unknown):
    reports = [
        dict(
            environment_steps_attempted=0,
            environment_steps=0,
            failure_work_may_be_unobserved=unknown,
        )
    ]
    work = pilot._work([], reports, old_task_readouts=True, return_to_a=True)
    pilot._add_wide_neural_work(work, "pc")
    assert work["wide_neural"]["native_calls_reported"] == 0
    assert work["wide_neural"]["nominal_latent_row_steps"] == (None if unknown else 0)
    assert work["wide_neural"]["pc_parameter_updates_derived"] == (None if unknown else 0)
    assert work["wide_neural"]["failure_work_may_be_unobserved"] is unknown
