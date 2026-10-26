"""Reserved E probes on isolated copies of matched public-history checkpoints."""

from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import key_door_pilot as shared
from src.adapters import key_door_public_history_pilot as pilot
from src.adapters.key_door_learnability import _bindings
from test_key_door_pilot import Learner as FeatureLearner, fake_environment
from test_torch_feature_rehearsal_pilot import forbidden


def environment(monkeypatch):
    """Real stream/probe applications with fake native heads and owned snapshots."""
    fake_environment(monkeypatch)
    evidence: dict[str, Any] = dict(
        sources=[],
        streams=[],
        collections=[],
        benchmark_trains=[],
        benchmark_scores=[],
        probe_attempts=[],
        probe_trains=[],
        probe_scores=[],
        cohorts=[],
        probe_budgets=[],
        benchmark_seal=None,
        on_probe_train=None,
        before_probe=None,
        fail_probe_at=None,
    )
    original_collect = shared._collect
    original_stream = pilot.AttributionStream
    original_checkpoint = pilot.run_checkpoint_probe
    original_probe = pilot.run_fresh_task_probe

    class Learner(FeatureLearner):
        def __init__(self, policy):
            assert (policy.feature_dim, policy.hidden_dim, policy.seed, policy.num_classes) == (
                16,
                8,
                17,
                5,
            )
            super().__init__(policy)
            evidence["sources"].append(self)

        def train_batch(self, features, targets):
            assert features.shape == (8, 16) and targets.shape == (8, 1)
            reserved = features[0, 7] == 0.75
            if reserved:
                assert (
                    len(evidence["benchmark_trains"]) == 32
                    and len(evidence["benchmark_scores"]) == 120
                )
                evidence["probe_attempts"].append(
                    (self.policy.method, self.state.completed_updates)
                )
                if evidence["fail_probe_at"] == len(evidence["probe_attempts"]):
                    raise RuntimeError("declared third probe update failed")
            else:
                assert not evidence["cohorts"], "benchmark training resumed after probing"
            before = self.state.completed_updates
            result = super().train_batch(features, targets)
            slot = (self.policy.method, before, features.copy(), targets.copy())
            evidence["probe_trains" if reserved else "benchmark_trains"].append(slot)
            if reserved and evidence["on_probe_train"] is not None:
                evidence["on_probe_train"](self)
            return result

        def predict(self, features):
            assert features.shape == (8, 16)
            assert len(evidence["benchmark_trains"]) == 32
            reserved = features[0, 7] == 0.75
            if reserved:
                assert len(evidence["benchmark_scores"]) == 120
            else:
                assert not evidence["cohorts"], "benchmark readout resumed after probing"
            result = super().predict(features)
            evidence["probe_scores" if reserved else "benchmark_scores"].append(
                (self.policy.method, self.state.completed_updates, features.copy())
            )
            return result

    def collect(identity, layout, cue, role):
        assert role in ("train", "probe_train", "probe_test"), (
            "final data entered a diagnostic route"
        )
        if role.startswith("probe_"):
            assert (
                len(evidence["benchmark_trains"]) == 32 and len(evidence["benchmark_scores"]) == 120
            )
        batch = original_collect(identity, layout, cue, role)
        evidence["collections"].append(batch)
        return batch

    def stream(source, training, heldout, **options):
        assert heldout == () and options["training_only"] is True
        instance = original_stream(source, training, heldout, **options)
        evidence["streams"].append(instance)
        return instance

    def checkpoint(row, source, state, data, **options):
        assert len(evidence["benchmark_trains"]) == 32 and len(evidence["benchmark_scores"]) == 120
        cohort = dict(row=row, source=source, state=state, data=data, options=options)
        evidence["cohorts"].append(cohort)
        if evidence["benchmark_seal"] is None:
            evidence["benchmark_seal"] = benchmark_signature(evidence)
        if evidence["before_probe"] is not None:
            evidence["before_probe"](cohort)
        return original_checkpoint(row, source, state, data, **options)

    def probe(*args, **options):
        budget = options["budget"]
        assert budget.updates_completed == 0 and budget.budget.max_training_updates == 16
        evidence["probe_budgets"].append(budget)
        return original_probe(*args, **options)

    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)
    monkeypatch.setattr(
        pilot, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(pilot, "run_checkpoint_probe", checkpoint)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_live", forbidden)
    monkeypatch.setattr(shared, "build_stream", forbidden)
    return evidence


def benchmark_signature(evidence):
    return pickle.dumps(
        [
            (
                stream._states,
                stream._runtime.candidate_snapshot().state,
                stream._runtime.applied_updates,
                _bindings(stream._budget, None),
            )
            for stream in evidence["streams"]
        ]
    )


def test_should_encode_all_reserved_packets_once_without_changing_roles_labels_ids_or_raw_arrays(
    monkeypatch,
):
    raw = shared.build_probe_stream()
    calls = []

    def collect_reserved():
        calls.append(True)
        return raw

    monkeypatch.setattr(shared, "build_probe_stream", collect_reserved)
    encoded = pilot.build_probe_stream()
    assert calls == [True] and [len(part) for part in encoded] == [16, 2]
    all_ids = []
    for original_part, transformed_part, role in zip(
        raw, encoded, ("probe_train", "probe_test"), strict=True
    ):
        for original, transformed in zip(original_part, transformed_part, strict=True):
            assert transformed.role == original.role == role
            assert (
                transformed.context == original.context
                and transformed.num_classes == original.num_classes == 5
            )
            assert transformed.row_ids == original.row_ids
            assert transformed.episode is not None and original.episode is not None
            assert transformed.episode.episode_id == original.episode.episode_id
            assert transformed.episode.role == original.episode.role == role
            assert transformed.features.shape == (8, 16)
            np.testing.assert_array_equal(
                transformed.features[:, :8], np.tile(original.features[0], (8, 1))
            )
            np.testing.assert_array_equal(transformed.features[:, 8:], original.features)
            np.testing.assert_array_equal(transformed.targets, original.targets)
            assert not np.shares_memory(transformed.features, original.features)
            assert not np.shares_memory(transformed.targets, original.targets)
            all_ids.extend(transformed.row_ids)
    assert len(all_ids) == len(set(all_ids)) == 144
    arrays = [
        array for part in encoded for batch in part for array in (batch.features, batch.targets)
    ]
    assert all(
        not np.shares_memory(left, right)
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    )


def test_should_preserve_benchmark_and_run_two_paid_private_probes_on_each_absolute_checkpoint(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    report = pilot.run_probe_pilot()
    assert report["status"] == "complete", report["global_errors"]
    assert len(evidence["sources"]) == len(evidence["streams"]) == 4
    assert len(evidence["benchmark_trains"]) == 32 and len(evidence["benchmark_scores"]) == 120
    assert len(evidence["probe_trains"]) == 128 and len(evidence["probe_scores"]) == 64
    assert len(evidence["cohorts"]) == len(evidence["probe_budgets"]) == 8
    assert len({id(budget) for budget in evidence["probe_budgets"]}) == 8
    assert all(
        budget.updates_completed == budget.budget.max_training_updates == 16
        for budget in evidence["probe_budgets"]
    )
    assert benchmark_signature(evidence) == evidence["benchmark_seal"]
    assert len(evidence["collections"]) == 26
    assert len(report["probe_demonstrations"]) == report["collected_probe_episodes"] == 18
    assert sum(batch.role == "train" for batch in evidence["collections"]) == 8
    assert sum(batch.role == "probe_train" for batch in evidence["collections"]) == 16
    assert sum(batch.role == "probe_test" for batch in evidence["collections"]) == 2
    assert report["final_data_constructed"] is False
    assert (
        report["shared_initial_parameters_exact"]
        is report["pc_neutral_exact"]
        is report["pc_neutral_probe_exact"]
        is True
    )
    original_data = evidence["cohorts"][0]["data"]
    for cohort in evidence["cohorts"][1:]:
        assert all(
            old is current
            for old_part, current_part in zip(original_data, cohort["data"], strict=True)
            for old, current in zip(old_part, current_part, strict=True)
        )
    for row, source, stream in zip(
        report["runs"], evidence["sources"], evidence["streams"], strict=True
    ):
        assert (
            row["status"] == "complete"
            and row["source_unchanged"] is row["actor_unchanged"] is True
        )
        assert row["wake_updates"] == stream._budget.updates_completed == 8
        assert source.state.completed_updates == source.state.training_examples == 0
        assert stream._runtime.candidate_snapshot().state.completed_updates == 8
        assert (
            row["stream_result"]["completed_updates"] == 8
            and row["stream_result"]["training_examples"] == 64
        )
        assert len(row["stream_result"]["applied_updates"]) == 8
        assert row["stream_result"]["score_forward_calls"] == 0
        assert row["work"]["score_forwards_completed"] == 30
        assert [
            before for method, before, _, _ in evidence["probe_trains"] if method == row["method"]
        ] == list(range(16)) + list(range(8, 24))
        assert [slot["checkpoint_updates"] for slot in row["probes"]] == [0, 8]
        for slot in row["probes"]:
            origin = slot["checkpoint_updates"]
            assert slot["status"] == "complete" and slot["charged_updates"] == 16
            assert slot["native_updates_before"] == origin
            assert slot["native_training_examples_before"] == origin * 8
            assert (
                slot["source_unchanged"]
                is slot["state_unchanged"]
                is slot["budget_binding_preserved"]
                is True
            )
            measured = slot["report"]
            assert measured["attempted_updates"] == measured["completed_updates"] == 16
            assert (
                measured["attempted_training_examples"]
                == measured["completed_training_examples"]
                == 128
            )
            assert measured["score_forward_calls_completed"] == 8
            assert measured["reserved_data_bytes"] == 18 * 8 * 17 * 8
            assert [point["updates"] for point in measured["checkpoints"]] == [0, 4, 8, 16]
            assert measured["decision_diagnostics_enabled"] is True
            assert measured["episode_scope"]["training_transitions"] == 128
            assert measured["episode_scope"]["evaluation_transitions"] == 16
    for left, right in zip(report["runs"][1]["probes"], report["runs"][2]["probes"], strict=True):
        assert left["outcomes"] == right["outcomes"]
        assert left["report"]["checkpoints"] == right["report"]["checkpoints"]
    assert report["scientific_advantage_verified"] is report["full_R43_complete"] is False


@pytest.mark.parametrize(
    "fault", ("prior_probe", "benchmark_receipt", "wake_budget", "probe_values", "probe_array")
)
def test_should_stop_corrupted_prior_evidence_or_probe_input_without_resetting_paid_work(
    monkeypatch, fault
):
    evidence = environment(monkeypatch)
    changed = []
    if fault == "prior_probe":

        def corrupt_prior(cohort):
            if len(evidence["cohorts"]) == 2:
                prior = evidence["cohorts"][0]["row"]["probes"][0]
                prior["report"]["completed_updates"] = 0
                changed.append(True)

        evidence["before_probe"] = corrupt_prior
    else:

        def corrupt_after_paid_update(owner):
            if changed:
                return
            stream = evidence["streams"][0]
            if fault == "benchmark_receipt":
                stream._runtime._inbox._applied.clear()
            elif fault == "wake_budget":
                object.__setattr__(stream._budget.budget, "max_training_updates", 9)
            else:
                batch = evidence["cohorts"][0]["data"][0][0]
                if fault == "probe_values":
                    batch.features[0, 0] = 0.123
                else:
                    object.__setattr__(batch, "features", batch.features.copy())
            changed.append(True)

        evidence["on_probe_train"] = corrupt_after_paid_update
    report = pilot.run_probe_pilot()
    assert changed and report["status"] == "incomplete"
    assert len(evidence["benchmark_trains"]) == 32 and len(evidence["benchmark_scores"]) == 120
    assert all(row["wake_updates"] == 8 for row in report["runs"])
    if fault == "prior_probe":
        assert len(evidence["probe_trains"]) == 16
        assert report["runs"][0]["probes"][0]["charged_updates"] == 16
        assert report["runs"][0]["probes"][0]["report"]["completed_updates"] == 0
    else:
        assert len(evidence["probe_trains"]) == 1
        assert report["runs"][0]["probes"][0]["charged_updates"] == 1
        assert report["runs"][0]["probes"][0]["report"]["completed_updates"] == 1
    assert all(slot["status"] == "not_run" for row in report["runs"][1:] for slot in row["probes"])
    assert all(budget.updates_completed <= 16 for budget in evidence["probe_budgets"])


def test_should_keep_partial_probe_updates_and_attempts_when_third_update_fails(monkeypatch):
    evidence = environment(monkeypatch)
    evidence["fail_probe_at"] = 3
    report = pilot.run_probe_pilot()
    assert report["status"] == "incomplete"
    slot = report["runs"][0]["probes"][0]
    assert slot["status"] == "failed" and slot["charged_updates"] == 2
    assert slot["report"]["attempted_updates"] == 3 and slot["report"]["completed_updates"] == 2
    assert slot["report"]["attempted_training_examples"] == 24
    assert slot["report"]["completed_training_examples"] == 16
    assert slot["report"]["score_forward_calls_completed"] == 2
    assert [point["updates"] for point in slot["report"]["checkpoints"]] == [0]
    assert len(evidence["probe_trains"]) == 2 and len(evidence["probe_attempts"]) == 3
    assert all(row["wake_updates"] == 8 for row in report["runs"])
    assert all(
        slot["status"] == "not_run"
        for row in report["runs"]
        for slot in row["probes"]
        if slot["report"] is None
    )
    assert report["runs"][0]["probes"][1]["status"] == "not_run"


def test_should_leave_ordinary_default_without_probe_data_or_calls(monkeypatch):
    evidence = environment(monkeypatch)
    monkeypatch.setattr(shared, "build_probe_stream", forbidden)
    monkeypatch.setattr(pilot, "run_checkpoint_probe", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "complete"
    assert len(evidence["benchmark_trains"]) == 32 and len(evidence["benchmark_scores"]) == 120
    assert not evidence["probe_trains"] and not evidence["probe_scores"]
    assert all("probes" not in row for row in report["runs"])
    assert "probe_demonstrations" not in report
    assert pilot.describe_protocol() == pilot.describe_protocol(with_probes=False)


@pytest.mark.parametrize("invalid", (None, 0, 1, "true", np.bool_(True)))
@pytest.mark.parametrize("describe", (False, True))
def test_should_refuse_non_bool_probe_mode_before_resources(monkeypatch, invalid, describe):
    monkeypatch.setattr(shared, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(with_probes=invalid)


def test_should_describe_fixed_probe_costs_and_wrappers_without_data_or_model_work(monkeypatch):
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "build_probe_stream", forbidden)
    monkeypatch.setattr(pilot, "build_training_stream", forbidden)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", forbidden)
    original = deepcopy(pilot.describe_protocol())
    description = pilot.describe_probe_protocol()
    assert description == pilot.describe_protocol(with_probes=True)
    assert description["probes"]["benchmark_checkpoints"] == [0, 8]
    assert description["probes"]["relative_updates"] == [0, 4, 8, 16]
    assert description["probes"]["per_probe_updates"] == 16
    assert (
        description["probes"]["training_packets"] == 16
        and description["probes"]["evaluation_packets"] == 2
    )
    assert description["probes"]["decision_diagnostics"] is True
    assert description["planned_work"] == dict(
        benchmark_updates=32,
        probe_updates=128,
        native_updates=160,
        training_rows=1280,
        score_calls=184,
        score_rows=1472,
        teacher_episodes=26,
        physics_steps=208,
    )
    assert original == pilot.describe_protocol()


@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_only_public_history_probe_cli(monkeypatch, capsys, describe):
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_public_history_probe_protocol", selected)
    monkeypatch.setattr(cli, "run_public_history_probe_pilot", selected)
    monkeypatch.setattr(cli, "run_public_history_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-public-history-probe"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["status"] == "complete"
