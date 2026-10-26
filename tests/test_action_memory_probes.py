"""Real bounded-memory E16 probes, with no live rerun or neural execution."""

import builtins
from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import key_door_learnability as checkpoint_helper
from src.adapters import key_door_memory_pilot as pilot
from src.adapters import key_door_pilot as shared
from src.adapters import key_door_public_history_pilot as history
from src.app import toy_execution_budget as budgets
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


def environment(monkeypatch):
    """Observe the production learner and existing two probe applications."""
    evidence: dict[str, Any] = dict(
        sources=[],
        streams=[],
        wake=[],
        benchmark_trains=[],
        benchmark_scores=[],
        probe_attempts=[],
        probe_trains=[],
        probe_scores=[],
        collections=[],
        cohorts=[],
        probe_budgets=[],
        benchmark_seal=None,
        on_probe_train=None,
        before_probe=None,
        fail_probe_at=None,
        fail_probe_collection_at=None,
        torch_imports=[],
    )
    native_class = pilot.ActionMemoryLearner
    original_train = native_class.train_batch
    original_predict = native_class.predict
    original_stream = pilot.AttributionStream
    original_collect = shared._collect
    original_session = shared._session
    original_checkpoint = pilot.run_checkpoint_probe
    original_probe = pilot.run_fresh_task_probe
    original_import = builtins.__import__
    clock = lambda: 1.0

    def import_without_torch(name: str, *args: Any, **options: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            evidence["torch_imports"].append(name)
            pytest.fail("count-memory probe runtime attempted to import Torch")
        return original_import(name, *args, **options)

    def source(*args, **options):
        result = native_class(*args, **options)
        evidence["sources"].append(result)
        return result

    def train(self, features, targets):
        assert features.shape == (8, 16) and targets.shape == (8, 1)
        reserved = features[0, 7] == 0.75
        before = self.snapshot_state()
        if reserved:
            assert (
                len(evidence["benchmark_trains"]) == 8 and len(evidence["benchmark_scores"]) == 30
            )
            evidence["probe_attempts"].append(before)
            if evidence["fail_probe_at"] == len(evidence["probe_attempts"]):
                raise RuntimeError("third probe storage update failed before commit")
        else:
            assert not evidence["benchmark_scores"] and not evidence["cohorts"]
        result = original_train(self, features, targets)
        record = (before, self.snapshot_state(), features.copy(), targets.copy())
        evidence["probe_trains" if reserved else "benchmark_trains"].append(record)
        if reserved and evidence["on_probe_train"] is not None:
            evidence["on_probe_train"](self)
        return result

    def predict(self, features):
        assert features.shape == (8, 16) and len(evidence["benchmark_trains"]) == 8
        reserved = features[0, 7] == 0.75
        if reserved:
            assert len(evidence["benchmark_scores"]) == 30
        else:
            assert not evidence["cohorts"]
        result = original_predict(self, features)
        evidence["probe_scores" if reserved else "benchmark_scores"].append(
            (self.snapshot_state(), features.copy(), result.copy())
        )
        return result

    def stream(source, training, heldout, **options):
        assert heldout == () and options["training_only"] is True
        result = original_stream(source, training, heldout, **options)
        evidence["streams"].append(result)
        return result

    def collect(identity, layout, cue, role):
        assert role in ("train", "probe_train", "probe_test")
        if role.startswith("probe_"):
            assert (
                len(evidence["benchmark_trains"]) == 8 and len(evidence["benchmark_scores"]) == 30
            )
            previous = sum(batch.role.startswith("probe_") for batch in evidence["collections"])
            if evidence["fail_probe_collection_at"] == previous + 1:
                raise RuntimeError("third reserved collection failed")
        batch = original_collect(identity, layout, cue, role)
        evidence["collections"].append(batch)
        return batch

    def session(*args, **options):
        result = original_session(*args, **options)
        evidence["wake"].append(result)
        return result

    def checkpoint(row, source, state, data, **options):
        assert len(evidence["benchmark_trains"]) == 8 and len(evidence["benchmark_scores"]) == 30
        assert options["decision_diagnostics"] is True
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
        options["timer"] = clock
        return original_probe(*args, **options)

    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(shared, "monotonic", clock)
    monkeypatch.setattr(checkpoint_helper, "monotonic", clock)
    monkeypatch.setattr(shared, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "TorchFeatureLearner", forbidden)
    monkeypatch.setattr(shared, "_live", forbidden)
    monkeypatch.setattr(shared, "build_stream", forbidden)
    monkeypatch.setattr(pilot, "ActionMemoryLearner", source)
    monkeypatch.setattr(native_class, "train_batch", train)
    monkeypatch.setattr(native_class, "predict", predict)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_session", session)
    monkeypatch.setattr(pilot, "run_checkpoint_probe", checkpoint)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    return evidence


def benchmark_signature(evidence):
    stream = evidence["streams"][0]
    return pickle.dumps(
        (
            stream._states,
            stream._runtime.candidate_snapshot().state,
            stream._runtime.applied_updates,
            pilot._budget_binding(stream._budget),
        )
    )


def test_should_fit_reserved_task_on_two_real_owned_copies_without_changing_benchmark_or_expanding_capacity(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    report = pilot.run_probe_pilot()
    assert report["status"] == "complete", report["global_errors"]
    row = report["runs"][0]
    assert row["method"] == "observation_memory" and row["status"] == "complete"
    assert all(
        name not in row
        for name in ("live_episodes", "live_work", "live_sessions_started", "live_outcomes")
    )
    assert len(evidence["sources"]) == len(evidence["streams"]) == len(evidence["wake"]) == 1
    assert len(evidence["benchmark_trains"]) == 8 and len(evidence["benchmark_scores"]) == 30
    assert len(evidence["probe_trains"]) == len(evidence["probe_attempts"]) == 32
    assert len(evidence["probe_scores"]) == 16
    assert [cohort["state"].completed_updates for cohort in evidence["cohorts"]] == [0, 8]
    assert len(evidence["probe_budgets"]) == 2
    assert evidence["probe_budgets"][0] is not evidence["probe_budgets"][1]
    assert all(
        budget.updates_completed == budget.budget.max_training_updates == 16
        for budget in evidence["probe_budgets"]
    )
    assert (
        evidence["wake"][0].updates_completed
        == evidence["wake"][0].budget.max_training_updates
        == 8
    )
    assert benchmark_signature(evidence) == evidence["benchmark_seal"]
    source, candidate = (
        evidence["sources"][0],
        evidence["streams"][0]._runtime.candidate_snapshot().state,
    )
    assert (
        source.snapshot_state().entries == ()
        and source.completed_updates == source.training_examples == 0
    )
    assert (
        candidate.completed_updates == 8
        and candidate.training_examples == 64
        and len(candidate.entries) == 48
    )
    initial_end, trained_end = evidence["probe_trains"][15][1], evidence["probe_trains"][31][1]
    assert (
        initial_end.completed_updates,
        initial_end.training_examples,
        len(initial_end.entries),
    ) == (16, 128, 16)
    assert (
        trained_end.completed_updates,
        trained_end.training_examples,
        len(trained_end.entries),
    ) == (24, 192, 64)
    assert 168 * len(initial_end.entries) == 2688 and 168 * len(trained_end.entries) == 10752
    assert trained_end.policy.max_entries == initial_end.policy.max_entries == 64
    assert trained_end.policy.max_payload_bytes == initial_end.policy.max_payload_bytes == 10752
    old = {entry.features: entry.counts for entry in candidate.entries}
    after = {entry.features: entry.counts for entry in trained_end.entries}
    assert all(after[key] == counts for key, counts in old.items())
    assert row["wake_updates"] == row["stream_result"]["completed_updates"] == 8
    assert (
        row["stream_result"]["training_examples"] == 64
        and len(row["stream_result"]["applied_updates"]) == 8
    )
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["work"]["score_forwards_completed"] == 30
    assert len(report["probe_demonstrations"]) == report["collected_probe_episodes"] == 18
    assert report["partial_probe_collection_work_may_be_unobserved"] is False
    assert report["reserved_probe_array_bytes"] == 19584
    assert [batch.role for batch in evidence["collections"]].count("train") == 8
    assert [batch.role for batch in evidence["collections"]].count("probe_train") == 16
    assert [batch.role for batch in evidence["collections"]].count("probe_test") == 2
    first_data, next_data = (cohort["data"] for cohort in evidence["cohorts"])
    assert all(
        first is second
        for a, b in zip(first_data, next_data, strict=True)
        for first, second in zip(a, b, strict=True)
    )
    train_ids = {identity for batch in first_data[0] for identity in batch.row_ids}
    test_ids = {identity for batch in first_data[1] for identity in batch.row_ids}
    assert len(train_ids) == 128 and len(test_ids) == 16 and train_ids.isdisjoint(test_ids)
    for slot in row["probes"]:
        point = slot["checkpoint_updates"]
        assert slot["status"] == "complete" and slot["charged_updates"] == 16
        assert (
            slot["native_updates_before"] == point
            and slot["native_training_examples_before"] == 8 * point
        )
        assert (
            slot["source_unchanged"]
            is slot["state_unchanged"]
            is slot["budget_binding_preserved"]
            is True
        )
        measured = slot["report"]
        assert measured["attempted_updates"] == measured["completed_updates"] == 16
        assert (
            measured["completed_training_examples"]
            == measured["attempted_training_examples"]
            == 128
        )
        assert (
            measured["score_forward_calls_completed"] == 8
            and measured["reserved_data_bytes"] == 19584
        )
        assert [entry["updates"] for entry in measured["checkpoints"]] == [0, 4, 8, 16]
        assert measured["decision_diagnostics_enabled"] is True
        assert slot["outcomes"]["ce_gain"] > 0 and slot["outcomes"]["final_accuracy"] == 1
    assert report["final_data_constructed"] is False and not evidence["torch_imports"]
    assert report["shared_initial_parameters_exact"] is report["pc_neutral_exact"] is None
    assert (
        report["scientific_advantage_verified"]
        is report["full_R42_complete"]
        is report["full_R43_complete"]
        is False
    )


@pytest.mark.parametrize(
    "fault",
    (
        "prior_outcome",
        "report_replacement",
        "receipt",
        "clock",
        "source",
        "probe_values",
        "probe_replacement",
    ),
)
def test_should_refuse_changed_prior_probe_or_original_evidence_and_keep_separate_paid_charges(
    monkeypatch, fault
):
    evidence = environment(monkeypatch)
    changed: list[bool] = []
    if fault in ("prior_outcome", "report_replacement"):

        def corrupt_prior(cohort):
            if len(evidence["cohorts"]) == 2:
                previous = evidence["cohorts"][0]["row"]["probes"][0]
                if fault == "prior_outcome":
                    previous["outcomes"]["ce_gain"] = -99.0
                else:
                    previous["report"] = deepcopy(previous["report"])
                changed.append(True)

        evidence["before_probe"] = corrupt_prior
    else:

        def corrupt_after_update(owner):
            if changed:
                return
            stream = evidence["streams"][0]
            if fault == "receipt":
                applied = stream._runtime._inbox._applied
                applied[next(iter(applied))] = object()
            elif fault == "clock":
                stream._budget.last_clock += 1.0
            elif fault == "source":
                evidence["sources"][0].restore_state(stream._runtime.candidate_snapshot().state)
            else:
                batch = evidence["cohorts"][0]["data"][0][0]
                if fault == "probe_values":
                    batch.features[0, 0] = 0.123
                else:
                    object.__setattr__(batch, "features", batch.features.copy())
            changed.append(True)

        evidence["on_probe_train"] = corrupt_after_update
    report = pilot.run_probe_pilot()
    row = report["runs"][0]
    assert changed and report["status"] == "incomplete" and row["status"] == "failed"
    assert row["wake_updates"] == evidence["wake"][0].updates_completed == 8
    assert len(row["stream_result"]["applied_updates"]) == 8
    assert len(evidence["benchmark_trains"]) == 8 and len(evidence["benchmark_scores"]) == 30
    if fault in ("prior_outcome", "report_replacement"):
        assert len(evidence["probe_trains"]) == row["probes"][0]["charged_updates"] == 16
        assert row["probes"][1]["charged_updates"] == 0
    else:
        assert len(evidence["probe_trains"]) == row["probes"][0]["charged_updates"] == 1
        assert row["probes"][0]["report"]["completed_updates"] == 1
        assert row["probes"][1]["status"] == "not_run"
    if fault == "receipt":
        assert row["partial_native_evidence"]["charged_updates"] == 8
        assert row["partial_native_evidence"]["readback_error"]
    assert all(budget.updates_completed <= 16 for budget in evidence["probe_budgets"])


def test_should_preserve_two_probe_commits_and_censored_third_attempt_without_retry(monkeypatch):
    evidence = environment(monkeypatch)
    evidence["fail_probe_at"] = 3
    report = pilot.run_probe_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete"
    first, later = row["probes"]
    assert first["status"] == "failed" and first["charged_updates"] == 2
    assert first["report"]["attempted_updates"] == 3 and first["report"]["completed_updates"] == 2
    assert (
        first["report"]["attempted_training_examples"] == 24
        and first["report"]["completed_training_examples"] == 16
    )
    assert [point["updates"] for point in first["report"]["checkpoints"]] == [0]
    assert first["report"]["score_forward_calls_completed"] == 2
    assert later["status"] == "not_run" and later["charged_updates"] == 0
    assert len(evidence["probe_trains"]) == 2 and len(evidence["probe_attempts"]) == 3
    assert evidence["wake"][0].updates_completed == row["wake_updates"] == 8
    assert evidence["streams"][0]._runtime.candidate_snapshot().state.completed_updates == 8


def test_should_keep_unknown_reserved_collection_cost_when_the_third_delegate_fails(monkeypatch):
    evidence = environment(monkeypatch)
    evidence["fail_probe_collection_at"] = 3
    report = pilot.run_probe_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["wake_updates"] == 8
    assert report["collected_probe_episodes"] is None
    assert report["partial_probe_collection_work_may_be_unobserved"] is True
    assert report["reserved_probe_array_bytes"] is None
    assert len([batch for batch in evidence["collections"] if batch.role.startswith("probe_")]) == 2
    assert not evidence["cohorts"] and not evidence["probe_trains"]
    assert all(slot["status"] == "not_run" for slot in row["probes"])
    assert len(row["training_points"]) == 5 and row["work"]["score_forwards_completed"] == 30


@pytest.mark.parametrize("invalid", (None, 0, 1, "true", np.bool_(True)))
@pytest.mark.parametrize("describe", (False, True))
def test_should_refuse_non_bool_probe_control_before_resources(monkeypatch, invalid: Any, describe):
    monkeypatch.setattr(shared, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(with_probes=invalid)


def test_should_describe_fixed_original_assay_costs_without_changing_accepted_live_defaults(
    monkeypatch,
):
    old = deepcopy(pilot.describe_protocol())
    monkeypatch.setattr(shared, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(history, "build_probe_stream", forbidden)
    monkeypatch.setattr(pilot, "ActionMemoryLearner", forbidden)
    protocol = pilot.describe_probe_protocol()
    assert protocol == pilot.describe_protocol(with_probes=True)
    assert protocol["protocol_id"] == "local_key_door_bounded_action_memory_probe_v1"
    assert protocol["policy"] == old["policy"]
    assert protocol["probes"]["benchmark_checkpoints"] == [0, 8]
    assert protocol["probes"]["relative_updates"] == [0, 4, 8, 16]
    assert protocol["probes"]["updates_per_cohort"] == 16
    assert (
        protocol["probes"]["training_packets"] == 16
        and protocol["probes"]["evaluation_packets"] == 2
    )
    assert protocol["probes"]["roles"] == ["probe_train", "probe_test"]
    assert protocol["probes"]["decision_diagnostics"] is True
    assert (
        protocol["reserved_probe_array_bytes"] == 19584
        and protocol["shared_encoded_array_bytes"] == 28288
    )
    assert protocol["probe_table_payload_limit_bytes"] == 10752
    assert protocol["planned_work"] == dict(
        benchmark_storage_updates=8,
        probe_storage_updates=32,
        storage_updates=40,
        training_rows=320,
        score_calls=46,
        score_rows=368,
        teacher_episodes=26,
        teacher_physics_steps=208,
        live_episodes=0,
    )
    assert old == pilot.describe_protocol() == pilot.describe_protocol(with_probes=False)


@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_only_reserved_memory_probe_cli(monkeypatch, capsys, describe):
    calls: list[bool] = []

    def selected():
        calls.append(True)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_action_memory_probe_protocol", selected)
    monkeypatch.setattr(cli, "run_action_memory_probe_pilot", selected)
    monkeypatch.setattr(cli, "run_action_memory_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "key-door-action-memory-probe"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["status"] == "complete"


def test_should_preserve_old_live_cli_dispatch_without_new_probe_flag(monkeypatch, capsys):
    calls: list[bool] = []

    def selected():
        calls.append(True)
        return dict(status="complete")

    monkeypatch.setattr(cli, "run_action_memory_pilot", selected)
    monkeypatch.setattr(cli, "run_action_memory_probe_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["run", "--scenario", "key-door-action-memory"])
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "flags",
    (
        ["--source-artifact", "old.json"],
        ["--environment-seed", "0", "--heldout-environment-seed", "23"],
    ),
)
def test_should_refuse_caller_artifact_or_seed_selection_before_probe_cli(monkeypatch, flags):
    monkeypatch.setattr(cli, "run_action_memory_probe_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["run", "--scenario", "key-door-action-memory-probe", *flags])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
