"""Fixed sixteen-update selection; all probe execution below is explicitly stubbed."""

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import src.adapters.learnability_pilot as pilot
from src.shared.process_memory import ProcessRssSegment


# Retained v1 protocol DATA, copied from r21-learnability-20261016/protocol.json.
# Keeping the independent whole dict catches accidental default/schema changes.
RETAINED_V1 = json.loads(r"""
{
  "protocol_id": "reserved_fresh_task_learnability_smoke_v1",
  "classification": "exploratory_fixed_data_engineering_smoke",
  "seeds": [
    23
  ],
  "methods": [
    "bp",
    "pc",
    "neutral_cpc",
    "gated_cpc"
  ],
  "benchmark_stream": "recurring_A_B_A",
  "benchmark_updates_per_phase": 6,
  "benchmark_checkpoint_updates": [
    0,
    6,
    12,
    18
  ],
  "benchmark_training_rng": 7301,
  "benchmark_final_rng": 9101,
  "hidden_dims": [
    8,
    8
  ],
  "learning_rate": 0.03,
  "precisions": [
    1.0,
    1.0
  ],
  "active_cpc_config": {
    "chemical_decay": 0.995,
    "chemical_buildup_rate": 0.02,
    "plasticity_sensitivity": 0.7,
    "min_plasticity": 0.2
  },
  "neutral_cpc_config": {
    "chemical_decay": 0.995,
    "chemical_buildup_rate": 0.02,
    "plasticity_sensitivity": 0.7,
    "min_plasticity": 1.0
  },
  "benchmark_batch_rows": 8,
  "benchmark_final_rows_per_context": 64,
  "benchmark_target_rules": {
    "A": "x0 >= 0",
    "B": "x1 >= 0"
  },
  "label_delay": 2,
  "inference_steps": 4,
  "inference_learning_rate": 0.2,
  "probe_target": "x0*x1 >= 0; fixed nonlinear composition of A/B inputs",
  "probe_training_rng": 17031,
  "probe_evaluation_rng": 17033,
  "diagnostic_features_rng": 17035,
  "probe_training_batches": 4,
  "probe_training_batch_rows": 8,
  "probe_evaluation_rows": 32,
  "probe_checkpoint_updates": [
    0,
    1,
    2,
    4
  ],
  "diagnostic_feature_rows": 32,
  "feature_health_definition": "mean absolute tanh activation; dead <=1e-3; active complement; saturated entry abs>=.99",
  "effective_rank_definition": "centered SVD singular VALUES entropy; s>eps*max(shape)*max(1,max(s)); numerical zero rank0",
  "effective_rank_limit": "absolute numerical cutoff is not invariant at tiny scales; diagnostic, not causal proof",
  "unavailable_diagnostics": {
    "importance": "Coupled models do not store native per-neuron importance.",
    "neuron_age": "Coupled models do not store native per-neuron ages."
  },
  "planned_work": {
    "benchmark_updates": 72,
    "probe_updates": 64,
    "total_updates": 136,
    "benchmark_training_row_exposures": 576,
    "probe_training_row_exposures": 512,
    "probe_clones": 16,
    "probe_score_forward_calls": 64,
    "benchmark_score_forward_calls": 32,
    "diagnostic_feedforward_passes": 16,
    "diagnostic_svd_calls": 32,
    "replay_examples": 0
  },
  "score_definitions": {
    "bce": "probability BCE clip1e-12 lower better",
    "accuracy": "threshold >=.5 higher better"
  },
  "comparison_directions": {
    "fresh_gain": "probe BCE at0 minus BCE at4; retain initial level and entire curve",
    "retention_loss": "A BCE afterB minus A BCE afterA; lower better",
    "return_gain": "A BCE afterB minus A BCE afterReturn; higher better",
    "new_B_gain": "B BCE initial minus B BCE afterB; higher better"
  },
  "isolation": "independent probe template; copied checkpoint and reserved probe data; probes finish before benchmark final scoring",
  "work_limit": "BP and PC have unequal native work; nominal counts and elapsed time are not matched compute",
  "reserved_data_array_bytes": 1536,
  "diagnostic_reference_array_bytes": 512,
  "stop_policy": "One invocation; preserve failed or unreached rows; no retry or tuning.",
  "resources": {
    "wall_seconds": 60.0,
    "training_updates": 136,
    "environment_steps": 0,
    "cpu_threads": 1,
    "process_memory_bytes": 268435456,
    "replay_bytes": 0,
    "model_download_bytes": 0,
    "storage_bytes": 33554432,
    "gpu_memory_bytes": 0
  },
  "limitations": "One prespecified engineering seed, reused fixed ABA data, no intervals or confirmation; C is not independent task generalization. Native importance/age and full R2.1 remain open. No tuning, promotion, replay, sleep or CPC advantage claim."
}
""")


EXTENDED_POINTS = (0, 1, 2, 4, 8, 16)


def forbidden(*args, **kwargs):
    pytest.fail("an unselected or native operational path was reached")


def test_should_preserve_the_entire_retained_four_update_description_without_operational_io(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "build_reserved_data", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert pilot.describe_protocol() == RETAINED_V1
    assert pilot.describe_protocol(extended=False) == RETAINED_V1
    assert pilot.PROTOCOL_ID == RETAINED_V1["protocol_id"]
    assert pilot.PROBE_CHECKPOINTS == (0, 1, 2, 4)


def test_should_describe_only_the_predeclared_extended_exposure_and_preserve_every_other_control():
    report = pilot.describe_protocol(extended=True)
    assert report["protocol_id"] == "reserved_fresh_task_learnability_sixteen_v1"
    assert report["predecessor_protocol_id"] == RETAINED_V1["protocol_id"]
    assert report["probe_training_batches"] == 16
    assert report["probe_checkpoint_updates"] == list(EXTENDED_POINTS)
    assert report["reserved_data_array_bytes"] == 3840
    assert report["extended_probe_limits"] == {
        "training_batches": 16,
        "checkpoint_updates": list(EXTENDED_POINTS),
        "no_endpoint_selection": True,
    }
    assert report["planned_work"] == {
        **RETAINED_V1["planned_work"],
        "probe_updates": 256,
        "total_updates": 328,
        "probe_training_row_exposures": 2048,
        "probe_score_forward_calls": 96,
    }
    assert report["resources"] == {**RETAINED_V1["resources"], "training_updates": 328}
    assert all(
        phrase in report["motivation"].lower()
        for phrase in (
            "negative",
            "fifteen-of-sixteen",
            "predeclared",
            "no favorable endpoint selection",
        )
    )
    changed = {
        "protocol_id",
        "probe_training_batches",
        "probe_checkpoint_updates",
        "reserved_data_array_bytes",
        "planned_work",
        "resources",
        "comparison_directions",
        "predecessor_protocol_id",
        "motivation",
        "extended_probe_limits",
    }
    assert {key: value for key, value in report.items() if key not in changed} == {
        key: value for key, value in RETAINED_V1.items() if key not in changed
    }
    assert report["comparison_directions"] == {
        **RETAINED_V1["comparison_directions"],
        "fresh_gain": "probe BCE at0 minus BCE at16; retain initial level and entire curve",
    }
    selection = pilot._scenario(True)
    assert type(selection) is tuple and selection == (
        "reserved_fresh_task_learnability_sixteen_v1",
        EXTENDED_POINTS,
        pilot.EXTENDED_RESOURCES,
    )
    assert pilot.EXTENDED_PROBE_CHECKPOINTS == EXTENDED_POINTS
    assert pilot.describe_protocol() == RETAINED_V1


def test_should_keep_four_batches_as_the_exact_prefix_and_all_reserved_storage_separate():
    before = pickle.dumps(np.random.get_state())
    old_training, old_evaluation, old_reference = pilot.build_reserved_data()
    training, evaluation, reference = pilot.build_reserved_data(extended=True)
    benchmark, final = pilot.build_stream("recurring")
    assert pickle.dumps(np.random.get_state()) == before
    assert len(training) == 16 and len(evaluation) == 1
    for index, batch in enumerate(training):
        assert batch.role == "probe_train" and batch.context == "C"
        assert batch.row_ids == tuple(f"probe-train-{index}-{row}" for row in range(8))
        assert batch.features.shape == (8, 2) and batch.targets.shape == (8, 1)
        assert batch.features.dtype == batch.targets.dtype == np.float64
        np.testing.assert_array_equal(
            batch.targets, (batch.features[:, [0]] * batch.features[:, [1]] >= 0)
        )
        if index < 4:
            assert batch.row_ids == old_training[index].row_ids
            np.testing.assert_array_equal(batch.features, old_training[index].features)
            np.testing.assert_array_equal(batch.targets, old_training[index].targets)
            assert not np.shares_memory(batch.features, old_training[index].features)
            assert not np.shares_memory(batch.targets, old_training[index].targets)
    assert evaluation[0].role == "probe_test" and evaluation[0].row_ids == old_evaluation[0].row_ids
    np.testing.assert_array_equal(evaluation[0].features, old_evaluation[0].features)
    np.testing.assert_array_equal(evaluation[0].targets, old_evaluation[0].targets)
    np.testing.assert_array_equal(reference, old_reference)
    assert reference.shape == (32, 2)
    batches = (*training, *evaluation, *benchmark, *final)
    ids = [row for batch in batches for row in batch.row_ids]
    assert len(ids) == len(set(ids))
    arrays = [array for batch in batches for array in (batch.features, batch.targets)] + [reference]
    assert all(
        not np.shares_memory(left, right)
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    )
    assert (
        sum(
            array.nbytes
            for batch in (*training, *evaluation)
            for array in (batch.features, batch.targets)
        )
        == 3840
    )
    assert reference.nbytes == 512


def test_should_generate_independent_deterministic_extended_data_on_each_call():
    left, right = pilot.build_reserved_data(extended=True), pilot.build_reserved_data(extended=True)
    for first, second in zip((*left[0], *left[1]), (*right[0], *right[1]), strict=True):
        assert first.row_ids == second.row_ids and first.role == second.role
        for original, fresh in ((first.features, second.features), (first.targets, second.targets)):
            assert not np.shares_memory(original, fresh)
            np.testing.assert_array_equal(original, fresh)
    assert not np.shares_memory(left[2], right[2])
    np.testing.assert_array_equal(left[2], right[2])


@pytest.mark.parametrize("entrypoint", ("describe_protocol", "build_reserved_data", "run_pilot"))
@pytest.mark.parametrize("bad", (None, 0, "true", np.bool_(True)))
def test_should_refuse_nonboolean_scenario_controls_before_allocation_or_native_paths(
    monkeypatch, entrypoint, bad
):
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, entrypoint)(extended=bad)


def probe_report(points=EXTENDED_POINTS, *, failed=False):
    selected = points[:-1] if failed else points
    bce = {0: 1.0, 1: 0.9, 2: 0.8, 4: 0.7, 8: 1.3, 16: 1.5}
    completed = selected[-1]
    return {
        "status": "failed" if failed else "complete",
        "reason": "stub failure after eight" if failed else None,
        "attempted_updates": completed,
        "completed_updates": completed,
        "attempted_training_examples": completed * 8,
        "completed_training_examples": completed * 8,
        "score_forward_calls_attempted": len(selected),
        "score_forward_calls_completed": len(selected),
        "checkpoints": [
            {"updates": point, "scores": {"C": {"bce": bce[point], "accuracy": 0.5}}}
            for point in selected
        ],
        "diagnostics": [],
        "elapsed_seconds": 0.0,
        "timing_status": "partial" if failed else "complete",
        "reserved_data_bytes": 3840,
    }


@pytest.mark.parametrize("extended", (False, True))
def test_should_supply_an_independent_fresh_budget_matching_the_fixed_probe_endpoint(
    monkeypatch, extended
):
    points = EXTENDED_POINTS if extended else (0, 1, 2, 4)
    training, evaluation, _ = pilot.build_reserved_data(extended=extended)
    template, state = object(), object()
    row: dict[str, Any] = {}
    calls: list[Any] = []
    returned = probe_report(points)

    def probe(
        source, checkpoint, observed_training, observed_evaluation, *, budget, checkpoint_updates
    ):
        assert source is template and checkpoint is state
        assert observed_training is training and observed_evaluation is evaluation
        assert budget.updates_completed == 0 and budget.budget.max_training_updates == points[-1]
        assert budget.budget.max_wall_seconds == 50.0
        assert checkpoint_updates == points
        calls.append(budget)
        return returned

    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    if extended:
        result = pilot._measure_probe(
            template, state, training, evaluation, row, 50.0, None, probe_checkpoints=points
        )
    else:
        result = pilot._measure_probe(template, state, training, evaluation, row, 50.0, None)
    assert result is row["probe"] is returned and row["probe_seconds"] == 0.0
    assert len(calls) == 1


class Template:
    def __init__(self):
        self.marker = 0

    def snapshot_state(self):
        return {"marker": self.marker}


@pytest.mark.parametrize(
    "case", ("complete", "failed", "checkpoint_mutation", "template_mutation", "heldout")
)
def test_should_preserve_full_extended_curves_and_refuse_failure_or_cross_role_state_changes(
    monkeypatch, case
):
    training, evaluation, reference = pilot.build_reserved_data(extended=True)
    template, state = Template(), {"marker": 0}
    rows = pilot._empty_observations()
    returned = probe_report(failed=case == "failed")

    def features(checkpoint, features, row):
        assert checkpoint is state and features is reference
        row.update(feature_health=[], diagnostic_seconds=0.0)

    def probe(
        source, checkpoint, observed_training, observed_evaluation, *, budget, checkpoint_updates
    ):
        assert source is template and checkpoint is state
        assert observed_training is training and observed_evaluation is evaluation
        assert checkpoint_updates == EXTENDED_POINTS and budget.budget.max_training_updates == 16
        if case == "checkpoint_mutation":
            checkpoint["marker"] = 1
        elif case == "template_mutation":
            template.marker = 1
        return returned

    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pilot, "_measure_features", forbidden if case == "heldout" else features)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden if case == "heldout" else probe)
    observe = pilot._observer(
        template, training, evaluation, reference, rows, 50.0, probe_checkpoints=EXTENDED_POINTS
    )
    if case == "complete":
        observe(0, state, None)
        assert rows[0]["status"] == "complete"
        assert rows[0]["fresh_initial_bce"] == 1.0 and rows[0]["fresh_gain"] == -0.5
        assert rows[0]["probe"] is returned
        assert [point["updates"] for point in returned["checkpoints"]] == list(EXTENDED_POINTS)
    elif case == "heldout":
        with pytest.raises(ValueError, match="training"):
            observe(0, state, SimpleNamespace(role="final_test"))
        assert rows[0]["status"] == "not_run"
    else:
        with pytest.raises(RuntimeError):
            observe(0, state, None)
        assert rows[0]["status"] == "failed" and rows[0]["probe"] is returned
        assert "fresh_gain" not in rows[0]
    assert all(row["status"] == "not_run" for row in rows[1:])


@dataclass
class Parameters:
    value: np.ndarray


class StubModel:
    def __init__(self, *args):
        pass

    def snapshot_parameters(self):
        return Parameters(np.array([0.0]))


class StubSampler:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def snapshot(self):
        return ProcessRssSegment(1, 1024, 1024, 1, 0.01)


class StubBudget:
    def __init__(self, budget, clock):
        self.budget, self.clock, self.updates_completed = budget, clock, 0

    def attach_memory(self, sampler):
        assert type(sampler) is StubSampler

    def before_final(self):
        pass


@pytest.mark.parametrize("failed", (False, True))
def test_should_compose_all_four_extended_runs_with_only_stubs_and_preserve_failed_rows(
    monkeypatch, failed
):
    calls: dict[str, Any] = {"probe": 0, "stream": 0, "models": []}
    benchmark_training, benchmark_final = object(), object()

    def learner(method, initial):
        result = Template()
        calls["models"].append(result)
        return result

    def probe(template, state, training, evaluation, *, budget, checkpoint_updates):
        calls["probe"] += 1
        assert checkpoint_updates == EXTENDED_POINTS and budget.budget.max_training_updates == 16
        assert len(training) == 16 and all(batch.role == "probe_train" for batch in training)
        assert len(evaluation) == 1 and evaluation[0].role == "probe_test"
        assert training is not benchmark_training and evaluation is not benchmark_final
        return probe_report(failed=failed)

    def stream(
        source, training, evaluation, *, checkpoint_updates, label_delay, budget, observe_checkpoint
    ):
        calls["stream"] += 1
        assert training is benchmark_training and evaluation is benchmark_final
        assert checkpoint_updates == (0, 6, 12, 18) and label_delay == 2
        for point in checkpoint_updates:
            budget.updates_completed = point
            observe_checkpoint(point, {"marker": point}, None)
        return {
            "checkpoints": [
                {"updates": point, "scores": {"A": {"bce": 1.0}, "B": {"bce": 1.0}}}
                for point in checkpoint_updates
            ]
        }

    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", StubModel)
    monkeypatch.setattr(pilot, "ProcessRssSampler", StubSampler)
    monkeypatch.setattr(pilot, "ToyBudgetSession", StubBudget)
    monkeypatch.setattr(pilot, "build_stream", lambda kind: (benchmark_training, benchmark_final))
    monkeypatch.setattr(pilot, "build_learner", learner)
    monkeypatch.setattr(pilot, "run_attribution_stream", stream)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    monkeypatch.setattr(
        pilot,
        "_measure_features",
        lambda state, reference, row: row.update(feature_health=[], diagnostic_seconds=0.0),
    )
    result = pilot.run_pilot(extended=True)
    assert (
        len(result["runs"]) == 4 and sum(len(row["observations"]) for row in result["runs"]) == 16
    )
    assert len(calls["models"]) == 8 and len({id(model) for model in calls["models"]}) == 8
    assert calls["stream"] == 4
    assert result["protocol"]["protocol_id"] == "reserved_fresh_task_learnability_sixteen_v1"
    assert result["scientific_advantage_verified"] is result["full_R21_complete"] is False
    if failed:
        assert result["status"] == "incomplete" and calls["probe"] == 4
        assert result["probe_completed_updates"] == result["probe_attempted_updates"] == 32
        assert result["total_completed_updates"] == 32
        assert (
            result["probe_score_forward_calls_attempted"]
            == result["probe_score_forward_calls_completed"]
            == 20
        )
        assert all(row["status"] == "failed" and row["outcomes"] is None for row in result["runs"])
        assert all(
            row["observations"][0]["probe"]["reason"] == "stub failure after eight"
            for row in result["runs"]
        )
        assert all(
            point["status"] == "not_run"
            for row in result["runs"]
            for point in row["observations"][1:]
        )
    else:
        assert result["status"] == "complete" and calls["probe"] == 16
        assert result["benchmark_completed_updates"] == 72
        assert result["probe_completed_updates"] == result["probe_attempted_updates"] == 256
        assert result["total_completed_updates"] == 328
        assert result["probe_training_row_exposures"] == 2048
        assert result["benchmark_training_row_exposures"] == 576
        assert (
            result["probe_score_forward_calls_attempted"]
            == result["probe_score_forward_calls_completed"]
            == 96
        )
        assert result["pc_neutral_curves_exact"] is True
        assert all(
            point["fresh_gain"] == -0.5 for row in result["runs"] for point in row["observations"]
        )


@pytest.mark.parametrize("phase", ("constructor", "enter"))
def test_should_keep_all_sixteen_observation_slots_when_extended_setup_fails(monkeypatch, phase):
    class FailedSampler:
        def __init__(self):
            if phase == "constructor":
                raise RuntimeError("declared sampler construction failure")

        def __enter__(self):
            raise RuntimeError("declared sampler enter failure")

        def __exit__(self, *args):
            return False

        def snapshot(self):
            raise RuntimeError("no RSS after declared failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", FailedSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    result = pilot.run_pilot(extended=True)
    assert result["status"] == "incomplete" and result["total_completed_updates"] == 0
    assert (
        len(result["runs"]) == 4 and sum(len(row["observations"]) for row in result["runs"]) == 16
    )
    assert all(row["status"] == "not_run" and row["outcomes"] is None for row in result["runs"])
    assert all(
        point["status"] == "not_run" and point["probe"] is None
        for row in result["runs"]
        for point in row["observations"]
    )
    assert result["pc_neutral_curves_exact"] is None
