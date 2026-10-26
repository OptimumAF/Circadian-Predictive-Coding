"""Checkpoint composition charges TRAIN cue gradients without extra updates.

The diagnostic port is a truthful scalar stub; its native mathematics is already
qualified separately. The existing stream/probe runner and owned fakes are reused.
"""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_gradient_readouts as gradients
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_recurrent_gradient_diagnostics import DEFINITION, GradientDiagnosticFailed
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID, TorchRecurrentPolicy
from test_minigrid_history_route import history_environment
from test_minigrid_learning_pilot import (
    Learner,
    POINTS,
    evaluator as evaluator,
    same_state,
)
from test_torch_feature_rehearsal_pilot import forbidden


PARAMETERS = ("weight_input", "weight_recurrent", "bias_hidden", "weight_output", "bias_output")


def diagnostic(pair, *, complete=True):
    rows = sum(len(episode.transitions) for episode in pair)
    geometry = dict(first_norm=1.0, second_norm=1.0, dot=-1.0, cosine=-1.0, cosine_reason=None)
    return dict(
        status="complete" if complete else "incomplete",
        reason=None if complete else "second native gradient forward failed",
        definition=DEFINITION,
        objective=TORCH_RECURRENT_CE_ID,
        losses=[
            dict(
                episode_id=episode.episode_id,
                context=episode.context,
                rows=len(episode.transitions),
                mean_ce=0.5,
            )
            for episode in (pair if complete else pair[:1])
        ],
        joint_geometry=dict(
            geometry, first_norm=float(np.sqrt(5)), second_norm=float(np.sqrt(5)), dot=-5.0
        )
        if complete
        else None,
        per_parameter_geometry={name: dict(geometry) for name in PARAMETERS} if complete else None,
        work=dict(
            snapshot_validations_attempted=1,
            snapshot_validations_completed=1,
            validation_temporary_optimizers=1,
            forward_calls_attempted=2,
            forward_calls_completed=2 if complete else 1,
            forward_rows_attempted=rows,
            forward_rows_completed=rows if complete else rows // 2,
            backward_calls_attempted=2 if complete else 1,
            backward_calls_completed=2 if complete else 1,
            backward_rows_attempted=rows if complete else rows // 2,
            backward_rows_completed=rows if complete else rows // 2,
            optimizer_steps=0,
            published_updates=0,
            failure_work_may_be_unobserved=not complete,
        ),
    )


def install_port(monkeypatch, *, callback=None):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(gradients, "_same_state", same_state)

    def read(state, pair):
        assert type(pair) is tuple and len(pair) == 2
        assert pair[0].context == pair[1].context in ("A", "B", "C_train")
        assert pair[0].episode_id != pair[1].episode_id
        assert all(
            episode.role == "train"
            and episode.num_classes == 7
            and len(episode.transitions) == 9
            and len(episode.transitions[0].observation) == 148
            for episode in pair
        )
        result = diagnostic(pair)
        call = dict(state=state, before=pickle.dumps(state), pair=pair, payload=result)
        calls.append(call)
        if callback:
            changed = callback(call, len(calls) - 1)
            if changed is not None:
                return changed
        return result

    monkeypatch.setattr(gradients, "diagnose_episode_gradients", read)
    return calls


def raw_case(monkeypatch, evaluator):
    monkeypatch.setattr(pilot, "_same_state", same_state)
    source = Learner(TorchRecurrentPolicy(148, 8, 7, 17))
    return source, source.snapshot_state(), evaluator.sources


def score(case, source, state, sources, *, check_boundary=lambda: None, **options):
    return pilot._score_checkpoint(
        source,
        state,
        method="recurrent_bc",
        updates=0,
        sources=sources,
        budget=case.budget,
        check_boundary=check_boundary,
        **options,
    )


def test_should_add_only_fifteen_recurrent_pairs_without_changing_streams_or_fresh_probes(
    monkeypatch, evaluator
):
    case = history_environment(monkeypatch, evaluator)
    calls = install_port(monkeypatch)
    report = pilot.run_pilot(public_history=True, gradient_geometry=True)
    assert report["status"] == "complete", report["reason"]
    assert len(case.work["trains"]) == 800 and len(case.work["predicts"]) == 230
    assert len(case.raw) == 25 and len(case.probes) == 10 and len(calls) == 15
    assert len(case.captured["finished"]) == 5 and case.captured["collection_calls"] == 1
    assert not case.work.get("sessions") and not case.work.get("environments")
    assert len(case.work["probe_environments"]) == 4
    assert report["work"]["native_train_calls"] == 800 and report["work"]["training_rows"] == 7200
    assert report["work"]["recurrent_unroll_steps"] == 2124
    assert report["work"]["known_raw_score_forwards_completed"] == 150
    assert report["work"]["probe_score_forwards_completed"] == 80
    recurrent = report["methods"][-1]
    assert recurrent["method"] == "recurrent_bc"
    for index, row in enumerate(report["methods"]):
        assert (
            row["charged_updates"] == 128
            and row["source_unchanged"] is row["actor_unchanged"] is True
        )
        assert row["final_evaluation"]["status"] == "not_run"
        assert row["work"]["cue_gradient_geometry"]["pair_calls_completed"] == (
            15 if index == 4 else 0
        )
        for slot in row["readouts"]:
            if index < 4:
                assert "cue_gradient_geometry" not in slot
                assert "cue_gradient_geometry" not in slot["work"]
                continue
            assert [entry["context"] for entry in slot["cue_gradient_geometry"]] == [
                "A",
                "B",
                "C_train",
            ]
            assert slot["work"]["score_forward_calls_completed"] == 6
            work = slot["work"]["cue_gradient_geometry"]
            assert work["pair_calls_attempted"] == work["pair_calls_completed"] == 3
            assert work["forward_rows_completed"] == work["backward_rows_completed"] == 54
            assert (
                work["input_array_equivalent_bytes"] == 64368 and work["train_rows_admitted"] == 54
            )
            for entry in slot["cue_gradient_geometry"]:
                assert entry["diagnostic"]["joint_geometry"]["cosine"] == -1.0
                assert entry["episode_ids"] == [
                    loss["episode_id"] for loss in entry["diagnostic"]["losses"]
                ]
        assert all("cue_gradient_geometry" not in slot["report"] for slot in row["probes"])
    assert [call["state"].completed_updates for call in calls] == [
        point for point in POINTS for _ in range(3)
    ]
    assert all(
        call["state"].policy.feature_dim == 148 and pickle.dumps(call["state"]) == call["before"]
        for call in calls
    )
    assert [call["pair"][0].context for call in calls] == ["A", "B", "C_train"] * 5
    work = report["work"]["cue_gradient_geometry"]
    assert work["pair_calls_attempted"] == work["pair_calls_completed"] == 15
    assert work["forward_rows_completed"] == work["backward_rows_completed"] == 270
    assert work["geometry_reductions_completed"] == 90
    assert work["snapshot_validations_completed"] == work["validation_temporary_optimizers"] == 15
    assert work["input_array_equivalent_bytes"] == 321840 and work["train_rows_admitted"] == 270
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is False
    assert report["scientific_advantage_verified"] is False
    calls[0]["payload"]["losses"][0]["mean_ce"] = 999.0
    assert (
        recurrent["readouts"][0]["cue_gradient_geometry"][0]["diagnostic"]["losses"][0]["mean_ce"]
        == 0.5
    )


@pytest.mark.parametrize("failure", ["native_exception", "incomplete_return", "generic"])
def test_should_keep_six_raw_scores_and_first_paid_pair_after_second_diagnostic_failure(
    monkeypatch, evaluator, failure
):
    source, state, sources = raw_case(monkeypatch, evaluator)

    def failed(call, index):
        if index != 1:
            return None
        if failure == "generic":
            raise RuntimeError("gradient port failed before observed readback")
        partial = diagnostic(call["pair"], complete=False)
        if failure == "native_exception":
            raise GradientDiagnosticFailed(partial)
        return partial

    calls = install_port(monkeypatch, callback=failed)
    report = score(evaluator, source, state, sources, gradient_geometry=True)
    assert report["status"] == "failed" and len(calls) == 2
    assert len(report["scores"]) == 6 and len(evaluator.work["predicts"]) == 6
    assert all(row["ce"] == pytest.approx(float(np.log(2))) for row in report["scores"])
    records = report["cue_gradient_geometry"]
    assert len(records) == 2 and records[0]["diagnostic"]["status"] == "complete"
    work = report["work"]["cue_gradient_geometry"]
    assert work["pair_calls_attempted"] == 2 and work["geometry_reductions_completed"] == 6
    assert work["failure_work_may_be_unobserved"] is True
    if failure == "generic":
        assert records[1]["diagnostic"] is None and work["forward_rows_completed"] == 18
    else:
        assert records[1]["diagnostic"]["status"] == "incomplete"
        assert len(records[1]["diagnostic"]["losses"]) == 1
        assert work["forward_rows_completed"] == work["backward_rows_completed"] == 27
    assert (
        work["optimizer_steps"]
        == work["published_updates"]
        == evaluator.budget.updates_completed
        == 0
    )


@pytest.mark.parametrize(
    "fault", ["source", "state", "diagnostic_state", "source_packet", "boundary"]
)
def test_should_preserve_returned_payload_before_refusing_late_state_or_evidence_change(
    monkeypatch, evaluator, fault
):
    source, state, sources = raw_case(monkeypatch, evaluator)

    def mutate(call, index):
        if fault == "source":
            source.state.optimizer_state["momentum"] += 1
        elif fault == "state":
            state.parameters["weight"] += 1
        elif fault == "diagnostic_state":
            call["state"].parameters["weight"] += 1
        elif fault == "source_packet":
            sources[0].features[0, 0] += 0.125

    calls = install_port(monkeypatch, callback=mutate)

    def guard():
        if calls and fault == "boundary":
            raise ValueError("preceding completed stream evidence changed")

    before = pickle.dumps(state)
    report = score(evaluator, source, state, sources, check_boundary=guard, gradient_geometry=True)
    assert report["status"] == "failed" and len(calls) == 1
    assert len(report["scores"]) == 6 and len(report["cue_gradient_geometry"]) == 1
    payload = report["cue_gradient_geometry"][0]["diagnostic"]
    assert payload["status"] == "complete" and payload["joint_geometry"]["cosine"] == -1.0
    assert report["work"]["cue_gradient_geometry"]["forward_rows_completed"] == 18
    assert report["work"]["cue_gradient_geometry"]["geometry_reductions_completed"] == 6
    assert report["work"]["cue_gradient_geometry"]["failure_work_may_be_unobserved"] is True
    assert evaluator.budget.updates_completed == 0 and not evaluator.work.get("trains")
    assert calls[0]["state"] is not state
    if fault != "state":
        assert pickle.dumps(state) == before


def test_should_preserve_default_raw_schema_and_explicit_false_without_diagnostic_entry(
    monkeypatch, evaluator
):
    source, state, sources = raw_case(monkeypatch, evaluator)
    monkeypatch.setattr(gradients, "diagnose_episode_gradients", forbidden)
    original = score(evaluator, source, state, sources)
    explicit = score(evaluator, source, state, sources, gradient_geometry=False)
    assert original == explicit and original["status"] == "complete"
    assert (
        "cue_gradient_geometry" not in original and "cue_gradient_geometry" not in original["work"]
    )


def test_should_refuse_work_counter_tamper_but_keep_returned_paid_native_payload(
    monkeypatch, evaluator
):
    source, state, sources = raw_case(monkeypatch, evaluator)
    raw = score(evaluator, source, state, sources)
    raw["cue_gradient_geometry"] = []
    raw["work"]["cue_gradient_geometry"] = gradients.empty_work()

    def corrupt(call, index):
        raw["work"]["cue_gradient_geometry"]["forward_rows_completed"] = 999

    calls = install_port(monkeypatch, callback=corrupt)
    with pytest.raises(ValueError, match="evidence changed"):
        gradients.run_gradient_readouts(raw, state, sources, check_boundary=lambda: None)
    assert len(calls) == len(raw["cue_gradient_geometry"]) == 1
    assert raw["cue_gradient_geometry"][0]["diagnostic"]["status"] == "complete"
    work = raw["work"]["cue_gradient_geometry"]
    assert work["forward_rows_completed"] == work["backward_rows_completed"] == 18
    assert work["pair_calls_completed"] == 1 and work["geometry_reductions_completed"] == 6
    assert work["failure_work_may_be_unobserved"] is True
    assert raw["work"]["score_forward_calls_completed"] == 6 and len(raw["scores"]) == 6


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_gradient_option_before_optional_work(
    monkeypatch, evaluator, value
):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(gradients, "diagnose_episode_gradients", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(gradient_geometry=value)
    with pytest.raises(ValueError):
        pilot.run_pilot(gradient_geometry=value)
    source, state, sources = raw_case(monkeypatch, evaluator)
    with pytest.raises(ValueError):
        score(evaluator, source, state, sources, gradient_geometry=value)
    assert not evaluator.work.get("predicts")


@pytest.mark.parametrize("fault", ["order", "missing", "role", "array_binding", "row_ids"])
def test_should_admit_only_original_six_train_packet_bindings_before_gradient_work(
    monkeypatch, evaluator, fault
):
    source, state, sources = raw_case(monkeypatch, evaluator)
    monkeypatch.setattr(gradients, "diagnose_episode_gradients", forbidden)
    if fault == "order":
        sources = (sources[1], sources[0], *sources[2:])
    elif fault == "missing":
        sources = sources[:-1]
    elif fault == "role":
        first = replace(
            sources[0], role="final_test", episode=replace(sources[0].episode, role="final_test")
        )
        sources = (first, *sources[1:])
    elif fault == "array_binding":
        sources[0].features[0, 0] += 0.125
    else:
        sources = (replace(sources[0], row_ids=sources[1].row_ids), *sources[1:])
    with pytest.raises(ValueError):
        score(evaluator, source, state, sources, gradient_geometry=True)
    assert not evaluator.work.get("predicts") and not evaluator.work.get("trains")


def test_should_refuse_non_recurrent_gradient_request_before_raw_prediction(monkeypatch, evaluator):
    monkeypatch.setattr(gradients, "diagnose_episode_gradients", forbidden)
    source, state, sources = raw_case(monkeypatch, evaluator)
    with pytest.raises(ValueError, match="recurrent_bc"):
        pilot._score_checkpoint(
            source,
            state,
            method="bp",
            updates=0,
            sources=sources,
            budget=evaluator.budget,
            check_boundary=lambda: None,
            gradient_geometry=True,
        )
    assert not evaluator.work.get("predicts")


def test_should_finish_all_optional_activation_reads_before_first_gradient_pair(
    monkeypatch, evaluator
):
    source, state, sources = raw_case(monkeypatch, evaluator)
    activations: list[bool] = []

    def hidden(snapshot, values):
        activations.append(True)
        return np.zeros((len(values), 8), dtype=np.float64)

    def after_health(call, index):
        assert len(activations) == 6 and len(evaluator.work["predicts"]) == 6

    monkeypatch.setattr(pilot, "read_native_activations", hidden)
    calls = install_port(monkeypatch, callback=after_health)
    report = score(
        evaluator, source, state, sources, activation_health=True, gradient_geometry=True
    )
    assert report["status"] == "complete", report["reason"]
    assert len(calls) == 3 and report["work"]["activation_svd_calls_completed"] == 6
    assert report["work"]["cue_gradient_geometry"]["geometry_reductions_completed"] == 18


@pytest.mark.parametrize("history", [False, True])
def test_should_describe_opt_in_without_changing_frozen_seed_dose_or_old_limits(
    monkeypatch, history
):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    original = pilot.describe_protocol(public_history=history)
    assert original == pilot.describe_protocol(public_history=history, gradient_geometry=False)
    assert "cue_gradient_geometry" not in original
    protocol = pilot.describe_protocol(public_history=history, gradient_geometry=True)
    assert protocol["protocol_id"] != original["protocol_id"]
    assert protocol["limits"] == original["limits"] and protocol["seed"] == original["seed"]
    work = deepcopy(protocol["planned_work"])
    diagnostic_work = work.pop("cue_gradient_geometry")
    work["recurrent_unroll_steps"] -= 270
    assert work == original["planned_work"]
    assert (
        protocol["protocol_id"]
        == original["protocol_id"].removesuffix("_v1") + "_cue_gradient_geometry_v1"
    )
    assert diagnostic_work["pair_calls"] == 15 and diagnostic_work["forward_rows"] == 270
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol


def test_should_distinguish_combinations_and_include_both_forward_diagnostic_costs():
    protocols = [
        pilot.describe_protocol(gradient_geometry=True),
        pilot.describe_protocol(public_history=True, gradient_geometry=True),
        pilot.describe_protocol(
            public_history=True, activation_health=True, gradient_geometry=True
        ),
    ]
    assert len({protocol["protocol_id"] for protocol in protocols}) == 3
    assert [protocol["planned_work"]["recurrent_unroll_steps"] for protocol in protocols] == [
        2124,
        2124,
        2394,
    ]


@pytest.mark.parametrize("health", [False, True])
def test_should_include_completed_diagnostic_forward_rows_in_actual_recurrent_total_only_when_opted_in(
    health,
):
    row = pilot._new_row("recurrent_bc")
    row.update(
        charged_updates=4,
        native_training_examples=36,
        wake_calls_attempted=4,
        wake_calls_returned=4,
        probes=[],
    )
    work: dict[str, Any] = dict(
        score_forward_calls_attempted=6,
        score_forward_calls_completed=6,
        score_rows_completed=54,
        failure_work_may_be_unobserved=False,
    )
    work["cue_gradient_geometry"] = gradients.empty_work()
    work["cue_gradient_geometry"]["forward_rows_completed"] = 12
    if health:
        work["activation_recurrent_hidden_steps_completed"] = 9
    row["readouts"] = [{"work": work}]
    result = pilot._work([row], [], activation_health=health, gradient_geometry=True)
    assert result["recurrent_unroll_steps"] == 90 + 12 + (9 if health else 0)
    assert result["training_rows"] == 36 and result["native_train_calls"] == 4
    legacy = pilot._work([row], [], activation_health=health)
    assert legacy["recurrent_unroll_steps"] == 90 and "cue_gradient_geometry" not in legacy
