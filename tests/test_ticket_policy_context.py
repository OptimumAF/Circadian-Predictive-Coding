"""Public context plumbing and exact default preservation, without model fits."""

from copy import deepcopy
import json
from typing import Any

import numpy as np
import pytest

from scripts import run_ticket_workflow as example
from src.app.ticket_workflow import run_workflow_episode
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.ticket_workflow import (
    ACTION_IDS,
    CATALOG,
    WorkflowCapabilities,
    WorkflowState,
    dispatch_workflow,
    workflow_request,
)
from test_ticket_workflow import decision


CAPS = WorkflowCapabilities(ACTION_IDS)


def test_should_keep_context_teacher_out_of_original_strict_diagnostic(monkeypatch):
    from scripts import diagnose_ticket_training as diagnosis

    cases = example.training_cases(include_policy_context=True)
    monkeypatch.setattr(diagnosis, "training_cases", lambda: cases)
    with pytest.raises(ValueError, match="original TRAIN teacher"):
        diagnosis._teacher(DecisionFeatureSpec(example.CONTEXT_FEATURES, CATALOG))


@pytest.fixture(scope="module")
def teachers():
    return {
        v: example.training_cases(policy_version=v, include_policy_context=True)
        for v in ("v1", "v2")
    }


def test_should_expose_only_public_context_and_preserve_other_observation_fields():
    for version in ("v1", "v2"):
        state = WorkflowState(
            "private-target-route-password-identifier", "invoice", policy_version=version
        )
        for _ in range(3):
            old = workflow_request(state, CAPS, "request")
            tagged = workflow_request(state, CAPS, "request", include_policy_context=True)
            assert workflow_request(state, CAPS, "request", include_policy_context=False) == old
            assert tagged.allowed_candidates == old.allowed_candidates
            assert tagged.request_id == old.request_id and set(vars(tagged)) == set(vars(old))
            assert tagged.query == old.query.replace(" feedback ", f" policy_{version} feedback ")
            assert state.ticket_id not in tagged.query
            target = (
                "open"
                if state.phase == "sealed"
                else "commit"
                if state.phase == "staged"
                else "route"
                if version == "v1"
                else "inspect"
            )
            state = dispatch_workflow(state, target, state.revision, CAPS).state


def test_should_align_every_context_teacher_with_public_episode_observations(teachers):
    spec = DecisionFeatureSpec(example.CONTEXT_FEATURES, CATALOG)
    assert len(example.FEATURES) == 7 and len(spec.vocabulary) == 9
    assert tuple(sorted((*example.FEATURES, "policy_v1", "policy_v2"))) == spec.vocabulary
    for version, cases in teachers.items():
        plain = example.training_cases(policy_version=version)
        assert [c.target_id for c in cases] == [c.target_id for c in plain]
        assert all(c.role == "train" and c.source_group.endswith("-context") for c in cases)
        for index, concept in enumerate(example.VOCABULARY):
            observed: list[Any] = []

            def predict(request):
                teacher = cases[3 * index + len(observed)]
                observed.append(request)
                assert teacher.request.allowed_candidates == request.allowed_candidates
                assert np.array_equal(
                    encode_decision(teacher.request, spec)[0], encode_decision(request, spec)[0]
                )
                return decision(request, teacher.target_id)

            report = run_workflow_episode(
                WorkflowState("public", concept, policy_version=version),
                predict,
                capabilities=CAPS,
                include_policy_context=True,
            )
            assert report["complete"] and report["success"] and report["policy_context_in_query"]
            assert (
                len(observed) == 3 and report["training_updates"] == report["external_actions"] == 0
            )


def test_should_remove_cross_policy_label_aliasing_without_changing_old_features(teachers):
    old_spec, spec = (
        DecisionFeatureSpec(words, CATALOG)
        for words in (example.FEATURES, example.CONTEXT_FEATURES)
    )
    table: dict[tuple[Any, ...], str] = {}
    for cases in teachers.values():
        for case in cases:
            key = tuple(encode_decision(case.request, spec)[0][0])
            assert table.setdefault(key, case.target_id) == case.target_id
    assert len(table) == 12
    invoice = [cases[3 * example.VOCABULARY.index("invoice") + 1] for cases in teachers.values()]
    assert [c.target_id for c in invoice] == ["route", "inspect"]
    old = [encode_decision(c.request, old_spec)[0] for c in invoice]
    contextual = [encode_decision(c.request, spec)[0] for c in invoice]
    assert np.array_equal(old[0], old[1])
    assert not np.array_equal(contextual[0], contextual[1])
    assert all(
        {"route", "inspect"}.issubset({a.candidate_id for a in c.request.allowed_candidates})
        for c in invoice
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("scenario", ["ordinary", "stale_once", "revoke_commit"])
def test_should_preserve_permission_revision_and_effect_semantics_with_context(version, scenario):
    def predict(request):
        words = request.query.split()
        action = (
            "open"
            if "sealed" in words
            else "commit"
            if "staged" in words
            else "route"
            if version == "v1"
            else "inspect"
        )
        return decision(request, action)

    state = WorkflowState("same-task", "invoice", policy_version=version)
    before = deepcopy(state)
    old = run_workflow_episode(state, predict, capabilities=CAPS, scenario=scenario)
    tagged = run_workflow_episode(
        state, predict, capabilities=CAPS, scenario=scenario, include_policy_context=True
    )
    assert state == before
    assert tagged["success"] == old["success"] == (scenario != "revoke_commit")
    for key in (
        "stopped_reason",
        "final_state",
        "final_capabilities",
        "error_recovery_demonstrated",
    ):
        assert tagged[key] == old[key]
    assert tagged["work"]["dispatches_completed"] == old["work"]["dispatches_completed"]
    assert all(f"policy_{version}" in s["request"]["query"] for s in tagged["steps"])


def test_should_keep_default_and_false_episode_reports_exactly_equal():
    state = WorkflowState("default", "invoice")

    def predict(request):
        words = request.query.split()
        return decision(
            request, "open" if "sealed" in words else "commit" if "staged" in words else "inspect"
        )

    assert run_workflow_episode(state, predict, capabilities=CAPS) == run_workflow_episode(
        state, predict, capabilities=CAPS, include_policy_context=False
    )


@pytest.mark.parametrize("bad", [None, 0, 1, "v1", [], np.bool_(True)])
def test_should_refuse_nonbool_context_before_any_request_teacher_or_predictor(bad, monkeypatch):
    with pytest.raises(ValueError, match="exact bool"):
        workflow_request(None, None, "bad", include_policy_context=bad)
    monkeypatch.setattr(
        example,
        "workflow_request",
        lambda *a, **k: pytest.fail("invalid option reached teacher requests"),
    )
    with pytest.raises(ValueError, match="exact bool"):
        example.training_cases(include_policy_context=bad)
    with pytest.raises(ValueError, match="exact bool"):
        run_workflow_episode(
            None,
            lambda *a: pytest.fail("prediction"),
            capabilities=None,
            include_policy_context=bad,
        )
    with pytest.raises(ValueError, match="exact bool"):
        example.run_example(include_policy_context=bad)


def test_should_pass_context_to_existing_fit_and_episodes_without_running_a_model(monkeypatch):
    calls: list[Any] = []
    spec = DecisionFeatureSpec(example.CONTEXT_FEATURES, CATALOG)

    def fit(report, budget, **options):
        calls.append(options)
        report["snapshots"] = [{}, {}]
        report["training_work"] = dict(latest_observed_mistakes=3)
        return spec, {0: "initial", example.PASSES: "trained"}

    def reader(specification, state):
        assert specification is spec
        return lambda request: decision(request, "open")

    def episode(state, predict, **options):
        assert options["include_policy_context"] is True
        assert state.policy_version == "v2"
        calls.append("episode")
        return dict(
            complete=True,
            success=False,
            scenario=options["scenario"],
            stopped_reason="fixture",
            error_recovery_demonstrated=False,
            work=dict(prediction_calls_validated=1),
        )

    monkeypatch.setattr(example, "fit_reference", fit)
    monkeypatch.setattr(example, "PerceptronDecisionPredictor", reader)
    monkeypatch.setattr(example, "run_workflow_episode", episode)
    report = example.run_example(include_policy_context=True)
    assert report["complete"] and report["definition"] == "local_learned_ticket_workflow_context_v1"
    assert report["policy_context_in_query"] and calls[0] == dict(include_policy_context=True)
    assert calls.count("episode") == 18
    assert report["costs"]["retained_snapshot_coefficient_bytes"] == 960
    assert report["costs"]["validated_readout_affine_terms"] == 900
    assert report["costs"]["committed_coefficient_addition_terms"] == 60


def test_should_build_context_batches_in_existing_fit_without_entering_learner(monkeypatch):
    import scripts.run_ticket_workflow as module

    def stop(policy):
        assert policy.feature_dim == 9
        raise RuntimeError("scripted boundary before learner construction")

    monkeypatch.setattr(module, "PerceptronLearner", stop)
    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="scripted boundary"):
        module.fit_reference(report, None, include_policy_context=True)
    assert len(report["feature_spec"]["vocabulary"]) == 9
    assert all(row["source_group"].endswith("-context") for row in report["training_cases"])


def test_should_dispatch_cli_option_and_preserve_fresh_output(monkeypatch, tmp_path):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(example, "run_example", run)
    out = tmp_path / "context.json"
    assert example.main(["--policy-context", "--output", str(out)]) == 0
    assert calls == [dict(include_policy_context=True)] and json.loads(out.read_bytes()) == dict(
        complete=True
    )
    with pytest.raises(FileExistsError):
        example.main(["--policy-context", "--output", str(out)])
    assert len(calls) == 1
