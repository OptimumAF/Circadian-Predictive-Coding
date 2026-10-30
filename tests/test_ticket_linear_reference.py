"""Fixed LS typed/task reference: admission, isolation, costs and paid failures."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import qualify_ticket_bp_fit as entry
from scripts import ticket_linear_reference as reference
from src.adapters import linear_fit_decision as adapter
from src.core.categorical_linear_fit import (
    CategoricalLinearFit,
    predict_categorical_linear,
    predict_categorical_linear_logits,
)
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionCandidate, DecisionRequest


def literal_reader():
    catalog = tuple(DecisionCandidate(key, key) for key in ("a", "b", "c"))
    spec = DecisionFeatureSpec(("cue",), catalog)
    # Large unavailable c logit demonstrates candidate-local mass normalization.
    model = CategoricalLinearFit(1, 3, 1, ((0.0, 0.0, 0.0), (1.0, 2.0, 1000.0)), 1, (1.0,), 0.0)
    return adapter.LinearFitDecisionPredictor(spec, model), DecisionRequest("r", "cue", catalog[:2])


def test_should_reuse_affine_scores_and_normalize_only_allowed_candidates():
    reader, request = literal_reader()
    x = np.ones((1, 1), dtype=np.float64)
    logits = predict_categorical_linear_logits(reader.model, x)
    assert logits.tolist() == [[1.0, 2.0, 1000.0]]
    probabilities = predict_categorical_linear(reader.model, x)
    assert probabilities[0, :2].sum() == 0.0
    result = reader(request)
    assert result.selected_id == "b"
    assert [s.candidate_id for s in result.scores] == ["a", "b"]
    assert sum(s.probability for s in result.scores) == pytest.approx(1.0)
    assert result.scores[1].probability == pytest.approx(1 / (1 + np.exp(-1)))
    assert reader.work.affine_attempts == reader.work.affine_returns == 1
    assert reader.work.rows_attempted == 1 and reader.work.nominal_affine_terms == 6


def test_should_own_readout_inputs_and_return_owned_logits():
    reader, request = literal_reader()
    x = np.ones((1, 1), dtype=np.float64)
    before = x.copy()
    logits = reader.predict_logits(x)
    logits[0, 0] = 999
    assert np.array_equal(x, before) and reader(request).selected_id == "b"
    work = reader.work
    object.__setattr__(work, "rows_attempted", 999)
    reader.assert_unchanged()
    assert reader.work.rows_attempted == 2


@pytest.mark.parametrize("fault", ["model", "model_replace", "spec", "work"])
def test_should_refuse_mutated_or_replaced_frozen_readout_before_affine_work(fault):
    reader, request = literal_reader()
    if fault == "model":
        object.__setattr__(reader.model, "squared_error", 1.0)
    elif fault == "model_replace":
        reader.model = deepcopy(reader.model)
    elif fault == "spec":
        object.__setattr__(reader.spec, "vocabulary", ("other",))
    else:
        object.__setattr__(reader._work, "rows_attempted", 999)
    with pytest.raises(ValueError, match="changed"):
        reader(request)
    assert reader.work.affine_attempts == 0


@pytest.mark.parametrize("fault", ["width", "dtype", "nonfinite"])
def test_should_keep_failed_affine_attempt_without_return(fault):
    reader, _ = literal_reader()
    x = (
        np.zeros((1, 2))
        if fault == "width"
        else np.zeros((1, 1), dtype=np.float32)
        if fault == "dtype"
        else np.array([[np.nan]])
    )
    with pytest.raises(ValueError):
        reader.predict_logits(x)
    assert reader.work.affine_attempts == 1 and reader.work.affine_returns == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(linear_reference=1),
        dict(linear_reference=True),
        dict(linear_reference=True, policy_context=1),
    ],
)
def test_should_refuse_invalid_mode_before_teacher(options, monkeypatch):
    monkeypatch.setattr(entry, "_context_teacher", lambda spec: pytest.fail("read teacher"))
    with pytest.raises(ValueError):
        entry.run_qualification(**options)


@pytest.mark.parametrize("linear", [False, True])
def test_should_forward_cli_without_changing_bp_default(linear, tmp_path, monkeypatch):
    seen = []

    def run(**kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_qualification", run)
    path = tmp_path / "one.json"
    args = ["--output", str(path), "--policy-context"] + (["--linear-reference"] if linear else [])
    assert entry.main(args) == 0
    assert seen == [dict(policy_context=True, **(dict(linear_reference=True) if linear else {}))]
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert len(seen) == 1


def test_should_refuse_cli_without_context_before_output(tmp_path):
    path = tmp_path / "refused.json"
    with pytest.raises(SystemExit) as error:
        entry.main(["--linear-reference", "--output", str(path)])
    assert error.value.code == 2 and not path.exists()


@pytest.mark.parametrize("fault", ["final", "probe", "target", "features"])
def test_should_refuse_other_roles_or_unaligned_packet_before_solve(fault, monkeypatch):
    original = entry._context_teacher

    def teacher(spec):
        cases, batch = original(spec)
        if fault in ("final", "probe"):
            role = "final_test" if fault == "final" else "probe_train"
            cases = (replace(cases[0], role=role), *cases[1:])
        elif fault == "target":
            batch.targets[0, 0] = (batch.targets[0, 0] + 1) % 6
        else:
            batch.features[0, 0] = 1 - batch.features[0, 0]
        return cases, batch

    monkeypatch.setattr(entry, "_context_teacher", teacher)
    monkeypatch.setattr(reference, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    r = entry.run_qualification(policy_context=True, linear_reference=True)
    assert not r["complete"] and r["fit_work"]["attempted_fits"] == 0


def test_should_complete_one_train_fit_and_real_tasks_without_native_owners():
    r = entry.run_qualification(policy_context=True, linear_reference=True)
    assert r["complete"], r.get("error")
    assert r["fit_work"] == dict(
        attempted_fits=1, returned_fits=1, rows_attempted=12, unreturned_work_unknown=False
    )
    assert r["model"]["fit_rows"] == 12 and r["model"]["feature_dim"] == 9
    assert r["model"]["num_classes"] == 6 and len(r["model"]["coefficients"]) == 10
    assert len(r["episodes"]) == 12 and all(e["episode"]["complete"] for e in r["episodes"])
    assert all(e["accepted"] and e["returned_tasks"] == 1 for e in r["episodes"])
    assert [(r["policy_version"], r["concept"]) for r in r["episodes"][::3]] == list(
        entry.CONTEXT_TASKS
    )
    assert r["numpy_rng_unchanged"] and r["python_rng_unchanged"]
    assert all(
        r[k] == 0
        for k in (
            "native_updates",
            "deployed_updates",
            "benchmark_updates",
            "final_examples",
            "probe_updates",
            "replay_examples",
        )
    )
    assert r["budget"]["committed_fits"] == 1 and r["budget"]["limits"]["max_wall_seconds"] == 30
    assert r["readout_work"]["rows_attempted"] == 12 + r["readout_work"]["decision_attempts"]
    assert r["readout_work"]["affine_attempts"] == r["readout_work"]["affine_returns"]
    assert not r["readout_work_unknown"] and len(entry._encode_report(r)) <= 512 * 1024


@pytest.mark.parametrize("fault", ["lost_solve", "input", "lost_read", "rng"])
def test_should_preserve_paid_fit_failure_and_refuse_changed_source(fault, monkeypatch):
    original = reference.fit_categorical_linear
    numpy_rng = deepcopy(np.random.get_state())

    def fit(x, y, k):
        result = original(x, y, k)
        if fault == "lost_solve":
            raise RuntimeError("lost solve return")
        if fault == "input":
            x[0, 0] = 1 - x[0, 0]
        if fault == "rng":
            state = np.random.get_state()
            assert isinstance(state, tuple)
            np.random.set_state((state[0], state[1], 0 if state[2] else 1, state[3], state[4]))
        return result

    monkeypatch.setattr(reference, "fit_categorical_linear", fit)
    if fault == "lost_read":
        original_read = adapter.predict_categorical_linear_logits

        def lost(model, x):
            original_read(model, x)
            raise RuntimeError("lost read return")

        monkeypatch.setattr(adapter, "predict_categorical_linear_logits", lost)
    try:
        r = entry.run_qualification(policy_context=True, linear_reference=True)
        assert not r["complete"] and r["fit_work"]["attempted_fits"] == 1
        assert r["fit_work"]["rows_attempted"] == 12 and len(r["episodes"]) == 0
        assert r["fit_work"]["returned_fits"] == int(fault != "lost_solve")
        if fault == "lost_read":
            assert r["readout_work_unknown"] and r["readout_work"]["rows_attempted"] == 12
        if fault == "rng":
            assert not r["numpy_rng_unchanged"]
    finally:
        np.random.set_state(numpy_rng)


@pytest.mark.parametrize("fault", ["fit_record", "episode_record"])
def test_should_refuse_previous_paid_evidence_and_preserve_original_prefix(fault, monkeypatch):
    saved: dict[str, Any] = {}
    original_fit = reference.teacher_fit
    original_episode = reference.run_workflow_episode
    original_copy = reference.deepcopy

    def capture_paid_owner(value):
        # Inject a foreign mutation of the actual published record, rather than
        # the producer's return (the runner now owns a detached copy of that).
        if type(value) is dict and value.get("accepted") is True and "episode" in value:
            saved["paid_owner"] = value
        return original_copy(value)

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        saved["fit"] = result
        return result

    def episode(*args, **kwargs):
        if fault == "episode_record" and "paid_owner" in saved:
            saved["paid_owner"]["episode"]["success"] = not saved["paid_owner"]["episode"][
                "success"
            ]
        result = original_episode(*args, **kwargs)
        if fault == "fit_record":
            saved["fit"]["global_mean_ce"] = 999.0
        saved["episode"] = result
        return result

    monkeypatch.setattr(reference, "teacher_fit", fit)
    monkeypatch.setattr(reference, "run_workflow_episode", episode)
    monkeypatch.setattr(reference, "deepcopy", capture_paid_owner)
    r = entry.run_qualification(policy_context=True, linear_reference=True)
    assert not r["complete"] and r["fit_work"]["attempted_fits"] == 1
    assert "paid" in r["error"]
    assert r["preserved_paid_evidence"]["teacher_fit"]["global_mean_ce"] != 999.0
    paid = r["preserved_paid_evidence"]["episodes"]
    assert len(paid) == 1 + int(fault == "episode_record")
    assert r["task_work"]["attempted_tasks"] == r["task_work"]["returned_tasks"] == len(paid)
    assert paid[-1]["returned_tasks"] == 1 and not paid[-1]["accepted"]
    assert paid[-1]["episode"]["work"]["prediction_calls_attempted"] >= 1


def test_should_keep_unknown_task_work_after_lost_return_and_stop(monkeypatch):
    original = reference.run_workflow_episode
    calls = []

    def lost(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        raise RuntimeError("lost task return")

    monkeypatch.setattr(reference, "run_workflow_episode", lost)
    r = entry.run_qualification(policy_context=True, linear_reference=True)
    assert not r["complete"] and len(calls) == 1
    assert r["task_work"] == dict(attempted_tasks=1, returned_tasks=0, unreturned_work_unknown=True)
    paid = r["preserved_paid_evidence"]["episodes"]
    assert len(paid) == 1 and not paid[0]["accepted"] and paid[0]["episode"] is None
