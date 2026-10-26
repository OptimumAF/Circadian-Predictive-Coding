"""Real native chronological recipe and partial-work/isolation checks."""

import json
from typing import Any

import pytest

import scripts.recurring_ticket_workflow_example as study
from scripts import ticket_workflow_native as native
from src.adapters.torch_feature_learner import TorchFeatureLearner


@pytest.fixture(scope="module", params=native.METHODS)
def completed(request):
    holder: dict[str, Any] = {}
    original = study.AttributionStream

    def capture(*args, **kwargs):
        stream = original(*args, **kwargs)
        holder.update(
            stream=stream, source=args[0], actor_before=stream._runtime.actor.snapshot_state().state
        )
        return stream

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "AttributionStream", capture)
        report = study.run_example(method=request.param)
    assert report["complete"], report.get("error")
    return report, holder


def test_actual_native_sequence_has_exact_original_task_schedule_and_paid_work(completed):
    report, holder = completed
    assert report["method"] in native.METHODS
    assert report["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=36,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert [p["source_updates"] for p in report["checkpoints"]] == list(study.CHECKPOINTS)
    assert [p["source_rows"] for p in report["checkpoints"]] == [0, 12, 24, 36]
    assert all(p["source_mistakes"] is None for p in report["checkpoints"])
    for point in study.CHECKPOINTS:
        actual = [
            (r["episode"]["scenario"], r["concept"])
            for r in report["episodes"]
            if r["source_updates"] == point
        ]
        assert tuple(actual) == study.episode_slots(point)
    assert report["summary"] == study.summarize(report["episodes"])
    assert len(report["episodes"]) == 11 and all(
        r["episode"]["complete"] for r in report["episodes"]
    )
    assert all(r["episode"]["training_updates"] == 0 for r in report["episodes"])
    assert (
        report["stream"]["score_forward_calls"] == 0
        and report["stream"]["evaluation_scope"] == "training_only"
    )
    receipts = report["applied_updates"]
    assert len(receipts) == 12 and all(r["applied_at"] > r["observed_at"] for r in receipts)
    assert (
        report["probe_updates"]
        == report["final_examples"]
        == report["replay_examples"]
        == report["external_actions"]
        == 0
    )
    assert report["numpy_rng_unchanged"] and report["torch_rng_unchanged"]
    assert holder["source"].completed_updates == 0
    assert native.same_state(
        holder["actor_before"], holder["stream"]._runtime.actor.snapshot_state().state
    )
    assert report["costs"]["attempted_training_nominal_affine_terms"] == 36 * 39
    assert "attempted_training_affine_terms" not in report["costs"]
    calls = report["episode_work"]["prediction_calls_validated"]
    work = [r["work"] for r in report["native_readout_work"]]
    assert calls == sum(w["native_forward_attempts"] for w in work) <= 66
    assert all(w["native_forward_attempts"] == w["decision_returns"] for w in work)
    assert report["costs"]["attempted_readout_nominal_affine_terms"] == calls * 39
    assert report["costs"]["retained_checkpoint_tensor_bytes"] == sum(
        p["tensor_bytes"] for p in report["checkpoints"]
    )
    # Includes the full CPC RNG state or full trained BP gradients/momentum.
    final = report["checkpoints"][-1]["state"]
    if report["method"] == "bp":
        assert final["gradients"]["bias_output"]["dtype"] == "torch.float32"
        assert len(final["optimizer_state"]["state"]) == 4
    elif report["method"] != "pc":
        assert len(final["head_state"]["split_generator_state"]["values"]) > 1000
        assert final["head_state"]["neuron_age"]["values"] == [12.0] * 3


def test_native_report_roundtrips_complete_json_safe_checkpoint(completed):
    report, _ = completed
    encoded = study._encode_report(report)
    assert len(encoded) <= 512 * 1024
    roundtrip = json.loads(encoded)
    assert roundtrip["checkpoints"][-1]["state"]["completed_updates"] == 12
    assert (
        roundtrip["summary"]["new_task_success_gain"] == report["summary"]["new_task_success_gain"]
    )


def test_paid_fifth_native_failure_keeps_real_prefix(monkeypatch):
    original = TorchFeatureLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("native fifth return lost")
        return original(self, features, targets)

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    r = study.run_example(method="bp")
    assert not r["complete"] and "return lost" in r["error"]
    assert (
        r["training_work"]["attempted_updates"] == 5 and r["training_work"]["attempted_rows"] == 15
    )
    assert r["training_work"]["returned_updates"] == r["budget"]["updates_completed"] == 4
    assert r["training_work"]["unreturned_work_unknown"]
    assert len(r["applied_updates"]) == 4 and len(r["episodes"]) == 4


def test_paid_prediction_fault_preserves_native_readout_attempt(monkeypatch):
    def fail(self, features):
        raise RuntimeError("native prediction return lost")

    monkeypatch.setattr(TorchFeatureLearner, "predict_logits", fail)
    r = study.run_example(method="pc")
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert r["native_readout_work"][0]["work"] == dict(
        decision_attempts=1, native_forward_attempts=1, native_forward_returns=0, decision_returns=0
    )
    assert r["episode_work"]["prediction_calls_attempted"] == 1


def test_mutated_training_source_refused_after_native_readout(monkeypatch):
    original = study.AttributionStream
    holder: dict[str, Any] = {}

    def capture(*args, **kwargs):
        stream = original(*args, **kwargs)
        holder["stream"] = stream
        return stream

    original_readout = native.TorchDecisionPredictor.__call__

    def corrupt(self, request):
        result = original_readout(self, request)
        holder["stream"]._runtime._candidate._head.bias_output.add_(1)
        return result

    monkeypatch.setattr(study, "AttributionStream", capture)
    monkeypatch.setattr(native.TorchDecisionPredictor, "__call__", corrupt)
    r = study.run_example(method="pc")
    assert not r["complete"] and "source or budget" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0
    assert r["native_readout_work"][0]["work"]["native_forward_returns"] == 1
    assert r["episode_work"]["prediction_calls_validated"] == 0


def test_final_train_trace_refused_before_native_initialization(monkeypatch):
    cases = study.training_cases()
    # A bypassed frozen record, so admission must revalidate rather than trust it.
    object.__setattr__(cases[0], "role", "final_test")
    monkeypatch.setattr(study, "training_cases", lambda: cases)
    monkeypatch.setattr(native, "counted_learner", lambda *a: pytest.fail("final reached native"))
    r = study.run_example(method="bp")
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert r["episodes"] == []


@pytest.mark.parametrize(
    "method,replay", [(True, "none"), ("unknown", "none"), ("bp", "recent"), ("pc", "coverage")]
)
def test_unsupported_native_configuration_refused_before_work(method, replay, monkeypatch):
    monkeypatch.setattr(native, "counted_learner", lambda *a: pytest.fail("invalid fit"))
    with pytest.raises(ValueError):
        study.run_example(method=method, replay=replay)


def test_cli_passes_explicit_native_method_without_changing_old_default(tmp_path, monkeypatch):
    seen = []

    def run(**kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    assert study.main(["--output", str(tmp_path / "native.json"), "--method", "neutral_cpc"]) == 0
    assert study.main(["--output", str(tmp_path / "default.json")]) == 0
    assert seen == [dict(method="neutral_cpc", replay="none"), {}]
