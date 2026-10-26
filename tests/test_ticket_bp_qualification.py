"""Tiny structural qualification fixtures; the96update scientific case is separate."""

import json
from typing import Any

import pytest

import scripts.qualify_ticket_bp_fit as study
import scripts.diagnose_ticket_training as diagnosis
from src.adapters.torch_feature_learner import TorchFeatureLearner


@pytest.fixture(scope="module")
def completed():
    original = study.AttributionStream
    holder: dict[str, Any] = {}

    def capture(*args, **kwargs):
        stream = original(*args, **kwargs)
        holder.update(stream=stream, seed=args[0])
        return stream

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "UPDATES", 2)
        patch.setattr(study, "AttributionStream", capture)
        report = study.run_qualification()
    assert report["complete"], report.get("error")
    return report, holder


def test_should_use_existing_delayed_stream_and_exact_joint_teacher(completed):
    r, holder = completed
    assert r["planned_updates"] == 2 and r["rows_per_update"] == 6
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=2, unreturned_work_unknown=False
    )
    assert r["training_data"]["role"] == "train" and r["training_data"]["all_available_at_outset"]
    assert [p["source_updates"] for p in r["checkpoints"]] == [0, 2]
    assert [p["source_rows"] for p in r["checkpoints"]] == [0, 12]
    assert [p["state"]["completed_updates"] for p in r["checkpoints"]] == [0, 2]
    assert (
        r["stream"]["evaluation_scope"] == "training_only"
        and r["stream"]["score_forward_calls"] == 0
    )
    receipts = r["applied_updates"]
    assert len(receipts) == 2 and all(x["applied_at"] > x["observed_at"] for x in receipts)
    assert all(
        x["diagnostic"]["definition"] == "torch_feature_cross_entropy_mean_v1" for x in receipts
    )
    assert holder["seed"].completed_updates == 0
    assert r["source_seed_unchanged"] and r["deployed_actor_unchanged"]
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert (
        r["final_examples"]
        == r["reserved_probe_updates"]
        == r["replay_examples"]
        == r["external_actions"]
        == 0
    )


def test_should_keep_fixed_readouts_and_honest_teacher_task_work(completed):
    r, _ = completed
    assert len(r["episodes"]) == 12
    expected = [
        (p, c, s)
        for p in (0, 2)
        for c in ("invoice", "parcel")
        for s in ("ordinary", "stale_once", "revoke_commit")
    ]
    assert [
        (e["source_updates"], e["concept"], e["episode"]["scenario"]) for e in r["episodes"]
    ] == expected
    assert all(
        e["episode"]["complete"] and e["episode"]["training_updates"] == 0 for e in r["episodes"]
    )
    assert r["teacher_reads"] == dict(attempted=2, returned=2, rows_attempted=12)
    calls = sum(e["episode"]["work"]["prediction_calls_attempted"] for e in r["episodes"])
    assert (
        calls == sum(e["work"]["native_forward_attempts"] for e in r["native_readout_work"]) <= 72
    )
    assert all(len(p["teacher_fit"]["rows"]) == 6 for p in r["checkpoints"])
    assert r["costs"]["attempted_training_nominal_affine_terms"] == 12 * 39
    assert json.loads(study._encode_report(r))["complete"]


def test_should_preserve_paid_lost_second_update_and_cli_prefix(tmp_path, monkeypatch):
    original = TorchFeatureLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second qualification return lost")
        return original(self, features, targets)

    monkeypatch.setattr(study, "UPDATES", 2)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    output = tmp_path / "failed.json"
    assert study.main(["--output", str(output)]) == 1
    r = json.loads(output.read_bytes())
    assert not r["complete"] and "return lost" in r["error"]
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=1, unreturned_work_unknown=True
    )
    assert r["budget"]["updates_completed"] == 1 and len(r["applied_updates"]) == 1
    assert len(r["episodes"]) == 6 and len(r["checkpoints"]) == 1


def test_should_charge_teacher_readout_failure_before_any_fit(monkeypatch):
    def fail(*args):
        raise RuntimeError("qualification teacher return lost")

    monkeypatch.setattr(study, "UPDATES", 2)
    monkeypatch.setattr(TorchFeatureLearner, "predict_logits", fail)
    r = study.run_qualification()
    assert not r["complete"] and "teacher return lost" in r["error"]
    assert r["teacher_reads"] == dict(attempted=1, returned=0, rows_attempted=6)
    assert r["training_work"]["attempted_updates"] == 0 and r["episodes"] == []


def test_should_refuse_changed_qualification_actor_after_readout(monkeypatch):
    original_stream = study.AttributionStream
    original_predict = study.native.TorchDecisionPredictor.__call__
    holder: dict[str, Any] = {}

    def capture(*args, **kwargs):
        stream = original_stream(*args, **kwargs)
        holder["stream"] = stream
        return stream

    def corrupt(self, request):
        result = original_predict(self, request)
        with study.native.require_torch().no_grad():
            holder["stream"]._runtime.actor._learner._head.bias_output.add_(1)
        return result

    monkeypatch.setattr(study, "UPDATES", 2)
    monkeypatch.setattr(study, "AttributionStream", capture)
    monkeypatch.setattr(study.native.TorchDecisionPredictor, "__call__", corrupt)
    r = study.run_qualification()
    assert not r["complete"] and "source/actor/seed/budget" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0
    assert r["native_readout_work"][0]["work"]["native_forward_returns"] == 1
    assert r["episodes"][0]["episode"]["work"]["prediction_calls_validated"] == 0


def test_should_refuse_probe_teacher_before_model_construction(monkeypatch):
    cases = diagnosis.training_cases()
    object.__setattr__(cases[0], "role", "probe_train")
    monkeypatch.setattr(diagnosis, "training_cases", lambda: cases)
    monkeypatch.setattr(
        study.native, "counted_learner", lambda *a: pytest.fail("probe teacher reached fit")
    )
    r = study.run_qualification()
    assert not r["complete"] and "TRAIN teacher" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


def test_should_refuse_occupied_output_before_native_calls(tmp_path, monkeypatch):
    output = tmp_path / "preserve.json"
    output.write_text("original")
    monkeypatch.setattr(
        study, "run_qualification", lambda: pytest.fail("occupied output reached native")
    )
    with pytest.raises(FileExistsError):
        study.main(["--output", str(output)])
    assert output.read_text() == "original"
