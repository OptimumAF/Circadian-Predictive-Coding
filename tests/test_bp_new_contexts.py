"""Full BP readback plus tiny new-context stream/failure integration."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import scripts.run_bp_new_contexts as study
from scripts import ticket_workflow_native as native
from src.adapters.torch_bp_record import bp_snapshot_from_record
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"
)


def record():
    return json.loads(SOURCE.read_bytes())["checkpoints"][-1]["state"]


def test_complete_bp_readback_preserves_all_values_and_detached_storage():
    original = record()
    before = deepcopy(original)
    snapshot = bp_snapshot_from_record(original)
    actual = json.loads(json.dumps(native.snapshot_record(snapshot)))
    assert (
        actual == before and snapshot.completed_updates == 96 and snapshot.training_examples == 576
    )
    original["head_state"]["bias_output"]["values"][0][0] = 0.0
    original["optimizer_state"]["state"]["0"]["momentum_buffer"]["values"][0][0] = 0.0
    original["gradients"]["bias_hidden"]["values"][0][0] = 0.0
    assert json.loads(json.dumps(native.snapshot_record(snapshot))) == before


def test_readback_exact_continuation_against_live_owned_snapshot():
    # Two live steps, not a warmup refit: compare actual pre-serialization state
    # with readback before continuation, including SGD momentum and gradients.
    snapshot = bp_snapshot_from_record(record())
    first = TorchFeatureLearner(snapshot.policy)
    first.restore_state(snapshot)
    in_memory = first.snapshot_state()
    decoded = bp_snapshot_from_record(json.loads(json.dumps(native.snapshot_record(in_memory))))
    assert native.same_state(in_memory, decoded)
    second = TorchFeatureLearner(snapshot.policy)
    second.restore_state(decoded)
    _, teacher, _ = study._teacher_batches(DecisionFeatureSpec(study.FEATURES, CATALOG))
    assert first.train_batch(teacher.features[:6], teacher.targets[:6]) == second.train_batch(
        teacher.features[:6], teacher.targets[:6]
    )
    assert native.same_state(first.snapshot_state(), second.snapshot_state())
    assert first.completed_updates == 97 and first.training_examples == 582


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "dtype",
        "shape",
        "nonfinite",
        "rounded",
        "gradient",
        "momentum",
        "id",
        "clock",
        "method",
    ],
)
def test_readback_refuses_corrupt_complete_state(fault):
    r = record()
    if fault == "extra":
        r["unknown"] = 1
    elif fault in ("dtype", "shape", "nonfinite", "rounded"):
        v = r["head_state"]["bias_output"]
        if fault == "dtype":
            v["dtype"] = "torch.float64"
        elif fault == "shape":
            v["shape"] = [6]
        elif fault == "nonfinite":
            v["values"][0][0] = float("nan")
        else:
            v["values"][0][0] = 0.123456789
    elif fault == "gradient":
        r["gradients"]["bias_hidden"] = None
    elif fault == "momentum":
        r["optimizer_state"]["param_groups"][0]["lr"] = 0.1
    elif fault == "id":
        r["optimizer_state"]["state"]["00"] = r["optimizer_state"]["state"].pop("0")
    elif fault == "clock":
        r["completed_updates"] = True
    else:
        r["policy"]["method"] = "pc"
    with pytest.raises(ValueError):
        bp_snapshot_from_record(r)


@pytest.fixture(scope="module")
def completed():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "BATCHES_PER_PHASE", 1)
        result = study.run_case(SOURCE)
    assert result["complete"], result.get("error")
    return result


def test_real_stream_preserves_native_offsets_and_prior_charges(completed):
    r = completed
    assert r["prior_training_work"] == dict(
        attempted_updates=96, attempted_rows=576, returned_updates=96, unreturned_work_unknown=False
    )
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=2, unreturned_work_unknown=False
    )
    assert [p["source_updates"] for p in r["checkpoints"]] == [0, 1, 2]
    assert [p["state"]["completed_updates"] for p in r["checkpoints"]] == [96, 97, 98]
    assert [p["source_rows"] for p in r["checkpoints"]] == [576, 582, 588]
    assert (
        r["new_budget"]["limits"]["max_training_updates"] == 2
        and r["new_budget"]["updates_completed"] == 2
    )
    assert len(r["applied_updates"]) == 2 and all(
        x["applied_at"] > x["observed_at"] for x in r["applied_updates"]
    )
    assert (
        r["source_seed_actor_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    )
    assert (
        r["final_examples"]
        == r["reserved_probe_updates"]
        == r["replay_examples"]
        == r["external_actions"]
        == 0
    )
    assert (
        r["stream"]["evaluation_scope"] == "training_only"
        and r["stream"]["score_forward_calls"] == 0
    )


def test_all_four_contexts_fixed_slots_teacher_and_summary(completed, monkeypatch):
    r = completed
    monkeypatch.setattr(study, "BATCHES_PER_PHASE", 1)
    expected = [(p, s, c) for p in (0, 1, 2) for s, c in study._slots(p)]
    assert [
        (x["source_updates"], x["episode"]["scenario"], x["concept"]) for x in r["episodes"]
    ] == expected
    assert len(r["episodes"]) == 20 and all(x["episode"]["complete"] for x in r["episodes"])
    assert r["summary"] == study._summary(r["episodes"])
    assert r["teacher_reads"] == dict(attempted=3, returned=3, rows_attempted=36)
    assert all(len(p["teacher_fit"]["rows"]) == 12 for p in r["checkpoints"])
    assert r["checkpoints"][0]["teacher_fit"]["rows"][6]["historical_train_exposures"] == 0
    assert r["checkpoints"][-1]["teacher_fit"]["rows"][0]["historical_train_exposures"] == 97
    assert json.loads(study._encode_report(r))["complete"]


def test_lost_second_update_keeps_native_and_local_prefix(monkeypatch):
    original = TorchFeatureLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("new-context return lost")
        return original(self, features, targets)

    monkeypatch.setattr(study, "BATCHES_PER_PHASE", 1)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    r = study.run_case(SOURCE)
    assert not r["complete"] and "return lost" in r["error"]
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=1, unreturned_work_unknown=True
    )
    assert r["new_budget"]["updates_completed"] == 1 and len(r["applied_updates"]) == 1
    assert [p["state"]["completed_updates"] for p in r["checkpoints"]] == [96, 97]
    assert r["prior_training_work"]["attempted_updates"] == 96


def test_wrong_native_offset_refused_before_more_readouts(monkeypatch):
    original = study.AttributionStream.run_until

    def corrupt(self, point):
        original(self, point)
        owner = self._runtime._candidate
        head, optimizer, updates, rows = owner._payload
        owner._payload = (head, optimizer, updates + 1, rows + 6)

    monkeypatch.setattr(study, "BATCHES_PER_PHASE", 1)
    monkeypatch.setattr(study.AttributionStream, "run_until", corrupt)
    r = study.run_case(SOURCE)
    assert not r["complete"] and "offset mismatch" in r["error"]
    assert r["training_work"]["returned_updates"] == 1 and len(r["checkpoints"]) == 1


def test_replaced_prior_source_refused_without_model(tmp_path, monkeypatch):
    source = tmp_path / "changed.json"
    source.write_bytes(SOURCE.read_bytes() + b" ")
    monkeypatch.setattr(
        study.native, "counted_learner", lambda *a: pytest.fail("changed source reached native")
    )
    r = study.run_case(source)
    assert not r["complete"] and "changed or replaced" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0 and r["episodes"] == []


def test_probe_teacher_refused_before_fit(monkeypatch):
    cases = study.training_cases()
    object.__setattr__(cases[0], "role", "probe_train")
    monkeypatch.setattr(study, "training_cases", lambda: cases)
    monkeypatch.setattr(
        study.native, "counted_learner", lambda *a: pytest.fail("probe reached fit")
    )
    r = study.run_case(SOURCE)
    assert not r["complete"] and "TRAIN teacher" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


def test_teacher_readout_failure_stays_paid_without_fit(monkeypatch):
    def fail(*args):
        raise RuntimeError("new teacher return lost")

    monkeypatch.setattr(study, "BATCHES_PER_PHASE", 1)
    monkeypatch.setattr(TorchFeatureLearner, "predict_logits", fail)
    r = study.run_case(SOURCE)
    assert not r["complete"] and "teacher return lost" in r["error"]
    assert r["teacher_reads"] == dict(attempted=1, returned=0, rows_attempted=12)
    assert r["training_work"]["attempted_updates"] == 0 and r["episodes"] == []


def test_occupied_output_refused_before_work(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_text("preserve")
    monkeypatch.setattr(study, "run_case", lambda *a: pytest.fail("occupied output reached work"))
    with pytest.raises(FileExistsError):
        study.main(["--input", str(SOURCE), "--output", str(path)])
    assert path.read_text() == "preserve"
