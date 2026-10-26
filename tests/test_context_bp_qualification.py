"""Two-update context fixtures; fixed96 scientific execution is separate."""

import json
from typing import Any

import pytest

from scripts import qualify_ticket_bp_fit as study
from src.adapters.torch_feature_learner import TorchFeatureLearner


@pytest.fixture(scope="module")
def completed():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "UPDATES", 2)
        r = study.run_qualification(policy_context=True)
    assert r["complete"], r.get("error")
    return r


def test_should_run_fixed_context_stream_and_preserve_all_native_clocks(completed):
    r = completed
    assert r["definition"] == "isolated_bp_context_joint_fit_v1"
    assert r["role"] == "disposable_context_train_fit_qualification"
    assert r["rows_per_update"] == 12 and r["policy_context_in_query"]
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=24, returned_updates=2, unreturned_work_unknown=False
    )
    assert [(p["source_updates"], p["source_rows"]) for p in r["checkpoints"]] == [(0, 0), (2, 24)]
    assert [p["state"]["policy"]["feature_dim"] for p in r["checkpoints"]] == [9, 9]
    assert r["teacher_reads"] == dict(attempted=2, returned=2, rows_attempted=24)
    assert r["training_data"]["policy_versions"] == ["v1", "v1", "v2", "v2"]
    assert r["training_data"]["concepts"] == ["invoice", "password", "invoice", "parcel"]
    assert (
        r["stream"]["evaluation_scope"] == "training_only"
        and r["stream"]["score_forward_calls"] == 0
    )
    assert all(a["applied_at"] > a["observed_at"] for a in r["applied_updates"])
    assert all(
        r[k]
        for k in (
            "source_seed_unchanged",
            "deployed_actor_unchanged",
            "numpy_rng_unchanged",
            "torch_rng_unchanged",
        )
    )
    assert r["costs"]["attempted_training_nominal_affine_terms"] == 24 * 45


def test_should_keep_complete_phase_bound_task_and_teacher_evidence(completed):
    r = completed
    expected = [
        (p, v, c, s)
        for p in (0, 2)
        for v, c in study.CONTEXT_TASKS
        for s in ("ordinary", "stale_once", "revoke_commit")
    ]
    assert [
        (e["source_updates"], e["policy_version"], e["concept"], e["episode"]["scenario"])
        for e in r["episodes"]
    ] == expected
    assert all(
        e["episode"]["complete"] and e["episode"]["policy_context_in_query"] for e in r["episodes"]
    )
    assert all(
        e["episode"]["initial_state"]["policy_version"] == e["policy_version"]
        for e in r["episodes"]
    )
    for point in r["checkpoints"]:
        rows = point["teacher_fit"]["rows"]
        assert len(rows) == 12
        assert [(x["policy_version"], x["concept"]) for x in rows] == [
            (v, c) for v, c in study.CONTEXT_TASKS for _ in range(3)
        ]
    assert all(
        r[k] == 0
        for k in ("final_examples", "reserved_probe_updates", "replay_examples", "external_actions")
    )
    assert json.loads(study._encode_report(r))["complete"]
    # Fixture correctness does not require fitted task success or margin sign.


@pytest.mark.parametrize("fault", ["role", "group", "schema", "alignment"])
def test_should_refuse_incompatible_context_teacher_before_native_construction(monkeypatch, fault):
    original = study.training_cases
    if fault in ("role", "group"):

        def corrupt(**kwargs):
            rows = original(**kwargs)
            object.__setattr__(
                rows[0], "role" if fault == "role" else "source_group", "probe_train"
            )
            return rows

        monkeypatch.setattr(study, "training_cases", corrupt)
    elif fault == "alignment":
        from scripts import run_ticket_workflow as teacher_module

        encode = teacher_module.encode_decision

        def corrupt_public(request, spec):
            values, count = encode(request, spec)
            if request.query.endswith("please"):
                values[0, 0] = 1 - values[0, 0]
            return values, count

        monkeypatch.setattr(teacher_module, "encode_decision", corrupt_public)
    else:
        with pytest.raises(ValueError, match="D9 schema"):
            study._context_teacher(study.DecisionFeatureSpec(study.FEATURES, study.CATALOG))
        return
    monkeypatch.setattr(
        study.native, "counted_learner", lambda *a: pytest.fail("bad teacher reached native")
    )
    r = study.run_qualification(policy_context=True)
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert "TRAIN" in r["error"]


def test_should_preserve_paid_context_failure_and_stop_future_reads(monkeypatch):
    original = TorchFeatureLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("context second return lost")
        return original(self, features, targets)

    monkeypatch.setattr(study, "UPDATES", 2)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    r = study.run_qualification(policy_context=True)
    assert not r["complete"] and "return lost" in r["error"]
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=24, returned_updates=1, unreturned_work_unknown=True
    )
    assert r["budget"]["updates_completed"] == len(r["applied_updates"]) == 1
    assert len(r["checkpoints"]) == 1 and len(r["episodes"]) == 12


def test_should_refuse_replaced_paid_initial_evidence_before_more_native_work(monkeypatch):
    checkpoint = study.native.checkpoint
    counted = study.native.counted_learner
    holder: dict[int, Any] = {}

    def capture(point, state):
        record = checkpoint(point, state)
        holder[point] = record
        return record

    def construct(method, d, k, before, after):
        def corrupt_after():
            after()
            holder[0]["source_rows"] = 1

        return counted(method, d, k, before, corrupt_after)

    monkeypatch.setattr(study, "UPDATES", 2)
    monkeypatch.setattr(study.native, "checkpoint", capture)
    monkeypatch.setattr(study.native, "counted_learner", construct)
    r = study.run_qualification(policy_context=True)
    assert not r["complete"] and "preceding contextual paid evidence" in r["error"]
    assert r["training_work"]["returned_updates"] == 1
    assert r["checkpoints"][0]["source_rows"] == 0
    assert len(r["episodes"]) == 12


@pytest.mark.parametrize("bad", [None, 0, 1, "v1"])
def test_should_refuse_invalid_mode_before_native_runtime(monkeypatch, bad):
    monkeypatch.setattr(
        study.native, "require_torch", lambda: pytest.fail("invalid mode reached runtime")
    )
    with pytest.raises(ValueError, match="exact bool"):
        study.run_qualification(policy_context=bad)


@pytest.mark.parametrize("context", [False, True])
def test_should_dispatch_cli_mode_and_protect_output_before_work(tmp_path, monkeypatch, context):
    calls: list[dict[str, Any]] = []

    def run(**kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_qualification", run)
    output = tmp_path / "context.json"
    args = (["--policy-context"] if context else []) + ["--output", str(output)]
    assert study.main(args) == 0
    assert calls == [dict(policy_context=True) if context else {}]
    with pytest.raises(FileExistsError):
        study.main(args)
    assert len(calls) == 1 and json.loads(output.read_bytes())["complete"]
