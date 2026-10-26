"""Tiny fresh native training, original peer admission and saved-view integration."""

from copy import deepcopy
import json

import pytest

from scripts import run_native_ticket_controls as study
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.app.key_door_inspector import project_key_door_report
from test_taskspace_memory_inspector import assert_sources, at, section


def deny_history(*args, **kwargs):
    pytest.fail("fresh learner requested historical payload")


@pytest.fixture(scope="module")
def completed():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "WARMUP", 1)
        patch.setattr(study.context, "BATCHES_PER_PHASE", 1)
        patch.setattr(study.context, "_source", deny_history)
        patch.setattr(study.context, "bp_snapshot_from_record", deny_history)
        result = study.run_case("bp", fresh_bp=True)
    assert result["complete"], result.get("error")
    return result


def test_fresh_learner_owns_all_local_work_without_saved_authority(completed):
    r = completed
    assert r["definition"] == "fresh_bp_ticket_demo_v1"
    assert r["method"] == "bp" and r["role"] == "fresh_train_demo"
    assert "prior_training_work" not in r and "prior_costs" not in r
    assert r["training_work"] == dict(
        attempted_updates=3, attempted_rows=18, returned_updates=3, unreturned_work_unknown=False
    )
    assert r["budget"]["updates_completed"] == r["budget"]["limits"]["max_training_updates"] == 3
    assert [c["state"]["completed_updates"] for c in r["checkpoints"]] == [1, 2, 3]
    assert [c["source_rows"] for c in r["checkpoints"]] == [6, 12, 18]
    assert [c["source_updates"] for c in r["checkpoints"]] == [0, 1, 2]
    assert all(c["state"]["policy"]["method"] == "bp" for c in r["checkpoints"])
    assert len(r["applied_updates"]) == 3 and all(
        a["observed_at"] < a["applied_at"] for a in r["applied_updates"]
    )
    assert (
        r["source_seed_actor_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    )
    assert (
        r["stream"]["score_forward_calls"] == 0
        and r["stream"]["evaluation_scope"] == "training_only"
    )
    assert (
        r["final_examples"]
        == r["reserved_probe_updates"]
        == r["replay_examples"]
        == r["external_actions"]
        == 0
    )
    assert len(r["episodes"]) == 20 and r["teacher_reads"] == dict(
        attempted=3, returned=3, rows_attempted=36
    )


def test_fresh_report_has_honest_inspector_ownership_and_source_links(completed):
    # The file boundary serializes native optimizer integer IDs as JSON keys.
    saved = json.loads(study._encode_report(completed))
    view = project_key_door_report(saved, source_label="fresh.json")
    assert view["schema_id"] == "ticket_workflow_inspector_v1"
    assert at(view, "/role")["value"] == "fresh_train_demo"
    assert at(view, "/training_work")["value"]["attempted_updates"] == 3
    assert "fresh" in section(view, "work")["note"].lower()
    assert "restores qualified history" not in section(view, "work")["note"]
    assert len(section(view, "episodes")["rows"]) == 20
    assert_sources(view)


@pytest.mark.parametrize("fault", ["fresh-method", "fresh-role", "peer-method"])
def test_false_fresh_or_peer_labels_are_refused(completed, fault):
    record = deepcopy(json.loads(study._encode_report(completed)))
    if fault == "fresh-method":
        record["method"] = "pc"
    elif fault == "fresh-role":
        record.pop("role")
    else:
        record["definition"] = "matched_native_ticket_controls_v1"
    with pytest.raises(ValueError):
        project_key_door_report(record, source_label="forged.json")


@pytest.mark.parametrize("method,flag", [("pc", True), ("bp", "yes"), ("bp", False)])
def test_fresh_admission_is_explicit_before_any_native_work(method, flag, monkeypatch):
    monkeypatch.setattr(study.native, "counted_learner", deny_history)
    with pytest.raises(ValueError):
        study.run_case(method, fresh_bp=flag)


def test_nontrain_teacher_still_refused_in_fresh_mode(monkeypatch):
    cases = study.context.training_cases()
    object.__setattr__(cases[0], "role", "probe_train")
    monkeypatch.setattr(study.context, "training_cases", lambda: cases)
    monkeypatch.setattr(study.native, "counted_learner", deny_history)
    r = study.run_case("bp", fresh_bp=True)
    assert not r["complete"] and "TRAIN teacher" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


def test_fresh_lost_second_update_keeps_paid_prefix(monkeypatch):
    original = TorchFeatureLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fresh update return lost")
        return original(self, features, targets)

    monkeypatch.setattr(study, "WARMUP", 1)
    monkeypatch.setattr(study.context, "BATCHES_PER_PHASE", 1)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    r = study.run_case("bp", fresh_bp=True)
    assert not r["complete"] and "return lost" in r["error"]
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=1, unreturned_work_unknown=True
    )
    assert r["budget"]["updates_completed"] == 1 and len(r["applied_updates"]) == 1
    assert [c["state"]["completed_updates"] for c in r["checkpoints"]] == [1]


def test_unobserved_fresh_clock_change_is_refused_before_readout(monkeypatch):
    original = study.AttributionStream.run_until

    def corrupt(self, point):
        original(self, point)
        owner = self._runtime._candidate
        head, optimizer, updates, rows = owner._payload
        owner._payload = (head, optimizer, updates + 1, rows + 6)

    monkeypatch.setattr(study, "WARMUP", 1)
    monkeypatch.setattr(study.context, "BATCHES_PER_PHASE", 1)
    monkeypatch.setattr(study.AttributionStream, "run_until", corrupt)
    r = study.run_case("bp", fresh_bp=True)
    assert not r["complete"] and "offset mismatch" in r["error"]
    assert r["training_work"]["returned_updates"] == 1 and not r["checkpoints"]


def test_public_fresh_choice_is_explicit_and_output_stays_exclusive(tmp_path, monkeypatch):
    calls: list[tuple[str, bool]] = []

    def run(method, *, fresh_bp=False):
        calls.append((method, fresh_bp))
        return dict(complete=True, method=method)

    monkeypatch.setattr(study, "run_case", run)
    output = tmp_path / "fresh.json"
    assert study.main(["--fresh-bp", "--output", str(output)]) == 0
    assert calls == [("bp", True)] and json.loads(output.read_bytes())["method"] == "bp"
    with pytest.raises(FileExistsError):
        study.main(["--fresh-bp", "--output", str(output)])
    ambiguous = tmp_path / "ambiguous.json"
    with pytest.raises(SystemExit):
        study.main(["--fresh-bp", "--method", "pc", "--output", str(ambiguous)])
    assert not ambiguous.exists() and len(calls) == 1
    assert study.main(["--method", "pc", "--output", str(tmp_path / "peer.json")]) == 0
    assert calls[-1] == ("pc", False)
