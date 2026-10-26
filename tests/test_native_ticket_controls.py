"""Tiny matched streams; no full warmup or baseline refit in fixtures."""

import json
from pathlib import Path

import pytest

from scripts import run_native_ticket_controls as study
from src.adapters.torch_feature_learner import TorchFeatureLearner


@pytest.fixture(scope="module")
def completed():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "WARMUP", 1)
        patch.setattr(study.context, "BATCHES_PER_PHASE", 1)
        results = {m: study.run_case(m) for m in study.METHODS}
    for result in results.values():
        assert result["complete"], result.get("error")
    return results


@pytest.mark.parametrize("method", study.METHODS)
def test_live_stream_has_exact_exposure_clocks_and_isolated_readouts(completed, method):
    r = completed[method]
    assert r["training_work"] == dict(
        attempted_updates=3, attempted_rows=18, returned_updates=3, unreturned_work_unknown=False
    )
    assert [p["source_updates"] for p in r["checkpoints"]] == [0, 1, 2]
    assert [p["state"]["completed_updates"] for p in r["checkpoints"]] == [1, 2, 3]
    assert [p["source_rows"] for p in r["checkpoints"]] == [6, 12, 18]
    assert r["budget"]["updates_completed"] == r["budget"]["limits"]["max_training_updates"] == 3
    assert len(r["applied_updates"]) == 3
    assert all(a["applied_at"] > a["observed_at"] for a in r["applied_updates"])
    assert (
        r["source_seed_actor_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    )
    assert r["stream"]["score_forward_calls"] == 0
    assert r["stream"]["evaluation_scope"] == "training_only"
    assert r["final_examples"] == r["reserved_probe_updates"] == r["replay_examples"] == 0
    assert r["teacher_reads"] == dict(attempted=3, returned=3, rows_attempted=36)
    assert len(r["episodes"]) == 20 and all(e["episode"]["complete"] for e in r["episodes"])
    assert [
        p["teacher_fit"]["rows"][0]["historical_train_exposures"] for p in r["checkpoints"]
    ] == [1, 1, 2]
    assert [
        p["teacher_fit"]["rows"][6]["historical_train_exposures"] for p in r["checkpoints"]
    ] == [0, 1, 1]
    assert json.loads(study._encode_report(r))["complete"]


def test_pc_neutral_parameters_diagnostics_teacher_and_tasks_match(completed):
    pc, neutral = completed["pc"], completed["neutral_cpc"]
    assert (
        pc["initial_parameters"]
        == neutral["initial_parameters"]
        == completed["gated_cpc"]["initial_parameters"]
    )
    assert pc["training_data"] == neutral["training_data"]
    assert pc["applied_updates"] == neutral["applied_updates"]
    source = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"
        ).read_bytes()
    )
    assert pc["initial_parameters"] == study.parameters(source["checkpoints"][0]["state"])
    assert pc["episodes"] == neutral["episodes"] and pc["summary"] == neutral["summary"]
    for a, b in zip(pc["checkpoints"], neutral["checkpoints"], strict=True):
        assert study.parameters(a["state"]) == study.parameters(b["state"])
        assert a["teacher_fit"] == b["teacher_fit"] and a["teacher_logits"] == b["teacher_logits"]


def test_original_training_contract_and_heldout_refusal(monkeypatch):
    spec = study.DecisionFeatureSpec(study.FEATURES, study.CATALOG)
    _, _, batches, data = study.training_sequence(spec)
    assert len(batches) == 112 and len({row for b in batches for row in b.row_ids}) == 672
    assert [b.context for b in batches] == ["joint-train"] * 96 + ["new-contexts"] * 8 + [
        "old-return"
    ] * 8
    assert all(b.features.shape == (6, 7) and b.targets.shape == (6, 1) for b in batches)
    assert data["warm"]["features"] == data["return"]["features"]
    assert data["warm"]["targets"] == data["return"]["targets"]
    source = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"
        ).read_bytes()
    )
    assert data["warm"]["features"] == source["training_data"]["features"]
    assert data["warm"]["targets"] == source["training_data"]["targets"]
    cases = study.context.training_cases()
    # Simulate corruption after the reference case's own admission validation.
    object.__setattr__(cases[0], "role", "probe_train")
    monkeypatch.setattr(study.context, "training_cases", lambda: cases)
    monkeypatch.setattr(
        study.native, "counted_learner", lambda *a: pytest.fail("probe reached fit")
    )
    r = study.run_case("pc")
    assert not r["complete"] and "TRAIN teacher" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


def test_second_update_failure_keeps_paid_prefix(monkeypatch):
    original = TorchFeatureLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("peer update return lost")
        return original(self, features, targets)

    monkeypatch.setattr(study, "WARMUP", 1)
    monkeypatch.setattr(study.context, "BATCHES_PER_PHASE", 1)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    r = study.run_case("pc")
    assert not r["complete"] and "return lost" in r["error"]
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=1, unreturned_work_unknown=True
    )
    assert r["budget"]["updates_completed"] == 1 and len(r["applied_updates"]) == 1
    assert [p["state"]["completed_updates"] for p in r["checkpoints"]] == [1]


def test_corrupt_native_clock_refused_before_readouts(monkeypatch):
    original = study.AttributionStream.run_until

    def corrupt(self, point):
        original(self, point)
        owner = self._runtime._candidate
        head, optimizer, updates, rows = owner._payload
        owner._payload = head, optimizer, updates + 1, rows + 6

    monkeypatch.setattr(study, "WARMUP", 1)
    monkeypatch.setattr(study.context, "BATCHES_PER_PHASE", 1)
    monkeypatch.setattr(study.AttributionStream, "run_until", corrupt)
    r = study.run_case("pc")
    assert not r["complete"] and (
        "offset mismatch" in r["error"] or "traffic is incompatible with update clock" in r["error"]
    )
    assert r["training_work"]["returned_updates"] == 1 and not r["checkpoints"]


def test_invalid_method_and_occupied_output_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study.native, "counted_learner", lambda *a: pytest.fail("invalid reached native")
    )
    with pytest.raises(ValueError, match="peer method"):
        study.run_case("bp")
    path = tmp_path / "existing.json"
    path.write_text("preserve")
    monkeypatch.setattr(study, "run_case", lambda *a: pytest.fail("occupied reached native"))
    with pytest.raises(FileExistsError):
        study.main(["--method", "pc", "--output", str(path)])
    assert path.read_text() == "preserve"
