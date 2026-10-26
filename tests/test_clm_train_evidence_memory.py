"""One matched literal pair; scalar refusal cases reuse its paid report."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import scripts.run_clm_temporal_sequence as study
from src.app.clm_train_evidence_memory import CLMTrainEvidenceMemory
from src.app.experience_inbox import ExperienceInbox
from src.core.experience import ExperiencePermissions
from test_clm_typed_stream_readout import execute


@pytest.fixture(scope="module")
def pair():
    torch = pytest.importorskip("torch")
    control = execute(torch, typed=False)
    actual = study.run_case
    observed = []
    record = ExperienceInbox.record_experience

    def consent(inbox, experience):
        observed.append(asdict(experience.permissions))
        return record(inbox, experience)

    def memory(*args, **kwargs):
        return actual(*args, **kwargs, train_evidence_memory=True)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "run_case", memory)
        patch.setattr(ExperienceInbox, "record_experience", consent)
        treatment = execute(torch, typed=False)
    return torch, control, treatment, observed


def test_should_preserve_actual_head_updates_weights_encoder_and_caller_rng(pair):
    torch, control, memory, observed = pair
    for key in (
        "rows",
        "stream",
        "raw_readouts",
        "frozen_A_probabilities",
        "learner_work",
        "encoder_work",
        "checkpoints",
        "contrasts",
    ):
        assert memory["report"][key] == control["report"][key], key
    for old, new in zip(
        control["handles"]["learner"]._bundle[:2],
        memory["handles"]["learner"]._bundle[:2],
        strict=True,
    ):
        assert all(
            torch.equal(a, b) for a, b in zip(old.parameters(), new.parameters(), strict=True)
        )
    assert torch.equal(control["rng"], memory["rng"])
    assert memory["report"]["caller_rng_unchanged"] is True
    assert "train_evidence_memory" not in control["report"]
    assert observed == [dict(training=True, replay=True, evaluation=False)] * 12
    assert memory["report"]["training_replay_consent"] is True
    assert control["report"].get("training_replay_consent", False) is False


def test_should_adapt_and_reuse_returned_context_with_explicit_unavailability(pair):
    record = pair[2]["report"]["train_evidence_memory"]
    assert record["complete"] is True
    assert [row["answered_count"] for row in record["scores"]] == [0, 3, 3, 4]
    assert [row["correct_count"] for row in record["scores"]] == [0, 3, 3, 4]
    assert [row["coverage"] for row in record["scores"]] == [0, 0.75, 0.75, 1]
    assert [row["accuracy_answered"] for row in record["scores"]] == [None, 1, 1, 1]
    baseline, a, b, returned = record["phases"]
    assert all(call["result"] is None for call in baseline["calls"])
    assert a["calls"][2]["result"] is None  # parcel has never arrived
    assert b["calls"][3]["result"] is None  # no v2 password label yet
    invoice = returned["calls"][0]
    assert invoice["result"]["selected_id"] == "inspect"
    parcel = returned["calls"][2]["selection"]["items"][0]["evidence"]
    assert parcel["request"]["query"] == "parcel" and parcel["target_id"] == "defer"
    assert parcel["policy_version"] == "v2" and parcel["recorded_at"] == 15
    assert returned["cutoff"] == 23


def test_should_pay_bounded_memory_and_record_phase_delays_without_backdating(pair):
    record = pair[2]["report"]["train_evidence_memory"]
    assert len(record["events"]) == 36
    assert 0 < record["storage_bytes"] <= 65536
    work = record["work"]
    assert work["writes_attempted"] == work["writes_completed"] == 36
    assert work["retrieval_attempts"] == work["retrievals_completed"] == 16
    assert work["metadata_scans"] == 288  # 4 * (0 + 12 + 24 + 36)
    assert 0 < work["rendered_bytes"] <= 16 * 768
    for index, event in enumerate(record["events"]):
        evidence = event["evidence"]
        batch = index // 3
        assert evidence["observed_at"] == 2 * batch
        assert evidence["available_at"] == 2 * batch + 1
        assert evidence["recorded_at"] == (7, 15, 23)[batch // 4]
        assert evidence["role"] == "train" and evidence["source_group"] == "train-token"
        assert evidence["permissions"] == dict(training=True, replay=True, evaluation=False)
        assert "please" not in evidence["request"]["query"]
        assert event["revision"] == 0
    for update, predictions, scored in pair[2]["observations"]:
        assert not predictions
        if scored:
            assert update == 12


def prefix(pair, count):
    source = pair[2]["report"]
    report = dict(
        rows=deepcopy(source["rows"][:count]),
        stream=dict(
            completed_updates=count,
            training_pairs=3 * count,
            receipts=deepcopy(source["stream"]["receipts"][:count]),
        ),
        training_objective=source["training_objective"],
        training_replay_consent=True,
        sequence_complete=False,
    )
    return report


def fresh(pair):
    report = prefix(pair, 0)
    hook = CLMTrainEvidenceMemory(
        report, permissions=ExperiencePermissions(training=True, replay=True)
    )
    hook("baseline", SimpleNamespace(completed_updates=0))
    report.update(prefix(pair, 4))
    return hook, report


@pytest.mark.parametrize(
    "field,value",
    [("arrived_at", 0), ("applied_at", 0), ("update_number", True), ("sample_id", "replacement")],
)
def test_should_refuse_corrupt_actual_receipts_before_releasing_more_labels(pair, field, value):
    hook, report = fresh(pair)
    report["stream"]["receipts"][0][field] = value
    with pytest.raises(ValueError, match="receipt"):
        hook("after_A", SimpleNamespace(completed_updates=4))
    assert report["train_evidence_memory"]["work"]["writes_attempted"] == 0


@pytest.mark.parametrize("change", ["replace", "alter", "store", "consent", "label"])
def test_should_refuse_replaced_or_corrupted_evidence_consent_and_train_labels(pair, change):
    hook, report = fresh(pair)
    if change == "replace":
        report["train_evidence_memory"] = deepcopy(report["train_evidence_memory"])
    elif change == "alter":
        report["train_evidence_memory"]["phases"][0]["cutoff"] = 1
    elif change == "store":
        hook._store._work["retrieval_attempts"] += 1
    elif change == "consent":
        report["training_replay_consent"] = False
    else:
        report["rows"][0]["candidate_indices"][0] = 0
    with pytest.raises(ValueError):
        hook("after_A", SimpleNamespace(completed_updates=4))
    assert hook._store.work["writes_attempted"] == 0


def test_should_keep_paid_store_failure_and_refuse_retry(pair, monkeypatch):
    hook, report = fresh(pair)
    original = hook._store.add

    def fail_after_write(evidence):
        original(evidence)
        raise RuntimeError("lost memory write return")

    monkeypatch.setattr(hook._store, "add", fail_after_write)
    with pytest.raises(RuntimeError, match="lost memory"):
        hook("after_A", SimpleNamespace(completed_updates=4))
    record = report["train_evidence_memory"]
    assert record["complete"] is False
    assert record["work"]["writes_attempted"] == record["work"]["writes_completed"] == 1
    assert len(record["events"]) == 1
    with pytest.raises(ValueError, match="stopped"):
        hook("after_A", SimpleNamespace(completed_updates=4))


def test_should_require_complete_train_before_scoring_and_never_renew_completed_work(pair):
    hook, report = fresh(pair)
    with pytest.raises(ValueError):
        hook.finish()
    for count, phase in ((4, "after_A"), (8, "after_B"), (12, "after_return")):
        report.update(prefix(pair, count))
        hook(phase, SimpleNamespace(completed_updates=count))
    assert "scores" not in report["train_evidence_memory"]
    report["sequence_complete"] = True
    hook.finish()
    with pytest.raises(ValueError, match="already completed"):
        hook.finish()
    with pytest.raises(ValueError, match="allowance"):
        hook("after_return", SimpleNamespace(completed_updates=12))


@pytest.mark.parametrize(
    "permissions",
    [
        ExperiencePermissions(),
        ExperiencePermissions(training=True),
        ExperiencePermissions(training=True, replay=True, evaluation=True),
    ],
)
def test_should_require_explicit_raw_reuse_consent_at_birth(permissions):
    with pytest.raises(ValueError, match="consent"):
        CLMTrainEvidenceMemory({}, permissions=permissions)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(catalog_training=False),
        dict(catalog_support="repeat"),
        dict(reserved_task_probes=True),
        dict(compare_catalog_support=True),
        dict(train_evidence_memory=1),
    ],
)
def test_should_refuse_unsupported_modes_before_loading(monkeypatch, kwargs):
    def forbidden(*args, **kwargs):
        raise AssertionError("model loader must not run")

    monkeypatch.setattr(study, "load_local_clm", forbidden)
    options = dict(catalog_training=True, train_evidence_memory=True)
    options.update(kwargs)
    with pytest.raises(ValueError, match="TRAIN evidence memory"):
        study.run_case(Path("unused"), Path("unused"), {}, **options)


def test_should_forward_the_cli_option_through_existing_runner(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    received = []

    def run(*args, **kwargs):
        received.append(kwargs)

    monkeypatch.setattr(study, "run_case", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--model-dir",
            "unused",
            "--head",
            "unused",
            "--output",
            str(output),
            "--catalog-training",
            "--train-evidence-memory",
        ],
    )
    study.main()
    assert received[0]["catalog_training"] is received[0]["train_evidence_memory"] is True
