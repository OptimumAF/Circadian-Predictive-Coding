"""Actual literal TRAIN→memory→frozen typed serving, with paid failures."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_head_learner as native
from src.app.clm_memory_serving import memory_serving_requests
from src.app.clm_train_evidence_memory import CLMTrainEvidenceMemory
from src.core.decision_evidence import DecisionEvidenceStore
from src.core.experience import ExperiencePermissions
from test_clm_frozen_encoder import FakeTokenizer
from test_clm_typed_stream_readout import execute


def joint(torch, *, fail_head=False):
    run = study.run_case
    initialize = FakeTokenizer.__init__
    bind = CLMTrainEvidenceMemory.bind_fallback
    holder: dict[str, Any] = {}

    def tokenizer(self, mapping=None):
        initialize(self, mapping)
        self.mapping["unknown workflow please\n\n" + study.INSTRUCTIONS] = [13]

    def selected(*args, **kwargs):
        return run(*args, **kwargs, train_evidence_memory=True)

    def observed_bind(self, predictor):
        holder["owner"] = self
        holder["source_before"] = deepcopy(self._record)
        return bind(self, predictor)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(FakeTokenizer, "__init__", tokenizer)
        patch.setattr(study, "run_case", selected)
        patch.setattr(CLMTrainEvidenceMemory, "bind_fallback", observed_bind)
        result = execute(torch, fail_head=fail_head)
    result.update(holder)
    return result


@pytest.fixture(scope="module")
def completed():
    return joint(pytest.importorskip("torch"))


@pytest.fixture(scope="module")
def failed():
    return joint(pytest.importorskip("torch"), fail_head=True)


def test_should_complete_actual_twelve_updates_and_four_memory_choices_then_one_head_call(
    completed,
):
    report = completed["report"]
    assert report["complete"] is report["sequence_complete"] is True
    assert report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 36
    record = report["typed_development"]
    assert record["complete"] is True and record["mode"] == "evidence_fallback"
    assert len(record["predictions"]) == 5
    assert [r["selected_id"] for r in record["predictions"][:4]] == [
        "inspect",
        "defer",
        "defer",
        "inspect",
    ]
    view = record["memory_view_report"]
    assert [call["choice_source"] for call in view["calls"]] == ["memory"] * 4 + ["fallback"]
    assert view["work"] == dict(
        attempted=5, returned=5, validated=5, callback_attempted=1, callback_returned=1
    )
    assert record["requests"][-1]["query"] == "unknown workflow please"
    assert record["unscored_request_ids"] == ["typed-after-return-unseen"]
    assert record["memory_view_report"]["calls"][-1]["supplied"] == record["requests"][-1]
    assert report["cached_typed_decodes"] == 88
    assert all(
        "unknown" not in c["request"]["query"]
        for point in report["checkpoints"]
        for c in point["learned"]["cases"]
    )


def test_should_keep_one_original_encoder_budget_and_charge_only_the_actual_fallback(completed):
    case, record = completed, completed["report"]["typed_development"]
    encoder = case["handles"]["encoders"][0]
    assert len(case["handles"]["encoders"]) == 1 and encoder._limits[2] == 5
    assert [len(call["ids"]) for call in case["model"].calls] == [12, 4]
    assert record["encoder_work_before"]["forward_returns"] == 1
    assert record["encoder_work_after"]["forward_returns"] == 2
    assert record["encoder_work_after"]["sequences_returned"] == 16
    work = record["predictor_work"]
    assert work["request_attempts"] == work["request_returns"] == 1
    assert (
        work["encoder_returns"] == 1
        and work["head_forward_returns"] == 2
        and work["projected_rows"] == 4
    )
    assert work["unreturned_work_unknown"] is False
    assert record["binding_work"]["snapshot_returns"] == 1
    assert record["source_guard_work"]["capture_returns"] == 7
    assert (
        4 * record["source_guard_work"]["tensor_elements_copied"]
        == 7 * record["binding_work"]["snapshot_tensor_elements_copied"]
    )
    assert case["handles"]["learner"].work["snapshot_returns"] == 1
    assert case["handles"]["learner"].work["commits"] == 12


def test_should_preserve_completed_phase_memory_and_pay_for_one_separate_bounded_view(completed):
    case, record = completed, completed["report"]["typed_development"]
    source = case["report"]["train_evidence_memory"]
    assert source == case["source_before"]
    assert record["memory_source"] == json.loads(json.dumps(source))
    assert source["work"]["writes_completed"] == 36 and source["work"]["retrievals_completed"] == 16
    work = record["memory_binding_work"]
    assert work["binding_attempts"] == work["binding_returns"] == 1
    assert work["copy_write_attempts"] == work["copy_write_returns"] == 36
    assert work["view_write_work"]["retrieval_attempts"] == 0
    assert work["view_storage_bytes"] == work["source_storage_bytes"] == source["storage_bytes"]
    assert work["copy_return_unknown"] is False
    assert work["source_work"] == source["work"]
    assert work["max_calls"] == 5 and work["cutoff"] == 23 and work["policy_version"] == "v2"
    with pytest.raises(ValueError, match="unspent"):
        case["owner"].bind_fallback(lambda _: None)
    assert case["owner"].serving_work == work


def test_should_preserve_original_heads_caller_rng_and_paid_binding(completed):
    torch = pytest.importorskip("torch")
    state, action, scale = completed["original"]
    assert native._equal(completed["loaded"].state_head.state_dict(), state)
    assert native._equal(completed["loaded"].action_head.state_dict(), action)
    assert torch.equal(completed["loaded"].logit_scale, scale)
    assert torch.equal(torch.get_rng_state(), completed["rng"])
    assert (
        completed["report"]["control_unchanged"]
        is completed["report"]["caller_rng_unchanged"]
        is True
    )
    assert completed["report"]["typed_development"][
        "max_abs_cached_difference"
    ] == study._typed_difference(
        completed["report"], completed["report"]["typed_development"]["predictions"][:4]
    )


def test_should_keep_four_memory_returns_and_paid_unknown_failure_without_more_training(failed):
    report = failed["report"]
    record = report["typed_development"]
    assert report["complete"] is False and record["complete"] is False
    assert report["sequence_complete"] is True and report["stream"]["completed_updates"] == 12
    assert len(record["predictions"]) == 4
    assert record["memory_view_report"]["work"] == dict(
        attempted=5, returned=4, validated=4, callback_attempted=1, callback_returned=0
    )
    assert record["memory_view_report"]["calls"][-1]["error"]
    assert (
        record["memory_view_report"]["calls"][-1]["store_work_after"]["retrievals_completed"] == 5
    )
    assert record["predictor_work"]["unreturned_work_unknown"] is True
    assert record["memory_binding_work"]["copy_write_returns"] == 36
    assert report["train_evidence_memory"] == failed["source_before"]
    assert (
        report["learner_work"]["commits"] == 12 and report["learner_work"]["snapshot_returns"] == 1
    )
    assert record["encoder_work_after"]["forward_returns"] == 2
    with pytest.raises(ValueError):
        study.score_sequence(report)


@pytest.mark.parametrize(
    "fault",
    [
        "source-event",
        "source-work",
        "copy-count",
        "copy-bytes",
        "callback-count",
        "source-choice",
        "selection",
        "unscored",
        "unknown-recipe",
        "encoder-count",
        "guard-count",
        "prediction-count",
        "store-prefix",
        "unknown-work",
    ],
)
def test_should_refuse_corrupt_combined_evidence_before_cached_scoring(
    completed, monkeypatch, fault
):
    report = deepcopy(completed["report"])
    r = report["typed_development"]
    if fault == "source-event":
        report["train_evidence_memory"]["events"][0]["evidence"]["target_id"] = "inspect"
    elif fault == "source-work":
        report["train_evidence_memory"]["work"]["retrievals_completed"] = 21
    elif fault == "copy-count":
        r["memory_binding_work"]["copy_write_returns"] = 35
    elif fault == "copy-bytes":
        r["memory_binding_work"]["view_storage_bytes"] += 1
    elif fault == "callback-count":
        r["memory_view_report"]["work"]["callback_attempted"] = 0
    elif fault == "source-choice":
        r["memory_view_report"]["calls"][0]["choice_source"] = "fallback"
    elif fault == "selection":
        r["memory_view_report"]["calls"][0]["selection"]["items"][0]["evidence"][
            "policy_version"
        ] = "v1"
    elif fault == "unscored":
        r["unscored_request_ids"] = []
    elif fault == "unknown-recipe":
        r["memory_view_report"]["calls"][-1]["supplied"]["query"] += " evidence"
    elif fault == "encoder-count":
        r["encoder_work_after"]["forward_returns"] += 1
    elif fault == "guard-count":
        r["source_guard_work"]["capture_returns"] = 6
    elif fault == "prediction-count":
        r["predictions"].pop()
    elif fault == "store-prefix":
        r["memory_view_report"]["calls"][0]["store_work_after"]["writes_attempted"] += 1
    else:
        r["memory_view_report"]["calls"][-1]["selection"]["work"]["token_visits"] -= 1

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted memory result reached scoring")

    monkeypatch.setattr(study, "_score_cached", forbidden)
    with pytest.raises(ValueError):
        study.score_sequence(report)


@pytest.mark.parametrize("fault", ["replace", "corrupt"])
def test_should_refuse_replaced_or_corrupt_original_memory_before_more_serving(
    completed, monkeypatch, fault
):
    owner = completed["owner"]
    old = owner._record
    if fault == "replace":
        monkeypatch.setitem(completed["report"], "train_evidence_memory", deepcopy(old))
    else:
        monkeypatch.setitem(old, "storage_bytes", old["storage_bytes"] + 1)
    before = owner.serving_work
    with pytest.raises(ValueError, match="preceding"):
        owner.require_serving_source()
    assert owner.serving_work == before


def test_should_forward_both_existing_cli_options_without_loading(tmp_path, monkeypatch):
    options: list[dict[str, Any]] = []
    monkeypatch.setattr(study, "run_case", lambda *args, **kwargs: options.append(kwargs))
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
            str(tmp_path / "result.json"),
            "--catalog-training",
            "--train-evidence-memory",
            "--typed-development-readout",
        ],
    )
    study.main()
    assert options[0]["train_evidence_memory"] is options[0]["typed_development_readout"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(catalog_support="repeat"),
        dict(reserved_task_probes=True),
        dict(compare_catalog_support=True),
        dict(train_state_only=True),
        dict(catalog_training=False),
    ],
)
def test_should_keep_unsupported_joint_modes_refused_before_loader(monkeypatch, kwargs):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported mode loaded a model")

    monkeypatch.setattr(study, "load_local_clm", forbidden)
    options = dict(
        catalog_training=True, train_evidence_memory=True, typed_development_readout=True
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        study.run_case(Path("unused"), Path("unused"), {}, **options)


def test_should_declare_four_known_requests_and_one_unlabeled_control():
    requests = memory_serving_requests()
    assert [r.query for r in requests] == [
        "invoice please",
        "legacy please",
        "parcel please",
        "password please",
        "unknown workflow please",
    ]
    assert all(r.allowed_candidates == requests[0].allowed_candidates for r in requests)


def test_should_keep_a_lost_view_copy_paid_and_refuse_another_bind(completed, monkeypatch):
    source = completed["report"]
    report: dict[str, Any] = dict(
        training_objective=source["training_objective"],
        training_replay_consent=True,
        sequence_complete=False,
    )
    memory = CLMTrainEvidenceMemory(
        report, permissions=ExperiencePermissions(training=True, replay=True)
    )
    # Scalar boundary control reuses actual paid receipts; no learner is invoked.
    for count, phase in ((0, "baseline"), (4, "after_A"), (8, "after_B"), (12, "after_return")):
        report.update(
            rows=deepcopy(source["rows"][:count]),
            stream=dict(
                completed_updates=count,
                training_pairs=3 * count,
                receipts=deepcopy(source["stream"]["receipts"][:count]),
            ),
        )
        memory(phase, SimpleNamespace(completed_updates=count))
    report["sequence_complete"] = True
    memory.finish()
    before = deepcopy(report["train_evidence_memory"])
    original = DecisionEvidenceStore.add

    def lost(store, evidence):
        original(store, evidence)
        raise RuntimeError("lost copied-store write return")

    monkeypatch.setattr(DecisionEvidenceStore, "add", lost)
    with pytest.raises(RuntimeError, match="lost copied"):
        memory.bind_fallback(lambda _: None)
    work = memory.serving_work
    assert work["binding_attempts"] == work["copy_write_attempts"] == 1
    assert work["binding_returns"] == work["copy_write_returns"] == 0
    assert work["copy_return_unknown"] is True
    assert work["view_write_work"]["writes_completed"] == 1
    assert work["view_storage_bytes"] > 2
    with pytest.raises(ValueError, match="unspent"):
        memory.bind_fallback(lambda _: None)
    assert report["train_evidence_memory"] == before
