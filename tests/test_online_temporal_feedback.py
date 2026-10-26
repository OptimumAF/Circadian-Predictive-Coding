"""Causal typed feedback and one shared fixed temporal reference fixture.

Scalar helper cases do not fit a model. The single integration fixture is the
prospectively declared twelve-update/36-row reference, executed only by root.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from src.app.decision_evidence import apply_decision_feedback, decision_from_evidence
from src.core.decision_evidence import DecisionEvidenceStore
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.temporal_decision_sequence import temporal_examples
from src.core.typed_decision import DecisionCandidate, DecisionRequest


CATALOG = (
    DecisionCandidate("inspect", "inspect the record"),
    DecisionCandidate("retire", "retire the record"),
    DecisionCandidate("route", "route the record"),
)


def feedback_pair():
    request = DecisionRequest("sample-1", "invoice", deepcopy(CATALOG))
    source = Experience(
        "sample-1",
        "episode-1",
        0,
        "model-0",
        request,
        "train",
        ExperiencePermissions(training=True, replay=True),
        candidate_ids=tuple(candidate.candidate_id for candidate in request.allowed_candidates),
    )
    label = LabelArrival("label-1", "sample-1", "episode-1", 1, "model-0", "route")
    return source, label


def apply(store, source, label, *, recorded_at=1):
    return apply_decision_feedback(
        store,
        source,
        label,
        policy_version="v1",
        source_group="train-token",
        recorded_at=recorded_at,
    )


def test_should_publish_a_bound_label_only_at_its_recorded_causal_time():
    source, label = feedback_pair()
    store = DecisionEvidenceStore()
    revision = apply(store, source, label)
    record = revision.evidence
    assert revision.revision == 0 and record.request == source.features
    assert record.observed_at == 0 and record.available_at == record.recorded_at == 1
    assert record.target_id == label.targets and record.role == "train"
    assert record.permissions == source.permissions and record.policy_version == "v1"
    assert record.source_group == "train-token"
    assert store.retrieve(source.features, cutoff=0, policy_version="v1").items == ()
    selected = store.retrieve(source.features, cutoff=1, policy_version="v1")
    result = decision_from_evidence(source.features, selected)
    assert result is not None and result.selected_id == "route"
    assert dict(result.probabilities) == {"inspect": 0.0, "retire": 0.0, "route": 1.0}


def test_should_own_feedback_after_public_sources_labels_and_returned_records_change():
    source, label = feedback_pair()
    before_source, before_label = deepcopy(source), deepcopy(label)
    store = DecisionEvidenceStore()
    revision = apply(store, source, label)
    assert source == before_source and label == before_label
    before = store.events
    object.__setattr__(
        source.features.allowed_candidates[0], "description", "changed caller description"
    )
    object.__setattr__(label, "targets", "inspect")
    object.__setattr__(revision.evidence, "target_id", "retire")
    assert store.events == before


@pytest.mark.parametrize(
    "fault",
    ["probe-source", "heldout-source", "probe-label", "final-label", "no-training", "no-replay"],
)
def test_should_require_train_roles_and_original_explicit_reuse_consent(fault):
    source, label = feedback_pair()
    if fault == "probe-source":
        object.__setattr__(source, "role", "probe_train")
    elif fault == "heldout-source":
        object.__setattr__(source, "role", "outer_selection")
    elif fault == "probe-label":
        object.__setattr__(label, "role", "probe_train")
    elif fault == "final-label":
        object.__setattr__(label, "role", "final_test")
    elif fault == "no-training":
        source = replace(source, permissions=ExperiencePermissions(replay=True))
    else:
        source = replace(source, permissions=ExperiencePermissions(training=True))
    store = DecisionEvidenceStore()
    with pytest.raises(ValueError):
        apply(store, source, label)
    assert store.events == ()


@pytest.mark.parametrize(
    "fault",
    [
        "sample-key",
        "episode-key",
        "model-version",
        "request-id",
        "candidate-order",
        "candidate-missing",
        "target",
        "boolean-observation",
        "early-label",
        "early-record",
    ],
)
def test_should_refuse_mismatched_feedback_keys_schema_or_causal_times_before_writing(fault):
    source, label = feedback_pair()
    recorded = 1
    if fault == "sample-key":
        label = replace(label, sample_id="foreign")
    elif fault == "episode-key":
        label = replace(label, episode_id="foreign")
    elif fault == "model-version":
        label = replace(label, model_version="foreign")
    elif fault == "request-id":
        source = replace(source, features=replace(source.features, request_id="foreign"))
    elif fault == "candidate-order":
        source = replace(source, candidate_ids=tuple(reversed(source.candidate_ids)))
    elif fault == "candidate-missing":
        source = replace(source, candidate_ids=())
    elif fault == "target":
        label = replace(label, targets="network")
    elif fault == "boolean-observation":
        object.__setattr__(source, "observed_at", True)
    elif fault == "early-label":
        source = replace(source, observed_at=2)
    else:
        recorded = 0
    store = DecisionEvidenceStore()
    with pytest.raises(ValueError):
        apply(store, source, label, recorded_at=recorded)
    assert store.events == ()


@pytest.mark.parametrize("record", ["source", "label"])
def test_should_refuse_extra_fields_on_bypassed_frozen_feedback_records(record):
    source, label = feedback_pair()
    object.__setattr__(source if record == "source" else label, "unadmitted", True)
    store = DecisionEvidenceStore()
    with pytest.raises(ValueError):
        apply(store, source, label)
    assert store.events == ()


def test_should_refuse_republishing_the_same_source_without_relabeling_old_history():
    source, label = feedback_pair()
    store = DecisionEvidenceStore()
    apply(store, source, label)
    before = store.events
    with pytest.raises(ValueError):
        apply(
            store,
            source,
            replace(label, event_id="different-label", targets="inspect"),
            recorded_at=2,
        )
    assert store.events == before and len(store.events) == 1


def test_should_distinguish_known_no_hit_from_probabilities_and_allow_later_available_feedback():
    source, label = feedback_pair()
    store = DecisionEvidenceStore()
    no_hit = store.retrieve(source.features, cutoff=0, policy_version="v1")
    assert decision_from_evidence(source.features, no_hit) is None
    apply(store, source, label)
    result = decision_from_evidence(
        source.features, store.retrieve(source.features, cutoff=1, policy_version="v1")
    )
    assert result is not None and result.selected_id == "route"
    assert (
        result.reported_confidence
        is result.reported_input_tokens
        is result.reported_output_tokens
        is None
    )


def test_should_keep_old_policy_memory_unavailable_instead_of_reusing_an_obsolete_target():
    source, label = feedback_pair()
    store = DecisionEvidenceStore()
    apply(store, source, label)
    current = temporal_examples("after_B", "outer_selection")[0].request
    selection = store.retrieve(current, cutoff=1, policy_version="v2")
    assert selection.items == () and decision_from_evidence(current, selection) is None


@pytest.mark.parametrize(
    "fault",
    [
        "mutable-items",
        "boolean-revision",
        "boolean-work",
        "extra-field",
        "foreign-target",
        "inactive-target",
        "utf8-overflow",
    ],
)
def test_should_revalidate_selection_and_current_allowed_ids_without_changing_store_history(fault):
    source, label = feedback_pair()
    store = DecisionEvidenceStore()
    apply(store, source, label)
    selection = store.retrieve(source.features, cutoff=1, policy_version="v1")
    before = store.events
    query = source.features
    if fault == "mutable-items":
        object.__setattr__(selection, "items", list(selection.items))
    elif fault == "boolean-revision":
        object.__setattr__(selection.items[0], "revision", True)
    elif fault == "boolean-work":
        selection.work["token_visits"] = True
    elif fault == "extra-field":
        object.__setattr__(selection, "unadmitted", True)
    elif fault == "foreign-target":
        object.__setattr__(selection.items[0].evidence, "target_id", "network")
    elif fault == "inactive-target":
        query = replace(query, allowed_candidates=query.allowed_candidates[:-1])
    else:
        object.__setattr__(selection, "context", "é" * 769)
    with pytest.raises(ValueError):
        decision_from_evidence(query, selection)
    assert store.events == before


@pytest.fixture(scope="module")
def online_example():
    from scripts import run_online_temporal_feedback as driver

    observed: dict[str, Any] = dict(
        owners=[],
        streams=[],
        stores=[],
        training_rows=[],
        readout_updates=[],
        queries=[],
        feedback=[],
        scores=[],
        events=[],
        finished=False,
    )
    original_init = driver.PerceptronLearner.__init__
    original_train = driver.PerceptronLearner.train_batch
    original_readout = driver.PerceptronDecisionPredictor.__call__
    original_stream = driver.AttributionStream.__init__
    original_finish = driver.AttributionStream.finish
    original_feedback = driver.apply_decision_feedback
    original_score = driver._score_view
    original_store = driver.DecisionEvidenceStore.__init__

    def construct(owner, policy):
        original_init(owner, policy)
        observed["owners"].append(owner)

    def train(owner, features, targets):
        observed["training_rows"].append(len(features))
        return original_train(owner, features, targets)

    def readout(owner, query):
        observed["readout_updates"].append(len(observed["training_rows"]))
        observed["queries"].append(deepcopy(query))
        assert set(vars(query)) == {"request_id", "query", "allowed_candidates"}
        return original_readout(owner, query)

    def create_stream(stream, *args, **kwargs):
        original_stream(stream, *args, **kwargs)
        observed["streams"].append(stream)

    def finish(stream):
        result = original_finish(stream)
        observed["finished"] = True
        return result

    def construct_store(store, *args, **kwargs):
        original_store(store, *args, **kwargs)
        observed["stores"].append(store)

    def publish(store, source, label, **kwargs):
        stream = observed["streams"][0]
        receipts = stream.applied_updates
        assert receipts and receipts[-1].update_number == len(observed["training_rows"])
        assert source.observed_at == receipts[-1].observed_at
        assert label.arrived_at == receipts[-1].arrived_at == kwargs["recorded_at"]
        observed["feedback"].append((deepcopy(source), deepcopy(label), dict(kwargs)))
        return original_feedback(store, source, label, **kwargs)

    def score(row, *, memory):
        assert observed["finished"] is True and observed["training_rows"] == [3] * 12
        observed["scores"].append((row["phase"], row["control"], memory))
        return original_score(row, memory=memory)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(driver.PerceptronLearner, "__init__", construct)
        patch.setattr(driver.PerceptronLearner, "train_batch", train)
        patch.setattr(driver.PerceptronDecisionPredictor, "__call__", readout)
        patch.setattr(driver.AttributionStream, "__init__", create_stream)
        patch.setattr(driver.AttributionStream, "finish", finish)
        patch.setattr(driver, "apply_decision_feedback", publish)
        patch.setattr(driver, "_score_view", score)
        patch.setattr(driver.DecisionEvidenceStore, "__init__", construct_store)
        report = driver.run_example(progress=observed["events"].append)
    return report, observed


def test_should_read_live_weight_prefixes_but_delay_every_score_until_training_finishes(
    online_example,
):
    report, observed = online_example
    assert report["complete"] is True, report.get("error")
    assert observed["training_rows"] == [3] * 12
    assert observed["readout_updates"] == [0] * 16 + [4] * 16 + [8] * 24 + [12] * 24
    assert len(observed["scores"]) == 16 and observed["finished"] is True
    assert "training_finished_before_scoring" in observed["events"]
    assert (
        report["readout_work"]["weight_attempted"]
        == report["readout_work"]["weight_returned"]
        == report["readout_work"]["weight_validated"]
        == 80
    )
    assert (
        report["work"]["committed_weight_updates"] == 12
        and report["work"]["committed_training_rows"] == 36
    )
    assert (
        report["final_examples"]
        == report["probe_updates"]
        == report["replay_examples"]
        == report["external_actions"]
        == report["work"]["model_random_draws"]
        == 0
    )
    assert len(observed["queries"]) == 80
    assert all(query.query.endswith(" please") for query in observed["queries"])
    assert len(observed["owners"]) == 3
    assert [owner.completed_updates for owner in observed["owners"]] == [0, 0, 12]
    assert [owner.training_examples for owner in observed["owners"]] == [0, 0, 36]
    assert observed["owners"][0].snapshot_state() == observed["owners"][1].snapshot_state()
    assert [row["update"] for row in report["snapshots"]] == [0, 4, 8, 12]
    assert report["work"]["retained_coefficient_bytes"] == 640
    assert report["work"]["latest_observed_candidate_state"]["completed_updates"] == 12
    assert report["work"]["original_source_state"]["completed_updates"] == 0
    json.dumps(report, allow_nan=False)


def test_should_bind_every_memory_add_to_an_original_raw_observation_and_completed_batch_receipt(
    online_example,
):
    report, observed = online_example
    assert (
        len(report["observations"])
        == len(report["feedback"])
        == len(report["evidence_events"])
        == len(observed["feedback"])
        == 36
    )
    sources = {row["source"]["sample_id"]: row["source"] for row in report["observations"]}
    expected = [
        example
        for phase in ("after_A", "after_B", "after_return")
        for exposure in range(4)
        for example in temporal_examples(phase, "train", exposure=exposure)
    ]
    assert set(sources) == {example.request.request_id for example in expected}
    for index, (row, event, example) in enumerate(
        zip(report["feedback"], report["evidence_events"], expected, strict=True)
    ):
        update = index // 3 + 1
        source = sources[row["source_id"]]
        fact = event["evidence"]
        assert row["parent_update"] == update and event["revision"] == 0
        assert source["features"] == asdict(example.request) and source["role"] == "train"
        assert source["permissions"] == dict(training=True, replay=True, evaluation=False)
        assert source["observed_at"] == fact["observed_at"] == update - 1
        assert (
            row["label"]["arrived_at"]
            == row["recorded_at"]
            == fact["available_at"]
            == fact["recorded_at"]
            == update
        )
        assert row["label"]["targets"] == fact["target_id"] == example.target_id
        assert row["label"]["model_version"] == source["model_version"]
        assert fact["policy_version"] == example.policy_version
        assert (
            fact["source_group"] == "train-token" and fact["permissions"] == source["permissions"]
        )
        assert row["event"] == event
    assert len(report["receipts"]) == 12
    assert [receipt["observed_at"] for receipt in report["receipts"]] == list(range(12))
    assert [receipt["arrived_at"] for receipt in report["receipts"]] == list(range(1, 13))
    assert [receipt["update_number"] for receipt in report["receipts"]] == list(range(1, 13))
    assert (
        report["evidence_work"]["writes_attempted"]
        == report["evidence_work"]["writes_completed"]
        == 36
    )
    assert report["evidence_storage_bytes"] <= 65536


def test_should_report_full_denominator_coverage_and_preserve_unavailable_policy_controls(
    online_example,
):
    report, _ = online_example
    assert [(row["phase"], row["control"], row["cutoff"]) for row in report["memory_views"]] == [
        ("baseline", "adaptive_memory", 0),
        ("after_A", "adaptive_memory", 4),
        ("after_B", "adaptive_memory", 8),
        ("after_B", "frozen_A_memory", 4),
        ("after_return", "adaptive_memory", 12),
        ("after_return", "frozen_A_memory", 4),
    ]
    assert [row["score"]["answered"] for row in report["memory_views"]] == [0, 3, 3, 0, 4, 0]
    assert [row["score"]["accuracy_all_cases"] for row in report["memory_views"]] == [
        0.0,
        0.75,
        0.75,
        0.0,
        1.0,
        0.0,
    ]
    assert (
        report["readout_work"]["memory_attempted"]
        == report["readout_work"]["memory_completed"]
        == 48
    )
    assert (
        report["readout_work"]["memory_validated"] == 20
        and report["readout_work"]["memory_unavailable"] == 28
    )
    assert (
        report["evidence_work"]["retrievals_completed"] == 48
        and report["evidence_work"]["metadata_scans"] == 1056
    )
    for row in report["memory_views"]:
        score = row["score"]
        assert score["cases"] == 4 and score["coverage"] == score["answered"] / 4
        assert score["proper_score_denominator"] == score["answered"]
        evaluation = score["answered_only_evaluation"]
        if score["answered"] == 0:
            assert evaluation is None
        else:
            assert len(evaluation["cases"]) == score["answered"]
            assert evaluation["metrics"]["accuracy"] == 1.0
        for pair in row["pairs"]:
            first, reversed_call = pair["calls"]
            assert (first["status"] == "answered") is (reversed_call["status"] == "answered")
            if first["status"] == "unavailable":
                assert "result" not in first and "returned" not in first
                assert first["selection"]["items"] == ()
            else:
                assert dict(first["result"]["probabilities"]) == dict(
                    reversed_call["result"]["probabilities"]
                )
    summary = report["summary"]
    assert summary["parcel_acquisition_after_B"] is summary["parcel_retained_at_return"] is True
    assert summary["password_answered"] == summary["password_correct"] == [True, False, True]
    assert summary["current_rule_old_domain_return_success"] is True
    for phase, result in summary["by_phase"].items():
        live = next(
            row
            for row in report["memory_views"]
            if row["phase"] == phase and row["control"] == "adaptive_memory"
        )
        learned = next(
            row
            for row in report["weight_views"]
            if row["phase"] == phase and row["control"] == "learned"
        )
        assert (
            result["memory_minus_learned_accuracy"]
            == live["score"]["accuracy_all_cases"] - learned["score"]["accuracy_all_cases"]
        )
        assert result["memory_coverage"] == live["score"]["coverage"]
        assert result["obsolete_rule_errors"] == 0


def test_should_score_weight_outputs_against_current_truth_without_forcing_a_favorable_model_result(
    online_example,
):
    report, _ = online_example
    assert len(report["weight_views"]) == 10
    for row in report["weight_views"]:
        examples = temporal_examples(row["phase"], "outer_selection")
        by_id = {example.request.request_id: example for example in examples}
        correct = sum(
            pair["calls"][0]["result"]["selected_id"] == by_id[pair["request_id"]].target_id
            for pair in row["pairs"]
        )
        score = row["score"]
        assert score["answered"] == score["proper_score_denominator"] == 4
        assert score["coverage"] == 1.0 and score["correct"] == correct
        assert score["accuracy_all_cases"] == correct / 4
        assert score["answered_only_evaluation"]["metrics"]["accuracy"] == correct / 4
        assert len(row["pairs"]) == 4 and all(len(pair["calls"]) == 2 for pair in row["pairs"])


def test_should_return_detached_committed_receipts_at_a_quiescent_owned_stream_boundary(
    online_example,
):
    report, observed = online_example
    stream = observed["streams"][0]
    receipts = stream.applied_updates
    assert type(receipts) is tuple and len(receipts) == 12
    assert [asdict(receipt) for receipt in receipts] == report["receipts"]
    original = deepcopy(stream.applied_updates)
    object.__setattr__(receipts[0], "update_number", 999)
    object.__setattr__(receipts[0].diagnostic, "value", 999.0)
    assert stream.applied_updates == original
    with stream._exclusive():
        with pytest.raises(ValueError):
            _ = stream.applied_updates


def test_should_check_original_stream_ownership_before_exposing_receipts(
    online_example, monkeypatch
):
    _, observed = online_example
    stream = observed["streams"][0]
    with monkeypatch.context() as patch:
        patch.setattr(stream, "_training_only", False)
        with pytest.raises(ValueError):
            _ = stream.applied_updates
    assert len(stream.applied_updates) == 12


@pytest.mark.parametrize("fault", ["boolean-receipt", "changed-feedback"])
def test_should_refuse_corrupted_published_prefix_without_changing_the_genuine_paid_evidence(
    online_example, fault
):
    from scripts import run_online_temporal_feedback as driver

    report, observed = online_example
    changed = deepcopy(report)
    if fault == "boolean-receipt":
        changed["receipts"][0]["update_number"] = True
    else:
        changed["feedback"][0]["event"]["evidence"]["target_id"] = "inspect"
    before = deepcopy(report)
    with pytest.raises(ValueError):
        driver._verify(observed["streams"][0], observed["stores"][0], changed, 12)
    assert report == before
    assert observed["training_rows"] == [3] * 12 and len(observed["queries"]) == 80


def test_should_refuse_existing_cli_output_before_constructing_a_stream(monkeypatch, tmp_path):
    from scripts import run_online_temporal_feedback as driver

    def forbidden(*_args, **_kwargs):
        pytest.fail("existing output must refuse before fitting")

    monkeypatch.setattr(driver, "run_example", forbidden)
    output = tmp_path / "paid.json"
    output.write_bytes(b"original paid output")
    with pytest.raises(FileExistsError):
        driver.main(["--output", str(output)])
    assert output.read_bytes() == b"original paid output"
