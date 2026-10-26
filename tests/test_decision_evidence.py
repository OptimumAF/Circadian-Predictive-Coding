"""Scalar causal evidence edits and protected typed predictor boundaries.

No model, fitting, array kernels, external tools or semantic injection resistance
are exercised by these tests.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import pytest

from src.app.decision_evidence import EvidenceDecisionPredictor
from src.core.decision_evidence import DecisionEvidence, DecisionEvidenceStore, EvidencePolicy
from src.core.experience import ExperiencePermissions
from src.core.typed_decision import (
    CandidateScore,
    DecisionCandidate,
    DecisionRequest,
    DecisionResult,
    ProbabilityDecisionResult,
)


CATALOG = (
    DecisionCandidate("defer", "defer the workflow"),
    DecisionCandidate("inspect", "inspect the workflow"),
    DecisionCandidate("route", "route the workflow"),
)


def request(query="parcel please", *, identity="query", candidates=CATALOG):
    return DecisionRequest(identity, query, deepcopy(candidates))


def evidence(
    identity="fact",
    *,
    target="inspect",
    query="parcel please",
    observed=1,
    available=2,
    recorded=3,
    policy="v2",
):
    return DecisionEvidence(
        identity,
        observed,
        available,
        recorded,
        policy,
        "train-facts",
        request(query, identity=f"train-{identity}"),
        target,
        permissions=ExperiencePermissions(training=True, replay=True),
    )


def store_with_fact(**kwargs):
    store = DecisionEvidenceStore()
    store.add(evidence(**kwargs))
    return store


def typed_result(query, selected="route", *, probability_only=False):
    probabilities = tuple(
        (candidate.candidate_id, float(candidate.candidate_id == selected))
        for candidate in query.allowed_candidates
    )
    if probability_only:
        return ProbabilityDecisionResult(query.request_id, selected, probabilities)
    return DecisionResult(
        query.request_id,
        selected,
        tuple(CandidateScore(identity, value, value) for identity, value in probabilities),
        1,
        len(probabilities),
    )


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "outer_selection", "final"])
def test_should_refuse_nontrain_evidence_before_retaining_it(role):
    source, store = evidence(), DecisionEvidenceStore()
    object.__setattr__(source, "role", role)
    with pytest.raises(ValueError):
        store.add(source)
    assert store.events == () and store.storage_bytes == len(b"[]")


@pytest.mark.parametrize(
    "permissions",
    [
        ExperiencePermissions(),
        ExperiencePermissions(training=True),
        ExperiencePermissions(replay=True),
    ],
)
def test_should_require_explicit_training_and_replay_consent(permissions):
    source, store = evidence(), DecisionEvidenceStore()
    object.__setattr__(source, "permissions", permissions)
    with pytest.raises(ValueError):
        store.add(source)
    assert store.events == ()


@pytest.mark.parametrize(
    "field,value",
    [("observed_at", True), ("available_at", 0), ("recorded_at", 1), ("target_id", "network")],
)
def test_should_revalidate_mutated_native_times_and_allowed_targets(field, value):
    source, store = evidence(), DecisionEvidenceStore()
    object.__setattr__(source, field, value)
    with pytest.raises(ValueError):
        store.add(source)
    assert store.events == ()


def test_should_own_admitted_evidence_and_return_detached_history_and_work():
    source, store = evidence(), DecisionEvidenceStore()
    store.add(source)
    before = deepcopy(store.events)
    object.__setattr__(source, "target_id", "route")
    object.__setattr__(
        source.request.allowed_candidates[0], "description", "changed caller definition"
    )
    assert store.events == before
    returned = store.events
    object.__setattr__(returned[0].evidence, "target_id", "route")
    assert store.events == before
    work = store.work
    key = next(iter(work))
    work[key] = "caller mutation"
    assert store.work[key] != "caller mutation"
    assert store.storage_bytes > 0


def test_should_keep_observation_source_policy_and_consent_when_only_the_label_is_edited():
    source, store = evidence(), DecisionEvidenceStore()
    first = store.add(source)
    second = store.edit("fact", "defer", available_at=5, recorded_at=6, expected_revision=0)
    assert first.revision == 0 and second.revision == 1
    old, new = asdict(first.evidence), asdict(second.evidence)
    for key in (
        "evidence_id",
        "observed_at",
        "policy_version",
        "source_group",
        "request",
        "role",
        "permissions",
    ):
        assert old[key] == new[key]
    assert new["target_id"] == "defer" and new["available_at"] == 5 and new["recorded_at"] == 6
    assert len(store.events) == 2


@pytest.mark.parametrize(
    "fault",
    ["stale", "missing", "unavailable-target", "backdated-record", "equal-record", "early-label"],
)
def test_should_refuse_invalid_edits_without_altering_owned_history(fault):
    store = store_with_fact()
    before = store.events
    options = dict(available_at=5, recorded_at=6, expected_revision=0)
    identity, target = "fact", "defer"
    if fault == "stale":
        options["expected_revision"] = 1
    elif fault == "missing":
        identity = "absent"
    elif fault == "unavailable-target":
        target = "network"
    elif fault == "backdated-record":
        options.update(available_at=2, recorded_at=2)
    elif fault == "equal-record":
        options.update(available_at=2, recorded_at=3)
    else:
        options["available_at"] = 0
    with pytest.raises(ValueError):
        store.edit(identity, target, **options)
    assert store.events == before


def test_should_preserve_old_cutoff_views_after_a_later_label_correction():
    store = store_with_fact()
    earlier = store.retrieve(request(), cutoff=3, policy_version="v2")
    before = deepcopy(earlier)
    store.edit("fact", "defer", available_at=5, recorded_at=6, expected_revision=0)
    still_earlier = store.retrieve(request(), cutoff=5, policy_version="v2")
    corrected = store.retrieve(request(), cutoff=6, policy_version="v2")
    assert earlier == before and still_earlier.items == earlier.items
    assert earlier.items[0].revision == 0 and earlier.items[0].evidence.target_id == "inspect"
    assert corrected.items[0].revision == 1 and corrected.items[0].evidence.target_id == "defer"
    assert json.loads(corrected.context)["decision_evidence"][0]["recorded_at"] == 6
    object.__setattr__(corrected.items[0].evidence, "target_id", "route")
    assert store.events[-1].evidence.target_id == "defer"


def test_should_wait_for_recording_time_even_when_a_label_was_available_earlier():
    store = store_with_fact(recorded=10)
    assert store.retrieve(request(), cutoff=2, policy_version="v2").items == ()
    assert store.retrieve(request(), cutoff=9, policy_version="v2").items == ()
    assert (
        store.retrieve(request(), cutoff=10, policy_version="v2").items[0].evidence.target_id
        == "inspect"
    )


@pytest.mark.parametrize("write", ["new-id", "edit"])
def test_should_preserve_queried_past_views_by_refusing_backfilled_records_and_labels(write):
    store = store_with_fact()
    before = store.retrieve(request(), cutoff=5, policy_version="v2")
    with pytest.raises(ValueError):
        if write == "new-id":
            store.add(evidence("backfilled", target="defer", recorded=5))
        else:
            store.edit("fact", "defer", available_at=5, recorded_at=5, expected_revision=0)
    assert store.retrieve(request(), cutoff=5, policy_version="v2").items == before.items
    store.add(evidence("new-observation", target="defer", observed=4, available=5, recorded=6))
    assert store.retrieve(request(), cutoff=5, policy_version="v2").items == before.items
    assert (
        store.retrieve(request(), cutoff=6, policy_version="v2").items[0].evidence.target_id
        == "defer"
    )


def test_should_seal_failed_retrieval_cutoffs_against_later_backfill():
    store = DecisionEvidenceStore(EvidencePolicy(max_context_bytes=32))
    store.add(evidence())
    with pytest.raises(ValueError):
        store.retrieve(request(), cutoff=10, policy_version="v2")
    with pytest.raises(ValueError):
        store.add(evidence("retroactive", recorded=9))
    assert len(store.events) == 1
    store.add(evidence("future", recorded=11))
    assert len(store.events) == 2


def test_should_filter_policy_before_retrieval_and_never_guess_current_rules():
    store = DecisionEvidenceStore()
    store.add(evidence("old", target="route", policy="v1"))
    store.add(evidence("new", target="inspect", policy="v2", recorded=4))
    assert (
        store.retrieve(request(), cutoff=100, policy_version="v1").items[0].evidence.target_id
        == "route"
    )
    assert store.retrieve(request(), cutoff=3, policy_version="v2").items == ()
    assert (
        store.retrieve(request(), cutoff=4, policy_version="v2").items[0].evidence.evidence_id
        == "new"
    )


def test_should_not_resurrect_an_old_label_when_the_latest_visible_target_is_now_inactive():
    store = store_with_fact()
    store.edit("fact", "defer", available_at=5, recorded_at=6, expected_revision=0)
    restricted = request(candidates=CATALOG[1:])
    assert (
        store.retrieve(restricted, cutoff=3, policy_version="v2").items[0].evidence.target_id
        == "inspect"
    )
    assert store.retrieve(restricted, cutoff=6, policy_version="v2").items == ()


@pytest.mark.parametrize("order", [("z", "a"), ("a", "z")], ids=["reversed-tie", "canonical-tie"])
def test_should_rank_positive_jaccard_then_recency_then_id_and_bound_top_k(order):
    store = DecisionEvidenceStore(EvidencePolicy(top_k=3))
    for identity in order:
        store.add(evidence(identity))
    store.add(evidence("recent", recorded=4))
    store.add(evidence("lower-overlap", query="parcel", recorded=5))
    store.add(evidence("no-overlap", query="billing", recorded=6))
    selection = store.retrieve(request(), cutoff=6, policy_version="v2")
    assert [item.evidence.evidence_id for item in selection.items] == ["recent", "a", "z"]
    assert selection.work["metadata_scans"] == selection.work["candidate_comparisons"] == 5
    assert selection.work["token_visits"] == 10
    assert selection.work["rendered_bytes"] == len(selection.context.encode("utf-8"))


def test_should_count_canonical_unicode_bytes_and_render_data_without_candidate_definitions():
    store = store_with_fact(query="café parcel")
    expected_bytes = json.dumps(
        [asdict(event) for event in store.events],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert store.storage_bytes == len(expected_bytes)
    selection = store.retrieve(request("CAFÉ parcel"), cutoff=3, policy_version="v2")
    data = json.loads(selection.context)
    row = data["decision_evidence"][0]
    assert row["query"] == "café parcel"
    assert "café" in selection.context and "\\u00e9" not in selection.context
    assert set(row) == {
        "evidence_id",
        "revision",
        "query",
        "target_id",
        "observed_at",
        "available_at",
        "recorded_at",
        "policy_version",
        "source_group",
    }
    assert "allowed_candidates" not in row and "instructions" not in row


def test_should_fail_on_event_capacity_without_evicting_or_publishing_an_edit():
    store = DecisionEvidenceStore(EvidencePolicy(max_events=1))
    store.add(evidence())
    before, size = store.events, store.storage_bytes
    with pytest.raises(ValueError):
        store.edit("fact", "defer", available_at=5, recorded_at=6, expected_revision=0)
    assert store.events == before and store.storage_bytes == size
    assert store.work["writes_attempted"] == 2 and store.work["writes_completed"] == 1


def test_should_admit_exact_canonical_storage_bytes_and_refuse_one_byte_less():
    probe = store_with_fact(query="café parcel")
    size = probe.storage_bytes
    exact = DecisionEvidenceStore(EvidencePolicy(max_storage_bytes=size))
    exact.add(evidence(query="café parcel"))
    assert exact.storage_bytes == size
    refused = DecisionEvidenceStore(EvidencePolicy(max_storage_bytes=size - 1))
    with pytest.raises(ValueError):
        refused.add(evidence(query="café parcel"))
    assert refused.events == () and refused.storage_bytes == len(b"[]")


def test_should_charge_full_rendering_then_refuse_context_overflow_without_truncation():
    store = DecisionEvidenceStore(EvidencePolicy(max_context_bytes=32))
    store.add(evidence())
    before = store.events
    with pytest.raises(ValueError):
        store.retrieve(request(), cutoff=3, policy_version="v2")
    assert store.events == before
    assert store.work["retrieval_attempts"] == 1 and store.work["retrievals_completed"] == 0
    assert store.work["rendered_bytes"] > 32 and store.work["metadata_scans"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "backdated-new-record"])
def test_should_refuse_duplicate_ids_or_nonmonotonic_new_records(fault):
    store = store_with_fact()
    before = store.events
    next_record = (
        evidence()
        if fault == "duplicate"
        else evidence("different", observed=0, available=1, recorded=2)
    )
    with pytest.raises(ValueError):
        store.add(next_record)
    assert store.events == before


@pytest.mark.parametrize(
    "options",
    [
        {"max_events": True},
        {"max_events": 0},
        {"max_storage_bytes": 63},
        {"max_context_bytes": 1537},
        {"top_k": 0},
        {"top_k": 9},
    ],
)
def test_should_refuse_invalid_native_policy_limits(options):
    with pytest.raises(ValueError):
        EvidencePolicy(**options)


def test_should_change_a_retrieval_decision_only_after_the_correction_is_recorded():
    store = store_with_fact()
    old = EvidenceDecisionPredictor(store, cutoff=3, policy_version="v2")
    before = old(request())
    store.edit("fact", "defer", available_at=5, recorded_at=6, expected_revision=0)
    current = EvidenceDecisionPredictor(store, cutoff=6, policy_version="v2")
    after = current(request())
    assert before.selected_id == "inspect" and after.selected_id == "defer"
    assert dict(before.probabilities) == {"defer": 0.0, "inspect": 1.0, "route": 0.0}
    assert dict(after.probabilities) == {"defer": 1.0, "inspect": 0.0, "route": 0.0}
    assert old(request()).selected_id == "inspect"
    assert (
        old.report["work"]["callback_attempted"]
        == current.report["work"]["callback_attempted"]
        == 0
    )


@pytest.mark.parametrize(
    "probability_only", [False, True], ids=["scored-result", "probability-result"]
)
def test_should_append_data_to_query_only_and_preserve_a_context_models_adverse_choice(
    probability_only,
):
    store = store_with_fact()
    original, received = request(), []

    def ignore_context(supplied):
        received.append(deepcopy(supplied))
        return typed_result(supplied, "route", probability_only=probability_only)

    predictor = EvidenceDecisionPredictor(
        store, cutoff=3, policy_version="v2", mode="context", predict=ignore_context
    )
    result = predictor(original)
    assert result.selected_id == "route"
    assert (
        received[0].request_id == original.request_id
        and received[0].allowed_candidates == original.allowed_candidates
    )
    assert received[0].query.startswith(original.query + " evidence ")
    context = received[0].query.split(" evidence ", 1)[1]
    assert json.loads(context)["decision_evidence"][0]["target_id"] == "inspect"
    assert original.query == "parcel please"
    report = predictor.report
    assert report["work"] == dict(
        attempted=1, returned=1, validated=1, callback_attempted=1, callback_returned=1
    )
    report["calls"][0]["request"]["query"] = "caller changed report"
    assert predictor.report["calls"][0]["request"]["query"] == original.query


@pytest.mark.parametrize(
    "fault",
    [
        "unreturned",
        "unsupported",
        "nonfinite",
        "changed-request",
        "changed-candidates",
        "evidence-edit",
    ],
)
def test_should_retain_born_callback_returns_and_stop_after_uncertain_context_calls(fault):
    store, calls = store_with_fact(), []

    def predict(supplied):
        calls.append(supplied.request_id)
        if fault == "unreturned":
            raise RuntimeError("controlled context callback failure")
        if fault == "unsupported":
            return object()
        result = typed_result(supplied, probability_only=True)
        if fault == "nonfinite":
            object.__setattr__(result, "probabilities", (("route", float("nan")),))
        elif fault == "changed-request":
            object.__setattr__(supplied, "request_id", "changed")
        elif fault == "changed-candidates":
            object.__setattr__(
                supplied.allowed_candidates[0], "description", "changed protected definition"
            )
        elif fault == "evidence-edit":
            store.edit("fact", "defer", available_at=5, recorded_at=6, expected_revision=0)
        return result

    predictor = EvidenceDecisionPredictor(
        store, cutoff=3, policy_version="v2", mode="context", predict=predict
    )
    original = request()
    with pytest.raises((ValueError, RuntimeError)):
        predictor(original)
    report = predictor.report
    assert report["work"]["attempted"] == report["work"]["callback_attempted"] == 1
    assert report["work"]["callback_returned"] == int(fault != "unreturned")
    assert report["work"]["validated"] == 0
    assert ("returned" in report["calls"][0]) is (fault != "unreturned")
    assert report["calls"][0]["error"] and original == request()
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError):
        predictor(original)
    assert calls == ["query"] and predictor.report["work"]["attempted"] == 1


def test_should_refuse_query_overflow_before_buying_a_context_callback():
    store = store_with_fact()

    def forbidden(_request):
        pytest.fail("query overflow must refuse before callback")

    predictor = EvidenceDecisionPredictor(
        store, cutoff=3, policy_version="v2", mode="context", predict=forbidden
    )
    with pytest.raises(ValueError):
        predictor(request("parcel " + "x" * 1990))
    assert predictor.report["work"]["attempted"] == 1
    assert predictor.report["work"]["callback_attempted"] == 0
    assert store.work["retrievals_completed"] == 1


def test_should_refuse_no_evidence_and_a_spent_wrapper_without_retrying_retrieval():
    store = DecisionEvidenceStore()
    absent = EvidenceDecisionPredictor(store, cutoff=3, policy_version="v2")
    with pytest.raises(ValueError):
        absent(request())
    with pytest.raises(ValueError):
        absent(request())
    assert store.work["retrieval_attempts"] == 1
    store.add(evidence(recorded=4))
    bounded = EvidenceDecisionPredictor(store, cutoff=4, policy_version="v2", max_calls=1)
    assert bounded(request()).selected_id == "inspect"
    with pytest.raises(ValueError):
        bounded(request())
    assert bounded.report["work"]["attempted"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "unknown"},
        {"mode": "context"},
        {"predict": lambda _request: None},
        {"max_calls": True},
        {"max_calls": 0},
        {"cutoff": True},
    ],
)
def test_should_refuse_invalid_predictor_configuration_before_retrieval(options):
    store = store_with_fact()
    arguments: dict[str, Any] = dict(cutoff=3, policy_version="v2")
    arguments.update(options)
    with pytest.raises(ValueError):
        EvidenceDecisionPredictor(store, **arguments)
    assert store.work["retrieval_attempts"] == 0
