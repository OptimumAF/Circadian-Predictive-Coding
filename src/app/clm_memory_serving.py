"""Fixed typed memory-serving requests and complete cached-record admission.

Consumes scalar reports from the existing stream/serving ports. No backend,
head update, scoring, label grant, store write or external effect occurs here.
"""

from dataclasses import asdict
import json

from src.app.clm_reserved_probe import _require_work
from src.core.temporal_decision_sequence import PHASES, temporal_examples
from src.core.typed_decision import DecisionRequest


def memory_serving_requests():
    known = tuple(row.request for row in temporal_examples("after_return", "outer_selection"))
    # No target is assigned: this control exercises actual fallback availability.
    return (
        *known,
        DecisionRequest(
            "typed-after-return-unseen", "unknown workflow please", known[0].allowed_candidates
        ),
    )


def _record(value):
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _source_events():
    events = []
    for batch in range(12):
        phase = PHASES[batch // 4]
        for example in temporal_examples(phase, "train", exposure=batch % 4):
            events.append(
                dict(
                    revision=0,
                    evidence=dict(
                        evidence_id=example.request.request_id,
                        observed_at=2 * batch,
                        available_at=2 * batch + 1,
                        recorded_at=(7, 15, 23)[batch // 4],
                        policy_version=example.policy_version,
                        source_group=example.source_group,
                        request=_record(example.request),
                        target_id=example.target_id,
                        role="train",
                        permissions=dict(training=True, replay=True, evaluation=False),
                    ),
                )
            )
    return events


def _require_source_record(report, record):
    """Validate the sealed phase source and one paid copied-store receipt."""
    source = report.get("train_evidence_memory")
    if type(source) is dict:
        source = json.loads(json.dumps(source, allow_nan=False))
    events = _source_events()
    storage = len(json.dumps(events, ensure_ascii=False, separators=(",", ":")).encode("utf8"))
    # These are the fixed four-phase source query costs, not a renewed budget.
    source_work = dict(
        writes_attempted=36,
        writes_completed=36,
        retrieval_attempts=16,
        retrievals_completed=16,
        metadata_scans=288,
        candidate_comparisons=192,
        token_visits=224,
        rendered_bytes=2335,
    )
    if (
        type(source) is not dict
        or source.get("complete") is not True
        or source.get("events") != events
        or source.get("storage_bytes") != storage
        or record.get("memory_source") != source
        or record.get("unscored_request_ids") != ["typed-after-return-unseen"]
    ):
        raise ValueError("original completed canonical TRAIN memory and unlabeled control required")
    _require_work(source.get("work"), source_work)
    _require_work(
        record.get("memory_binding_work"),
        dict(
            binding_attempts=1,
            binding_returns=1,
            copy_write_attempts=36,
            copy_write_returns=36,
            source_events=36,
            source_storage_bytes=storage,
            view_storage_bytes=storage,
            source_work=source_work,
            cutoff=23,
            policy_version="v2",
            max_calls=5,
            copy_return_unknown=False,
            view_write_work=dict(
                writes_attempted=36,
                writes_completed=36,
                retrieval_attempts=0,
                retrievals_completed=0,
                metadata_scans=0,
                candidate_comparisons=0,
                token_visits=0,
                rendered_bytes=0,
            ),
        ),
    )
    return events


def require_memory_serving(report, record):
    """Bind observed serving calls to the sealed source and copied-store budget."""
    events = _require_source_record(report, record)
    view = record.get("memory_view_report")
    if (
        type(view) is not dict
        or view.get("mode") != "fallback"
        or view.get("cutoff") != 23
        or view.get("policy_version") != "v2"
        or type(view.get("calls")) is not list
        or len(view["calls"]) != 5
    ):
        raise ValueError("complete finite memory/fallback serving view required")
    _require_work(
        view.get("work"),
        dict(attempted=5, returned=5, validated=5, callback_attempted=1, callback_returned=1),
    )
    rendered = tokens = 0
    for index, (request, call, result) in enumerate(
        zip(memory_serving_requests(), view["calls"], record["predictions"], strict=True)
    ):
        known = index < 4
        if (
            call.get("request") != _record(request)
            or call.get("result") != result
            or call.get("choice_source") != ("memory" if known else "fallback")
        ):
            raise ValueError("memory serving request/result/source changed")
        selection = call.get("selection", {})
        items, context = selection.get("items"), selection.get("context")
        if type(items) is not list or len(items) != int(known) or type(context) is not str:
            raise ValueError("canonical known/unknown selection required")
        if known:
            item = items[0]
            if (
                item not in events
                or item["evidence"]["policy_version"] != "v2"
                or item["evidence"]["request"]["query"] != request.query.removesuffix(" please")
            ):
                raise ValueError("selection is not eligible original TRAIN evidence")
            target = item["evidence"]["target_id"]
            if result["selected_id"] != target or result["probabilities"] != [
                [candidate.candidate_id, float(candidate.candidate_id == target)]
                for candidate in request.allowed_candidates
            ]:
                raise ValueError("memory result differs from observed eligible label")
        elif call.get("supplied") != _record(request):
            raise ValueError("fallback changed its original encoder input recipe")
        expected_context = dict(
            decision_evidence=[
                dict(
                    evidence_id=item["evidence"]["evidence_id"],
                    revision=item["revision"],
                    query=item["evidence"]["request"]["query"],
                    **{
                        key: item["evidence"][key]
                        for key in (
                            "target_id",
                            "observed_at",
                            "available_at",
                            "recorded_at",
                            "policy_version",
                            "source_group",
                        )
                    },
                )
                for item in items
            ]
        )
        if context != json.dumps(expected_context, ensure_ascii=False, separators=(",", ":")):
            raise ValueError("serving context differs from its observed TRAIN selection")
        count = len(context.encode("utf8"))
        if count > 768:
            raise ValueError("serving context exceeds its original byte bound")
        _require_work(
            selection.get("work"),
            dict(
                metadata_scans=36,
                candidate_comparisons=24,
                token_visits=26 if known else 27,
                rendered_bytes=count,
            ),
        )
        rendered += count
        tokens += 26 if known else 27
        _require_work(
            call.get("store_work_after"),
            dict(
                writes_attempted=36,
                writes_completed=36,
                retrieval_attempts=index + 1,
                retrievals_completed=index + 1,
                metadata_scans=36 * (index + 1),
                candidate_comparisons=24 * (index + 1),
                token_visits=tokens,
                rendered_bytes=rendered,
            ),
        )
