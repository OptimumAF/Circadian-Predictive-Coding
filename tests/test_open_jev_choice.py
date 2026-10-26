"""Analytic native-wire fixtures; no upstream imports, transport or inference.

Open-Jev insertion-order ties and native confidence differ from the local lexical
DecisionResult contract. These tests intentionally retain that distinction.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import json
from typing import Any

import pytest

from src.adapters.open_jev_choice import (
    build_open_jev_choice,
    decode_open_jev_choice,
)
from src.core.typed_decision import DecisionCandidate, DecisionRequest


def request() -> DecisionRequest:
    return DecisionRequest(
        "local-request-1",
        "Café invoice needs review",
        (
            DecisionCandidate("z", "Billing café"),
            DecisionCandidate("a", "Account access"),
        ),
    )


def response(
    *,
    probabilities: dict[str, Any] | None = None,
    choice: str = "z",
    confidence: Any = 0.4,
    question_id: str = "decision",
) -> dict[str, Any]:
    return {
        "answers": {
            question_id: {
                "type": "choice",
                "choice": choice,
                "probabilities": {"a": 0.3, "z": 0.7} if probabilities is None else probabilities,
                "confidence": confidence,
            }
        }
    }


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf8")


def test_should_build_only_declared_model_inputs_preserving_native_candidate_order():
    original = request()
    before = deepcopy(original)
    wire = build_open_jev_choice(original, "Which team should handle this?", question_id="route")
    assert wire == {
        "model": "open-jev",
        "state": "Café invoice needs review",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"z": "Billing café", "a": "Account access"},
            }
        },
    }
    assert list(wire["questions"]["route"]["criteria"]) == ["z", "a"]
    assert "local-request-1" not in json.dumps(wire)
    assert not {"target_id", "role", "source_group", "labels"}.intersection(wire)
    wire["questions"]["route"]["criteria"]["z"] = "changed outgoing copy"
    assert original == before


def test_should_decode_native_probabilities_without_softmax_or_invented_evidence():
    original = request()
    body = encoded(response())
    reply = decode_open_jev_choice(original, body)
    assert reply.request_id == original.request_id
    assert reply.question_id == "decision" and reply.choice == "z"
    assert reply.probabilities == (("z", 0.7), ("a", 0.3))
    assert reply.confidence == 0.4 and reply.raw_response == body
    assert {field.name for field in fields(reply)} == {
        "request_id",
        "question_id",
        "choice",
        "probabilities",
        "confidence",
        "raw_response",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(reply, "choice", "a")


def test_should_use_request_order_for_ties_even_when_reply_map_is_reversed():
    body = encoded(response(probabilities={"a": 0.5, "z": 0.5}, confidence=0.0))
    assert decode_open_jev_choice(request(), body).choice == "z"
    reversed_request = replace(
        request(), allowed_candidates=tuple(reversed(request().allowed_candidates))
    )
    with pytest.raises(ValueError):
        decode_open_jev_choice(reversed_request, body)
    reverse_body = encoded(response(probabilities={"z": 0.5, "a": 0.5}, choice="a", confidence=0.0))
    reply = decode_open_jev_choice(reversed_request, reverse_body)
    assert reply.choice == "a" and reply.probabilities == (("a", 0.5), ("z", 0.5))


def test_should_preserve_mass_roundoff_but_check_confidence_on_normalized_mass():
    probabilities = {"z": 0.7000003, "a": 0.3000003}
    confidence = (probabilities["z"] / sum(probabilities.values()) - 0.5) / 0.5
    reply = decode_open_jev_choice(
        request(), encoded(response(probabilities=probabilities, confidence=confidence))
    )
    assert reply.probabilities == tuple(probabilities.items())
    assert sum(value for _, value in reply.probabilities) != 1.0
    assert reply.confidence == confidence


def test_should_use_native_three_way_uniform_baseline_confidence():
    query = replace(
        request(),
        allowed_candidates=(*request().allowed_candidates, DecisionCandidate("m", "Shipping")),
    )
    reply = decode_open_jev_choice(
        query, encoded(response(probabilities={"a": 0.2, "m": 0.1, "z": 0.7}, confidence=0.55))
    )
    assert reply.probabilities == (("z", 0.7), ("a", 0.2), ("m", 0.1))
    assert reply.confidence == 0.55


def test_should_admit_singleton_choice_with_native_confidence_one():
    query = replace(request(), allowed_candidates=request().allowed_candidates[:1])
    reply = decode_open_jev_choice(query, encoded(response(probabilities={"z": 1}, confidence=1)))
    assert reply.probabilities == (("z", 1.0),) and reply.confidence == 1.0


def test_should_keep_uninterpreted_root_metadata_only_in_original_bytes():
    payload = response(question_id="route")
    payload.update(
        model="unattested-upstream-name",
        usage={"input_tokens": 38, "output_tokens": 0},
        provenance={"note": "café"},
    )
    body = b"  " + encoded(payload) + b"\n"
    reply = decode_open_jev_choice(request(), body, question_id="route")
    assert reply.raw_response == body and reply.question_id == "route"
    assert not hasattr(reply, "model") and not hasattr(reply, "token_visits")
    assert not hasattr(reply, "raw_logits")


def test_should_not_claim_request_nonce_authentication_from_anonymous_native_reply():
    query = replace(request(), request_id="another-local-id")
    reply = decode_open_jev_choice(query, encoded(response()))
    assert reply.request_id == "another-local-id"


@pytest.mark.parametrize(
    "kind",
    [
        "mass",
        "negative",
        "above_one",
        "bool_probability",
        "bool_confidence",
        "wrong_confidence",
        "nonmax",
        "extra_candidate",
        "missing_candidate",
        "unknown_choice",
        "type",
        "extra_answer",
        "wrong_question",
        "extra_field",
    ],
)
def test_should_refuse_forged_or_mismatched_native_answers(kind):
    payload = response()
    answer = payload["answers"]["decision"]
    if kind == "mass":
        answer["probabilities"] = {"z": 0.7, "a": 0.31}
    elif kind == "negative":
        answer["probabilities"] = {"z": 1.1, "a": -0.1}
    elif kind == "above_one":
        answer["probabilities"] = {"z": 1.0000001, "a": 0.0}
    elif kind == "bool_probability":
        answer["probabilities"]["z"] = True
    elif kind == "bool_confidence":
        answer["confidence"] = False
    elif kind == "wrong_confidence":
        answer["confidence"] = 0.7
    elif kind == "nonmax":
        answer["choice"] = "a"
    elif kind == "extra_candidate":
        answer["probabilities"]["other"] = 0.0
    elif kind == "missing_candidate":
        del answer["probabilities"]["a"]
    elif kind == "unknown_choice":
        answer["choice"] = "other"
    elif kind == "type":
        answer["type"] = "score"
    elif kind == "extra_answer":
        payload["answers"]["extra"] = deepcopy(answer)
    elif kind == "wrong_question":
        payload["answers"] = {"different": answer}
    else:
        answer["raw_logits"] = [1.0, 0.0]
    with pytest.raises(ValueError):
        decode_open_jev_choice(request(), encoded(payload))


@pytest.mark.parametrize(
    "body",
    [b"", b"x" * 65537, b"\xff", b"{} trailing", b"[]", b"null", b'{"answers":{},"answers":{}}'],
    ids=[
        "empty",
        "oversized",
        "invalid-utf8",
        "trailing-text",
        "array-root",
        "null-root",
        "duplicate-root-key",
    ],
)
def test_should_refuse_malformed_or_unbounded_json(body):
    with pytest.raises(ValueError):
        decode_open_jev_choice(request(), body)


@pytest.mark.parametrize(
    "metadata",
    [b'{"x":1,"x":2}', b"NaN", b"Infinity", b"1e400", b"9" * 400, b"9007199254740992"],
    ids=["duplicate-key", "nan", "infinity", "float-overflow", "huge-integer", "unsafe-integer"],
)
def test_should_reject_duplicate_or_unsafe_numbers_in_metadata_of_an_otherwise_valid_reply(
    metadata,
):
    body = encoded(response())[:-1] + b',"metadata":' + metadata + b"}"
    with pytest.raises(ValueError):
        decode_open_jev_choice(request(), body)


def test_should_refuse_duplicate_probability_ids_before_they_can_be_overwritten():
    body = b'{"answers":{"decision":{"type":"choice","choice":"z","confidence":0.4,"probabilities":{"z":0.1,"z":0.7,"a":0.3}}}}'
    with pytest.raises(ValueError):
        decode_open_jev_choice(request(), body)


def test_should_accept_exact_body_limit_without_discarding_raw_whitespace():
    body = encoded(response())
    body += b" " * (65536 - len(body))
    assert decode_open_jev_choice(request(), body).raw_response == body


def test_should_refuse_excessive_json_nesting_as_a_value_error():
    payload = encoded(response())[:-1] + b',"metadata":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}"
    with pytest.raises(ValueError):
        decode_open_jev_choice(request(), payload)


def test_should_accept_exactly_sixty_four_container_levels():
    # The root object is level one; the metadata value adds 63 array levels.
    body = encoded(response())[:-1] + b',"metadata":' + b"[" * 63 + b"0" + b"]" * 63 + b"}"
    assert decode_open_jev_choice(request(), body).raw_response == body


def test_should_refuse_sixty_five_container_levels_before_json_parser_recursion():
    body = encoded(response())[:-1] + b',"metadata":' + b"[" * 64 + b"0" + b"]" * 64 + b"}"
    with pytest.raises(ValueError, match="64 container levels"):
        decode_open_jev_choice(request(), body)


def test_should_ignore_brackets_inside_quoted_text_with_escaped_quotes_and_backslashes():
    payload = response()
    payload["metadata"] = {"text": "[" * 70 + '\\"' + "}" * 70 + '"quoted"' + "\\" + "]" * 70}
    body = encoded(payload)
    assert decode_open_jev_choice(request(), body).raw_response == body


@pytest.mark.parametrize("field", ["query", "candidate_id", "description"])
def test_should_revalidate_bypassed_frozen_request_fields_at_both_boundaries(field):
    query = request()
    target = query if field == "query" else query.allowed_candidates[0]
    object.__setattr__(target, field, "invalid\ntext")
    with pytest.raises(ValueError):
        build_open_jev_choice(query, "Route")
    with pytest.raises(ValueError):
        decode_open_jev_choice(query, encoded(response()))


@pytest.mark.parametrize("question_id", ["", "bad\nquestion", True])
def test_should_refuse_invalid_question_identifiers_at_both_boundaries(question_id):
    with pytest.raises(ValueError):
        build_open_jev_choice(request(), "Route", question_id=question_id)
    with pytest.raises(ValueError):
        decode_open_jev_choice(request(), encoded(response()), question_id=question_id)


@pytest.mark.parametrize(
    "instructions", ["", b"Route", "x" * 2049], ids=["empty", "bytes", "oversized"]
)
def test_should_refuse_unsupported_instructions_instead_of_coercing_them(instructions):
    with pytest.raises(ValueError):
        build_open_jev_choice(request(), instructions)


@pytest.mark.parametrize("body", ["{}", bytearray(b"{}")], ids=["text", "bytearray"])
def test_should_refuse_nonbytes_response_bodies(body):
    with pytest.raises(ValueError):
        decode_open_jev_choice(request(), body)


def test_should_detach_decoded_identity_and_values_from_later_caller_record_changes():
    query = request()
    reply = decode_open_jev_choice(query, encoded(response()))
    object.__setattr__(query, "request_id", "later-request")
    object.__setattr__(query.allowed_candidates[0], "candidate_id", "later-candidate")
    assert reply.request_id == "local-request-1"
    assert reply.probabilities == (("z", 0.7), ("a", 0.3))
