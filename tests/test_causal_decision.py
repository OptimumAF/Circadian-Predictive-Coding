"""Causal text/identity/usage contract on plain immutable fixtures, no models."""

from dataclasses import FrozenInstanceError, asdict, replace
from itertools import permutations
from typing import Any

import pytest

from src.core.causal_decision import (
    MAX_RESPONSE_BYTES,
    CausalModelReference,
    CausalGeneration,
    GeneratedDecisionResult,
    decode_causal_decision,
    validate_generated_decision_result,
)
from src.core.typed_decision import DecisionCandidate, DecisionRequest

REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def model():
    return CausalModelReference("Qwen/Qwen3.5-2B", REVISION)


def request():
    return DecisionRequest(
        "r1",
        "Route this request",
        (
            DecisionCandidate("approve", "Approve request"),
            DecisionCandidate("reject", "Reject request"),
        ),
    )


def generation(**changes):
    values: dict[str, Any] = dict(
        request_id="r1",
        model=model(),
        text="approve",
        finish_reason="stop",
        reported_input_tokens=8,
        reported_output_tokens=2,
    )
    values.update(changes)
    return CausalGeneration(**values)


def decode(raw=None, query=None, **changes):
    options: dict[str, Any] = dict(expected_model=model(), max_input_tokens=8, max_output_tokens=2)
    options.update(changes)
    return decode_causal_decision(
        request() if query is None else query, generation() if raw is None else raw, **options
    )


def test_should_preserve_raw_evidence_without_inventing_probability_or_work():
    raw = generation()
    result = decode(raw)
    assert result.request_id == "r1" and result.selected_id == "approve"
    assert result.generation == raw and result.generation is not raw
    assert result.generation.model is not raw.model
    assert set(asdict(result)) == {"request_id", "selected_id", "generation"}
    assert not any(
        hasattr(result, field) for field in ("scores", "probabilities", "confidence", "logits")
    )
    validate_generated_decision_result(
        request(), result, expected_model=model(), max_input_tokens=8, max_output_tokens=2
    )
    assert (result.generation.reported_input_tokens, result.generation.reported_output_tokens) == (
        8,
        2,
    )


def test_should_keep_candidate_order_independent_and_exact_unicode_ids():
    query = request()
    for candidates in permutations(query.allowed_candidates):
        assert decode(query=replace(query, allowed_candidates=candidates)).selected_id == "approve"
    query = replace(query, allowed_candidates=(DecisionCandidate("承認", "Approve"),))
    assert decode(generation(text="承認"), query).selected_id == "承認"
    boundary = "a" * 64
    assert (
        decode(
            generation(text=boundary),
            replace(query, allowed_candidates=(DecisionCandidate(boundary, "x"),)),
        ).selected_id
        == boundary
    )


@pytest.mark.parametrize(
    "text",
    [
        " approve",
        "approve ",
        "approve\n",
        "\napprove",
        '"approve"',
        "The answer is approve",
        "approve\nreject",
        "",
        "APPROVE",
        "foreign",
        "approve\x00",
    ],
)
def test_should_refuse_nonexact_text_without_destroying_raw_return(text):
    raw = generation(text=text)
    with pytest.raises(ValueError, match="exactly one allowed"):
        decode(raw)
    assert raw.text == text


@pytest.mark.parametrize("reason", ["length", "error", None, "eos"])
def test_should_refuse_noncompleted_or_unknown_stop_reason(reason):
    raw = generation(finish_reason=reason)
    with pytest.raises(ValueError):
        decode(raw)
    assert raw.finish_reason == reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("reported_input_tokens", None),
        ("reported_output_tokens", None),
        ("reported_input_tokens", 0),
        ("reported_output_tokens", 0),
    ],
)
def test_should_keep_unknown_or_zero_raw_usage_but_refuse_qualified_selection(field, value):
    raw = generation(**{field: value})
    with pytest.raises(ValueError, match="known positive"):
        decode(raw)
    assert getattr(raw, field) is value


@pytest.mark.parametrize("field", ["reported_input_tokens", "reported_output_tokens"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1", 2**53])
def test_should_refuse_nonexact_or_unsafe_reported_counts(field, value):
    with pytest.raises(ValueError):
        generation(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        dict(max_input_tokens=7),
        dict(max_output_tokens=1),
        dict(max_input_tokens=0),
        dict(max_output_tokens=False),
        dict(max_input_tokens=1.5),
        dict(max_output_tokens=2**53),
    ],
)
def test_should_enforce_explicit_safe_positive_token_limits(changes):
    with pytest.raises(ValueError):
        decode(**changes)


@pytest.mark.parametrize(
    "raw",
    [
        generation(request_id="r2"),
        generation(model=CausalModelReference("Other/model", REVISION)),
        generation(model=CausalModelReference("Qwen/Qwen3.5-2B", "a" * 40)),
    ],
)
def test_should_refuse_request_or_pinned_model_mismatch(raw):
    with pytest.raises(ValueError, match="request or pinned model"):
        decode(raw)


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "g" * 40, True])
def test_should_refuse_moving_or_nonexact_model_revision(revision):
    with pytest.raises(ValueError):
        CausalModelReference("Qwen/Qwen3.5-2B", revision)


@pytest.mark.parametrize("text", ["x" * 4097, "é" * 2049, "\ud800"])
def test_should_bound_raw_utf8_bytes_before_decoding(text):
    with pytest.raises(ValueError):
        generation(text=text)


def test_should_retain_bounded_failed_raw_text_at_exact_byte_limit():
    raw = generation(text="é" * 2048, finish_reason="length", reported_output_tokens=None)
    assert len(raw.text.encode("utf8")) == MAX_RESPONSE_BYTES
    with pytest.raises(ValueError):
        decode(raw)
    assert raw.reported_output_tokens is None and raw.finish_reason == "length"


@pytest.mark.parametrize("fault", ["text", "count", "revision", "model", "request", "candidate"])
def test_should_revalidate_bypassed_frozen_fields_before_acceptance(fault):
    raw, query = generation(), request()
    if fault == "text":
        object.__setattr__(raw, "text", b"approve")
    elif fault == "count":
        object.__setattr__(raw, "reported_input_tokens", True)
    elif fault == "revision":
        object.__setattr__(raw.model, "revision", "main")
    elif fault == "model":
        object.__setattr__(raw, "model", None)
    elif fault == "request":
        object.__setattr__(query, "request_id", "r2")
    else:
        object.__setattr__(query.allowed_candidates[0], "candidate_id", "different")
    with pytest.raises(ValueError):
        decode(raw, query)


def test_should_detach_accepted_output_from_input_record_shells():
    raw = generation()
    result = decode(raw)
    object.__setattr__(raw, "text", "reject")
    object.__setattr__(raw.model, "model_id", "changed")
    assert result.selected_id == result.generation.text == "approve"
    assert result.generation.model == model()
    validate_generated_decision_result(
        request(), result, expected_model=model(), max_input_tokens=8, max_output_tokens=2
    )
    with pytest.raises(FrozenInstanceError):
        setattr(result, "selected_id", "reject")
    with pytest.raises(AttributeError):
        object.__setattr__(result.generation, "foreign", True)


def test_should_refuse_changed_result_or_receiver_request_without_redecoding_choice():
    result, query = decode(), request()
    object.__setattr__(result.generation, "text", "foreign")
    object.__setattr__(result, "selected_id", "foreign")
    with pytest.raises(ValueError):
        validate_generated_decision_result(
            query, result, expected_model=model(), max_input_tokens=8, max_output_tokens=2
        )
    result = decode()
    object.__setattr__(query, "allowed_candidates", (DecisionCandidate("reject", "x"),))
    with pytest.raises(ValueError):
        validate_generated_decision_result(
            query, result, expected_model=model(), max_input_tokens=8, max_output_tokens=2
        )


def test_should_refuse_subclasses_or_wrong_record_objects():
    class ForeignModel(CausalModelReference):
        pass

    wrong: Any = {}
    with pytest.raises(ValueError):
        ForeignModel("Qwen/Qwen3.5-2B", REVISION)
    with pytest.raises(ValueError):
        decode_causal_decision(
            wrong, generation(), expected_model=model(), max_input_tokens=8, max_output_tokens=2
        )
    with pytest.raises(ValueError):
        decode(expected_model={})
    with pytest.raises(ValueError):
        validate_generated_decision_result(
            request(), wrong, expected_model=model(), max_input_tokens=8, max_output_tokens=2
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(selected_id="reject"),
        dict(request_id="other"),
        dict(generation=generation(finish_reason="length")),
        dict(generation=generation(reported_output_tokens=None)),
    ],
)
def test_should_refuse_inconsistent_qualified_result_constructor(changes):
    values: dict[str, Any] = dict(request_id="r1", selected_id="approve", generation=generation())
    values.update(changes)
    with pytest.raises(ValueError):
        GeneratedDecisionResult(**values)
