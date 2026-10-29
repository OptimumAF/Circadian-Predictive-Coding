"""Literal local bridge behavior; no numerical library or model construction."""

import json
from types import SimpleNamespace

import pytest

from src.adapters.causal_choice import causal_messages, observe_causal_choice
from src.core.causal_decision import CausalModelReference, decode_causal_decision
from src.core.typed_decision import DecisionCandidate, DecisionRequest


class Tokens:
    def __init__(self, row, shape=None):
        self.row = row
        self.shape = (1, len(row)) if shape is None else shape
        self.copies = 0

    def tolist(self):
        self.copies += 1
        return [list(self.row)]


class Tokenizer:
    def __init__(self, text="password"):
        self.inputs = {"input_ids": Tokens([2, 3]), "attention_mask": Tokens([1, 1])}
        self.text = text
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.inputs

    def decode(self, ids, **kwargs):
        self.decoded = (ids, kwargs)
        return self.text


class Model:
    training = False

    def __init__(self, row=None):
        self.output = Tokens([2, 3, 4, 9] if row is None else row)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


@pytest.fixture
def case():
    request = DecisionRequest(
        "bridge-1",
        "Login help?",
        (DecisionCandidate("password", "Login"), DecisionCandidate("billing", "Bills")),
    )
    return SimpleNamespace(
        request=request,
        reference=CausalModelReference("Qwen/Qwen3.5-2B", "1" * 40),
        model=Model(),
        tokenizer=Tokenizer(),
    )


def observe(case, **changes):
    kwargs = dict(
        model=case.model,
        tokenizer=case.tokenizer,
        reference=case.reference,
        max_input_tokens=2,
        max_output_tokens=2,
        eos_token_id=9,
        vocab_size=10,
    )
    kwargs.update(changes)
    return observe_causal_choice(case.request, **kwargs)


def test_should_compose_exact_prompt_and_observed_tokens_with_core_decoder(case):
    events = []
    raw = observe(case, before_generate=lambda: events.append("admitted"))
    result = decode_causal_decision(
        case.request, raw, expected_model=case.reference, max_input_tokens=2, max_output_tokens=2
    )
    assert result.selected_id == "password"
    assert raw.reported_input_tokens == raw.reported_output_tokens == 2
    assert events == ["admitted"] and raw.model is not case.reference
    assert case.tokenizer.decoded == (
        [4],
        dict(skip_special_tokens=False, clean_up_tokenization_spaces=False),
    )
    assert case.tokenizer.calls[0][1] == dict(
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )
    kwargs = case.model.calls[0]
    assert kwargs == dict(
        **case.tokenizer.inputs,
        max_new_tokens=2,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        logits_to_keep=1,
        eos_token_id=9,
        pad_token_id=9,
        return_dict_in_generate=False,
        output_scores=False,
        output_logits=False,
    )


def test_should_preserve_candidate_order_and_unicode_without_targets(case):
    request = DecisionRequest("unicode", "密码?", tuple(reversed(case.request.allowed_candidates)))
    messages = causal_messages(request)
    payload = json.loads(messages[1]["content"])
    assert "密码" in messages[1]["content"]
    assert payload == dict(
        request_id="unicode",
        query="密码?",
        allowed_candidates=[
            dict(candidate_id="billing", description="Bills"),
            dict(candidate_id="password", description="Login"),
        ],
    )
    assert messages[0] == dict(
        role="system",
        content="Select the support intent. Reply with exactly one allowed candidate_id and no other text.",
    )


@pytest.mark.parametrize(
    "text",
    ["password ", '"password"', "The answer is password", "<special>password", "", "foreign"],
)
def test_should_retain_raw_failed_text_without_repair(case, text):
    case.tokenizer.text = text
    raw = observe(case)
    assert raw.text == text and raw.finish_reason == "stop"
    with pytest.raises(ValueError):
        decode_causal_decision(
            case.request,
            raw,
            expected_model=case.reference,
            max_input_tokens=2,
            max_output_tokens=2,
        )


@pytest.mark.parametrize(
    ("row", "reason", "count", "content"),
    [
        ([2, 3, 4, 5], "length", 2, [4, 5]),
        ([2, 3, 4], "unobserved", 1, [4]),
        ([2, 3, 9], "stop", 1, []),
    ],
)
def test_should_report_observed_stop_and_usage_without_guessing(case, row, reason, count, content):
    case.model.output = Tokens(row)
    raw = observe(case)
    assert (raw.finish_reason, raw.reported_output_tokens) == (reason, count)
    assert case.tokenizer.decoded[0] == content


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_input_tokens", 0),
        ("max_output_tokens", True),
        ("max_output_tokens", 1.0),
        ("vocab_size", -1),
        ("vocab_size", 262145),
        ("eos_token_id", 10),
        ("eos_token_id", False),
        ("reference", object()),
    ],
)
def test_should_refuse_invalid_configuration_before_tokenization(case, key, value):
    with pytest.raises(ValueError):
        observe(case, **{key: value})
    assert not case.model.calls and not case.tokenizer.calls


def test_should_refuse_training_model_before_tokenization(case):
    case.model.training = True
    with pytest.raises(ValueError, match="eval"):
        observe(case)
    assert not case.tokenizer.calls


def test_should_refuse_bypassed_request_before_tokenization(case):
    object.__setattr__(case.request, "query", "")
    with pytest.raises(ValueError):
        observe(case)
    assert not case.tokenizer.calls


@pytest.mark.parametrize(
    ("key", "row", "shape"),
    [
        ("input_ids", [2, 3, 4], None),
        ("input_ids", [], None),
        ("input_ids", [2], (2, 1)),
        ("input_ids", [True, 3], None),
        ("input_ids", [10, 3], None),
        ("attention_mask", [1, 0], None),
        ("attention_mask", [1], None),
    ],
)
def test_should_refuse_bad_inputs_before_admitting_model_work(case, key, row, shape):
    case.tokenizer.inputs[key] = Tokens(row, shape)
    with pytest.raises(ValueError):
        observe(case)
    assert not case.model.calls
    if shape == (2, 1):
        assert case.tokenizer.inputs[key].copies == 0


def test_should_refuse_multimodal_or_additional_inputs(case):
    case.tokenizer.inputs["pixel_values"] = object()
    with pytest.raises(ValueError, match="text-only"):
        observe(case)
    assert not case.model.calls


def test_should_refuse_work_when_budget_callback_refuses(case):
    def refuse():
        raise RuntimeError("budget spent")

    with pytest.raises(RuntimeError, match="budget spent"):
        observe(case, before_generate=refuse)
    assert len(case.tokenizer.calls) == 1 and not case.model.calls


@pytest.mark.parametrize(
    "row", [[2, 3], [3, 2, 4, 9], [2, 3, 4, 4, 9], [2, 3, 9, 9], [2, 3, 4, True], [2, 3, 4, 10]]
)
def test_should_refuse_invalid_generated_sequence_without_another_call(case, row):
    case.model.output = Tokens(row)
    with pytest.raises(ValueError):
        observe(case)
    assert len(case.model.calls) == 1 and not hasattr(case.tokenizer, "decoded")


def test_should_bound_output_copy_before_detachment(case):
    case.model.output = Tokens([2, 3, 4, 9], (100, 4))
    with pytest.raises(ValueError):
        observe(case)
    assert case.model.output.copies == 0


def test_should_freeze_reference_and_request_binding_before_third_party_work(case):
    def mutate():
        object.__setattr__(case.reference, "model_id", "changed")
        object.__setattr__(case.request, "request_id", "changed")

    raw = observe(case, before_generate=mutate)
    assert raw.model.model_id == "Qwen/Qwen3.5-2B" and raw.request_id == "bridge-1"


def test_should_propagate_model_failure_without_retry(case):
    def fail(**kwargs):
        case.model.calls.append(kwargs)
        raise RuntimeError("native failure")

    case.model.generate = fail
    with pytest.raises(RuntimeError, match="native failure"):
        observe(case)
    assert len(case.model.calls) == 1
