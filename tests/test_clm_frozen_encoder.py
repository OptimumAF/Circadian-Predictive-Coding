"""Fake-only checks for the pinned plain-text CLM boundary; no real encoder."""

import ast
from copy import deepcopy
from dataclasses import replace
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
from torch import nn

from src.adapters.clm_decision import CLMDecisionPredictor, make_clm_head
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
from src.core.clm_head import FrozenEncoderIdentity
from src.core.typed_decision import DecisionCandidate, DecisionRequest

ROOT = Path(__file__).resolve().parents[1]
CACHED = ROOT / "artifacts/runs/native-decision-contracts-20261022/clm"


def definition(path, name, namespace):
    """Compile only the named cached definition, never its upstream imports."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    node = next(item for item in tree.body if getattr(item, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def identity(max_tokens=4):
    return FrozenEncoderIdentity(
        "fake-encoder",
        "fixture-revision",
        "fixture-tokenizer",
        "last",
        f"clm-plain-state-tail-action-head-v1:max_len={max_tokens}:cap={max_tokens - 1}:l2-plus-1e-12",
        2,
    )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []
        self.fault = None

    def __call__(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        if self.fault:
            return self.fault(text)
        ids = self.mapping.get(text)
        if ids is None:
            ids = [int(word) for word in text.split()] if text.strip() else []
        if text == " ":
            ids = [7]
        return {"input_ids": list(ids)}


def backbone(torch):
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
            self.config = SimpleNamespace(hidden_size=2)
            self.calls = []
            self.fault = None

        def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
            self.calls.append(
                {
                    "ids": input_ids.clone(),
                    "mask": attention_mask.clone(),
                    "use_cache": use_cache,
                    "return_dict": return_dict,
                    "grad_enabled": torch.is_grad_enabled(),
                }
            )
            self.last_input_ids = input_ids
            hidden = torch.stack((input_ids.float(), torch.ones_like(input_ids).float()), dim=-1)
            if self.fault:
                hidden = self.fault(hidden)
            return SimpleNamespace(last_hidden_state=hidden)

    return FakeBackbone().eval()


def encoder(torch, *, mapping=None, forwards=1, rows=16):
    model, tokenizer = backbone(torch), FakeTokenizer(mapping)
    owner = FrozenCLMEncoder(
        model,
        tokenizer,
        identity=identity(),
        max_tokens=4,
        max_sequences=rows,
        max_forward_calls=forwards,
    )
    return owner, model, tokenizer


def request(*, reverse=False, tied=False):
    candidates = (
        DecisionCandidate("z", "same" if tied else "first"),
        DecisionCandidate("a", "same" if tied else "second"),
    )
    return DecisionRequest("fixture-request", "query", candidates[::-1] if reverse else candidates)


def test_should_match_cached_plain_recipe_last_nonpadding_and_l2(torch):
    owner, model, tokenizer = encoder(torch)
    texts, keeps = ("1 2 3 4 5", "1 2 3 4 5", ""), ("head", "tail", "head")
    caller_rng = torch.get_rng_state().clone()
    result = owner.embed_texts(texts, keeps)
    Recipe = definition(CACHED / "train__embed_utils.py", "Recipe", {})
    reference = Recipe.__new__(Recipe)
    reference.tok, reference.cap = FakeTokenizer(), 3
    ids = [reference.text_ids(text, keep) for text, keep in zip(texts, keeps)]
    assert ids == [[1, 2, 3], [3, 4, 5], [7]]
    call = model.calls[0]
    assert call["ids"].tolist() == [[1, 2, 3], [3, 4, 5], [7, 0, 0]]
    assert call["mask"].tolist() == [[1, 1, 1], [1, 1, 1], [1, 0, 0]]
    expected = [
        [value / (sqrt(value * value + 1) + 1e-12), 1 / (sqrt(value * value + 1) + 1e-12)]
        for value in (3, 5, 7)
    ]
    torch.testing.assert_close(result, torch.tensor(expected), rtol=0, atol=1e-7)
    assert (
        result.dtype == torch.float32 and result.device.type == "cpu" and not result.requires_grad
    )
    assert (
        call["grad_enabled"] is False and call["use_cache"] is False and call["return_dict"] is True
    )
    assert tokenizer.calls == [(texts[0], False), (texts[1], False), ("", False), (" ", False)]
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert not model.training and model.anchor.item() == 0
    assert owner.work == {
        "embedding_attempts": 1,
        "embedding_returns": 1,
        "tokenization_attempts": 4,
        "tokenization_returns": 4,
        "token_ids_visited": 11,
        "retained_token_ids": 7,
        "forward_attempts": 1,
        "forward_returns": 1,
        "sequences_attempted": 3,
        "sequences_returned": 3,
        "padded_tokens_attempted": 9,
        "pooled_values": 6,
        "unreturned_work_unknown": False,
    }
    result.zero_()
    assert call["ids"][0, -1].item() == 3


def test_should_pair_train_rows_in_one_forward_without_reinterpreting_actions(torch):
    owner, model, _ = encoder(torch)
    batch = owner.training_batch(("1 2 3 4", "2"), ("3 4 5 6", "4"))
    assert batch.role == "train" and batch.identity == identity()
    assert batch.states.shape == batch.actions.shape == (2, 2)
    assert model.calls[0]["ids"].tolist() == [[2, 3, 4], [2, 0, 0], [3, 4, 5], [4, 0, 0]]
    batch.states.zero_()
    assert bool(batch.actions.abs().sum() > 0)


@pytest.mark.parametrize("role", ["probe_train", "final", "outer_selection"])
def test_should_refuse_nontrain_before_tokenization(torch, role):
    owner, model, tokenizer = encoder(torch)
    before = owner.work
    with pytest.raises(ValueError):
        owner.training_batch(("1", "2"), ("3", "4"), role=role)
    assert owner.work == before and not tokenizer.calls and not model.calls


@pytest.mark.parametrize(
    "texts,keeps",
    [
        (["1"], ("head",)),
        (("1",), ["head"]),
        ((), ()),
        (("1",), ("middle",)),
        (("1",), ()),
        ((1,), ("head",)),
    ],
)
def test_should_admit_entire_text_batch_before_tokenization(torch, texts, keeps):
    owner, model, tokenizer = encoder(torch)
    before = owner.work
    with pytest.raises(ValueError):
        owner.embed_texts(texts, keeps)
    assert owner.work == before and not tokenizer.calls and not model.calls


def test_should_not_renew_paid_forward_allowance_through_work_copy(torch):
    owner, model, _ = encoder(torch)
    owner.embed_texts(("1",), ("head",))
    paid = deepcopy(owner.work)
    detached = owner.work
    detached.clear()
    with pytest.raises(ValueError):
        owner.embed_texts(("2",), ("tail",))
    assert owner.work == paid and len(model.calls) == 1


@pytest.mark.parametrize("fault", ["tokenizer", "forward", "nonfinite", "parameter", "input"])
def test_should_preserve_paid_failure_and_refuse_retry(torch, fault):
    owner, model, tokenizer = encoder(torch, forwards=2)
    caller_rng = torch.get_rng_state().clone()
    if fault == "tokenizer":

        def fail_tokenizer(text):
            raise RuntimeError("tokenizer lost return")

        tokenizer.fault = fail_tokenizer
    else:

        def fail_native(hidden):
            if fault == "forward":
                raise RuntimeError("forward lost return")
            if fault == "nonfinite":
                return hidden * float("nan")
            if fault == "parameter":
                model.anchor.add_(1)
            if fault == "input":
                model.last_input_ids.add_(1)
            return hidden

        model.fault = fail_native
    try:
        with pytest.raises((ValueError, RuntimeError)):
            owner.embed_texts(("1",), ("head",))
        paid = deepcopy(owner.work)
        assert paid["embedding_attempts"] == 1 and paid["embedding_returns"] == 0
        assert paid["tokenization_attempts"] == 1
        assert paid["tokenization_returns"] == (0 if fault == "tokenizer" else 1)
        assert paid["forward_attempts"] == (0 if fault == "tokenizer" else 1)
        assert paid["forward_returns"] == (0 if fault in ("tokenizer", "forward") else 1)
        assert paid["unreturned_work_unknown"] is True
        token_calls, native_calls = len(tokenizer.calls), len(model.calls)
        with pytest.raises((ValueError, RuntimeError)):
            owner.embed_texts(("2",), ("head",))
        assert owner.work == paid
        assert len(tokenizer.calls) == token_calls and len(model.calls) == native_calls
    finally:
        torch.set_rng_state(caller_rng)


@pytest.mark.parametrize("mutation", ["training", "requires_grad", "recipe"])
def test_should_refuse_unfrozen_or_mismatched_encoder_birth(torch, mutation):
    model, tokenizer, declared = backbone(torch), FakeTokenizer(), identity()
    if mutation == "training":
        model.train()
    elif mutation == "requires_grad":
        model.anchor.requires_grad_(True)
    else:
        declared = replace(declared, recipe="different")
    with pytest.raises(ValueError):
        FrozenCLMEncoder(model, tokenizer, identity=declared, max_tokens=4)
    assert not tokenizer.calls and not model.calls


@pytest.fixture(scope="module")
def decision_case(torch):
    owner, model, tokenizer = encoder(
        torch,
        mapping={"query\n\nChoose.": [1], "first": [1], "second": [2]},
        forwards=2,
    )
    source_state, source_action = torch.nn.Identity(), torch.nn.Identity()
    scale = torch.tensor(0.0)
    predictor = CLMDecisionPredictor(
        owner,
        source_state,
        source_action,
        scale,
        instructions="Choose.",
        head_revision="fixture-head",
    )
    rng = torch.get_rng_state().clone()
    first = predictor(request())
    second = predictor(request(reverse=True))
    return (
        predictor,
        owner,
        model,
        tokenizer,
        source_state,
        source_action,
        scale,
        rng,
        first,
        second,
    )


def test_should_match_native_probability_decode_and_preserve_order(torch, decision_case):
    predictor, _, model, tokenizer, _, _, _, rng, first, second = decision_case
    logits = [1.0, 3 / sqrt(10)]
    import math

    namespace: dict[str, Any] = {"math": math}
    for name in ("softmax", "confidence", "answer_from_probs", "answer_from_logits"):
        definition(CACHED / "src__clm__schema.py", name, namespace)
    native = namespace["answer_from_logits"]({"type": "choice"}, ["z", "a"], logits)
    assert first.selected_id == native["choice"] == "z"
    assert dict(first.probabilities) == pytest.approx(native["probabilities"], abs=1e-7)
    assert first.reported_confidence == pytest.approx(native["confidence"], abs=1e-7)
    assert [key for key, _ in first.probabilities] == ["z", "a"]
    assert [key for key, _ in second.probabilities] == ["a", "z"]
    assert dict(first.probabilities) == pytest.approx(dict(second.probabilities), abs=1e-7)
    assert first.reported_input_tokens is None and first.reported_output_tokens is None
    assert tokenizer.calls == [
        ("query\n\nChoose.", False),
        ("first", False),
        ("second", False),
        ("query\n\nChoose.", False),
        ("second", False),
        ("first", False),
    ]
    assert len(model.calls) == 2 and torch.equal(rng, torch.get_rng_state())
    assert predictor.last_readout is not None


def test_should_detach_head_sources_work_and_single_readout_slot(decision_case):
    predictor, _, _, _, source_state, source_action, scale, _, _, _ = decision_case
    assert source_state.training and source_action.training and scale.item() == 0
    assert predictor._state_head is not source_state and predictor._action_head is not source_action
    assert not predictor._state_head.training and not predictor._action_head.training
    before_work, before_readout = deepcopy(predictor.work), deepcopy(predictor.last_readout)
    predictor.work.clear()
    predictor.last_readout.clear()
    assert predictor.work == before_work and predictor.last_readout == before_readout
    assert len(before_readout) > 0


def test_should_refuse_corrupted_request_before_encoding(torch, decision_case):
    predictor, owner, model, tokenizer, *_ = decision_case
    malformed = request()
    object.__setattr__(malformed.allowed_candidates[0], "description", "")
    before = (
        deepcopy(owner.work),
        deepcopy(predictor.work),
        len(model.calls),
        len(tokenizer.calls),
    )
    with pytest.raises(ValueError):
        predictor(malformed)
    assert (owner.work, predictor.work, len(model.calls), len(tokenizer.calls)) == before


def test_should_keep_native_first_insertion_ties_in_both_orders(torch):
    owner, _, _ = encoder(
        torch,
        mapping={"query\n\nChoose.": [1], "same": [1]},
        forwards=2,
    )
    predictor = CLMDecisionPredictor(
        owner,
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.tensor(0.0),
        instructions="Choose.",
        head_revision="fixture-ties",
    )
    first, reversed_result = (
        predictor(request(tied=True)),
        predictor(request(tied=True, reverse=True)),
    )
    assert first.selected_id == "z" and reversed_result.selected_id == "a"
    assert first.probabilities == (("z", 0.5), ("a", 0.5))
    assert first.reported_confidence == reversed_result.reported_confidence == 0


@pytest.fixture(scope="module")
def head_case(torch):
    native_factory = definition(
        CACHED / "src__clm__heads.py", "make_head", {"PROJ_DIM": 2, "HIDDEN": 2}
    )
    caller_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng():
        torch.manual_seed(23)
        actual = make_clm_head(width=3, depth=2, proj=2, hidden=2)
        torch.manual_seed(23)
        reference = native_factory(width=3, depth=2, proj=2, hidden=2)
    assert torch.equal(caller_rng, torch.get_rng_state())
    return actual, reference


def test_should_match_cached_head_architecture_without_touching_caller_rng(torch, head_case):
    actual, reference = head_case
    caller_rng = torch.get_rng_state().clone()
    assert list(actual.state_dict()) == list(reference.state_dict())
    for key, value in actual.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key])
    sample = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    with torch.no_grad():
        torch.testing.assert_close(actual(sample), reference(sample), rtol=0, atol=0)
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert sum(value.numel() for value in actual.parameters()) == 17


@pytest.mark.parametrize("location", ["forward", "tokenizer", "tokenizer_raise"])
def test_should_restore_caller_rng_after_opaque_code_rng_use(torch, location):
    owner, model, _ = encoder(torch)
    caller_rng = torch.get_rng_state().clone()

    def use_private_native_rng(hidden):
        torch.set_rng_state(torch.Generator().manual_seed(99).get_state())
        return hidden

    if location == "forward":
        model.fault = use_private_native_rng
    else:

        def use_tokenizer_rng(text):
            torch.set_rng_state(torch.Generator().manual_seed(99).get_state())
            if location == "tokenizer_raise":
                raise RuntimeError("tokenizer failed after RNG change")
            return {"input_ids": [1]}

        owner._tokenizer.fault = use_tokenizer_rng
    if location == "tokenizer_raise":
        with pytest.raises(RuntimeError):
            owner.embed_texts(("1",), ("head",))
    else:
        owner.embed_texts(("1",), ("head",))
        assert owner.work["embedding_returns"] == 1
    assert torch.equal(caller_rng, torch.get_rng_state())


def test_should_keep_paid_encoder_and_first_head_prefix_when_second_head_fails(torch):
    owner, model, _ = encoder(
        torch,
        mapping={"query\n\nChoose.": [1], "first": [1], "second": [2]},
        forwards=2,
    )
    predictor = CLMDecisionPredictor(
        owner,
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.tensor(0.0),
        instructions="Choose.",
        head_revision="fixture-paid-head",
    )

    def fail_head(module, args, output):
        raise RuntimeError("action head lost return")

    predictor._action_head.register_forward_hook(fail_head)
    with pytest.raises(RuntimeError, match="lost return"):
        predictor(request())
    paid = predictor.work
    assert paid["encoder_attempts"] == paid["encoder_returns"] == 1
    assert paid["head_forward_attempts"] == 2 and paid["head_forward_returns"] == 1
    assert paid["request_attempts"] == 1 and paid["request_returns"] == 0
    assert paid["unreturned_work_unknown"] is True
    assert predictor.last_readout["encoder_work"] == owner.work
    assert predictor.last_readout["logits"] is None
    with pytest.raises(ValueError):
        predictor(request())
    assert predictor.work == paid and len(model.calls) == 1


def test_should_own_parameter_heads_without_mutating_or_aliasing_sources(torch, head_case):
    actual, reference = head_case
    sources = deepcopy(actual.state_dict()), deepcopy(reference.state_dict())
    owner, _, _ = encoder(
        torch,
        mapping={"query\n\nChoose.": [1], "first": [1], "second": [2]},
    )
    predictor = CLMDecisionPredictor(
        owner,
        actual,
        reference,
        torch.tensor(0.0),
        instructions="Choose.",
        head_revision="fixture-parameter-heads",
    )
    # Caller mutations remain outside the owned predictor; this is not source attestation.
    for source, owned in ((actual, predictor._state_head), (reference, predictor._action_head)):
        for first, copied in zip(source.parameters(), owned.parameters()):
            assert first is not copied and first.data_ptr() != copied.data_ptr()
            assert torch.equal(first, copied) and not copied.requires_grad
    result = predictor(request())
    assert result.request_id == request().request_id
    for source, before in zip((actual, reference), sources):
        for name, value in source.state_dict().items():
            assert torch.equal(value, before[name])
    parameter = next(actual.parameters())
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.add_(1.0)
        assert not torch.equal(parameter, next(predictor._state_head.parameters()))
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def test_should_refuse_missing_released_assets_before_loading_any_native_model(tmp_path):
    from src.infra.clm_local_loader import load_local_clm

    def forbidden_factory(**kwargs):
        raise AssertionError("missing assets must not construct a projection")

    with pytest.raises(ValueError, match="missing or wrong-size"):
        load_local_clm(
            tmp_path / "missing", tmp_path / "missing.pt", head_factory=forbidden_factory
        )


def test_should_refuse_same_size_replaced_local_asset_before_model_loading(tmp_path, monkeypatch):
    import hashlib
    from src.infra import clm_local_loader as loader

    # This one-byte IO fixture is never represented as a released model payload.
    asset, head = tmp_path / "config.json", tmp_path / "head.pt"
    asset.write_bytes(b"a")
    head.write_bytes(b"h")
    monkeypatch.setattr(
        loader,
        "ENCODER_FILES",
        {
            "config.json": dict(
                bytes=1, kind="git_blob_sha1", digest=hashlib.sha1(b"blob 1\0a").hexdigest()
            )
        },
    )
    monkeypatch.setattr(loader, "HEAD_BYTES", 1)
    monkeypatch.setattr(loader, "HEAD_SHA256", hashlib.sha256(b"h").hexdigest())
    loader.validate_local_clm(tmp_path, head)
    asset.write_bytes(b"b")
    with pytest.raises(ValueError, match="digest mismatch"):
        loader.validate_local_clm(tmp_path, head)


def test_should_retain_transformers_set_loading_information_as_json_safe_facts():
    import json
    from src.infra.clm_local_loader import _plain_loading_info

    native = dict(
        missing_keys=set(), unexpected_keys={"lm_head.weight"}, mismatched_keys=set(), error_msgs=[]
    )
    result = _plain_loading_info(native)
    assert json.loads(json.dumps(result)) == dict(
        missing_keys=[], unexpected_keys=["lm_head.weight"], mismatched_keys=[], error_msgs=[]
    )
    assert native["unexpected_keys"] == {"lm_head.weight"}
