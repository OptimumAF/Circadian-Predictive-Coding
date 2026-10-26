"""Synchronous frozen CPU encoder with the pinned plain-text CLM recipe.

The loader transfers an already frozen model; this adapter never clones its
weights, downloads data, trains, caches projections or attests source provenance.
Tokenizer/model Python code is trusted, not contained by these checks.
"""

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from src.core.clm_head import FrozenEmbeddingBatch, FrozenEncoderIdentity, _exact


def text_recipe(max_tokens: int) -> str:
    return (
        f"clm-plain-state-tail-action-head-v1:max_len={max_tokens}:"
        f"cap={max_tokens - 1}:l2-plus-1e-12"
    )


def _bound(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a native integer in1..{maximum}")


def _model_signature(model):
    return tuple(
        (name, parameter, parameter.untyped_storage().data_ptr(), parameter._version)
        for name, parameter in model.named_parameters()
    )


class FrozenCLMEncoder:
    def __init__(
        self, model, tokenizer, *, identity, max_tokens=2048, max_sequences=16, max_forward_calls=1
    ):
        _exact(identity, FrozenEncoderIdentity)
        _bound(max_tokens, "max_tokens", 2048)
        if max_tokens < 2:
            raise ValueError("max_tokens requires room for the pinned max_tokens-1 cap")
        _bound(max_sequences, "max_sequences", 16)
        _bound(max_forward_calls, "max_forward_calls", 128)
        if identity.pooling != "last" or identity.recipe != text_recipe(max_tokens):
            raise ValueError("encoder identity must bind last pooling and the exact text recipe")
        if not isinstance(model, nn.Module) or not callable(tokenizer):
            raise ValueError("expected a native frozen module and supplied tokenizer")
        if getattr(getattr(model, "config", None), "hidden_size", None) != identity.embedding_dim:
            raise ValueError("backbone declared hidden size differs from encoder identity")
        self._model, self._tokenizer = model, tokenizer
        self._identity = deepcopy(identity)
        self._limits = (max_tokens, max_sequences, max_forward_calls)
        self._birth = (model, tokenizer, self._identity, deepcopy(identity), self._limits)
        self._signature = _model_signature(model)
        self._buffers = tuple(
            (name, value, value.untyped_storage().data_ptr(), value._version)
            for name, value in model.named_buffers()
        )
        self._failed = False
        self._work: dict[str, int | bool] = dict(
            embedding_attempts=0,
            embedding_returns=0,
            tokenization_attempts=0,
            tokenization_returns=0,
            token_ids_visited=0,
            retained_token_ids=0,
            forward_attempts=0,
            forward_returns=0,
            sequences_attempted=0,
            sequences_returned=0,
            padded_tokens_attempted=0,
            pooled_values=0,
            unreturned_work_unknown=False,
        )
        self._last_readout: dict[str, Any] | None = None
        self._check()

    @property
    def identity(self):
        return deepcopy(self._identity)

    @property
    def work(self):
        return dict(self._work)

    @property
    def last_readout(self):
        return deepcopy(self._last_readout)

    def _check(self):
        if self._failed:
            raise ValueError("frozen encoder stopped after uncertain native work")
        if (
            self._model is not self._birth[0]
            or self._tokenizer is not self._birth[1]
            or self._identity is not self._birth[2]
            or self._identity != self._birth[3]
            or self._limits != self._birth[4]
        ):
            raise ValueError("frozen encoder owner, identity or limits changed")
        if any(module.training is not False for module in self._model.modules()):
            raise ValueError("backbone must remain in eval mode")
        if (
            getattr(getattr(self._model, "config", None), "hidden_size", None)
            != self._identity.embedding_dim
        ):
            raise ValueError("backbone hidden-size binding changed")
        now = _model_signature(self._model)
        if len(now) != len(self._signature):
            raise ValueError("backbone parameter schema changed")
        for current, original in zip(now, self._signature, strict=True):
            name, parameter, pointer, version = current
            if (
                name != original[0]
                or parameter is not original[1]
                or pointer != original[2]
                or version != original[3]
                or parameter.device.type != "cpu"
                or parameter.requires_grad
            ):
                raise ValueError("backbone parameter owner, value version or freeze changed")
        buffers = tuple(self._model.named_buffers())
        if len(buffers) != len(self._buffers):
            raise ValueError("backbone buffer schema changed")
        for (name, value), original in zip(buffers, self._buffers, strict=True):
            if (
                name != original[0]
                or value is not original[1]
                or value.untyped_storage().data_ptr() != original[2]
                or value._version != original[3]
                or value.device.type != "cpu"
            ):
                raise ValueError("frozen backbone buffer owner or version changed")

    def _ids(self, text):
        self._work["tokenization_attempts"] += 1
        with torch.random.fork_rng(devices=[]):
            result = self._tokenizer(text, add_special_tokens=False)
        self._work["tokenization_returns"] += 1
        ids = result["input_ids"]
        if type(ids) is list and ids and type(ids[0]) is list:
            if len(ids) != 1:
                raise ValueError("tokenizer returned multiple sequences for one string")
            ids = ids[0]
        if type(ids) is not list or len(ids) > 65536:
            raise ValueError("tokenizer requires a bounded native token-ID list")
        self._work["token_ids_visited"] += len(ids)
        if any(type(value) is not int or not 0 <= value < 2**31 for value in ids):
            raise ValueError("tokenizer IDs must be nonnegative bounded native integers")
        return list(ids)

    def embed_texts(self, texts, keeps):
        self._check()
        if (
            type(texts) is not tuple
            or type(keeps) is not tuple
            or not 1 <= len(texts) <= self._limits[1]
            or len(texts) != len(keeps)
            or any(type(text) is not str or len(text.encode("utf8")) > 16384 for text in texts)
            or any(type(keep) is not str or keep not in ("head", "tail") for keep in keeps)
        ):
            raise ValueError("expected bounded tuple text/keep pairs")
        if self._work["forward_attempts"] >= self._limits[2]:
            raise ValueError("frozen encoder forward allowance exhausted")
        self._work["embedding_attempts"] += 1
        self._work["sequences_attempted"] += len(texts)
        self._work["unreturned_work_unknown"] = True
        self._last_readout = dict(
            keeps=keeps,
            original_token_lengths=[],
            token_ids=[],
            retained_lengths=[],
            forward_returned=False,
        )
        try:
            rows = []
            for text, keep in zip(texts, keeps, strict=True):
                self._check()
                ids = self._ids(text)
                original_length = len(ids)
                if not ids:
                    ids = self._ids(" ")
                self._check()
                if not ids:
                    raise ValueError("blank fallback produced no token IDs")
                retained = (
                    ids[: self._limits[0] - 1] if keep == "head" else ids[-(self._limits[0] - 1) :]
                )
                rows.append(retained)
                self._work["retained_token_ids"] += len(retained)
                self._last_readout["original_token_lengths"].append(original_length)
                self._last_readout["token_ids"].append(tuple(retained))
                self._last_readout["retained_lengths"].append(len(retained))
            padding = getattr(self._tokenizer, "pad_token_id", None)
            if padding is None:
                padding = getattr(self._tokenizer, "eos_token_id", 0)
            if type(padding) is not int or not 0 <= padding < 2**31:
                raise ValueError("tokenizer requires a native padding or EOS token ID")
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), padding, dtype=torch.int64)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(rows):
                ids[i, : len(row)] = torch.tensor(row, dtype=torch.int64)
                mask[i, : len(row)] = 1
            self._last_readout["padded_tokens"] = ids.numel()
            self._last_readout["attention_mask"] = mask.tolist()
            self._last_readout["padded_token_ids"] = ids.tolist()
            self._check()
            expected_ids, expected_mask = ids.clone(), mask.clone()
            self._work["forward_attempts"] += 1
            self._work["padded_tokens_attempted"] += ids.numel()
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                output = self._model(
                    input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
                )
            self._work["forward_returns"] += 1
            self._last_readout["forward_returned"] = True
            self._check()
            if not torch.equal(ids, expected_ids) or not torch.equal(mask, expected_mask):
                raise ValueError("backbone changed its original token IDs or attention mask")
            hidden = output.last_hidden_state
            if (
                type(hidden) is not torch.Tensor
                or hidden.device.type != "cpu"
                or hidden.dtype not in (torch.float32, torch.bfloat16, torch.float16)
                or hidden.shape != (len(rows), width, self._identity.embedding_dim)
            ):
                raise ValueError("backbone returned incompatible last hidden state")
            last = hidden[torch.arange(len(rows)), mask.sum(dim=1) - 1].float().detach().clone()
            self._last_readout["last_token_states"] = last.clone()
            if not torch.isfinite(last).all().item():
                raise ValueError("backbone last-token features are nonfinite")
            features = last / (last.norm(dim=-1, keepdim=True) + 1e-12)
            if not torch.isfinite(features).all().item():
                raise ValueError("normalized encoder features are nonfinite")
            self._last_readout["embeddings"] = features.clone()
            self._work["pooled_values"] += features.numel()
            self._check()
            self._work["embedding_returns"] += 1
            self._work["sequences_returned"] += len(rows)
            self._work["unreturned_work_unknown"] = False
            return features.clone()
        except Exception:
            self._failed = True
            raise

    def training_batch(self, states, actions, *, role="train"):
        if type(role) is not str or role != "train":
            raise ValueError("paired encoder features admit TRAIN only")
        if (
            type(states) is not tuple
            or type(actions) is not tuple
            or not 2 <= len(states) <= 8
            or len(states) != len(actions)
        ):
            raise ValueError("training features require2..8 paired text strings")
        features = self.embed_texts(
            states + actions, ("tail",) * len(states) + ("head",) * len(actions)
        )
        return FrozenEmbeddingBatch(
            self.identity, features[: len(states)].clone(), features[len(states) :].clone(), role
        )
