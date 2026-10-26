"""Native CLM projection heads and typed, insertion-order choice decoding.

Owns small detached frozen heads, not an encoder copy or a projection cache.
Uses the plain TRAIN/HF encoder recipe; no vLLM parity, calibration, training,
promotion or physical provenance is asserted by a typed prediction.

make_clm_head derives from Contrastive-LM/CLM src/clm/heads.py at revision
d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6, under Apache-2.0. Modifications add
argument validation and the local typed wrapper. Original license and source
attribution are retained in THIRD_PARTY_NOTICES.md.
"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
from src.core.clm_head import _exact
from src.core.clm_action_features import FrozenActionFeatures
from src.core.typed_decision import (
    DecisionCandidate,
    DecisionRequest,
    ProbabilityDecisionResult,
    _text,
    validate_probability_decision_result,
)


def make_clm_head(
    width: int,
    depth: int = 2,
    proj: int = 512,
    activation: str = "gelu",
    layernorm: bool = False,
    residual: bool = False,
    hidden: int = 4096,
):
    """Pinned make_head architecture; honors the caller's CPU or meta device."""
    for name, value in (("width", width), ("depth", depth), ("proj", proj), ("hidden", hidden)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive native size")
    if type(layernorm) is not bool or type(residual) is not bool:
        raise ValueError("head options require native booleans")
    if type(activation) is not str or activation not in ("gelu", "relu", "silu"):
        raise ValueError("unsupported native CLM activation")
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList(
                (nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2)
            )
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


def _geometry(head, input_dim):
    supported = (
        nn.Linear,
        nn.Sequential,
        nn.ModuleList,
        nn.LayerNorm,
        nn.Identity,
        nn.GELU,
        nn.ReLU,
        nn.SiLU,
    )
    if not isinstance(head, nn.Module):
        raise ValueError("expected a deterministic native projection module")
    for module in head.modules():
        if type(module) not in supported and not (module is head and type(head).__name__ == "Head"):
            raise ValueError("unsupported stochastic or nonnative projection module")
        if tuple(module.buffers(recurse=False)):
            raise ValueError("projection buffers are not supported")
    linears = [module for module in head.modules() if type(module) is nn.Linear]
    if not linears:
        if type(head) is not nn.Identity:
            raise ValueError("only Identity may be a parameter-free fixture head")
        return input_dim
    if linears[0].in_features != input_dim:
        raise ValueError("head input width differs from frozen embedding identity")
    return linears[-1].out_features


def _signature(head):
    return tuple(
        (name, parameter, parameter.untyped_storage().data_ptr(), parameter._version)
        for name, parameter in head.named_parameters()
    )


def _action_feature_pin(identity, supplied):
    _exact(supplied, FrozenActionFeatures)
    value = supplied.features
    if (
        supplied.identity != identity
        or type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or value.requires_grad
        or tuple(value.shape) != (len(supplied.candidates), identity.embedding_dim)
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.untyped_storage().nbytes() != value.numel() * value.element_size()
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(
            "raw action features require matching encoder and bounded finite CPUfloat32MxD storage"
        )
    return (
        supplied,
        supplied.identity,
        supplied.candidates,
        value,
        value.untyped_storage().data_ptr(),
        value._version,
        deepcopy((supplied.identity, supplied.candidates)),
    )


def _require_action_feature_pin(identity, supplied, before):
    now = _action_feature_pin(identity, supplied)
    if any(now[index] is not before[index] for index in range(4)) or now[4:] != before[4:]:
        raise ValueError("original frozen action metadata, storage or version changed")


class CLMDecisionPredictor:
    def __init__(
        self,
        encoder,
        state_head,
        action_head,
        logit_scale,
        *,
        instructions,
        head_revision,
        action_features=None,
    ):
        if type(encoder) is not FrozenCLMEncoder:
            raise ValueError("expected exact FrozenCLMEncoder")
        cache_pin = (
            None
            if action_features is None
            else _action_feature_pin(encoder.identity, action_features)
        )
        _text(instructions, "instructions", 2048)
        _text(head_revision, "head_revision", 128)
        identity = encoder.identity
        projection = _geometry(state_head, identity.embedding_dim)
        if _geometry(action_head, identity.embedding_dim) != projection:
            raise ValueError("state/action projection widths differ")
        if identity.embedding_dim == 4096 and projection != 512:
            raise ValueError("released native geometry requires4096->512")
        parameters = [*state_head.parameters(), *action_head.parameters()]
        if sum(parameter.numel() for parameter in parameters) + 1 > 32000000:
            raise ValueError("projection parameter allowance exceeded")
        if len({p.untyped_storage().data_ptr() for p in parameters}) != len(parameters):
            raise ValueError("head parameters cannot share storage")
        for parameter in parameters:
            if (
                parameter.device.type != "cpu"
                or parameter.dtype != torch.float32
                or not torch.isfinite(parameter).all().item()
            ):
                raise ValueError("projection parameters require finite CPU float32")
        if (
            type(logit_scale) not in (torch.Tensor, nn.Parameter)
            or logit_scale.device.type != "cpu"
            or logit_scale.dtype != torch.float32
            or logit_scale.shape != ()
            or not torch.isfinite(logit_scale).item()
        ):
            raise ValueError("logit_scale requires a finite CPU float32 scalar")
        self._encoder = encoder
        self._state_head, self._action_head = deepcopy(state_head), deepcopy(action_head)
        for head in (self._state_head, self._action_head):
            head.eval()
            head.requires_grad_(False)
        self._scale = logit_scale.detach().clone()
        self._projection = projection
        self._instructions, self._revision = instructions, head_revision
        self._birth = (
            encoder,
            self._state_head,
            self._action_head,
            self._scale,
            identity,
            instructions,
            head_revision,
            projection,
        )
        self._signatures = (_signature(self._state_head), _signature(self._action_head))
        self._scale_version = self._scale._version
        self._scale_pointer = self._scale.untyped_storage().data_ptr()
        self._scale_value = self._scale.clone()
        self._failed = False
        self._last_readout = None
        self._work = dict(
            request_attempts=0,
            request_returns=0,
            encoder_attempts=0,
            encoder_returns=0,
            head_forward_attempts=0,
            head_forward_returns=0,
            projected_rows=0,
            readout_attempts=0,
            readout_returns=0,
            unreturned_work_unknown=False,
        )
        if action_features is not None:
            self._action_features = action_features
            self._action_pin = cache_pin
            self._work.update(
                action_feature_copy_attempts=1,
                action_feature_copy_returns=0,
                action_feature_elements_copied=0,
                action_feature_read_attempts=0,
                action_feature_read_returns=0,
                action_feature_rows_used=0,
                action_feature_guard_attempts=0,
                action_feature_guard_returns=0,
                action_feature_guard_values_checked=0,
            )
            self._action_values = action_features.features.detach().clone()
            self._action_values_birth = (
                self._action_values,
                self._action_values.untyped_storage().data_ptr(),
                self._action_values._version,
            )
            self._work["action_feature_copy_returns"] = 1
            self._work["action_feature_elements_copied"] = self._action_values.numel()
            self._require_action_values()

    def _require_action_values(self):
        if not hasattr(self, "_action_features"):
            return
        self._work["action_feature_guard_attempts"] += 1
        try:
            _require_action_feature_pin(
                self._encoder.identity, self._action_features, self._action_pin
            )
            if (
                self._action_values is not self._action_values_birth[0]
                or self._action_values.untyped_storage().data_ptr() != self._action_values_birth[1]
                or self._action_values._version != self._action_values_birth[2]
                or self._action_values.requires_grad
                or not torch.equal(self._action_features.features, self._action_values)
            ):
                raise ValueError("owned frozen raw action features changed")
            self._work["action_feature_guard_returns"] += 1
            self._work["action_feature_guard_values_checked"] += self._action_values.numel()
        except Exception:
            self._failed = True
            raise

    @property
    def work(self):
        return dict(self._work)

    @property
    def last_readout(self):
        return deepcopy(self._last_readout)

    def _check(self):
        if self._failed:
            raise ValueError("CLM predictor stopped after uncertain native work")
        if (
            self._encoder is not self._birth[0]
            or self._state_head is not self._birth[1]
            or self._action_head is not self._birth[2]
            or self._scale is not self._birth[3]
            or self._encoder.identity != self._birth[4]
            or (self._instructions, self._revision, self._projection) != self._birth[5:]
            or self._scale._version != self._scale_version
            or self._scale.untyped_storage().data_ptr() != self._scale_pointer
            or not torch.equal(self._scale, self._scale_value)
        ):
            raise ValueError("CLM predictor frozen ownership or recipe changed")
        for head, signature in zip(
            (self._state_head, self._action_head), self._signatures, strict=True
        ):
            current = _signature(head)
            if len(current) != len(signature) or any(module.training for module in head.modules()):
                raise ValueError("CLM projection owner or mode changed")
            for now, before in zip(current, signature, strict=True):
                if (
                    now[0] != before[0]
                    or now[1] is not before[1]
                    or now[2:] != before[2:]
                    or now[1].requires_grad
                ):
                    raise ValueError("CLM projection parameter owner, value or freeze changed")
        try:
            self._encoder._check()
        except Exception:
            if hasattr(self, "_action_features"):
                self._failed = True
            raise
        self._require_action_values()

    def __call__(self, request):
        _exact(request, DecisionRequest)
        for candidate in request.allowed_candidates:
            _exact(candidate, DecisionCandidate)
        self._check()
        cached_actions = hasattr(self, "_action_features")
        if cached_actions and request.allowed_candidates != self._action_features.candidates:
            raise ValueError(
                "request candidate IDs, descriptions or order differ from frozen actions"
            )
        owned = deepcopy(request)
        state_text = owned.query.strip() + "\n\n" + self._instructions.strip()
        descriptions = tuple(candidate.description for candidate in owned.allowed_candidates)
        ids = tuple(candidate.candidate_id for candidate in owned.allowed_candidates)
        self._last_readout = dict(
            request_id=owned.request_id,
            candidate_ids=ids,
            state_text=state_text,
            head_revision=self._revision,
            logits=None,
            probabilities=None,
            confidence=None,
        )
        self._work["request_attempts"] += 1
        self._work["unreturned_work_unknown"] = True
        try:
            self._work["encoder_attempts"] += 1
            if cached_actions:
                features = self._encoder.embed_texts((state_text,), ("tail",))
            else:
                features = self._encoder.embed_texts(
                    (state_text,) + descriptions, ("tail",) + ("head",) * len(ids)
                )
            self._work["encoder_returns"] += 1
            self._last_readout["encoder_work"] = self._encoder.work
            self._check()
            actions = features[1:]
            if cached_actions:
                self._work["action_feature_read_attempts"] += 1
                actions = self._action_values
                self._work["action_feature_rows_used"] += len(ids)
                self._work["action_feature_read_returns"] += 1
                self._last_readout["action_features"] = "frozen_raw_catalog"
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                projections = []
                for head, rows in (
                    (self._state_head, features[:1]),
                    (self._action_head, actions),
                ):
                    self._work["head_forward_attempts"] += 1
                    projected = head(rows)
                    self._work["head_forward_returns"] += 1
                    if (
                        type(projected) is not torch.Tensor
                        or projected.device.type != "cpu"
                        or projected.dtype != torch.float32
                        or projected.shape != (len(rows), self._projection)
                        or not torch.isfinite(projected).all().item()
                    ):
                        raise ValueError("head returned incompatible or nonfinite projections")
                    self._work["projected_rows"] += len(rows)
                    projections.append(F.normalize(projected, dim=-1))
                    self._check()
                self._work["readout_attempts"] += 1
                logits = (self._scale.exp().clamp(max=100.0) * projections[0] @ projections[1].t())[
                    0
                ]
                values = [float(value) for value in logits]
                self._last_readout["logits"] = values
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("CLM logits are nonfinite")
                # Native schema uses stable scalar softmax and insertion-order ties.
                peak = max(values)
                exponentials = [math.exp(value - peak) for value in values]
                total = sum(exponentials)
                probabilities = [value / total for value in exponentials]
                winner = max(range(len(ids)), key=probabilities.__getitem__)
                rest = [value for i, value in enumerate(probabilities) if i != winner]
                confidence = (
                    max(0.0, min(1.0, probabilities[winner] - sum(rest) / len(rest)))
                    if rest
                    else 1.0
                )
                self._last_readout.update(probabilities=probabilities, confidence=confidence)
                self._work["readout_returns"] += 1
            if request != owned:
                raise ValueError("prediction changed the original typed request")
            self._check()
            result = ProbabilityDecisionResult(
                owned.request_id,
                ids[winner],
                tuple(zip(ids, probabilities, strict=True)),
                reported_confidence=confidence,
            )
            validate_probability_decision_result(owned, result)
            self._work["request_returns"] += 1
            self._work["unreturned_work_unknown"] = False
            return result
        except Exception:
            self._failed = True
            raise
