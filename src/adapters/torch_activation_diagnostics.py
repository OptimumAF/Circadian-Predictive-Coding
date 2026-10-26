"""Read native hidden activations from owned snapshots and feature rows.

Inputs are exact supported CPU snapshots and their bound float64 feature arrays.
Outputs are detached float64 hidden rows. No labels, roles, learner construction,
updates, clocks, RNG changes or IO belong here; callers admit TRAIN data separately.
"""

import numpy as np

from src.adapters import torch_feature_learner as feature
from src.adapters import torch_recurrent_learner as recurrent
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureSnapshot,
    copy_invariant_torch_feature_snapshot,
)
from src.core.action_memory import _feature_keys
from src.core.coupled_energy import Array
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch


def _feature_activations(owned: feature.TorchFeatureSnapshot, features: Array) -> Array:
    values = feature._features(features, owned.policy.feature_dim)
    torch = require_torch()
    with torch.no_grad():
        linear = (
            values @ owned.head_state["weight_feature_hidden"] + owned.head_state["bias_hidden"]
        )
        if not bool(torch.isfinite(linear).all().item()):
            raise FloatingPointError("nonfinite feedforward hidden preactivation")
        hidden = torch.tanh(linear)
        if not bool(torch.isfinite(hidden).all().item()):
            raise FloatingPointError("nonfinite feedforward hidden activation")
        return hidden.detach().cpu().numpy().astype(np.float64, copy=True)


def _invariant_features(features: Array) -> Array:
    # Why: validate/project every row before evaluating hidden priors, exactly
    # as the bound learner does; a malformed later row cannot be skipped.
    rows = _feature_keys(features, 296)
    return np.array([canonicalize_public_history(row) for row in rows], dtype=np.float64)


def read_native_activations(snapshot: object, features: Array) -> Array:
    """Return actual float32 hidden states as an owned BxH float64 array.

    Feedforward values are tanh priors, including for PC/CPC, rather than
    label-conditioned relaxed states. Recurrent values start at zero for each
    complete episode and follow its row order. The existing recurrent forward
    also computes one discarded output projection per row; this is diagnostic
    work, not a free hidden-only pass.

    Encoding-bound feature snapshots apply their explicit canonical projection
    to raw public296 rows before the same native float32 prior calculation.
    """
    if type(snapshot) is InvariantTorchFeatureSnapshot:
        bound = copy_invariant_torch_feature_snapshot(snapshot)
        return _feature_activations(bound.native_state, _invariant_features(features))
    if type(snapshot) is feature.TorchFeatureSnapshot:
        owned = feature._snapshot(snapshot)
        return _feature_activations(owned, features)
    if type(snapshot) is recurrent.TorchRecurrentSnapshot:
        owned_recurrent = recurrent._snapshot(snapshot)
        values = recurrent._features(features, owned_recurrent.policy)
        torch = require_torch()
        with torch.no_grad():
            hidden = torch.zeros(
                (1, owned_recurrent.policy.hidden_dim), dtype=torch.float32, device="cpu"
            )
            rows = []
            for row in values:
                # Why: single-row reuse exposes every ordered hidden state
                # without duplicating or approximating native recurrence math.
                logits, hidden = recurrent._forward(
                    owned_recurrent.parameters, row.reshape(1, -1), hidden, check_finite=True
                )
                if not bool(torch.isfinite(logits).all().item()) or not bool(
                    torch.isfinite(hidden).all().item()
                ):
                    raise FloatingPointError("nonfinite recurrent diagnostic output")
                rows.append(hidden)
            return torch.cat(rows, dim=0).detach().cpu().numpy().astype(np.float64, copy=True)
    raise ValueError(
        "native activations require an exact feature, bound feature or recurrent snapshot"
    )
