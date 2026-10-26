"""TRAIN-role catalog learner with an explicit state-only optimization scope.

Accepts the ordinary frozen catalog TRAIN records and owns the native update,
readout and restore lifecycle. Action-head values and temperature remain pinned
to this learner's birth. This module does not load features, promote probes,
select examples or change the default all-head catalog learner.
"""

from copy import deepcopy
from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from src.adapters.clm_catalog_learner import (
    CLMCatalogLearner,
    OBJECTIVE,
    _snapshot_fields,
)
from src.adapters.clm_head_learner import (
    CLMHeadLearner,
    CLMHeadSnapshot,
    _equal,
    _params,
    _tensor,
)


def state_only_catalog_head_loss(state_head, action_head, logit_scale, states, actions, indices):
    """Same numeric catalog CE, with gradients restricted to state parameters."""
    state_z = F.normalize(state_head(states), dim=-1)
    # Why: retaining requires_grad declarations avoids silently freezing inputs;
    # the forward and optimizer explicitly define which parameters may change.
    with torch.no_grad():
        action_z = F.normalize(action_head(actions), dim=-1)
    logits = logit_scale.detach().exp().clamp(max=100.0) * state_z @ action_z.t()
    return F.cross_entropy(logits, indices), logits, indices


@dataclass(frozen=True)
class CLMStateOnlyCatalogSnapshot(CLMHeadSnapshot):
    """Owned native state tagged with catalog objective and state-only scope."""


class CLMStateOnlyCatalogLearner(CLMCatalogLearner):
    """Update state parameters while preserving the owned action/scale birth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only the copied owners are cleared: callers retain their gradients.
        for parameter in (*self._bundle[1].parameters(), self._bundle[2]):
            parameter.grad = None
        # A single bounded pin survives updates and restores; snapshots do not
        # redefine the fixed target representation of the receiving learner.
        self._fixed_birth = (
            deepcopy(self._bundle[1].state_dict()),
            _tensor(self._bundle[2], shape=()),
        )

    def _optimizer_parameters(self, bundle):
        return list(bundle[0].parameters())

    def _validate_update_parameters(self, bundle):
        optimized = self._optimizer_parameters(bundle)
        actual = [parameter for group in bundle[3].param_groups for parameter in group["params"]]
        if (
            len(actual) != len(optimized)
            or any(left is not right for left, right in zip(actual, optimized, strict=True))
            or bundle[4].optimizer is not bundle[3]
            or any(not any(key is parameter for parameter in optimized) for key in bundle[3].state)
        ):
            raise ValueError(
                "state-only catalog optimizer contains unoptimized parameter state or owners"
            )
        if (
            not _equal(bundle[1].state_dict(), self._fixed_birth[0])
            or not _equal(bundle[2].detach(), self._fixed_birth[1])
            or any(
                parameter.grad is not None or not parameter.requires_grad
                for parameter in (*bundle[1].parameters(), bundle[2])
            )
        ):
            raise ValueError("state-only catalog fixed action parameters or scale changed")

    def _check(self):
        super()._check()
        self._validate_update_parameters(self._bundle)

    @property
    def parameter_scope(self):
        self._check()
        return "state_only"

    @property
    def parameter_counts(self):
        self._check()
        total = sum(parameter.numel() for parameter in _params(self._bundle))
        optimized = sum(parameter.numel() for parameter in self._optimizer_parameters(self._bundle))
        return {"optimized": optimized, "fixed": total - optimized, "total": total}

    @property
    def fixed_parameters_unchanged(self):
        self._check()
        return True

    def train_batch(self, batch, targets, *, record_gradient_norm=False):
        states, actions, indices = self._catalog_batch(batch, targets)
        return self._train_embeddings(
            states,
            actions,
            indices,
            objective=OBJECTIVE,
            loss_fn=state_only_catalog_head_loss,
            record_gradient_norm=record_gradient_norm,
        )

    def snapshot_state(self):
        owned = CLMHeadLearner.snapshot_state(self)
        return CLMStateOnlyCatalogSnapshot(**_snapshot_fields(owned))

    def restore_state(self, snapshot):
        if type(snapshot) is not CLMStateOnlyCatalogSnapshot or set(vars(snapshot)) != {
            field.name for field in fields(CLMStateOnlyCatalogSnapshot)
        }:
            raise ValueError("expected exact complete CLMStateOnlyCatalogSnapshot")
        # Why: the base engine preserves physical charges and validates staged
        # fixed values before publication; the all-head catalog wrapper has a
        # deliberately different exact snapshot family.
        CLMHeadLearner.restore_state(self, CLMHeadSnapshot(**_snapshot_fields(snapshot)))
