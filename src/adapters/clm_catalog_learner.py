"""Opt-in full-catalog CLM learning through the owned native head engine.

Inputs bind each TRAIN query to a catalog index. Updates share the paired
learner's atomic AdamW/OneCycle implementation, but use a directional objective.
This adapter neither loads an encoder nor supplies data, scoring or promotion.
"""

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from src.adapters.clm_head_learner import CLMHeadLearner, CLMHeadSnapshot
from src.core.clm_catalog import (
    CATALOG_OBJECTIVE,
    CLMCatalogTargets,
    FrozenCatalogTrainingBatch,
)
from src.core.clm_head import _exact

OBJECTIVE = CATALOG_OBJECTIVE


def catalog_head_loss(state_head, action_head, logit_scale, states, actions, indices):
    """Row-mean CE over every catalog candidate; no reverse or masked term."""
    state_z = F.normalize(state_head(states), dim=-1)
    action_z = F.normalize(action_head(actions), dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * state_z @ action_z.t()
    return F.cross_entropy(logits, indices), logits, indices


@dataclass(frozen=True)
class CLMCatalogSnapshot(CLMHeadSnapshot):
    """Same owned state fields, explicitly tagged with the catalog objective."""


def _snapshot_fields(snapshot):
    # Base snapshots already own their tensors; asdict would copy them again.
    return {field.name: getattr(snapshot, field.name) for field in fields(CLMHeadSnapshot)}


class CLMCatalogLearner(CLMHeadLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._work["catalog_candidate_rows_attempted"] = 0
        self._work["catalog_logit_values_attempted"] = 0

    def _catalog_batch(self, batch, targets):
        return self._catalog_embeddings(batch, targets, kind=FrozenCatalogTrainingBatch)

    def _catalog_embeddings(self, batch, targets, *, kind):
        # Explicit record kinds share tensor admission without changing their roles.
        self._check()
        _exact(batch, kind)
        _exact(targets, CLMCatalogTargets)
        if batch.candidate_ids != targets.candidate_ids:
            raise ValueError("catalog target order differs from feature catalog IDs")
        # Check all binding metadata before the shared validator scans/copies tensors.
        for value in (batch.states, batch.actions):
            if (
                type(value) not in (torch.Tensor, nn.Parameter)
                or value.ndim != 2
                or not 2 <= value.shape[0] <= self._policy.max_rows
            ):
                raise ValueError("catalog learning requires bounded NxD and MxD tensors")
        if (
            len(batch.candidate_ids) != batch.actions.shape[0]
            or len(targets.candidate_indices) != batch.states.shape[0]
        ):
            raise ValueError("catalog IDs and target indices must match feature rows")
        states, actions = self._choice_embeddings(batch, kind=kind)
        indices = torch.tensor(targets.candidate_indices, dtype=torch.int64)
        return states, actions, indices

    def _charge_training_inputs(self, states, actions):
        self._work["catalog_candidate_rows_attempted"] += len(actions)
        self._work["catalog_logit_values_attempted"] += len(states) * len(actions)

    def train_batch(self, batch, targets, *, record_gradient_norm=False):
        states, actions, indices = self._catalog_batch(batch, targets)
        return self._train_embeddings(
            states,
            actions,
            indices,
            objective=OBJECTIVE,
            loss_fn=catalog_head_loss,
            record_gradient_norm=record_gradient_norm,
        )

    def snapshot_state(self):
        return CLMCatalogSnapshot(**_snapshot_fields(super().snapshot_state()))

    def restore_state(self, snapshot):
        if type(snapshot) is not CLMCatalogSnapshot or set(vars(snapshot)) != {
            field.name for field in fields(CLMCatalogSnapshot)
        }:
            raise ValueError("expected exact complete CLMCatalogSnapshot")
        super().restore_state(CLMHeadSnapshot(**_snapshot_fields(snapshot)))
