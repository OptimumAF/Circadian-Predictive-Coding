"""Disposable warm-weight catalog CLM probes with independent native charges.

Reuses catalog loss and the existing owned probe initialization. The public
surface offers no snapshot, restore, export, fork or installation authority.
Private native objects remain trusted Python, not code containment.
"""

from copy import deepcopy

import torch
from torch.nn import functional as F

from src.adapters.clm_catalog_learner import CLMCatalogLearner, OBJECTIVE
from src.adapters.clm_head_learner import _equal, _params, _tensor
from src.adapters.clm_head_probe import CLMHeadProbe, _admit_probe_policy
from src.adapters.clm_state_only_catalog_learner import CLMStateOnlyCatalogLearner
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_probe import FrozenProbeChoiceBatch


class _ProbeCatalogLearner(CLMCatalogLearner):
    def _catalog_batch(self, batch, targets):
        return self._catalog_embeddings(batch, targets, kind=FrozenProbeCatalogBatch)

    def _choice_batch(self, batch):
        return self._choice_embeddings(batch, kind=FrozenProbeChoiceBatch)


def _state_only_catalog_loss(state_head, action_head, logit_scale, states, actions, indices):
    state_z = F.normalize(state_head(states), dim=-1)
    # Why: keep the same numeric objective while preventing action/scale updates.
    with torch.no_grad():
        action_z = F.normalize(action_head(actions), dim=-1)
    logits = logit_scale.detach().exp().clamp(max=100.0) * state_z @ action_z.t()
    return F.cross_entropy(logits, indices), logits, indices


class _StateOnlyProbeCatalogLearner(_ProbeCatalogLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A warm source may retain gradients; the disposable fixed owners do not.
        for parameter in (*self._bundle[1].parameters(), self._bundle[2]):
            parameter.grad = None
        self._fixed_birth = (
            deepcopy(self._bundle[1].state_dict()),
            _tensor(self._bundle[2], shape=()),
        )

    def _optimizer_parameters(self, bundle):
        return list(bundle[0].parameters())

    def _validate_update_parameters(self, bundle):
        optimized = self._optimizer_parameters(bundle)
        actual = [p for group in bundle[3].param_groups for p in group["params"]]
        if (
            len(actual) != len(optimized)
            or any(left is not right for left, right in zip(actual, optimized, strict=True))
            or bundle[4].optimizer is not bundle[3]
            or any(not any(key is parameter for parameter in optimized) for key in bundle[3].state)
        ):
            raise ValueError(
                "state-only probe optimizer contains unoptimized parameter state or owners"
            )
        if (
            not _equal(bundle[1].state_dict(), self._fixed_birth[0])
            or not _equal(bundle[2].detach(), self._fixed_birth[1])
            or any(
                parameter.grad is not None or not parameter.requires_grad
                for parameter in (*bundle[1].parameters(), bundle[2])
            )
        ):
            raise ValueError("state-only probe fixed action parameters or scale changed")

    def _check(self):
        super()._check()
        self._validate_update_parameters(self._bundle)

    def train_batch(self, batch, targets, *, record_gradient_norm=False):
        states, actions, indices = self._catalog_batch(batch, targets)
        return self._train_embeddings(
            states,
            actions,
            indices,
            objective=OBJECTIVE,
            loss_fn=_state_only_catalog_loss,
            record_gradient_norm=record_gradient_norm,
        )


class CLMCatalogProbe(CLMHeadProbe):
    def __init__(self, source, *, policy, max_readouts, parameter_scope="all"):
        if type(parameter_scope) is not str or parameter_scope not in ("all", "state_only"):
            raise ValueError("catalog probe parameter_scope must be all or state_only")
        if type(source) not in (CLMCatalogLearner, CLMStateOnlyCatalogLearner):
            raise ValueError("a catalog probe requires an exact catalog TRAIN learner source")
        _admit_probe_policy(source, policy, max_readouts)
        learner_kind = (
            _ProbeCatalogLearner if parameter_scope == "all" else _StateOnlyProbeCatalogLearner
        )
        self._initialize(
            source, policy=policy, max_readouts=max_readouts, learner_kind=learner_kind
        )
        self._parameter_scope = parameter_scope
        self._scope_birth = (parameter_scope, learner_kind)

    def _check(self):
        super()._check()
        if (
            type(self._parameter_scope) is not str
            or self._parameter_scope != self._scope_birth[0]
            or type(self._learner) is not self._scope_birth[1]
        ):
            raise ValueError("catalog probe parameter scope or learner kind changed")

    @property
    def parameter_scope(self):
        self._check()
        return self._parameter_scope

    @property
    def parameter_counts(self):
        self._check()
        total = sum(parameter.numel() for parameter in _params(self._learner._bundle))
        optimized = sum(
            parameter.numel()
            for parameter in self._learner._optimizer_parameters(self._learner._bundle)
        )
        return {"optimized": optimized, "fixed": total - optimized, "total": total}

    @property
    def fixed_parameters_unchanged(self):
        self._check()
        return True
