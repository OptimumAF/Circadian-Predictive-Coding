"""Fixed union-catalog linear TRAIN learner for the existing temporal inbox.

One cold solve followed by warm residual solves; active candidate IDs explicitly
map to persistent union columns. No native head, final/probe inputs, replay store,
optimizer, snapshot/history chain or scientific allowance. Readout bindings are
declared once and bounded to eight; they carry no labels into training.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import torch

from src.adapters.clm_linear_fit_control import (
    CLMLinearFitControl,
    fit_clm_linear_control,
    _capture,
    _features,
    _require_original,
    _tensor,
)
from src.adapters.clm_linear_update import update_clm_linear_control
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import FrozenChoiceBatch, FrozenEncoderIdentity, _exact
from src.core.clm_linear_catalog import CLMLinearCatalogPolicy, OBJECTIVE, linear_catalog_work
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.temporal_decision_sequence import CATALOG, policy_for_phase
from src.core.wide_categorical_linear_fit import predict_wide_categorical_linear


class CLMLinearCatalogLearner:
    def __init__(
        self,
        identity,
        actions,
        choice_bindings,
        *,
        policy=CLMLinearCatalogPolicy(),
        candidate_ids=None,
    ):
        _exact(identity, FrozenEncoderIdentity)
        _exact(policy, CLMLinearCatalogPolicy)
        explicit = candidate_ids is not None
        if explicit and (
            type(candidate_ids) is not tuple
            or not 2 <= len(candidate_ids) <= 8
            or any(type(key) is not str or not key or key.strip() != key for key in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or choice_bindings != ()
            or type(choice_bindings) is not tuple
        ):
            raise ValueError(
                "explicit TRAIN catalog requires2..8 unique IDs and no choice bindings"
            )
        _tensor(actions, len(candidate_ids) if explicit else 4, identity.embedding_dim)
        if not explicit and (
            type(choice_bindings) is not tuple or not 1 <= len(choice_bindings) <= 8
        ):
            raise ValueError("linear catalog requires1..8 original explicit choice bindings")
        self.identity, self.policy = deepcopy(identity), deepcopy(policy)
        self._identity_owner, self._policy_owner = self.identity, self.policy
        self._identity_values, self._policy_values = asdict(self.identity), asdict(self.policy)
        self._actions = actions.detach().clone()
        self._action_owner = self._actions
        self._action_values = self._actions.clone()
        self._action_trait = (self._actions._version, self._actions.data_ptr())
        self._ids = candidate_ids if explicit else tuple(c.candidate_id for c in CATALOG)
        self._catalog_birth = (self._ids, explicit)
        self._explicit_catalog = explicit
        bindings: list[tuple[Any, Any, Any]] = []
        for batch, ids in choice_bindings:
            self._admit_choice(batch, ids)
            if any(batch is old[0] for old in bindings):
                raise ValueError("duplicate linear choice binding")
            bindings.append((batch, ids, _capture(batch)))
        self._bindings = tuple(bindings)
        self._control: CLMLinearFitControl | None = None
        self._control_owner: CLMLinearFitControl | None = None
        self._model_owner: object | None = None
        self._control_values: dict[str, Any] | None = None
        self._work = linear_catalog_work()
        self._last: dict[str, Any] = {}

    @property
    def completed_updates(self):
        return self._work["commits"]

    @property
    def training_examples(self):
        return self._work["training_rows_attempted"]

    @property
    def generation(self):
        return self.completed_updates

    @property
    def work(self):
        return self._work.copy()

    @property
    def last_operation(self):
        return deepcopy(self._last)

    @property
    def current_control(self):
        """Borrow the immutable committed control; no export or model history."""
        self._ready()
        if self._control is None:
            raise ValueError("linear catalog has no committed TRAIN control")
        return self._control

    def _ready(self):
        if self._work["unreturned_work_unknown"]:
            raise ValueError("linear catalog owner stopped after uncertain work; no retry")
        if (
            self._ids is not self._catalog_birth[0]
            or self._explicit_catalog is not self._catalog_birth[1]
            or self.identity is not self._identity_owner
            or self.policy is not self._policy_owner
            or asdict(self.identity) != self._identity_values
            or asdict(self.policy) != self._policy_values
            or self._actions is not self._action_owner
            or (self._actions._version, self._actions.data_ptr()) != self._action_trait
            or not torch.equal(self._actions, self._action_values)
            or self._control is not self._control_owner
        ):
            raise ValueError("original linear catalog owner or values changed")
        if self._control is not None and (
            self._control.model is not self._model_owner
            or asdict(self._control) != self._control_values
        ):
            raise ValueError("original linear catalog coefficients changed")

    def _active(self, actions, ids):
        # Why: generic workflows serve through the typed provider. Their TRAIN
        # batches bind the entire union, while temporal choices retain three IDs.
        if self._explicit_catalog:
            if (
                type(ids) is not tuple
                or ids != self._ids
                or not torch.equal(actions, self._actions)
            ):
                raise ValueError("explicit TRAIN requires the original full union catalog")
            return list(range(len(ids)))
        if (
            type(ids) is not tuple
            or len(ids) != 3
            or len(set(ids)) != 3
            or any(key not in self._ids for key in ids)
        ):
            raise ValueError("linear temporal readout requires three explicit union catalog IDs")
        indices = [self._ids.index(key) for key in ids]
        if not torch.equal(actions, self._actions[indices]):
            raise ValueError("active action features differ from original union catalog")
        return indices

    def _admit_choice(self, batch, ids):
        _exact(batch, FrozenChoiceBatch)
        if batch.identity != self.identity or batch.role not in ("train", "outer_selection"):
            raise ValueError("linear choice requires bound TRAIN/development identity and role")
        _features(batch, ids, min_rows=1)
        self._active(batch.actions, ids)

    def train_batch(self, batch, targets):
        self._ready()
        _exact(batch, FrozenCatalogTrainingBatch)
        _exact(targets, CLMCatalogTargets)
        _features(batch, batch.candidate_ids)
        if (
            batch.identity != self.identity
            or targets.candidate_ids != batch.candidate_ids
            or len(targets.candidate_indices) != len(batch.states)
        ):
            raise ValueError("linear TRAIN identity or ordered target binding differs")
        indices = self._active(batch.actions, batch.candidate_ids)
        if (
            self.completed_updates >= self.policy.max_updates
            or len(batch.states) > self.policy.max_rows
        ):
            raise ValueError("linear lifetime solve or batch row allowance exhausted")
        pin = _capture(batch, targets)
        global_batch = FrozenCatalogTrainingBatch(
            deepcopy(self.identity), batch.states.detach().clone(), self._actions.clone(), self._ids
        )
        global_targets = CLMCatalogTargets(
            self._ids, tuple(indices[i] for i in targets.candidate_indices)
        )
        self._last = {}
        self._work["train_attempts"] += 1
        self._work["solve_attempts"] += 1
        self._work["training_rows_attempted"] += len(batch.states)
        key = "linear_fit" if self._control is None else "linear_update"
        try:
            control = (
                fit_clm_linear_control(global_batch, global_targets, self._last)
                if self._control is None
                else update_clm_linear_control(
                    self._control, global_batch, global_targets, self._last
                )
            )
            self._work["solve_returns"] += 1
            _require_original(batch, targets, pin)
            self._ready()
            _exact(control, CLMLinearFitControl)
            if (
                control.role != "train"
                or control.candidate_ids != self._ids
                or control.identity != self.identity
            ):
                raise ValueError("returned linear catalog control differs")
            diagnostic = TrainingDiagnostic(
                OBJECTIVE, control.model.squared_error / len(batch.states)
            )
            self._control = self._control_owner = control
            self._model_owner = control.model
            self._control_values = asdict(control)
            self._work["commits"] += 1
            return diagnostic
        except Exception as error:
            self._work["solve_returns"] = max(
                self._work["solve_returns"],
                self.completed_updates + int(self._last.get(key, {}).get("returns", 0)),
            )
            self._work["unreturned_work_unknown"] = True
            raise TrainingUpdateFailed(str(error), OBJECTIVE, self.work) from error

    def predict_choices(self, batch):
        self._ready()
        binding = next((b for b in self._bindings if b[0] is batch), None)
        if binding is None or self._work["prediction_attempts"] >= self.policy.max_readouts:
            raise ValueError("unbound linear choice or lifetime readout allowance exhausted")
        _, ids, pin = binding
        self._admit_choice(batch, ids)
        _require_original(batch, None, pin)
        indices = self._active(batch.actions, ids)
        self._work["prediction_attempts"] += 1
        self._work["choice_prediction_attempts"] += 1
        self._work["prediction_rows_attempted"] += len(batch.states)
        self._work["choice_candidate_rows_attempted"] += len(ids)
        try:
            if self._control is None:
                probabilities = np.full((len(batch.states), 3), 1 / 3)
            else:
                # Slice logits BEFORE softmax: an inactive extreme logit cannot
                # underflow every active probability during renormalization.
                model = replace(
                    self._control.model,
                    num_classes=3,
                    coefficients=tuple(
                        tuple(row[i] for i in indices) for row in self._control.model.coefficients
                    ),
                )
                probabilities = predict_wide_categorical_linear(
                    model, batch.states.detach().numpy().astype(np.float64)
                )
            _require_original(batch, None, pin)
            self._ready()
            result = torch.tensor(probabilities, dtype=torch.float32)
            self._work["prediction_returns"] += 1
            self._work["choice_prediction_returns"] += 1
            return result
        except Exception:
            self._work["unreturned_work_unknown"] = True
            raise


def run_linear_catalog_stream(
    identity,
    actions,
    train,
    choices,
    report,
    *,
    train_choices=None,
    before_step=lambda: None,
    save=lambda: None,
):
    """Compose the existing twelve-packet stream; no new scheduler or scoring."""
    from src.app.clm_temporal_sequence import run_temporal_stream

    bindings = tuple(
        (batch, policy_for_phase(phase)["allowed_ids"]) for phase, batch in choices.items()
    )
    if train_choices is not None:
        bindings += tuple(
            (batch, policy_for_phase(phase)["allowed_ids"])
            for phase, batch in train_choices.items()
        )
    learner = CLMLinearCatalogLearner(identity, actions, bindings)
    try:
        return run_temporal_stream(
            learner,
            train,
            choices,
            report,
            before_step=before_step,
            save=save,
            catalog_training=True,
            linear_catalog=True,
            train_choice_batches=train_choices,
        )
    finally:
        report["linear_last_operation"] = learner.last_operation
