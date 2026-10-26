"""Owned CPU CLM head updates over explicitly bound frozen embeddings.

Implements the pinned paired group-masked CLM loss, AdamW, gradient clipping and
OneCycle schedule. Snapshots are local native state, not a durable checkpoint or
encoder attestation. Head Python code is trusted; this is not code containment.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from src.core.clm_head import (
    CLMHeadPolicy,
    CLMTargets,
    FrozenChoiceBatch,
    FrozenEmbeddingBatch,
    FrozenEncoderIdentity,
    _exact,
)
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.clm_training_observation import CLMGradientObservation, CLMObservedUpdate

OBJECTIVE = "clm_paired_group_masked_symmetric_cross_entropy_v1"


def clm_head_loss(state_head, action_head, logit_scale, states, actions, codes):
    """Exact native CLM objective; same-group off-diagonal negatives are masked."""
    state_z = F.normalize(state_head(states), dim=-1)
    action_z = F.normalize(action_head(actions), dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * state_z @ action_z.t()
    labels = torch.arange(len(states), device=states.device)
    same = codes.unsqueeze(0) == codes.unsqueeze(1)
    same.fill_diagonal_(False)
    forward = F.cross_entropy(logits.masked_fill(same, float("-inf")), labels)
    backward = F.cross_entropy(logits.t().masked_fill(same, float("-inf")), labels)
    return (forward + backward) / 2, logits, labels


def _tensor(value, *, shape=None):
    if (
        type(value) not in (torch.Tensor, nn.Parameter)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or (shape is not None and tuple(value.shape) != tuple(shape))
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError("expected a finite CPU float32 tensor with declared shape")
    return value.detach().clone()


def _params(bundle):
    return [*bundle[0].parameters(), *bundle[1].parameters(), bundle[2]]


def _owners(bundle):
    return (bundle, *bundle, *bundle[0].modules(), *bundle[1].modules(), *_params(bundle))


def _copy_bundle(bundle):
    copied = deepcopy(bundle)
    # Parameter deepcopy omits .grad; gradients are part of native continuation.
    for source, target in zip(_params(bundle), _params(copied), strict=True):
        target.grad = None if source.grad is None else source.grad.detach().clone()
    return copied


def _modes(bundle):
    return tuple(tuple(module.training for module in head.modules()) for head in bundle[:2])


def _set_modes(bundle, modes):
    if type(modes) is not tuple or len(modes) != 2:
        raise ValueError("expected both head mode vectors")
    for head, flags in zip(bundle[:2], modes, strict=True):
        modules = tuple(head.modules())
        if (
            type(flags) is not tuple
            or len(flags) != len(modules)
            or any(type(flag) is not bool for flag in flags)
        ):
            raise ValueError("head mode vector differs from the admitted modules")
        for module, flag in zip(modules, flags, strict=True):
            module.training = flag


def _head(head, *, validate_tensors=True):
    if not isinstance(head, nn.Module):
        raise ValueError("CLM heads must be native Torch modules")
    supported = (
        nn.Linear,
        nn.ModuleList,
        nn.Sequential,
        nn.LayerNorm,
        nn.Identity,
        nn.GELU,
        nn.ReLU,
        nn.SiLU,
    )
    for module in head.modules():
        # The pinned make_head root is a local Head class; its deterministic
        # native children are explicit. Arbitrary Python forward code is trusted.
        if type(module) not in supported and not (module is head and type(head).__name__ == "Head"):
            raise ValueError("unsupported head module; deterministic native heads required")
        if type(module.training) is not bool or tuple(module.buffers(recurse=False)):
            raise ValueError("head buffers or malformed modes are unsupported")
    parameters = tuple(head.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("heads require trainable native parameters")
    if validate_tensors:
        for parameter in parameters:
            _tensor(parameter)
            if parameter.grad is not None:
                _tensor(parameter.grad, shape=parameter.shape)


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or not torch.isfinite(value).all().item():
            raise ValueError("snapshot contains nonfinite or non-CPU tensor state")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_tree(item)
    elif type(value) is float:
        import math

        if not math.isfinite(value):
            raise ValueError("snapshot contains nonfinite scalar state")
    elif value is not None and type(value) not in (int, bool, str):
        raise ValueError("snapshot contains unsupported native state")


def _equal(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        return (
            first.dtype == second.dtype
            and first.device == second.device
            and first.shape == second.shape
            and torch.equal(first, second)
        )
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _equal(a, b) for a, b in zip(first, second, strict=True)
        )
    return first == second


@dataclass(frozen=True)
class CLMHeadSnapshot:
    identity: FrozenEncoderIdentity
    policy: CLMHeadPolicy
    state_head: dict[str, Any]
    action_head: dict[str, Any]
    logit_scale: Any
    gradients: tuple[Any, ...]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    modes: tuple[tuple[bool, ...], ...]
    cpu_rng: Any
    completed_updates: int
    training_examples: int


class CLMHeadLearner:
    def __init__(self, state_head, action_head, logit_scale, *, identity, policy):
        _exact(identity, FrozenEncoderIdentity)
        _exact(policy, CLMHeadPolicy)
        _head(state_head, validate_tensors=False)
        _head(action_head, validate_tensors=False)
        scale = _tensor(logit_scale, shape=())
        params = [*state_head.parameters(), *action_head.parameters()]
        for head in (state_head, action_head):
            linears = [module for module in head.modules() if type(module) is nn.Linear]
            if (
                not linears
                or linears[0].in_features != identity.embedding_dim
                or linears[-1].out_features != policy.projection_dim
            ):
                raise ValueError("head input/output width differs from the fixed identity/policy")
        if len({p.untyped_storage().data_ptr() for p in params}) != len(params):
            raise ValueError("CLM heads cannot share parameter owners")
        if sum(p.numel() for p in params) + 1 > policy.max_parameters:
            raise ValueError("CLM head parameter allowance exceeded")
        _head(state_head)
        _head(action_head)
        state, action = deepcopy(state_head), deepcopy(action_head)
        owned_scale = nn.Parameter(scale)
        for original, copied in zip(
            params, [*state.parameters(), *action.parameters()], strict=True
        ):
            copied.grad = None if original.grad is None else original.grad.detach().clone()
        owned_scale.grad = None if logit_scale.grad is None else _tensor(logit_scale.grad, shape=())
        owned_params = self._optimizer_parameters((state, action, owned_scale))
        optimizer = torch.optim.AdamW(
            owned_params, lr=policy.learning_rate, weight_decay=policy.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=policy.learning_rate,
            total_steps=policy.max_updates,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self._bundle = (state, action, owned_scale, optimizer, scheduler)
        self._owners = _owners(self._bundle)
        self._storage = tuple(p.untyped_storage().data_ptr() for p in _params(self._bundle))
        self._identity, self._policy = deepcopy(identity), deepcopy(policy)
        self._declarations = (self._identity, self._policy, deepcopy(identity), deepcopy(policy))
        self._rng = torch.get_rng_state().clone()
        self._completed_updates = self._training_examples = self._generation = 0
        self._failed = False
        self._work = dict.fromkeys(
            (
                "train_attempts",
                "training_rows_attempted",
                "loss_forward_attempts",
                "loss_forward_returns",
                "prediction_attempts",
                "prediction_returns",
                "prediction_rows_attempted",
                "choice_prediction_attempts",
                "choice_prediction_returns",
                "choice_candidate_rows_attempted",
                "backward_attempts",
                "backward_returns",
                "optimizer_attempts",
                "optimizer_returns",
                "scheduler_attempts",
                "scheduler_returns",
                "commits",
                "snapshot_attempts",
                "snapshot_returns",
                "restore_attempts",
                "restore_returns",
            ),
            0,
        )
        self._work["unreturned_work_unknown"] = False

    @property
    def completed_updates(self):
        return self._completed_updates

    @property
    def training_examples(self):
        return self._training_examples

    @property
    def generation(self):
        return self._generation

    @property
    def identity(self):
        return deepcopy(self._identity)

    @property
    def policy(self):
        return deepcopy(self._policy)

    @property
    def work(self):
        return dict(self._work)

    def _optimizer_parameters(self, bundle):
        """Select optimized owners; ordinary learners retain every parameter."""
        return _params(bundle)

    def _validate_update_parameters(self, bundle):
        """Objective-specific staged constraints before native publication."""

    def _check(self):
        if self._failed:
            raise ValueError("CLM learner stopped after uncertain work")
        if (
            self._identity is not self._declarations[0]
            or self._policy is not self._declarations[1]
            or self._identity != self._declarations[2]
            or self._policy != self._declarations[3]
        ):
            raise ValueError("CLM encoder identity or policy changed")
        _head(self._bundle[0])
        _head(self._bundle[1])
        _tensor(self._bundle[2], shape=())
        owners = _owners(self._bundle)
        if len(owners) != len(self._owners) or any(
            a is not b for a, b in zip(owners, self._owners, strict=True)
        ):
            raise ValueError("CLM head module or parameter owner changed")
        parameters = _params(self._bundle)
        if tuple(p.untyped_storage().data_ptr() for p in parameters) != self._storage:
            raise ValueError("CLM parameter storage owner changed")
        optimized = self._optimizer_parameters(self._bundle)
        actual = [
            parameter for group in self._bundle[3].param_groups for parameter in group["params"]
        ]
        if (
            len(actual) != len(optimized)
            or any(a is not b for a, b in zip(actual, optimized, strict=True))
            or self._bundle[4].optimizer is not self._bundle[3]
        ):
            raise ValueError("CLM optimizer/scheduler parameter owners differ")

    def _batch(self, batch, targets=None):
        return self._paired_embeddings(batch, targets, kind=FrozenEmbeddingBatch)

    def _paired_embeddings(self, batch, targets=None, *, kind):
        # Why: disposable probe owners reuse native math without changing TRAIN admission.
        self._check()
        _exact(batch, kind)
        if batch.identity != self._identity:
            raise ValueError("frozen embedding encoder identity differs")
        for value in (batch.states, batch.actions):
            if (
                type(value) not in (torch.Tensor, nn.Parameter)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or value.ndim != 2
                or not 2 <= value.shape[0] <= self._policy.max_rows
                or value.shape[1] != self._identity.embedding_dim
            ):
                raise ValueError("CLM batch requires bounded CPU float32 NxD embeddings")
        if batch.states.shape != batch.actions.shape:
            raise ValueError("CLM batch requires paired NxD embeddings within fixed bounds")
        states, actions = _tensor(batch.states), _tensor(batch.actions)
        codes = None
        if targets is not None:
            _exact(targets, CLMTargets)
            if len(targets.group_codes) != len(states):
                raise ValueError("group codes must match paired rows")
            codes = torch.tensor(targets.group_codes, dtype=torch.int64)
        return states, actions, codes

    def _choice_batch(self, batch):
        return self._choice_embeddings(batch, kind=FrozenChoiceBatch)

    def _choice_embeddings(self, batch, *, kind):
        self._check()
        _exact(batch, kind)
        if batch.identity != self._identity:
            raise ValueError("choice encoder identity differs")
        for value, minimum in ((batch.states, 1), (batch.actions, 2)):
            if (
                type(value) not in (torch.Tensor, nn.Parameter)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or value.ndim != 2
                or not minimum <= value.shape[0] <= self._policy.max_rows
                or value.shape[1] != self._identity.embedding_dim
            ):
                raise ValueError("choice readout requires bounded CPU float32 NxD/MxD features")
        return _tensor(batch.states), _tensor(batch.actions)

    def _capture(self):
        return CLMHeadSnapshot(
            deepcopy(self._identity),
            deepcopy(self._policy),
            deepcopy(self._bundle[0].state_dict()),
            deepcopy(self._bundle[1].state_dict()),
            self._bundle[2].detach().clone(),
            tuple(
                None if p.grad is None else p.grad.detach().clone() for p in _params(self._bundle)
            ),
            deepcopy(self._bundle[3].state_dict()),
            deepcopy(self._bundle[4].state_dict()),
            _modes(self._bundle),
            self._rng.clone(),
            self._completed_updates,
            self._training_examples,
        )

    def snapshot_state(self):
        self._check()
        self._work["snapshot_attempts"] += 1
        result = self._capture()
        self._work["snapshot_returns"] += 1
        return result

    def _require_original(self, owner, before):
        if self._bundle is not owner or not _equal(vars(before), vars(self._capture())):
            raise ValueError("CLM operation changed original state before publication")
        self._check()

    def _project(self, bundle, states, actions):
        state_z = F.normalize(bundle[0](states), dim=-1)
        action_z = F.normalize(bundle[1](actions), dim=-1)
        if state_z.shape != (len(states), self._policy.projection_dim) or action_z.shape != (
            len(actions),
            self._policy.projection_dim,
        ):
            raise ValueError("head output differs from fixed projection dimension")
        logits = bundle[2].exp().clamp(max=100.0) * state_z @ action_z.t()
        _tensor(logits, shape=(len(states), len(actions)))
        return logits

    def train_batch(self, batch, targets, *, record_gradient_norm=False):
        _exact(targets, CLMTargets)
        states, actions, codes = self._batch(batch, targets)
        return self._train_embeddings(
            states,
            actions,
            codes,
            objective=OBJECTIVE,
            loss_fn=clm_head_loss,
            record_gradient_norm=record_gradient_norm,
        )

    def train_batch_observed(self, batch, targets):
        """Return the original loss plus scalars; retain no observation model state."""
        return self.train_batch(batch, targets, record_gradient_norm=True)

    def _charge_training_inputs(self, states, actions):
        """Additional objective-specific exposures; paired work keeps its schema."""

    def _train_embeddings(
        self, states, actions, labels, *, objective, loss_fn, record_gradient_norm=False
    ):
        # Why: explicit supervised catalog learning shares the owned native update,
        # while each public TRAIN port retains its own record/role admission.
        if type(record_gradient_norm) is not bool:
            raise ValueError("record_gradient_norm must be an exact boolean")
        if self._work["train_attempts"] >= self._policy.max_updates:
            raise ValueError("physical CLM update allowance exhausted")
        owner, before = self._bundle, self._capture()
        self._work["train_attempts"] += 1
        self._work["training_rows_attempted"] += len(states)
        self._work["unreturned_work_unknown"] = True
        try:
            self._charge_training_inputs(states, actions)
            staged = _copy_bundle(owner)
            modes = _modes(staged)
            staged[0].train()
            staged[1].train()
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._rng)
                self._work["loss_forward_attempts"] += 1
                loss, logits, _ = loss_fn(staged[0], staged[1], staged[2], states, actions, labels)
                self._work["loss_forward_returns"] += 1
                _tensor(loss, shape=())
                _tensor(logits, shape=(len(states), len(actions)))
                # Validate width using native module output declarations without
                # another head forward; the loss matrix alone cannot reveal width.
                for head in staged[:2]:
                    last = [m for m in head.modules() if type(m) is nn.Linear][-1]
                    if last.out_features != self._policy.projection_dim:
                        raise ValueError("head output differs from fixed projection dimension")
                staged[3].zero_grad(set_to_none=True)
                self._work["backward_attempts"] += 1
                loss.backward()
                self._work["backward_returns"] += 1
                optimized = self._optimizer_parameters(staged)
                for p in optimized:
                    if p.grad is None:
                        raise ValueError("CLM update omitted a trainable gradient")
                    _tensor(p.grad, shape=p.shape)
                gradient_norm = nn.utils.clip_grad_norm_(optimized, 1.0, error_if_nonfinite=True)
                # Why: use the already-computed scalar, never another reduction/pass.
                observed_norm = float(gradient_norm.detach()) if record_gradient_norm else None
                self._work["optimizer_attempts"] += 1
                staged[3].step()
                self._work["optimizer_returns"] += 1
                self._work["scheduler_attempts"] += 1
                staged[4].step()
                self._work["scheduler_returns"] += 1
                rng = torch.get_rng_state().clone()
            _set_modes(staged, modes)
            for p in _params(staged):
                _tensor(p)
            _finite_tree(staged[3].state_dict())
            _finite_tree(staged[4].state_dict())
            diagnostic = TrainingDiagnostic(objective, float(loss.detach()))
            self._validate_update_parameters(staged)
            self._require_original(owner, before)
            result: TrainingDiagnostic | CLMObservedUpdate = diagnostic
            if record_gradient_norm:
                assert observed_norm is not None
                result = CLMObservedUpdate(
                    diagnostic,
                    CLMGradientObservation(
                        self._completed_updates + 1,
                        len(states),
                        len(optimized),
                        observed_norm,
                    ),
                )
            self._bundle, self._rng = staged, rng
            self._owners = _owners(staged)
            self._storage = tuple(p.untyped_storage().data_ptr() for p in _params(staged))
            self._completed_updates += 1
            self._training_examples += len(states)
            self._generation += 1
            self._work["commits"] += 1
            self._work["unreturned_work_unknown"] = False
            return result
        except Exception as error:
            self._failed = True
            raise TrainingUpdateFailed(str(error), objective, self.work) from error

    def predict(self, batch):
        states, actions, _ = self._batch(batch)
        owner, before = self._bundle, self._capture()
        self._work["prediction_attempts"] += 1
        self._work["prediction_rows_attempted"] += len(states)
        self._work["unreturned_work_unknown"] = True
        try:
            staged = _copy_bundle(owner)
            staged[0].eval()
            staged[1].eval()
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.set_rng_state(self._rng)
                logits = self._project(staged, states, actions)
                probabilities = torch.softmax(logits, dim=-1)
                self._work["prediction_returns"] += 1
                result = _tensor(probabilities, shape=(len(states), len(states)))
            self._validate_update_parameters(staged)
            self._require_original(owner, before)
            self._work["unreturned_work_unknown"] = False
            return result
        except Exception:
            self._failed = True
            raise

    def predict_choices(self, batch):
        """Read fixed query/candidate features without admitting a training batch."""
        states, actions = self._choice_batch(batch)
        owner, before = self._bundle, self._capture()
        self._work["prediction_attempts"] += 1
        self._work["prediction_rows_attempted"] += len(states)
        self._work["choice_prediction_attempts"] += 1
        self._work["choice_candidate_rows_attempted"] += len(actions)
        self._work["unreturned_work_unknown"] = True
        try:
            staged = _copy_bundle(owner)
            staged[0].eval()
            staged[1].eval()
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.set_rng_state(self._rng)
                logits = self._project(staged, states, actions)
                probabilities = torch.softmax(logits, dim=-1)
                self._work["prediction_returns"] += 1
                self._work["choice_prediction_returns"] += 1
                result = _tensor(probabilities, shape=(len(states), len(actions)))
            self._validate_update_parameters(staged)
            self._require_original(owner, before)
            self._work["unreturned_work_unknown"] = False
            return result
        except Exception:
            self._failed = True
            raise

    def restore_state(self, snapshot):
        self._check()
        if type(snapshot) is not CLMHeadSnapshot or set(vars(snapshot)) != set(
            CLMHeadSnapshot.__dataclass_fields__
        ):
            raise ValueError("expected exact complete CLMHeadSnapshot")
        _exact(snapshot.identity, FrozenEncoderIdentity)
        _exact(snapshot.policy, CLMHeadPolicy)
        if snapshot.identity != self._identity or snapshot.policy != self._policy:
            raise ValueError("snapshot encoder identity or policy differs")
        for value in (snapshot.completed_updates, snapshot.training_examples):
            if type(value) is not int or not 0 <= value <= 2**53 - 1:
                raise ValueError("snapshot counters must be bounded native integers")
        if (
            snapshot.completed_updates > self._policy.max_updates
            or not 2 * snapshot.completed_updates
            <= snapshot.training_examples
            <= self._policy.max_rows * snapshot.completed_updates
        ):
            raise ValueError("snapshot update and row counters disagree")
        if self._work["restore_attempts"] >= self._policy.max_restores:
            raise ValueError("CLM restore allowance exhausted")
        owner, before = self._bundle, self._capture()
        self._work["restore_attempts"] += 1
        self._work["unreturned_work_unknown"] = True
        try:
            staged = _copy_bundle(owner)
            for head, state in zip(
                staged[:2], (snapshot.state_head, snapshot.action_head), strict=True
            ):
                current = head.state_dict()
                if type(state) is not type(current) or state.keys() != current.keys():
                    raise ValueError("snapshot head schema differs")
                for key in state:
                    _tensor(state[key], shape=current[key].shape)
                head.load_state_dict(deepcopy(state), strict=True)
            staged[2].data.copy_(_tensor(snapshot.logit_scale, shape=()))
            parameters = _params(staged)
            if type(snapshot.gradients) is not tuple or len(snapshot.gradients) != len(parameters):
                raise ValueError("snapshot gradient schema differs")
            for parameter, gradient in zip(parameters, snapshot.gradients, strict=True):
                parameter.grad = (
                    None if gradient is None else _tensor(gradient, shape=parameter.shape)
                )
            if (
                type(snapshot.optimizer_state) is not dict
                or set(snapshot.optimizer_state) != {"state", "param_groups"}
                or type(snapshot.optimizer_state["param_groups"]) is not list
                or len(snapshot.optimizer_state["param_groups"]) != 1
                or type(snapshot.scheduler_state) is not dict
                or set(snapshot.scheduler_state) != set(staged[4].state_dict())
                or snapshot.scheduler_state["total_steps"] != self._policy.max_updates
                or snapshot.scheduler_state["last_epoch"] != snapshot.completed_updates
            ):
                raise ValueError("snapshot optimizer/scheduler schema or counters differ")
            for key, value in staged[4].state_dict().items():
                if key not in ("last_epoch", "_step_count", "_last_lr") and not _equal(
                    snapshot.scheduler_state[key], value
                ):
                    raise ValueError("snapshot changes the fixed OneCycle schedule")
            if snapshot.scheduler_state["_step_count"] != snapshot.completed_updates + 1:
                raise ValueError("snapshot scheduler step count differs")
            group = snapshot.optimizer_state["param_groups"][0]
            template = staged[3].state_dict()["param_groups"][0]
            if (
                type(group) is not dict
                or set(group) != set(template)
                or group["params"] != template["params"]
                or group["lr"] != snapshot.scheduler_state["_last_lr"][0]
            ):
                raise ValueError("snapshot optimizer parameter groups or schedule differ")
            for key, value in template.items():
                if key not in ("params", "lr", "betas") and not _equal(group[key], value):
                    raise ValueError("snapshot changes fixed AdamW configuration")
            if (
                type(group["betas"]) is not tuple
                or len(group["betas"]) != 2
                or group["betas"][1] != template["betas"][1]
                or type(group["betas"][0]) is not float
                or not 0 <= group["betas"][0] < 1
            ):
                raise ValueError("snapshot changes AdamW beta configuration")
            state = snapshot.optimizer_state["state"]
            expected_ids = group["params"] if snapshot.completed_updates else []
            if type(state) is not dict or set(state) != set(expected_ids):
                raise ValueError("snapshot AdamW state does not cover the committed parameters")
            optimized = self._optimizer_parameters(staged)
            for key, parameter in zip(group["params"], optimized, strict=True):
                if snapshot.completed_updates:
                    item = state[key]
                    if type(item) is not dict or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
                        raise ValueError("snapshot AdamW moment schema differs")
                    _tensor(item["exp_avg"], shape=parameter.shape)
                    second_moment = _tensor(item["exp_avg_sq"], shape=parameter.shape)
                    if torch.any(second_moment < 0).item():
                        raise ValueError("snapshot second moment must be nonnegative")
                    if float(_tensor(item["step"], shape=())) != snapshot.completed_updates:
                        raise ValueError("snapshot AdamW step differs from committed updates")
            _finite_tree(snapshot.optimizer_state)
            _finite_tree(snapshot.scheduler_state)
            staged[3].load_state_dict(deepcopy(snapshot.optimizer_state))
            staged[4].load_state_dict(deepcopy(snapshot.scheduler_state))
            incoming_lr, incoming_beta = group["lr"], group["betas"][0]
            # Why: matching two supplied fields does not prove the fixed recipe.
            # Native get_lr also writes beta1, so compare the saved incoming value
            # on this disposable staged optimizer before admitting either state.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                expected_lr = staged[4].get_lr()
            if (
                expected_lr != [incoming_lr]
                or staged[3].param_groups[0]["betas"][0] != incoming_beta
            ):
                raise ValueError("snapshot learning rate or beta1 differs from native OneCycle")
            _set_modes(staged, snapshot.modes)
            if (
                type(snapshot.cpu_rng) is not torch.Tensor
                or snapshot.cpu_rng.dtype != torch.uint8
                or snapshot.cpu_rng.device.type != "cpu"
                or snapshot.cpu_rng.shape != self._rng.shape
            ):
                raise ValueError("snapshot CPU RNG schema differs")
            rng = snapshot.cpu_rng.clone()
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(rng)
            self._validate_update_parameters(staged)
            self._require_original(owner, before)
            self._bundle, self._rng = staged, rng
            self._owners = _owners(staged)
            self._storage = tuple(p.untyped_storage().data_ptr() for p in _params(staged))
            self._completed_updates = snapshot.completed_updates
            self._training_examples = snapshot.training_examples
            self._generation += 1
            self._work["restore_returns"] += 1
            self._work["unreturned_work_unknown"] = False
        except Exception:
            self._failed = True
            raise
