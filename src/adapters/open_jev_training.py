"""CPU native Choice training through the existing opaque-input learner port.

Inputs are typed requests and hard candidate IDs; outputs are native diagnostics
and calibrated inference probabilities. Owns bounded in-memory LoRA/head/AdamW/
gradient/RNG snapshots and opt-in disposable owners. Does not release labels,
calibrate, load weights, authenticate a frozen base cryptographically or provide
portable resume.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

from src.adapters.open_jev_choice import build_open_jev_choice
from src.adapters.open_jev_consolidation import (
    admit_adapter_reference,
    admit_consolidation,
    new_consolidation_work,
)
from src.adapters.open_jev_owned_fork import fork_native_learner
from src.adapters.open_jev_projection import (
    bind_projection_modules,
    capture_native_adapter_reference,
    check_projection_bindings,
    new_projection_work,
    project_native_adapters,
)
from src.adapters.open_jev_update_health import capture_update_health, finish_update_health
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.typed_decision import DecisionRequest, ProbabilityDecisionResult

DEFINITION = "open_jev_hard_choice_nll_plus_0.1_brier_v1"


@dataclass(frozen=True)
class OpenJevTrainingPolicy:
    max_updates: int = 1
    expected_lora_rank: int = 8
    max_optimizer_resets: int = 0
    record_update_health: bool = False
    max_forks: int = 0
    max_adapter_projections: int = 0
    max_adapter_consolidations: int = 0

    def __post_init__(self):
        if type(self.max_updates) is not int or not 1 <= self.max_updates <= 128:
            raise ValueError("max_updates must be an integer in 1..128")
        if type(self.expected_lora_rank) is not int or not 1 <= self.expected_lora_rank <= 64:
            raise ValueError("expected_lora_rank must be an integer in 1..64")
        if type(self.max_optimizer_resets) is not int or self.max_optimizer_resets not in (0, 1):
            raise ValueError("max_optimizer_resets must be integer 0 or 1")
        if type(self.record_update_health) is not bool:
            raise ValueError("record_update_health must be a boolean")
        if type(self.max_forks) is not int or self.max_forks not in (0, 1):
            raise ValueError("max_forks must be integer 0 or 1")
        if type(self.max_adapter_projections) is not int or self.max_adapter_projections not in (
            0,
            1,
        ):
            raise ValueError("max_adapter_projections must be integer 0 or 1")
        if type(
            self.max_adapter_consolidations
        ) is not int or self.max_adapter_consolidations not in (0, 1):
            raise ValueError("max_adapter_consolidations must be integer 0 or 1")
        if self.max_adapter_consolidations and not (
            self.max_adapter_projections and self.max_optimizer_resets
        ):
            raise ValueError("consolidation requires explicit projection and reset authority")


@dataclass(frozen=True)
class OpenJevTrainingSnapshot:
    owner: int
    weights: dict[str, Any]
    gradients: dict[str, Any]
    optimizer: dict[str, Any]
    rng: Any


def native_choice_loss(logits, target_index: int):
    """Pinned native train.py math: temperature1 NLL + .1 squared Brier sum."""
    import torch

    if logits.ndim != 1 or not 1 <= logits.numel() <= 8:
        raise ValueError("expected 1..8 native Choice logits")
    if type(target_index) is not int or not 0 <= target_index < logits.numel():
        raise ValueError("target index outside candidate set")
    logits = logits.float()
    target = torch.zeros_like(logits)
    target[target_index] = 1
    return (
        -(target * logits.log_softmax(-1)).sum() + 0.1 * ((logits.softmax(-1) - target) ** 2).sum()
    )


class OpenJevTrainingLearner:
    def __init__(
        self,
        model,
        compile_request: Callable,
        *,
        instructions: str,
        policy: OpenJevTrainingPolicy = OpenJevTrainingPolicy(),
        temperature: float = 1.518796342858676,
    ):
        if type(policy) is not OpenJevTrainingPolicy:
            raise ValueError("expected exact OpenJevTrainingPolicy")
        policy.__post_init__()
        if not callable(compile_request) or type(instructions) is not str or not instructions:
            raise ValueError("native compiler and nonempty instructions required")
        if type(temperature) not in (int, float) or not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        import torch

        if model.lora_rank != policy.expected_lora_rank:
            raise ValueError("native LoRA rank differs from declared policy")
        named = dict(model.named_parameters())
        if not named or any(p.is_meta or p.device.type != "cpu" for p in named.values()):
            raise ValueError("native learner requires materialized CPU parameters")
        trainable = {
            name: p
            for name, p in named.items()
            if name.startswith("head.") or ".lora_A." in name or ".lora_B." in name
        }
        if (
            not any(name.startswith("head.") for name in trainable)
            or not any(".lora_A." in name for name in trainable)
            or not any(".lora_B." in name for name in trainable)
            or sum(p.numel() for p in trainable.values()) > 5_000_000
        ):
            raise ValueError("expected bounded native LoRA A/B and scalar head parameters")
        if any(p.dtype != torch.float32 for p in named.values()):
            raise ValueError("this declared training path requires CPU float32")
        model.requires_grad_(False)
        for parameter in trainable.values():
            parameter.requires_grad_(True)
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
        self.model, self.policy = model, policy
        self._birth_policy = policy
        self._policy_values = (
            policy.max_updates,
            policy.expected_lora_rank,
            policy.max_optimizer_resets,
            policy.record_update_health,
            policy.max_forks,
            policy.max_adapter_projections,
            policy.max_adapter_consolidations,
        )
        self._compile, self._instructions = compile_request, instructions
        self._temperature, self._torch = float(temperature), torch
        self._named, self._trainable = named, trainable
        # Why: frozen weights are large and unchanged. Identity/version guards
        # avoid copying the base; normal Tensor writes/replacement are refused.
        # .data writes and arbitrary hostile Python are outside this local port.
        self._base = {
            name: (id(p), p._version) for name, p in named.items() if name not in trainable
        }
        self._optimizer = self._new_optimizer()
        self._rng = torch.get_rng_state().clone()
        self._work = dict(
            training_attempts=0,
            forward_calls=0,
            backward_calls=0,
            optimizer_steps=0,
            committed_updates=0,
        )
        self._poisoned = False
        self._optimizer_reset_work = dict(optimizer_reset_attempts=0, optimizer_resets_completed=0)
        self._update_health_work = dict(
            update_health_attempts=0,
            parameter_copy_elements_attempted=0,
            gradient_reduction_elements_attempted=0,
            step_reduction_elements_attempted=0,
            update_health_reports_completed=0,
        )
        self.last_update: dict[str, Any] | None = None
        self._fork_work = dict(
            fork_attempts=0,
            parameter_copy_elements_attempted=0,
            gradient_copy_elements_attempted=0,
            buffer_copy_elements_attempted=0,
            optimizer_copy_elements_attempted=0,
            rng_copy_elements_attempted=0,
            forks_completed=0,
        )
        self._adapter_projection_work = new_projection_work()
        self.last_adapter_projection: dict[str, Any] | None = None
        self._projection_bindings = (
            bind_projection_modules(self) if policy.max_adapter_projections else ()
        )
        self._adapter_consolidation_work = new_consolidation_work()
        self._adapter_reference = self._adapter_reference_origin = None
        self._last_adapter_reference: dict[str, Any] | None = None

    @property
    def work(self) -> dict[str, int]:
        return dict(self._work)

    @property
    def optimizer_reset_work(self) -> dict[str, int]:
        """Permanent intervention work; deliberately absent from local snapshots."""
        return dict(self._optimizer_reset_work)

    @property
    def update_health_work(self) -> dict[str, int]:
        """Permanent conservative source-element charges; never snapshot-restored."""
        return dict(self._update_health_work)

    def _new_optimizer(self):
        groups = [
            dict(
                params=[p for n, p in self._trainable.items() if not n.startswith("head.")], lr=5e-5
            ),
            dict(params=[p for n, p in self._trainable.items() if n.startswith("head.")], lr=1e-4),
        ]
        return self._torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    @property
    def fork_work(self) -> dict[str, int]:
        """Permanent attempted copy work; inherited by copies, never restored."""
        return dict(self._fork_work)

    def fork(self):
        self._guard()
        if self._adapter_reference is not None:
            raise ValueError("native fork refuses a pending adapter reference")
        if self._fork_work["fork_attempts"] >= self.policy.max_forks:
            raise ValueError("native fork allowance disabled or exhausted")
        return fork_native_learner(self)

    @property
    def adapter_projection_work(self) -> dict[str, int]:
        """Permanent intervention work, absent from restorable model snapshots."""
        return dict(self._adapter_projection_work)

    def project_adapters(self, effective_rank: int) -> dict[str, Any]:
        """Explicitly project owned adapters and consume existing reset authority."""
        return project_native_adapters(self, effective_rank)

    @property
    def adapter_consolidation_work(self) -> dict[str, int]:
        """Permanent capture/consolidation charges, outside restorable snapshots."""
        return dict(self._adapter_consolidation_work)

    @property
    def last_adapter_reference(self) -> dict[str, Any] | None:
        """Detached bounded historical metadata; never exposes reference tensors."""
        return deepcopy(self._last_adapter_reference)

    def capture_adapter_reference(self) -> dict[str, Any]:
        return capture_native_adapter_reference(self)

    def consolidate_adapters(self, effective_rank: int) -> dict[str, Any]:
        """Interpolate complete prior/current operators at fixed half/half weights."""
        reference = admit_consolidation(self)
        return project_native_adapters(self, effective_rank, reference=reference)

    def _guard(self):
        if self._poisoned:
            raise ValueError("native learner stopped after uncertain mutation")
        if (
            self.policy is not self._birth_policy
            or (
                self.policy.max_updates,
                self.policy.expected_lora_rank,
                self.policy.max_optimizer_resets,
                self.policy.record_update_health,
                self.policy.max_forks,
                self.policy.max_adapter_projections,
                self.policy.max_adapter_consolidations,
            )
            != self._policy_values
        ):
            raise ValueError("native training policy replacement refused")
        self.policy.__post_init__()
        named = dict(self.model.named_parameters())
        if named.keys() != self._named.keys() or any(
            named[n] is not p for n, p in self._named.items()
        ):
            raise ValueError("native model parameter replacement refused")
        for name, (identity, version) in self._base.items():
            p = named[name]
            if id(p) != identity or p._version != version or p.requires_grad:
                raise ValueError("frozen base changed")
        for p in self._trainable.values():
            if not p.requires_grad or p.device.type != "cpu" or p.dtype != self._torch.float32:
                raise ValueError("trainable parameter contract changed")
        if self.policy.max_adapter_projections:
            check_projection_bindings(self)
        if self._adapter_reference is not None or self._adapter_reference_origin is not None:
            admit_adapter_reference(self, self._adapter_reference)

    def _trainable_versions(self):
        return tuple(
            (name, id(p), p._version, None if p.grad is None else (id(p.grad), p.grad._version))
            for name, p in self._trainable.items()
        )

    def reset_optimizer_state(self, *, _adapter_reference=None) -> dict[str, int]:
        """Opt-in moment/step reset; never renew model-update or reset allowances.

        Why: a declared history intervention must preserve the learned model,
        gradient buffers and randomness while varying only owned AdamW state.
        This is not a native training step or an automatic metric-based policy.
        """
        self._guard()
        if self._adapter_reference is not _adapter_reference:
            raise ValueError("pending adapter reference reserves reset for consolidation")
        if (
            self._optimizer_reset_work["optimizer_reset_attempts"]
            >= self.policy.max_optimizer_resets
        ):
            raise ValueError("optimizer reset allowance disabled or exhausted")
        self._optimizer_reset_work["optimizer_reset_attempts"] += 1
        try:
            versions = self._trainable_versions()
            global_rng = self._torch.get_rng_state().clone()
            private_rng = self._rng.clone()
            was_training = self.model.training
            original = self._optimizer
            groups = deepcopy(original.state_dict()["param_groups"])
            bindings = tuple(tuple(id(p) for p in g["params"]) for g in original.param_groups)
            staged = self._new_optimizer()
            state = staged.state_dict()
            if (
                type(staged) is not self._torch.optim.AdamW
                or staged is original
                or state["state"]
                or state["param_groups"] != groups
                or tuple(tuple(id(p) for p in g["params"]) for g in staged.param_groups) != bindings
            ):
                raise ValueError("reset requires a new empty AdamW with the unchanged recipe")
            self._guard()
            if (
                versions != self._trainable_versions()
                or was_training != self.model.training
                or not self._torch.equal(global_rng, self._torch.get_rng_state())
                or not self._torch.equal(private_rng, self._rng)
            ):
                raise ValueError("optimizer reset changed weights, gradients, mode or RNG")
            self._optimizer = staged
            self._optimizer_reset_work["optimizer_resets_completed"] += 1
            return self.optimizer_reset_work
        except Exception as error:
            self._poisoned = True
            raise TrainingUpdateFailed(
                str(error),
                "open_jev_adamw_state_reset_v1",
                {**self.work, **self.optimizer_reset_work},
            ) from error

    def _record(self, request: DecisionRequest):
        wire = build_open_jev_choice(request, self._instructions)
        records = self._compile(wire["state"], wire["questions"])
        self._guard()
        keys = [c.candidate_id for c in request.allowed_candidates]
        if (
            type(records) is not list
            or len(records) != 1
            or records[0].get("kind") != "choice"
            or records[0].get("answer_keys") != keys
            or "target" in records[0]
        ):
            raise ValueError("native compiler changed the unlabeled Choice contract")
        return records[0]

    def train_batch(self, features: DecisionRequest, targets: str) -> TrainingDiagnostic:
        self._guard()
        record = self._record(features)
        if type(targets) is not str or targets not in record["answer_keys"]:
            raise ValueError("training target must name an allowed candidate")
        if self._work["training_attempts"] >= self.policy.max_updates:
            raise ValueError("permanent native update allowance exhausted")
        target_index = record["answer_keys"].index(targets)
        record["target"] = [float(i == target_index) for i in range(len(record["answer_keys"]))]
        torch = self._torch
        was_training = self.model.training
        self._work["training_attempts"] += 1
        try:
            # Preserve caller RNG. Private dropout state advances even on failure;
            # paid attempts and physical work are deliberately never restorable.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._rng)
                try:
                    self.model.train()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._work["forward_calls"] += 1
                    logits = self.model([record])[0]
                    if logits.shape != (len(record["answer_keys"]),):
                        raise ValueError("native output shape changed")
                    loss = native_choice_loss(logits, target_index)
                    if not torch.isfinite(loss).item():
                        raise ValueError("nonfinite native training loss")
                    self._work["backward_calls"] += 1
                    loss.backward()
                    health = (
                        capture_update_health(self._trainable, self._update_health_work)
                        if self.policy.record_update_health
                        else None
                    )
                    norm = torch.nn.utils.clip_grad_norm_(list(self._trainable.values()), 1.0)
                    if not torch.isfinite(norm).item():
                        raise ValueError("nonfinite native gradient norm")
                    self._work["optimizer_steps"] += 1
                    self._optimizer.step()
                    if not all(torch.isfinite(p).all().item() for p in self._trainable.values()):
                        raise ValueError("nonfinite updated native parameters")
                    self._guard()
                    value = float(loss.detach().item())
                    update: dict[str, Any] = dict(
                        loss=value, gradient_norm_before_clip=float(norm.item())
                    )
                    if health is not None:
                        before, groups = health
                        update["update_health"] = finish_update_health(
                            self._trainable, before, groups, self._update_health_work
                        )
                    self.last_update = update
                    self._work["committed_updates"] += 1
                    return TrainingDiagnostic(DEFINITION, value)
                finally:
                    self._rng = torch.get_rng_state().clone()
        except Exception as error:
            self._poisoned = True
            work = self.work
            if self._policy_values[3]:
                work.update(self.update_health_work)
            raise TrainingUpdateFailed(str(error), DEFINITION, work) from error
        finally:
            self.model.train(was_training)

    def predict(self, features: DecisionRequest) -> ProbabilityDecisionResult:
        self._guard()
        record = self._record(features)
        was_training = self.model.training
        try:
            self.model.eval()
            self._work["forward_calls"] += 1
            with self._torch.inference_mode():
                logits = self.model([record])[0].float()
                probabilities = (logits / self._temperature).softmax(-1).tolist()
            keys = record["answer_keys"]
            selected = keys[max(range(len(keys)), key=lambda i: probabilities[i])]
            result = ProbabilityDecisionResult(
                features.request_id, selected, tuple(zip(keys, probabilities, strict=True))
            )
        finally:
            self.model.train(was_training)
        self._guard()
        return result

    def snapshot_state(self) -> OpenJevTrainingSnapshot:
        self._guard()
        return OpenJevTrainingSnapshot(
            id(self),
            {n: p.detach().clone() for n, p in self._trainable.items()},
            {
                n: None if p.grad is None else p.grad.detach().clone()
                for n, p in self._trainable.items()
            },
            deepcopy(self._optimizer.state_dict()),
            self._rng.clone(),
        )

    def _check_tensor(self, value, template):
        torch = self._torch
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype != template.dtype
            or value.shape != template.shape
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("invalid native snapshot tensor")

    def restore_state(self, snapshot: OpenJevTrainingSnapshot) -> None:
        self._guard()
        if type(snapshot) is not OpenJevTrainingSnapshot or snapshot.owner != id(self):
            raise ValueError("snapshot belongs to another native learner")
        if (
            snapshot.weights.keys() != self._trainable.keys()
            or snapshot.gradients.keys() != self._trainable.keys()
        ):
            raise ValueError("snapshot parameter keys changed")
        for name, parameter in self._trainable.items():
            self._check_tensor(snapshot.weights[name], parameter)
            if snapshot.gradients[name] is not None:
                self._check_tensor(snapshot.gradients[name], parameter)
        self._check_tensor(snapshot.rng, self._rng)
        # Validate the RNG and optimizer on temporary owners before mutating live
        # weights. No forward/backward/initialization or base clone occurs here.
        generator = self._torch.Generator(device="cpu")
        generator.set_state(snapshot.rng)
        staged = self._new_optimizer()
        copied = deepcopy(snapshot.optimizer)
        expected_groups = staged.state_dict()["param_groups"]
        if type(copied) is not dict or copied.get("param_groups") != expected_groups:
            raise ValueError("snapshot optimizer recipe changed")
        parameters = [p for group in staged.param_groups for p in group["params"]]
        states = copied.get("state")
        if type(states) is not dict or not set(states).issubset(range(len(parameters))):
            raise ValueError("invalid optimizer state indices")
        for index, state in states.items():
            if (
                type(index) is not int
                or type(state) is not dict
                or set(state) != {"step", "exp_avg", "exp_avg_sq"}
            ):
                raise ValueError("invalid AdamW state fields")
            self._check_tensor(state["step"], self._torch.tensor(0.0))
            step = state["step"].item()
            if not 0 <= step <= self._work["optimizer_steps"] or step != int(step):
                raise ValueError("invalid AdamW step")
            for name in ("exp_avg", "exp_avg_sq"):
                self._check_tensor(state[name], parameters[index])
            if (state["exp_avg_sq"] < 0).any().item():
                raise ValueError("invalid AdamW squared moment")
        staged.load_state_dict(copied)
        try:
            with self._torch.no_grad():
                for name, parameter in self._trainable.items():
                    parameter.copy_(snapshot.weights[name])
                    grad = snapshot.gradients[name]
                    parameter.grad = None if grad is None else grad.clone()
            self._optimizer = staged
            self._rng = snapshot.rng.clone()
        except Exception as error:
            self._poisoned = True
            raise TrainingUpdateFailed(str(error), DEFINITION, self.work) from error
