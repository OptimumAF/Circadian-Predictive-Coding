"""Owned ordinary NumPy student plus a wake-only parameter EMA teacher.

The explicit replay method uses affine BCE targets on the existing native
objectives; prediction always uses the student. Complete snapshots bind native
family, geometry and update/chemistry policy, but contain no application budget,
data authorization or recovery authority. No Torch, sleep, structural changes,
target cache or DER/DER++ replication is supplied.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

import numpy as np

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.core.circadian_predictive_coding import replay_sample_id
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
    _copy_wake_vector,
    _plasticity_gates,
)
from src.core.coupled_energy import Array, CoupledParameters, predict_coupled
from src.core.coupled_learner_state import copy_update_policy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import (
    MatchedBackpropNetwork,
    MatchedBackpropSnapshot,
    _copy_parameters,
    _learning_rate,
)
from src.core.slow_teacher import (
    SlowTeacherPolicy,
    TeacherTargets,
    blend_teacher_targets,
    copy_slow_teacher_policy,
    copy_teacher_targets,
)

MIXED_COUPLED_ENERGY_ID = "numpy_coupled_binary_energy_slow_teacher_mixture_v1"
MIXED_BACKPROP_LOSS_ID = "numpy_matched_binary_bce_slow_teacher_mixture_preupdate_v1"
Student = CoupledLearner | MatchedBackpropLearner
StudentState = CoupledSnapshot | MatchedBackpropSnapshot
_BINDING_FIELDS = {
    "family",
    "input_dim",
    "hidden_dims",
    "update_policy",
    "precisions",
    "circadian_config",
    "initial_native_updates",
}


def _record(value: Any, kind: Any) -> None:
    if type(value) is not kind or vars(value).keys() != {item.name for item in fields(kind)}:
        raise ValueError(f"slow teacher requires an exact complete {kind.__name__}")


def _count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a native nonnegative integer")


def _copy_binding(binding: dict[str, Any]) -> dict[str, Any]:
    if type(binding) is not dict or binding.keys() != _BINDING_FIELDS:
        raise ValueError("student binding requires its complete field set")
    family = binding["family"]
    if type(family) is not str or family not in {"matched_bp", "coupled_pc", "coupled_cpc"}:
        raise ValueError("student binding family is unsupported")
    if type(binding["input_dim"]) is not int or binding["input_dim"] < 1:
        raise ValueError("student input dimension must be a positive native integer")
    widths = binding["hidden_dims"]
    if type(widths) is not tuple or not widths or any(type(n) is not int or n < 1 for n in widths):
        raise ValueError("student hidden dimensions must be a positive native integer tuple")
    copied = dict(binding)
    if family == "matched_bp":
        update = binding["update_policy"]
        if type(update) is not dict or update.keys() != {"learning_rate"}:
            raise ValueError("BP update policy requires only its fixed learning rate")
        copied["update_policy"] = {"learning_rate": _learning_rate(update["learning_rate"])}
        if any(
            binding[name] is not None
            for name in ("precisions", "circadian_config", "initial_native_updates")
        ):
            raise ValueError("BP binding cannot contain PC/CPC metadata")
    else:
        copied["update_policy"] = copy_update_policy(binding["update_policy"])
        precisions = binding["precisions"]
        if (
            type(precisions) is not tuple
            or len(precisions) != len(widths)
            or any(
                type(value) not in {int, float} or not 1e-6 <= value <= 1e6 or not isfinite(value)
                for value in precisions
            )
        ):
            raise ValueError("student precisions must match the native bounded policy")
        if family == "coupled_pc":
            if (
                binding["circadian_config"] is not None
                or binding["initial_native_updates"] is not None
            ):
                raise ValueError("PC binding cannot contain CPC metadata")
        else:
            _record(binding["circadian_config"], CoupledCircadianConfig)
            copied["circadian_config"] = replace(binding["circadian_config"])
            _count(binding["initial_native_updates"], "initial_native_updates")
    return copied


def _binding(student: Student, initial_native_updates: int | None) -> dict[str, Any]:
    if type(student) is MatchedBackpropLearner:
        if type(student._model) is not MatchedBackpropNetwork:
            raise ValueError("slow teacher requires an exact ordinary matched BP model")
        return _copy_binding(
            dict(
                family="matched_bp",
                input_dim=student._model.input_dim,
                hidden_dims=student._model.hidden_dims,
                update_policy={"learning_rate": student._learning_rate},
                precisions=None,
                circadian_config=None,
                initial_native_updates=None,
            )
        )
    if type(student) is not CoupledLearner or type(student._model) not in {
        CoupledPredictiveCodingNetwork,
        CoupledCircadianPredictiveCodingNetwork,
    }:
        raise ValueError("slow teacher requires exact ordinary coupled or matched BP learners")
    model = student._model
    cpc = type(model) is CoupledCircadianPredictiveCodingNetwork
    return _copy_binding(
        dict(
            family="coupled_cpc" if cpc else "coupled_pc",
            input_dim=model.input_dim,
            hidden_dims=model.hidden_dims,
            update_policy=student.checkpoint_policy(),
            precisions=model.precisions,
            circadian_config=model.config
            if isinstance(model, CoupledCircadianPredictiveCodingNetwork)
            else None,
            initial_native_updates=initial_native_updates if cpc else None,
        )
    )


def _parameters(parameters: CoupledParameters, binding: dict[str, Any]) -> CoupledParameters:
    copied = _copy_parameters(parameters)
    if (
        copied.hidden_weights[0].shape[0] != binding["input_dim"]
        or tuple(value.shape[1] for value in copied.hidden_weights) != binding["hidden_dims"]
    ):
        raise ValueError("parameter geometry differs from the student binding")
    return copied


def _copy_student_state(state: StudentState, binding: dict[str, Any], updates: int) -> StudentState:
    bp = binding["family"] == "matched_bp"
    _record(state, MatchedBackpropSnapshot if bp else CoupledSnapshot)
    if type(state.format_version) is not int or state.format_version != 1:
        raise ValueError("slow teacher requires native snapshot format 1")
    parameters = _parameters(state.parameters, binding)
    if bp:
        return MatchedBackpropSnapshot(1, parameters)
    assert isinstance(state, CoupledSnapshot)
    if (
        type(state.precisions) is not tuple
        or len(state.precisions) != len(binding["precisions"])
        or any(
            type(value) not in {int, float} or not 1e-6 <= value <= 1e6 or not isfinite(value)
            for value in state.precisions
        )
        or state.precisions != binding["precisions"]
    ):
        raise ValueError("native precision policy differs from the student binding")
    if binding["family"] == "coupled_pc":
        if state.wake_state is not None:
            raise ValueError("PC native snapshot cannot contain CPC wake state")
        return CoupledSnapshot(1, parameters, state.precisions)
    wake = state.wake_state
    _record(wake, CoupledWakeState)
    assert isinstance(wake, CoupledWakeState)
    config = CoupledCircadianConfig(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
    )
    if config != binding["circadian_config"]:
        raise ValueError("native CPC policy differs from the student binding")
    _count(wake.wake_updates, "native CPC wake_updates")
    if wake.wake_updates != binding["initial_native_updates"] + updates:
        raise ValueError("native CPC counter differs from complete student update history")
    width = binding["hidden_dims"][-1]
    chemicals = _copy_wake_vector(wake.chemical_state, width, "chemical_state")
    gates = _copy_wake_vector(wake.plasticity_state, width, "plasticity_state")
    if wake.wake_updates == 0 and (np.any(chemicals != 0) or np.any(gates != 1)):
        raise ValueError("zero native updates require zero chemistry and unit gates")
    if not np.array_equal(gates, _plasticity_gates(chemicals, config)):
        raise ValueError("native CPC gates differ from its chemistry")
    return CoupledSnapshot(
        1,
        parameters,
        state.precisions,
        CoupledWakeState(
            config.chemical_decay,
            config.chemical_buildup_rate,
            config.plasticity_sensitivity,
            config.min_plasticity,
            chemicals,
            gates,
            wake.wake_updates,
        ),
    )


@dataclass(frozen=True)
class SlowTeacherSnapshot:
    format_version: int
    policy: SlowTeacherPolicy
    student_binding: dict[str, Any]
    student_state: StudentState
    teacher_parameters: CoupledParameters | None
    wake_updates: int
    replay_updates: int
    teacher_revision: int

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("slow teacher snapshot requires native format 1")
        policy = copy_slow_teacher_policy(self.policy)
        binding = _copy_binding(self.student_binding)
        for name in ("wake_updates", "replay_updates", "teacher_revision"):
            _count(getattr(self, name), name)
        if self.teacher_revision != (self.wake_updates if policy.mixture_weight > 0 else 0):
            raise ValueError("teacher revision differs from wake-only teacher history")
        student_state = _copy_student_state(
            self.student_state, binding, self.wake_updates + self.replay_updates
        )
        if policy.mixture_weight == 0:
            if self.teacher_parameters is not None:
                raise ValueError("zero mixture must not retain teacher parameters")
            teacher = None
        else:
            if self.teacher_parameters is None:
                raise ValueError("positive mixture requires complete teacher parameters")
            teacher = _parameters(self.teacher_parameters, binding)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "student_binding", binding)
        object.__setattr__(self, "student_state", student_state)
        object.__setattr__(self, "teacher_parameters", teacher)


def _snapshot(snapshot: SlowTeacherSnapshot) -> SlowTeacherSnapshot:
    _record(snapshot, SlowTeacherSnapshot)
    return replace(snapshot)


def _batch(features: Array, targets: Array, width: int) -> tuple[Array, Array]:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 128
        or features.shape[1] != width
        or type(targets) is not np.ndarray
        or targets.dtype != np.float64
        or targets.shape != (len(features), 1)
    ):
        raise ValueError("slow teacher requires exact float64 BxD/Bx1 arrays with 1..128 rows")
    if (
        not np.isfinite(features).all()
        or not np.isfinite(targets).all()
        or np.any((targets < 0) | (targets > 1))
    ):
        raise ValueError("slow teacher requires finite features and native targets in [0,1]")
    return features.copy(), targets.copy()


def _sample_ids(features: Array, targets: Array) -> tuple[str, ...]:
    return tuple(
        replay_sample_id(features[i : i + 1], targets[i : i + 1]) for i in range(len(features))
    )


def _ema_parameters(
    previous: CoupledParameters, current: CoupledParameters, decay: float
) -> CoupledParameters:
    pairs = (
        (
            *previous.hidden_weights,
            *previous.hidden_biases,
            previous.output_weight,
            previous.output_bias,
        ),
        (
            *current.hidden_weights,
            *current.hidden_biases,
            current.output_weight,
            current.output_bias,
        ),
    )
    if len(pairs[0]) != len(pairs[1]) or any(a.shape != b.shape for a, b in zip(*pairs)):
        raise ValueError("teacher EMA geometry differs from the student")
    with np.errstate(over="raise", invalid="raise"):
        return _copy_parameters(
            CoupledParameters(
                tuple(
                    decay * old + (1 - decay) * new
                    for old, new in zip(previous.hidden_weights, current.hidden_weights)
                ),
                tuple(
                    decay * old + (1 - decay) * new
                    for old, new in zip(previous.hidden_biases, current.hidden_biases)
                ),
                decay * previous.output_weight + (1 - decay) * current.output_weight,
                decay * previous.output_bias + (1 - decay) * current.output_bias,
            )
        )


class SlowTeacherLearner:
    """Trusted single-owner student/teacher staging with explicit replay calls."""

    def __init__(self, student: Student, policy: SlowTeacherPolicy) -> None:
        policy = copy_slow_teacher_policy(policy)
        # Reject unsupported families before any fork/model work.
        if type(student) not in {CoupledLearner, MatchedBackpropLearner}:
            raise ValueError("slow teacher requires an exact ordinary NumPy learner")
        native_count = (
            student._model.wake_updates
            if type(student) is CoupledLearner
            and type(student._model) is CoupledCircadianPredictiveCodingNetwork
            else None
        )
        binding = _binding(student, native_count)
        owned = student.fork()
        if owned is student or _binding(owned, native_count) != binding:
            raise ValueError("student fork must own the same complete native policy")
        state = _copy_student_state(owned.snapshot_state(), binding, 0)
        teacher = _parameters(state.parameters, binding) if policy.mixture_weight > 0 else None
        self._policy, self._binding = policy, binding
        self._payload = (owned, teacher, 0, 0, 0)

    @property
    def teacher_policy(self) -> SlowTeacherPolicy:
        return copy_slow_teacher_policy(self._policy)

    @property
    def _student(self) -> Student:
        return self._payload[0]

    @property
    def _teacher_parameters(self) -> CoupledParameters | None:
        return self._payload[1]

    @property
    def wake_updates(self) -> int:
        return self._payload[2]

    @property
    def replay_updates(self) -> int:
        return self._payload[3]

    @property
    def teacher_revision(self) -> int:
        return self._payload[4]

    @property
    def teacher_storage_bytes(self) -> int:
        parameters = self._teacher_parameters
        return (
            0
            if parameters is None
            else sum(
                value.nbytes
                for value in (
                    *parameters.hidden_weights,
                    *parameters.hidden_biases,
                    parameters.output_weight,
                    parameters.output_bias,
                )
            )
        )

    def snapshot_state(self) -> SlowTeacherSnapshot:
        binding = _copy_binding(self._binding)
        if _binding(self._student, binding["initial_native_updates"]) != binding:
            raise ValueError("current student policy differs from its original binding")
        return SlowTeacherSnapshot(
            1,
            self.teacher_policy,
            binding,
            self._student.snapshot_state(),
            self._teacher_parameters,
            self.wake_updates,
            self.replay_updates,
            self.teacher_revision,
        )

    def restore_state(self, snapshot: SlowTeacherSnapshot) -> None:
        owned = _snapshot(snapshot)
        if owned.policy != self.teacher_policy or owned.student_binding != self._binding:
            raise ValueError("slow teacher snapshot policy or student binding differs")
        self.snapshot_state()
        student = self._student.fork()
        if student is self._student:
            raise ValueError("student restore staging must own its native state")
        if isinstance(student, CoupledLearner):
            assert isinstance(owned.student_state, CoupledSnapshot)
            student.restore_state(owned.student_state)
        else:
            assert isinstance(owned.student_state, MatchedBackpropSnapshot)
            student.restore_state(owned.student_state)
        if _binding(student, self._binding["initial_native_updates"]) != self._binding:
            raise ValueError("restored student policy differs")
        _copy_student_state(
            student.snapshot_state(), self._binding, owned.wake_updates + owned.replay_updates
        )
        self._payload = (
            student,
            owned.teacher_parameters,
            owned.wake_updates,
            owned.replay_updates,
            owned.teacher_revision,
        )

    def fork(self) -> "SlowTeacherLearner":
        self.snapshot_state()
        # Only exact ordinary native learners and owned array/scalar records are
        # retained; deepcopy also preserves the original pre-wrapper CPC clock.
        copied = deepcopy(self)
        copied.snapshot_state()
        return copied

    def _publish(
        self,
        student: Student,
        teacher: CoupledParameters | None,
        wakes: int,
        replays: int,
        revision: int,
    ) -> None:
        if _binding(student, self._binding["initial_native_updates"]) != self._binding:
            raise ValueError("trained student policy differs from its original binding")
        checked = SlowTeacherSnapshot(
            1,
            self.teacher_policy,
            self._binding,
            student.snapshot_state(),
            teacher,
            wakes,
            replays,
            revision,
        )
        self._payload = (student, checked.teacher_parameters, wakes, replays, revision)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        features, targets = _batch(features, targets, self._binding["input_dim"])
        before = self.snapshot_state()
        student = self._student.fork()
        if student is self._student:
            raise ValueError("student training staging must own its native state")
        diagnostic = student.train_batch(features, targets)
        _record(diagnostic, TrainingDiagnostic)
        diagnostic = replace(diagnostic)
        teacher = None
        if before.policy.mixture_weight > 0:
            assert before.teacher_parameters is not None
            teacher = _ema_parameters(
                before.teacher_parameters,
                student.snapshot_state().parameters,
                before.policy.ema_decay,
            )
        self._publish(
            student,
            teacher,
            before.wake_updates + 1,
            before.replay_updates,
            before.teacher_revision + (1 if before.policy.mixture_weight > 0 else 0),
        )
        return diagnostic

    def make_teacher_targets(
        self, features: Array, targets: Array, *, role: str, feature_schema: str, output_schema: str
    ) -> TeacherTargets | None:
        policy = self.teacher_policy
        if type(role) is not str or role != "train":
            raise ValueError("teacher targets require the arrived train role")
        if (
            type(feature_schema) is not str
            or feature_schema != policy.feature_schema
            or type(output_schema) is not str
            or output_schema != policy.output_schema
        ):
            raise ValueError("teacher feature/output schema differs from the original binding")
        features, targets = _batch(features, targets, self._binding["input_dim"])
        state = self.snapshot_state()
        if policy.mixture_weight == 0:
            return None
        assert state.teacher_parameters is not None
        probabilities = predict_coupled(state.teacher_parameters, features)
        return TeacherTargets(
            1,
            policy,
            state.teacher_revision,
            state.wake_updates + state.replay_updates,
            _sample_ids(features, targets),
            probabilities,
        )

    def train_distillation_batch(
        self, features: Array, targets: Array, teacher_targets: TeacherTargets | None
    ) -> TrainingDiagnostic:
        features, targets = _batch(features, targets, self._binding["input_dim"])
        before = self.snapshot_state()
        packet = None if teacher_targets is None else copy_teacher_targets(teacher_targets)
        if before.policy.mixture_weight > 0:
            if (
                packet is None
                or packet.policy != before.policy
                or packet.teacher_revision != before.teacher_revision
                or packet.student_updates != before.wake_updates + before.replay_updates
                or packet.sample_ids != _sample_ids(features, targets)
            ):
                raise ValueError(
                    "teacher targets have stale policy, revision, student state or samples"
                )
        mixed = blend_teacher_targets(targets, packet, before.policy)
        student = self._student.fork()
        if student is self._student:
            raise ValueError("student replay staging must own its native state")
        diagnostic = student.train_batch(features, mixed)
        _record(diagnostic, TrainingDiagnostic)
        diagnostic = replace(diagnostic)
        if before.policy.mixture_weight > 0:
            definition = (
                MIXED_BACKPROP_LOSS_ID
                if before.student_binding["family"] == "matched_bp"
                else MIXED_COUPLED_ENERGY_ID
            )
            diagnostic = TrainingDiagnostic(definition, diagnostic.value)
        self._publish(
            student,
            before.teacher_parameters,
            before.wake_updates,
            before.replay_updates + 1,
            before.teacher_revision,
        )
        return diagnostic

    def predict(self, features: Array) -> Array:
        # Native prediction validation and objective remain exactly student-owned.
        return self._student.predict(features)
