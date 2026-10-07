"""Compose gating, replay and A-boundary sleep without evaluation.

Inputs are fixed original settings and arrived train/inner roles. Outputs are
unchanged training trajectories and unscored legacy cost/decision facts.
Source construction, checkpoint copies, joint validation and IO live outside.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from src.app import continual_gating_pilot as gating
from src.app import continual_replay_factor_pilot as replay
from src.app import continual_sleep_factor_preflight as sleep
from src.app.continual_confirmation_state import (
    GuardWitness,
    capture_model,
    finite_json,
    guard_witness,
    legacy_role_facts,
)
from src.app.continual_replay_factor_pilot import Model
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer
from src.infra.continual_roles import PhaseDecisionRoles


class GatingTrajectory:
    def __init__(self, seed: int) -> None:
        self.manifest = gating.fixed_gating_pilot_manifest()
        self.arms = self.manifest.arms
        self._group = gating._new_models(seed, self.manifest.source.training.hidden_dim)
        self.models: dict[str, Model] = dict(
            zip(
                self.arms,
                (self._group.ordinary, self._group.neutral, self._group.gating),
                strict=True,
            )
        )
        self.guards: list[GuardWitness] = []
        self.initial = {}
        for name, model in self.models.items():
            if isinstance(model, BackpropMLP):
                raise ValueError("confirmation gating model kind differs")
            self.initial[name] = gating._parameter_hash(model)
        self.minimum = 1.0

    def train_phase(self, roles: PhaseDecisionRoles) -> None:
        training = self.manifest.source.training
        count = training.phase_a_epochs if roles.phase == "a" else training.phase_b_epochs
        self.minimum = min(
            self.minimum,
            gating._train_phase(self._group, roles.train.input, roles.train.target, count),
        )

    def finish(self, seed: int, a: PhaseDecisionRoles, b: PhaseDecisionRoles) -> dict[str, Any]:
        if not self.minimum < 1.0:
            raise ValueError("confirmation chemical gating did not become active")
        hashes, counts = legacy_role_facts(a, b)
        training = self.manifest.source.training
        updates = training.phase_a_epochs + training.phase_b_epochs
        presentations = training.phase_a_epochs * len(
            a.train.input
        ) + training.phase_b_epochs * len(b.train.input)
        methods = []
        for name in self.arms:
            model = self.models[name]
            if isinstance(model, BackpropMLP):
                raise ValueError("confirmation gating model kind differs")
            if isinstance(model, CircadianPredictiveCodingNetwork):
                clocks = model.get_sleep_clocks()
                if (
                    clocks.wake_batches,
                    clocks.wake_examples,
                    clocks.replay_updates,
                    clocks.sleep_events,
                ) != (updates, presentations, 0, 0):
                    raise ValueError("confirmation gating work clocks differ")
            methods.append(
                {
                    "method": name,
                    "initial_parameter_sha256": self.initial[name],
                    "final_parameter_sha256": gating._parameter_hash(model),
                    "wake_updates": updates,
                    "train_presentations": presentations,
                    "latent_iterations": updates * training.pc_inference_steps,
                    "example_inference_iterations": presentations * training.pc_inference_steps,
                    "sleep_attempts": 0,
                    "replay_updates": 0,
                    "hidden_width_start": training.hidden_dim,
                    "hidden_width_end": int(model.weight_hidden_output.shape[0]),
                    "parameter_count_start": 4 * training.hidden_dim + 1,
                    "parameter_count_end": gating._parameter_count(model),
                    "minimum_plasticity": None
                    if name == "ordinary_pc"
                    else 1.0
                    if name == "neutral_circadian"
                    else self.minimum,
                }
            )
        return finite_json(
            {
                "seed": seed,
                "source_role_hashes": hashes,
                "source_role_counts": counts,
                "methods": methods,
            }
        )


class ReplayTrajectory:
    def __init__(self, seed: int) -> None:
        self.manifest = replay.fixed_replay_pilot_manifest()
        self.arms = self.manifest.arms
        self.models = replay._models(seed, self.manifest)
        self.guards: list[GuardWitness] = []
        self.initial = {name: replay._parameter_hash(model) for name, model in self.models.items()}
        self.shared = SharedReplayBuffer(
            2, ReplayRetentionBudget(8, 192), ReplayRetentionPolicy("recent_fifo")
        )
        self.boundaries: tuple[replay.ReplayBoundaryResult, ...] = ()

    def train_phase(self, roles: PhaseDecisionRoles) -> None:
        self.boundaries += replay._train_phase(
            self.models, roles, roles.phase, self.manifest, self.shared
        )

    def finish(self, seed: int, a: PhaseDecisionRoles, b: PhaseDecisionRoles) -> dict[str, Any]:
        hashes, counts = legacy_role_facts(a, b)
        methods = tuple(self._method_facts(name, a, b) for name in self.arms)
        return finite_json(
            {
                "seed": seed,
                "source_role_hashes": hashes,
                "source_role_counts": counts,
                "boundaries": [asdict(row) for row in self.boundaries],
                "methods": methods,
            }
        )

    def _method_facts(
        self, name: str, a: PhaseDecisionRoles, b: PhaseDecisionRoles
    ) -> dict[str, Any]:
        model = self.models[name]
        training = self.manifest.source.training
        updates = training.phase_a_epochs + training.phase_b_epochs
        presentations = training.phase_a_epochs * len(
            a.train.input
        ) + training.phase_b_epochs * len(b.train.input)
        replay_updates = sum(len(row.applied_ids_by_method[name]) for row in self.boundaries)
        is_pc = not isinstance(model, BackpropMLP)
        width = int(model.weight_hidden_output.shape[0])
        expected = self.manifest.planned_width if "width12" in name else self.manifest.width
        if width != expected or replay._parameter_count(model) != 4 * expected + 1:
            raise ValueError("confirmation replay fixed/planned capacity differs")
        sleep_attempts = (
            len(self.boundaries) if isinstance(model, CircadianPredictiveCodingNetwork) else 0
        )
        if isinstance(model, CircadianPredictiveCodingNetwork):
            clocks = model.get_sleep_clocks()
            if (
                clocks.wake_batches,
                clocks.wake_examples,
                clocks.replay_updates,
                clocks.sleep_events,
            ) != (updates, presentations, replay_updates, sleep_attempts):
                raise ValueError("confirmation replay work clocks differ")
        return {
            "method": name,
            "initial_parameter_sha256": self.initial[name],
            "final_parameter_sha256": replay._parameter_hash(model),
            "wake_updates": updates,
            "wake_presentations": presentations,
            "wake_inference_loops": updates * training.pc_inference_steps if is_pc else 0,
            "wake_example_inference_iterations": presentations * training.pc_inference_steps
            if is_pc
            else 0,
            "replay_updates": replay_updates,
            "replay_presentations": replay_updates,
            "replay_inference_loops": 2 * replay_updates if is_pc else 0,
            "sleep_attempts": sleep_attempts,
            "hidden_width_start": expected,
            "hidden_width_end": width,
            "parameter_count_start": 4 * expected + 1,
            "parameter_count_end": replay._parameter_count(model),
        }


class SleepTrajectory:
    def __init__(self, seed: int) -> None:
        self.manifest = sleep.fixed_sleep_factor_manifest()
        self.arms = self.manifest.arms
        self.models = sleep._new_models(seed, self.manifest)
        self.guards: list[GuardWitness] = []
        self.initial = {name: sleep._parameter_hash(model) for name, model in self.models.items()}
        self.pre_sleep: dict[str, str] = {}
        self.post_sleep: dict[str, str] = {}
        self.minimum: dict[str, float] = {}
        self.sleeps: dict[str, sleep.GuardedSleepFacts] = {}

    def train_phase(self, roles: PhaseDecisionRoles) -> None:
        sleep._train_phase(self.models, roles, self.manifest, before_sleep=roles.phase == "a")
        if roles.phase == "a":
            self._sleep_after_a(roles)

    def _sleep_after_a(self, roles: PhaseDecisionRoles) -> None:
        self.pre_sleep = {name: sleep._parameter_hash(model) for name, model in self.models.items()}
        self.minimum = {
            name: float(np.min(model.get_plasticity_state()))
            for name in sleep.GATING_ARMS
            if isinstance(model := self.models[name], CircadianPredictiveCodingNetwork)
        }
        if any(not 0.2 <= value < 1.0 for value in self.minimum.values()):
            raise ValueError("confirmation conditional chemical reset gate was inactive")
        before = {name: capture_model(self.models[name]) for name in sleep.SLEEP_ARMS}
        self.sleeps = sleep._apply_guarded_sleep(self.models, roles, self.manifest)
        self.guards.extend(
            guard_witness(
                name,
                "a",
                self.manifest.epochs_per_phase,
                self.sleeps[name].outcome,
                before[name],
                self.models[name],
            )
            for name in sleep.SLEEP_ARMS
        )
        self.post_sleep = {
            name: sleep._parameter_hash(model) for name, model in self.models.items()
        }

    def finish(self, seed: int, a: PhaseDecisionRoles, b: PhaseDecisionRoles) -> dict[str, Any]:
        hashes, counts = legacy_role_facts(a, b)
        arms = tuple(
            sleep._arm_facts(
                name,
                self.models[name],
                self.initial[name],
                self.pre_sleep[name],
                self.post_sleep[name],
                self.minimum.get(name),
                self.sleeps.get(name),
                a,
                b,
                self.manifest,
            )
            for name in self.arms
        )
        return finite_json(sleep.SleepSeedFacts(seed, hashes, counts, arms, False))
