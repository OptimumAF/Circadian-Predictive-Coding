"""Compose frozen schedule, combined and parent trajectories without scores.

Inputs are existing settings/helpers and arrived train/inner roles. Outputs
preserve original decision, work and selector facts. This module adds only
complete schedule rejection witnesses; it owns no sources, copies or IO.
"""

from __future__ import annotations

from typing import Any

from src.app import continual_combined_factor_preflight as combined
from src.app import continual_parent_factor_preflight as parent
from src.app import continual_schedule_factor_preflight as schedule
from src.app.continual_combined_factor_manifest import fixed_combined_manifest
from src.app.continual_confirmation_state import (
    GuardWitness,
    capture_model,
    finite_json,
    guard_witness,
    legacy_role_facts,
)
from src.app.continual_parent_factor_manifest import fixed_parent_manifest
from src.app.continual_replay_factor_pilot import _parameter_hash
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer, SharedReplaySelection
from src.infra.continual_roles import PhaseDecisionRoles


class ScheduleTrajectory:
    def __init__(self, seed: int) -> None:
        self.manifest = schedule.fixed_schedule_factor_manifest()
        self.arms = self.manifest.arms
        self.models = schedule._new_models(self.manifest, seed)
        self.guards: list[GuardWitness] = []
        self.initial = {name: _parameter_hash(model) for name, model in self.models.items()}
        self.after_a: dict[str, str] = {}
        self.shared = SharedReplayBuffer(
            2,
            ReplayRetentionBudget(self.manifest.memory_examples, self.manifest.memory_bytes),
            ReplayRetentionPolicy("recent_fifo"),
        )
        self.opportunities: tuple[schedule.ScheduleOpportunityFacts, ...] = ()

    def _decision(
        self,
        roles: PhaseDecisionRoles,
        policy: str,
        epoch: int,
        global_epoch: int,
        selected: SharedReplaySelection,
    ) -> schedule.ScheduleDecisionFacts:
        name = f"neutral_{policy}"
        before = capture_model(self.models[name])
        decision = schedule._decide_and_apply(
            self.models, roles, self.manifest, policy, epoch, global_epoch, self.shared, selected
        )
        self.guards.append(
            guard_witness(name, roles.phase, epoch, decision.outcome, before, self.models[name])
        )
        return decision

    def train_phase(self, roles: PhaseDecisionRoles) -> None:
        training = self.manifest.source.training
        offset = 0 if roles.phase == "a" else training.phase_a_epochs
        count = training.phase_a_epochs if roles.phase == "a" else training.phase_b_epochs
        rows = []
        for epoch in range(1, count + 1):
            schedule._train_wake(self.models, roles, self.manifest)
            self.shared.observe_train_batch(roles.train.input, roles.train.target)
            selected = self.shared.select_recent(self.manifest.replay_updates_per_attempt)
            decisions = tuple(
                self._decision(roles, policy, epoch, offset + epoch, selected)
                for policy in self.manifest.policies
            )
            rows.append(
                schedule.ScheduleOpportunityFacts(
                    roles.phase,
                    epoch,
                    offset + epoch,
                    roles.split_hashes["train"],
                    self.shared.retained_order_ids,
                    selected.sample_ids,
                    self.shared.retention.example_count,
                    self.shared.retention.retained_bytes,
                    decisions,
                )
            )
        self.opportunities += tuple(rows)
        if roles.phase == "a":
            self.after_a = {name: _parameter_hash(model) for name, model in self.models.items()}

    def finish(self, seed: int, a: PhaseDecisionRoles, b: PhaseDecisionRoles) -> dict[str, Any]:
        hashes, counts = legacy_role_facts(a, b)
        methods = tuple(
            schedule._method_facts(
                name,
                self.models[name],
                self.initial[name],
                self.after_a[name],
                self.opportunities,
                self.manifest,
            )
            for name in self.arms
        )
        executed = sum(
            row.wake_updates + row.applied_replay_updates + row.rejected_executed_replay_updates
            for row in methods
        )
        return finite_json(
            schedule.ScheduleSeedFacts(seed, hashes, counts, self.opportunities, methods, executed)
        )


class CombinedTrajectory:
    def __init__(self, seed: int) -> None:
        self.manifest = fixed_combined_manifest()
        self.arms = tuple(arm.name for arm in self.manifest.arms)
        self._progress = combined._new_progress(self.manifest, seed)
        self.models = self._progress.models
        self.guards: list[GuardWitness] = []
        self.after_a: dict[str, str] = {}

    def train_phase(self, roles: PhaseDecisionRoles) -> None:
        combined._train_phase(self._progress, roles)
        if roles.phase == "a":
            self.after_a = {name: _parameter_hash(model) for name, model in self.models.items()}

    def finish(self, seed: int, a: PhaseDecisionRoles, b: PhaseDecisionRoles) -> dict[str, Any]:
        return finite_json(combined._finish_seed(self._progress, seed, a, b, self.after_a))


class ParentTrajectory:
    def __init__(self, seed: int) -> None:
        self.manifest = fixed_parent_manifest()
        self.arms = tuple(arm.name for arm in self.manifest.arms)
        self._progress = parent._new_progress(self.manifest, seed)
        self.models = self._progress.models
        self.guards: list[GuardWitness] = []

    def train_phase(self, roles: PhaseDecisionRoles) -> None:
        parent._train_phase(self._progress, roles)

    def finish(self, seed: int, a: PhaseDecisionRoles, b: PhaseDecisionRoles) -> dict[str, Any]:
        hashes, counts = legacy_role_facts(a, b)
        methods = tuple(parent._method_facts(self._progress, arm) for arm in self.manifest.arms)
        return finite_json(
            parent.ParentSeedFacts(
                seed,
                hashes,
                counts,
                methods,
                tuple(self._progress.opportunities),
                sum(row["wake_updates"] for row in methods),
            )
        )
