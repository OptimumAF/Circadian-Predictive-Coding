"""Bounded causal encoding between an existing prediction and native return.

Owns one pending raw input and O(1) executed-action history. No learner, simulator,
fit, label selection, budget renewal or IO. The caller keeps original native work
receipts when encoding/commit fails; an uncertain control never advances history.
"""

from dataclasses import asdict
import json
import struct

from src.core import pick_cube_action_history as history
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.pick_cube_guard_context import pick_cube_guard_context
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_mass_context import (
    PickCubeMassContext,
    require_mass_binding,
    validate_pick_cube_mass_context,
)
from src.core.continuous_demonstration import (
    ContinuousTransition,
    _copy_transition,
    copy_continuous_episode,
)


class ActionHistoryEncoding:
    def __init__(
        self,
        recipe,
        *,
        train_seed=None,
        guard_seed=None,
        probe_test_seed=None,
        mass_context: PickCubeMassContext | None = None,
    ):
        self.recipe = history.validate_action_history_recipe(recipe)
        self._recipe_binding = json.dumps(asdict(self.recipe), sort_keys=True, allow_nan=False)
        if sum(value is not None for value in (train_seed, guard_seed, probe_test_seed)) > 1:
            raise ValueError(
                "causal history requires separate TRAIN, guard or probe-test reset binding"
            )
        self._mass_request = mass_context
        self.mass_context = None
        if mass_context is not None:
            mass = validate_pick_cube_mass_context(mass_context)
            if (
                type(train_seed) is not int
                or train_seed != mass.seed
                or guard_seed is not None
                or probe_test_seed is not None
                or not history.is_taskspace_recipe(self.recipe)
                or self.recipe.mode != "executed"
            ):
                raise ValueError("mass history requires explicit TRAIN17 executed taskspace12")
            self.mass_context = require_mass_binding(
                mass_context, context=mass.context, seed=train_seed, role="train"
            )
        self._mass_birth = (
            self._mass_request,
            self.mass_context,
            json.dumps(asdict(self.mass_context), sort_keys=True, allow_nan=False)
            if self.mass_context is not None
            else None,
        )
        probe_binding = None
        if probe_test_seed is not None:
            probe_binding = pick_cube_probe_binding("probe_test")
            if type(probe_test_seed) is not int or probe_test_seed != probe_binding.seed:
                raise ValueError("causal probe-test history requires the exact native seed24")
            if not history.is_taskspace_recipe(self.recipe) or self.recipe.mode != "executed":
                raise ValueError("causal probe-test history requires fixed executed taskspace12")
        self.native_role = (
            "probe_test"
            if probe_binding is not None
            else "inner_guard"
            if guard_seed is not None
            else "train"
            if train_seed is not None
            else None
        )
        self.native_binding = (
            (probe_binding.seed, probe_binding.context, "pick-cube-probe-test-smoke-24")
            if probe_binding is not None
            else (
                guard_seed,
                pick_cube_guard_context(guard_seed),
                f"pick-cube-guard-smoke-{guard_seed}",
            )
            if guard_seed is not None
            else (
                train_seed,
                self.mass_context.context
                if self.mass_context is not None
                else pick_cube_training_context(train_seed),
                f"pick-cube-smoke-{train_seed}",
            )
            if train_seed is not None
            else None
        )
        # Why: later callbacks cannot turn a scoring-only trace into TRAIN authority or
        # replace the admitted reset while preserving a coherent public report.
        self._native_binding_seal = json.dumps(
            (self.native_role, self.native_binding), sort_keys=True
        )
        self.state = history.ActionHistoryState()
        self.pending: tuple[float, ...] | None = None
        self.attempts = self.returns = self.committed = 0

    def _check_mass_context(self):
        requested, owned, seal = self._mass_birth
        if self._mass_request is not requested or self.mass_context is not owned:
            raise ValueError("original causal mass metadata owner changed")
        if owned is not None:
            if requested is None:
                raise ValueError("original causal mass request is missing")
            for value in (requested, owned):
                validated = validate_pick_cube_mass_context(value)
                if json.dumps(asdict(validated), sort_keys=True, allow_nan=False) != seal:
                    raise ValueError("original causal mass metadata changed")

    def report(self):
        self._check_mass_context()
        if (
            json.dumps((self.native_role, self.native_binding), sort_keys=True)
            != self._native_binding_seal
        ):
            raise ValueError("original causal history role/reset binding changed")
        if (
            json.dumps(
                asdict(history.validate_action_history_recipe(self.recipe)),
                sort_keys=True,
                allow_nan=False,
            )
            != self._recipe_binding
        ):
            raise ValueError("original causal action history recipe changed")
        result = dict(
            recipe=asdict(history.validate_action_history_recipe(self.recipe)),
            state=asdict(history.validate_action_history_state(self.state)),
            pending=self.pending is not None,
            work=dict(
                encoding_attempts=self.attempts,
                encoding_returns=self.returns,
                committed_controls=self.committed,
                source_values_validated=42 * self.returns,
                model_values_emitted=self.recipe.feature_dim * self.returns,
                model_numeric_bytes_emitted=self.recipe.feature_dim * 8 * self.returns,
            ),
        )

        if self.mass_context is not None:
            result["mass_context"] = asdict(self.mass_context)
        return result

    def check(self, actual):
        # JSON distinguishes native bool/int/float metadata; the report is never
        # the authority for the next observation or any permanently paid count.
        if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
            self.report(), sort_keys=True, allow_nan=False
        ):
            raise ValueError("causal action history metadata changed")

    def validate_executions(self, records, expected_raw):
        """Bind current state to the existing native port's successful prefix.

        The port already owns at most50 raw records. Reusing it avoids keeping
        a second growing receipt chain, while refusing coherent public/ledger
        replacements before another prediction or control is admitted.
        """
        self.report()
        if (
            any(type(value) is not int for value in (self.attempts, self.returns, self.committed))
            or not 0 <= self.committed <= 50
            or len(records) != self.committed
            or self.attempts != self.returns
            or self.returns != self.committed + int(self.pending is not None)
        ):
            raise ValueError("causal action history counters differ from successful native prefix")
        state = history.ActionHistoryState()
        for index, record in enumerate(records):
            if type(record) is not ContinuousTransition or record.step != index:
                raise ValueError("causal action history native prefix changed")
            state = history.advance_action_history(state, record.action)
        if state != history.validate_action_history_state(self.state):
            raise ValueError("causal action history state differs from successful native prefix")
        if self.pending is not None and self.pending != expected_raw:
            raise ValueError("causal action history pending input differs from native observation")

    def encode(self, raw):
        if self.pending is not None or self.attempts >= 50:
            raise ValueError("causal action history has a pending input or reached50 controls")
        self.attempts += 1
        encoded = history.action_history_observation(raw, self.state, self.recipe)
        self.pending = tuple(raw)
        self.returns += 1
        return encoded

    def commit(self, record: ContinuousTransition, requested):
        if self.pending is None:
            raise ValueError("causal action history requires one observed pending input")
        owned = _copy_transition(record)
        if type(requested) is not tuple or len(requested) != 4:
            raise ValueError("causal action history requires the original four-axis command")
        sent = tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in requested)
        if (
            owned.step != self.committed
            or owned.observation != self.pending
            or owned.action != sent
        ):
            raise ValueError("returned execution differs from the observed input or sent command")
        next_state = history.advance_action_history(self.state, owned.action)
        self.state = next_state
        self.pending = None
        self.committed += 1

    def verify_episode(self, episode):
        self.report()
        owned = copy_continuous_episode(episode)
        history.action_history_spec(owned.spec, self.recipe)
        if self.native_binding is not None:
            _, context, episode_id = self.native_binding
            if (owned.role, owned.context, owned.episode_id) != (
                self.native_role,
                context,
                episode_id,
            ):
                raise ValueError("causal episode differs from its admitted role/reset binding")
        if self.pending is not None or len(owned.transitions) != self.committed:
            raise ValueError("complete causal action history lost an observed native return")
        state = history.ActionHistoryState()
        for record in owned.transitions:
            state = history.advance_action_history(state, record.action)
        if state != self.state or self.attempts != self.returns or self.returns != self.committed:
            raise ValueError("complete causal action history differs from executed native controls")
