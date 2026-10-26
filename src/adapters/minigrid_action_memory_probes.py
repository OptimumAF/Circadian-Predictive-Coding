"""Reserved MiniGrid count probes through the existing fresh-task helper.

Default routes use owned raw refusal/FIFO64 snapshots for E16 or E8/A8.
The explicit saved route uses genuine invariant FIFO36 states for F16/A16.
Outputs retain isolated curves, failed prefixes, permanent charges and costs.
No file IO, benchmark training, final-data access, tuning or promotion occurs.
"""

from copy import deepcopy
from dataclasses import asdict, fields, replace
import pickle
import random
from math import isfinite

import numpy as np
from time import monotonic
from typing import Any, Callable, cast

from src.adapters import minigrid_learnability as reserved
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.adapters.minigrid_saved_fifo_inputs import SavedFifoInputs
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.adapters.minigrid_action_memory_pilot import memory_policy
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.fresh_task_probe import _validate_batches, run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_return_phase import paid_report_guard
from src.core.action_memory import ActionMemorySnapshot, copy_action_memory_snapshot
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
)
from src.core.fifo_action_memory import (
    FIFO_RETIREMENT_METADATA_BYTES,
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
    convert_action_memory_to_fifo,
    copy_fifo_action_memory_snapshot,
)
from src.core.minigrid_key_door import predefined_probe_layouts, predefined_distinct_probe_layouts
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.process_memory import ProcessRssSampler

BODY_SECONDS = 90.0
POINTS = (0, 4, 8, 16)
RETURN_POINTS = (0, 4, 8, 12, 16)


def describe_protocol() -> dict[str, Any]:
    """Describe without reading artifacts or constructing learners/resources."""
    return dict(
        protocol_id="minigrid_saved_action_memory_E16_capacity_v1",
        classification="reserved_same_domain_capacity_not_generalization",
        method="observation_memory",
        policy=asdict(memory_policy()),
        seed=17,
        source_updates=[0, 128],
        native_training_examples_before=[0, 1152],
        benchmark_updates=0,
        checkpoint_updates=list(POINTS),
        updates_per_cohort=16,
        probe_layouts=[asdict(layout) for layout in predefined_probe_layouts()],
        source="typed owned initial/end128 snapshots from accepted complete memory-live case",
        training="original two probe_train nine-step E cues copied to16 alternating packets",
        evaluation="two separately collected same-E probe_test packets; diagnostics reuse predictions",
        roles=["probe_train", "probe_test"],
        isolation="separate original16-update budgets/disposable native forks; no benchmark/final/live/replay/promotion",
        metric_direction="CE lower; accuracy/exact_sequence higher; gain=CE0-CE16 positive better; missing endpoint has no gain",
        policy_interpretation="54 mature keys leave10slots; capacity refusal is valid and does not establish neural plasticity loss",
        planned_work=dict(
            maximum_storage_updates=32,
            maximum_training_rows=288,
            maximum_score_calls=16,
            maximum_score_rows=144,
            teacher_episodes=4,
            expected_teacher_steps=36,
            maximum_teacher_steps=64,
            benchmark_updates=0,
        ),
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=120,
            process_rss_bytes=shared.RSS_LIMIT,
            output_bytes=32 * 1024**2,
            cpu_threads=1,
            gpu=0,
        ),
        limitations="deterministic reserved fit; unequal capacity/compute to neural controls; no unseen-layout/CPC/serving/recovery advantage",
    )


def describe_return_protocol() -> dict[str, Any]:
    """Fixed exploratory E8→A8 comparison; original E16 descriptor stays intact."""
    protocol = describe_protocol()
    protocol.update(
        protocol_id="minigrid_saved_memory_E8_A8_return_v1",
        classification="exploratory_known_TRAIN_retention_and_reserved_fit",
        method="paired_retained_memory_controls",
        methods=["observation_memory", "observation_memory_fifo"],
        cohort_order=[
            [0, "observation_memory"],
            [0, "observation_memory_fifo"],
            [128, "observation_memory"],
            [128, "observation_memory_fifo"],
        ],
        checkpoint_updates=list(RETURN_POINTS),
        training="eight alternating E probe_train packets, then eight owned original TRAIN A copies",
        evaluation="E-blue/red probe_test plus known TRAIN A-blue/red copies; A is retention/resubstitution, not heldout",
        source_role_transformation="original TRAIN A DTOs preserved; owned copies become probe_train/probe_test with disjoint IDs/arrays and A-blue/red report contexts only",
        metric_direction="CE lower; accuracy/exact_sequence higher; new_E_gain=CE0-CE8; old_A_retention_loss=CE8-CE0; old_A_return_gain=CE8-CE16; new_E_retention_loss=CE16-CE8; missing endpoints unavailable",
        continuation="only verified atomic exact-memory capacity refusal may precede another isolated cohort; any failure keeps top incomplete/CLI2; unknown/corruption stops",
        fifo_retirement_metadata_bytes=FIFO_RETIREMENT_METADATA_BYTES,
        policy_interpretation="same retained-table cap; FIFO metadata adds16 logical bytes plus overhead, not equal total model memory",
        limitations="previously inspected E/A exploratory deterministic fit; no independent confirmation/generalization/neural plasticity/CPC/serving/recovery advantage",
    )
    protocol["planned_work"] = dict(
        maximum_storage_updates=64,
        maximum_training_rows=576,
        maximum_score_calls=80,
        maximum_score_rows=720,
        teacher_episodes=4,
        expected_teacher_steps=36,
        maximum_teacher_steps=64,
        reused_TRAIN_A_episodes=2,
        reused_TRAIN_A_new_physics_steps=0,
        benchmark_updates=0,
    )
    return protocol


def describe_saved_fifo_return_protocol() -> dict[str, Any]:
    """Declare the explicit saved invariant control without constructing a learner."""
    return dict(
        protocol_id="minigrid_saved_invariant_fifo36_F16_A16_v1",
        classification="isolated_known_TRAIN_retention_and_reserved_F_fit",
        method="public_invariant_observation_memory_fifo",
        seed=17,
        source_updates=[0, 128],
        native_training_examples_before=[0, 1152],
        benchmark_updates=0,
        checkpoint_updates=list(POINTS),
        probe_layouts=[asdict(layout) for layout in predefined_distinct_probe_layouts()],
        phases=[
            dict(name="F", updates=16, role="probe_train"),
            dict(name="A_return", updates=16, role="train"),
        ],
        evaluation="separately collected F probe_test and read-only original A TRAIN copies after0/4/8/16 of each phase",
        isolation="one owned fork per0/128 cohort; distinct permanent16-update F/A budgets; no source/benchmark/final/replay/serving/promotion updates",
        metric_direction="CE lower; accuracy/exact higher; new_F_gain=F_CE0-F_CE16; old_A_retention_loss=A_CEF16-A_CEF0; old_A_return_gain=A_CEA0-A_CEA16; new_F_retention_loss=F_CEA16-F_CEF16",
        policy=asdict(
            FifoActionMemoryPolicy(
                feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
            )
        ),
        retirement_metadata_bytes=16,
        planned_work=dict(
            maximum_storage_updates=64,
            maximum_training_rows=576,
            maximum_score_calls=64,
            maximum_score_rows=576,
            teacher_episodes=4,
            expected_teacher_steps=36,
            maximum_teacher_steps=64,
            benchmark_updates=0,
            replay_examples=0,
        ),
        limits=dict(
            body_seconds=120.0,
            parent_seconds=150,
            process_rss_bytes=shared.RSS_LIMIT,
            output_bytes=16 * 1024**2,
            cpu_threads=1,
            gpu=0,
        ),
        limitations="exploratory single-seed reserved fit/known-A retention; no neural, heldout, live gameplay, equal-compute or CPC benefit established; old incomplete/spent run remains unchanged",
    )


def _saved_fifo_snapshots(inputs: SavedFifoInputs) -> tuple[InvariantMemorySnapshot, ...]:
    if type(inputs) is not SavedFifoInputs or set(vars(inputs)) != {
        field.name for field in fields(SavedFifoInputs)
    }:
        raise ValueError("saved FIFO probe requires exact decoded inputs")
    if type(inputs.snapshots) is not tuple or len(inputs.snapshots) != 5:
        raise ValueError("saved FIFO probe requires five actual complete snapshots")
    pairs = tuple((context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red"))
    scores = tuple((point, *pair) for point in (96, 128) for pair in pairs)
    games = tuple(
        (point, *pair)
        for point, group in ((64, pairs[2:]), (96, pairs), (128, pairs))
        for pair in group
    )
    completed = tuple(
        (point, *pair)
        for point, group in ((0, pairs), (32, pairs), (64, pairs[:2]))
        for pair in group
    )
    if (
        not _same_value(inputs.score_slots, scores)
        or not _same_value(inputs.game_slots, games)
        or not _same_value(inputs.completed_games, completed)
        or not _same_value(inputs.censored_games, ((64, "B", "blue"),))
    ):
        raise ValueError("saved FIFO original readout/censored cell evidence differs")
    expected = FifoActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
    )
    states = tuple(copy_invariant_memory_snapshot(state) for state in inputs.snapshots)
    for point, state in zip((0, 32, 64, 96, 128), states, strict=True):
        if (
            state.policy.native_policy != expected
            or state.completed_updates != point
            or state.training_examples != 9 * point
        ):
            raise ValueError("saved FIFO probe policy/native update-row clocks differ")
    if (
        type(inputs.original_training_updates) is not int
        or inputs.original_training_updates != 128
        or type(inputs.original_training_rows) is not int
        or inputs.original_training_rows != 1152
    ):
        raise ValueError("saved FIFO original training charges differ")
    if (
        type(inputs.episodes) is not tuple
        or len(inputs.episodes) != 6
        or inputs.origin_status != "incomplete"
        or type(inputs.origin_reason) is not str
        or type(inputs.original_elapsed_seconds) not in (int, float)
        or not isfinite(inputs.original_elapsed_seconds)
        or inputs.original_elapsed_seconds < 0
    ):
        raise ValueError("saved FIFO original source/failure evidence differs")
    for episode, (context, cue) in zip(
        inputs.episodes,
        ((c, q) for c in ("A", "B", "C_train") for q in ("blue", "red")),
        strict=True,
    ):
        owned = copy_episode(episode)
        if (
            owned.role != "train"
            or owned.context != context
            or owned.episode_id != f"minigrid-source-{context}-{cue}"
            or owned.num_classes != 7
            or len(owned.transitions) != 9
            or any(len(t.observation) != 148 for t in owned.transitions)
        ):
            raise ValueError("saved FIFO original TRAIN source cohort differs")
    return states[0], states[-1]


class _InvariantFifoProbeSource(InvariantActionMemoryLearner):
    """Capture only this cohort's one disposable fork; no benchmark training."""

    def __init__(self, state: InvariantMemorySnapshot):
        super().__init__(state.policy)
        self.restore_state(state)
        self.child: InvariantActionMemoryLearner | None = None
        self.fork_calls = 0

    def fork(self) -> InvariantActionMemoryLearner:
        self.fork_calls += 1
        if self.fork_calls != 1:
            raise ValueError("saved FIFO cohort permits exactly one disposable fork")
        self.child = super().fork()
        return self.child


def _saved_fifo_return_packets(sources):
    # Why: monitor labels stay read-only TRAIN; only disjoint return templates
    # train the disposable candidate after the original F phase has completed.
    monitors = tuple(
        reserved._packet(
            replace(source, context="A-" + cue, episode_id=f"{source.episode_id}:saved-monitor")
        )
        for source, cue in zip(sources, ("blue", "red"), strict=True)
    )
    templates = tuple(
        reserved._packet(
            replace(source, context="A-" + cue, episode_id=f"{source.episode_id}:saved-return")
        )
        for source, cue in zip(sources, ("blue", "red"), strict=True)
    )
    return monitors, templates


def _saved_fifo_outcomes(raw):
    if raw is None:
        return None

    def mean(phase, field, point):
        points = (
            phase.get(field, {}).get("checkpoints", []) if field else phase.get("checkpoints", [])
        )
        record = next((row for row in points if row["updates"] == point), None)
        return (
            None
            if record is None
            else sum(score["ce"] for score in record["scores"].values()) / len(record["scores"])
        )

    def difference(left, right):
        return None if left is None or right is None else left - right

    returning = raw.get("return_phase", {})
    return dict(
        new_F_gain=difference(mean(raw, "", 0), mean(raw, "", 16)),
        old_A_retention_loss=difference(
            mean(raw, "training_readouts", 16), mean(raw, "training_readouts", 0)
        ),
        old_A_return_gain=difference(
            mean(returning, "training_readouts", 0), mean(returning, "training_readouts", 16)
        ),
        new_F_retention_loss=difference(mean(returning, "", 16), mean(raw, "", 16)),
    )


def run_saved_fifo_return_pilot(
    inputs: SavedFifoInputs, *, max_seconds: float = 120.0
) -> dict[str, Any]:
    """Use genuine0/128 FIFO36 states in the existing isolated F16/A16 helper."""
    if (
        type(max_seconds) not in (int, float)
        or not isfinite(max_seconds)
        or not 0 < max_seconds <= 120
    ):
        raise ValueError("saved FIFO max_seconds must be finite in (0,120]")
    states = _saved_fifo_snapshots(inputs)
    return _run(
        states,
        return_sources=inputs.episodes[:2],
        saved_fifo=inputs,
        max_seconds=float(max_seconds),
    )


def _snapshots(value: tuple[ActionMemorySnapshot, ActionMemorySnapshot]):
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("exact initial/end128 snapshot tuple required")
    owned = tuple(copy_action_memory_snapshot(state) for state in value)
    for point, state in zip((0, 128), owned, strict=True):
        if (
            state.policy != memory_policy()
            or state.completed_updates != point
            or state.training_examples != 9 * point
            or len(state.entries) != (0 if point == 0 else 54)
        ):
            raise ValueError("snapshots must bind accepted0/end128 geometry/cap/rows/entries")
    return owned


class _ProbeSource(ActionMemoryLearner):
    """Observe the helper's one native fork without changing its update logic."""

    def __init__(self, state: ActionMemorySnapshot):
        super().__init__(state.policy)
        self.restore_state(state)
        self.child: ActionMemoryLearner | None = None
        self.fork_calls = 0

    def fork(self) -> ActionMemoryLearner:
        self.fork_calls += 1
        if self.fork_calls != 1:
            raise ValueError("each cohort permits exactly one disposable probe fork")
        # Why: generic probe reports expose progress but not the final table.
        # Retain one owned fork for measured readback; no helper/algorithm change.
        self.child = super().fork()
        return self.child


class _FifoProbeSource(FifoActionMemoryLearner):
    """Observe the same one-fork boundary for the separate FIFO control."""

    def __init__(self, state: FifoActionMemorySnapshot):
        super().__init__(state.policy)
        self.restore_state(state)
        self.child: FifoActionMemoryLearner | None = None
        self.fork_calls = 0

    def fork(self) -> FifoActionMemoryLearner:
        self.fork_calls += 1
        if self.fork_calls != 1:
            raise ValueError("each cohort permits exactly one disposable probe fork")
        self.child = super().fork()
        return self.child


def _cohort(
    state,
    training,
    evaluation,
    sampler,
    started,
    common_guard,
    row,
    *,
    returning=False,
    prior_guard=None,
    saved_fifo=False,
    wall_seconds=BODY_SECONDS,
    training_readouts=(),
    return_training=(),
):
    source = (
        _InvariantFifoProbeSource(state)
        if saved_fifo
        else _FifoProbeSource(state)
        if type(state) is FifoActionMemorySnapshot
        else _ProbeSource(state)
    )
    origin = state.completed_updates
    observed_clock: float | None = None
    original_wall_clock = monotonic

    def clock():
        nonlocal observed_clock
        observed_clock = original_wall_clock()
        return observed_clock

    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=16,
            max_wall_seconds=max(0.0, wall_seconds - (monotonic() - started)),
            max_process_rss_bytes=shared.RSS_LIMIT,
            **({"max_replay_examples": 0} if saved_fifo else {}),
        ),
        clock,
    )
    budget.attach_memory(sampler)
    return_budget = None
    return_last_clock = None
    return_binding = None
    return_config = None
    return_values = None
    return_start = None

    def return_clock():
        nonlocal return_last_clock
        return_last_clock = original_wall_clock()
        return return_last_clock

    if saved_fifo:
        return_budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=16,
                max_replay_examples=0,
                max_wall_seconds=max(0.0, wall_seconds - (monotonic() - started)),
                max_process_rss_bytes=shared.RSS_LIMIT,
            ),
            return_clock,
        )
        return_budget.attach_memory(sampler)
        return_config, return_values, return_start = (
            return_budget.budget,
            asdict(return_budget.budget),
            return_budget.started_at,
        )
        return_binding = (
            return_budget.clock,
            return_budget.progress,
            return_budget.process_rss_sampler,
        )

    def total_charges():
        return budget.updates_completed + (
            0 if return_budget is None else return_budget.updates_completed
        )

    config, config_values, start = budget.budget, asdict(budget.budget), budget.started_at
    child = None
    raw_record = raw_payload = None
    raw_guard = None
    known_charges = 0
    known_snapshot = deepcopy(state)
    row.update(status="running", charged_updates=0)

    def budget_binding():
        if (
            budget.budget is not config
            or not _same_value(asdict(config), config_values)
            or budget.clock is not clock
            or not _same_value(budget.started_at, start)
            or not _same_value(budget.last_clock, observed_clock)
            or budget.progress is not None
            or budget.process_rss_sampler is not sampler
            or type(budget.updates_completed) is not int
            or not 0 <= budget.updates_completed <= 16
            or total_charges() < known_charges
            or not _same_value(budget.replay_examples_completed, 0)
        ):
            raise ValueError("original probe source/budget/config/clock binding changed")
        if return_budget is not None and (
            return_binding is None
            or return_budget.budget is not return_config
            or not _same_value(asdict(return_budget.budget), return_values)
            or not _same_value(return_budget.started_at, return_start)
            or return_budget.clock is not return_binding[0]
            or return_budget.progress is not return_binding[1]
            or return_budget.process_rss_sampler is not return_binding[2]
            or not _same_value(return_budget.last_clock, return_last_clock)
            or type(return_budget.updates_completed) is not int
            or not 0 <= return_budget.updates_completed <= 16
            or type(return_budget.replay_examples_completed) is not int
            or return_budget.replay_examples_completed != 0
            or (return_budget.updates_completed > 0 and budget.updates_completed != 16)
        ):
            raise ValueError("original return phase budget/config/clock binding changed")

    def read_native(owner):
        # Typed core validation precedes native dataclass/tuple equality. This
        # retains exact field/type checks (including bool-vs-int corruption),
        # without recursively rebuilding report dictionaries at every boundary.
        observed = owner.snapshot_state()
        if saved_fifo:
            return copy_invariant_memory_snapshot(observed)
        if type(state) is FifoActionMemorySnapshot:
            return copy_fifo_action_memory_snapshot(observed)
        return copy_action_memory_snapshot(observed)

    def local_guard():
        nonlocal child, known_charges, known_snapshot
        common_guard()
        if raw_guard is not None:
            raw_guard()
        if raw_payload is not None and pickle.dumps(raw_record, protocol=5) != raw_payload:
            raise ValueError("preceding paid helper report changed")
        budget_binding()
        if state != read_native(source):
            raise ValueError("original probe source changed")
        if type(source.fork_calls) is not int or source.fork_calls not in (0, 1):
            raise ValueError("probe native fork count changed")
        if source.child is not None:
            if saved_fifo and source.fork_calls != 1:
                raise ValueError("saved FIFO captured child requires its one original fork")
            if child is None:
                child = source.child
            if source.child is not child or child is source:
                raise ValueError("original disposable probe fork was replaced")
            current = read_native(child)
            if (
                current.completed_updates != origin + total_charges()
                or current.training_examples != state.training_examples + 9 * total_charges()
            ):
                raise ValueError("probe native and permanent budget clocks disagree")
            if total_charges() == known_charges:
                if current != known_snapshot:
                    raise ValueError("probe state changed without an observed paid update")
            elif total_charges() != known_charges + 1:
                raise ValueError("probe counter advanced without one observed paid update")
        # The candidate readback is an external callback; recheck source last.
        if state != read_native(source):
            raise ValueError("probe source changed during candidate readback")
        common_guard()

        if saved_fifo and (
            type(source.fork_calls) is not int
            or (child is not None and (source.child is not child or source.fork_calls != 1))
            or (child is None and (source.child is not None or source.fork_calls != 0))
        ):
            raise ValueError("saved FIFO disposable fork ownership changed during readback")
        # Pure checks after the last external readback catch callback changes;
        # never hide an altered clock by reading it again first.
        budget_binding()
        if child is not None and (
            current.completed_updates != origin + total_charges()
            or current.training_examples != state.training_examples + 9 * total_charges()
        ):
            raise ValueError("probe budget changed during terminal source readback")
        advanced = total_charges() != known_charges
        known_charges = total_charges()
        if child is not None and advanced:
            known_snapshot = deepcopy(current)

    def guard():
        # Why: stored preceding-cohort checks must be local. Recursing through
        # their own predecessors multiplies work as cohorts accumulate.
        if prior_guard is not None:
            prior_guard()
        local_guard()
        if prior_guard is not None:
            prior_guard()

    try:
        raw = run_fresh_task_probe(
            source,
            state,
            training,
            evaluation,
            budget=budget,
            checkpoint_updates=RETURN_POINTS if returning and not saved_fifo else POINTS,
            decision_diagnostics=True,
            check_boundary=guard,
            **(
                dict(
                    training_readouts=training_readouts,
                    return_training=return_training,
                    return_budget=return_budget,
                )
                if saved_fifo
                else {}
            ),
        )
        raw_record = raw
        if saved_fifo:
            raw_guard, _ = paid_report_guard(raw)
        raw_payload = pickle.dumps(raw, protocol=5)
        row["charged_updates"] = total_charges()
        guard()
        end = None if child is None else read_native(child)
        guard()
        if saved_fifo:
            row["end_snapshot"] = None if end is None else asdict(end)
        if (
            saved_fifo
            and raw["status"] == "complete"
            and raw.get("return_phase", {}).get("status") != "complete"
        ):
            raise ValueError(
                "saved FIFO A return did not complete: "
                + str(raw.get("return_phase", {}).get("reason"))
            )
        row.update(
            status=raw["status"],
            reason=raw["reason"],
            source_unchanged=True,
            budget_binding_preserved=True,
            end_snapshot=None if end is None else asdict(end),
        )
        if (
            raw["completed_updates"] != budget.updates_completed
            or raw["completed_training_examples"] != 9 * budget.updates_completed
            or (
                return_budget is not None
                and (
                    raw.get("return_phase", {}).get("completed_updates", 0)
                    != return_budget.updates_completed
                    or raw.get("return_phase", {}).get("completed_training_examples", 0)
                    != 9 * return_budget.updates_completed
                )
            )
        ):
            raise ValueError("probe reported commits differ from original permanent charges")
    except Exception as error:
        row.update(
            status="failed",
            reason=f"{type(error).__name__}: {error}",
            budget_binding_preserved=False,
        )
        try:
            row["source_unchanged"] = state == read_native(source)
        except Exception as read_error:
            row["source_readback_error"] = f"{type(read_error).__name__}: {read_error}"
        if raw_payload is None:
            row["unreported_native_work_may_be_unobserved"] = True
    finally:
        observed_charge = total_charges()
        row["budget_updates_observed"] = observed_charge
        try:
            # A missing helper return can still leave an observable native commit.
            # Accept its charge only with the original bindings and matching rows;
            # a rejected counter jump remains a raw observation, never a lower bound.
            budget_binding()
            retained = source.child
            if retained is not None:
                if source.fork_calls != 1 or (child is not None and retained is not child):
                    raise ValueError("charge readback lost its original disposable fork")
                observed = read_native(retained)
                if (
                    observed.completed_updates != origin + observed_charge
                    or observed.training_examples != state.training_examples + 9 * observed_charge
                ):
                    raise ValueError("charge readback disagrees with native commits")
                if observed_charge == known_charges and observed != known_snapshot:
                    raise ValueError("charge readback changed the preceding paid state")
            elif observed_charge != 0:
                raise ValueError("charge readback has no observed native fork")
            budget_binding()
            if total_charges() != observed_charge:
                raise ValueError("charge changed during native readback")
            known_charges = observed_charge
        except Exception as charge_error:
            row["charge_readback_unknown"] = True
            row["charge_readback_error"] = f"{type(charge_error).__name__}: {charge_error}"
            row.update(
                status="failed",
                budget_binding_preserved=False,
                reason=row.get("reason") or row["charge_readback_error"],
            )
        row["charged_updates"] = known_charges
        if saved_fifo:
            row["phase_charges"] = dict(
                F=budget.updates_completed,
                A_return=0 if return_budget is None else return_budget.updates_completed,
            )
        row["source_updates"] = origin
        row["native_training_examples_before"] = state.training_examples
        if raw_payload is not None:
            row["report"] = pickle.loads(raw_payload)
        else:
            row["report"] = None
    frozen_end = None if row["end_snapshot"] is None else deepcopy(end)

    def completed_guard():
        local_guard()
        if frozen_end is not None and (child is None or read_native(child) != frozen_end):
            raise ValueError("preceding disposable probe state changed")
        local_guard()

    return completed_guard


def _return_sources(value, mature):
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("exact original TRAIN A-blue/red episode tuple required")
    owned = []
    invariant = type(mature) is InvariantMemorySnapshot
    native = mature.native_state if invariant else mature
    table = {entry.features: entry.counts for entry in native.entries}
    for episode, cue in zip(value, ("blue", "red"), strict=True):
        if (
            type(episode) is not DemonstrationEpisode
            or set(vars(episode)) != {f.name for f in fields(DemonstrationEpisode)}
            or type(episode.transitions) is not tuple
            or len(episode.transitions) != 9
            or any(
                type(t) is not DemonstrationTransition
                or set(vars(t)) != {f.name for f in fields(DemonstrationTransition)}
                for t in episode.transitions
            )
        ):
            raise ValueError("exact original TRAIN A episode/transition records required")
        detached = copy_episode(episode)
        if (
            detached.role != "train"
            or detached.context != "A"
            or detached.num_classes != 7
            or detached.episode_id != f"minigrid-source-A-{cue}"
            or len(detached.transitions[0].observation) != 148
        ):
            raise ValueError("return episodes must retain original TRAIN A identities/geometry")
        first = detached.transitions[0].observation
        if any(
            table.get(
                canonicalize_public_history(first + t.observation)
                if invariant
                else first + t.observation,
                (0,) * 7,
            )[t.action]
            <= 0
            for t in detached.transitions
        ):
            raise ValueError("original A episodes disagree with the mature source counts")
        owned.append(detached)
    return tuple(owned)


def _return_packets(training, evaluation, sources):
    """Own known TRAIN copies for helper roles; preserve originals separately."""
    old_training = tuple(
        reserved._packet(
            replace(
                sources[i % 2],
                role="probe_train",
                context="A-" + ("blue", "red")[i % 2],
                episode_id=f"{sources[i % 2].episode_id}:return-train:{i}",
            )
        )
        for i in range(8, 16)
    )
    old_evaluation = tuple(
        reserved._packet(
            replace(
                source,
                role="probe_test",
                context="A-" + cue,
                episode_id=f"{source.episode_id}:return-test",
            )
        )
        for source, cue in zip(sources, ("blue", "red"), strict=True)
    )
    return training[:8] + old_training, evaluation + old_evaluation


def _return_outcomes(raw):
    points = {} if raw is None else {p["updates"]: p for p in raw["checkpoints"]}

    def mean(at, context):
        if at not in points:
            return None
        return sum(points[at]["scores"][context + "-" + cue]["ce"] for cue in ("blue", "red")) / 2

    def difference(left, right):
        return None if left is None or right is None else left - right

    return dict(
        new_E_gain=difference(mean(0, "E"), mean(8, "E")),
        old_A_retention_loss=difference(mean(8, "A"), mean(0, "A")),
        old_A_return_gain=difference(mean(8, "A"), mean(16, "A")),
        new_E_retention_loss=difference(mean(16, "E"), mean(8, "E")),
    )


def _verified_capacity_refusal(row):
    raw = row["report"]
    return (
        row["method"] == "observation_memory"
        and row["status"] == "failed"
        and row["source_unchanged"] is True
        and row["budget_binding_preserved"] is True
        and not row.get("charge_readback_unknown", False)
        and raw is not None
        and raw["status"] == "failed"
        and raw["reason"] == "ValueError: memory batch exceeds entry or logical payload capacity"
        and raw["attempted_updates"] == raw["completed_updates"] + 1
        and row["charged_updates"] == raw["completed_updates"]
        and row["end_snapshot"] is not None
    )


def run_pilot(snapshots: tuple[ActionMemorySnapshot, ActionMemorySnapshot]) -> dict[str, Any]:
    """Execute original E16 probes with original stop-on-failure behavior."""
    return _run(snapshots)


def run_return_pilot(snapshots, return_sources) -> dict[str, Any]:
    """Compare fixed E8→A8 controls; keep verified capacity negatives failed."""
    if return_sources is None:
        raise ValueError("original TRAIN A-blue/red sources are required for return comparison")
    return _run(snapshots, return_sources=return_sources)


def _run(snapshots, *, return_sources=None, saved_fifo=None, max_seconds=120.0) -> dict[str, Any]:
    invariant = saved_fifo is not None
    wall_seconds = max_seconds if invariant else BODY_SECONDS
    states: tuple[Any, ...] = (
        _saved_fifo_snapshots(saved_fifo) if invariant else _snapshots(snapshots)
    )  # Pure preflight before any resource/teacher call.
    returning = return_sources is not None
    sources: Any = _return_sources(return_sources, states[-1]) if returning else None
    if returning and not invariant:
        states = tuple(
            item for state in states for item in (state, convert_action_memory_to_fifo(state))
        )
    expected_bindings = tuple((s.completed_updates, s.training_examples) for s in states)
    input_payload = pickle.dumps((snapshots, return_sources), protocol=5)
    original_input_payload = pickle.dumps(saved_fifo, protocol=5)
    python_rng = random.getstate() if invariant else None
    numpy_rng = pickle.dumps(np.random.get_state(), protocol=5) if invariant else None
    started = monotonic()
    rows: list[Any] = [
        dict(
            source_updates=s.completed_updates,
            native_training_examples_before=s.training_examples,
            status="not_run",
            reason=None,
            report=None,
            charged_updates=None,
            end_snapshot=None,
            source_unchanged=None,
            budget_binding_preserved=None,
        )
        for s in states
    ]
    if returning:
        for row, state in zip(rows, states, strict=True):
            row.update(
                method=state.policy.method,
                fifo_retirement_metadata_bytes=(
                    FIFO_RETIREMENT_METADATA_BYTES
                    if type(state) in (FifoActionMemorySnapshot, InvariantMemorySnapshot)
                    else 0
                ),
            )
    report: dict[str, Any] = dict(
        protocol=describe_saved_fifo_return_protocol()
        if invariant
        else describe_return_protocol()
        if returning
        else describe_protocol(),
        status="running",
        reason=None,
        cohorts=rows,
        source_reports=[],
        probe_data=[],
        retained_input_array_bytes=None,
        work=None,
        total_seconds=None,
        observed_process_rss=None,
        scientific_advantage_verified=False,
    )
    origin_payload = None
    if invariant:
        report["origin"] = dict(
            status=saved_fifo.origin_status,
            reason=saved_fifo.origin_reason,
            elapsed_seconds=saved_fifo.original_elapsed_seconds,
            training_updates=saved_fifo.original_training_updates,
            training_rows=saved_fifo.original_training_rows,
            budget_status="spent; not restored",
            censored_games=[list(slot) for slot in saved_fifo.censored_games],
        )
        origin_payload = pickle.dumps(report["origin"], protocol=5)
    if returning:
        report["return_sources"] = [asdict(source) for source in sources]
    return_payload = pickle.dumps(report.get("return_sources"), protocol=5)
    protocol, protocol_payload = report["protocol"], pickle.dumps(report["protocol"], protocol=5)
    source_reports = report["source_reports"]
    saved: list[bytes | None] = [None] * len(states)
    slots = list(rows)
    completed: list[Callable[[], None]] = []
    cohorts = []
    data_payload = source_payload = None
    sampler = None
    collection_started = collection_finished = False
    closing = False
    monitor_training: tuple[AttributionBatch, ...] = ()
    return_training: tuple[AttributionBatch, ...] = ()

    def invariant_evidence_guard():
        if not invariant:
            return
        if (
            pickle.dumps(saved_fifo, protocol=5) != original_input_payload
            or pickle.dumps(report.get("origin"), protocol=5) != origin_payload
        ):
            raise ValueError("saved FIFO original evidence or historical charges changed")
        if (
            random.getstate() != python_rng
            or pickle.dumps(np.random.get_state(), protocol=5) != numpy_rng
        ):
            raise ValueError("saved FIFO probe changed global RNG state")

    def common_guard():
        invariant_evidence_guard()
        if not closing and monotonic() - started > wall_seconds:
            raise ValueError("fixed reserved-assay wall budget exhausted")
        if sampler is not None and not closing:
            sampler.sample()
            if sampler.snapshot().peak_bytes > shared.RSS_LIMIT:
                raise ValueError("fixed reserved-assay RSS budget exhausted")
        if (
            report["protocol"] is not protocol
            or pickle.dumps(protocol, protocol=5) != protocol_payload
            or report["cohorts"] is not rows
            or len(rows) != len(states)
            or report["source_reports"] is not source_reports
            or pickle.dumps((snapshots, return_sources), protocol=5) != input_payload
            or pickle.dumps(report.get("return_sources"), protocol=5) != return_payload
        ):
            raise ValueError("original protocol/snapshots/publication bindings changed")
        for i, slot in enumerate(slots):
            if rows[i] is not slot or (
                saved[i] is None
                and not _same_value(
                    (rows[i]["source_updates"], rows[i]["native_training_examples_before"]),
                    expected_bindings[i],
                )
            ):
                raise ValueError("original cohort or preceding result was replaced")
            if (
                returning
                and saved[i] is None
                and (
                    rows[i]["method"] != states[i].policy.method
                    or rows[i]["fifo_retirement_metadata_bytes"]
                    != (
                        FIFO_RETIREMENT_METADATA_BYTES
                        if type(states[i]) in (FifoActionMemorySnapshot, InvariantMemorySnapshot)
                        else 0
                    )
                )
            ):
                raise ValueError("original cohort method or metadata allowance changed")
        if source_payload is not None and (
            pickle.dumps(source_reports, protocol=5) != source_payload
            or pickle.dumps(report["probe_data"], protocol=5) != data_payload
            or any(not shared._check_signature(data, sig) for data, sig in cohorts)
        ):
            raise ValueError("original reserved source evidence or packets changed")
        invariant_evidence_guard()

    def all_completed():
        common_guard()
        for check in completed:
            check()
        common_guard()

    try:
        with ProcessRssSampler() as active_sampler:
            sampler = active_sampler
            collection_started = True
            raw_training, raw_evaluation = reserved._collect_probe_sources(
                common_guard, source_reports, **({"distinct_probe": True} if invariant else {})
            )
            collection_finished = True
            # Validate raw roles before the history encoder reconstructs packets
            # from episode metadata; contradictory role evidence must be refused.
            _validate_batches(raw_training, raw_evaluation)
            if returning and not invariant:
                raw_training, raw_evaluation = _return_packets(
                    raw_training, raw_evaluation, sources
                )
                _validate_batches(raw_training, raw_evaluation)
            training, evaluation = (
                reserved._history_packets(raw_training),
                reserved._history_packets(raw_evaluation),
            )
            _validate_batches(training, evaluation)
            raw_monitors: tuple[AttributionBatch, ...] = ()
            raw_templates: tuple[AttributionBatch, ...] = ()
            if invariant:
                raw_monitors, raw_templates = _saved_fifo_return_packets(sources)
                monitor_training, return_training = (
                    reserved._history_packets(raw_monitors),
                    reserved._history_packets(raw_templates),
                )
                _validate_batches(training, evaluation, monitor_training, return_training)
            cohorts = [
                (data, shared._signature(data))
                for data in (
                    raw_training,
                    raw_evaluation,
                    training,
                    evaluation,
                    raw_monitors,
                    raw_templates,
                    monitor_training,
                    return_training,
                )
            ]
            report["probe_data"] = [
                asdict(shared._episode(batch))
                for batch in (
                    (*raw_training, *raw_evaluation, *raw_monitors, *raw_templates)
                    if returning
                    else (raw_training[0], raw_training[1], *raw_evaluation)
                )
            ]
            source_payload, data_payload = (
                pickle.dumps(source_reports, protocol=5),
                pickle.dumps(report["probe_data"], protocol=5),
            )
            report["retained_input_array_bytes"] = sum(
                b.features.nbytes + b.targets.nbytes
                for batches in (
                    raw_training,
                    raw_evaluation,
                    training,
                    evaluation,
                    raw_monitors,
                    raw_templates,
                    monitor_training,
                    return_training,
                )
                for b in batches
            )
            for i, state in enumerate(states):
                all_completed()
                preceding = tuple(completed)

                def before_cohort(preceding=preceding):
                    common_guard()
                    for check_previous in preceding:
                        check_previous()
                    common_guard()

                current = slots[i]
                check = _cohort(
                    state,
                    training,
                    evaluation,
                    sampler,
                    started,
                    common_guard,
                    current,
                    returning=returning,
                    prior_guard=before_cohort,
                    **(
                        dict(
                            saved_fifo=True,
                            wall_seconds=wall_seconds,
                            training_readouts=monitor_training,
                            return_training=return_training,
                        )
                        if invariant
                        else {}
                    ),
                )
                if returning:
                    current["outcomes"] = (
                        _saved_fifo_outcomes(current["report"])
                        if invariant
                        else _return_outcomes(current["report"])
                    )
                payload = pickle.dumps(current, protocol=5)
                common_guard()
                rows[i] = slots[i] = payload
                saved[i] = payload
                if pickle.loads(payload)["status"] != "complete":
                    report.update(status="incomplete", reason=pickle.loads(payload)["reason"])
                    if not returning or not _verified_capacity_refusal(pickle.loads(payload)):
                        break
                completed.append(check)
                all_completed()
            else:
                all_completed()
                report["status"] = (
                    "incomplete"
                    # Reaching loop-else means every slot has a frozen payload.
                    if any(pickle.loads(cast(bytes, p))["status"] != "complete" for p in saved)
                    else "complete"
                )
        if invariant:
            # Closing sampler/time ports cannot publish a success after changing
            # protected sources, cumulative phase charges or preceding reports.
            all_completed()
            final_rss = sampler.snapshot()
            final_elapsed = monotonic() - started
            closing = True
            all_completed()
            if final_rss.peak_bytes > shared.RSS_LIMIT or final_elapsed >= wall_seconds:
                raise ValueError("saved FIFO closing resource allowance exhausted")
            report["observed_process_rss"] = asdict(final_rss)
            report["total_seconds"] = final_elapsed
    except Exception as error:
        report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
    finally:
        report["cohorts"] = [
            pickle.loads(payload) if payload is not None else deepcopy(slot)
            for payload, slot in zip(saved, slots, strict=True)
        ]
        probes = [r["report"] for r in report["cohorts"] if r["report"] is not None]
        unreported = any(
            r.get("unreported_native_work_may_be_unobserved", False) for r in report["cohorts"]
        )
        if invariant and any(r.get("charge_readback_unknown", False) for r in report["cohorts"]):
            unreported = True
        report["work"] = dict(
            benchmark_updates=0,
            attempted_storage_updates=sum(p["attempted_updates"] for p in probes),
            completed_storage_updates=sum(p["completed_updates"] for p in probes),
            attempted_training_rows=sum(p["attempted_training_examples"] for p in probes),
            completed_training_rows=sum(p["completed_training_examples"] for p in probes),
            score_calls_attempted=sum(p["score_forward_calls_attempted"] for p in probes),
            score_calls_completed=sum(p["score_forward_calls_completed"] for p in probes),
            score_rows_completed=9 * sum(p["score_forward_calls_completed"] for p in probes),
            teacher_episodes_observed=len(source_reports),
            teacher_steps_returned=sum(p.get("environment_steps") or 0 for p in source_reports),
            teacher_counts_are_lower_bounds=(collection_started and not collection_finished)
            or any(p.get("environment_steps") is None for p in source_reports),
            observed_charged_updates_lower_bound=sum(
                r["charged_updates"]
                for r in report["cohorts"]
                if type(r["charged_updates"]) is int and r["charged_updates"] >= 0
            ),
            probe_totals_are_unknown=unreported,
        )
        if returning:
            report["work"].update(reused_TRAIN_A_episodes=2, reused_TRAIN_A_new_physics_steps=0)
        if invariant:
            report["origin"] = pickle.loads(cast(bytes, origin_payload))
            phases = [
                phase
                for probe in probes
                for phase in (probe, probe.get("return_phase"))
                if phase is not None
            ]
            monitors = [phase.get("training_readouts", {}) for phase in phases]
            for key, field in (
                ("attempted_storage_updates", "attempted_updates"),
                ("completed_storage_updates", "completed_updates"),
                ("attempted_training_rows", "attempted_training_examples"),
                ("completed_training_rows", "completed_training_examples"),
            ):
                report["work"][key] = sum(phase.get(field, 0) for phase in phases)
            report["work"]["score_calls_attempted"] = sum(
                phase.get("score_forward_calls_attempted", 0) for phase in (*phases, *monitors)
            )
            report["work"]["score_calls_completed"] = sum(
                phase.get("score_forward_calls_completed", 0) for phase in (*phases, *monitors)
            )
            report["work"]["score_rows_completed"] = 9 * report["work"]["score_calls_completed"]
            report["work"]["replay_examples"] = 0
        if unreported:
            # A missing helper return can hide attempted rows and score calls.
            # Keep known charges separately rather than inventing zero work.
            for name in (
                "attempted_storage_updates",
                "completed_storage_updates",
                "attempted_training_rows",
                "completed_training_rows",
                "score_calls_attempted",
                "score_calls_completed",
                "score_rows_completed",
            ):
                report["work"][name] = None
        if not invariant:
            report["total_seconds"] = monotonic() - started
            if sampler is not None:
                report["observed_process_rss"] = asdict(sampler.snapshot())
    return report
