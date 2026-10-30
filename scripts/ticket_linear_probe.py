"""LS source and paid-read guards for the existing ticket probe API.

No training loop, data loader, evaluation selection or budget authority. The
existing generic probe owns roles, clocks, private updates and return admission.
"""

from copy import deepcopy
from typing import Any

import numpy as np

from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.adapters.context_linear_fit_decision import ContextLinearFitDecisionPredictor
from src.adapters.linear_fit_learner import _rng_signature
from src.app.coupled_attribution import _same_value
from src.app.fresh_task_probe import _validate_batches
from src.core.context_linear_state import copy_context_linear_state
from src.core.decision_features import DecisionFeatureSpec
from scripts.run_ticket_workflow import CONTEXT_FEATURES, CATALOG


def _validate_transfer_cohorts(cohorts, spec, head_tasks, *, episode_cohorts=False):
    """Certify supplied fixed task/head rows; never replace their labels/roles."""
    from scripts.probe_ticket_policy_learning import TRANSFER_HEAD_TASKS, TASKS
    from scripts.context_head_probe import TRANSFER_TASKS
    from scripts.ticket_workflow_rehearsal import _ticket_trace
    from src.core.ticket_workflow import ACTION_IDS

    if not _same_value(head_tasks, TRANSFER_HEAD_TASKS):
        raise ValueError("linear_fit transfer requires its fixed explicit task/head tuple")
    if tuple(len(group) for group in cohorts) != (16, 2, 2, 2):
        raise ValueError("linear_fit transfer requires original16/2/2/2 cohort recipe")
    # Why: validate six authentic traces once at admission, not on every source
    # guard. The diagnostic sandbox work is paid; supplied arrays remain targets.
    traces = {}
    for version, concept in (*TASKS, *TRANSFER_TASKS):
        observations, actions = _ticket_trace(spec, version, concept)
        traces[(version, concept)] = (
            np.array(observations, dtype=np.float64),
            np.array([[float(ACTION_IDS.index(action))] for action in actions]),
        )
    old = tuple(
        (
            np.concatenate([traces[task][0] for task in TASKS[2 * i : 2 * i + 2]]),
            np.concatenate([traces[task][1] for task in TASKS[2 * i : 2 * i + 2]]),
        )
        for i in range(2)
    )
    for group_index, group in enumerate(cohorts):
        for index, batch in enumerate(group):
            if group_index < 2:
                version, concept = TRANSFER_TASKS[index % 2]
                context, pair = f"{concept}-{version}", traces[(version, concept)]
            else:
                context, pair = f"old-v{index + 1}", old[index]
            if (
                batch.context != context
                or (batch.episode is None if episode_cohorts else batch.episode is not None)
                or batch.packet_episodes
                or batch.num_classes != 6
                or not np.array_equal(batch.features, pair[0])
                or not np.array_equal(batch.targets, pair[1])
            ):
                raise ValueError(
                    "linear_fit transfer cohorts differ from fixed authentic task/head rows"
                )


def bind_source(seed, state, cohorts, spec, boundary, *, head_tasks=None, episode_cohorts=False):
    if type(episode_cohorts) is not bool or episode_cohorts and head_tasks is None:
        raise ValueError("episode cohorts require explicit fixed transfer tasks")
    if type(spec) is not DecisionFeatureSpec or spec != DecisionFeatureSpec(
        CONTEXT_FEATURES, CATALOG
    ):
        raise ValueError("linear probe requires the fixed contextual ticket spec")
    if not isinstance(seed, ContextLinearFitLearner):
        raise ValueError("linear probe requires an owned ContextLinearFitLearner")
    original = copy_context_linear_state(state)
    if seed.snapshot_state() != original:
        raise ValueError("linear probe state differs from current source")
    if type(cohorts) is not tuple or len(cohorts) != 4:
        raise ValueError("linear probe requires original four cohort groups")
    _validate_batches(*cohorts)
    if head_tasks is not None:
        _validate_transfer_cohorts(cohorts, spec, head_tasks, episode_cohorts=episode_cohorts)
    # Construction validates vocabulary/context order without performing a read.
    ContextLinearFitDecisionPredictor(spec, original)
    if original.policy.feature_dim != 9 or original.policy.num_classes != 6:
        raise ValueError("linear ticket probe requires D9/K6 context geometry")
    source_owners = (id(seed._policy), id(seed._state))
    inputs = deepcopy(cohorts)
    spec_pin = deepcopy(spec)

    def owners():
        return tuple(
            (id(b), id(b.features), id(b.targets), id(b.episode)) for g in cohorts for b in g
        )

    original_owners = owners()
    rng = _rng_signature()

    def guard():
        boundary()
        if (
            source_owners != (id(seed._policy), id(seed._state))
            or seed.snapshot_state() != original
            or not _same_value(state, original)
            or not _same_value(spec, spec_pin)
            or original_owners != owners()
            or not _same_value(cohorts, inputs)
            or _rng_signature() != rng
        ):
            raise ValueError("linear probe source, original inputs or RNG changed")

    guard()
    return guard


def bind_readouts(row, pins, boundary):
    episode_pins: list[Any] = []
    containers = (
        ("readout_work", row["readout_work"], pins),
        ("episodes", row["episodes"], episode_pins),
    )
    header = {
        key: deepcopy(row[key])
        for key in (
            "mode",
            "prior_native_updates",
            "prior_native_rows",
            *(("selected_head_tasks",) if "selected_head_tasks" in row else ()),
        )
    }

    def check_paid():
        if any(
            row.get(key) is not owner
            or len(owner) != len(records)
            or any(
                owner[i] is not r or not _same_value(r, value)
                for i, (r, value) in enumerate(records)
            )
            for key, owner, records in containers
        ) or any(not _same_value(row.get(key), value) for key, value in header.items()):
            for key, owner, records in containers:
                owner[:] = [r for r, _ in records]
                for r, value in records:
                    r.clear()
                    r.update(deepcopy(value))
                row[key] = owner
            row.update(deepcopy(header))
            raise ValueError("linear probe paid endpoint evidence changed")

    def guard():
        check_paid()
        try:
            boundary()
        finally:
            # Recheck even a throwing callback before admitting more work.
            check_paid()

    return guard, episode_pins
