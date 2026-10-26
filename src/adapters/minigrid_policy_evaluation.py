"""Frozen MiniGrid readouts through existing learner and episode ports.

Inputs are one owned checkpoint, completed-work guard and explicit data role.
Outputs retain raw TRAIN scores and bounded target-free live decisions/costs.
No teacher, update, selection, replay, promotion or file IO is performed here.
"""

from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, cast

import numpy as np

from src.adapters.invariant_torch_feature_learner import copy_invariant_torch_feature_snapshot
from src.adapters.invariant_action_memory_learner import (
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.adapters.minigrid_endpoint_timing import (
    MiniGridEndpointTiming,
    NativeMiniGridEndpointTiming,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_feature_learner import TorchFeaturePolicy, _snapshot as _feature_snapshot
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import AttributionBatch, _same_value, _validate_batch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_endpoint_observer import EndpointBoundary
from src.core.action_memory import (
    ActionMemoryPolicy,
    ActionMemorySnapshot,
    copy_action_memory_snapshot,
)
from src.core.categorical_probability_scores import probability_scores, validate_class_probabilities
from src.core.demonstration_episode import episode_arrays
from src.core.fifo_action_memory import (
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
    copy_fifo_action_memory_snapshot,
)
from src.core.minigrid_key_door import (
    MiniGridKeyDoorLayout,
    predefined_distinct_probe_layouts,
    predefined_layouts,
    predefined_probe_layouts,
)
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.core.minigrid_public_history import (
    PublicInitialFrameHistory,
    encode_public_history_episode,
)
from src.infra.minigrid_key_door import MiniGridKeyDoor

_WARM_METHODS = ("recurrent_warm_reward", "recurrent_warm_momentum_control")
_PAIRED_METHOD = "recurrent_paired_bc"
_RECURRENT_METHODS = (
    "recurrent_bc",
    "recurrent_weighted_bc",
    _PAIRED_METHOD,
    *_WARM_METHODS,
    "recurrent_public_cue",
)
_MEMORY_METHOD = "observation_memory"


class MiniGridEvaluationFailed(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        super().__init__(str(report["reason"]))


def _validate_native_rows(method: str, updates: int, native_training_examples: int | None) -> None:
    """Describe the fixed warm/reward clocks; the caller still proves receipts.

    Why: eight sampled reward episodes contain1..16 rows each. Their native row
    clock cannot use the existing demonstration-only nine-rows-per-update law.
    Numeric metadata never replaces original receipt, budget or state guards.
    """
    if method not in _WARM_METHODS:
        if native_training_examples is not None:
            raise ValueError("existing MiniGrid methods forbid a native row override")
        return
    if (
        type(method) is not str
        or type(updates) is not int
        or type(native_training_examples) is not int
        or not (
            (updates == 128 and native_training_examples == 1152)
            or (updates == 136 and 1160 <= native_training_examples <= 1280)
        )
    ):
        raise ValueError("warm reward readout requires native clocks128/1152 or136/1160..1280")


def _checkpoint_metadata(
    state: Any,
    method: str,
    updates: int,
    *,
    public_history: bool = False,
    native_training_examples: int | None = None,
) -> None:
    _validate_native_rows(method, updates, native_training_examples)
    if method == _MEMORY_METHOD:
        if type(method) is not str or not public_history:
            raise ValueError("observation memory readout requires public history")
        memory_state = copy_action_memory_snapshot(state)
        expected_policy = ActionMemoryPolicy(
            feature_dim=296, num_classes=7, max_entries=64, max_payload_bytes=155136
        )
        if memory_state.policy != expected_policy:
            raise ValueError("memory checkpoint geometry/capacity differs from the fixed protocol")
        if (
            memory_state.completed_updates != updates
            or memory_state.training_examples != 9 * updates
        ):
            raise ValueError(
                "memory checkpoint native update/row clocks differ from its readout label"
            )
        return
    if method == "recurrent_warm_momentum_control" and updates != 136:
        raise ValueError("momentum control readout requires its distinct136 endpoint")
    policy = getattr(state, "policy", None)
    kind = TorchRecurrentPolicy if method in _RECURRENT_METHODS else TorchFeaturePolicy
    if type(policy) is not kind:
        raise ValueError("owned checkpoint policy does not match the declared method")
    policy.__post_init__()
    if (
        policy.feature_dim
        != (
            296
            if public_history and (method not in _RECURRENT_METHODS or method == _PAIRED_METHOD)
            else 148
        )
        or policy.hidden_dim != 8
        or policy.num_classes != 7
        or policy.seed != 17
        or policy.learning_rate != 0.03
        or policy.momentum != 0.9
        or (isinstance(policy, TorchRecurrentPolicy) and policy.max_episode_steps != 16)
        or (
            isinstance(policy, TorchFeaturePolicy)
            and (
                policy.method != method
                or policy.inference_steps != 2
                or policy.inference_learning_rate != 0.04
            )
        )
    ):
        raise ValueError(
            "checkpoint architecture/method/seed/objective settings differ from protocol"
        )
    if method == _PAIRED_METHOD:
        if public_history:
            from src.adapters.torch_recurrent_episode_batch import _snapshot

            owned = _snapshot(state)
            if (owned.completed_updates, owned.training_episodes, owned.training_examples) != (
                updates,
                2 * updates,
                18 * updates,
            ):
                raise ValueError("paired history checkpoint clocks differ from its readout label")
        else:
            from src.adapters.minigrid_learning_pilot import _paired_counter

            _paired_counter(state, updates)
    elif (
        type(state.completed_updates) is not int
        or state.completed_updates != updates
        or type(state.training_examples) is not int
        or state.training_examples
        != (native_training_examples if method in _WARM_METHODS else 9 * updates)
    ):
        raise ValueError("checkpoint native update/row clocks differ from its readout label")


def _admit(method, updates, layouts, role, sources, check_boundary):
    if method not in ("bp", "pc", "neutral_cpc", "gated_cpc", *_RECURRENT_METHODS, _MEMORY_METHOD):
        raise ValueError("unsupported MiniGrid readout method")
    points = (
        (136,)
        if method == "recurrent_warm_momentum_control"
        else (
            (0, 16, 32, 48, 64)
            if method == _PAIRED_METHOD
            else ((128, 136) if method in _WARM_METHODS else (0, 32, 64, 96, 128))
        )
    )
    if type(updates) is not int or updates not in points:
        raise ValueError("readout requires a declared checkpoint")
    if (
        type(layouts) is not tuple
        or not layouts
        or any(type(v) is not MiniGridKeyDoorLayout for v in layouts)
    ):
        raise ValueError("readout requires declared layout records")
    for layout in layouts:
        layout.__post_init__()
    contexts = tuple(layout.context for layout in layouts)
    if method == _MEMORY_METHOD and (
        type(method) is not str
        or type(role) is not str
        or role != "train"
        or layouts != predefined_layouts()[:3]
        or any(
            set(vars(layout)) != set(vars(expected))
            for layout, expected in zip(layouts, predefined_layouts()[:3])
        )
    ):
        # This count-memory control has only its original six TRAIN traces.
        # Geometry names do not authorize A-only, held-out or changed layouts.
        raise ValueError(
            "observation memory readout requires the exact predefined TRAIN layout triple"
        )
    if method in _WARM_METHODS and (
        type(role) is not str
        or role != "train"
        or layouts != predefined_layouts()[:3]
        or any(
            set(vars(layout)) != set(vars(expected))
            for layout, expected in zip(layouts, predefined_layouts()[:3])
        )
    ):
        raise ValueError("warm reward readout requires the exact predefined TRAIN layout triple")
    # Why: the new weighted objective has only this separately declared TRAIN
    # acquisition control. It grants no new final/D or arbitrary-layout route.
    if method in ("recurrent_weighted_bc", _PAIRED_METHOD) and (
        type(role) is not str or role != "train" or layouts != predefined_layouts()[:1]
    ):
        raise ValueError("weighted/paired recurrent readout requires predefined TRAIN A only")
    if role == "train":
        # Why: fixed-task acquisition needs only its original A cue pair. Keep
        # this extra admission tied to the predefined geometry, not arbitrary
        # context subsets or modified layouts that happen to carry the A name.
        if contexts == ("A",):
            if (
                layouts != predefined_layouts()[:1]
                or type(sources) is not tuple
                or len(sources) != 2
            ):
                raise ValueError(
                    "A-only readout requires predefined A and two TRAIN source packets"
                )
        elif contexts != ("A", "B", "C_train") or type(sources) is not tuple or len(sources) != 6:
            raise ValueError("known readout requires three TRAIN layouts and six source packets")
        seen: set[str] = set()
        pairs = []
        for packet in sources:
            _validate_batch(packet, "train", seen)
            if (
                packet.num_classes != 7
                or packet.features.shape != (9, 148)
                or packet.episode is None
            ):
                raise ValueError("TRAIN source must bind the original nine-step148/K7 packet")
            cue = packet.episode.episode_id.rsplit("-", 1)[-1]
            pairs.append((packet.context, cue))
        if pairs != [(context, cue) for context in contexts for cue in ("blue", "red")]:
            raise ValueError("TRAIN source order/identity must bind each context and cue")
    elif role == "final_test":
        if contexts != ("D",) or updates != 128 or type(sources) is not tuple or sources:
            raise ValueError("final readout requires endpoint D without any labels")
    else:
        raise ValueError("readout role must be train or final_test")
    if not callable(check_boundary):
        raise ValueError("completed training guard must be callable")


def evaluate_checkpoint(
    source: Any,
    state: Any,
    *,
    method: str,
    updates: int,
    layouts: tuple[MiniGridKeyDoorLayout, ...],
    role: str,
    sources: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    public_history: bool = False,
    native_training_examples: int | None = None,
) -> dict[str, Any]:
    """Score TRAIN traces and run live raw-argmax policies without update authority."""
    if type(public_history) is not bool:
        raise ValueError("public_history must be an exact bool")
    _validate_native_rows(method, updates, native_training_examples)
    if method == _MEMORY_METHOD and not public_history:
        raise ValueError("observation memory readout requires public_history=True")
    if method in _WARM_METHODS and public_history:
        raise ValueError("warm reward readout uses original148 recurrent inputs without history")
    _admit(method, updates, layouts, role, sources, check_boundary)
    if method == _MEMORY_METHOD:
        # Pure snapshot admission precedes callbacks, budget checks, forks and
        # environment work. This does not grant new data-role release authority.
        _checkpoint_metadata(state, method, updates, public_history=public_history)
    if method == _PAIRED_METHOD:
        # New snapshots have a different clock contract; refuse them before copying.
        check_boundary()
        budget.before_final()
        _checkpoint_metadata(state, method, updates, public_history=public_history)
    return _run_checkpoint(
        source,
        state,
        method=method,
        updates=updates,
        layouts=layouts,
        role=role,
        sources=sources,
        budget=budget,
        check_boundary=check_boundary,
        public_history=public_history,
        native_training_examples=native_training_examples,
    )


def evaluate_invariant_memory_checkpoint(
    source: Any,
    state: Any,
    *,
    updates: int,
    layouts: tuple[MiniGridKeyDoorLayout, ...],
    role: str,
    sources: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
) -> dict[str, Any]:
    """Score the original recurring TRAIN stream through the bound count port.

    Reusing raw-memory admission keeps this route tied to the original triple
    and six raw teacher packets. The typed encoding identity does not authorize
    new geometry, final labels, clock overrides or a different memory backend.
    """
    _admit(_MEMORY_METHOD, updates, layouts, role, sources, check_boundary)
    expected_policy = ActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=64, max_payload_bytes=155136
    )

    def copy_state(value: Any) -> Any:
        owned = copy_invariant_memory_snapshot(value)
        if (
            type(owned.native_state) is not ActionMemorySnapshot
            or owned.policy.native_policy != expected_policy
            or owned.policy.encoding_id != INVARIANT_HISTORY_ENCODING_ID
            or type(owned.completed_updates) is not int
            or owned.completed_updates not in (0, 32, 64, 96, 128)
            or type(owned.training_examples) is not int
            or owned.training_examples != 9 * owned.completed_updates
        ):
            raise ValueError(
                "recurring invariant memory requires its fixed policy and native clocks"
            )
        return owned

    owned = copy_state(state)
    if owned.completed_updates != updates:
        raise ValueError("recurring memory checkpoint differs from its declared native point")
    return _run_checkpoint(
        source,
        state,
        method=owned.policy.method,
        updates=updates,
        layouts=layouts,
        role=role,
        sources=sources,
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        state_equal=_same_value,
        representation_encoding=INVARIANT_HISTORY_ENCODING_ID,
        require_initial_source=True,
        return_metadata=dict(
            native_training_examples=owned.training_examples,
            model_encoding=INVARIANT_HISTORY_ENCODING_ID,
            readout_scope="original recurring A/B/C_train TRAIN traces and target-free known games; no final/probe data, teaching during readout, selection or generalization claim",
        ),
    )


def _recurring_fifo_copier() -> Callable[[Any], Any]:
    expected_policy = FifoActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
    )

    def copy_state(value: Any) -> Any:
        owned = copy_invariant_memory_snapshot(value)
        if (
            type(owned.native_state) is not FifoActionMemorySnapshot
            or owned.policy.native_policy != expected_policy
            or owned.policy.encoding_id != INVARIANT_HISTORY_ENCODING_ID
            or type(owned.completed_updates) is not int
            or owned.completed_updates not in (0, 32, 64, 96, 128)
            or type(owned.training_examples) is not int
            or owned.training_examples != 9 * owned.completed_updates
        ):
            raise ValueError(
                "recurring invariant FIFO requires its distinct36-entry policy and native clocks"
            )
        return owned

    return copy_state


def evaluate_invariant_fifo_checkpoint(
    source: Any,
    state: Any,
    *,
    updates: int,
    layouts: tuple[MiniGridKeyDoorLayout, ...],
    role: str,
    sources: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    score_pairs: tuple[tuple[str, str], ...] | None = None,
    game_pairs: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, Any]:
    """Read only the distinct FIFO36 recurring TRAIN control.

    Its exact native snapshot includes retirement counters and insertion order;
    matching clocks or public encoding alone never authorize refusal/FIFO64/raw
    states. The existing loop retains all prediction and cleanup costs.
    """
    canonical = tuple(
        (context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    explicit = score_pairs is not None or game_pairs is not None
    for pairs in (score_pairs, game_pairs):
        if pairs is not None and (
            type(pairs) is not tuple
            or any(
                type(pair) is not tuple or len(pair) != 2 or any(type(v) is not str for v in pair)
                for pair in pairs
            )
            or tuple(pair for pair in canonical if pair in pairs) != pairs
        ):
            raise ValueError(
                "FIFO readout selections require unique canonical TRAIN pairs in order"
            )
    budget_guard = None
    if explicit:
        budget_guard = _return_budget_guard(budget)
        if (
            type(budget.budget.max_replay_examples) is not int
            or budget.budget.max_replay_examples != 0
        ):
            raise ValueError("selected FIFO readouts require a zero-replay budget")
    _admit(_MEMORY_METHOD, updates, layouts, role, sources, check_boundary)
    # Why: selected scores may omit sources which still define the frozen cohort.
    # Import locally because the shared runner imports this evaluator at startup.
    from src.adapters.minigrid_learning_pilot import _signature, _check_signature

    signature = _signature(sources) if explicit else None

    def selection_guard() -> None:
        if budget_guard is not None:
            budget_guard()
        if signature is not None and not _check_signature(sources, signature):
            raise ValueError("original six FIFO TRAIN sources changed")

    selected_scores = canonical if score_pairs is None else score_pairs
    selected_games = canonical if game_pairs is None else game_pairs
    selected_sources = tuple(
        packet for packet, pair in zip(sources, canonical, strict=True) if pair in selected_scores
    )
    live_plan = (
        tuple(
            (layout, cue, role)
            for layout in layouts
            for cue in ("blue", "red")
            if (layout.context, cue) in selected_games
        )
        if explicit
        else None
    )
    copy_state = _recurring_fifo_copier()

    owned = copy_state(state)
    if owned.completed_updates != updates:
        raise ValueError("recurring FIFO checkpoint differs from its declared native point")
    native = owned.native_state
    return _run_checkpoint(
        source,
        state,
        method=owned.policy.method,
        updates=updates,
        layouts=layouts,
        role=role,
        sources=selected_sources if explicit else sources,
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        extra_check=selection_guard if explicit else None,
        live_plan=live_plan,
        memory_copier=copy_state,
        state_equal=_same_value,
        representation_encoding=INVARIANT_HISTORY_ENCODING_ID,
        require_initial_source=True,
        return_metadata=dict(
            native_training_examples=owned.training_examples,
            model_encoding=INVARIANT_HISTORY_ENCODING_ID,
            native_evictions=native.evictions,
            native_retired_examples=native.retired_examples,
            native_retained_entries=len(native.entries),
            native_retained_logical_payload_bytes=2424 * len(native.entries),
            native_retirement_metadata_bytes=16,
            readout_scope="original recurring TRAIN FIFO36 control: six A/B/C_train traces and known target-free games; no final/probe data, readout updates, selection or generalization claim",
        ),
    )


def evaluate_heldout_fifo_checkpoint(
    source: Any,
    state: Any,
    *,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
) -> dict[str, Any]:
    """Read exactly D blue/red from frozen benchmark128, never TRAIN or probes.

    Caller must freeze its protocol/model before opening this final readout.
    There is no training, selection, promotion, teacher or metric tuning here.
    """
    copy_fifo = _recurring_fifo_copier()
    owned = copy_fifo(state)
    if owned.completed_updates != 128 or owned.training_examples != 1152:
        raise ValueError(
            "held-out FIFO readout requires benchmark128/1152, not a probe or selected prefix"
        )
    binding = _return_budget_guard(budget)
    if type(budget.budget.max_replay_examples) is not int or budget.budget.max_replay_examples != 0:
        raise ValueError("held-out FIFO readout requires an explicit zero-replay budget")
    if not callable(check_boundary):
        raise ValueError("held-out FIFO readout requires the original boundary callback")

    def copy_state(value: Any) -> Any:
        current = copy_fifo(value)
        if current.completed_updates not in (0, 128):
            raise ValueError("held-out FIFO port requires original source0 or benchmark128")
        return current

    # Why: a separate finite plan preserves the old TRAIN-only admission.
    # D's public geometry is fixed before any outcomes; targets never enter actions.
    layout = predefined_layouts()[3]
    native = owned.native_state
    # The shared TRAIN reader owns execution; label these frames as final.
    frame_scope = "one148-value held-out D final frame per active episode; never training; numeric bytes exclude Python/copy overhead"
    try:
        report = _run_checkpoint(
            source,
            state,
            method=owned.policy.method,
            updates=128,
            layouts=(),
            role="final_test",
            sources=(),
            budget=budget,
            check_boundary=check_boundary,
            public_history=True,
            memory_copier=copy_state,
            state_equal=_same_value,
            extra_check=binding,
            require_initial_source=True,
            live_plan=((layout, "blue", "final_test"), (layout, "red", "final_test")),
            representation_encoding=INVARIANT_HISTORY_ENCODING_ID,
            return_metadata=dict(
                native_training_examples=1152,
                model_encoding=INVARIANT_HISTORY_ENCODING_ID,
                native_evictions=native.evictions,
                native_retired_examples=native.retired_examples,
                native_retained_entries=len(native.entries),
                native_retained_logical_payload_bytes=2424 * len(native.entries),
                native_retirement_metadata_bytes=16,
                readout_scope="final D blue/red from fixed FIFO36 benchmark128; no training/probe/selection/promotion; not a globally unseen benchmark",
            ),
        )
    except MiniGridEvaluationFailed as error:
        error.report["public_history"]["storage_scope"] = frame_scope
        raise
    report["public_history"]["storage_scope"] = frame_scope
    return report


def _return_snapshot_copier(state: Any) -> Callable[[Any], Any]:
    kind: type[Any]
    copier: Callable[[Any], Any]
    policy: Any
    if type(state) is ActionMemorySnapshot:
        kind = ActionMemorySnapshot
        copier = copy_action_memory_snapshot
        policy = ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
    elif type(state) is FifoActionMemorySnapshot:
        kind = FifoActionMemorySnapshot
        copier = copy_fifo_action_memory_snapshot
        policy = FifoActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
    else:
        raise ValueError("saved-return readout requires an exact refusal or FIFO memory snapshot")

    def copy_state(value: Any) -> Any:
        if type(value) is not kind:
            raise ValueError("saved-return snapshot port changed its exact native type")
        owned = cast(Callable[[Any], Any], copier)(value)
        if owned.policy != policy:
            raise ValueError("saved-return memory policy differs from fixed296/K7/64/155136")
        if owned.training_examples != 9 * owned.completed_updates:
            raise ValueError("saved-return snapshot port changed its nine-row native clock law")
        return owned

    return copy_state


def _return_budget_guard(budget: ToyBudgetSession) -> Callable[[], None]:
    if type(budget) is not ToyBudgetSession or type(budget.budget) is not ToyExecutionBudget:
        raise ValueError("saved-return readout requires an exact readout-only budget")
    config = budget.budget
    expected = replace(config)
    if type(config.max_training_updates) is not int or config.max_training_updates != 0:
        raise ValueError("saved-return readout requires a zero-training-update budget")
    clock, started, progress = budget.clock, budget.started_at, budget.progress
    sampler = budget.process_rss_sampler

    def guard() -> None:
        if (
            budget.budget is not config
            or not _same_value(vars(config), vars(expected))
            or budget.clock is not clock
            or budget.progress is not progress
            or budget.process_rss_sampler is not sampler
            or type(budget.started_at) is not type(started)
            or budget.started_at != started
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 0
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
        ):
            raise ValueError(
                "saved-return readout budget binding, clock or permanent charges changed"
            )

    guard()
    return guard


def evaluate_memory_return_checkpoint(
    source: Any,
    state: Any,
    *,
    source_updates: int,
    relative_updates: int,
    source_status: str,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
) -> dict[str, Any]:
    """Run fixed target-free A/E games from actual saved exploratory endpoints."""
    if (
        type(source_updates) is not int
        or type(relative_updates) is not int
        or type(source_status) is not str
        or not callable(check_boundary)
    ):
        raise ValueError(
            "saved-return source/relative/status labels and boundary guard must be exact"
        )
    copier = _return_snapshot_copier(state)
    owned = copier(state)
    method = owned.policy.method
    allowed = {
        (0, 0, "complete", "observation_memory"),
        (0, 16, "complete", "observation_memory"),
        (0, 0, "complete", "observation_memory_fifo"),
        (0, 16, "complete", "observation_memory_fifo"),
        (128, 0, "complete", "observation_memory"),
        (128, 1, "failed", "observation_memory"),
        (128, 0, "complete", "observation_memory_fifo"),
        (128, 16, "complete", "observation_memory_fifo"),
    }
    if (source_updates, relative_updates, source_status, method) not in allowed:
        raise ValueError("saved-return readout requires a declared actual source endpoint")
    updates = source_updates + relative_updates
    if owned.completed_updates != updates or owned.training_examples != 9 * updates:
        raise ValueError("saved-return native clocks differ from the actual saved endpoint")
    budget_guard = _return_budget_guard(budget)
    return _run_checkpoint(
        source,
        state,
        method=method,
        updates=updates,
        layouts=(),
        role="saved_return_exploratory",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copier,
        extra_check=budget_guard,
        live_plan=_known_a_e_plan(),
        return_metadata=dict(
            source_updates=source_updates,
            relative_updates=relative_updates,
            source_status=source_status,
            source_failed=source_status == "failed",
            native_training_examples=owned.training_examples,
            readout_scope="exploratory_saved_memory_return_live; known TRAIN A and previously inspected E; no raw scores, final data or training",
        ),
    )


def _known_a_e_plan() -> tuple[tuple[MiniGridKeyDoorLayout, str, str], ...]:
    a = predefined_layouts()[0]
    blue, red = predefined_probe_layouts()
    return (
        (a, "blue", "train"),
        (a, "red", "train"),
        (blue, "blue", "probe_test"),
        (red, "red", "probe_test"),
    )


def evaluate_memory_probe_checkpoint(
    source: Any,
    state: Any,
    *,
    source_updates: int,
    phase: str,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    saved_fifo: bool = False,
) -> dict[str, Any]:
    """Read fixed A/F games at a genuine same-fork count-probe endpoint.

    The caller proves the preceding paid probe/return. This narrow admission
    binds every native port to its actual endpoint. The explicit saved FIFO36
    route admits only the genuinely retained after-return endpoints. Default
    refusal64 admission and older saved-return clocks remain unchanged.
    """
    if (
        type(saved_fifo) is not bool
        or type(source_updates) is not int
        or source_updates not in (0, 128)
        or type(phase) is not str
        or phase not in ("after_probe", "after_return")
        or (saved_fifo and phase != "after_return")
        or not callable(check_boundary)
    ):
        raise ValueError(
            "memory probe readout requires exact flag/source0/128, fixed phase and boundary; saved FIFO36 has only after_return states"
        )
    relative = 16 if phase == "after_probe" else 32
    updates = source_updates + relative
    expected_policy = (
        FifoActionMemoryPolicy(
            feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
        )
        if saved_fifo
        else ActionMemoryPolicy(
            feature_dim=296, num_classes=7, max_entries=64, max_payload_bytes=155136
        )
    )
    saved_endpoint = copy_invariant_memory_snapshot(state) if saved_fifo else None

    def copy_state(value: Any) -> Any:
        owned = copy_invariant_memory_snapshot(value)
        if (
            type(owned.native_state)
            is not (FifoActionMemorySnapshot if saved_fifo else ActionMemorySnapshot)
            or owned.policy.native_policy != expected_policy
            or owned.policy.encoding_id != INVARIANT_HISTORY_ENCODING_ID
            or type(owned.completed_updates) is not int
            or owned.completed_updates != updates
            or type(owned.training_examples) is not int
            or owned.training_examples != 9 * updates
        ):
            raise ValueError("memory probe port differs from its exact invariant count endpoint")
        if saved_fifo and not _same_value(owned, saved_endpoint):
            raise ValueError("saved FIFO36 port differs from its complete supplied endpoint")
        return owned

    owned = copy_state(state)
    metadata: dict[str, Any] = {}
    if saved_fifo:
        native = cast(FifoActionMemorySnapshot, owned.native_state)
        # Why: finite table recall and retirement costs stay distinct from games.
        # No F16 full state was saved, so this route never implies that readout.
        metadata = dict(
            memory_policy=asdict(native.policy),
            native_evictions=native.evictions,
            native_retired_examples=native.retired_examples,
            native_retained_entries=len(native.entries),
            native_retained_logical_payload_bytes=2424 * len(native.entries),
            native_retirement_metadata_bytes=16,
        )
    budget_guard = _return_budget_guard(budget)
    if type(budget.budget.max_replay_examples) is not int or budget.budget.max_replay_examples != 0:
        raise ValueError("memory probe readout requires zero replay authority")
    a = predefined_layouts()[0]
    blue, red = predefined_distinct_probe_layouts()
    return _run_checkpoint(
        source,
        state,
        method=owned.policy.method,
        updates=updates,
        layouts=(),
        role="saved_fifo36_probe_exploratory" if saved_fifo else "same_fork_probe_exploratory",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        extra_check=budget_guard,
        state_equal=_same_value,
        live_plan=(
            (a, "blue", "train"),
            (a, "red", "train"),
            (blue, "blue", "probe_test"),
            (red, "red", "probe_test"),
        ),
        representation_encoding=INVARIANT_HISTORY_ENCODING_ID,
        return_metadata=dict(
            **metadata,
            source_updates=source_updates,
            relative_updates=relative,
            phase=phase,
            native_training_examples=owned.training_examples,
            model_encoding=INVARIANT_HISTORY_ENCODING_ID,
            readout_scope="same-fork endpoint known TRAIN A and fixed F probe_test target-free games; no teachers, labels, updates, final data or acquired-skill/generalization claim",
        ),
    )


def _copy_wide_neural_probe_snapshot(
    state: Any, *, source_updates: int, phase: str, method: str
) -> Any:
    """Own the full fixed H32 endpoint, including its mixed TRAIN/probe clocks.

    Why: the benchmark uses54 rows per pass, while each disposable F/A update
    uses nine. A valid snapshot declares structure; its owner still proves the
    actual preceding paid phase and the historical origin of these weights.
    """
    if (
        type(source_updates) is not int
        or source_updates not in (0, 128)
        or type(phase) is not str
        or phase not in ("after_probe", "after_return")
        or type(method) is not str
        or method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
    ):
        raise ValueError("wide probe readout requires exact source0/128, phase and native method")
    relative = 16 if phase == "after_probe" else 32
    owned = copy_invariant_torch_feature_snapshot(state)
    if (
        owned.policy.native_policy != TorchFeaturePolicy(method, 296, 32, 17, num_classes=7)
        or type(owned.completed_updates) is not int
        or owned.completed_updates != source_updates + relative
        or type(owned.training_examples) is not int
        or owned.training_examples != 54 * source_updates + 9 * relative
    ):
        raise ValueError("wide probe port differs from its fixed policy or mixed native clocks")
    return owned


def evaluate_wide_neural_probe_checkpoint(
    source: Any,
    state: Any,
    *,
    source_updates: int,
    phase: str,
    method: str,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    qualified_boundary: EndpointBoundary | None = None,
    timings: NativeMiniGridEndpointTiming | None = None,
) -> dict[str, Any]:
    """Run fixed target-free A/F games on the actual disposable H32 endpoint.

    The existing caller guard proves the paid probe/return before every boundary.
    All ports must retain the complete supplied endpoint, including optimizer,
    gradients and adaptive RNG state. This grants no benchmark update authority.
    """
    if not callable(check_boundary):
        raise ValueError("completed probe boundary must be callable")
    if qualified_boundary is not None and not callable(qualified_boundary):
        raise ValueError("qualified probe boundary must be callable")
    if timings is not None and type(timings) is not NativeMiniGridEndpointTiming:
        raise ValueError(
            "wide probe timing requires an exact NativeMiniGridEndpointTiming recorder"
        )
    owned = _copy_wide_neural_probe_snapshot(
        state, source_updates=source_updates, phase=phase, method=method
    )

    def copy_state(value: Any) -> Any:
        current = _copy_wide_neural_probe_snapshot(
            value, source_updates=source_updates, phase=phase, method=method
        )
        # Why: the source is the actual disposable endpoint, unlike the older
        # TRAIN reader whose source intentionally stays at initialization.
        if not _same_state(current, owned):
            raise ValueError("wide probe source/scorer/checkpoint differs from the frozen endpoint")
        return current

    budget_guard = _return_budget_guard(budget)
    if type(budget.budget.max_replay_examples) is not int or budget.budget.max_replay_examples != 0:
        raise ValueError("wide probe readout requires zero replay authority")
    a = predefined_layouts()[0]
    blue, red = predefined_distinct_probe_layouts()
    relative = 16 if phase == "after_probe" else 32
    return _run_checkpoint(
        source,
        state,
        method=owned.policy.method,
        updates=owned.completed_updates,
        layouts=(),
        role="same_fork_probe_exploratory",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        extra_check=budget_guard,
        state_equal=_same_state,
        qualified_boundary=qualified_boundary,
        timings=timings,
        live_plan=(
            (a, "blue", "train"),
            (a, "red", "train"),
            (blue, "blue", "probe_test"),
            (red, "red", "probe_test"),
        ),
        representation_encoding=INVARIANT_HISTORY_ENCODING_ID,
        return_metadata=dict(
            source_updates=source_updates,
            relative_updates=relative,
            phase=phase,
            native_method=method,
            wide_neural_method=method,
            hidden_dim=32,
            catalogue_rows_per_update=54,
            probe_rows_per_update=9,
            native_training_examples=owned.training_examples,
            model_encoding=INVARIANT_HISTORY_ENCODING_ID,
            readout_scope="same-fork H32 endpoint known TRAIN A and fixed F probe_test target-free games; no teachers, labels, updates, final data or acquired-skill/generalization claim",
        ),
    )


def _representation_snapshot_copier(state: Any) -> Callable[[Any], Any]:
    if type(state) is ActionMemorySnapshot:
        native_copy = _return_snapshot_copier(state)

        def copy_raw(value: Any) -> Any:
            owned = native_copy(value)
            if owned.completed_updates not in (0, 32, 64, 96, 128):
                raise ValueError("raw representation port requires a declared native checkpoint")
            return owned

        return copy_raw
    if type(state) is not InvariantMemorySnapshot:
        raise ValueError("representation readout requires exact raw or encoding-bound memory")
    policy = copy_invariant_memory_snapshot(state).policy
    native_kind = type(policy.native_policy)
    if native_kind not in (ActionMemoryPolicy, FifoActionMemoryPolicy):
        raise ValueError("representation readout requires refusal or FIFO bound memory")
    expected_native = native_kind(
        feature_dim=296, num_classes=7, max_entries=64, max_payload_bytes=155136
    )
    if (
        policy.native_policy != expected_native
        or policy.encoding_id != INVARIANT_HISTORY_ENCODING_ID
    ):
        raise ValueError("representation memory policy differs from fixed296/K7/64/155136")

    def copy_state(value: Any) -> Any:
        owned = copy_invariant_memory_snapshot(value)
        if owned.policy != policy:
            raise ValueError("representation snapshot port changed its bound policy")
        if owned.training_examples != 9 * owned.completed_updates:
            raise ValueError("representation snapshot port changed its nine-row native clock law")
        if owned.completed_updates not in (0, 32, 64, 96, 128):
            raise ValueError("bound representation port requires a declared native checkpoint")
        return owned

    return copy_state


def evaluate_representation_checkpoint(
    source: Any,
    state: Any,
    *,
    updates: int,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
) -> dict[str, Any]:
    """Read fixed A/E games from raw or explicitly bound public representations.

    Why: raw stored keys must never be silently interpreted as invariant keys.
    The native snapshot type binds the encoding; decisions retain raw inputs.
    """
    if type(updates) is not int or updates not in (0, 32, 64, 96, 128):
        raise ValueError("representation readout requires a declared native checkpoint")
    if not callable(check_boundary):
        raise ValueError("completed training guard must be callable")
    copier = _representation_snapshot_copier(state)
    owned = copier(state)
    if owned.completed_updates != updates or owned.training_examples != 9 * updates:
        raise ValueError("representation native clocks differ from its checkpoint label")
    encoding = (
        "original_raw_history"
        if type(owned) is ActionMemorySnapshot
        else INVARIANT_HISTORY_ENCODING_ID
    )
    return _run_checkpoint(
        source,
        state,
        method=owned.policy.method,
        updates=updates,
        layouts=(),
        role="representation_exploratory",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copier,
        extra_check=_return_budget_guard(budget),
        live_plan=_known_a_e_plan(),
        representation_encoding=encoding,
        return_metadata=dict(
            native_training_examples=owned.training_examples,
            model_encoding=encoding,
            readout_scope="exploratory_public_representation_live; known TRAIN A and previously inspected E; no raw scores, final data or training",
        ),
    )


def evaluate_raw_neural_representation_checkpoint(
    source: Any,
    state: Any,
    *,
    updates: int,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
) -> dict[str, Any]:
    """Give the raw neural baseline the same target-free A/E readout contract.

    Why: the older raw route includes TRAIN scores and different layouts. A
    representation comparison must use the same games and zero-update budget.
    """
    if type(updates) is not int or updates not in (0, 32, 64, 96, 128):
        raise ValueError("raw neural representation requires a declared native checkpoint")
    if not callable(check_boundary):
        raise ValueError("completed training guard must be callable")
    owned = _feature_snapshot(state)
    policy = owned.policy
    if policy != TorchFeaturePolicy(policy.method, 296, 8, 17, num_classes=7):
        raise ValueError("raw neural representation policy differs from the fixed recipe")

    def copy_state(value: Any) -> Any:
        current = _feature_snapshot(value)
        if current.policy != policy:
            raise ValueError("raw neural snapshot port changed its fixed policy")
        if (
            current.completed_updates not in (0, 32, 64, 96, 128)
            or current.training_examples != 9 * current.completed_updates
        ):
            raise ValueError("raw neural snapshot port changed its native checkpoint clocks")
        return current

    owned = copy_state(owned)
    if owned.completed_updates != updates:
        raise ValueError("raw neural native clock differs from its checkpoint label")
    return _run_checkpoint(
        source,
        state,
        method=policy.method,
        updates=updates,
        layouts=(),
        role="representation_exploratory",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        state_equal=_same_state,
        extra_check=_return_budget_guard(budget),
        live_plan=_known_a_e_plan(),
        representation_encoding="original_raw_history",
        return_metadata=dict(
            native_method=policy.method,
            native_training_examples=owned.training_examples,
            model_encoding="original_raw_history",
            readout_scope="exploratory matched raw neural representation; known A and inspected E; no scores, final data or training",
        ),
    )


def evaluate_context_checkpoint(source, state, *, updates, budget, check_boundary):
    """Read fixed known TRAIN A games from the declared public-cue control."""
    from src.adapters.minigrid_context_perceptron import copy_snapshot
    from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID
    from src.app.probe_return_phase import budget_binding

    if type(budget) is not ToyBudgetSession or type(budget.budget) is not ToyExecutionBudget:
        raise ValueError("context games require the original fully spent32-update budget")
    config = budget.budget
    expected_config = replace(config)
    original_binding = budget_binding(budget)

    def budget_guard():
        if (
            budget_binding(budget) != original_binding
            or vars(config) != vars(expected_config)
            or type(config.max_training_updates) is not int
            or config.max_training_updates != 32
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 32
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("original spent context budget/clock/charges changed")

    budget_guard()

    def copy_state(value):
        owned = copy_snapshot(value)
        if (
            owned.completed_updates not in (0, 8, 16, 24, 32)
            or owned.training_examples != owned.completed_updates * 9
        ):
            raise ValueError("context game readout requires declared nine-row receipt clocks")
        return owned

    owned = copy_state(state)
    if type(updates) is not int or updates != owned.completed_updates:
        raise ValueError("context game readout label differs from its private checkpoint")
    layout = predefined_layouts()[0]
    return _run_checkpoint(
        source,
        state,
        method="public_cue_context_perceptron",
        updates=updates,
        layouts=(),
        role="knownTRAIN_A_only",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        extra_check=budget_guard,
        live_plan=((layout, "blue", "train"), (layout, "red", "train")),
        representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
        representation_feature_dim=298,
        require_initial_source=True,
        return_metadata=dict(
            native_method="categorical_unit_mistake_perceptron",
            native_training_examples=owned.training_examples,
            model_encoding=CONTEXT_HISTORY_ENCODING_ID,
            readout_scope="known TRAIN A only; raw seven-action argmax; no updates/final/probe",
        ),
    )


def evaluate_context_linear_checkpoint(source, state, *, budget, check_boundary):
    """Two known-A games from fitted affine heads; no training authority."""
    from src.adapters.minigrid_context_linear_policy import (
        ContextLinearReader,
        copy_context_linear_snapshot,
    )
    from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID
    from src.app.probe_return_phase import budget_binding

    if type(source) is not ContextLinearReader or type(budget) is not ToyBudgetSession:
        raise ValueError("affine games require exact readonly owner and readout budget")
    config = budget.budget
    if type(config) is not ToyExecutionBudget:
        raise ValueError("affine games require an explicit readout-only budget")
    expected_config, binding = replace(config), budget_binding(budget)

    def guard():
        if (
            budget_binding(budget) != binding
            or not _same_value(vars(config), vars(expected_config))
            or config.max_training_updates != 0
            or type(config.max_training_updates) is not int
            or config.max_replay_examples != 0
            or type(config.max_replay_examples) is not int
            or config.max_wall_seconds != 90
            or config.max_process_rss_bytes != 512 * 1024**2
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 0
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("affine zero-update readout budget binding changed")

    guard()
    owned = copy_context_linear_snapshot(state)
    layout = predefined_layouts()[0]
    return _run_checkpoint(
        source,
        owned,
        method="public_cue_affine_fit",
        updates=0,
        layouts=(),
        role="knownTRAIN_A_only",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_context_linear_snapshot,
        extra_check=guard,
        live_plan=((layout, "blue", "train"), (layout, "red", "train")),
        representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
        representation_feature_dim=298,
        return_metadata=dict(
            source_fit_calls=2,
            source_fit_rows=54,
            new_optimizer_updates=0,
            model_encoding=CONTEXT_HISTORY_ENCODING_ID,
            readout_scope="known TRAIN A only; fixed fitted heads, raw K7 argmax; no refit/final/probe",
        ),
    )


def evaluate_heldout_context_linear_checkpoint(source, state, *, budget, check_boundary):
    """Final D only from the frozen ordinary ten-fit endpoint; never refit/select."""
    from src.adapters.minigrid_context_linear_learner import (
        ContextLinearLearner,
        copy_context_linear_training_snapshot,
    )
    from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
    from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID

    if type(source) is not ContextLinearLearner or not callable(check_boundary):
        raise ValueError("held-out affine games require exact learner and original boundary")
    guard = _return_budget_guard(budget)
    expected_budget = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    if not _same_value(vars(budget.budget), vars(expected_budget)):
        raise ValueError("held-out affine games require original150s/512MiB zero-work budget")
    policy = ContextLinearPolicy()

    def copy_state(value):
        owned = copy_context_linear_training_snapshot(value)
        if (
            owned.policy != policy
            or owned.failed_work is not None
            or (owned.completed_updates, owned.training_examples) not in ((0, 0), (10, 252))
            or (
                owned.completed_updates == 10
                and any(head is None or head.fit_rows != 36 for head in owned.heads)
            )
        ):
            raise ValueError("held-out affine games require original0 or fixed10/252/head36 data")
        return owned

    owned = copy_state(state)
    initial = copy_state(source.snapshot_state())
    if owned.completed_updates != 10 or initial.completed_updates != 0:
        raise ValueError("held-out affine readout requires source0 and ordinary final10 endpoint")
    guard()
    layout = predefined_layouts()[3]
    frame_scope = (
        "one148-value held-out D final frame per active episode; never training; "
        "numeric bytes exclude native/Python/copy overhead"
    )
    try:
        report = _run_checkpoint(
            source,
            owned,
            method="arrived_public_cue_affine_fit",
            updates=10,
            layouts=(),
            role="final_test",
            sources=(),
            budget=budget,
            check_boundary=check_boundary,
            public_history=True,
            memory_copier=copy_state,
            state_equal=_same_value,
            extra_check=guard,
            require_initial_source=True,
            live_plan=((layout, "blue", "final_test"), (layout, "red", "final_test")),
            representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
            representation_feature_dim=298,
            return_metadata=dict(
                source_fit_calls=10,
                source_fit_rows=252,
                new_fit_calls=0,
                new_optimizer_updates=0,
                model_encoding=CONTEXT_HISTORY_ENCODING_ID,
                readout_scope="fixed final D blue/red from ordinary affine10/252; no training/probe/selection/promotion; not a globally unseen benchmark",
            ),
        )
    except MiniGridEvaluationFailed as error:
        error.report["public_history"]["storage_scope"] = frame_scope
        raise
    report["public_history"]["storage_scope"] = frame_scope
    return report


def evaluate_context_linear_stream_checkpoint(
    source,
    state,
    *,
    budget,
    check_boundary,
    retained_catalogue=False,
    after_c_capacity=False,
    recurring_catalogue=False,
    development_a4=False,
    a4_training=False,
    a4_current_only=False,
):
    """Known current-phase or retained old/new games; no new updates."""
    from src.adapters.minigrid_context_linear_learner import (
        ContextLinearLearner,
        copy_context_linear_training_snapshot,
    )
    from src.app.probe_return_phase import budget_binding
    from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID

    if type(source) is not ContextLinearLearner or type(budget) is not ToyBudgetSession:
        raise ValueError("causal affine games require exact learner and readout budget")
    if type(a4_current_only) is not bool or (
        a4_current_only
        and any(
            (
                a4_training,
                retained_catalogue,
                after_c_capacity,
                recurring_catalogue,
                development_a4,
            )
        )
    ):
        raise ValueError("current-only A4 readout requires an exact distinct mode")
    if type(a4_training) is not bool or (
        a4_training
        and any(
            (
                retained_catalogue,
                after_c_capacity,
                recurring_catalogue,
                development_a4,
            )
        )
    ):
        raise ValueError("A4 TRAIN readout requires an exact distinct mode")
    if type(development_a4) is not bool or (development_a4 and recurring_catalogue is not True):
        raise ValueError("development A4 requires an exact flag and ordinary recurring endpoint")
    if (
        type(retained_catalogue) is not bool
        or type(after_c_capacity) is not bool
        or type(recurring_catalogue) is not bool
        or (retained_catalogue and after_c_capacity)
        or (recurring_catalogue and (retained_catalogue or after_c_capacity))
    ):
        raise ValueError("exact mutually exclusive retained/after-C flags required")
    owned = copy_context_linear_training_snapshot(state)
    updates = owned.completed_updates
    if (
        updates
        not in (
            (10,)
            if recurring_catalogue
            else (6,)
            if after_c_capacity
            else (4,)
            if retained_catalogue
            else (2, 4, 6)
        )
        or owned.training_examples
        != (
            {2: 18, 4: 54, 6: 90}.get(updates, -1)
            if a4_training
            else 252
            if recurring_catalogue
            else 108
            if after_c_capacity
            else 54
            if retained_catalogue
            else updates * 9
        )
        or owned.failed_work is not None
        or any(
            head is None
            or head.fit_rows
            != (
                9
                if a4_training and updates == 2
                else 18
                if a4_training
                else 36
                if recurring_catalogue
                else 27
                if after_c_capacity
                else 18
                if retained_catalogue
                else 9
            )
            for head in owned.heads
        )
    ):
        raise ValueError("causal game checkpoint fit/row geometry differs from its declared recipe")
    config = budget.budget
    if type(config) is not ToyExecutionBudget:
        raise ValueError("causal affine games require an explicit readout-only budget")
    expected_config, binding = replace(config), budget_binding(budget)

    def guard():
        if (
            budget_binding(budget) != binding
            or not _same_value(vars(config), vars(expected_config))
            or type(config.max_training_updates) is not int
            or config.max_training_updates != 0
            or type(config.max_replay_examples) is not int
            or config.max_replay_examples != 0
            or config.max_wall_seconds != 150
            or config.max_process_rss_bytes != 512 * 1024**2
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 0
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("causal affine zero-update readout budget binding changed")

    def copy_state(value):
        current = copy_context_linear_training_snapshot(value)
        if current.policy != owned.policy or current.failed_work is not None:
            raise ValueError("causal affine checkpoint policy or failure evidence changed")
        return current

    guard()
    from src.core.minigrid_key_door import predefined_recurring_train_layout

    layout = predefined_layouts()[1 if updates == 4 else 0]
    live_plan = (
        tuple(
            (item, cue, "train")
            for item in (predefined_layouts()[0], MiniGridKeyDoorLayout("A", branch_row=4))
            for cue in ("blue", "red")
        )
        if a4_training or a4_current_only
        else tuple(
            (MiniGridKeyDoorLayout("A", branch_row=4), cue, "probe_test") for cue in ("blue", "red")
        )
        if development_a4
        else tuple(
            (item, cue, "train")
            for item in (predefined_recurring_train_layout(), predefined_layouts()[0])
            for cue in ("blue", "red")
        )
        if recurring_catalogue
        else tuple(
            (item, cue, "train") for item in predefined_layouts()[1:3] for cue in ("blue", "red")
        )
        if after_c_capacity
        else tuple(
            (item, cue, "train") for item in predefined_layouts()[:2] for cue in ("blue", "red")
        )
        if retained_catalogue
        else ((layout, "blue", "train"), (layout, "red", "train"))
    )
    return _run_checkpoint(
        source,
        owned,
        method="arrived_public_cue_affine_fit",
        updates=updates,
        layouts=(),
        role="knownTRAIN_A3A4_current_only"
        if a4_current_only
        else "knownTRAIN_A3A4_adaptation"
        if a4_training
        else "development_A4_from_ordinary_MAIN"
        if development_a4
        else "knownTRAIN_GA_return"
        if recurring_catalogue
        else "knownTRAIN_BC_afterC"
        if after_c_capacity
        else "knownTRAIN_AB_retained"
        if retained_catalogue
        else "knownTRAIN_ABA_current_phase",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        extra_check=guard,
        require_initial_source=True,
        live_plan=live_plan,
        representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
        representation_feature_dim=298,
        return_metadata=dict(
            **({"evaluation_purpose": "development"} if development_a4 else {}),
            source_fit_calls=updates,
            source_fit_rows=owned.training_examples,
            new_fit_calls=0,
            new_optimizer_updates=0,
            model_encoding=CONTEXT_HISTORY_ENCODING_ID,
            readout_scope=(
                "known TRAIN A3/A4 at fixed causal checkpoint; no refit/final/probe"
                if a4_training or a4_current_only
                else "A branch4 development observations via probe_test role; ordinary main state; no probe training or refit/final"
                if development_a4
                else "known G_train then returned A TRAIN; ten252-row causal fit; no refit/final/probe"
                if recurring_catalogue
                else "known B and NEW C_train games afterC; capacity2 saved fit; raw K7 argmax; no refit/final/probe"
                if after_c_capacity
                else "known OLD A and NEW B TRAIN games afterB; retained saved fit; raw K7 argmax; no refit/final/probe"
                if retained_catalogue
                else "known current TRAIN phase only; causal saved fits; raw K7 argmax; no refit/final/probe"
            ),
        ),
    )


def evaluate_context_linear_probe_checkpoint(source, state, *, phase, budget, check_boundary):
    """Execute fixed F probe_test games from an owned before/after snapshot.

    No probe fitting or labels enter this binding. The ordinary game loop owns
    scorer restoration, raw K7 decisions, public history, cleanup and RNG checks.
    """
    from src.adapters.minigrid_context_linear_learner import (
        ContextLinearLearner,
        copy_context_linear_training_snapshot,
    )
    from src.app.probe_return_phase import budget_binding
    from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID
    from src.core.minigrid_key_door import predefined_distinct_probe_layouts

    if type(source) is not ContextLinearLearner or type(budget) is not ToyBudgetSession:
        raise ValueError("F games require exact learner and readout budget")
    if type(phase) is not str or phase not in ("before_probe", "after_probe"):
        raise ValueError("F games require an explicit before_probe or after_probe phase")
    owned = copy_context_linear_training_snapshot(state)
    updates, rows, head_rows = (6, 108, 27) if phase == "before_probe" else (8, 126, 9)
    if (
        (owned.completed_updates, owned.training_examples) != (updates, rows)
        or owned.failed_work is not None
        or any(head is None or head.fit_rows != head_rows for head in owned.heads)
    ):
        raise ValueError("F game checkpoint geometry differs from the paid probe recipe")
    config = budget.budget
    if type(config) is not ToyExecutionBudget:
        raise ValueError("F games require an explicit readout-only budget")
    expected_config, binding = replace(config), budget_binding(budget)

    def guard():
        if (
            budget_binding(budget) != binding
            or not _same_value(vars(config), vars(expected_config))
            or type(config.max_training_updates) is not int
            or config.max_training_updates != 0
            or type(config.max_replay_examples) is not int
            or config.max_replay_examples != 0
            or config.max_wall_seconds != 150
            or config.max_process_rss_bytes != 512 * 1024**2
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 0
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("F game zero-update readout budget binding changed")

    def copy_state(value):
        current = copy_context_linear_training_snapshot(value)
        if current.policy != owned.policy or current.failed_work is not None:
            raise ValueError("F game checkpoint policy or failure evidence changed")
        return current

    def label_history(report):
        # Why: the shared loop also serves TRAIN games; this invocation owns
        # only a target-free probe_test initial frame, never a TRAIN packet.
        if "public_history" in report:
            report["public_history"]["storage_scope"] = (
                "one148-value public probe_test initial frame per active episode; "
                "numericfloat64 bytes exclude native/Python/report/copy overhead; no readout updates"
            )
        return report

    guard()
    try:
        report = _run_checkpoint(
            source,
            owned,
            method="arrived_public_cue_affine_fit",
            updates=updates,
            layouts=(),
            role="isolated_affine_F_probe_test",
            sources=(),
            budget=budget,
            check_boundary=check_boundary,
            public_history=True,
            memory_copier=copy_state,
            extra_check=guard,
            require_initial_source=True,
            live_plan=tuple(
                (layout, cue, "probe_test")
                for layout, cue in zip(predefined_distinct_probe_layouts(), ("blue", "red"))
            ),
            representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
            representation_feature_dim=298,
            return_metadata=dict(
                probe_phase=phase,
                source_fit_calls=updates,
                source_fit_rows=rows,
                new_fit_calls=0,
                new_optimizer_updates=0,
                model_encoding=CONTEXT_HISTORY_ENCODING_ID,
                readout_scope="fixed F probe_test games; raw K7 argmax; no refit/final/promotion",
            ),
        )
    except MiniGridEvaluationFailed as error:
        label_history(error.report)
        raise
    return label_history(report)


def evaluate_context_linear_rehearsal_checkpoint(source, state, *, arm, budget, check_boundary):
    """Read fixed F then returned A from the matched private two-fit endpoints."""
    from src.adapters.minigrid_context_linear_learner import (
        ContextLinearLearner,
        copy_context_linear_training_snapshot,
    )
    from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID
    from src.core.minigrid_key_door import predefined_distinct_probe_layouts

    if (
        type(source) is not ContextLinearLearner
        or type(arm) is not str
        or arm not in ("current_only", "joint_ABC_F")
    ):
        raise ValueError("private return games require exact learner and matched arm")
    guard = _return_budget_guard(budget)
    expected_budget = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    if not _same_value(vars(budget.budget), vars(expected_budget)):
        raise ValueError(
            "private return games require original150s/512MiB zero-work readout budget"
        )
    owned = copy_context_linear_training_snapshot(state)
    if (
        (owned.completed_updates, owned.training_examples) != (8, 180)
        or owned.failed_work is not None
        or any(h is None or h.fit_rows != 36 for h in owned.heads)
    ):
        raise ValueError("private return games require actual8/180/two36-row heads")

    def copy_state(value):
        current = copy_context_linear_training_snapshot(value)
        if current.policy != owned.policy or current.failed_work is not None:
            raise ValueError("private return policy or failed-work evidence changed")
        return current

    def label_history(report):
        if "public_history" in report:
            report["public_history"]["storage_scope"] = (
                "one148-value public initial frame per active F probe_test or A TRAIN episode; "
                "numericfloat64 bytes exclude native/Python/report/copy overhead; no readout updates"
            )
        return report

    plan = tuple(
        (layout, cue, "probe_test")
        for layout, cue in zip(predefined_distinct_probe_layouts(), ("blue", "red"))
    ) + tuple((predefined_layouts()[0], cue, "train") for cue in ("blue", "red"))
    try:
        report = _run_checkpoint(
            source,
            owned,
            method="arrived_public_cue_affine_fit",
            updates=8,
            layouts=(),
            role="private_F_then_returned_A",
            sources=(),
            budget=budget,
            check_boundary=check_boundary,
            public_history=True,
            memory_copier=copy_state,
            extra_check=guard,
            require_initial_source=True,
            live_plan=plan,
            representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
            representation_feature_dim=298,
            return_metadata=dict(
                rehearsal_arm=arm,
                source_fit_calls=8,
                source_fit_rows=180,
                new_fit_calls=0,
                new_optimizer_updates=0,
                model_encoding=CONTEXT_HISTORY_ENCODING_ID,
                readout_scope="fixed F probe_test then known returned A TRAIN; raw K7 argmax; no fit/final/promotion",
            ),
        )
    except MiniGridEvaluationFailed as error:
        label_history(error.report)
        raise
    return label_history(report)


def evaluate_standardized_neural_checkpoint(
    source: Any,
    state: Any,
    *,
    updates: int,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
) -> dict[str, Any]:
    """Read both known TRAIN A games from the explicit standardized BP control."""
    from src.adapters.standardized_minigrid_learner import ENCODING_ID, copy_snapshot

    owned = copy_snapshot(state)
    if type(updates) is not int or updates not in (0, 128):
        raise ValueError("standardized live readout requires exact source0 or endpoint128")
    policy = owned.policy
    if (
        policy.statistics.fitted_rows != 54
        or owned.completed_updates != updates
        or owned.training_examples != updates * 54
    ):
        raise ValueError("standardized live readout requires the fixed54-row source/clock")

    def copy_state(value: Any) -> Any:
        current = copy_snapshot(value)
        if current.policy != policy or (
            current.completed_updates,
            current.training_examples,
        ) not in ((0, 0), (128, 6912)):
            raise ValueError("standardized live port changed its transform/native clock")
        return current

    layout = predefined_layouts()[0]
    return _run_checkpoint(
        source,
        state,
        method=policy.method,
        updates=updates,
        layouts=(),
        role="knownTRAIN_A_only",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        state_equal=_same_state,
        extra_check=_return_budget_guard(budget),
        live_plan=((layout, "blue", "train"), (layout, "red", "train")),
        representation_encoding=ENCODING_ID,
        require_initial_source=True,
        return_metadata=dict(
            native_method="bp",
            native_training_examples=owned.training_examples,
            model_encoding=ENCODING_ID,
            standardization_fitted_rows=54,
            readout_scope="known TRAIN A only; raw argmax, no live updates/final/probe/promotion",
        ),
    )


def _copy_public_cue_probe_snapshot(state, *, source_updates, phase):
    """Only the fixed two-warm-update, balanced F16/A16 probe recipe."""
    from src.adapters.minigrid_public_cue_recurrent_learner import (
        PublicCueRecurrentPolicy,
        copy_snapshot,
    )

    if (
        type(source_updates) is not int
        or source_updates != 2
        or type(phase) is not str
        or phase not in ("after_probe", "after_return")
    ):
        raise ValueError("public-cue probe requires source2 and an exact paid F/return phase")
    count = source_updates + (16 if phase == "after_probe" else 32)
    owned = copy_snapshot(state)
    if (
        owned.policy != PublicCueRecurrentPolicy()
        or owned.failed_work is not None
        or (owned.completed_updates, owned.training_examples) != (count, 9 * count)
        or any(
            (h.completed_updates, h.training_examples) != (count // 2, 9 * count // 2)
            for h in owned.heads
        )
    ):
        raise ValueError("public-cue probe policy/native/per-head clocks differ from paid work")
    return owned


def evaluate_public_cue_probe_checkpoint(
    source, state, *, source_updates, phase, budget, check_boundary, qualified_boundary=None
):
    """Target-free F after probe or A after return on the actual disposable state.

    Why this: a probe source is already trained. Require exact full equality
    with its paid endpoint instead of applying the benchmark's cold-source rule.
    Native session/argmax/cleanup/preservation behavior stays in the shared kernel.
    """
    from src.adapters.minigrid_public_cue_recurrent_learner import METHOD, PublicCueRecurrentLearner

    if (
        type(source) is not PublicCueRecurrentLearner
        or not callable(check_boundary)
        or qualified_boundary is not None
        and not callable(qualified_boundary)
    ):
        raise ValueError("public-cue probe requires its exact learner and completed-work boundary")
    owned = _copy_public_cue_probe_snapshot(state, source_updates=source_updates, phase=phase)
    guard = _return_budget_guard(budget)
    expected_budget = ToyExecutionBudget(
        max_training_updates=0, max_replay_examples=0, max_wall_seconds=40.0
    )
    if not _same_value(vars(budget.budget), vars(expected_budget)):
        raise ValueError("public-cue probe games require original40s zero-work readout budget")

    def copy_state(value):
        current = _copy_public_cue_probe_snapshot(value, source_updates=source_updates, phase=phase)
        if not _same_value(current, owned):
            raise ValueError("public-cue live source/scorer/checkpoint differs from paid endpoint")
        return current

    a = predefined_layouts()[0]
    blue, red = predefined_distinct_probe_layouts()
    role = "probe_test" if phase == "after_probe" else "train"
    plan = (
        ((blue, "blue", role), (red, "red", role))
        if phase == "after_probe"
        else ((a, "blue", role), (a, "red", role))
    )
    return _run_checkpoint(
        source,
        owned,
        method=METHOD,
        updates=owned.completed_updates,
        layouts=(),
        role=role,
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=False,
        memory_copier=copy_state,
        state_equal=_same_value,
        extra_check=guard,
        qualified_boundary=qualified_boundary,
        live_plan=plan,
        return_metadata=dict(
            source_updates=source_updates,
            phase=phase,
            native_training_updates=owned.completed_updates,
            native_training_examples=owned.training_examples,
            native_head_updates=[h.completed_updates for h in owned.heads],
            logical_probe_updates=16,
            logical_return_updates=16 if phase == "after_return" else 0,
            readout_scope="same disposable public-cue learner; target-free F probe_test after F16 or A TRAIN after return16; no final/update/promotion",
        ),
    )


def evaluate_public_cue_recurrent_checkpoint(
    source,
    state,
    *,
    updates,
    sources,
    budget,
    check_boundary,
    recurring_training=False,
    recurrent_replay=None,
    class_balanced_training=False,
):
    """Read fixed-A or explicitly declared recurring known-TRAIN checkpoints."""
    from src.adapters.minigrid_public_cue_recurrent_learner import (
        METHOD,
        PublicCueRecurrentLearner,
        PublicCueRecurrentPolicy,
        copy_snapshot,
    )

    policy = PublicCueRecurrentPolicy()
    if type(source) is not PublicCueRecurrentLearner or not callable(check_boundary):
        raise ValueError("public-cue readout requires its exact learner and completed-work guard")
    if type(recurring_training) is not bool:
        raise ValueError("recurring_training must be an exact boolean")
    if type(class_balanced_training) is not bool or (
        class_balanced_training and (not recurring_training or recurrent_replay is not None)
    ):
        raise ValueError("class-balanced readout requires recurring TRAIN without replay")
    if recurrent_replay is not None:
        from src.adapters import minigrid_recurrent_replay as replay

        replay.validate_mode(recurrent_replay)
        if not recurring_training:
            raise ValueError("replay readout requires the recurring TRAIN recipe")
    points = (0, 32, 64, 96, 128) if recurring_training else (0, 128)
    native_points = (
        tuple(replay.native_updates(p) for p in points) if recurrent_replay is not None else points
    )
    contexts = ("A", "B", "C_train") if recurring_training else ("A",)
    if type(updates) is not int or updates not in points:
        raise ValueError("public-cue readout requires fixed0/128 checkpoint")
    if type(sources) is not tuple or len(sources) != 2 * len(contexts):
        raise ValueError("public-cue readout requires original two TRAIN A sources")
    seen: set[str] = set()
    for i, packet in enumerate(sources):
        context, cue = contexts[i // 2], ("blue", "red")[i % 2]
        _validate_batch(packet, "train", seen)
        if (
            packet.context != context
            or packet.num_classes != 7
            or packet.features.shape != (9, 148)
            or packet.episode is None
            or packet.episode.episode_id != f"minigrid-source-{context}-{cue}"
        ):
            raise ValueError("public-cue readout requires ordered original raw148 TRAIN A pair")
        if recurring_training:
            from src.adapters.minigrid_public_cue_recurrent_learner import _cue

            if _cue(packet.features[0]) != i % 2:
                raise ValueError("recurring readout source lost its actual initial public cue")

    def copy_state(value):
        owned = copy_snapshot(value)
        if (
            owned.policy != policy
            or owned.failed_work is not None
            or owned.completed_updates not in native_points
            or any(
                (h.completed_updates, h.training_examples)
                != (owned.completed_updates // 2, owned.completed_updates * 9 // 2)
                for h in owned.heads
            )
        ):
            raise ValueError(
                "public-cue readout requires original policy and exact per-head clocks"
            )
        return owned

    owned = copy_state(state)
    expected_updates = replay.native_updates(updates) if recurrent_replay is not None else updates
    if owned.completed_updates != expected_updates:
        raise ValueError("public-cue checkpoint and declared point differ")
    guard = _return_budget_guard(budget)
    expected = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=200 if recurring_training else 100,
        max_process_rss_bytes=768 * 1024**2,
    )
    if not _same_value(vars(budget.budget), vars(expected)):
        raise ValueError("public-cue readout requires original100s/768MiB zero-work budget")
    layouts = (
        predefined_layouts()[:3]
        if recurring_training and updates == 128
        else (
            predefined_layouts()[:1]
            if (
                not recurring_training
                or updates == 32
                or (class_balanced_training and updates in (64, 96))
            )
            else ()
        )
    )
    replay_options: dict[str, Any] = {}
    if class_balanced_training:
        replay_options["return_metadata"] = dict(class_balanced_training=True)
    if recurrent_replay is not None:
        replay_options["return_metadata"] = dict(
            recurrent_replay=recurrent_replay,
            logical_wake_updates=updates,
            native_training_updates=owned.completed_updates,
            native_training_examples=owned.training_examples,
            native_head_updates=[h.completed_updates for h in owned.heads],
        )
    return _run_checkpoint(
        source,
        owned,
        method=METHOD,
        updates=updates,
        layouts=(),
        role="train",
        sources=sources,
        budget=budget,
        check_boundary=check_boundary,
        public_history=False,
        memory_copier=copy_state,
        state_equal=_same_value,
        extra_check=guard,
        require_initial_source=True,
        live_plan=tuple((layout, cue, "train") for layout in layouts for cue in ("blue", "red")),
        **replay_options,
    )


def evaluate_context_neural_bp_checkpoint(source, state, *, budget, check_boundary):
    """Known A only from the declared conventional public-cue BP endpoint."""
    from src.adapters.minigrid_context_torch_feature_learner import (
        ContextTorchFeatureLearner,
        ContextTorchFeaturePolicy,
        copy_context_torch_feature_snapshot,
    )
    from src.adapters.torch_feature_learner import TorchFeaturePolicy
    from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID

    expected_policy = ContextTorchFeaturePolicy(
        TorchFeaturePolicy("bp", 296, 32, 17, num_classes=7)
    )
    if type(source) is not ContextTorchFeatureLearner:
        raise ValueError("context neural games require the exact two-head learner")
    owned = copy_context_torch_feature_snapshot(state)
    if (
        owned.policy != expected_policy
        or owned.failed_work is not None
        or (owned.completed_updates, owned.training_examples) != (128, 1152)
        or any((h.completed_updates, h.training_examples) != (64, 576) for h in owned.heads)
    ):
        raise ValueError(
            "context neural known-A endpoint requires declared BP128/1152/two64-update heads"
        )
    guard = _return_budget_guard(budget)
    expected_budget = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=768 * 1024**2,
    )
    if not _same_value(vars(budget.budget), vars(expected_budget)):
        raise ValueError("context neural games require original150s/768MiB zero-work budget")

    def copy_state(value):
        current = copy_context_torch_feature_snapshot(value)
        if current.policy != expected_policy or current.failed_work is not None:
            raise ValueError("context neural readout policy or failure evidence changed")
        return current

    layout = predefined_layouts()[0]
    return _run_checkpoint(
        source,
        owned,
        method=owned.policy.method,
        updates=128,
        layouts=(),
        role="knownTRAIN_A_context_neural_bp",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        state_equal=_same_value,
        extra_check=guard,
        require_initial_source=True,
        live_plan=((layout, "blue", "train"), (layout, "red", "train")),
        representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
        representation_feature_dim=298,
        return_metadata=dict(
            source_native_updates=128,
            source_native_rows=1152,
            new_fit_calls=0,
            new_optimizer_updates=0,
            native_method="bp",
            readout_scope="known TRAIN A only; two public-cue native BP heads; no refit/final/probe/promotion",
        ),
    )


def evaluate_neural_representation_checkpoint(
    source: Any,
    state: Any,
    *,
    updates: int,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    known_a_only: bool = False,
    catalogue_method: str | None = None,
    wide_method: str | None = None,
) -> dict[str, Any]:
    """Read fixed A/E games from an explicitly encoding-bound native Torch head.

    Original raw Torch/count routes keep their own admission. Native objectives
    and complete optimizer/adaptive/RNG state are validated before any callback.
    The optional known-A readout uses only the original two TRAIN game slots;
    restricting the fixed plan adds no input or action-selection authority.
    Explicit H32 methods bind only their fixed128-pass/6912-row TRAIN endpoint.
    """
    if type(known_a_only) is not bool:
        raise ValueError("known_a_only must be an exact bool")
    if wide_method is not None and (
        type(wide_method) is not str
        or wide_method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
        or known_a_only is not True
        or catalogue_method is not None
        or type(updates) is not int
        or updates != 128
    ):
        raise ValueError("wide live readout requires an exact native method at128/A only")
    if catalogue_method is not None:
        from src.adapters.minigrid_learning_pilot import _INVARIANT_FEATURE_METHODS

        if (
            type(catalogue_method) is not str
            or catalogue_method not in _INVARIANT_FEATURE_METHODS
            or known_a_only is not True
            or type(updates) is not int
            or updates != 128
        ):
            raise ValueError(
                "catalogue live readout requires an exact declared method at128/A only"
            )
    if type(updates) is not int or updates not in (0, 32, 64, 96, 128):
        raise ValueError("neural representation requires a declared native checkpoint")
    if not callable(check_boundary):
        raise ValueError("completed training guard must be callable")
    owned = copy_invariant_torch_feature_snapshot(state)
    policy = owned.policy
    native = policy.native_policy
    if catalogue_method is None and native != TorchFeaturePolicy(
        native.method if wide_method is None else wide_method,
        296,
        8 if wide_method is None else 32,
        17,
        num_classes=7,
    ):
        raise ValueError("neural representation policy differs from the fixed recipe")

    def copy_state(value: Any) -> Any:
        current = copy_invariant_torch_feature_snapshot(value)
        if current.policy != policy:
            raise ValueError("neural representation snapshot port changed its bound policy")
        if wide_method is not None:
            # Why: the frozen H32 joint TRAIN recipe uses54 rows per pass. Its
            # original source remains0; no intermediate or inherited H8 clock
            # can stand in for the genuine128/6912 endpoint.
            if (
                type(current.completed_updates) is not int
                or type(current.training_examples) is not int
                or (current.completed_updates, current.training_examples)
                not in ((0, 0), (128, 6912))
            ):
                raise ValueError("wide live port requires source0/0 or endpoint128/6912")
        elif catalogue_method is not None:
            from src.adapters.minigrid_learning_pilot import _invariant_feature_counter

            # Why: keep exact full recipe and54-row exposure admission in the
            # shared TRAIN authority; aliases never stand in for native identity.
            if type(current.completed_updates) is not int or current.completed_updates not in (
                0,
                128,
            ):
                raise ValueError("catalogue live port requires source0 or endpoint128")
            _invariant_feature_counter(
                current, current.completed_updates, catalogue_method, catalogue_rows=54
            )
        elif (
            current.completed_updates not in (0, 32, 64, 96, 128)
            or current.training_examples != 9 * current.completed_updates
        ):
            raise ValueError("neural representation port changed its native checkpoint clocks")
        return current

    owned = copy_state(owned)
    if owned.completed_updates != updates:
        raise ValueError("neural representation native clock differs from its label")
    plan: tuple[tuple[MiniGridKeyDoorLayout, str, str], ...]
    if known_a_only:
        layout = predefined_layouts()[0]
        plan = ((layout, "blue", "train"), (layout, "red", "train"))
    else:
        plan = _known_a_e_plan()
    return _run_checkpoint(
        source,
        state,
        method=policy.method if catalogue_method is None else catalogue_method,
        updates=updates,
        layouts=(),
        role="knownTRAIN_A_only" if known_a_only else "representation_exploratory",
        sources=(),
        budget=budget,
        check_boundary=check_boundary,
        public_history=True,
        memory_copier=copy_state,
        state_equal=_same_state,
        extra_check=_return_budget_guard(budget),
        live_plan=plan,
        representation_encoding=INVARIANT_HISTORY_ENCODING_ID,
        require_initial_source=catalogue_method is not None or wide_method is not None,
        return_metadata=dict(
            native_method=native.method,
            native_training_examples=owned.training_examples,
            model_encoding=INVARIANT_HISTORY_ENCODING_ID,
            **({} if catalogue_method is None else dict(catalogue_rows_per_update=54)),
            **(
                {}
                if wide_method is None
                else dict(
                    wide_neural_method=wide_method,
                    hidden_dim=32,
                    catalogue_rows_per_update=54,
                )
            ),
            readout_scope=(
                "knownTRAIN_A_only: original TRAIN A blue/red games only; no E, probe or final data, raw scores or training; exploratory fit-to-live diagnostic, no acquisition/generalization claim"
                if known_a_only
                else "exploratory neural representation; known TRAIN A and inspected E; no raw scores, final data or training"
            ),
        ),
    )


def _live_slot(layout: MiniGridKeyDoorLayout, cue: str, role: str, status: str) -> dict[str, Any]:
    return dict(
        context=layout.context,
        cue=cue,
        source_role=role,
        status=status,
        reason="not reached" if status == "not_run" else None,
        predictions=[],
        prediction_calls_attempted=0,
        prediction_calls_returned=0,
        environment=None,
        state_unchanged=None,
    )


def _run_checkpoint(
    source: Any,
    state: Any,
    *,
    method: str,
    updates: int,
    layouts: tuple[MiniGridKeyDoorLayout, ...],
    role: str,
    sources: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    public_history: bool,
    native_training_examples: int | None = None,
    memory_copier: Callable[[Any], Any] | None = None,
    extra_check: Callable[[], None] | None = None,
    live_plan: tuple[tuple[MiniGridKeyDoorLayout, str, str], ...] | None = None,
    return_metadata: dict[str, Any] | None = None,
    representation_encoding: str | None = None,
    representation_feature_dim: int = 296,
    state_equal: Callable[[Any, Any], bool] | None = None,
    require_initial_source: bool = False,
    qualified_boundary: EndpointBoundary | None = None,
    timings: MiniGridEndpointTiming | None = None,
) -> dict[str, Any]:
    """Execute already-admitted old readouts or the private fixed A/E contract."""
    if type(representation_feature_dim) is not int or representation_feature_dim not in (296, 298):
        raise ValueError("bound MiniGrid projection dimension must be296 or298")
    if timings is not None and type(timings) not in (
        MiniGridEndpointTiming,
        NativeMiniGridEndpointTiming,
    ):
        raise ValueError("readout timing requires an exact MiniGridEndpointTiming recorder")
    history_features = public_history and (
        method not in _RECURRENT_METHODS or method == _PAIRED_METHOD
    )
    work = dict(
        raw_score_forwards_attempted=0,
        raw_score_forwards_completed=0,
        raw_score_rows_attempted=0,
        raw_score_rows_completed=0,
        live_prediction_calls_attempted=0,
        live_prediction_calls_returned=0,
        live_sessions_started=0,
        environment_resets_attempted=0,
        environment_resets_completed=0,
        environment_steps_attempted=0,
        environment_steps_returned=0,
        failure_work_may_be_unobserved=False,
    )
    report: dict[str, Any] = dict(
        updates=updates,
        role=role,
        method=method,
        status="running",
        reason=None,
        scores=[],
        episodes=[],
        work=work,
        state_unchanged=None,
        live_encoding="full148_normalized_before_after; redundant_raw_next_images_omitted",
    )
    if return_metadata is not None:
        report.update(return_metadata)
    if representation_encoding is not None:
        work.update(
            internal_projection_rows_returned=0,
            internal_projection_array_bytes_returned=0,
        )
        report["internal_projection_accounting"] = (
            f"additional{representation_feature_dim}-float64 array bytes for returned bound prediction calls only; "
            "raw baseline adds zero; failed-attempt allocations and Python/native/copy overhead unknown"
        )
    if live_plan is not None:
        report["episodes"].extend(
            _live_slot(layout, cue, slot_role, "not_run") for layout, cue, slot_role in live_plan
        )
    if method in _WARM_METHODS:
        report["native_training_examples"] = native_training_examples
    if public_history:
        report["public_history"] = dict(
            enabled=True,
            applied=history_features,
            observation_dim=148,
            prediction_feature_dim=296 if history_features else 148,
            initial_numeric_bytes=148 * 8 if history_features else 0,
            storage_scope="one148-value public initial frame per active FF episode; numericfloat64 bytes exclude Python/native/report/copy overhead; RNN retains original148 inputs",
        )
        work.update(
            history_episodes_started=0,
            history_live_feature_rows=0,
            history_live_feature_array_bytes=0,
            history_raw_episodes_encoded=0,
            history_raw_feature_array_bytes=0,
            history_raw_target_array_bytes=0,
        )
        if history_features:
            report["live_encoding"] = (
                "raw148_normalized_before_after; features296_public_initial_plus_current; redundant_raw_next_images_omitted"
            )
            if method == _PAIRED_METHOD:
                report["public_history"]["storage_scope"] = (
                    "one148-value public initial frame per active paired recurrent episode; "
                    "numericfloat64 bytes exclude recurrent activation/native/Python/report/copy overhead"
                )
    if return_metadata is not None and public_history:
        report["public_history"]["storage_scope"] = (
            "one148-value public initial frame per saved-memory episode; numericfloat64 bytes "
            "exclude native/Python/report/copy overhead; no training authority"
        )
        if require_initial_source:
            report["public_history"]["storage_scope"] = (
                "one148-value public development initial frame per active episode; numericfloat64 bytes "
                "exclude native/Python/report/copy overhead; no training authority"
                if role == "development_A4_from_ordinary_MAIN"
                else "one148-value original TRAIN initial frame per active episode; numericfloat64 bytes "
                "exclude native/Python/report/copy overhead; raw sources remain148 and no readout updates"
            )
    expected = deepcopy(state) if memory_copier is None else memory_copier(state)
    source_before = scorer = None
    # Count memory has no Torch backend. Neural paths retain their original
    # exact comparator, including its dependency and exception behavior.
    same_state: Callable[[Any, Any], bool] = (
        (_same_value if method == _MEMORY_METHOD else _same_state)
        if memory_copier is None
        else lambda left, right: bool(left == right)
    )

    if state_equal is not None:
        same_state = state_equal

    def checkpoint_state() -> Any:
        return state if memory_copier is None else memory_copier(state)

    def read_snapshot(owner: Any) -> Any:
        value = owner.snapshot_state()
        owned = value if memory_copier is None else memory_copier(value)
        if (
            require_initial_source
            and owner is source
            and (
                type(owned.completed_updates) is not int
                or owned.completed_updates != 0
                or type(owned.training_examples) is not int
                or owned.training_examples != 0
            )
        ):
            raise ValueError("recurring memory readout requires its unchanged native0 source")
        return owned

    def boundary() -> None:
        if extra_check is not None:
            extra_check()
        budget.before_final()
        if extra_check is not None:
            extra_check()
        check_boundary()
        if extra_check is not None:
            extra_check()

    def read_native_state() -> None:
        if not same_state(expected, checkpoint_state()):
            raise ValueError("supplied readout checkpoint changed")
        if source_before is not None and not same_state(source_before, read_snapshot(source)):
            raise ValueError("readout source learned state changed")
        if scorer is not None and not same_state(expected, read_snapshot(scorer)):
            raise ValueError("frozen scorer learned state changed")
        # Snapshot ports may invoke callbacks. Recheck the caller-owned state
        # after the last one, before admitting more work or publishing success.
        if not same_state(expected, checkpoint_state()):
            raise ValueError("supplied readout checkpoint changed during snapshot readback")

    def check_preservation() -> None:
        if qualified_boundary is not None:
            calls = 0
            final_calls = 0
            failure: Exception | None = None

            def reads() -> None:
                nonlocal calls, failure
                calls += 1
                try:
                    if calls != 1:
                        raise ValueError("qualified native reads must run exactly once")
                    if extra_check is not None:
                        extra_check()
                    budget.before_final()
                    if extra_check is not None:
                        extra_check()
                    read_native_state()
                except Exception as error:
                    failure = error
                    raise

            def final_reads() -> None:
                nonlocal final_calls, failure
                final_calls += 1
                try:
                    if final_calls != 1:
                        raise ValueError("qualified final native reads must run exactly once")
                    read_native_state()
                except Exception as error:
                    failure = error
                    raise

            result = qualified_boundary(reads, after_clocks=final_reads)
            if failure is not None:
                raise failure
            if calls != 1 or final_calls != 1 or result is not None:
                raise ValueError("qualified native reads were omitted or returned a value")
            if extra_check is not None:
                extra_check()
            return
        boundary()
        read_native_state()
        if extra_check is not None or require_initial_source:
            # Native snapshot ports can invoke callbacks into the parent owner.
            # Revalidate its prior reports/future inputs after the final port.
            check_boundary()
            if extra_check is not None:
                extra_check()

    def check() -> None:
        # Why: aggregate the existing guard as one span without changing its
        # callback order or granting the diagnostic clock budget authority.
        with timings.measure("preservation") if timings is not None else nullcontext():
            check_preservation()

    try:
        check()
        if memory_copier is None:
            _checkpoint_metadata(
                expected,
                method,
                updates,
                public_history=public_history,
                native_training_examples=native_training_examples,
            )
        source_before = read_snapshot(source)
        if memory_copier is None:
            source_before = deepcopy(source_before)
        if not same_state(expected.policy, source_before.policy):
            raise ValueError("checkpoint and source native policies differ")
        check()
        scorer = source.fork()
        if scorer is source:
            raise ValueError("readout requires an independent scorer fork")
        # The new fork initially owns source state, so inspect the original guard
        # before binding this fork to a potentially older owned checkpoint.
        boundary()
        if not same_state(expected, checkpoint_state()):
            raise ValueError("fork changed the supplied readout checkpoint")
        if not same_state(source_before, read_snapshot(source)):
            raise ValueError("fork changed the readout source")
        scorer.restore_state(
            deepcopy(expected) if memory_copier is None else memory_copier(expected)
        )
        check()
        for packet in sources:
            check()
            episode = packet.episode
            if episode is None:
                raise ValueError("TRAIN readout source lost its complete episode")
            with timings.measure("input_preparation") if timings is not None else nullcontext():
                if history_features:
                    # Each trace owns its own first public frame; neither labels nor
                    # another cue/episode can supply its history.
                    features, history_targets = episode_arrays(
                        encode_public_history_episode(episode)
                    )
                    work["history_raw_episodes_encoded"] += 1
                    work["history_raw_feature_array_bytes"] += features.nbytes
                    work["history_raw_target_array_bytes"] += history_targets.nbytes
                else:
                    features = packet.features
            work["raw_score_forwards_attempted"] += 1
            work["raw_score_rows_attempted"] += len(packet.features)
            with timings.measure("input_preparation") if timings is not None else nullcontext():
                raw_prediction_features = features if history_features else features.copy()
            with timings.measure("prediction") if timings is not None else nullcontext():
                probabilities = scorer.predict(raw_prediction_features)
            work["raw_score_forwards_completed"] += 1
            work["raw_score_rows_completed"] += len(packet.features)
            if representation_encoding == INVARIANT_HISTORY_ENCODING_ID:
                work["internal_projection_rows_returned"] += len(features)
                work["internal_projection_array_bytes_returned"] += features.nbytes
            check()
            validate_class_probabilities(probabilities, 7)
            if probabilities.shape != (9, 7):
                raise ValueError("raw scorer must return nine rows of seven probabilities")
            metrics = probability_scores(probabilities, packet.targets, 7)
            report["scores"].append(
                dict(
                    context=packet.context,
                    cue=episode.episode_id.rsplit("-", 1)[-1],
                    ce=metrics["ce"],
                    accuracy=metrics["accuracy"],
                    exact_sequence_accuracy=float(
                        np.array_equal(np.argmax(probabilities, axis=1), packet.targets[:, 0])
                    ),
                    probabilities=probabilities.tolist(),
                )
            )
        sessions: list[Any] = []
        plan = (
            live_plan
            if live_plan is not None
            else tuple((layout, cue, role) for layout in layouts for cue in ("blue", "red"))
        )
        for slot_index, (layout, cue, episode_role) in enumerate(plan):
            if live_plan is None:
                slot = _live_slot(layout, cue, episode_role, "running")
                report["episodes"].append(slot)
            else:
                slot = report["episodes"][slot_index]
                slot.update(status="running", reason=None)
            environment = None
            current = scorer
            problem: Exception | None = None
            try:
                check()
                if method in _RECURRENT_METHODS:
                    current = scorer.start_episode()
                    work["live_sessions_started"] += 1
                    if current is scorer or any(current is old for old in sessions):
                        raise ValueError("recurrent readout requires a fresh independent session")
                    sessions.append(current)
                    check()
                    if type(current.completed_steps) is not int or current.completed_steps != 0:
                        raise ValueError("recurrent session must start at zero steps")
                    if not same_state(expected, read_snapshot(current)):
                        raise ValueError("recurrent session learned state differs")
                check()
                work["environment_resets_attempted"] += 1
                environment = MiniGridKeyDoor(
                    layout,
                    cue,
                    seed=17,
                    max_steps=16,
                    role=episode_role,
                    episode_id=(
                        f"live-{method}-{updates}-{layout.context}-row{layout.branch_row}-{cue}"
                        if role in ("knownTRAIN_A3A4_adaptation", "knownTRAIN_A3A4_current_only")
                        else f"live-{method}-{updates}-{layout.context}-{cue}"
                    ),
                )
                work["environment_resets_completed"] += 1
                history = None
                if history_features:
                    with (
                        timings.measure("input_preparation")
                        if timings is not None
                        else nullcontext()
                    ):
                        # A new encoder per environment makes cue/episode reset
                        # explicit and bounds retained memory to one first frame.
                        history = PublicInitialFrameHistory()
                        history.reset(environment.observation)
                        work["history_episodes_started"] += 1
                for _ in range(16):
                    if qualified_boundary is None or current is not scorer:
                        check()
                        if not same_state(expected, read_snapshot(current)):
                            raise ValueError("live session learned state changed")
                        if memory_copier is not None:
                            check_boundary()
                            if extra_check is not None:
                                extra_check()
                    if method in _RECURRENT_METHODS and (
                        type(current.completed_steps) is not int
                        or current.completed_steps != slot["prediction_calls_returned"]
                    ):
                        raise ValueError("recurrent activation counter changed outside prediction")
                    with (
                        timings.measure("input_preparation")
                        if timings is not None
                        else nullcontext()
                    ):
                        observation = environment.observation
                        decision: dict[str, Any] = dict(
                            observation=observation, probabilities=None, action=None
                        )
                        prediction_features = observation
                        if history is not None:
                            prediction_features = history.encode(observation)
                            decision["features"] = prediction_features
                        slot["predictions"].append(decision)
                    if representation_encoding is not None:
                        # Observation/snapshot ports may call back into the
                        # owner. Seal the parent boundary before charging the
                        # next prediction, with its raw public input recorded.
                        check()
                    slot["prediction_calls_attempted"] += 1
                    work["live_prediction_calls_attempted"] += 1
                    with (
                        timings.measure("input_preparation")
                        if timings is not None
                        else nullcontext()
                    ):
                        prediction_array = np.array([prediction_features], dtype=np.float64)
                        if history is not None:
                            work["history_live_feature_rows"] += 1
                            work["history_live_feature_array_bytes"] += prediction_array.nbytes
                    with timings.measure("prediction") if timings is not None else nullcontext():
                        probabilities = current.predict(prediction_array)
                    slot["prediction_calls_returned"] += 1
                    work["live_prediction_calls_returned"] += 1
                    if (
                        representation_encoding == INVARIANT_HISTORY_ENCODING_ID
                        or representation_feature_dim == 298
                    ):
                        work["internal_projection_rows_returned"] += 1
                        work["internal_projection_array_bytes_returned"] += (
                            representation_feature_dim * 8
                        )
                    check()
                    if (qualified_boundary is None or current is not scorer) and not same_state(
                        expected, read_snapshot(current)
                    ):
                        raise ValueError("live prediction changed learned state")
                    if method in _RECURRENT_METHODS and (
                        type(current.completed_steps) is not int
                        or current.completed_steps != slot["prediction_calls_returned"]
                    ):
                        raise ValueError("one live prediction must advance activation exactly once")
                    validate_class_probabilities(probabilities, 7)
                    if probabilities.shape != (1, 7):
                        raise ValueError("live predictor must return exactly one K7 row")
                    decision.update(
                        probabilities=probabilities[0].tolist(),
                        action=int(np.argmax(probabilities[0])),
                    )
                    if qualified_boundary is None or current is not scorer:
                        check()
                    if environment.legal_actions != tuple(range(7)):
                        raise ValueError("MiniGrid must expose all seven valid native inputs")
                    with timings.measure("physics") if timings is not None else nullcontext():
                        environment.step(decision["action"])
                    check()
                    if environment.terminal:
                        break
                observed = environment.report
                if observed["status"] != "complete":
                    raise ValueError("native environment did not signal its bounded ending")
                check()
                if not same_state(expected, read_snapshot(current)):
                    raise ValueError("live session state changed after episode")
                slot.update(
                    status="complete",
                    state_unchanged=True,
                    success=bool(
                        observed["terminated"]
                        and not observed["truncated"]
                        and observed["total_return"] > 0
                    ),
                    **{"return": observed["total_return"]},
                )
            except Exception as error:
                problem = error
                slot.update(
                    status="failed",
                    reason=f"{type(error).__name__}: {error}",
                    state_unchanged=False,
                )
                raise
            finally:
                if environment is not None:
                    cleanup_problem = None
                    try:
                        environment.close()
                    except Exception as error:
                        cleanup_problem = error
                        slot["close_error"] = f"{type(error).__name__}: {error}"
                    try:
                        observed = environment.report
                        for transition in observed["transitions"]:
                            transition.pop("raw_next_observation", None)
                        slot["environment"] = observed
                        work["environment_steps_attempted"] += observed[
                            "environment_steps_attempted"
                        ]
                        work["environment_steps_returned"] += observed["environment_steps"]
                        work["failure_work_may_be_unobserved"] |= observed[
                            "failure_work_may_be_unobserved"
                        ]
                    except Exception as error:
                        slot["readback_error"] = f"{type(error).__name__}: {error}"
                        work["failure_work_may_be_unobserved"] = True
                        if cleanup_problem is None:
                            cleanup_problem = error
                    if cleanup_problem is not None:
                        if problem is None:
                            slot.update(
                                status="failed",
                                reason=f"{type(cleanup_problem).__name__}: {cleanup_problem}",
                                state_unchanged=None,
                            )
                            raise cleanup_problem
                        # Preserve the primary execution fault; cleanup/readback
                        # failures remain separately visible in its partial slot.
        check()
        report.update(status="complete", state_unchanged=True)
    except Exception as error:
        work["failure_work_may_be_unobserved"] = True
        report.update(
            status="failed", reason=f"{type(error).__name__}: {error}", state_unchanged=False
        )
        if timings is not None:
            report["phase_timings"] = timings.report()
        raise MiniGridEvaluationFailed(report) from error
    if timings is not None:
        report["phase_timings"] = timings.report()
    return report
