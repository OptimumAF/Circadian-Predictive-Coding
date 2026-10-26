"""Saved A32 representation comparison on the existing delayed TRAIN runner.

Inputs are validated original raw0/A32 states and two original TRAIN episodes.
Outputs are one fresh invariant count stream and four frozen target-free readouts.
There is no teacher collection, final evaluation, tuning, replay or file IO here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from time import monotonic
from typing import Any

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.invariant_action_memory_learner import InvariantActionMemoryLearner
from src.adapters.minigrid_action_memory_pilot import _history_packets
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    _return_budget_guard,
    evaluate_representation_checkpoint,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemorySnapshot, copy_action_memory_snapshot
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.shared.process_memory import ProcessRssSampler

METHOD = "public_invariant_observation_memory"
BODY_SECONDS = 120.0


def describe_protocol() -> dict[str, Any]:
    """Declare the fixed exploratory slice without constructing resources."""
    return dict(
        protocol_id="minigrid_saved_A32_public_invariant_control_v1",
        classification="exploratory_known_A_and_previously_inspected_E_nuisance_transfer",
        method=METHOD,
        encoding_id=INVARIANT_HISTORY_ENCODING_ID,
        raw_baseline="original genuine raw0/A32; no baseline training in this invocation",
        stream=dict(training_only=True, heldout=[], label_delay=1, updates=32, rows=288),
        checkpoint_updates=[0, 32],
        readout_order=[["raw", 0], ["raw", 32], ["invariant", 0], ["invariant", 32]],
        slots=[
            ["A", "blue", "train"],
            ["A", "red", "train"],
            ["E-blue", "blue", "probe_test"],
            ["E-red", "red", "probe_test"],
        ],
        seed=17,
        max_steps=16,
        action_rule="raw seven-action first argmax; no mask/repair/retry",
        metrics="A/E success counts/2, return, steps; end-minus-initial and invariant-minus-raw end success",
        limits=dict(
            body_seconds=120,
            parent_work_seconds=140,
            shared_cleanup_seconds=150,
            process_rss_bytes=shared.RSS_LIMIT,
            stage_output_bytes=32 * 1024**2,
            cpu_threads=1,
            gpu=0,
        ),
        planned_work=dict(
            native_train_calls=32,
            training_rows=288,
            raw_score_forwards=0,
            environment_resets=16,
            maximum_predictions=256,
            maximum_environment_steps=256,
        ),
        limitations="same row dose/table cap is not equal compute, total memory or training time; historical baseline costs remain in original run; no independent generalization, plasticity, RL or CPC advantage claim",
    )


def _admit(
    raw_states: tuple[ActionMemorySnapshot, ActionMemorySnapshot],
    sources: tuple[DemonstrationEpisode, DemonstrationEpisode],
) -> tuple[tuple[ActionMemorySnapshot, ...], tuple[Any, ...]]:
    if type(raw_states) is not tuple or len(raw_states) != 2:
        raise ValueError("pilot requires original raw0/A32 snapshots")
    owned = tuple(copy_action_memory_snapshot(state) for state in raw_states)
    for point, state in zip((0, 32), owned, strict=True):
        shared._memory_counter(state, point)
    if type(sources) is not tuple or len(sources) != 2:
        raise ValueError("pilot requires original A-blue/A-red TRAIN episodes")
    episodes = tuple(copy_episode(episode) for episode in sources)
    for cue, episode in zip(("blue", "red"), episodes, strict=True):
        if (
            episode.episode_id != f"minigrid-source-A-{cue}"
            or episode.context != "A"
            or episode.role != "train"
            or episode.num_classes != 7
            or len(episode.transitions) != 9
            or len(episode.transitions[0].observation) != 148
        ):
            raise ValueError("pilot requires the fixed nine-row original TRAIN A pair")
    packets = _history_packets(
        tuple(
            shared._batch(replace(episodes[index % 2], episode_id=f"minigrid-training-{index}"))
            for index in range(32)
        )
    )
    # Why: bind the reused endpoint to this exact dose without retraining it or
    # inferring counts from clocks alone. This is a finite admission calculation.
    expected: dict[tuple[float, ...], list[int]] = {}
    for packet in packets:
        for features, target in zip(packet.features, packet.targets, strict=True):
            counts = expected.setdefault(tuple(features.tolist()), [0] * 7)
            counts[int(target[0])] += 1
    actual = {entry.features: list(entry.counts) for entry in owned[1].entries}
    if actual != expected:
        raise ValueError("saved raw A32 counts differ from the original alternating A packets")
    return owned, packets


def _contrasts(readouts: list[dict[str, Any]]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for index, name in ((0, "raw"), (2, "invariant")):
        changes[name] = {}
        for context, labels in (("A", ("A",)), ("E", ("E-blue", "E-red"))):
            counts = [
                sum(slot["success"] for slot in item["episodes"] if slot["context"] in labels)
                for item in readouts[index : index + 2]
            ]
            changes[name][context] = dict(
                initial_successes=counts[0],
                end_successes=counts[1],
                end_minus_initial=counts[1] - counts[0],
                episodes_per_endpoint=2,
            )
    changes["invariant_minus_raw_end"] = {
        context: changes["invariant"][context]["end_successes"]
        - changes["raw"][context]["end_successes"]
        for context in ("A", "E")
    }
    return changes


def run_pilot(
    raw_states: tuple[ActionMemorySnapshot, ActionMemorySnapshot],
    sources: tuple[DemonstrationEpisode, DemonstrationEpisode],
) -> dict[str, Any]:
    """Train the explicit control before any readout, preserving failed paid work."""
    started = monotonic()
    states, packets = _admit(raw_states, sources)
    input_bytes = pickle.dumps((raw_states, sources), protocol=5)
    owned_state_bytes = pickle.dumps(states, protocol=5)
    bindings = (*raw_states, *sources)
    packet_signature = shared._signature(packets)
    row = shared._row(METHOD)
    row["readouts"] = [dict(updates=point, status="not_run") for point in (0, 32)]
    slots = [
        dict(representation=name, updates=point, status="not_run", episodes=[], work=None)
        for name, point in describe_protocol()["readout_order"]
    ]
    report: dict[str, Any] = dict(
        protocol=describe_protocol(),
        status="running",
        reason=None,
        training=row,
        readouts=slots,
        contrasts={},
        scientific_advantage_verified=False,
    )
    frozen_protocol = pickle.dumps(report["protocol"], protocol=5)
    slot_objects = list(slots)
    frozen_slots = [pickle.dumps(slot, protocol=5) for slot in slots]
    private_slots = deepcopy(slots)
    arm: dict[str, Any] | None = None
    training_capture: bytes | None = None
    readout_budget: ToyBudgetSession | None = None
    budget_guard = None
    sampler = None
    raw_source = None
    final_rss = None
    active_index: int | None = None

    def sealed() -> None:
        if (
            tuple(id(value) for value in (*raw_states, *sources))
            != tuple(id(value) for value in bindings)
            or pickle.dumps((raw_states, sources), protocol=5) != input_bytes
            or not shared._check_signature(packets, packet_signature)
        ):
            raise ValueError("original saved inputs or TRAIN packets changed")
        if (
            report["training"] is not row
            or report["readouts"] is not slots
            or pickle.dumps(report["protocol"], protocol=5) != frozen_protocol
            or len(slots) != 4
        ):
            raise ValueError("fixed pilot publication slots changed")
        for index, slot in enumerate(slots):
            if (
                slot is not slot_objects[index]
                or pickle.dumps(slot, protocol=5) != frozen_slots[index]
            ):
                raise ValueError("preceding or planned readout evidence changed")
        if training_capture is not None and pickle.dumps(row, protocol=5) != training_capture:
            raise ValueError("preceding completed training report changed")
        if pickle.dumps(states, protocol=5) != owned_state_bytes:
            raise ValueError("owned saved readout states changed")

    def check() -> None:
        nonlocal final_rss
        if monotonic() - started > BODY_SECONDS:
            raise ValueError("fixed whole-case wall-time budget exhausted")
        sealed()
        if sampler is not None:
            sampler.sample()
            observed_rss = sampler.snapshot()
            final_rss = asdict(observed_rss)
            if observed_rss.peak_bytes > shared.RSS_LIMIT:
                raise ValueError("fixed whole-case process RSS budget exhausted")
        if arm is not None:
            arm["guard"]()
        if raw_source is not None and raw_source.snapshot_state() != states[0]:
            raise ValueError("original raw source changed")
        if budget_guard is not None:
            budget_guard()
            assert readout_budget is not None
            readout_budget.before_final()
            budget_guard()
        sealed()

    def capture(index: int, value: dict[str, Any]) -> None:
        slots[index] = value
        slot_objects[index] = value
        frozen_slots[index] = pickle.dumps(value, protocol=5)
        private_slots[index] = deepcopy(value)

    try:
        with ProcessRssSampler() as active_sampler:
            sampler = active_sampler
            check()
            arm = shared._train_arm(
                row,
                packets,
                sampler,
                started + BODY_SECONDS,
                check,
                learner_factory=lambda _: InvariantActionMemoryLearner(),
                checkpoint_updates=(0, 32),
            )
            training_capture = pickle.dumps(row, protocol=5)
            check()
            readout_budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_wall_seconds=max(0.0, started + BODY_SECONDS - monotonic()),
                    max_process_rss_bytes=shared.RSS_LIMIT,
                ),
                monotonic,
            )
            readout_budget.attach_memory(sampler)
            budget_guard = _return_budget_guard(readout_budget)
            raw_source = ActionMemoryLearner(states[0].policy)
            inputs = (
                (raw_source, states[0]),
                (raw_source, states[1]),
                (arm["source"], arm["states"][0][1]),
                (arm["source"], arm["states"][1][1]),
            )
            for index, (source, state) in enumerate(inputs):
                check()
                active_index = index
                try:
                    value = evaluate_representation_checkpoint(
                        source,
                        state,
                        updates=(0, 32)[index % 2],
                        budget=readout_budget,
                        check_boundary=check,
                    )
                except MiniGridEvaluationFailed as error:
                    capture(
                        index, dict(error.report, representation=("raw", "invariant")[index // 2])
                    )
                    raise
                capture(index, dict(value, representation=("raw", "invariant")[index // 2]))
                check()
                active_index = None
            check()
        check()  # Closing sampler callbacks also precede complete publication.
        final_rss = None if sampler is None else asdict(sampler.snapshot())
        check()
        report.update(status="complete", contrasts=_contrasts(private_slots))
    except Exception as error:
        report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        if active_index is not None and private_slots[active_index]["status"] == "not_run":
            private_slots[active_index].update(
                status="failed",
                reason=report["reason"],
                delegated_report_unavailable=True,
                work=dict(
                    failure_work_may_be_unobserved=True,
                    live_prediction_calls_attempted=None,
                    environment_resets_attempted=None,
                    environment_steps_attempted=None,
                ),
            )
        report["changed_readout_indices"] = [
            index
            for index, slot in enumerate(slots)
            if index >= len(slot_objects)
            or slot is not slot_objects[index]
            or pickle.dumps(slot, protocol=5) != frozen_slots[index]
        ]
        if training_capture is not None:
            report["training"] = pickle.loads(training_capture)
    finally:
        # Decode only private captures, never bytes supplied through a callback.
        report["readouts"] = private_slots
        report["readout_training_updates"] = (
            None if readout_budget is None else readout_budget.updates_completed
        )
        report["observed_process_rss"] = final_rss
        report["total_seconds"] = monotonic() - started
        report["training_projection_accounting"] = dict(
            returned_training_rows=report["training"].get("native_training_examples"),
            projected_array_bytes_returned=None
            if report["training"].get("native_training_examples") is None
            else 2368 * report["training"]["native_training_examples"],
            scope="returned training feature arrays only; validation/copies/Python objects/failed attempts and historical raw training cost not included",
        )
    return report
