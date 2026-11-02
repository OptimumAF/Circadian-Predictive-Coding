"""Disposable F/old-A/return diagnosis for the frozen recurrent readout.

Inputs are an owned two-fit A reader and its original TRAIN packets. The existing
probe owns one separate fork and two original short budgets. The existing game
kernel observes both private endpoints. No source update, final data, promotion,
policy selection, solver or new orchestration loop is provided here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from time import monotonic
from typing import Any, Callable

from src.adapters.minigrid_learning_pilot import _signature, _check_signature
from src.adapters.minigrid_learnability import _collect_probe_sources
from src.adapters.minigrid_return_fit import return_training_sources
from src.adapters.minigrid_recurrent_readout import (
    METHOD,
    RecurrentReadoutLearner,
    RecurrentReadoutPolicy,
    copy_snapshot,
    _owners,
)
from src.adapters.minigrid_policy_evaluation import _run_checkpoint, _return_budget_guard
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_key_door import predefined_layouts, predefined_distinct_probe_layouts


def describe_readout_probe():
    return dict(
        definition="frozen_recurrent_readout_F2_A2_probe_v1",
        source="original A-blue/red two fits18rows; raw148/H8/K7/seed17, frozen0SGD",
        checkpoints=[0, 1, 2],
        probe_updates=2,
        return_updates=2,
        fit_rows_per_phase=18,
        old_monitors="two original A TRAIN episodes, readonly until explicit return",
        collection="four distinct F probe_train/probe_test teachers, nine steps each",
        native_games="F blue/red probe_test then A blue/red TRAIN after each phase",
        budget_seconds=60,
        final_available=False,
        deployed_learning=False,
        qualification="disposable diagnosis; not reliability/generalization/CPC/equal-compute evidence",
    )


def _arrays(packet):
    # Why: the generic probe admits roles before calling the existing array port;
    # complete-episode TRAIN admission on the learner remains unchanged.
    return replace(
        packet, features=packet.features.copy(), targets=packet.targets.copy(), episode=None
    )


def _packet_refs(packets):
    # Hold references so equal array replacement cannot reuse a discarded id.
    return tuple(
        value
        for packet in packets
        for value in (
            packet,
            packet.features,
            packet.targets,
            packet.row_ids,
            packet.episode,
            packet.packet_episodes,
        )
    )


def run_readout_probe(source, state, sources, *, check_boundary, memory):
    if type(source) is not RecurrentReadoutLearner or not callable(check_boundary):
        raise ValueError("readout probe requires its exact A reader and original guard")
    check_boundary()
    owned = copy_snapshot(state)
    if (
        owned.policy != RecurrentReadoutPolicy()
        or owned.failure is not None
        or owned.fit_updates != (1, 1)
        or owned.fit_rows != (9, 9)
        or any(model is None or model.fit_rows != 9 for model in owned.readouts)
    ):
        raise ValueError("readout probe requires the complete original two-fit A endpoint")
    templates = return_training_sources(sources)
    source_signature, state_owners = _signature(sources), _owners(state)
    source_refs = _packet_refs(sources)
    live_owner = source._state
    live_pins = _owners(live_owner)
    report: dict[str, Any] = dict(
        protocol=describe_readout_probe(),
        status="incomplete",
        reason=None,
        collection=[],
        probe=None,
        native_readouts=[],
        budgets=None,
        source_unchanged=None,
    )
    collection, endpoints = report["collection"], report["native_readouts"]
    protocol_owner = report["protocol"]
    protocol_value = deepcopy(protocol_owner)
    protected: list[Callable[[], None]] = []
    signatures: list[tuple[tuple[AttributionBatch, ...], tuple[Any, ...], tuple[Any, ...]]] = []
    budgets = []

    def check_values():
        if (
            report["protocol"] is not protocol_owner
            or not _same_value(protocol_owner, protocol_value)
            or report["collection"] is not collection
            or report["native_readouts"] is not endpoints
            or _owners(state) != state_owners
            or not _same_value(state, owned)
            or source._state is not live_owner
            or _owners(source._state) != live_pins
            or not _same_value(source._state, owned)
            or not _check_signature(sources, source_signature)
            or any(a is not b for a, b in zip(_packet_refs(sources), source_refs, strict=True))
        ):
            raise ValueError("original readout source/protocol/evidence changed or was replaced")
        for check in protected:
            check()
        for packets, signature, refs in signatures:
            if not _check_signature(packets, signature) or any(
                a is not b for a, b in zip(_packet_refs(packets), refs, strict=True)
            ):
                raise ValueError("original readout probe packets changed")

    def guard():
        # A returned detached snapshot cannot qualify mutations made by its port.
        check_values()
        check_boundary()
        check_values()
        observed = source.snapshot_state()
        check_values()
        if not _same_value(observed, owned):
            raise ValueError("returned source snapshot differs from the original readout")

    try:
        guard()
        for updates in (2, 2, 0):
            limits = ToyExecutionBudget(
                max_training_updates=updates,
                max_replay_examples=0,
                max_wall_seconds=60,
                max_process_rss_bytes=512 * 1024**2,
            )
            budget = ToyBudgetSession(limits, monotonic)
            budgets.append(budget)
            budget.attach_memory(memory)
        probe_budget, return_budget, readout_budget = budgets
        native_budget_check = _return_budget_guard(readout_budget)
        training, evaluation = _collect_probe_sources(guard, collection, distinct_probe=True)
        if (
            type(training) is not tuple
            or len(training) != 16
            or type(evaluation) is not tuple
            or len(evaluation) != 2
        ):
            raise ValueError("readout probe requires the original collector's fixed16/2 packets")
        if len(collection) != 4 or any(
            row.get("status") != "complete"
            or row.get("closed") is not True
            or row.get("environment_steps") != 9
            or row.get("role") != role
            or row.get("cue") != cue
            for row, (role, cue) in zip(
                collection,
                (
                    ("probe_train", "blue"),
                    ("probe_train", "red"),
                    ("probe_test", "blue"),
                    ("probe_test", "red"),
                ),
                strict=True,
            )
        ):
            raise ValueError("readout probe requires four complete closed fixed F sources")
        signatures.extend(
            (packets, _signature(packets), _packet_refs(packets))
            for packets in (training, evaluation)
        )
        collection_check, _ = paid_report_guard(dict(records=collection))
        protected.append(collection_check)
        probe_training = tuple(_arrays(packet) for packet in training[:2])
        probe_test = tuple(_arrays(packet) for packet in evaluation)
        monitors = tuple(_arrays(packet) for packet in sources)
        returning = tuple(_arrays(packet) for packet in templates)
        signatures.extend(
            (packets, _signature(packets), _packet_refs(packets))
            for packets in (probe_training, probe_test, monitors, returning)
        )

        def endpoint(learner, phase, boundary):
            guard()
            boundary()
            snapshot = copy_snapshot(learner.snapshot_state())
            count = 2 if phase == "after_probe" else 3
            if (
                len(endpoints) != count - 2
                or phase != ("after_probe", "after_return")[len(endpoints)]
                or snapshot.failure is not None
                or snapshot.fit_updates != (count, count)
                or snapshot.fit_rows != (9 * count, 9 * count)
                or not _same_value(snapshot.native, owned.native)
            ):
                raise ValueError("private readout endpoint differs from the fixed F2/A2 schedule")
            endpoint_pins = _owners(snapshot)
            endpoint_value = copy_snapshot(snapshot)

            def endpoint_guard():
                guard()
                native_budget_check()
                if _owners(snapshot) != endpoint_pins or not _same_value(snapshot, endpoint_value):
                    raise ValueError("private readout endpoint changed or was replaced")

            f_blue, f_red = predefined_distinct_probe_layouts()
            old_a = predefined_layouts()[0]
            row = dict(phase=phase, updates=snapshot.completed_updates, games=None)
            endpoints.append(row)
            try:
                row["games"] = _run_checkpoint(
                    learner,
                    snapshot,
                    method=METHOD,
                    updates=snapshot.completed_updates,
                    layouts=(f_blue, f_red, old_a),
                    role="private_F_then_returned_A",
                    sources=(),
                    budget=readout_budget,
                    check_boundary=guard,
                    public_history=False,
                    native_training_examples=snapshot.training_examples,
                    memory_copier=copy_snapshot,
                    extra_check=endpoint_guard,
                    state_equal=_same_value,
                    live_plan=(
                        (f_blue, "blue", "probe_test"),
                        (f_red, "red", "probe_test"),
                        (old_a, "blue", "train"),
                        (old_a, "red", "train"),
                    ),
                    qualified_boundary=boundary,
                )
            except Exception as error:
                if hasattr(error, "report"):
                    row["games"] = deepcopy(error.report)
                raise
            # Pin the returned observation before any source/clock/parent port.
            endpoint_check, original_row = paid_report_guard(row)
            protected.append(endpoint_check)
            try:
                endpoint_guard()
                endpoint_check()
                if (
                    row["games"]["status"] != "complete"
                    or row["games"]["state_unchanged"] is not True
                    or [e.get("source_role") for e in row["games"].get("episodes", [])]
                    != ["probe_test", "probe_test", "train", "train"]
                    or any(
                        e.get("status") != "complete"
                        or e.get("environment", {}).get("closed") is not True
                        for e in row["games"]["episodes"]
                    )
                ):
                    raise ValueError("private readout games did not complete unchanged")
                endpoint_check()
            except Exception:
                row.clear()
                row.update(original_row)
                raise

        report["probe"] = run_fresh_task_probe(
            source,
            state,
            probe_training,
            probe_test,
            budget=probe_budget,
            checkpoint_updates=(0, 1, 2),
            decision_diagnostics=True,
            check_boundary=guard,
            training_readouts=monitors,
            return_training=returning,
            return_budget=return_budget,
            short_readout_return=True,
            endpoint_observer=endpoint,
        )
        guard()
        result = report["probe"]
        if result["status"] != "complete" or result["return_phase"]["status"] != "complete":
            raise ValueError("readout F/return probe incomplete; retain paid partial evidence")
        if len(endpoints) != 2:
            raise ValueError("readout probe lacks both fixed native endpoints")
        report.update(status="complete", source_unchanged=True)
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            report["budgets"] = [
                dict(
                    limits=asdict(b.budget),
                    birth=b.started_at,
                    last_clock=b.last_clock,
                    updates=b.updates_completed,
                    replay=b.replay_examples_completed,
                )
                for b in budgets
            ]
        except Exception as error:
            report["budget_capture_error"] = f"{type(error).__name__}: {error}"
            report.update(status="incomplete", reason="paid readout budget capture unavailable")
    return report
