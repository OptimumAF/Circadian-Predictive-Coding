"""Fixed reserved E probes around the existing acquired reward/quota runner.

Inputs are owned initial/end native states and reserved role-separated packets.
Outputs retain independent probe charges and original benchmark work. No final
data, policy promotion, checkpoint loading or new training loop belongs here.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any

from src.adapters import key_door_acquired_quota as acquired
from src.adapters import key_door_reward_pilot as recipe
from src.adapters import key_door_pilot as shared
from src.adapters.key_door_learnability import (
    probe_slots,
    run_checkpoint_probe,
    valid_probe,
    valid_probes,
)
from src.adapters.recurrent_probe_source import recurrent_probe_source
from src.app.fresh_task_probe import run_fresh_task_probe, _validate_batches

WALL_SECONDS = 180.0


def describe_protocol() -> dict[str, Any]:
    return dict(
        protocol_id="key_door_acquired_quota_reserved_e_probe_v1",
        classification="single_recurrent_learning_capacity_diagnostic_not_paired_adaptation",
        prefix=deepcopy(acquired.describe_protocol()["prefix"]),
        benchmark=dict(
            initializer_seed=17,
            sampling_seed=29,
            training_environment=0,
            collection_steps=96,
            maximum_native_updates=208,
            maximum_training_rows=1632,
            original_clock_and_budget=True,
        ),
        probes=dict(
            checkpoints="genuine owned native0 and actual end-of96-steps native count",
            task="existing fixed reserved E layout; deterministic role-separated packets",
            updates_each=16,
            points=[0, 4, 8, 16],
            total_updates=32,
            total_training_rows=256,
            score_calls=16,
            score_rows=128,
            data_episodes=18,
            data_physical_steps=144,
            diagnostics="decision-class summaries from the same returned probe_test predictions",
        ),
        maximum_wall_seconds=WALL_SECONDS,
        final_evaluation="unavailable: no final teacher packets or live sessions constructed",
        isolation="disposable full-state array learners; no benchmark/deployed updates, promotion or selection",
        metric_direction="probe CE lower/accuracy higher; report both raw curves and all failures",
        limitations="initial0 to end contrast includes both prefix and reward history, not causal quota attribution; one fixed instance/initializer; deterministic E resubstitution is not generalization; no matched trained BP/PC/CPC, retention/recovery or advantage claim",
    )


def run_pilot() -> dict[str, Any]:
    started = monotonic()
    deadline = started + WALL_SECONDS
    row = acquired._row(False)
    row.update(probes=probe_slots(checkpoint_updates=(0, 208)), episodes=[])
    errors: list[str] = []
    completed: list[Any] = []
    sampler = memory = None
    data = None
    reserved_started = False
    data_seal = None
    plans = deepcopy(row["probes"])
    probe_owners = (id(row["probes"]), tuple(id(slot) for slot in row["probes"]))
    active = -1

    def check_data() -> None:
        if monotonic() > deadline:
            raise ValueError("quota diagnostic original invocation deadline exhausted")
        if data is None or not recipe._same_state(
            data_seal, [[acquired._signature(batch) for batch in group] for group in data]
        ):
            raise ValueError("original reserved probe packets changed")
        if (id(row["probes"]), tuple(id(slot) for slot in row["probes"])) != probe_owners:
            raise ValueError("original quota probe records replaced")
        for index, expected in enumerate(completed):
            if not recipe._same_state(expected, row["probes"][index]):
                raise ValueError("preceding completed quota probe evidence changed")
        for index in range(len(completed), 2):
            if index != active and not recipe._same_state(plans[index], row["probes"][index]):
                raise ValueError("unreached quota probe plan changed")

    def probe(source: Any, state: Any, budget: Any, position: str, boundary: Any) -> Any:
        nonlocal active
        check_data()
        index = len(completed)
        if position != ("initial" if index == 0 else "end") or index >= 2:
            raise ValueError("quota probes require original initial/end order")
        if state.completed_updates != (0 if index == 0 else len(row["training"])):
            raise ValueError("quota probe is not at its original owned native checkpoint")
        active = index
        if index:
            row["probes"][1]["checkpoint_updates"] = state.completed_updates

        def guard() -> None:
            boundary()
            check_data()
            if not recipe._same_state(
                row["probes"][index]["checkpoint_updates"], state.completed_updates
            ):
                raise ValueError("active quota probe native identity changed")

        guard()
        bridge_started = monotonic()
        owner = recurrent_probe_source(state, check_boundary=guard)
        row["probes"][index]["bridge_seconds"] = monotonic() - bridge_started
        assert data is not None
        run_checkpoint_probe(
            row,
            owner,
            state,
            data,
            checkpoint=state.completed_updates,
            wake=budget,
            replay=None,
            deadline=deadline,
            sampler=sampler,
            same_state=recipe._same_state,
            probe_runner=run_fresh_task_probe,
            decision_diagnostics=True,
            check_boundary=guard,
        )
        if not valid_probe(row["probes"][index]):
            raise ValueError("quota probe incomplete")
        guard()
        completed.append(deepcopy(row["probes"][index]))
        active = -1
        return check_data

    try:
        sampler = recipe.ProcessRssSampler()
        with sampler:
            torch = recipe.require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            reserved_started = True
            data = shared.build_probe_stream()
            _validate_batches(*data)
            data_seal = [[acquired._signature(batch) for batch in group] for group in data]
            check_data()
            recipe._run_arm(row, None, deadline, sampler, checkpoint_probe=probe, probe_only=True)
            check_data()
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        try:
            if sampler is not None:
                memory = asdict(sampler.snapshot())
        except Exception as error:
            errors.append(f"readback {type(error).__name__}: {error}")
    complete = (
        not errors
        and row["status"] == "complete"
        and len(completed) == 2
        and valid_probes(row)
        and row["source_unchanged"] is True
        and row["actor_unchanged"] is True
        and not row["episodes"]
        and row["stream_result"] is None
        and memory is not None
        and memory["peak_bytes"] <= shared.MAX_RSS
        and monotonic() <= deadline
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        runs=[row],
        global_errors=errors,
        total_seconds=monotonic() - started,
        observed_process_rss=memory,
        final_data_constructed=False,
        probe_updates_charged=sum(slot["charged_updates"] for slot in row["probes"]),
        reserved_data_episodes=(None if reserved_started else 0)
        if data is None
        else sum(len(group) for group in data),
        reserved_data_work_may_be_unobserved=reserved_started and data is None,
        scientific_advantage_verified=False,
        full_R43_complete=False,
    )
