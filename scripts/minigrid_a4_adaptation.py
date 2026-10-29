"""Compose original delayed affine learning with fresh A4 TRAIN and fixed games.

Inputs are the exact historical170 TRAIN report or freshly collected typed A4,
plus bundled A3 TRAIN. Outputs retain original costs/receipts and fixed-phase
or explicit chooser-trial/application readouts. No learner math, source relabeling, private promotion or
alternative stream lives in this module.
"""

from dataclasses import asdict, replace
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_context_linear_stream_checkpoint,
    _require_phase,
)
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.reference_controller import ReferencePolicy
from src.core.minigrid_public_history import encode_public_history_episode
from src.infra.minigrid_affine_example import example_affine_training
from src.infra.minigrid_memory import _require_minigrid
from src.shared.process_memory import ProcessRssSampler

SOURCE_BYTES = 428191
SOURCE_SHA = "bce498c34c23d9ef128f9344bf7164a70edc6afa156d42e7534bb708b2e3c26f"


def describe_adaptation(*, current_only=False, current_repeated=False):
    from scripts.run_minigrid_linear_fit_control import (
        describe_context_linear_a4,
        describe_context_linear_a4_current_only,
        describe_context_linear_a4_current_repeated,
    )

    if (
        type(current_only) is not bool
        or type(current_repeated) is not bool
        or (current_only and current_repeated)
    ):
        raise ValueError("current_only/current_repeated require distinct exact bools")

    return dict(
        definition="minigrid_a3_a4_a3_current_repeated_and_games_v1"
        if current_repeated
        else "minigrid_a3_a4_a3_current_only_and_games_v1"
        if current_only
        else "minigrid_a3_a4_a3_adaptation_and_games_v1",
        training=describe_context_linear_a4_current_repeated()
        if current_repeated
        else describe_context_linear_a4_current_only()
        if current_only
        else describe_context_linear_a4(),
        source_bytes=SOURCE_BYTES,
        source_sha256=SOURCE_SHA,
        native_plan="at2/4/6: A3blue/red thenA4blue/red TRAIN; seed17/max16/K7firstargmax",
        game_budget="one cumulative original150s512MiB zero-update/replay owner across all phases",
        interpretation="known TRAIN acquisition/retention/return; recovery requires loss/regain; no CPC/generalization/matchedcompute claim",
    )


def _admit_a4_record(record):
    if (
        record.get("status") != "complete"
        or record.get("new_fits") != 0
        or record.get("new_replay") != 0
        or record.get("history_rows") != 18
        or len(record.get("episodes", ())) != 2
        or record["budget"]["updates"] != 0
        or record["budget"]["replay"] != 0
        or record["budget"]["started_at"] != record["budget"]["observed_started_at"]
        or record["counts"]["forbidden"] != 0
    ):
        raise ValueError("complete from-birth170 A4 TRAIN collection required")
    packets = _admit_a4_native_records(record.get("collection", ()))
    if json.loads(json.dumps([asdict(p) for p in packets])) != record["episodes"]:
        raise ValueError("exported A4 TRAIN history differs from original native evidence")
    provenance = dict(
        source_bytes=SOURCE_BYTES,
        source_sha256=SOURCE_SHA,
        counts=record["counts"],
        budget=record["budget"],
        seconds=record["seconds"],
        sampled_peak_rss_bytes=record["sampled_peak_rss_bytes"],
        qualification="original18teachersteps/2env; no learnerfit; sunk collection cost, not new authority",
    )
    return packets, provenance


def _admit_a4_native_records(records):
    """Decode the same exact native TRAIN evidence for file and live producers."""
    from scripts.run_minigrid_linear_fit_control import _transition

    if len(records) != 2:
        raise ValueError("two complete A4 TRAIN native records required")
    packets = []
    for native, cue in zip(records, ("blue", "red"), strict=True):
        identity = f"minigrid-source-A4-train-{cue}"
        if (
            (native["episode_id"], native["context"], native["role"], native["cue"])
            != (identity, "A", "train", cue)
            or native["layout"]
            != dict(context="A", rotation=0, branch_row=4, swap_keys=False, wall_color="grey")
            or native["status"] != "complete"
            or native["closed"] is not True
            or native["terminated"] is not True
            or native["truncated"] is not False
            or native["failure_work_may_be_unobserved"] is not False
            or native["environment_steps"] != 9
            or native["environment_steps_attempted"] != 9
            or len(native["transitions"]) != 9
            or native["total_return"] <= 0
        ):
            raise ValueError("exact complete closed fresh A4 TRAIN native record required")
        transitions = []
        for step, row in enumerate(native["transitions"]):
            if row["applied"] is not True:
                raise ValueError("A4 TRAIN transition must have returned")
            # Why: native reports own tuples; JSON exports own lists. Normalize
            # only that representation at the handoff, keeping codec validation.
            observations = {
                name: list(row[name]) if type(row[name]) is tuple else row[name]
                for name in ("observation", "next_observation")
            }
            transitions.append(
                _transition(
                    dict(
                        step=step,
                        action=row["action"],
                        reward=row["reward"],
                        terminal=row["terminated"],
                        num_classes=7,
                        **observations,
                    )
                )
            )
        packet = encode_public_history_episode(
            DemonstrationEpisode(identity, "A", "train", tuple(transitions), 7)
        )
        if sum(t.reward for t in packet.transitions) != native["total_return"]:
            raise ValueError("exported A4 TRAIN history differs from original native evidence")
        packets.append(packet)
    return tuple(packets)


def read_a4_training(path):
    with Path(path).open("rb") as source:
        raw = source.read(SOURCE_BYTES + 1)
    if len(raw) != SOURCE_BYTES or sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("exact qualified170 from-birth A4 TRAIN report required")
    return _admit_a4_record(json.loads(raw))


def summarize_games(phases):
    """Fixed paired outcomes; an incomplete phase is never a zero-success task."""
    if [phase.get("tick", phase["updates"]) for phase in phases] != [2, 4, 6]:
        return dict(
            complete=False,
            acquisition_gain=None,
            old_retention_delta=None,
            return_gain=None,
            learning_recovery=None,
        )
    outcomes = []
    for phase in phases:
        rows = phase["games"]["episodes"]
        if len(rows) != 4 or any(
            r["status"] != "complete" or r["environment"]["closed"] is not True for r in rows
        ):
            raise ValueError("paired A4 metrics require four complete closed episodes per phase")
        outcomes.append([r["success"] for r in rows])
    rates = [dict(old_A3=sum(row[:2]) / 2, new_A4=sum(row[2:]) / 2) for row in outcomes]
    return dict(
        complete=True,
        success_by_stage=rates,
        acquisition_gain=rates[1]["new_A4"] - rates[0]["new_A4"],
        old_retention_delta=rates[1]["old_A3"] - rates[0]["old_A3"],
        return_gain=rates[2]["old_A3"] - rates[1]["old_A3"],
        learning_recovery=[
            None
            if "tick" in phases[0] and not (outcomes[0][i] and not outcomes[1][i])
            else outcomes[0][i] and not outcomes[1][i] and outcomes[2][i]
            for i in range(2)
        ],
        qualification="two fixed cues/known TRAIN only; retention is not recovery; no matchedcompute/CPC/generalization",
    )


def _collect_fresh_a4(check_boundary, records):
    from scripts.minigrid_affine_workflow import _collect_recurring_train_sources
    from src.core.minigrid_key_door import MiniGridKeyDoorLayout

    started = monotonic()
    packets = _collect_recurring_train_sources(
        check_boundary, records, layout=MiniGridKeyDoorLayout("A", branch_row=4)
    )
    check_boundary()
    admitted = _admit_a4_native_records(records)
    if packets != admitted or any(r.get("seed") != 17 or r.get("max_steps") != 16 for r in records):
        raise ValueError("fresh A4 producer packets or original native settings differ")
    return packets, dict(
        origin="current_run_A4_TRAIN",
        collection_seconds=monotonic() - started,
        teacher_steps=sum(r["environment_steps"] for r in records),
        environment_episodes=len(records),
        history_rows=sum(len(p.transitions) for p in packets),
        qualification="fresh TRAIN teacher collection charged within original cumulative workflow clock; no historical file binding",
    )


def describe_fresh_adaptation(
    *,
    reference_policy=None,
    controller_mode=None,
    retained_probe=False,
    return_old=False,
    native_games=False,
    retained_rehearsal=False,
):
    if type(retained_rehearsal) is not bool or (
        retained_rehearsal
        and (
            retained_probe is not False
            or return_old is not False
            or controller_mode is not None
            or type(reference_policy) is not ReferencePolicy
            or reference_policy != ReferencePolicy("drift")
        )
    ):
        raise ValueError(
            "retained_rehearsal requires original default drift and no other private recipe"
        )
    if type(native_games) is not bool or (
        native_games and return_old is not True and retained_rehearsal is not True
    ):
        raise ValueError(
            "native_games requires an exact bool and a declared return_old or retained_rehearsal recipe"
        )
    if type(return_old) is not bool or (return_old and retained_probe is not True):
        raise ValueError("return_old requires an exact bool and retained_probe=True")
    if type(retained_probe) is not bool or (
        retained_probe
        and (
            controller_mode is not None
            or type(reference_policy) is not ReferencePolicy
            or reference_policy != ReferencePolicy("drift")
        )
    ):
        raise ValueError(
            "retained_probe requires an exact bool and the original default drift reference"
        )
    if controller_mode is not None:
        if (
            type(controller_mode) is not str
            or controller_mode not in ("current_only", "retain_arrived")
            or reference_policy is not None
        ):
            raise ValueError(
                "fresh controller requires an exact current_only or retain_arrived mode without a reference"
            )
        from scripts.minigrid_controller_learning import describe_controller_learning

        return dict(
            definition="portable_affine_a4_controller_learning_v1",
            controller_mode=controller_mode,
            training=describe_controller_learning(
                retain_arrived=controller_mode == "retain_arrived"
            ),
            source_origin="current_run_A4_TRAIN",
            historical_source_reads=0,
            collection="fresh A4 TRAIN blue/red, seed17/max16; exact nine rows each required",
            teacher_step_limit=32,
            fit_updates_upper_bound=14,
            fit_rows_upper_bound=234 if controller_mode == "retain_arrived" else 126,
            native_plan="four same-warm-base trials then same-owner application; at most seven distinct four-game TRAIN readouts",
            game_budget="original cumulative150s512MiB zero-update/replay owner before preparation",
            interpretation="known TRAIN private candidate; include setup/discarded costs, no controller/CPC/generalization/efficiency claim",
        )
    if reference_policy is not None:
        from scripts.run_minigrid_linear_fit_control import describe_context_linear_reference

        training = describe_context_linear_reference(
            reference_policy, a4_training=True, retain_arrived=True, ready_feedback=True
        )
        result = dict(
            definition="portable_affine_a4_reference_adaptation_v1",
            training=training,
            source_origin="current_run_A4_TRAIN",
            historical_source_reads=0,
            collection="fresh A4 TRAIN blue/red, seed17/max16; exact nine rows each required",
            teacher_step_limit=32,
            native_plan="logical2/4/6 actual prefixes: four A3/A4 TRAIN games per distinct complete state",
            game_budget="original cumulative150s512MiB zero-update/replay owner before preparation",
            interpretation="known TRAIN only; pending is not complete training; no matchedcompute/CPC/generalization",
        )
        if retained_probe:
            from src.adapters.minigrid_affine_probe import describe_protocol

            result.update(
                definition="portable_affine_a4_retained_F_diagnostic_v1",
                retained_probe=describe_protocol(
                    retained_a4=True,
                    **(dict(return_old=True) if return_old else {}),
                    **(dict(native_games=True) if native_games else {}),
                ),
                probe_scope="one isolated F diagnostic after all source games; no F native games/return/promotion/final/tuning",
            )
            if return_old:
                result.update(
                    definition="portable_affine_a4_F_then_old_TRAIN_return_v1",
                    probe_scope="isolated F2 then fixed old TRAIN2 return after source games; no F/returned native games, promotion, final or tuning",
                )
                if native_games:
                    result.update(
                        definition="portable_affine_a4_F_old_return_native_games_v1",
                        probe_scope="same private F2/oldTRAIN2 learner; F/A3/A4 target-free native readouts afterF andreturn, no feedback into fitting/promotion/final",
                    )
        if retained_rehearsal:
            from src.adapters.minigrid_affine_rehearsal import (
                describe_protocol as describe_rehearsal,
            )

            result.update(
                definition="portable_affine_a4_retained_rehearsal_v1",
                retained_rehearsal=describe_rehearsal(retained_a4=True),
                rehearsal_scope="original validated F16+2 release then both private equal-row arms after source games; no extraFwarmup/nativegames/promotion/final/tuning",
            )
            if native_games:
                result.update(
                    definition="portable_affine_a4_retained_rehearsal_games_v1",
                    rehearsal_scope="both fixed private arms complete before readonly F/A3/A4 games; no feedback into fitting/promotion/final/tuning",
                )
        return result
    protocol = describe_adaptation()
    del protocol["source_bytes"], protocol["source_sha256"]
    return dict(
        protocol,
        definition="portable_affine_a4_adaptation_v1",
        collection="fresh A4 TRAIN blue/red, seed17/max16; exact nine rows each required by existing stream",
        source_origin="current_run_A4_TRAIN",
        historical_source_reads=0,
        teacher_step_limit=32,
    )


def run_fresh_adaptation(
    *,
    reference_policy=None,
    controller_mode=None,
    retained_probe=False,
    return_old=False,
    native_games=False,
    retained_rehearsal=False,
):
    return _run_adaptation(
        _collect_fresh_a4,
        describe_fresh_adaptation(
            reference_policy=reference_policy,
            controller_mode=controller_mode,
            **(dict(retained_probe=retained_probe) if retained_probe is not False else {}),
            **(dict(return_old=return_old) if return_old is not False else {}),
            **(dict(native_games=native_games) if native_games is not False else {}),
            **(
                dict(retained_rehearsal=retained_rehearsal)
                if retained_rehearsal is not False
                else {}
            ),
        ),
        reference_policy=reference_policy,
        **(dict(controller_mode=controller_mode) if controller_mode is not None else {}),
        **(dict(retained_probe=True) if retained_probe else {}),
        **(dict(return_old=True) if return_old else {}),
        **(dict(native_games=True) if native_games else {}),
        **(dict(retained_rehearsal=True) if retained_rehearsal else {}),
    )


def _controller_window_summary(row):
    if row is None:
        return None
    window, budget = row.get("window") or {}, row.get("budget") or {}
    checkpoints = row.get("checkpoints") or []
    n = window.get("completed_updates")
    return dict(
        status=row.get("status"),
        reported_updates=budget.get("updates"),
        reported_native_rows=window.get("training_examples"),
        latest_native_checkpoint=checkpoints[-1] if checkpoints else None,
        unfitted_packets=len(row.get("source_episode_ids", ())) - n if type(n) is int else None,
        window_complete=window.get("window_complete"),
        training_complete=window.get("training_complete"),
        outcomes=row.get("outcomes"),
        bank_committed=(row.get("catalogue") or {}).get("committed"),
        readout_reuse=[
            dict(tick=p["tick"], reused_from=p["reused_from"]) for p in row.get("phases", ())
        ],
    )


def _controller_summary(training, *, complete):
    if training is None:
        return dict(
            complete=False,
            training_result_available=False,
            reported_updates=None,
            reported_native_rows=None,
            warmup=None,
            discarded_trials=None,
            application=None,
        )
    trials = training.get("trials") or []
    rows = [training.get("warmup"), *trials, training.get("application")]
    native_rows = [
        r.get("window", {}).get("training_examples")
        for r in rows
        if r is not None and r.get("window") is not None
    ]
    budgets = training.get("budgets") or []
    updates = [b.get("updates") for b in budgets]
    return dict(
        complete=complete and training.get("status") == "complete",
        training_result_available=True,
        private_candidate_only=True,
        choices=(training.get("fitted") or {}).get("choices"),
        estimates=(training.get("fitted") or {}).get("estimates"),
        reported_updates=sum(updates) if updates and all(type(n) is int for n in updates) else None,
        reported_native_rows=sum(native_rows)
        if native_rows and all(type(n) is int for n in native_rows)
        else None,
        cost_scope="reported completed work only, not unknown attempted work or a new utility; raw child report retains partial/failure/diagnostic/preparation costs",
        warmup=_controller_window_summary(training.get("warmup")),
        discarded_trials=[
            dict(
                context=(t.get("request") or {}).get("context"),
                action=(t.get("request") or {}).get("action_id"),
                **_controller_window_summary(t),
            )
            for t in trials
        ],
        application=_controller_window_summary(training.get("application")),
        game_work=training.get("game_work"),
        context_work=training.get("context_work"),
    )


def _reference_phase_states(training, expected_protocol):
    """Admit actual paid prefixes, never infer a model from an update count."""
    from scripts.run_minigrid_linear_fit_control import _context_linear_training_snapshot

    window = training.get("stream") or {}
    receipts = training.get("committed_receipts")
    bank = training.get("catalogue") or {}
    if (
        training.get("status") != "complete"
        or training.get("protocol") != expected_protocol
        or training.get("source_unchanged") is not True
        or training.get("actor_unchanged") is not True
        or training.get("receipt_capture_error") is not None
        or not isinstance(receipts, list)
        or window.get("status") != "complete"
        or window.get("window_complete") is not True
        or type(window.get("completed_updates")) is not int
        or not 0 <= window["completed_updates"] <= 6
        or len(receipts) != window["completed_updates"]
        or receipts != window.get("applied_updates")
        or window.get("unreturned_work_unknown") is not False
        or window.get("score_forward_calls") != 0
        or window.get("budget_updates") != len(receipts)
        or window.get("next_tick") != 7
        or training["budget"]["updates_completed"] != len(receipts)
        or bank.get("failure_work_may_be_unobserved") is not False
    ):
        raise ValueError("complete actual A4 reference window and paid receipts required")
    count = len(receipts)
    if any(
        r.get("sample_id") != f"batch-{i}"
        or r.get("update_number") != i + 1
        or r.get("observed_at") != i
        or r.get("arrived_at") != i + 1
        or type(r.get("applied_at")) is not int
        or not i + 1 <= r["applied_at"] <= 6
        for i, r in enumerate(receipts)
    ):
        raise ValueError("original delayed A4 FIFO receipts required")
    prior_rows = max(count - 2, 0) * 9
    if (
        bank.get("committed_rows")
        != dict(arrivals=9 * count, replay=prior_rows, native=9 * count + prior_rows)
        or window.get("training_examples") != 9 * count + prior_rows
        or window.get("pending_unapplied") != 6 - count
        or window.get("training_complete") is not (count == 6)
    ):
        raise ValueError("A4 reference committed costs or pending prefix disagree")
    checkpoints = training.get("unscored_checkpoints", ())
    if any(type(row["updates"]) is not int for row in checkpoints) or [
        row["updates"] for row in checkpoints
    ] != list(range(count + 1)):
        raise ValueError("all actual A4 reference prefixes required")
    states = {}
    for row in checkpoints:
        state = _context_linear_training_snapshot(row["state"])
        n = row["updates"]
        if (
            state.completed_updates != n
            or state.training_examples != 9 * n + max(n - 2, 0) * 9
            or state.failed_work is not None
        ):
            raise ValueError("actual A4 reference checkpoint disagrees with paid prefix")
        states[n] = state
    decisions = window.get("decisions", ())
    if len(decisions) != 7 or [r["observation"]["tick"] for r in decisions] != list(range(7)):
        raise ValueError("original seven A4 reference opportunities required")
    if any(
        row.get("returned") is not True
        or type(row.get("completed_updates")) is not int
        or row["completed_updates"] != sum(r["applied_at"] <= tick for r in receipts)
        for tick, row in enumerate(decisions)
    ):
        raise ValueError("A4 logical decisions disagree with actual paid receipt times")
    phases = []
    for tick in (2, 4, 6):
        n = decisions[tick]["completed_updates"]
        if type(n) is not int or n not in states:
            raise ValueError("logical A4 observation lacks its actual completed checkpoint")
        phases.append((tick, n, states[n]))
    return phases


def run_adaptation(path, *, current_only=False, current_repeated=False):
    # Why: fixed scientific file admission keeps its original byte/SHA contract;
    # live collection hands off typed sources without claiming that old binding.
    return _run_adaptation(
        lambda check, records: read_a4_training(path),
        describe_adaptation(current_only=current_only, current_repeated=current_repeated),
        current_only=current_only,
        current_repeated=current_repeated,
    )


def _run_adaptation(
    produce_source,
    protocol,
    *,
    current_only=False,
    current_repeated=False,
    reference_policy=None,
    controller_mode=None,
    retained_probe=False,
    return_old=False,
    native_games=False,
    retained_rehearsal=False,
):
    if type(retained_rehearsal) is not bool or (
        retained_rehearsal
        and (
            protocol.get("source_origin") != "current_run_A4_TRAIN"
            or current_only
            or current_repeated
            or retained_probe is not False
            or return_old is not False
            or controller_mode is not None
            or type(reference_policy) is not ReferencePolicy
            or reference_policy != ReferencePolicy("drift")
        )
    ):
        raise ValueError(
            "retained_rehearsal requires original fresh default drift and no other private recipe"
        )
    if type(native_games) is not bool or (
        native_games and return_old is not True and retained_rehearsal is not True
    ):
        raise ValueError(
            "native_games requires an exact bool and a declared return_old or retained_rehearsal recipe"
        )
    if type(return_old) is not bool or (return_old and retained_probe is not True):
        raise ValueError("return_old requires an exact bool and retained_probe=True")
    if type(retained_probe) is not bool or (
        retained_probe
        and (
            protocol.get("source_origin") != "current_run_A4_TRAIN"
            or controller_mode is not None
            or current_only
            or current_repeated
            or type(reference_policy) is not ReferencePolicy
            or reference_policy != ReferencePolicy("drift")
        )
    ):
        raise ValueError("retained probe requires the original fresh default-drift source")
    from scripts.run_minigrid_linear_fit_control import (
        _context_linear_training_snapshot,
        read_training_sources,
        run_context_linear_stream,
    )

    protocol_check, original_protocol = paid_report_guard(protocol)
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        source=None,
        training=None,
        games=[],
    )
    proofs: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    # Two owned collection records only, never an unbounded success chain.
    collection: list[dict[str, Any]] = []
    collection_proofs: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    controller_attempts = controller_returns = 0
    controller_work = dict(attempts=0, returned=0, unreturned_work_unknown=False)
    if controller_mode is not None:
        report["controller_work"] = controller_work
    probe_attempts = probe_returns = 0
    probe_work = dict(attempts=0, returned=0, unreturned_work_unknown=False)
    probe_owner = probe_check = detached_probe = None
    if retained_probe:
        report.update(probe=None, probe_work=probe_work)
    rehearsal_attempts = rehearsal_returns = 0
    rehearsal_work = dict(attempts=0, returned=0, unreturned_work_unknown=False)
    rehearsal_owner = rehearsal_check = detached_rehearsal = None
    if retained_rehearsal:
        report.update(rehearsal=None, rehearsal_work=rehearsal_work)
    is_fresh = protocol.get("source_origin") == "current_run_A4_TRAIN"
    if is_fresh:
        report["collection"] = collection
    source_check = training_check = None
    detached_source = detached_training = None
    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        binding, expected_limits = budget_binding(budget), replace(limits)
        games = report["games"]

        def check_rehearsal_evidence():
            if report.get("rehearsal_work") is not rehearsal_work or not _same_value(
                rehearsal_work,
                dict(
                    attempts=rehearsal_attempts,
                    returned=rehearsal_returns,
                    unreturned_work_unknown=rehearsal_attempts != rehearsal_returns,
                ),
            ):
                raise ValueError("original rehearsal-call costs changed")
            if report.get("rehearsal") is not rehearsal_owner:
                raise ValueError("preceding paid rehearsal report was replaced")
            if rehearsal_check is not None:
                rehearsal_check()

        def guard():
            protocol_check()
            if retained_rehearsal:
                check_rehearsal_evidence()
            if retained_probe:
                if report.get("probe_work") is not probe_work or not _same_value(
                    probe_work,
                    dict(
                        attempts=probe_attempts,
                        returned=probe_returns,
                        unreturned_work_unknown=probe_attempts != probe_returns,
                    ),
                ):
                    raise ValueError("original isolated probe-call costs changed")
                if probe_check is not None:
                    if report.get("probe") is not probe_owner:
                        raise ValueError("preceding paid probe report was replaced")
                    probe_check()
            if controller_mode is not None and (
                report.get("controller_work") is not controller_work
                or not _same_value(
                    (
                        controller_work.get("attempts"),
                        controller_work.get("returned"),
                        controller_work.get("unreturned_work_unknown"),
                    ),
                    (
                        controller_attempts,
                        controller_returns,
                        controller_attempts > controller_returns,
                    ),
                )
            ):
                raise ValueError("original controller-call costs changed")
            if reference_policy is not None and not _same_value(
                asdict(reference_policy), original_protocol["training"]["reference_policy"]
            ):
                raise ValueError("original fresh A4 reference policy changed")
            if report["protocol"] is not protocol or report["games"] is not games:
                raise ValueError("original A4 protocol or phase list replaced")
            if (
                budget_binding(budget) != binding
                or not _same_value(asdict(limits), asdict(expected_limits))
                or not _same_value(asdict(budget.budget), asdict(expected_limits))
            ):
                raise ValueError("original cumulative A4 readout budget changed")
            if not _same_value(
                (budget.updates_completed, budget.replay_examples_completed), (0, 0)
            ):
                raise ValueError("A4 readout is zero update/replay")
            budget.before_final()
            if budget_binding(budget) != binding:
                raise ValueError("A4 clock or memory binding changed")
            if is_fresh:
                if (
                    report.get("collection") is not collection
                    or not len(collection_proofs) <= len(collection) <= 2
                ):
                    raise ValueError("original bounded A4 collection prefix replaced")
                for current, (owner, check, _) in zip(collection, collection_proofs):
                    if current is not owner:
                        raise ValueError("original paid A4 collection record replaced")
                    check()
                for current in collection[len(collection_proofs) :]:
                    if (
                        current.get("status") != "complete"
                        or current.get("closed") is not True
                        or current.get("primary_error")
                        or current.get("cleanup_error")
                    ):
                        raise ValueError("preceding A4 collection incomplete")
                    check, detached = paid_report_guard(current)
                    collection_proofs.append((current, check, detached))
            if source_check is not None:
                if report["source"] is not provenance:
                    raise ValueError("original A4 source provenance replaced")
                source_check()
            if training_check is not None:
                if report["training"] is not training:
                    raise ValueError("original A4 training evidence replaced")
                training_check()
            if len(games) != len(proofs) or len(games) > 3:
                raise ValueError("original paid A4 phase prefix changed")
            for current, (owner, check, _) in zip(games, proofs):
                if current is not owner:
                    raise ValueError("original paid A4 phase replaced")
                check()
            if retained_rehearsal:
                # The original resource clock and source ports above can call
                # back after the first proof. Recheck paid evidence and fees.
                check_rehearsal_evidence()

        try:
            guard()
            _require_minigrid()
            fresh, provenance = produce_source(guard, collection)
            report["source"] = provenance
            source_check, detached_source = paid_report_guard(provenance)
            guard()
            with example_affine_training() as bundled:
                original = read_training_sources(bundled)[:2]
            if controller_mode is not None:
                from scripts.minigrid_controller_learning import run_controller_learning

                # Why: the caller's original clock already includes collection;
                # the fitter receives that owner rather than creating a new one.
                controller_attempts = 1
                controller_work.update(attempts=1, unreturned_work_unknown=True)
                training = run_controller_learning(
                    original + fresh,
                    readout_budget=budget,
                    check_boundary=guard,
                    retain_arrived=controller_mode == "retain_arrived",
                )
                controller_returns = 1
                controller_work.update(returned=1, unreturned_work_unknown=False)
                report["training"] = training
                training_check, detached_training = paid_report_guard(training)
                guard()
                if (
                    training.get("status") != "complete"
                    or training.get("capture_error") is not None
                    or training.get("protocol") != protocol["training"]
                    or training.get("source_unchanged") is not True
                ):
                    raise ValueError("complete source-bound controller learning evidence required")
                report["status"] = "complete"
                return report
            training_options: dict[str, Any] = (
                dict(
                    a4_training=True,
                    retain_arrived=True,
                    reference_policy=reference_policy,
                    ready_feedback=True,
                )
                if reference_policy is not None
                else dict(a4_current_repeated=True)
                if current_repeated
                else dict(a4_current_only=True)
                if current_only
                else dict(retain_arrived=True, a4_training=True)
            )
            training = run_context_linear_stream(
                original + fresh,
                **training_options,
                check_boundary=guard,
            )
            report["training"] = training
            training_check, detached_training = paid_report_guard(training)
            guard()
            if reference_policy is not None:
                states = _reference_phase_states(training, protocol["training"])
            else:
                costs_valid = (
                    "catalogue" not in training
                    and (training.get("stream") or {}).get("training_examples") == 54
                    if current_only
                    else (training.get("catalogue") or {}).get("committed_rows")
                    == dict(arrivals=54, replay=36, native=90)
                )
                if current_repeated:
                    repeated = training.get("current_repetition") or {}
                    costs_valid = (
                        "catalogue" not in training
                        and training.get("definition") == protocol["training"]["definition"]
                        and training.get("protocol") == protocol["training"]
                        and (training.get("stream") or {}).get("training_examples") == 90
                        and all(
                            type(repeated.get(key)) is int and repeated[key] == expected
                            for key, expected in (
                                ("arrival_row_exposures", 54),
                                ("repeated_current_row_exposures", 36),
                                ("fit_row_exposures", 90),
                                ("prior_packet_replay_rows", 0),
                                ("retained_bank_rows", 0),
                            )
                        )
                        and repeated.get("committed_rows")
                        == dict(arrivals=54, repeated_current=36, native=90)
                        and repeated.get("verified_rows") == repeated.get("committed_rows")
                        and repeated.get("verified_receipts") == training.get("committed_receipts")
                        and repeated.get("failure_work_may_be_unobserved") is False
                        and repeated.get("failed_attempt_repeated_rows_upper_bound") is None
                    )
                if (
                    training["status"] != "complete"
                    or training["source_unchanged"] is not True
                    or training["actor_unchanged"] is not True
                    or training["receipt_capture_error"] is not None
                    or len(training["committed_receipts"]) != 6
                    or training["budget"]["updates_completed"] != 6
                    or not costs_valid
                ):
                    raise ValueError("complete actual six-update A4 recipe/cost evidence required")
                states = [
                    (
                        row["updates"],
                        row["updates"],
                        _context_linear_training_snapshot(row["state"]),
                    )
                    for row in training["endpoints"]
                    if row["updates"] != 0
                ]
                if [n for _, n, _ in states] != [2, 4, 6]:
                    raise ValueError("all original typed A4 stage parameters required")
            owner = ContextLinearLearner()
            readouts: dict[int, tuple[Any, dict[str, Any], int]] = {}
            for tick, n, state in states:
                guard()
                reused = None
                try:
                    if reference_policy is not None and n in readouts:
                        expected, observed, reused = readouts[n]
                        if state != expected:
                            raise ValueError("complete A4 reference state changed before reuse")
                    else:
                        observed = evaluate_context_linear_stream_checkpoint(
                            owner,
                            state,
                            budget=budget,
                            check_boundary=guard,
                            **(
                                dict(reference_window=True, reference_catalogue=True)
                                if reference_policy is not None
                                else dict(a4_current_repeated=True)
                                if current_repeated
                                else dict(a4_current_only=True)
                                if current_only
                                else dict(a4_training=True)
                            ),
                        )
                except MiniGridEvaluationFailed as error:
                    report["failed_phase"] = dict(updates=n, games=error.report)
                    raise
                guard()
                try:
                    _require_phase(observed)
                except Exception:
                    report["failed_phase"] = dict(updates=n, games=observed)
                    raise
                phase = dict(updates=n, games=observed)
                if reference_policy is not None:
                    # Why: a completed window may wait at the same full state;
                    # logical observations are not extra executed games.
                    phase.update(tick=tick, readout_reused_from_tick=reused)
                    if n not in readouts:
                        readouts[n] = (deepcopy(state), observed, tick)
                check, detached = paid_report_guard(phase)
                games.append(phase)
                proofs.append((phase, check, detached))
            guard()
            if retained_probe:
                from src.adapters.minigrid_affine_probe import run_affine_fresh_probe

                # Why: this workflow already owns the admitted inputs and final
                # source. Reuse them directly; no loader, interception or refit.
                probe_attempts = 1
                probe_work.update(attempts=1, unreturned_work_unknown=True)
                report["probe"] = run_affine_fresh_probe(
                    original + fresh,
                    states[-1][2],
                    retained_a4=True,
                    check_boundary=guard,
                    **(dict(return_old=True) if return_old else {}),
                    **(dict(native_games=True, readout_budget=budget) if native_games else {}),
                )
                probe_returns = 1
                probe_work.update(returned=1, unreturned_work_unknown=False)
                probe_owner = report["probe"]
                probe_check, detached_probe = paid_report_guard(probe_owner)
                guard()
                if (
                    probe_owner.get("status") != "complete"
                    or probe_owner.get("source_native_state_unchanged") is not True
                    or type(probe_owner.get("source_fork_calls")) is not int
                    or probe_owner["source_fork_calls"] != 1
                    or not _same_value(
                        (
                            (probe_owner.get("probe") or {}).get("completed_updates"),
                            (probe_owner.get("probe") or {}).get("completed_training_examples"),
                        ),
                        (2, 18),
                    )
                ):
                    raise ValueError("isolated retained F diagnostic is incomplete")
                guard()
            if retained_rehearsal:
                from src.adapters.minigrid_affine_rehearsal import run_retained_rehearsal

                rehearsal_attempts = 1
                rehearsal_work.update(attempts=1, unreturned_work_unknown=True)
                report["rehearsal"] = rehearsal_owner = run_retained_rehearsal(
                    original + fresh,
                    states[-1][2],
                    readout_budget=budget,
                    check_boundary=guard,
                    **(dict(native_games=True) if native_games else {}),
                )
                rehearsal_returns = 1
                rehearsal_work.update(returned=1, unreturned_work_unknown=False)
                rehearsal_check, detached_rehearsal = paid_report_guard(rehearsal_owner)
                guard()
                compared = rehearsal_owner.get("comparison") or {}
                if (
                    rehearsal_owner.get("status") != "complete"
                    or rehearsal_owner.get("source_native_state_unchanged") is not True
                    or compared.get("status") != "complete"
                    or not _same_value(
                        rehearsal_owner.get("comparison_work"),
                        dict(attempts=1, returned=1, unreturned_work_unknown=False),
                    )
                    or not _same_value(
                        (
                            (compared.get("budget") or {}).get("updates_completed"),
                            (compared.get("budget") or {}).get("replay_examples_completed"),
                        ),
                        (4, 36),
                    )
                ):
                    raise ValueError("retained rehearsal is incomplete")
                guard()
            report["status"] = "complete"
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            try:
                guard()
            except Exception as error:
                report.update(
                    status="failed", capture_error=f"{type(error).__name__}: {error}"[:512]
                )
            if retained_probe:
                report["probe"] = detached_probe
                report["probe_work"] = dict(
                    attempts=probe_attempts,
                    returned=probe_returns,
                    unreturned_work_unknown=probe_attempts != probe_returns,
                )
            if retained_rehearsal:
                report["rehearsal"] = detached_rehearsal
                report["rehearsal_work"] = dict(
                    attempts=rehearsal_attempts,
                    returned=rehearsal_returns,
                    unreturned_work_unknown=rehearsal_attempts != rehearsal_returns,
                )
            report["protocol"] = original_protocol
            if is_fresh:
                report["collection"] = [p[2] for p in collection_proofs] + deepcopy(
                    collection[len(collection_proofs) : 2]
                )
            report["source"], report["training"] = detached_source, detached_training
            report["games"] = [detached for _, _, detached in proofs]
            report["summary"] = (
                _controller_summary(report["training"], complete=report["status"] == "complete")
                if controller_mode is not None
                else summarize_games(report["games"] if report["status"] == "complete" else ())
            )
            report["readout_budget"] = dict(
                initial_limits=asdict(expected_limits),
                limits=asdict(budget.budget),
                started_at=binding[2],
                observed_started_at=budget.started_at,
                updates=budget.updates_completed,
                replay=budget.replay_examples_completed,
                ended_at=monotonic(),
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report


def _emit_fresh_cli_result(result, destination, label):
    # Why this: indentation alone exceeded the byte cap after paid native work.
    # Share the compact writer and explicit LF so both commands measure written bytes.
    payload = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    if len(payload.encode("utf8")) > 8 * 1024**2:
        raise ValueError(f"{label} exceeds8MiB; executed work must not retry")
    if destination is not None:
        with destination.open("x", encoding="utf8", newline="\n") as output:
            output.write(payload)
    print(payload, end="")
    if result["status"] != "complete":
        raise SystemExit(2)


def handle_rehearsal_cli(args, parser):
    allowed = {"example_affine_a4", "context_linear_retained_rehearsal", "describe", "output"}
    if not args.example_affine_a4 or any(
        v is not None and v is not False for k, v in vars(args).items() if k not in allowed
    ):
        parser.error("fresh retained rehearsal accepts only its mode and describe/output")
    options = dict(
        reference_policy=ReferencePolicy("drift"),
        retained_rehearsal=True,
        native_games=args.context_linear_retained_rehearsal == "games",
    )
    if args.describe:
        if args.output is not None:
            parser.error("describe requires no output")
        print(json.dumps(describe_fresh_adaptation(**options), indent=2, allow_nan=False))
        return
    if args.output is not None and args.output.exists():
        raise FileExistsError("fresh retained rehearsal output already exists")
    result = run_fresh_adaptation(**options)
    _emit_fresh_cli_result(result, args.output, "fresh retained rehearsal")


def handle_controller_cli(args, parser):
    allowed = {"example_affine_a4", "context_linear_controller", "describe", "output"}
    if not args.example_affine_a4 or any(
        v is not None and v is not False for k, v in vars(args).items() if k not in allowed
    ):
        parser.error("fresh A4 controller accepts only its mode and describe/output")
    if args.describe:
        if args.output is not None:
            parser.error("describe requires no output")
        print(
            json.dumps(
                describe_fresh_adaptation(controller_mode=args.context_linear_controller),
                indent=2,
                allow_nan=False,
            )
        )
        return
    if args.output is not None and args.output.exists():
        raise FileExistsError("fresh A4 controller output already exists")
    result = run_fresh_adaptation(controller_mode=args.context_linear_controller)
    _emit_fresh_cli_result(result, args.output, "fresh A4 controller")


def handle_reference_cli(args, parser):
    allowed = {
        "example_affine_a4",
        "context_linear_reference",
        "ready_feedback",
        "describe",
        "output",
    }
    if any(v is not None and v is not False for k, v in vars(args).items() if k not in allowed):
        parser.error(
            "fresh A4 reference accepts only its policy, --ready-feedback and describe/output"
        )
    if not args.ready_feedback:
        parser.error("fresh A4 reference requires explicit --ready-feedback")
    policy = ReferencePolicy(args.context_linear_reference)
    if args.describe:
        if args.output is not None:
            parser.error("describe requires no output")
        print(
            json.dumps(
                describe_fresh_adaptation(reference_policy=policy), indent=2, allow_nan=False
            )
        )
        return
    if args.output is not None and args.output.exists():
        raise FileExistsError("fresh A4 reference output already exists")
    result = run_fresh_adaptation(reference_policy=policy)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf8")) > 8 * 1024**2:
        raise ValueError("fresh A4 reference exceeds8MiB; work must not retry")
    if args.output is not None:
        with args.output.open("x", encoding="utf8") as output:
            output.write(payload)
    print(payload)
    if result["status"] != "complete":
        raise SystemExit(2)


def handle_cli(args, parser):
    current_only = getattr(args, "context_linear_a4_current_only", False)
    current_repeated = getattr(args, "context_linear_a4_current_repeated", False)
    allowed = {
        "describe",
        "source",
        "output",
        "context_linear_a4_adaptation",
        "context_linear_a4_current_only",
        "context_linear_a4_current_repeated",
    }
    if any(v is not None and v is not False for k, v in vars(args).items() if k not in allowed):
        parser.error("A4 adaptation accepts no other modes or fit options")
    if args.describe:
        if args.source is not None or args.output is not None:
            parser.error("describe requires no paths")
        print(
            json.dumps(
                describe_adaptation(current_only=current_only, current_repeated=current_repeated),
                indent=2,
                allow_nan=False,
            )
        )
        return
    if args.source is None:
        parser.error("A4 adaptation requires fresh170 TRAIN --source")
    if args.output is not None and args.output.exists():
        raise FileExistsError("A4 adaptation output already exists")
    result = (
        run_adaptation(args.source, current_repeated=True)
        if current_repeated
        else run_adaptation(args.source, current_only=True)
        if current_only
        else run_adaptation(args.source)
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf8")) > 8 * 1024**2:
        raise ValueError("A4 adaptation exceeds8MiB; work must not retry")
    if args.output is not None:
        with args.output.open("x", encoding="utf8") as out:
            out.write(payload)
    print(payload)
    if result["status"] != "complete":
        raise SystemExit(2)
