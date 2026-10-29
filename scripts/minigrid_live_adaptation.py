"""Fresh A4 adaptation and explicit guarded deployment with the existing actor.

Composes existing finite catalogue, live runtime, collector and game ports. The
readouts never select updates or publications. The explicit deployment mode uses
the original current-actor start and promotion controller. No checkpoint restore,
benchmark learner, probe or final evaluation belongs here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import random
from time import monotonic
from typing import Any

from scripts import minigrid_affine_serving as published
from scripts.minigrid_a4_adaptation import _collect_fresh_a4, _require_phase, summarize_games
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_context_linear_stream_checkpoint,
)
from src.adapters.minigrid_serving import run_promotable_minigrid_episode
from src.adapters.minigrid_train_catalogue import compose_arrived_cue_catalogue
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import _record_tick, _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor, PromotionRejected
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LogicalClock
from src.core.minigrid_key_door import MiniGridKeyDoorLayout
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler


def describe_live_adaptation(*, deploy=False):
    if type(deploy) is not bool:
        raise ValueError("deploy requires an exact bool")
    protocol = dict(
        definition="portable_live_published_affine_a4_v1",
        source_sha256=published.TRAIN_SHA,
        source="bundled A3 TRAIN fit/B inner guards/C unused; fresh A4 TRAIN collected after publication",
        recipe="A3 blue/red, fresh A4 blue/red, retained A3 blue/red return; one earlier packet per cue",
        label_delay=1,
        fit_calls=6,
        fit_rows=90,
        original_unique_fit_rows=36,
        arrival_row_exposures=54,
        replay_row_exposures=36,
        teacher_steps=18,
        private_games="fixed A3/A4 TRAIN blue/red at updates2/4/6; seed17/max16/rawK7 firstargmax",
        actor_games="actual shared published A3 TRAIN blue/red after updates4/6; actor remains2/18",
        publications=1,
        native_wall_seconds=120,
        readout_wall_seconds=150,
        work_deadline_seconds=140,
        max_process_rss_bytes=512 * 1024**2,
        qualification="known TRAIN integration; retention requires learned old success; no generalization/matchedcompute/CPC/release claim",
    )
    if deploy:
        protocol.update(
            definition="portable_live_affine_a4_deployment_v1",
            recipe="A3 blue/red then fresh A4 blue/red; one earlier TRAIN packet per cue",
            fit_calls=4,
            fit_rows=54,
            arrival_row_exposures=36,
            replay_row_exposures=18,
            private_games="none; readouts use the actual shared published actor",
            actor_games="A3/A4 TRAIN blue/red before and after second publication; seed17/max16/rawK7 firstargmax",
            publications=2,
            candidate_start="original publication receipt/current actor, offset2/capacity2, same original resources",
            guard="unchanged B sentinel guards; not learned-old/A4 guards; no outcome-driven selection",
        )
    return protocol


def _plan(episodes, *, return_to_old=True):
    # Why: preserve the accepted A4 arrival/replay recipe and exposure IDs. Its
    # first two wakes are identical before and after fresh source acquisition.
    indices = (
        (0, 1) if len(episodes) == 2 else (0, 1, 2, 3, 0, 1) if return_to_old else (0, 1, 2, 3)
    )
    scheduled = tuple(
        replace(episodes[j], episode_id=f"{episodes[j].episode_id}:wake:{i}")
        for i, j in enumerate(indices)
    )
    return compose_arrived_cue_catalogue(scheduled)


def run_live_adaptation(*, deploy=False):
    protocol = describe_live_adaptation(deploy=deploy)
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        phase="preflight",
        reason=None,
        collection=[],
        games=[],
        actor_games=[],
        updates=[],
        summary=None,
        preservation=None,
        work=dict(
            guard_predictions_returned=0,
            guard_rows_returned=0,
            guard_rows_may_be_unknown=False,
            train_work_may_be_unobserved=False,
        ),
    )
    rng = None
    try:
        published._require_minigrid()
        if deploy:
            rng = random.getstate(), published.np.random.get_state()
        return _run_live(report)
    except Exception as error:
        report.update(
            status="failed",
            reason=f"{type(error).__name__}: {error}"[:512],
            preservation=False,
            summary=None,
        )
        return report
    finally:
        if rng is not None:
            report["global_rng"] = dict(
                python_unchanged=_same_value(random.getstate(), rng[0]),
                numpy_unchanged=_same_value(published.np.random.get_state(), rng[1]),
            )
            if not all(report["global_rng"].values()):
                report.update(
                    status="failed",
                    preservation=False,
                    summary=None,
                    reason="live deployment global RNG changed",
                )


def _deployment_summary(rows):
    expected = [
        (stage, branch, cue) for stage in (2, 4) for branch in (3, 4) for cue in ("blue", "red")
    ]
    if [(r["updates"], r["branch"], r["cue"]) for r in rows] != expected:
        raise ValueError("deployment summary requires all eight ordered actual-actor games")
    rates = []
    for stage in (2, 4):
        successes = {3: 0, 4: 0}
        for row in rows:
            if row["updates"] != stage:
                continue
            game = row["game"]
            if game["status"] != "complete" or len(game["episodes"]) != 1:
                raise ValueError("complete deployment game evidence required")
            episode = game["episodes"][0]
            environment, serving = episode["environment"], game["serving"]
            if (
                episode["status"] != "complete"
                or episode["source_role"] != "train"
                or type(episode["success"]) is not bool
                or environment["closed"] is not True
                or episode["cue"] != row["cue"]
                or environment["cue"] != row["cue"]
                or environment["role"] != "train"
                or environment["seed"] != 17
                or environment["max_steps"] != 16
                or environment["failure_work_may_be_unobserved"] is not False
                or environment["layout"]
                != asdict(MiniGridKeyDoorLayout("A", branch_row=row["branch"]))
                or (
                    serving["actor_version"],
                    serving["generation"],
                    serving["native_updates"],
                    serving["native_rows"],
                )
                != (("c0", 1, 2, 18) if stage == 2 else ("c1", 2, 4, 54))
            ):
                raise ValueError("closed known TRAIN actor outcomes required")
            successes[row["branch"]] += episode["success"]
        rates.append(dict(updates=stage, old_A3=successes[3] / 2, new_A4=successes[4] / 2))
    return dict(
        complete=True,
        success_by_stage=rates,
        acquisition_gain=rates[1]["new_A4"] - rates[0]["new_A4"],
        old_retention_delta=rates[1]["old_A3"] - rates[0]["old_A3"],
        learning_recovery=None,
        qualification="two known TRAIN cues; no return/loss-regain phase, generalization/matchedcompute/CPC/release claim",
    )


def _run_live(report):
    from scripts.run_minigrid_linear_fit_control import read_training_sources

    deployment = report["protocol"]["publications"] == 2
    total_updates = 4 if deployment else 6
    started = monotonic()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        native = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=total_updates,
                max_wall_seconds=120,
                max_process_rss_bytes=512 * 1024**2,
            ),
            monotonic,
        )
        readout = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=150,
                max_process_rss_bytes=512 * 1024**2,
            ),
            monotonic,
        )
        native.attach_memory(memory)
        readout.attach_memory(memory)
        pins = (budget_binding(native), budget_binding(readout))
        limits = (replace(native.budget), replace(readout.budget))
        protocol = report["protocol"]
        protocol_check, original_protocol = paid_report_guard(protocol)
        lists = {name: report[name] for name in ("collection", "games", "actor_games", "updates")}
        # Fixed finite output slots: 2 collection + 3 private + 4 actor + 6 polls
        # + 4 named records. No state/success-record chain grows with serving.
        proofs: list[tuple[str, Any, Any, Any, Any]] = []
        source = gate = None
        actor: Any = None
        runtime: Any = None
        shared: Any = None
        expected_state = actor_pin = source_pin = episodes_pin = None
        plan_pin = None
        plan = None
        episodes: tuple[Any, ...] = ()
        expected_updates = 0
        receipts: tuple[Any, ...] = ()
        receipt_offset = 0
        expected_base, expected_candidate = "a0", "c0"
        retired: Any = None
        retired_cursor: Any = None
        retired_state: Any = None
        gate_limits = gate_resource = None
        gate_lock = None

        def retain(name, value, index=None):
            check, detached = paid_report_guard(value)
            proofs.append((name, index, value, check, detached))

        def boundary():
            if monotonic() - started >= 140:
                raise TimeoutError("live A4 deadline; original cleanup time reserved")
            if report["protocol"] is not protocol:
                raise ValueError("original live protocol replaced")
            protocol_check()
            for owner, pin, original in zip((native, readout), pins, limits, strict=True):
                if budget_binding(owner) != pin or owner.budget != original:
                    raise ValueError("original live lifetime budget controls changed")
            readout.before_final()
            if readout.updates_completed or readout.replay_examples_completed:
                raise ValueError("live readout cannot update or replay")
            for name, owner in lists.items():
                if report[name] is not owner:
                    raise ValueError("original paid list replaced")
                if name != "collection" and len(owner) != sum(key == name for key, *_ in proofs):
                    raise ValueError("unobserved paid list entry or missing prefix")
            for name, index, owner, check, _ in proofs:
                observed = report[name] if index is None else report[name][index]
                if observed is not owner:
                    raise ValueError("preceding paid live evidence replaced")
                check()
            # Collector appends only after native close; protect each completed
            # record before its callback can admit the next teacher step/source.
            known = sum(name == "collection" for name, *_ in proofs)
            if not known <= len(lists["collection"]) <= 2:
                raise ValueError("bounded original collection prefix changed")
            for index in range(known, len(lists["collection"])):
                record = lists["collection"][index]
                if record.get("status") != "complete" or record.get("closed") is not True:
                    raise ValueError("preceding native collection incomplete")
                retain("collection", record, index)
            if source is not None and not _same_value(source.snapshot_state(), source_pin):
                raise ValueError("original source model changed")
            if episodes_pin is not None and not _same_value(episodes, episodes_pin):
                raise ValueError("original TRAIN sources changed")
            if plan_pin is not None and not _same_value(plan, plan_pin):
                raise ValueError("original arrived catalogue changed")
            if actor is not None and not _same_value(actor.serving_snapshot(), actor_pin):
                raise ValueError("published actor changed")
            if runtime is not None:
                if (
                    runtime._budget is not native
                    or runtime._inbox._budget is not native
                    or runtime._inbox._clock is not clock
                    or runtime.actor is not actor
                    or runtime._base_actor_version != expected_base
                    or runtime._candidate_version != expected_candidate
                    or native.updates_completed != expected_updates
                    or not _same_value(runtime.candidate_snapshot().state, expected_state)
                    or not _same_value(runtime.applied_updates, receipts)
                ):
                    raise ValueError("private candidate, original charge or receipt changed")
                if (
                    shared._runtime is not runtime
                    or shared._sharing is not gate
                    or gate._limits is not gate_limits
                    or gate._resource is not gate_resource
                    or gate._gate is not gate_lock
                    or gate.snapshot().limits != SharingLimits(1, total_updates, 1)
                    or gate.snapshot().admitted_updates != expected_updates
                ):
                    raise ValueError("original sharing controls or admissions changed")
            if retired is not None and (
                not retired._retired
                or retired._budget is not native
                or retired._inbox._clock is not clock
                or not _same_value(retired._inbox.capture_cursor(), retired_cursor)
                or not _same_value(retired._candidate.snapshot_state(), retired_state)
            ):
                raise ValueError("preceding retired candidate history or state changed")

        def tick(training, logical_tick):
            nonlocal expected_state, expected_updates, receipts
            boundary()
            clock.advance_to(logical_tick)
            _record_tick(runtime, training, logical_tick, 1)
            report["work"]["train_work_may_be_unobserved"] = True
            poll = shared.train_ready()
            report["updates"].append(asdict(poll))
            retain("updates", report["updates"][-1], len(report["updates"]) - 1)
            if poll.deferred_reason is not None or len(poll.updates) != 1:
                raise ValueError("one ordinary delayed paid update required")
            observed_receipts = runtime.applied_updates
            if (
                len(observed_receipts) != expected_updates + 1 - receipt_offset
                or not _same_value(observed_receipts[: expected_updates - receipt_offset], receipts)
                or not _same_value(poll.updates, observed_receipts[-1:])
                or poll.updates[0].update_number != expected_updates + 1
            ):
                raise ValueError("preceding paid receipt prefix changed during update")
            expected_updates += 1
            expected_state = runtime.candidate_snapshot().state
            receipts = deepcopy(observed_receipts)
            expected_rows = (0, 9, 18, 36, 54, 72, 90)[expected_updates]
            if (
                expected_state.completed_updates,
                expected_state.training_examples,
                expected_state.failed_work,
            ) != (expected_updates, expected_rows, None):
                raise ValueError("private native work differs from fixed A4 recipe")
            report["work"]["train_work_may_be_unobserved"] = False
            boundary()

        def private_games():
            boundary()
            try:
                games = evaluate_context_linear_stream_checkpoint(
                    source,
                    expected_state,
                    budget=readout,
                    check_boundary=boundary,
                    a4_training=True,
                )
            except MiniGridEvaluationFailed as error:
                report["failed_phase"] = dict(updates=expected_updates, games=error.report)
                raise
            try:
                _require_phase(games)
            except Exception:
                report["failed_phase"] = dict(updates=expected_updates, games=games)
                raise
            phase = dict(updates=expected_updates, games=games)
            report["games"].append(phase)
            retain("games", phase, len(report["games"]) - 1)
            boundary()

        def publish(guard_key, publication_key, *, now, version, generation):
            nonlocal actor_pin
            boundary()
            try:
                ticket = controller.prepare(
                    runtime,
                    *published._inner_guards(episodes),
                    now=now,
                    training_ids=train_ids,
                    metadata={"workflow": "live_a4"},
                )
            except PromotionRejected as error:
                report[guard_key] = asdict(error.report)
                raise
            except Exception:
                report["work"]["guard_work_may_be_unobserved"] = True
                raise
            report[guard_key] = asdict(ticket.report)
            retain(guard_key, report[guard_key])
            boundary()
            if deployment:
                report["work"]["publication_work_may_be_unobserved"] = True
            receipt = controller.commit(runtime, ticket)
            report[publication_key] = asdict(receipt)
            actor_pin = actor.serving_snapshot()
            if (actor_pin.actor_version, actor_pin.generation) != (
                version,
                generation,
            ) or not _same_value(actor_pin.model_state, expected_state):
                raise ValueError("published actor differs from accepted TRAIN candidate")
            retain(publication_key, report[publication_key])
            boundary()
            if deployment:
                report["work"]["publication_work_may_be_unobserved"] = False
            return receipt

        def actor_game(stage, cue, branch=3):
            report["phase"] = (
                f"actor-{stage}-{branch}-{cue}" if deployment else f"actor-{stage}-{cue}"
            )
            boundary()
            try:
                game = run_promotable_minigrid_episode(
                    actor,
                    cue=cue,
                    episode_id=f"deploy-a4-{stage}-{branch}-{cue}"
                    if deployment
                    else f"live-a4-{stage}-{cue}",
                    budget=readout,
                    check_boundary=boundary,
                    shared_runtime=shared,
                    **(
                        {"layout": MiniGridKeyDoorLayout("A", branch_row=branch)}
                        if deployment
                        else {}
                    ),
                )
            except MiniGridEvaluationFailed as error:
                report["failed_actor_game"] = dict(updates=stage, cue=cue, report=error.report)
                if deployment:
                    report["failed_actor_game"]["branch"] = branch
                raise
            row = dict(updates=stage, game=game)
            if deployment:
                row.update(branch=branch, cue=cue)
            report["actor_games"].append(row)
            retain("actor_games", row, len(report["actor_games"]) - 1)
            boundary()

        try:
            boundary()
            report["phase"] = "source"
            with published.example_affine_training() as path:
                episodes = read_training_sources(path, expected_sha256=published.TRAIN_SHA)
            published._training_batches(episodes)
            episodes_pin = deepcopy(episodes)
            prefix = _plan(episodes[:2])
            plan, plan_pin = prefix, deepcopy(prefix)
            source = ContextLinearLearner()
            source_pin = source.snapshot_state()
            actor = PromotableActor(
                source,
                version="a0",
                configuration=ServingConfiguration(3, 2),
                feature_digest=published._digest,
                metadata={"workflow": "live_a4", "phase": "cold"},
            )
            actor_pin = actor.serving_snapshot()
            clock = LogicalClock()
            runtime = ActorShadowRuntime(
                source,
                actor_version="a0",
                candidate_version="c0",
                clock=clock,
                budget=native,
                actor=actor,
                max_experiences=total_updates,
                max_consolidations=0,
            )
            expected_state = runtime.candidate_snapshot().state
            gate_resource = lambda: True
            gate = ServingPriorityGate(
                SharingLimits(1, total_updates, 1), resource_available=gate_resource
            )
            gate_limits, gate_lock = gate._limits, gate._gate
            shared = ResourceSharedRuntime(runtime, gate)
            receipts = deepcopy(runtime.applied_updates)
            report["phase"] = "initial-train"
            _record_tick(runtime, prefix.training, 0, 1)
            tick(prefix.training, 1)
            tick(prefix.training, 2)
            report["phase"] = "publication"
            controller = published._controller(actor, report["work"])
            train_ids = frozenset(
                (e.episode_id, f"{e.episode_id}:step:{t.step}")
                for e in episodes[:2]
                for t in e.transitions
            )
            publication = publish("guards", "publication", now=2, version="c0", generation=1)
            if deployment:
                for branch in (3, 4):
                    for cue in ("blue", "red"):
                        actor_game(2, cue, branch)
                report["phase"] = "current-actor-start"
                boundary()
                gate.pause()
                cursor_before = runtime._inbox.capture_cursor()
                state_before = deepcopy(expected_state)
                previous = runtime
                report["work"]["start_work_may_be_unobserved"] = True
                prepared = controller.start_candidate(shared, publication, candidate_version="c1")
                if (
                    shared._runtime is not prepared
                    or not previous._retired
                    or prepared._budget is not native
                    or prepared._inbox._clock is not clock
                    or (
                        prepared._base_actor_version,
                        prepared._candidate_version,
                        prepared._inbox._capacity,
                    )
                    != ("c0", "c1", 2)
                    or not _same_value(prepared.candidate_snapshot().state, state_before)
                ):
                    raise ValueError(
                        "current-actor start did not preserve the original resources/state"
                    )
                cursor_after = prepared._inbox.capture_cursor()
                if (
                    cursor_after.completed_before_generation != 2
                    or cursor_after.completed_updates != 2
                    or cursor_after.applied
                ):
                    raise ValueError(
                        "current-actor start requires the explicit preceding two-update offset"
                    )
                runtime, retired = prepared, previous
                retired_cursor, retired_state = deepcopy(cursor_before), state_before
                receipt_offset = 2
                expected_base, expected_candidate = "c0", "c1"
                receipts = ()
                report["start"] = dict(
                    base="c0",
                    candidate="c1",
                    completed_before_generation=2,
                    remaining_capacity=2,
                    old_receipts=[asdict(r) for r in cursor_before.applied],
                    old_native_updates=state_before.completed_updates,
                    old_native_rows=state_before.training_examples,
                )
                retain("start", report["start"])
                boundary()
                report["work"]["start_work_may_be_unobserved"] = False
                gate.resume()
            else:
                private_games()
            report["phase"] = "fresh-A4-collection"
            fresh, provenance = _collect_fresh_a4(boundary, lists["collection"])
            report["source"] = provenance
            retain("source", provenance)
            boundary()
            plan = _plan(episodes[:2] + fresh, return_to_old=not deployment)
            if not _same_value(plan.training[:2], prefix.training):
                raise ValueError("fresh plan replaced the preceding paid catalogue prefix")
            plan_pin = deepcopy(plan)
            report["catalogue"] = dict(
                wakes=[asdict(w) for w in plan.wakes],
                arrival_rows=plan.arrival_row_exposures,
                replay_rows=plan.replay_row_exposures,
                native_rows=plan.fit_row_exposures,
                input_array_bytes=plan.input_array_bytes,
                peak_bank_rows=plan.peak_bank_rows,
                peak_bank_float64_equivalent_bytes=plan.peak_bank_observation_target_float64_equivalent_bytes,
            )
            retain("catalogue", report["catalogue"])
            # The initial prefix already released label1 at tick2. Queue the
            # freshly observed A4 batch at that same tick without relabeling it.
            runtime.record_experience(
                Experience(
                    "batch-2",
                    "attribution",
                    2,
                    expected_base,
                    plan.training[2].features,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            for stage in (4,) if deployment else (4, 6):
                report["phase"] = f"private-train-{stage}"
                for logical_tick in (stage - 1, stage):
                    tick(plan.training, logical_tick)
                if deployment:
                    report["phase"] = "publication-after-A4"
                    publish("guards_after", "publication_after", now=4, version="c1", generation=2)
                    for branch in (3, 4):
                        for cue in ("blue", "red"):
                            actor_game(stage, cue, branch)
                else:
                    private_games()
                    for cue in ("blue", "red"):
                        actor_game(stage, cue)
            report["summary"] = (
                _deployment_summary(report["actor_games"])
                if deployment
                else summarize_games(report["games"])
            )
            boundary()
            report.update(status="complete", preservation=True)
        except Exception as error:
            report.update(
                status="failed",
                reason=f"{type(error).__name__}: {error}"[:512],
                preservation=False,
                summary=None,
            )
        finally:
            try:
                if report["status"] == "complete":
                    boundary()
                report["sharing"] = None if gate is None else asdict(gate.snapshot())
                if deployment:
                    report["work"]["candidate_owner_may_be_unknown"] = (
                        shared is not None and shared._runtime is not runtime
                    )
                if runtime is not None:
                    # A stopped candidate still has readable permanent receipts.
                    # Capture them before attempting a model read that may refuse.
                    report["private"] = dict(
                        native_updates=None,
                        native_rows=None,
                        failed_work=None,
                        base_version=runtime._base_actor_version,
                        receipts=[asdict(r) for r in runtime.applied_updates],
                        last_valid_receipts=[asdict(r) for r in receipts],
                        readback_uncertain=True,
                    )
                    if deployment and actor is not None:
                        observed_actor = actor.serving_snapshot()
                        report["actor"] = dict(
                            version=observed_actor.actor_version,
                            generation=observed_actor.generation,
                            native_updates=observed_actor.model_state.completed_updates,
                            native_rows=observed_actor.model_state.training_examples,
                        )
                    state = runtime.candidate_snapshot().state
                    report["private"].update(
                        native_updates=state.completed_updates,
                        native_rows=state.training_examples,
                        failed_work=state.failed_work,
                        readback_uncertain=False,
                    )
                    if deployment:
                        report["private"]["completed_before_generation"] = receipt_offset
            except Exception as error:
                report.update(
                    status="failed",
                    preservation=False,
                    summary=None,
                    capture_error=f"{type(error).__name__}: {error}"[:512],
                )
            # Recover the original finite paid prefix even after tampering.
            for name, owner in lists.items():
                report[name] = [
                    detached
                    for key, index, _, _, detached in proofs
                    if key == name and index is not None
                ]
                if name == "collection":
                    report[name] += deepcopy(owner[len(report[name]) : 2])
            for name, index, _, _, detached in proofs:
                if index is None:
                    report[name] = detached
            report["protocol"] = original_protocol
            report["budgets"] = dict(
                native=published._budget_report(native, pins[0], limits[0]),
                readout=published._budget_report(readout, pins[1], limits[1]),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report
