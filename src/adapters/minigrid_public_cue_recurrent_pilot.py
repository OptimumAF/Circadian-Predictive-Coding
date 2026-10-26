"""Fixed-A and recurring two-head recipes using existing stream/readout ports.

Inputs are the accepted original TRAIN source archive; output is one bounded
engineering report. No new teacher, optimizer, stream framework, final access,
probe, promotion, historical refit or parameter search is provided here.
"""

from copy import deepcopy
from dataclasses import asdict
from math import fsum, isfinite
import pickle
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_public_cue_health as health
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_public_cue_recurrent_checkpoint,
    _return_budget_guard,
)
from src.adapters.minigrid_public_cue_recurrent_learner import (
    METHOD,
    PublicCueRecurrentLearner,
    copy_snapshot,
)
from src.adapters.minigrid_weighted_bc_pilot import _admit_reference, _load_reference
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingUpdateFailed
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

BODY_SECONDS, TRAIN_SECONDS, READOUT_SECONDS = 300, 200, 100
RSS_LIMIT = 768 * 1024**2
RECURRING_BODY_SECONDS, RECURRING_TRAIN_SECONDS, RECURRING_READOUT_SECONDS = 420, 400, 200
REPLAY_BODY_SECONDS, REPLAY_TRAIN_SECONDS = 700, 660


def describe_protocol(
    *,
    typed_input: bool = False,
    recurring_training: bool = False,
    recurrent_replay: str | None = None,
    class_balanced_training: bool = False,
    activation_health: bool = False,
) -> dict[str, Any]:
    """Metadata only, before model construction or source IO."""
    if type(typed_input) is not bool:
        raise ValueError("typed_input must be an exact boolean")
    if type(recurring_training) is not bool or (recurring_training and not typed_input):
        raise ValueError("recurring_training requires exact bool and explicit typed TRAIN input")
    if type(class_balanced_training) is not bool or (
        class_balanced_training and (not recurring_training or recurrent_replay is not None)
    ):
        raise ValueError("class-balanced training requires typed recurring input without replay")
    if type(activation_health) is not bool or (activation_health and not recurring_training):
        raise ValueError("activation_health requires exact bool and typed recurring TRAIN input")
    if recurrent_replay is not None:
        from src.adapters import minigrid_recurrent_replay as replay

        replay.validate_mode(recurrent_replay)
        if not recurring_training:
            raise ValueError("recurrent replay requires explicit recurring typed input")
    result: dict[str, Any] = dict(
        protocol_id="minigrid_public_cue_recurrent_a128_v1",
        classification="single_seed_conventional_public_cue_specialization_engineering_control",
        method=METHOD,
        seed=17,
        feature_dim=148,
        hidden_dim_per_head=8,
        heads=2,
        num_classes=7,
        learning_rate=0.03,
        momentum=0.9,
        max_episode_steps=16,
        parameter_counts=dict(per_head=1319, total=2638, old_raw=1319, old_history=2503),
        label_delay=1,
        role="train",
        training_checkpoints=list(shared.POINTS),
        readout_order=[0, 128],
        training="original numbered128 TRAIN A episodes; one native complete-episode CE/SGD update each;64updates576rows per public-cue head; raw148 unchanged",
        routing="first actual PUBLIC observation only; select once per genuine recurrent session; no environment context/cue/teacher oracle",
        readout="AFTER all fits: original Ablue/Ared9row TRAIN scores and target-free native games at cold0 and endpoint128; seed17/max16/all7firstargmax",
        acquisition="both exact teacher sequences AND both native goal successes at128",
        planned_work=dict(
            updates=128,
            training_rows=1152,
            raw_calls=4,
            raw_rows=36,
            live_episodes=4,
            max_physics_steps=64,
            teacher_episodes=0,
            replay_rows=0,
            probe_updates=0,
            final_episodes=0,
            promotions=0,
        ),
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=320,
            training_seconds=TRAIN_SECONDS,
            readout_seconds=READOUT_SECONDS,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=8 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        limitations="different capacity/initializations/native update and momentum schedule/compute from old raw or paired/history controls; no equal-compute/capacity or isolated causal/CPC/retention/recovery/generalization/release claim; all old failures retained",
    )
    if typed_input:
        result.update(
            protocol_id="minigrid_public_cue_recurrent_typed_a128_v1",
            source_mode="caller_owned_six_complete_raw148_TRAIN_episodes",
            provenance="typed metadata/role/geometry and PUBLIC cue validation; caller supplies collection provenance; no historical model/archive/budget authority",
        )
    if recurring_training:
        result.update(
            protocol_id="minigrid_public_cue_recurrent_typed_abca128_v1",
            classification="single_seed_recurring_known_TRAIN_conventional_engineering",
            context_sequence=list(shared.CONTEXTS),
            readout_order=list(shared.POINTS),
            training="original A32/B32/C_train32/A32 chronological complete raw148 TRAIN episodes;128nativeCE/SGD updates1152rows;64updates/head/delay1",
            readout="AFTER all fits: six known TRAIN score packets at all five points; native Ablue/red at32 then A/B/C_trainblue/red at128 ONLY; all7firstargmax/seed17/max16",
            native_readout_points={"32": ["A"], "128": ["A", "B", "C_train"]},
            acquisition="per-context teacher exactness and native goals reported separately; native forgetting at64/96 is unmeasured",
            limitations="known TRAIN contexts/single seed; different capacity/optimizer schedule/compute from old controls; teacher fit is not native success; native64/96 unmeasured; no isolated causal/CPC/heldout/reliability/release claim",
        )
        result["planned_work"].update(
            raw_calls=30, raw_rows=270, live_episodes=8, max_physics_steps=128
        )
        result["limits"].update(
            body_seconds=RECURRING_BODY_SECONDS,
            parent_seconds=440,
            training_seconds=RECURRING_TRAIN_SECONDS,
            readout_seconds=RECURRING_READOUT_SECONDS,
        )
    if recurrent_replay is not None:
        result.update(
            protocol_id=f"minigrid_public_cue_recurrent_abca_{recurrent_replay}_v1",
            replay=dict(
                mode=recurrent_replay,
                policy=asdict(replay.policy(recurrent_replay)),
                extra_updates=32,
                extra_rows=288,
                extra_updates_per_head=16,
                selection_information="released TRAIN context/cue metadata only; stronger coverage selector information, never native routing",
            ),
            training="original128 incoming TRAIN updates plus16 replay events of two complete nine-row episodes;160native updates1440rows/80updates per public-cue head",
        )
        result["planned_work"].update(
            incoming_updates=128,
            updates=160,
            incoming_rows=1152,
            training_rows=1440,
            replay_updates=32,
            replay_rows=288,
        )
    if class_balanced_training:
        result.update(
            protocol_id="minigrid_public_cue_class_balanced_typed_abca128_v1",
            class_balanced_training=True,
            training="original A32/B32/C_train32/A32;128 complete TRAIN weightedCE/SGD updates1152rows; fixed current-episode n/(present_classes*count), absent1;64updates/head/delay1",
            readout="AFTER all128fits: six TRAIN scores at all five points; native Ablue/red32/64/96 then A/B/C_trainblue/red128; all7firstargmax/seed17/max16",
            native_readout_points={
                "32": ["A"],
                "64": ["A"],
                "96": ["A"],
                "128": ["A", "B", "C_train"],
            },
            acquisition="report teacher exactness and native goals separately; recovery requires prior observed acquisition, intervening loss and later regain",
            limitations="known TRAIN/single seed; accepted132meanCE control reused only where assumptions unchanged; extra middle A games and weighted schedule/performance overhead differ, not matched compute/time; preserve all negatives; no causal/CPC/heldout/reliability/RL/release benefit claim",
        )
        result["planned_work"].update(live_episodes=12, max_physics_steps=192)
    if activation_health:
        result["protocol_id"] = result["protocol_id"].removesuffix("_v1") + "_activation_health_v1"
        result["activation_health"] = dict(
            definition="public_cue_TRAIN_ordered_recurrent_activation_health_v1",
            order="all fits first; six TRAIN health reads before ordinary readout at each declared point",
            thresholds="existing active/dead1e-3 and saturation.99; existing centered singular-value entropy rank",
            routing="actual first public observation selects native head, never TRAIN context or targets",
            budgets="same original born-once readout and whole-case budgets; no renewal",
            stored="bounded detached metrics only; no raw hidden history",
        )
        result["planned_work"].update(
            activation_reads=30,
            activation_forward_calls=270,
            activation_forward_rows=270,
            activation_output_projections=270,
            activation_svd_calls=30,
            activation_array_bytes_returned=30 * 9 * 8 * 8,
        )
    return result


def _typed_sources(episodes: tuple[DemonstrationEpisode, ...]) -> tuple:
    if type(episodes) is not tuple or len(episodes) != 6:
        raise ValueError("typed recurrent input requires six complete ordered raw TRAIN episodes")
    owned = tuple(copy_episode(e) for e in episodes)
    sources = tuple(shared._batch(e) for e in owned)
    shared._packets(sources)  # Reuse the complete original role/context/shape admission.
    from src.adapters.minigrid_public_cue_recurrent_learner import _cue, _raw, _consistent_cues

    for i, packet in enumerate(sources):
        cue = ("blue", "red")[i % 2]
        _raw(packet.features)
        _consistent_cues(packet.features, i % 2)
        if (
            packet.episode is None
            or packet.episode.episode_id != f"minigrid-source-{packet.context}-{cue}"
            or _cue(packet.features[0]) != i % 2
        ):
            raise ValueError("typed recurrent source ID/order or actual PUBLIC cue differs")
    return sources


def _outcomes(records: list[dict[str, Any]]) -> dict[str, Any]:
    if [r.get("updates") for r in records] != [0, 128]:
        raise ValueError("public-cue outcomes require both fixed endpoints")
    for record in records:
        if (
            record.get("status") != "complete"
            or record.get("method") != METHOD
            or record.get("role") != "train"
            or record.get("state_unchanged") is not True
            or len(record.get("scores", ())) != 2
            or len(record.get("episodes", ())) != 2
        ):
            raise ValueError("public-cue outcomes require complete preserved TRAIN evidence")
        for cue, score, episode in zip(
            ("blue", "red"), record["scores"], record["episodes"], strict=True
        ):
            if (
                (score["context"], score["cue"]) != ("A", cue)
                or (episode["context"], episode["cue"]) != ("A", cue)
                or episode["status"] != "complete"
                or episode.get("environment", {}).get("closed") is not True
                or type(episode["success"]) is not bool
            ):
                raise ValueError("public-cue outcome cue/order/cleanup differs")
    cold, final = records
    exact = all(s["exact_sequence_accuracy"] == 1 for s in final["scores"])
    goals = all(e["success"] is True for e in final["episodes"])
    return dict(
        raw_both_cues_exact=exact,
        live_both_cues_goals=goals,
        joint_acquisition=exact and goals,
        ce_gain=fsum(s["ce"] for s in cold["scores"]) / 2
        - fsum(s["ce"] for s in final["scores"]) / 2,
        live_success_count_change=sum(e["success"] for e in final["episodes"])
        - sum(e["success"] for e in cold["episodes"]),
        mean_return_gain=fsum(e["return"] for e in final["episodes"]) / 2
        - fsum(e["return"] for e in cold["episodes"]) / 2,
    )


def _recurring_outcomes(
    records: list[dict[str, Any]], *, class_balanced_training: bool = False
) -> dict[str, Any]:
    """Describe fixed TRAIN phases; never infer native outcomes at unplayed points."""
    if type(class_balanced_training) is not bool or (
        len(records) != 5
        or any(type(r.get("updates")) is not int for r in records)
        or [r["updates"] for r in records] != list(shared.POINTS)
    ):
        raise ValueError("recurring outcomes require all five original checkpoint readouts")
    contexts = ("A", "B", "C_train")
    keys = [(context, cue) for context in contexts for cue in ("blue", "red")]
    curves: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    for record in records:
        point = record["updates"]
        live_contexts = (
            contexts
            if point == 128
            else ("A",)
            if (point == 32 or (class_balanced_training and point in (64, 96)))
            else ()
        )
        if (
            record.get("method") != METHOD
            or record.get("role") != "train"
            or record.get("status") != "complete"
            or record.get("state_unchanged") is not True
            or [(s.get("context"), s.get("cue")) for s in record.get("scores", ())] != keys
            or [(e.get("context"), e.get("cue")) for e in record.get("episodes", ())]
            != [(c, cue) for c in live_contexts for cue in ("blue", "red")]
        ):
            raise ValueError("recurring readout roles/context/complete fixed plan differ")
        for score in record["scores"]:
            if any(
                type(score.get(k)) not in (float, int) or not isfinite(score[k])
                for k in ("ce", "accuracy", "exact_sequence_accuracy")
            ) or (
                score["ce"] < 0
                or not 0 <= score["accuracy"] <= 1
                or score["exact_sequence_accuracy"] not in (0, 1)
            ):
                raise ValueError("recurring TRAIN scores must be finite bounded native metrics")
        for context in contexts:
            pair = [s for s in record["scores"] if s["context"] == context]
            curves.append(
                dict(
                    updates=point,
                    context=context,
                    ce=fsum(s["ce"] for s in pair) / 2,
                    accuracy=fsum(s["accuracy"] for s in pair) / 2,
                    exact_cues=sum(int(s["exact_sequence_accuracy"]) for s in pair),
                )
            )
        for episode in record["episodes"]:
            if (
                episode.get("status") != "complete"
                or episode.get("state_unchanged") is not True
                or episode.get("environment", {}).get("closed") is not True
                or type(episode.get("success")) is not bool
                or type(episode.get("return")) not in (float, int)
                or not isfinite(episode["return"])
            ):
                raise ValueError(
                    "recurring native outcomes require complete preserved closed episodes"
                )
        for context in live_contexts:
            pair = [e for e in record["episodes"] if e["context"] == context]
            native.append(
                dict(
                    updates=point,
                    context=context,
                    successes=sum(e["success"] for e in pair),
                    mean_return=fsum(e["return"] for e in pair) / 2,
                )
            )
    by_key = {(r["updates"], r["context"]): r for r in curves}
    changes = []
    for before, after, trained in zip(
        shared.POINTS[:-1], shared.POINTS[1:], shared.CONTEXTS, strict=True
    ):
        for context in contexts:
            a, b = by_key[before, context], by_key[after, context]
            changes.append(
                dict(
                    before=before,
                    after=after,
                    trained_context=trained,
                    evaluated_context=context,
                    ce_gain=a["ce"] - b["ce"],
                    exact_cue_change=b["exact_cues"] - a["exact_cues"],
                )
            )
    return dict(
        TRAIN_curves=curves,
        phase_changes=changes,
        native_outcomes=native,
        direction="positive CE gain/exact-cue change means improvement; negative retained-context values mean loss",
        native64_96=(
            "A observed at64/96; recovery requires prior native acquisition, intervening loss and later regain"
            if class_balanced_training
            else "unmeasured; TRAIN forgetting/recovery is not proof of native forgetting/recovery"
        ),
        qualification="known TRAIN/single seed; no heldout reliability or matched-compute/CPC advantage",
    )


def run_pilot(
    episodes: tuple[DemonstrationEpisode, ...] | None = None,
    *,
    recurring_training: bool = False,
    recurrent_replay: str | None = None,
    class_balanced_training: bool = False,
    activation_health: bool = False,
) -> dict[str, Any]:
    """One fixed run; preceding full-state/receipts guarded before each readout."""
    from scripts.run_minigrid_history_evaluation_smoke import _fixed_a_packets

    started = monotonic()
    if type(recurring_training) is not bool or (recurring_training and episodes is None):
        raise ValueError("recurring_training requires exact bool and explicit typed TRAIN input")
    protocol = describe_protocol(
        typed_input=episodes is not None,
        recurring_training=recurring_training,
        recurrent_replay=recurrent_replay,
        class_balanced_training=class_balanced_training,
        activation_health=activation_health,
    )
    # Validate typed roles/frames before importing Torch or admitting any native work.
    typed_sources = None if episodes is None else _typed_sources(episodes)
    torch = require_torch()
    torch_rng, numpy_rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    body_seconds, train_seconds, readout_seconds = (
        protocol["limits"][key] for key in ("body_seconds", "training_seconds", "readout_seconds")
    )
    readout_points = tuple(protocol["readout_order"])
    report: dict[str, Any] = dict(
        protocol=protocol,
        status="running",
        reason=None,
        training=shared._row(METHOD),
        evaluations=[],
        outcomes=None,
        failure_work=None,
        readout_budget=None,
        endpoint=None,
        readback_error=None,
    )
    report["training"].pop("final_evaluation", None)
    protocol_bytes = pickle.dumps(protocol, protocol=5)
    signatures: list[Any] = []
    arm = None
    budget = None
    owners: dict[str, Any] = {}
    row = report["training"]
    row_bytes = None
    evaluations = report["evaluations"]
    evaluation_bytes = pickle.dumps(evaluations, protocol=5)
    reader_binding: tuple[Any, Any] | None = None
    health_records: list[dict[str, Any]] = []
    health_bytes = pickle.dumps(health_records, protocol=5)
    health_failure = None
    if activation_health:
        report["activation_health"] = health_records

    def boundary() -> None:
        peak = memory.peak_bytes
        if monotonic() - started >= body_seconds or peak is None or peak > RSS_LIMIT:
            raise ValueError("public-cue original whole-case time/RSS cap exhausted")
        if (
            report["protocol"] is not protocol
            or pickle.dumps(protocol, protocol=5) != protocol_bytes
            or report["training"] is not row
            or (row_bytes is not None and pickle.dumps(row, protocol=5) != row_bytes)
            or report["evaluations"] is not evaluations
            or pickle.dumps(evaluations, protocol=5) != evaluation_bytes
            or any(not shared._check_signature(v, s) for v, s in signatures)
            or (arm is not None and any(arm.get(k) is not v for k, v in owners.items()))
            or (
                activation_health
                and (
                    report.get("activation_health") is not health_records
                    or pickle.dumps(health_records, protocol=5) != health_bytes
                    or pickle.dumps(np.random.get_state()) != numpy_rng
                    or not bool(torch.equal(torch.get_rng_state(), torch_rng))
                )
            )
        ):
            raise ValueError("public-cue preceding evidence/input/runtime binding changed")

    def guard() -> None:
        boundary()
        if arm is not None:
            arm["guard"]()
        if reader_binding is not None:
            wrapper, native = reader_binding
            if (
                arm is None
                or arm["source"] is not wrapper
                or type(wrapper) is not TorchTrainWeightedLearner
                or wrapper._learner is not native
                or type(native) is not PublicCueRecurrentLearner
                or not _same_value(wrapper.snapshot_state(), native.snapshot_state())
            ):
                raise ValueError("original weighted wrapper/native readout binding changed")
        boundary()

    with ProcessRssSampler() as memory:
        try:
            if typed_sources is None:
                admitted = _admit_reference(_load_reference())
                sources = admitted["sources"]
            else:
                admitted, sources = None, typed_sources
            packets = shared._packets(sources) if recurring_training else _fixed_a_packets(sources)
            if recurrent_replay is not None:
                from src.adapters import minigrid_recurrent_replay as replay

                packets = replay.prepare_packets(packets)
            signatures.extend((v, shared._signature(v)) for v in (sources, packets))
            report["reference"] = dict(
                path=None if admitted is None else admitted["provenance"]["reference_path"],
                scope="caller-owned typed TRAIN episodes; no historical reads/budget authority"
                if admitted is None
                else "original TRAIN data only; no historical model reconstruction/refit or budget renewal",
            )

            def factory(method: str) -> PublicCueRecurrentLearner | TorchTrainWeightedLearner:
                if method != METHOD:
                    raise ValueError("public-cue factory refuses other methods")
                if class_balanced_training:
                    from src.adapters.minigrid_public_cue_recurrent_learner import (
                        PublicCueRecurrentPolicy,
                    )

                    return TorchTrainWeightedLearner(
                        PublicCueRecurrentPolicy(), tuple(shared._episode(p) for p in packets)
                    )
                return PublicCueRecurrentLearner()

            replay_options: dict[str, Any] = {}
            if recurrent_replay is not None:
                replay_options["recurrent_replay"] = recurrent_replay
            arm = shared._train_arm(
                row,
                packets,
                memory,
                started + train_seconds,
                boundary,
                learner_factory=factory,
                **replay_options,
            )
            row_bytes = pickle.dumps(row, protocol=5)
            owners = {k: arm[k] for k in ("source", "states", "stream", "budget", "guard")}
            reader = arm["source"]
            if class_balanced_training:
                if type(reader) is not TorchTrainWeightedLearner:
                    raise ValueError("class-balanced source lost its exact weighted schedule owner")
                reader_binding = (reader, reader._learner)
                reader = reader._learner
                guard()
            if [n for n, _ in arm["states"]] != list(shared.POINTS):
                raise ValueError("public-cue stream lost an original checkpoint")
            by_point = dict(arm["states"])
            endpoint = copy_snapshot(by_point[128])
            expected_updates = 160 if recurrent_replay is not None else 128
            if (endpoint.completed_updates, endpoint.training_examples) != (
                expected_updates,
                9 * expected_updates,
            ) or any(
                (h.completed_updates, h.training_examples)
                != (expected_updates // 2, 9 * expected_updates // 2)
                for h in endpoint.heads
            ):
                raise ValueError("public-cue endpoint requires two64-update/576row heads")
            report["endpoint"] = dict(
                policy=asdict(endpoint.policy),
                completed_updates=endpoint.completed_updates,
                training_examples=endpoint.training_examples,
                heads=[
                    dict(
                        completed_updates=h.completed_updates, training_examples=h.training_examples
                    )
                    for h in endpoint.heads
                ],
                storage_scope="full owned native tensors retained in guarded stream states during readout; JSON metadata is not a resumable checkpoint",
            )
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=0,
                    max_wall_seconds=readout_seconds,
                    max_process_rss_bytes=RSS_LIMIT,
                ),
                monotonic,
            )
            budget.attach_memory(memory)
            budget_guard = _return_budget_guard(budget)
            for point in readout_points:
                guard()
                budget_guard()
                if activation_health:
                    try:
                        health_failure = health.read_checkpoint_health(
                            reader,
                            by_point[point],
                            updates=point,
                            sources=sources,
                            budget=budget,
                            check_boundary=guard,
                            recurrent_replay=recurrent_replay,
                        )
                    except health.ActivationHealthFailed as error:
                        health_failure = deepcopy(error.report)
                        raise
                    # Keep returned paid work even if this outer guard refuses.
                    guard()
                    health_records.append(health_failure)
                    health_bytes = pickle.dumps(health_records, protocol=5)
                    health_failure = None
                    guard()
                    budget_guard()
                try:
                    current = evaluate_public_cue_recurrent_checkpoint(
                        reader,
                        by_point[point],
                        updates=point,
                        sources=sources if recurring_training else sources[:2],
                        budget=budget,
                        check_boundary=guard,
                        **({"recurring_training": True} if recurring_training else {}),
                        **({"class_balanced_training": True} if class_balanced_training else {}),
                        **(
                            {"recurrent_replay": recurrent_replay}
                            if recurrent_replay is not None
                            else {}
                        ),
                    )
                except MiniGridEvaluationFailed as error:
                    evaluations.append(deepcopy(error.report))
                    evaluation_bytes = pickle.dumps(evaluations, protocol=5)
                    raise
                evaluations.append(current)
                evaluation_bytes = pickle.dumps(evaluations, protocol=5)
                guard()
                budget_guard()
            report["outcomes"] = (
                _recurring_outcomes(evaluations, class_balanced_training=class_balanced_training)
                if recurring_training
                else _outcomes(evaluations)
            )
            budget.complete_memory()
            budget_guard()
            report["readout_budget"] = dict(
                limits=asdict(budget.budget),
                started_at=budget.started_at,
                last_clock=budget.last_clock,
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
            )
            guard()
            report["status"] = "complete"
        except Exception as error:
            report.update(
                status="incomplete", reason=f"{type(error).__name__}: {error}", outcomes=None
            )
            if isinstance(error, TrainingUpdateFailed):
                report["failure_work"] = deepcopy(error.work)
        finally:
            try:
                guard()
            except Exception as error:
                report.update(
                    status="incomplete",
                    readback_error=f"{type(error).__name__}: {error}",
                    outcomes=None,
                )
            report["observed_process_rss"] = asdict(memory.snapshot())
            if budget is not None:
                report["readout_budget"] = dict(
                    limits=asdict(budget.budget),
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                    updates_completed=budget.updates_completed,
                    replay_examples_completed=budget.replay_examples_completed,
                )
    report["numpy_rng_unchanged"] = pickle.dumps(np.random.get_state()) == numpy_rng
    report["torch_rng_unchanged"] = bool(torch.equal(torch.get_rng_state(), torch_rng))
    if not report["numpy_rng_unchanged"] or not report["torch_rng_unchanged"]:
        report.update(status="incomplete", reason="global RNG changed", outcomes=None)
    report["total_seconds"] = monotonic() - started
    # Export the detached observed prefix; never repair or re-admit live owners.
    if row_bytes is not None:
        report["training"] = pickle.loads(row_bytes)
    report["evaluations"] = pickle.loads(evaluation_bytes)
    if activation_health:
        report["activation_health"] = pickle.loads(health_bytes)
        report["activation_health_failure"] = deepcopy(health_failure)
    if recurring_training:
        report["final_preservation"] = True if report["status"] == "complete" else None
    return report


def describe_replay_comparison() -> dict[str, Any]:
    """Fixed paired protocol; describing it grants no experiment allowance."""
    from src.adapters.minigrid_recurrent_replay import MODES

    return dict(
        protocol_id="minigrid_public_cue_recurrent_balanced_replay_v1",
        controls=[
            describe_protocol(typed_input=True, recurring_training=True, recurrent_replay=m)
            for m in MODES
        ],
        order="fit recent, fit coverage, then read recent0/32/64/96/128, then coverage0/32/64/96/128",
        limits=dict(
            body_seconds=REPLAY_BODY_SECONDS,
            training_lifetime_seconds=REPLAY_TRAIN_SECONDS,
            readout_seconds_per_arm=RECURRING_READOUT_SECONDS,
            process_rss_bytes=RSS_LIMIT,
        ),
        saved_no_replay=dict(
            issue=132,
            path="artifacts/runs/recurrent-context-stream-20261025/native-result.json",
            sha256="0b76d534e37b4c13aee5bd61bfdd48477e7ff82d20b8ace65c9aeb7ca1725763",
            reuse="saved outcomes/costs only, no refit/rescore/reconstruction",
        ),
        matching="recent versus coverage: same architecture/seed, incoming and extra update/row/per-public-cue exposures; selection/copy/storage overhead and episode identity differ; labelled TRAIN contexts available to coverage selection only",
        limitations="extra fitting versus saved132 is NOT matched compute; native64/96 unmeasured; no isolated causal/CPC/heldout/reliability/RL/release claim",
    )


def run_replay_comparison(episodes: tuple[DemonstrationEpisode, ...]) -> dict[str, Any]:
    """Fit both fixed controls through the existing runner before any readout.

    Why this: calling two single-arm pilots would expose first-arm outcomes
    before fitting the second. This finite composition keeps the original
    learner/runtime/budget owners and checks every completed peer before work.
    """
    from src.adapters import minigrid_recurrent_replay as replay

    started = monotonic()
    sources = _typed_sources(episodes)  # Roles and geometry before Torch or model work.
    packets = replay.prepare_packets(shared._packets(sources))
    signatures = [(value, shared._signature(value)) for value in (sources, packets)]
    protocol = describe_replay_comparison()
    frozen_protocol = pickle.dumps(protocol, protocol=5)
    torch = require_torch()
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    rows = [shared._row(METHOD) for _ in replay.MODES]
    records: list[list[dict[str, Any]]] = [[], []]
    for row in rows:
        row.pop("final_evaluation", None)
    report: dict[str, Any] = dict(
        protocol=protocol,
        status="running",
        reason=None,
        controls=[
            dict(mode=mode, training=row, evaluations=values, outcomes=None)
            for mode, row, values in zip(replay.MODES, rows, records, strict=True)
        ],
        failure_work=None,
        final_preservation=None,
    )
    arms: list[dict[str, Any]] = []
    arm_owners: list[dict[str, Any]] = []
    frozen_rows: list[bytes] = []
    frozen_records = [pickle.dumps(v, protocol=5) for v in records]
    budgets: list[ToyBudgetSession] = []
    budget_guards: list[Callable[[], None]] = []
    control_list = report["controls"]
    controls = tuple(control_list)
    outcome_bytes = [pickle.dumps(None, protocol=5) for _ in controls]

    def boundary():
        if (
            monotonic() - started >= REPLAY_BODY_SECONDS
            or memory.peak_bytes is None
            or memory.peak_bytes > RSS_LIMIT
        ):
            raise ValueError("original paired replay whole-case time/RSS cap exhausted")
        if (
            report["protocol"] is not protocol
            or pickle.dumps(protocol, protocol=5) != frozen_protocol
            or report["controls"] is not control_list
            or len(report["controls"]) != 2
            or any(a is not b for a, b in zip(report["controls"], controls, strict=True))
            or any(not shared._check_signature(v, s) for v, s in signatures)
        ):
            raise ValueError("paired replay original protocol/input/control binding changed")
        for i, control in enumerate(controls):
            if (
                control["mode"] != replay.MODES[i]
                or control["training"] is not rows[i]
                or control["evaluations"] is not records[i]
                or pickle.dumps(records[i], protocol=5) != frozen_records[i]
                or pickle.dumps(control["outcomes"], protocol=5) != outcome_bytes[i]
                or i < len(frozen_rows)
                and pickle.dumps(rows[i], protocol=5) != frozen_rows[i]
            ):
                raise ValueError("preceding paired replay evidence changed")
        for arm, owners in zip(arms, arm_owners, strict=True):
            if any(arm.get(k) is not v for k, v in owners.items()):
                raise ValueError("original paired replay arm was replaced")
            arm["guard"]()
        for check in budget_guards:
            check()

    def factory(method):
        if method != METHOD:
            raise ValueError("paired replay factory refuses other methods")
        return PublicCueRecurrentLearner()

    with ProcessRssSampler() as memory:
        try:
            for i, mode in enumerate(replay.MODES):
                boundary()
                arm = shared._train_arm(
                    rows[i],
                    packets,
                    memory,
                    started + REPLAY_TRAIN_SECONDS,
                    boundary,
                    learner_factory=factory,
                    recurrent_replay=mode,
                )
                if [n for n, _ in arm["states"]] != list(shared.POINTS):
                    raise ValueError("paired replay lost an original checkpoint")
                replay.check_snapshot(dict(arm["states"])[128], 128)
                arms.append(arm)
                arm_owners.append(
                    {k: arm[k] for k in ("source", "states", "stream", "budget", "guard", "replay")}
                )
                frozen_rows.append(pickle.dumps(rows[i], protocol=5))
                boundary()
            # Both fits have returned and passed their guards before the first scorer exists.
            for i, (mode, arm) in enumerate(zip(replay.MODES, arms, strict=True)):
                boundary()
                budget = ToyBudgetSession(
                    ToyExecutionBudget(
                        max_training_updates=0,
                        max_replay_examples=0,
                        max_wall_seconds=RECURRING_READOUT_SECONDS,
                        max_process_rss_bytes=RSS_LIMIT,
                    ),
                    monotonic,
                )
                budget.attach_memory(memory)
                budgets.append(budget)
                budget_guards.append(_return_budget_guard(budget))
                for point, state in arm["states"]:
                    boundary()
                    try:
                        result = evaluate_public_cue_recurrent_checkpoint(
                            arm["source"],
                            state,
                            updates=point,
                            sources=sources,
                            budget=budget,
                            check_boundary=boundary,
                            recurring_training=True,
                            recurrent_replay=mode,
                        )
                    except MiniGridEvaluationFailed as error:
                        records[i].append(deepcopy(error.report))
                        frozen_records[i] = pickle.dumps(records[i], protocol=5)
                        raise
                    records[i].append(result)
                    frozen_records[i] = pickle.dumps(records[i], protocol=5)
                    boundary()
                controls[i]["outcomes"] = _recurring_outcomes(records[i])
                outcome_bytes[i] = pickle.dumps(controls[i]["outcomes"], protocol=5)
                budget.complete_memory()
            boundary()
            report.update(status="complete", final_preservation=True)
        except Exception as error:
            report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
            if isinstance(error, TrainingUpdateFailed):
                report["failure_work"] = deepcopy(error.work)
        finally:
            try:
                boundary()
            except Exception as error:
                report.update(
                    status="incomplete",
                    final_preservation=None,
                    readback_error=f"{type(error).__name__}: {error}",
                )
            report["observed_process_rss"] = asdict(memory.snapshot())
            for i, budget in enumerate(budgets):
                controls[i]["readout_budget"] = dict(
                    limits=asdict(budget.budget),
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                    updates_completed=budget.updates_completed,
                    replay_examples_completed=budget.replay_examples_completed,
                )
    report["torch_rng_unchanged"] = bool(torch.equal(torch.get_rng_state(), rng[0]))
    report["numpy_rng_unchanged"] = pickle.dumps(np.random.get_state()) == rng[1]
    if not report["torch_rng_unchanged"] or not report["numpy_rng_unchanged"]:
        report.update(status="incomplete", reason="global RNG changed", final_preservation=None)
    if report["status"] != "complete":
        for control in controls:
            control["outcomes"] = None
    # Export observed detached prefixes; never repair live owners or renew budgets.
    for i, control in enumerate(controls):
        control["training"] = (
            pickle.loads(frozen_rows[i]) if i < len(frozen_rows) else deepcopy(rows[i])
        )
        control["evaluations"] = pickle.loads(frozen_records[i])
    report["total_seconds"] = monotonic() - started
    return report
