"""Opt-in joint TRAIN exposure through existing MiniGrid learner/runtime ports.

Owns option/protocol composition and batch preparation, not model objectives,
collection, budgets, probe updates, evaluation, promotion, IO or a new runner.
"""

from dataclasses import asdict
from typing import Any

from src.adapters.minigrid_train_catalogue import CatalogueTrainingPlan, compose_catalogue_training
from src.adapters import minigrid_relaxation_control as relaxation
from src.app.coupled_attribution import AttributionBatch, _validate_batch


def validate_option(flag: bool, **options: Any) -> None:
    if type(flag) is not bool:
        raise ValueError("joint_catalogue must be an exact bool")
    if not flag:
        return
    if options.get("weighted_pc", False) is not False:
        raise ValueError("joint_catalogue does not admit weighted_pc episode semantics")
    if (
        any(
            options.get(name) is not True
            for name in (
                "public_history",
                "invariant_history",
                "distinct_probe",
                "old_task_readouts",
                "return_to_a",
            )
        )
        or type(options.get("qualification_method")) is not str
        or options["qualification_method"]
        not in ("bp", "pc", "neutral_cpc", "gated_cpc", *relaxation.METHODS)
        or any(
            options.get(name) is not False
            for name in (
                "pc_qualification",
                "acquisition_only",
                "weighted_bp",
                "memory_control",
                "activation_health",
                "gradient_geometry",
                "probe_rollouts",
                "deferred_probe_rollouts",
            )
        )
    ):
        raise ValueError(
            "joint_catalogue requires both history modes, one explicit neural method, "
            "distinct_probe/old_task_readouts/return_to_a, without other training/rollout diagnostics"
        )
    # Why: prior CE diagnostics already accept the full H32 state and six TRAIN
    # episodes. Admit that explicit composition without broadening older recipes.
    prior = options.get("prior_gradient_geometry", False)
    if type(prior) is not bool or (
        prior
        and (
            options.get("wide_neural") is not True
            or options["qualification_method"] not in ("bp", "pc", "neutral_cpc", "gated_cpc")
            or options.get("pc_update_diagnostics", False) is not False
        )
    ):
        raise ValueError("joint prior gradients require the explicit original H32 recipe alone")
    if type(options.get("train_a_rollout", False)) is not bool or (
        options.get("train_a_rollout") is True
        and any(
            options.get(name, False) is not False
            for name in ("train_action_margins", "pc_update_diagnostics")
        )
    ):
        raise ValueError("joint TRAIN A games require an exact bool without extra diagnostics")


def describe(protocol: dict[str, Any]) -> None:
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_joint_catalogue_v1"
    protocol.update(
        classification="exploratory_joint_train_catalogue_acquisition_and_isolated_return",
        task_sequence=["joint_train_catalogue"] * 4,
        evaluation="one joint54-row stream completes128 updates before reserved F collection/readouts; all six TRAIN traces at0/32/64/96/128; isolated F16/old-A/A16 from0/128; no final D",
        training="128 delayed1 independent-row wakes; each sees all six original TRAIN sources in fixed order,54rows; native objective unchanged",
        acquisition_premise="all six TRAIN traces exact at128; no sequential A/B/C forgetting interpretation",
        metric_direction="original CE lower; accuracy/exact trace higher; F gain positive and old-A loss lower desirable; no live success",
        joint_catalogue=dict(
            source_order=[
                f"{context}-{cue}" for context in ("A", "B", "C_train") for cue in ("blue", "red")
            ],
            rows_per_update=54,
            updates=128,
            training_row_exposures=6912,
            input_array_bytes=54 * 297 * 8 * 128,
            source_template_array_bytes=54 * 297 * 8,
            validation_projection_rows=108,
            validation_projected_array_equivalent_bytes=108 * 296 * 8,
            comparison="joint54-row exposure differs from online9-row wakes; not equal exposure/compute or a replacement baseline",
            isolation="existing training_only stream; source and actor unchanged; F16 and A16 only on independently budgeted disposable forks",
        ),
    )
    protocol["public_history"].update(
        retained_benchmark_array_bytes=54 * 149 * 8 + 54 * 297 * 8 + 54 * 297 * 8 * 128,
        computation="54rows per joint native wake; original native method/seed/recipe, explicitly different exposure and compute",
    )
    protocol["planned_work"]["training_rows"] += (54 - 9) * 128
    protocol["planned_work"].update(
        benchmark_training_rows=6912,
        joint_benchmark_training_rows=6912,
        catalogue_validation_projection_rows=108,
        catalogue_input_array_bytes=54 * 297 * 8 * 128,
    )
    protocol.pop("differences", None)
    protocol["probe"].update(
        source_stages=["initial", "post_joint128"],
        comparison="fixed F CE0-CE16 at initial and joint128 checkpoints; additional A16 return and old-A fit are isolated; no advantage assumed",
        support_gate="compare all six original TRAIN sources to all18 F packets before probe updates; every F episode must have9of9 canonical TRAIN-unseen rows; every joint wake is the guarded concatenation of those same six sources",
    )
    # Why: repeated wakes change exposure, but not the input support set. The
    # existing support reducer receives complete source episodes, not a false
    # trajectory fabricated from a joint independent-row transport batch.
    protocol["probe_support"].update(
        training_packets=6,
        training_rows=54,
        history_rows=216,
        canonical_projection_rows=216,
        canonical_projected_array_equivalent_bytes=216 * 2368,
        training_scope="all six original complete TRAIN sources, analyzed once",
        exposure_binding="each of128 guarded joint wakes contains every original54 source row in order; same support set,6912 paid exposures",
    )
    protocol["limitations"] += (
        "; pooled TRAIN catalogue is a distinct acquisition control, not online adaptation, matched-compute evidence or final generalization"
    )


def compose(
    packets: tuple[AttributionBatch, ...], *, cue_relative: bool = False
) -> CatalogueTrainingPlan:
    if type(cue_relative) is not bool:
        raise ValueError("cue_relative must be an exact bool")
    if type(packets) is not tuple or len(packets) != 6:
        raise ValueError("joint catalogue requires all six original TRAIN cue sources")
    seen: set[str] = set()
    episodes = []
    for index, packet in enumerate(packets):
        _validate_batch(packet, "train", seen)
        episode = packet.episode
        if (
            episode is None
            or packet.features.shape != (9, 296)
            or packet.num_classes != 7
            or episode.context != ("A", "B", "C_train")[index // 2]
            or episode.episode_id
            != f"minigrid-source-{episode.context}-{('blue', 'red')[index % 2]}"
        ):
            raise ValueError("joint catalogue requires original six TRAIN296/K7 nine-row cue order")
        episodes.append(episode)
    # Admission projects both causal views, but transport remains raw: the
    # cue-relative learner owns the single non-idempotent native projection.
    return (
        compose_catalogue_training(tuple(episodes), 128, cue_relative=True)
        if cue_relative
        else compose_catalogue_training(tuple(episodes), 128)
    )


def accounting(plan: CatalogueTrainingPlan) -> dict[str, Any]:
    # Why: publish bounded scalar/span evidence, never transport arrays a second time.
    return dict(
        source_spans=[asdict(span) for span in plan.source_spans],
        source_rows=plan.source_rows,
        row_exposures=plan.row_exposures,
        input_array_bytes=plan.input_array_bytes,
        source_template_array_bytes=plan.source_template_array_bytes,
        validation_projection_rows=plan.validation_projection_rows,
        validation_projected_array_equivalent_bytes=plan.validation_projected_array_equivalent_bytes,
    )
