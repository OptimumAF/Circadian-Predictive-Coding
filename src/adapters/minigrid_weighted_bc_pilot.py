"""Weighted fixed-A MiniGrid acquisition against an accepted recorded BC control.

Inputs are the frozen local protocol, six original TRAIN demonstrations and the
same128 A packets as the recorded control. Outputs retain original stream receipts,
raw unweighted fit and target-free live readouts. No new optimizer, objective
math, trainer, final/probe data, replay, tuning or selection is introduced.
"""

from copy import deepcopy
from dataclasses import asdict
from math import fsum
import json
import pickle
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.minigrid_learnability import _score_checkpoint
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed, evaluate_checkpoint
from src.adapters.torch_recurrent_learner import (
    TORCH_RECURRENT_CE_ID,
    TorchRecurrentLearner,
    TorchRecurrentPolicy,
)
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.minigrid_key_door import predefined_layouts
from src.app.minigrid_inspector_projection import project_minigrid_report
from src.infra.key_door_inspector_files import (
    MAX_INPUT_BYTES,
    _finite_float,
    _reject_constant,
    _unique_object,
)
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

METHODS = ("recurrent_bc", "recurrent_weighted_bc")
POINTS = (0, 32, 64, 96, 128)
OBJECTIVES = dict(
    recurrent_bc=TORCH_RECURRENT_CE_ID, recurrent_weighted_bc=TORCH_RECURRENT_WEIGHTED_CE_ID
)
BODY_SECONDS = 180.0
RSS_LIMIT = 1024**3
REFERENCE_PATH = "artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json"


def describe_protocol() -> dict[str, Any]:
    """Declare this new control pair without constructing Torch or MiniGrid."""
    return dict(
        protocol_id="minigrid_reference_weighted_a128_acquisition_v1",
        classification="exploratory_recorded_control_distinct_recurrent_bc_objective",
        reference_path=REFERENCE_PATH,
        reference_protocol_id="minigrid_fixed_a128_acquisition_v1",
        initial_condition_check="new weighted0 raw matrices and native observation/action/probability trajectory must match recorded unweighted0, absolute probability tolerance1e-7/relative0; no retry or tolerance relaxation",
        methods=list(METHODS),
        objectives=deepcopy(OBJECTIVES),
        feature_dim=148,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        max_episode_steps=16,
        checkpoints=list(POINTS),
        label_delay=1,
        source_role="train",
        layouts=[asdict(layout) for layout in predefined_layouts()[:3]],
        training="reuse accepted unweighted A128 control, original six TRAIN source records and five readouts; reconstruct SAME128 A packets; train ONLY new independent weighted stream before any new readout; no fresh source collection or baseline training/scoring",
        weighting="current complete TRAIN episode only: n/(present_classes*class_count), absent classes1; native sum(weight*CE)/sum(observed target weights); no weight tuning",
        scoring="original raw unweighted CE/accuracy/exact sequence at all5points; native target-free first-argmax all7 A live at0/128; fresh RNN session per cue",
        acquisition="endpoint BOTH raw cue sequences exact AND BOTH native goals; each flag retained separately; no retention/recovery inference without acquisition",
        comparison="recurrent_weighted_bc-minus-recurrent_bc changes in CE0-CE128, endpoint-minus-initial success count and mean return; positive gain desirable; all fixed points/cues retained",
        planned_work=dict(
            native_training_updates=128,
            training_rows=1152,
            raw_score_forwards=10,
            raw_score_rows=90,
            live_episodes=4,
            maximum_live_predictions=64,
            maximum_live_physics=64,
            teacher_episodes=0,
            expected_teacher_physics=0,
            maximum_teacher_physics=0,
            maximum_environment_resets=4,
            maximum_total_physics=64,
            maximum_recurrent_unroll_steps=1306,
            probe_updates=0,
            replay_updates=0,
            final_episodes=0,
            reward_updates=0,
        ),
        retained_historical_exposure=dict(
            unweighted_updates=128,
            unweighted_training_rows=1152,
            raw_score_forwards=10,
            raw_score_rows=90,
            live_episodes=4,
            teacher_episodes=6,
            teacher_physics=54,
            total_control_pair_updates=256,
            total_control_pair_training_rows=2304,
        ),
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=210,
            process_rss_bytes=RSS_LIMIT,
            inherited_per_arm_rss_bytes=shared.RSS_LIMIT,
            output_bytes=16 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        limitations="one seed/fixed familiar TRAIN task; weighting is a separate objective/control, not an improved historical baseline; equal update/row exposure is not equal computation, wrapper validation/copies/RSS counted; no CPC, retention/recovery/generalization or benefit guarantee; preserve original negative acquisition",
    )


def _new_report() -> dict[str, Any]:
    methods = [
        dict(shared._row(method), objective_definition=OBJECTIVES[method]) for method in METHODS
    ]
    for row in methods:
        row["final_evaluation"]["reason"] = "not requested by TRAIN-only paired protocol"
    return dict(
        protocol=describe_protocol(),
        status="incomplete",
        reason=None,
        methods=methods,
        source_reports=[],
        training_sources=[],
        evaluations=[
            dict(method=method, updates=point, role="train", status="not_run")
            for method in METHODS
            for point in POINTS
        ],
        acquisition=None,
        work=None,
        new_work=None,
        historical_work=None,
        reference=None,
        retained_input_array_bytes=None,
        objective_evidence=None,
        observed_process_rss=None,
        seconds=None,
        readback_error=None,
        scientific_advantage_verified=False,
    )


def _load_reference() -> dict[str, Any]:
    """Read the one fixed bounded unique/finite JSON artifact, once per invocation."""
    path = Path(__file__).resolve().parents[2] / REFERENCE_PATH
    with path.open("rb") as handle:
        payload = handle.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("fixed reference exceeds8MiB")
    report = json.loads(
        payload.decode("utf8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    if type(report) is not dict:
        raise ValueError("fixed reference must be a JSON object")
    return report


def _admit_reference(reference: dict[str, Any]) -> dict[str, Any]:
    """Reuse the existing recorded-law validator, then bind the selected control."""
    # Why: this already accepted pure projector checks saved raw probabilities,
    # action/observation chains, native end flags, counters and displayed metrics.
    # It runs once on admitted input, never on every training boundary.
    project_minigrid_report(reference, source_label=REFERENCE_PATH)
    protocol = reference["protocol"]
    for key, expected in dict(
        protocol_id="minigrid_fixed_a128_acquisition_v1",
        seed=17,
        hidden_dim=8,
        num_classes=7,
        learning_rate=0.03,
        momentum=0.9,
        label_delay=1,
        role="train",
    ).items():
        if type(protocol.get(key)) is not type(expected) or protocol[key] != expected:
            raise ValueError(f"fixed reference protocol {key} changed")
    if (
        reference["status"] != "complete"
        or protocol["checkpoints"] != list(POINTS)
        or protocol["feature_dim_by_method"].get("recurrent_bc") != 148
    ):
        raise ValueError("fixed reference must be the completed original148/A128 case")
    selected = [
        (index, row)
        for index, row in enumerate(reference["methods"])
        if row["method"] == "recurrent_bc"
    ]
    if len(selected) != 1:
        raise ValueError("fixed reference requires exactly one original recurrent BC control")
    row_index, original_row = selected[0]
    if (
        original_row["status"] not in ("trained", "complete")
        or type(original_row["charged_updates"]) is not int
        or original_row["charged_updates"] != 128
        or type(original_row["native_training_examples"]) is not int
        or original_row["native_training_examples"] != 1152
        or original_row["source_unchanged"] is not True
        or original_row["actor_unchanged"] is not True
        or original_row["checkpoint_updates"] != list(POINTS)
    ):
        raise ValueError("recorded recurrent source/actor/counters differ from accepted control")
    row = dict(deepcopy(original_row), objective_definition=TORCH_RECURRENT_CE_ID)
    _check_receipts(dict(row=row))
    result = row["stream_result"]
    if (
        result["evaluation_scope"] != "training_only"
        or result["score_forward_calls"] != 0
        or result["checkpoints"] != []
        or result["num_classes"] != 7
    ):
        raise ValueError("reference training stream contains an undeclared evaluation")
    for index, receipt in enumerate(result["applied_updates"]):
        if (
            receipt["sample_id"] != f"batch-{index}"
            or receipt["episode_id"] != f"minigrid-fixed-a-training-{index}"
            or receipt["event_id"] != f"label-{index}"
            or any(
                type(receipt[key]) is not int or receipt[key] != value
                for key, value in (
                    ("observed_at", index),
                    ("arrived_at", index + 1),
                    ("applied_at", index + 1),
                    ("update_number", index + 1),
                )
            )
            or receipt["actor_version"] != "a0"
            or receipt["learner_version"] != "c0"
        ):
            raise ValueError(
                "reference receipt release identity/clock differs from original stream"
            )
    sources = []
    if len(reference["training_sources"]) != 6 or len(reference["source_reports"]) != 6:
        raise ValueError("reference requires exactly six original TRAIN source records")
    for index, raw in enumerate(reference["training_sources"]):
        transitions = tuple(
            DemonstrationTransition(
                **dict(
                    t,
                    observation=tuple(t["observation"]),
                    next_observation=tuple(t["next_observation"]),
                )
            )
            for t in raw["transitions"]
        )
        episode = DemonstrationEpisode(**dict(raw, transitions=transitions))
        expected_context = ("A", "B", "C_train")[index // 2]
        cue = ("blue", "red")[index % 2]
        if (
            episode.role != "train"
            or episode.context != expected_context
            or episode.num_classes != 7
            or episode.episode_id != f"minigrid-source-{expected_context}-{cue}"
            or len(transitions) != 9
            or any(len(t.observation) != 148 for t in transitions)
        ):
            raise ValueError("reference source DTO identity/role/geometry differs from original")
        physical = reference["source_reports"][index]
        if (
            physical.get("role") != "train"
            or physical.get("context") != expected_context
            or physical.get("cue") != cue
            or physical.get("closed") is not True
            or physical.get("status") != "complete"
            or physical.get("environment_steps") != 9
            or physical.get("reset_calls") != 1
            or physical.get("seed") != 17
            or physical.get("max_steps") != 16
            or physical.get("observation_dim") != 148
            or physical.get("num_classes") != 7
            or physical.get("layout") != asdict(predefined_layouts()[index // 2])
        ):
            raise ValueError("reference physical TRAIN source assumptions changed")
        native_rows = physical.get("transitions")
        if type(native_rows) is not list or len(native_rows) != 9:
            raise ValueError("reference physical TRAIN source requires nine actual transitions")
        for transition, native in zip(transitions, native_rows, strict=True):
            if (
                native.get("applied") is not True
                or tuple(native["observation"]) != transition.observation
                or tuple(native["next_observation"]) != transition.next_observation
                or native.get("action") != transition.action
                or native.get("reward") != transition.reward
                or native.get("terminated") is not transition.terminal
                or native.get("truncated") is not False
            ):
                raise ValueError("reference TRAIN DTO differs from its recorded physics")
        sources.append(shared._batch(episode))
    shared._packets(tuple(sources))  # Existing complete source/array/role admission.
    selected_evaluations = [
        (index, r)
        for index, r in enumerate(reference["evaluations"])
        if r["method"] == "recurrent_bc"
    ]
    records = [deepcopy(record) for _, record in selected_evaluations]
    if tuple(r["updates"] for r in records) != POINTS or any(
        r["status"] != "complete" or r["role"] != "train" for r in records
    ):
        raise ValueError("reference requires all five original TRAIN raw readouts")
    return dict(
        row=row,
        evaluations=records,
        sources=tuple(sources),
        provenance=dict(
            reference_path=REFERENCE_PATH,
            protocol_id=protocol["protocol_id"],
            row_pointer=f"/methods/{row_index}",
            readout_pointers=[f"/evaluations/{index}" for index, _ in selected_evaluations],
            historical_control=deepcopy(original_row),
            source_reports=deepcopy(reference["source_reports"]),
            training_sources=deepcopy(reference["training_sources"]),
            reference_protocol=deepcopy(protocol),
            scope="accepted recorded unweighted control/source/readouts reused; no baseline training/scoring/physics or old budget renewal",
        ),
    )


def _learner_factory(packets):
    episodes = tuple(shared._episode(packet) for packet in packets)

    def factory(method: str) -> Any:
        policy = TorchRecurrentPolicy(148, 8, 7, 17)
        if method == "recurrent_bc":
            return TorchRecurrentLearner(policy)
        if method == "recurrent_weighted_bc":
            return TorchTrainWeightedLearner(policy, episodes)
        raise ValueError("paired acquisition requires a declared recurrent control")

    return factory


def _check_receipts(arm: dict[str, Any]) -> None:
    row = arm["row"]
    result = row["stream_result"]
    receipts = result["applied_updates"]
    if (
        result["completed_updates"] != 128
        or result["training_examples"] != 1152
        or len(receipts) != 128
        or any(
            receipt["update_number"] != i + 1
            or receipt["diagnostic"]["definition"] != OBJECTIVES[row["method"]]
            for i, receipt in enumerate(receipts)
        )
    ):
        raise ValueError("paired control receipts differ from fixed dose or native objective")


def _outcomes(evaluations: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    outcomes = []
    for method in METHODS:
        records = [record for record in evaluations if record["method"] == method]
        available = (
            complete and len(records) == 5 and all(r["status"] == "complete" for r in records)
        )
        value: dict[str, Any] = dict(
            method=method,
            status="complete" if available else "unavailable",
            raw_both_cues_exact=None,
            live_both_cues_goals=None,
            joint_acquisition=None,
            ce_gain=None,
            live_success_count_change=None,
            mean_return_gain=None,
        )
        if available:
            if tuple(r["updates"] for r in records) != POINTS or any(
                [(s["context"], s["cue"]) for s in r["scores"]] != [("A", "blue"), ("A", "red")]
                for r in records
            ):
                raise ValueError("paired raw readout identities differ from frozen A cue order")
            initial, final = records[0], records[-1]
            for endpoint in (initial, final):
                if [(e["context"], e["cue"]) for e in endpoint["episodes"]] != [
                    ("A", "blue"),
                    ("A", "red"),
                ] or any(e["status"] != "complete" for e in endpoint["episodes"]):
                    raise ValueError("paired endpoint native evidence is incomplete")
            raw = all(score["exact_sequence_accuracy"] == 1 for score in final["scores"])
            live = all(episode["success"] is True for episode in final["episodes"])
            value.update(
                raw_both_cues_exact=raw,
                live_both_cues_goals=live,
                joint_acquisition=raw and live,
                ce_gain=fsum(s["ce"] for s in initial["scores"]) / 2
                - fsum(s["ce"] for s in final["scores"]) / 2,
                live_success_count_change=sum(e["success"] for e in final["episodes"])
                - sum(e["success"] for e in initial["episodes"]),
                mean_return_gain=fsum(e["return"] for e in final["episodes"]) / 2
                - fsum(e["return"] for e in initial["episodes"]) / 2,
            )
        outcomes.append(value)
    return dict(
        methods=outcomes,
        signed_comparisons={
            "recurrent_weighted_bc-minus-recurrent_bc": {
                key: outcomes[1][key] - outcomes[0][key] if complete else None
                for key in ("ce_gain", "live_success_count_change", "mean_return_gain")
            }
        },
    )


def _work(methods, evaluations, source_reports, source_unknown: bool) -> dict[str, Any]:
    raw_attempted = raw_returned = raw_rows_attempted = raw_rows_returned = 0
    live_attempted = live_returned = physics = resets = sessions = 0
    unknown = source_unknown
    for record in evaluations:
        work = record.get("work", {})
        raw_attempted += work.get(
            "raw_score_forwards_attempted", work.get("score_forward_calls_attempted", 0)
        )
        raw_returned += work.get(
            "raw_score_forwards_completed", work.get("score_forward_calls_completed", 0)
        )
        raw_rows_attempted += work.get(
            "raw_score_rows_attempted", work.get("score_rows_attempted", 0)
        )
        raw_rows_returned += work.get(
            "raw_score_rows_completed", work.get("score_rows_completed", 0)
        )
        live_attempted += work.get("live_prediction_calls_attempted", 0)
        live_returned += work.get("live_prediction_calls_returned", 0)
        physics += work.get("environment_steps_returned", 0)
        resets += work.get("environment_resets_completed", 0)
        sessions += work.get("live_sessions_started", 0)
        unknown |= work.get("failure_work_may_be_unobserved", False)
    updates = sum(row.get("charged_updates") or 0 for row in methods)
    rows = sum(row.get("native_training_examples") or 0 for row in methods)
    return dict(
        charged_updates=updates,
        observed_native_training_rows=rows,
        wake_calls_attempted=sum(row["wake_calls_attempted"] for row in methods),
        wake_calls_returned=sum(row["wake_calls_returned"] for row in methods),
        raw_score_forwards_attempted=raw_attempted,
        raw_score_forwards_completed=raw_returned,
        raw_score_rows_attempted=raw_rows_attempted,
        raw_score_rows_completed=raw_rows_returned,
        live_prediction_calls_attempted=live_attempted,
        live_prediction_calls_returned=live_returned,
        live_sessions_started=sessions,
        live_environment_resets_completed=resets,
        live_environment_steps_returned=physics,
        teacher_reports=len(source_reports),
        teacher_environment_steps_observed=sum(
            r.get("environment_steps", 0) or 0 for r in source_reports
        ),
        observed_recurrent_unroll_steps=rows + raw_rows_returned + live_returned,
        failure_work_may_be_unobserved=unknown
        or any(row["status"] == "incomplete" for row in methods),
        accounting="observed rows/returns only; failed native or constructor work may be unobserved; source/validation/ownership/copy overhead included in elapsed and process RSS",
    )


def _check_initial_condition(recorded, current) -> None:
    """Bind the new random initial policy's observable law to the saved control."""

    def probabilities(left, right) -> None:
        a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-7):
            raise ValueError("recorded/new initial probability law differs")

    for before, after in zip(recorded["scores"], current["scores"], strict=True):
        if (before["context"], before["cue"]) != (after["context"], after["cue"]):
            raise ValueError("recorded/new initial raw identity differs")
        probabilities(before["probabilities"], after["probabilities"])
    for before, after in zip(recorded["episodes"], current["episodes"], strict=True):
        if (before["context"], before["cue"], before["success"], before["return"]) != (
            after["context"],
            after["cue"],
            after["success"],
            after["return"],
        ):
            raise ValueError("recorded/new initial native outcome differs")
        for a, b in zip(before["predictions"], after["predictions"], strict=True):
            if a["observation"] != b["observation"] or a["action"] != b["action"]:
                # JSON lists and new native tuples carry the same148 coordinates.
                if tuple(a["observation"]) != tuple(b["observation"]) or a["action"] != b["action"]:
                    raise ValueError("recorded/new initial native decision chain differs")
            probabilities(a["probabilities"], b["probabilities"])
        for a, b in zip(
            before["environment"]["transitions"], after["environment"]["transitions"], strict=True
        ):
            if any(
                a[key] != b[key] for key in ("action", "reward", "terminated", "truncated")
            ) or any(tuple(a[key]) != tuple(b[key]) for key in ("observation", "next_observation")):
                raise ValueError("recorded/new initial native physics chain differs")


def run_pilot() -> dict[str, Any]:
    """Run both fixed streams before their bounded TRAIN-only readouts."""
    # Import the existing packet builder only in explicit execution: its legacy
    # script module also sets BLAS environment defaults at import time.
    from scripts.run_minigrid_history_evaluation_smoke import _fixed_a_packets

    report = _new_report()
    started = monotonic()
    rows, evaluations, source_reports = (
        report["methods"],
        report["evaluations"],
        report["source_reports"],
    )
    row_owners = tuple(rows)
    evaluation_owners = list(evaluations)
    evaluation_payloads = [pickle.dumps(r, protocol=5) for r in evaluations]
    source_captures: list[tuple[Any, bytes]] = []
    original_protocol = report["protocol"]
    protocol_bytes = pickle.dumps(original_protocol, protocol=5)
    originals: dict[str, tuple[Any, bytes]] = {}
    signatures: list[tuple[Any, Any]] = []
    arms: list[dict[str, Any]] = []
    paid_rows: dict[int, bytes] = {}
    failed_rows: dict[int, bytes] = {}
    initial_rows = tuple(pickle.dumps(row, protocol=5) for row in rows)
    active_training = None
    memory: Any = None
    primary_error: str | None = None
    reference_admitted = False
    initial_condition_match = None

    def common() -> None:
        if monotonic() - started >= BODY_SECONDS:
            raise ValueError("paired whole-case wall-time limit exhausted")
        if memory is not None:
            memory.sample()
            if memory.snapshot().peak_bytes > RSS_LIMIT:
                raise ValueError("paired whole-case process RSS limit exhausted")
        if (
            report["protocol"] is not original_protocol
            or pickle.dumps(original_protocol, protocol=5) != protocol_bytes
            or report["methods"] is not rows
            or tuple(id(r) for r in rows) != tuple(id(r) for r in row_owners)
            or tuple(r["method"] for r in rows) != METHODS
            or any(r["objective_definition"] != OBJECTIVES[r["method"]] for r in rows)
            or report["evaluations"] is not evaluations
            or len(evaluations) != 10
            or any(
                r is not owner or pickle.dumps(r, protocol=5) != payload
                for r, owner, payload in zip(
                    evaluations, evaluation_owners, evaluation_payloads, strict=True
                )
            )
            or report["source_reports"] is not source_reports
            or len(source_reports) > 6
            or len(source_reports) < len(source_captures)
        ):
            raise ValueError("paired protocol, control slots or preceding readout evidence changed")
        for index, (owner, payload) in enumerate(source_captures):
            if source_reports[index] is not owner or pickle.dumps(owner, protocol=5) != payload:
                raise ValueError("previous physical TRAIN source evidence changed")
        while len(source_captures) < len(source_reports):
            owner = source_reports[len(source_captures)]
            source_captures.append((owner, pickle.dumps(owner, protocol=5)))
        if any(
            not shared._check_signature(batches, signature) for batches, signature in signatures
        ):
            raise ValueError("original paired TRAIN arrays/episodes changed")
        if any(
            report[key] is not owner or pickle.dumps(owner, protocol=5) != payload
            for key, (owner, payload) in originals.items()
        ):
            raise ValueError("original paired source/objective/cost metadata changed")
        for index, payload in paid_rows.items():
            if pickle.dumps(rows[index], protocol=5) != payload:
                raise ValueError("previous complete training report changed")
        for index, payload in enumerate(initial_rows):
            if (
                index != active_training
                and index not in paid_rows
                and index not in failed_rows
                and pickle.dumps(rows[index], protocol=5) != payload
            ):
                raise ValueError("unreached paired training report changed")

    def completed() -> None:
        common()
        for arm in arms:
            arm["guard"]()
            _check_receipts(arm)
        common()

    def bind(key, value) -> None:
        report[key] = value
        originals[key] = (value, pickle.dumps(value, protocol=5))

    try:
        with ProcessRssSampler() as memory:
            common()
            admitted = _admit_reference(_load_reference())
            sources = admitted["sources"]
            provenance = admitted["provenance"]
            rows[0].clear()
            rows[0].update(admitted["row"])
            paid_rows[0] = pickle.dumps(rows[0], protocol=5)
            for index, record in enumerate(admitted["evaluations"]):
                evaluations[index] = record
                evaluation_owners[index] = record
                evaluation_payloads[index] = pickle.dumps(record, protocol=5)
            source_reports.extend(deepcopy(provenance["source_reports"]))
            bind("reference", provenance)
            reference_admitted = True
            common()
            require_torch()
            packets = _fixed_a_packets(sources)
            raw_sources = sources[:2]
            signatures.extend(
                (batches, shared._signature(batches)) for batches in (sources, packets)
            )
            bind("training_sources", deepcopy(provenance["training_sources"]))
            bind(
                "retained_input_array_bytes",
                dict(
                    original_sources=sum(p.features.nbytes + p.targets.nbytes for p in sources),
                    shared_training=sum(p.features.nbytes + p.targets.nbytes for p in packets),
                ),
            )
            bind(
                "objective_evidence",
                dict(
                    source_role="train",
                    derivation="current complete episode only",
                    cue_weights=[
                        dict(
                            cue=shared._episode(p).episode_id.rsplit("-", 1)[-1],
                            weights=list(episode_class_weights(shared._episode(p))),
                        )
                        for p in raw_sources
                    ],
                    weighted_definition=TORCH_RECURRENT_WEIGHTED_CE_ID,
                    readout_definition="raw unweighted categorical CE/accuracy/sequences and native all7firstargmax",
                ),
            )
            factory = _learner_factory(packets)
            for index, row in ((1, rows[1]),):
                completed()
                active_training = index
                # Each peer guard inspects only its own arm; avoid recursive
                # callback graphs when the next stream invokes the shared runner.
                prior = tuple(arm["guard"] for arm in arms)

                def before(prior=prior):
                    common()
                    for check in prior:
                        check()
                    common()

                arm = shared._train_arm(
                    row,
                    packets,
                    memory,
                    started + BODY_SECONDS,
                    before,
                    learner_factory=factory,
                    checkpoint_updates=POINTS,
                )
                arms.append(arm)
                arm["guard"]()
                _check_receipts(arm)
                paid_rows[index] = pickle.dumps(row, protocol=5)
                active_training = None
            completed()
            for arm in arms:
                arm_index = 1
                for point_index, (point, state) in enumerate(arm["states"]):
                    completed()
                    method = METHODS[arm_index]
                    try:
                        if point in (0, 128):
                            result = evaluate_checkpoint(
                                arm["source"],
                                state,
                                method=method,
                                updates=point,
                                layouts=predefined_layouts()[:1],
                                role="train",
                                sources=raw_sources,
                                budget=arm["budget"],
                                check_boundary=completed,
                            )
                        else:
                            result = _score_checkpoint(
                                arm["source"],
                                state,
                                method=method,
                                updates=point,
                                sources=raw_sources,
                                budget=arm["budget"],
                                check_boundary=completed,
                            )
                    except MiniGridEvaluationFailed as error:
                        result = error.report
                    slot = arm_index * 5 + point_index
                    evaluations[slot] = result
                    evaluation_owners[slot] = result
                    evaluation_payloads[slot] = pickle.dumps(result, protocol=5)
                    if result["status"] != "complete":
                        raise ValueError("paired readout failed; original paid prefix retained")
                    if point == 0:
                        _check_initial_condition(evaluations[0], result)
                        initial_condition_match = True
                    completed()
            completed()
    except Exception as error:
        primary_error = f"{type(error).__name__}: {error}"
        if "initial" in str(error) and initial_condition_match is None:
            initial_condition_match = False
        if active_training is not None and active_training not in paid_rows:
            failed_rows[active_training] = pickle.dumps(row_owners[active_training], protocol=5)
        # The shared collector appends its last paid report during cleanup. Keep
        # that observed tail on failure even when resources forbid another guard.
        while len(source_captures) < min(len(source_reports), 6):
            owner = source_reports[len(source_captures)]
            source_captures.append((owner, pickle.dumps(owner, protocol=5)))
    # Final external readbacks precede the last whole-record guard. Private
    # captures are published only afterwards, even if continuation was refused.
    try:
        if memory is not None and memory.start_bytes is not None:
            report["observed_process_rss"] = asdict(memory.snapshot())
        elapsed = monotonic() - started
        completed()
    except Exception as error:
        report["readback_error"] = f"{type(error).__name__}: {error}"
        primary_error = primary_error or report["readback_error"]
        elapsed = monotonic() - started
    report["protocol"] = pickle.loads(protocol_bytes)
    report["methods"] = [
        pickle.loads(paid_rows.get(i, failed_rows.get(i, initial_rows[i]))) for i in range(2)
    ]
    report["evaluations"] = [pickle.loads(payload) for payload in evaluation_payloads]
    report["source_reports"] = [pickle.loads(payload) for _, payload in source_captures]
    for key, (_, payload) in originals.items():
        report[key] = pickle.loads(payload)
    complete = (
        primary_error is None
        and reference_admitted
        and len(arms) == 1
        and initial_condition_match is True
        and all(r["status"] == "complete" for r in report["evaluations"])
    )
    report.update(status="complete" if complete else "incomplete", reason=primary_error)
    report["acquisition"] = _outcomes(report["evaluations"], complete)
    report["work"] = _work(
        report["methods"],
        report["evaluations"],
        report["source_reports"],
        len(source_captures) != 6,
    )
    report["new_work"] = _work(report["methods"][1:], report["evaluations"][5:], [], False)
    report["historical_work"] = _work(
        report["methods"][:1],
        report["evaluations"][:5],
        report["source_reports"],
        not reference_admitted,
    )
    report["cost_scope"] = (
        "work combines accepted historical exposure plus new weighted work; new_work alone is charged to this invocation; historical teacher/SGD/readouts are not rerun or charged to a renewed budget; old walltime/RSS/native weights are not matched"
    )
    report["initial_condition_match"] = initial_condition_match
    report["seconds"] = elapsed
    if complete and report["seconds"] >= BODY_SECONDS:
        report.update(status="incomplete", reason="paired whole-case wall-time limit exhausted")
        report["acquisition"] = _outcomes(report["evaluations"], False)
    return report
