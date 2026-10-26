"""Reconstruct the accepted unweighted MiniGrid prefix for reward continuation.

The JSON reference lacks native parameters, gradients and optimizer momentum,
so reproducing its fixed128 TRAIN updates is newly charged integration work.
All five raw readouts and both endpoint native cue pairs must match exactly.
No teacher, dose extension, model selection, weighted objective or reward SGD
belongs here; the existing stream and native objectives remain authoritative.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
import json
import pickle
from time import monotonic
from typing import Any

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.minigrid_learnability import _score_checkpoint
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed, evaluate_checkpoint
from src.adapters.minigrid_weighted_bc_pilot import _admit_reference, _load_reference, _work
from src.adapters.torch_demonstration_reward_learner import TorchDemonstrationRewardLearner
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID, TorchRecurrentPolicy
from src.core.minigrid_key_door import predefined_layouts

POINTS = (0, 32, 64, 96, 128)


def _canonical(value: Any) -> str:
    # Native reports contain only JSON values; tuples/lists encode the same
    # coordinates. No float tolerance, rounding or outcome-based selection.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parity(recorded: dict[str, Any], current: dict[str, Any]) -> None:
    if current["status"] != "complete" or _canonical(recorded["scores"]) != _canonical(
        current["scores"]
    ):
        raise ValueError(
            "warm-prefix raw TRAIN scores/probabilities/actions differ from accepted reference"
        )
    if current["updates"] in (0, 128):
        before, after = recorded["episodes"], current["episodes"]
        if len(before) != 2 or len(after) != 2:
            raise ValueError("warm-prefix parity requires both native cue episodes")
        for first, second in zip(before, after, strict=True):
            keys = ("context", "cue", "source_role", "status", "predictions", "success", "return")
            if _canonical({key: first[key] for key in keys}) != _canonical(
                {key: second[key] for key in keys}
            ):
                raise ValueError(
                    "warm-prefix native decision/outcome law differs from accepted reference"
                )
            # Episode IDs identify this reconstruction, not a policy input. All
            # actual physics, public observations, flags and costs remain exact.
            if _canonical(
                {k: v for k, v in first["environment"].items() if k != "episode_id"}
            ) != _canonical({k: v for k, v in second["environment"].items() if k != "episode_id"}):
                raise ValueError("warm-prefix native physics differs from accepted reference")


def prepare_warm_start(
    report: dict[str, Any], memory: Any, deadline: float, check_boundary: Callable[[], None]
) -> dict[str, Any]:
    """Populate owned partial evidence and return a fully restored frozen source.

    The supplied report remains the same container. Failure updates its observed
    prefix before raising; callers must not admit reward work after a refusal.
    """
    if type(report) is not dict or report or not callable(check_boundary):
        raise ValueError("warm start requires an empty report and callable boundary guard")
    if type(deadline) not in (int, float) or not 0 <= deadline < float("inf"):
        raise ValueError("warm start requires a finite nonnegative original deadline")
    from scripts.run_minigrid_history_evaluation_smoke import _fixed_a_packets

    row = dict(shared._row("recurrent_bc"), objective_definition=TORCH_RECURRENT_CE_ID)
    evaluations = [
        dict(method="recurrent_bc", updates=point, role="train", status="not_run")
        for point in POINTS
    ]
    parity_checks: list[bool | None] = [None] * 5
    report.update(
        status="running",
        reason=None,
        training=row,
        evaluations=evaluations,
        reference=None,
        historical_work=None,
        work=None,
        parity_checks=parity_checks,
        parity_matched=None,
        full_state_restored=False,
        retained_input_array_bytes=None,
        observed_process_rss=None,
        readback_error=None,
    )
    owners = list(evaluations)
    payloads = [pickle.dumps(value, protocol=5) for value in evaluations]
    expected_parity = list(parity_checks)
    metadata: dict[str, tuple[Any, bytes]] = {}
    signatures: list[tuple[Any, Any]] = []
    paid_training: bytes | None = None
    completed_report: bytes | None = None
    arm: Any = None
    source: Any = None
    expected_source: Any = None
    primary_error: str | None = None

    def common() -> None:
        check_boundary()
        if monotonic() >= deadline:
            raise ValueError("original warm-start whole-case deadline exhausted")
        if (
            report.get("training") is not row
            or report.get("evaluations") is not evaluations
            or report.get("parity_checks") is not parity_checks
            or parity_checks != expected_parity
            or len(evaluations) != 5
            or any(
                value is not owner or pickle.dumps(value, protocol=5) != payload
                for value, owner, payload in zip(evaluations, owners, payloads, strict=True)
            )
            or any(
                report[key] is not owner or pickle.dumps(owner, protocol=5) != payload
                for key, (owner, payload) in metadata.items()
            )
            or any(
                not shared._check_signature(values, signature) for values, signature in signatures
            )
            or (paid_training is not None and pickle.dumps(row, protocol=5) != paid_training)
            or (
                completed_report is not None
                and pickle.dumps(report, protocol=5) != completed_report
            )
        ):
            raise ValueError("original warm-prefix inputs or preceding evidence changed")

    def guard() -> None:
        common()
        if arm is not None:
            arm["guard"]()
        if source is not None and not _same_state(expected_source, source.snapshot_state()):
            raise ValueError("restored warm source full learned/optimizer state changed")
        common()

    def bind(key, value) -> None:
        report[key] = value
        metadata[key] = (value, pickle.dumps(value, protocol=5))

    try:
        common()
        admitted = _admit_reference(_load_reference())
        sources = admitted["sources"]
        packets = _fixed_a_packets(sources)
        raw_sources = sources[:2]
        signatures.extend((values, shared._signature(values)) for values in (sources, packets))
        bind("reference", admitted["provenance"])
        bind(
            "historical_work",
            _work(
                [admitted["row"]],
                admitted["evaluations"],
                admitted["provenance"]["source_reports"],
                False,
            ),
        )
        bind(
            "retained_input_array_bytes",
            dict(
                original_sources=sum(p.features.nbytes + p.targets.nbytes for p in sources),
                training=sum(p.features.nbytes + p.targets.nbytes for p in packets),
            ),
        )
        guard()
        arm = shared._train_arm(row, packets, memory, deadline, common, checkpoint_updates=POINTS)
        arm["guard"]()
        receipts = row["stream_result"]["applied_updates"]
        if len(receipts) != 128 or any(
            r["diagnostic"]["definition"] != TORCH_RECURRENT_CE_ID for r in receipts
        ):
            raise ValueError("warm prefix must preserve all128 original mean-CE receipts")
        paid_training = pickle.dumps(row, protocol=5)
        for index, (point, state) in enumerate(arm["states"]):
            guard()
            try:
                if point in (0, 128):
                    current = evaluate_checkpoint(
                        arm["source"],
                        state,
                        method="recurrent_bc",
                        updates=point,
                        layouts=predefined_layouts()[:1],
                        role="train",
                        sources=raw_sources,
                        budget=arm["budget"],
                        check_boundary=guard,
                    )
                else:
                    current = _score_checkpoint(
                        arm["source"],
                        state,
                        method="recurrent_bc",
                        updates=point,
                        sources=raw_sources,
                        budget=arm["budget"],
                        check_boundary=guard,
                    )
            except MiniGridEvaluationFailed as error:
                current = error.report
            evaluations[index] = owners[index] = current
            payloads[index] = pickle.dumps(current, protocol=5)
            try:
                _parity(admitted["evaluations"][index], current)
            except Exception:
                parity_checks[index] = expected_parity[index] = False
                raise
            parity_checks[index] = expected_parity[index] = True
            guard()
        expected_source = deepcopy(arm["states"][-1][1])
        if expected_source.completed_updates != 128 or expected_source.training_examples != 1152:
            raise ValueError("warm native checkpoint counters differ from128/1152")
        guard()
        restored = TorchDemonstrationRewardLearner(TorchRecurrentPolicy(148, 8, 7, 17))
        guard()
        restored.restore_state(deepcopy(expected_source))
        source = restored
        guard()
        report["full_state_restored"] = True
    except Exception as error:
        primary_error = f"{type(error).__name__}: {error}"
        if paid_training is None:
            paid_training = pickle.dumps(row, protocol=5)
    try:
        observed_rss = asdict(memory.snapshot())
        guard()
    except Exception as error:
        observed_rss = None
        report["readback_error"] = f"{type(error).__name__}: {error}"
        primary_error = primary_error or report["readback_error"]
    # Publish only private paid captures after the last external callback. On a
    # later reward fault the caller retains this completed report separately.
    report["training"] = pickle.loads(paid_training) if paid_training is not None else row
    report["evaluations"] = [pickle.loads(payload) for payload in payloads]
    report["parity_checks"] = list(expected_parity)
    for key, (_, payload) in metadata.items():
        report[key] = pickle.loads(payload)
    report["observed_process_rss"] = observed_rss
    report["work"] = _work([report["training"]], report["evaluations"], [], False)
    report.update(
        status="complete" if primary_error is None else "incomplete",
        reason=primary_error,
        parity_matched=all(value is True for value in expected_parity),
        full_state_restored=primary_error is None and source is not None,
    )
    if primary_error is not None:
        raise RuntimeError(primary_error)
    # Rebind the completed public containers once, without changing their data.
    row = report["training"]
    evaluations = report["evaluations"]
    parity_checks = report["parity_checks"]
    owners = list(evaluations)
    paid_training = pickle.dumps(row, protocol=5)
    metadata = {key: (report[key], payload) for key, (_, payload) in metadata.items()}
    completed_report = pickle.dumps(report, protocol=5)
    return dict(source=source, guard=guard, updates=128, rows=1152, sources=sources)
