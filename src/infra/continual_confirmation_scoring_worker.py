"""Run the fixed scored composition while held models/views remain budgeted.

Inputs are the exact saved request/scope and repository paths. Outputs are
scientific JSON plus independently observed worker metadata. No publication,
reference-body decode, setting selection or partial scientific CLI belongs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

from src.app.continual_confirmation_execution import json_value, verify_process_memory
from src.app.continual_confirmation_final_observation import verify_final_observation
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_scoring import ScoredConfirmation, evaluate_confirmation
from src.app.continual_confirmation_scoring_execution import elapsed_seconds, verify_scored_updates
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    fixed_scoring_manifest,
)
from src.app.continual_confirmation_scoring_state import (
    TrainingStateProof,
    verify_scoring_training_state,
)
from src.app.continual_confirmation_training import TrainedConfirmation, train_confirmation
from src.core.confirmation_final_roles import capture_final_role
from src.infra.continual_confirmation_final_runtime import FinalExecutionObserver
from src.infra.continual_confirmation_io import parse_json
from src.infra.continual_confirmation_runtime import ExecutionObserver
from src.infra.continual_confirmation_scoring_bindings import checked_scoring_request
from src.infra.continual_confirmation_training_references import stream_file_identity
from src.shared.process_memory import ProcessRssSampler


@dataclass(frozen=True)
class _ScoringPorts:
    """Private component-test seam; the fixed worker always uses public gates."""

    evaluate: Callable[
        [
            TrainedConfirmation,
            ConfirmationScoringManifest,
            FinalExecutionObserver,
            Callable[[str], None],
        ],
        ScoredConfirmation,
    ]
    verify_state: Callable[[TrainedConfirmation, ConfirmationScoringManifest], TrainingStateProof]
    verify_json: Callable[[Any, Any, ConfirmationScoringManifest], Any]


@dataclass(frozen=True)
class _ScoredParts:
    scored: ScoredConfirmation
    observer: FinalExecutionObserver
    result_json: str
    observations: dict[str, Any]


def _evaluate(
    trained: TrainedConfirmation,
    manifest: ConfirmationScoringManifest,
    observer: FinalExecutionObserver,
    checkpoint: Callable[[str], None],
) -> ScoredConfirmation:
    return evaluate_confirmation(trained, manifest, observer.release, observer.evaluate, checkpoint)


def _default_ports() -> _ScoringPorts:
    return _ScoringPorts(_evaluate, verify_scoring_training_state, verify_final_observation)


def _require_retained(parts: _ScoredParts) -> None:
    observer = parts.observer
    require(observer.used and not observer.active, "scored final observer did not finish")
    for binding in observer.bindings:
        current = getattr(binding.item, binding.attribute)
        require(
            current is binding.original and current._source is binding.source,
            "scored original role/source changed after guard restoration",
        )
        role, before = observer.views[binding.key]
        require(
            role.input is binding.observed.values["test_input"]
            and role.target is binding.observed.values["test_target"]
            and capture_final_role(role, before.count) == before,
            "scored retained final content/source arrays changed",
        )
    for request in observer.predictions:
        models = (
            request.item.models_after_a
            if request.checkpoint == "a"
            else request.item.models_after_b
        )
        require(models.get(request.arm) is request.model, "scored held model identity changed")
    same_json(observer.observations(), parts.observations, "scored late actual observations")
    same_json(
        json_value(asdict(parts.scored)),
        parse_json(parts.result_json),
        "scored serialized/live endpoint links",
    )


def _require_complete_state(
    trained: TrainedConfirmation,
    manifest: ConfirmationScoringManifest,
    parts: _ScoredParts,
    ports: _ScoringPorts,
) -> None:
    proof = ports.verify_state(trained, manifest)
    same_json(
        asdict(proof),
        asdict(parts.scored.training_after_evaluation),
        "scored post-serialization complete training proof",
    )
    _require_retained(parts)


def _score_held(
    trained: TrainedConfirmation,
    manifest: ConfirmationScoringManifest,
    budget: ExecutionObserver,
    check_bindings: Callable[[], None],
    ports: _ScoringPorts | None = None,
) -> _ScoredParts:
    """Private composition seam; no fixture policy is exposed by the worker/CLI."""
    selected = _default_ports() if ports is None else ports
    final = FinalExecutionObserver(trained, budget.checkpoint)

    def checkpoint(stage: str) -> None:
        if stage in {"before_final_release", "before_final_evaluation", "after_final_evaluation"}:
            check_bindings()
        final.checkpoint(stage)

    with final.observe():
        scored = selected.evaluate(trained, manifest, final, checkpoint)
        final.verify_result(scored)
        # Both encodings/decoded validation and their allocations occur with
        # the models/copies/final arrays retained inside the sampler window.
        result_json = json.dumps(asdict(scored), sort_keys=True, allow_nan=False)
        observed = final.observations()
        selected.verify_json(observed, parse_json(result_json), manifest)
        check_bindings()
        budget.checkpoint()
        final.verify_result(scored)
    parts = _ScoredParts(scored, final, result_json, observed)
    # This follows ALL observer callbacks, including normal context exit.
    _require_complete_state(trained, manifest, parts, selected)
    return parts


def worker_parts(root: Path, request_file: Path, scope_file: Path) -> tuple[str, dict[str, Any]]:
    """The full source-bound child; never accepts fixture scope/digest overrides."""
    manifest = fixed_scoring_manifest()
    ports = _default_ports()
    with ProcessRssSampler(
        interval_seconds=manifest.train_manifest.rss_interval_seconds
    ) as sampler:
        budget = ExecutionObserver(manifest.train_manifest, sampler)
        request = checked_scoring_request(root, request_file, scope_file)
        request_identity = stream_file_identity(request_file)

        def recheck() -> None:
            same_json(
                checked_scoring_request(root, request_file, scope_file, request_identity),
                request,
                "scored original/current request",
            )

        with budget.observe_updates():
            trained = train_confirmation(manifest.train_manifest)
            verify_scored_updates(budget.updates(), request)
            recheck()
            parts = _score_held(trained, manifest, budget, recheck)
            recheck()
            budget.checkpoint()
            _require_complete_state(trained, manifest, parts, ports)
            verify_scored_updates(budget.updates(), request)
            budget.checkpoint()
    # Include the exit sample and retained graphs before framing stdout. The
    # original resource contract excludes framing and parent publication only.
    budget.checkpoint()
    memory = asdict(sampler.snapshot())
    verify_process_memory(memory, manifest.train_manifest)
    elapsed = budget.elapsed()
    elapsed_seconds(elapsed, "worker", manifest.train_manifest.wall_limit_seconds)
    metadata = {
        "request_sha256": request_identity["sha256"],
        "reference_report_sha256": request["reference_report_sha256"],
        "source_map_sha256": request["source_map_sha256"],
        "observed_updates": budget.updates(),
        "final_observation": parts.observations,
        "process_rss": memory,
        "worker_elapsed_seconds": elapsed,
    }
    return parts.result_json, metadata
