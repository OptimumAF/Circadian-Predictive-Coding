"""Opt-in toy CLI lifecycle around the checked app budget and checkpoint.

Inputs are a resolved scientific request, local artifact paths, and an
execution budget. Outputs are a completed result or an explicit stop/error
state. This adapter does not choose models, seeds, metrics, or final scores.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, cast
import sys

from src.app.experiment_runner import ExperimentConfig, ExperimentResult
from src.app.toy_checkpoint import toy_checkpoint_width_work, toy_config_digest
from src.app.toy_execution_budget import (
    ToyExecutionBudget,
    ToyExecutionProgress,
    ToyExecutionStopped,
    ToyProcessRssUnavailable,
)
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore
from src.infra.local_result_json import write_local_json_payload
from src.infra.toy_result_files import write_toy_result_json
from src.infra.toy_run_state_files import ToyRunStateFile
from src.shared.process_memory import ProcessRssSegment


TOY_RUN_STATE_SCHEMA = "toy_cli_run_state_v1"
_CHECKPOINT_RESERVATION = b"CIRCADIAN_TOY_CHECKPOINT_RESERVED_V1\n"


def validate_budget_paths(
    state_path: Path,
    checkpoint_path: Path | None,
    result_path: Path | None,
    config_path: Path | None,
) -> None:
    """Reject overlapping artifact roles before any path is written."""
    paths = {
        "run-state": state_path,
        "run-state lock": state_path.with_name(f"{state_path.name}.lock"),
        "checkpoint": checkpoint_path,
        "JSON result": result_path,
        "resolved config": config_path,
    }
    seen: dict[Path, str] = {}
    for role, path in paths.items():
        if path is None:
            continue
        identity = path.resolve()
        if identity in seen:
            raise ValueError(f"toy {role} and {seen[identity]} must use different paths")
        seen[identity] = role


def run_budgeted_toy_cli(
    *,
    config: ExperimentConfig,
    resolved_record: Mapping[str, object],
    budget: ToyExecutionBudget,
    state_path: Path,
    checkpoint_path: Path | None,
    result_path: Path | None,
    config_path: Path | None,
    resume: bool,
    input_tokens: list[str],
    runner: Callable[..., ExperimentResult],
) -> ExperimentResult:
    """Claim one run state, then publish only a fully scored result."""
    state_file = ToyRunStateFile(state_path)
    store = TrustedLocalToyCheckpointStore(checkpoint_path) if checkpoint_path else None
    paths = _artifact_paths(checkpoint_path, result_path, config_path)
    with state_file.exclusive():
        if resume:
            state = state_file.load()
            _validate_resume(state, config, resolved_record, paths, store)
            assert store is not None
            config_published = _validate_existing_config_artifact(config_path, state)
        else:
            if state_path.exists():
                raise FileExistsError(f"toy run state already exists: {state_path}")
            if checkpoint_path is not None and checkpoint_path.exists():
                raise FileExistsError(f"toy checkpoint already exists: {checkpoint_path}")
            state = _initial_state(config, resolved_record, paths)
            config_published = False
            if checkpoint_path is not None:
                # Why this: claim the checkpoint name before the replaceable
                # store can overwrite a path created by another local run.
                _reserve_checkpoint_path(checkpoint_path)
            try:
                state_file.create(state)
            except BaseException:
                if checkpoint_path is not None:
                    _remove_own_reservation(checkpoint_path)
                raise

        _start_attempt(state, budget, resume, input_tokens)
        state_file.replace(state)
        progress = ToyExecutionProgress()
        started_at = perf_counter()
        try:
            result = runner(
                config=config,
                checkpoint_store=store,
                resume_from_checkpoint=resume,
                execution_budget=budget,
                execution_progress=progress,
            )
            if config_path is not None and not config_published:
                write_local_json_payload(state["resolved_config"], config_path)
            if result_path is not None:
                write_toy_result_json(
                    result, result_path, cast(Mapping[str, object], state["resolved_config"])
                )
        except ToyExecutionStopped as exc:
            _finish_attempt(
                state,
                status="incomplete",
                reason=exc.stop.reason,
                error_type=None,
                progress=progress,
                elapsed_seconds=exc.stop.elapsed_seconds,
                store=store,
            )
            state_file.replace(state)
            print(str(exc), file=sys.stderr)
            raise SystemExit(3) from exc
        except Exception as exc:
            _finish_attempt(
                state,
                status="error",
                reason=(
                    "process_rss_unavailable"
                    if isinstance(exc, ToyProcessRssUnavailable)
                    else "exception"
                ),
                error_type=type(exc).__name__,
                progress=progress,
                elapsed_seconds=perf_counter() - started_at,
                store=store,
            )
            state_file.replace(state)
            raise

        _finish_attempt(
            state,
            status="completed",
            reason="completed",
            error_type=None,
            progress=progress,
            elapsed_seconds=perf_counter() - started_at,
            store=store,
        )
        state_file.replace(state)
        return result


def _artifact_paths(
    checkpoint_path: Path | None, result_path: Path | None, config_path: Path | None
) -> dict[str, str | None]:
    return {
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "result": str(result_path.resolve()) if result_path else None,
        "resolved_config": str(config_path.resolve()) if config_path else None,
    }


def _reserve_checkpoint_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as reserved:
        reserved.write(_CHECKPOINT_RESERVATION)
        reserved.flush()
        os.fsync(reserved.fileno())


def _remove_own_reservation(path: Path) -> None:
    try:
        if path.read_bytes() == _CHECKPOINT_RESERVATION:
            path.unlink()
    except OSError:
        # Preserve the original state-create error and never delete a file
        # whose exact reservation bytes could not be confirmed.
        pass


def _initial_state(
    config: ExperimentConfig,
    resolved_record: Mapping[str, object],
    paths: Mapping[str, str | None],
) -> dict[str, Any]:
    return {
        "schema_id": TOY_RUN_STATE_SCHEMA,
        "config_digest": toy_config_digest(config),
        "resolved_config": dict(resolved_record),
        "artifacts": dict(paths),
        "status": "running",
        "reason": None,
        "error_type": None,
        "checkpoint_error_type": None,
        "budget": None,
        "work": _work(0, 0, 0.0, 0, 0),
        "checkpoint": None,
        "attempts": [],
    }


def _start_attempt(
    state: dict[str, Any], budget: ToyExecutionBudget, resume: bool, tokens: list[str]
) -> None:
    requested_budget = asdict(budget)
    state["status"] = "running"
    state["reason"] = None
    state["error_type"] = None
    state["checkpoint_error_type"] = None
    state["budget"] = requested_budget
    state["attempts"].append(
        {
            "resume": resume,
            "explicit_inputs": list(tokens),
            "budget": requested_budget,
            "status": "running",
            "reason": None,
            "error_type": None,
            "checkpoint_error_type": None,
            "work": state["work"],
            "checkpoint": state["checkpoint"],
        }
    )


def _finish_attempt(
    state: dict[str, Any],
    *,
    status: str,
    reason: str,
    error_type: str | None,
    progress: ToyExecutionProgress,
    elapsed_seconds: float,
    store: TrustedLocalToyCheckpointStore | None,
) -> None:
    checkpoint_error_type = None
    try:
        checkpoint = _checkpoint_identity(store, state["config_digest"])
    except Exception as exc:
        # Why this: an unreadable checkpoint must not suppress the truthful
        # stop/error record or pretend that durable work was verified.
        checkpoint = None
        checkpoint_error_type = type(exc).__name__
    checkpointed = (
        cast(int, checkpoint["training_updates"])
        if checkpoint
        else (None if checkpoint_error_type else 0)
    )
    checkpointed_replay = (
        cast(int, checkpoint["replay_examples"])
        if checkpoint
        else (None if checkpoint_error_type else 0)
    )
    checkpointed_width = cast(int, checkpoint["hidden_width"]) if checkpoint else None
    checkpointed_peak_width = cast(int, checkpoint["peak_hidden_width"]) if checkpoint else None
    work = _work(
        progress.updates_completed,
        checkpointed,
        elapsed_seconds,
        progress.replay_examples_completed,
        checkpointed_replay,
        hidden_width=progress.hidden_width_observed,
        peak_hidden_width=progress.peak_hidden_width_observed,
        checkpointed_hidden_width=checkpointed_width,
        checkpointed_peak_hidden_width=checkpointed_peak_width,
        rejected_proposed_hidden_width=progress.rejected_proposed_hidden_width,
        process_rss_segment=progress.process_rss_segment,
    )
    state.update(
        status=status,
        reason=reason,
        error_type=error_type,
        checkpoint_error_type=checkpoint_error_type,
        work=work,
        checkpoint=checkpoint,
    )
    state["attempts"][-1].update(
        status=status,
        reason=reason,
        error_type=error_type,
        checkpoint_error_type=checkpoint_error_type,
        work=work,
        checkpoint=checkpoint,
    )


def _work(
    observed: int,
    checkpointed: int | None,
    elapsed: float,
    replay_observed: int,
    replay_checkpointed: int | None,
    *,
    hidden_width: int | None = None,
    peak_hidden_width: int | None = None,
    checkpointed_hidden_width: int | None = None,
    checkpointed_peak_hidden_width: int | None = None,
    rejected_proposed_hidden_width: int | None = None,
    process_rss_segment: ProcessRssSegment | None = None,
) -> dict[str, Any]:
    return {
        "training_updates_observed": observed,
        "checkpointed_training_updates": checkpointed,
        "replay_examples_observed": replay_observed,
        "checkpointed_replay_examples": replay_checkpointed,
        "hidden_width_observed": hidden_width,
        "peak_hidden_width_observed": peak_hidden_width,
        "checkpointed_hidden_width": checkpointed_hidden_width,
        "checkpointed_peak_hidden_width": checkpointed_peak_hidden_width,
        "rejected_proposed_hidden_width": rejected_proposed_hidden_width,
        "process_rss": (
            {"scope": "absolute_current_process_per_invocation", **asdict(process_rss_segment)}
            if process_rss_segment is not None
            else None
        ),
        "elapsed_seconds": elapsed,
    }


def _checkpoint_identity(
    store: TrustedLocalToyCheckpointStore | None, config_digest: str
) -> dict[str, object] | None:
    if store is None or not store.path.exists():
        return None
    raw = store.path.read_bytes()
    if raw == _CHECKPOINT_RESERVATION:
        return None
    digest = sha256(raw).hexdigest()
    checkpoint = store.load()
    if sha256(store.path.read_bytes()).hexdigest() != digest:
        raise ValueError("toy checkpoint changed while reading its identity")
    if checkpoint.runner_config_digest != config_digest:
        raise ValueError("toy checkpoint config differs from run state")
    hidden_width, peak_hidden_width = toy_checkpoint_width_work(checkpoint)
    return {
        "path": str(store.path.resolve()),
        "sha256": digest,
        "position": asdict(checkpoint.combined.position),
        "training_updates": sum(len(history) for history in checkpoint.losses),
        "replay_examples": sum(event.replay.applied_examples for event in checkpoint.sleep_events),
        "hidden_width": hidden_width,
        "peak_hidden_width": peak_hidden_width,
    }


def _validate_resume(
    state: dict[str, object],
    config: ExperimentConfig,
    resolved_record: Mapping[str, object],
    paths: Mapping[str, str | None],
    store: TrustedLocalToyCheckpointStore | None,
) -> None:
    if state.get("schema_id") != TOY_RUN_STATE_SCHEMA:
        raise ValueError("incompatible toy run state schema")
    if state.get("status") not in {"incomplete", "error"}:
        raise ValueError("toy run state is not an incomplete/error run")
    if state.get("config_digest") != toy_config_digest(config):
        raise ValueError("toy resume config differs from run state")
    saved_record = state.get("resolved_config")
    requested_record = json.loads(json.dumps(resolved_record, allow_nan=False))
    if type(saved_record) is not dict or any(
        saved_record.get(key) != requested_record.get(key)
        for key in ("config", "preset", "mode", "seeds", "noise_levels", "trial_configs")
    ):
        raise ValueError("toy resume resolved config differs from run state")
    if state.get("artifacts") != dict(paths):
        raise ValueError("toy resume artifact paths differ from run state")
    saved_checkpoint = state.get("checkpoint")
    if store is None or type(saved_checkpoint) is not dict:
        raise ValueError("toy run is non-resumable without a saved checkpoint")
    current = _checkpoint_identity(store, toy_config_digest(config))
    # Why this: earlier v1 states had the same byte hash and cursor but
    # lacked the later additive replay and width work fields.
    comparable_current = dict(current) if current is not None else None
    if comparable_current is not None:
        for key in ("replay_examples", "hidden_width", "peak_hidden_width"):
            if key not in saved_checkpoint:
                comparable_current.pop(key)
    if comparable_current != saved_checkpoint:
        raise ValueError("toy checkpoint identity differs from run state")


def _validate_existing_config_artifact(path: Path | None, state: dict[str, object]) -> bool:
    """Reuse an exact config file left by a failed result publication."""
    if path is None or not path.exists():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("toy resolved config artifact cannot be read for resume") from exc
    if saved != state["resolved_config"]:
        raise ValueError("toy resolved config artifact differs from run state")
    return True
