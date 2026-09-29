"""Prepare and run one fixed, locally budgeted strict-online A/B confirmation.

The saved request is the experiment input. ``prepare`` writes it without
opening a source; ``run`` accepts only that unchanged request and writes every
outer trial and final score. Checkpoints and result files are never reused.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import pickle
from time import monotonic
from typing import Any, Callable, Iterator
from unittest.mock import patch

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_arrived_selection as selection
from src.app import continual_shift_benchmark as base
from src.core.circadian_predictive_coding import CircadianConfig
from src.infra.circadian_checkpoint_files import TrustedLocalArrivedSelectionCheckpointStore


_REQUEST_SCHEMA = "continual_arrived_confirmation_request_v1"
_OBJECTIVE = "mean_0.5_phase_a_post_plus_0.5_phase_b_post_outer_accuracy"
_SEEDS = [17, 19]
_MAX_ELAPSED_SECONDS = 180
_UPDATES_PER_ORDER_PATH = 24


def _build_candidates(
    model_order: tuple[str, ...],
) -> tuple[selection.ArrivedSelectionCandidate, ...]:
    # Why this: use the same tiny fixed grid as the previously recorded c3a smoke.
    training = base.ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=1,
        phase_b_epochs=1,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=1,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=1,
        ),
        replay_max_examples=4,
        replay_max_bytes=96,
        model_order=model_order,
    )
    first = arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2)
    second = replace(
        first,
        training=replace(
            training,
            backprop_learning_rate=training.backprop_learning_rate * 0.7,
            pc_learning_rate=training.pc_learning_rate * 0.7,
            circadian_learning_rate=training.circadian_learning_rate * 0.7,
        ),
    )
    return (
        selection.ArrivedSelectionCandidate("default", first),
        selection.ArrivedSelectionCandidate("lower_rate", second),
    )


def _model_orders() -> dict[str, tuple[str, ...]]:
    return {
        "forward": base.CONTINUAL_MODEL_ORDER,
        "reverse": tuple(reversed(base.CONTINUAL_MODEL_ORDER)),
    }


def _predeclared_request() -> dict[str, Any]:
    orders = _model_orders()
    return {
        "schema": _REQUEST_SCHEMA,
        "protocol_id": selection.ARRIVED_SELECTION_PROTOCOL,
        "selection_objective": _OBJECTIVE,
        "selection_tie_rule": "first_predeclared_candidate",
        "seeds": _SEEDS,
        "model_orders": {name: list(order) for name, order in orders.items()},
        "candidates_by_order": {
            name: [asdict(candidate) for candidate in _build_candidates(order)]
            for name, order in orders.items()
        },
        "budget": {
            "candidate_count": 2,
            "candidate_seed_trials_per_method": 4,
            "model_updates_per_order_path": _UPDATES_PER_ORDER_PATH,
            "total_model_updates": 4 * _UPDATES_PER_ORDER_PATH,
            "max_elapsed_seconds": _MAX_ELAPSED_SECONDS,
            "checkpoint_interruptions": ["phase_a_wake", "phase_b_wake"],
            "paths_per_order": ["ordinary", "checkpoint_resume"],
        },
        "required_outcomes": [
            "all_outer_trials_and_exposures",
            "per_method_frozen_choices",
            "all_per_seed_final_metrics_and_role_hashes",
            "replay_ids_counts_bytes_and_task_information",
            "phase_a_and_b_interruption_cursors",
            "trained_state_digests_and_model_order_isolation",
            "final_source_read_ledger",
            "circadian_minus_each_baseline_per_seed",
        ],
        "scope": "tiny_synthetic_local_confirmation_no_general_performance_claim",
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def prepare_request(path: Path) -> None:
    """Save the exact fixed settings before any confirmation source is opened."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(_predeclared_request()))


def _read_request(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError("predeclared confirmation request is missing")
    raw = path.read_bytes()
    try:
        request: dict[str, Any] = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("predeclared confirmation request is invalid") from error
    if raw.replace(b"\r\n", b"\n") != _canonical_json(_predeclared_request()).encode("utf-8"):
        raise ValueError("predeclared confirmation request changed")
    return request, sha256(raw).hexdigest()


@contextmanager
def _sealed_final_sources(is_frozen: Callable[[], bool]) -> Iterator[list[dict[str, Any]]]:
    reads: list[dict[str, Any]] = []
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source

    class SealedSource:
        def __init__(self, source: Any, phase: str, seed: int) -> None:
            self.source = source
            self.phase = phase
            self.seed = seed
            self.train_input = source.train_input
            self.train_target = source.train_target

        def _read_final(self, field: str) -> Any:
            if not is_frozen():
                raise AssertionError("final source opened before all candidates froze")
            reads.append({"seed": self.seed, "phase": self.phase, "field": field})
            return getattr(self.source, f"test_{'target' if field == 'label' else field}")

        @property
        def test_input(self) -> Any:
            return self._read_final("input")

        @property
        def test_target(self) -> Any:
            return self._read_final("label")

    def source_a(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_a(*args, **kwargs), "a", kwargs["seed"])

    def source_b(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_b(*args, **kwargs), "b", args[1] - 101)

    with (
        patch.object(arrived, "generate_two_cluster_dataset_with_transform", source_a),
        patch.object(arrived, "_generate_phase_b_source", source_b),
    ):
        yield reads


@contextmanager
def _count_model_updates() -> Iterator[list[str]]:
    updates: list[str] = []
    original = base._train_named_model_epoch

    def counted(*args: Any, **kwargs: Any) -> None:
        updates.append(args[-1])
        if len(updates) > _UPDATES_PER_ORDER_PATH:
            raise AssertionError("predeclared model-update budget exceeded")
        original(*args, **kwargs)

    with patch.object(base, "_train_named_model_epoch", counted):
        yield updates


def _hash_training_state(state: Any) -> dict[str, str]:
    names = (
        "backprop_model",
        "predictive_model",
        "circadian_model",
        "backprop_after_a",
        "predictive_after_a",
        "circadian_after_a",
    )

    def hash_model(model: Any) -> str:
        fields = model.state if hasattr(model, "state") else model.__dict__
        parts: list[tuple[str, Any]] = []
        for field_name, value in sorted(fields.items()):
            if field_name == "_replay_memory":
                replay = [
                    (
                        sha256(pickle.dumps(row.input_batch, protocol=5)).hexdigest(),
                        sha256(pickle.dumps(row.target_batch, protocol=5)).hexdigest(),
                        row.priority,
                        row.positive_fraction,
                    )
                    for row in value
                ]
                parts.append((field_name, replay))
            else:
                parts.append((field_name, sha256(pickle.dumps(value, protocol=5)).hexdigest()))
        return sha256(_canonical_json(parts).encode("utf-8")).hexdigest()

    return {name: hash_model(getattr(state, name)) for name in names}


def _run_ordinary(
    candidates: tuple[selection.ArrivedSelectionCandidate, ...],
) -> tuple[selection.ArrivedOuterSelectionResult, dict[str, Any]]:
    frozen = False
    state_hashes: dict[str, dict[str, str]] = {}
    original_freeze = selection._freeze_selection
    original_train = arrived._train_arrived_seed

    def capture_freeze(*args: Any, **kwargs: Any) -> Any:
        nonlocal frozen
        result = original_freeze(*args, **kwargs)
        frozen = True
        return result

    def capture_state(config: Any, seed: int) -> Any:
        pending = original_train(config, seed)
        candidate_id = next(item.candidate_id for item in candidates if item.config == config)
        state_hashes[f"{candidate_id}:{seed}"] = _hash_training_state(pending.state)
        return pending

    with (
        _sealed_final_sources(lambda: frozen) as reads,
        _count_model_updates() as updates,
        patch.object(selection, "_freeze_selection", capture_freeze),
        patch.object(arrived, "_train_arrived_seed", capture_state),
    ):
        result = selection.run_arrived_outer_selection(candidates, _SEEDS)
    if not frozen or len(reads) != 8 or len(updates) != _UPDATES_PER_ORDER_PATH:
        raise AssertionError("ordinary selection did not meet the fixed release/work contract")
    return result, {"reads": reads, "updates": len(updates), "states": state_hashes}


class _PlannedInterruption(Exception):
    """Stop after the declared durable A or B wake checkpoint."""


class _InterruptingStore:
    def __init__(self, path: Path) -> None:
        self.store = TrustedLocalArrivedSelectionCheckpointStore(path)
        self.boundary: str | None = "a"

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        active = checkpoint.active_v6
        if (
            active is not None
            and checkpoint.candidate_index == 0
            and active.seed_index == 0
            and active.phase == self.boundary
            and active.stage == "wake"
        ):
            raise _PlannedInterruption()


def _run_checkpointed(
    candidates: tuple[selection.ArrivedSelectionCandidate, ...], path: Path
) -> tuple[selection.ArrivedOuterSelectionResult, dict[str, Any]]:
    store = _InterruptingStore(path)
    interruptions: list[str] = []

    def is_frozen() -> bool:
        return path.exists() and store.load().stage == "frozen"

    with _sealed_final_sources(is_frozen) as reads, _count_model_updates() as updates:
        for phase in ("a", "b"):
            store.boundary = phase
            try:
                selection.run_arrived_outer_selection(
                    candidates,
                    _SEEDS,
                    checkpoint_store=store,
                    resume_from_checkpoint=phase == "b",
                )
            except _PlannedInterruption:
                checkpoint = store.load()
                active = checkpoint.active_v6
                if (
                    checkpoint.stage != "training"
                    or checkpoint.freeze is not None
                    or active is None
                    or active.phase != phase
                    or active.stage != "wake"
                    or reads
                ):
                    raise AssertionError("A/B interruption crossed the final-source seal") from None
                interruptions.append(f"phase_{phase}_wake")
            else:
                raise AssertionError(f"missing phase {phase} checkpoint interruption")
        store.boundary = None
        result = selection.run_arrived_outer_selection(
            candidates, _SEEDS, checkpoint_store=store, resume_from_checkpoint=True
        )
    checkpoint = store.load()
    if (
        checkpoint.stage != "frozen"
        or checkpoint.freeze != result.freeze
        or len(reads) != 8
        or len(updates) != _UPDATES_PER_ORDER_PATH
    ):
        raise AssertionError("checkpointed confirmation did not meet release/work contract")
    state_hashes = {
        f"{candidate.candidate_id}:{seed.seed}": _hash_training_state(seed.state)
        for candidate in checkpoint.completed_candidates
        for seed in candidate.unscored_seeds
    }
    return result, {
        "reads": reads,
        "updates": len(updates),
        "states": state_hashes,
        "interruptions": interruptions,
        "checkpoint_manifest_digest": checkpoint.manifest_digest,
        "checkpoint_stage": checkpoint.stage,
    }


def _assert_model_order_isolation(
    forward: selection.ArrivedOuterSelectionResult,
    reverse: selection.ArrivedOuterSelectionResult,
    forward_states: dict[str, dict[str, str]],
    reverse_states: dict[str, dict[str, str]],
) -> None:
    for first_trial, second_trial in zip(forward.trials, reverse.trials, strict=True):
        if replace(first_trial, role_accesses=()) != replace(second_trial, role_accesses=()):
            raise AssertionError("model order changed an outer trial")
    if forward.selections != reverse.selections or forward_states != reverse_states:
        raise AssertionError("model order changed a choice or trained state")
    for first_seed, second_seed in zip(
        forward.final_seed_results, reverse.final_seed_results, strict=True
    ):
        if (
            first_seed.role_ids != second_seed.role_ids
            or first_seed.role_hashes != second_seed.role_hashes
        ):
            raise AssertionError("model order changed final role identity")
        for method in base.CONTINUAL_MODEL_ORDER:
            if getattr(first_seed.metrics, method) != getattr(second_seed.metrics, method):
                raise AssertionError("model order changed a final method outcome")


def _seed_differences(
    result: selection.ArrivedOuterSelectionResult,
) -> list[dict[str, float | int]]:
    return [
        {
            "seed": row.seed,
            "circadian_minus_backprop_balanced": (
                row.metrics.circadian_predictive_coding.balanced_score
                - row.metrics.backprop.balanced_score
            ),
            "circadian_minus_predictive_balanced": (
                row.metrics.circadian_predictive_coding.balanced_score
                - row.metrics.predictive_coding.balanced_score
            ),
        }
        for row in result.final_seed_results
    ]


def run_confirmation(request_path: Path, result_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    """Execute the unchanged request; preserve each negative score difference."""
    request, request_sha256 = _read_request(request_path)
    if result_path.exists() or checkpoint_dir.exists():
        raise FileExistsError("confirmation result or checkpoint directory already exists")
    checkpoint_dir.mkdir(parents=True)
    started = monotonic()
    orders: dict[str, Any] = {}
    raw_results: dict[str, selection.ArrivedOuterSelectionResult] = {}
    state_sets: dict[str, dict[str, dict[str, str]]] = {}
    for name, model_order in _model_orders().items():
        candidates = _build_candidates(model_order)
        ordinary, ordinary_audit = _run_ordinary(candidates)
        checkpointed, checkpoint_audit = _run_checkpointed(
            candidates, checkpoint_dir / f"{name}.ckpt"
        )
        if ordinary != checkpointed or ordinary_audit["states"] != checkpoint_audit["states"]:
            raise AssertionError("checkpoint resume changed the ordinary outcome or trained state")
        if monotonic() - started > request["budget"]["max_elapsed_seconds"]:
            raise TimeoutError("predeclared local confirmation wall-time budget exceeded")
        raw_results[name] = ordinary
        state_sets[name] = ordinary_audit["states"]
        orders[name] = {
            "result": asdict(ordinary),
            "seed_differences": _seed_differences(ordinary),
            "checkpoint_matches_ordinary": True,
            "model_updates": {
                "ordinary": ordinary_audit["updates"],
                "checkpoint_resume": checkpoint_audit["updates"],
            },
            "trained_state_sha256": ordinary_audit["states"],
            "final_source_reads": {
                "ordinary": ordinary_audit["reads"],
                "checkpoint_resume": checkpoint_audit["reads"],
            },
            "interruptions": checkpoint_audit["interruptions"],
            "checkpoint_manifest_digest": checkpoint_audit["checkpoint_manifest_digest"],
            "checkpoint_stage": checkpoint_audit["checkpoint_stage"],
        }
    _assert_model_order_isolation(
        raw_results["forward"],
        raw_results["reverse"],
        state_sets["forward"],
        state_sets["reverse"],
    )
    if sha256(request_path.read_bytes()).hexdigest() != request_sha256:
        raise ValueError("predeclared request changed during confirmation")
    report: dict[str, Any] = {
        "schema": "continual_arrived_confirmation_result_v1",
        "request_sha256": request_sha256,
        "request_path": str(request_path),
        "checks": {
            "model_order_isolated": True,
            "total_model_updates": sum(
                sum(order["model_updates"].values()) for order in orders.values()
            ),
            "final_source_sealed_until_freeze": True,
        },
        "orders": orders,
        "elapsed_seconds": monotonic() - started,
    }
    if report["checks"]["total_model_updates"] != request["budget"]["total_model_updates"]:
        raise AssertionError("confirmation model-update total differs from request")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--request", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--request", required=True, type=Path)
    run.add_argument("--result", required=True, type=Path)
    run.add_argument("--checkpoint-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_request(args.request)
        print(args.request)
    else:
        report = run_confirmation(args.request, args.result, args.checkpoint_dir)
        print(json.dumps({"result": str(args.result), "checks": report["checks"]}, sort_keys=True))


if __name__ == "__main__":
    main()
