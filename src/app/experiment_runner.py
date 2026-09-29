"""Use-case orchestration for model comparison experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

from src.app.comparison_scope import NumpyComparisonScope, scope_for_hidden_dims
from src.app.circadian_checkpoint import (
    CircadianResumePosition,
    capture_circadian_checkpoint,
    restore_circadian_checkpoint,
)
from src.app.sleep_schedule import decide_sleep_attempt
from src.app.numpy_sleep_decisions import describe_unguarded_numpy_sleep_decision
from src.app.toy_checkpoint import (
    ToyCheckpointStore,
    ToyRunnerCheckpoint,
    toy_config_digest,
    toy_data_digest,
    validate_toy_checkpoint,
)
from src.core.backprop_mlp import BackpropMLP, NUMPY_BACKPROP_LOSS_ID
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    NUMPY_CIRCADIAN_ENERGY_ID,
)
from src.core.neuron_adaptation import (
    LayerTraffic,
    NeuronAdaptationPolicy,
    NoOpNeuronAdaptationPolicy,
)
from src.core.predictive_coding import NUMPY_PC_ENERGY_ID, PredictiveCodingNetwork
from src.core.sleep_clocks import SleepEpochProgress
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.datasets import LabeledData, generate_two_cluster_dataset, split_training_validation

TOY_VALIDATION_PROTOCOL = "toy_validation_v1"
TOY_LEGACY_PROTOCOL = "toy_legacy_train_test_v0"
TOY_MODEL_ORDER = ("backprop", "predictive_coding", "circadian_predictive_coding")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configurable parameters for the baseline comparison."""

    sample_count: int = 400
    protocol_id: str = TOY_VALIDATION_PROTOCOL
    validation_fraction: float = 0.20
    noise_scale: float = 0.8
    hidden_dim: int = 12
    hidden_dims: tuple[int, ...] | None = None
    epoch_count: int = 160
    backprop_learning_rate: float = 0.12
    pc_learning_rate: float = 0.05
    pc_inference_steps: int = 25
    pc_inference_learning_rate: float = 0.2
    circadian_learning_rate: float = 0.05
    circadian_inference_steps: int = 25
    circadian_inference_learning_rate: float = 0.2
    circadian_sleep_interval: int = 40
    circadian_force_sleep: bool = True
    circadian_use_policy_for_sleep: bool = False
    circadian_config: CircadianConfig | None = None
    random_seed: int = 7
    model_order: tuple[str, ...] = TOY_MODEL_ORDER


@dataclass(frozen=True)
class ModelReport:
    """Result metrics for one model."""

    loss_history: list[float]
    training_metric_id: str
    test_accuracy: float
    traffic_by_layer: list[LayerTraffic]
    validation_accuracy: float | None = None


@dataclass(frozen=True)
class CircadianSleepSummary:
    """Sleep-event summary for the circadian model."""

    event_count: int
    total_splits: int
    total_prunes: int
    hidden_dim_start: int
    hidden_dim_end: int
    sleep_mode: str = "legacy"
    events: tuple[SleepEventTelemetry, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class ExperimentResult:
    """End-to-end comparison output."""

    backprop: ModelReport
    predictive_coding: ModelReport
    circadian_predictive_coding: ModelReport
    circadian_sleep: CircadianSleepSummary
    protocol_id: str
    split_hashes: dict[str, str]
    comparison_scope: NumpyComparisonScope
    training_order: tuple[str, ...] = TOY_MODEL_ORDER


@dataclass(frozen=True)
class _ToyTrainingOutcome:
    """Trained models and permitted development metrics, without final test data."""

    backprop_model: BackpropMLP
    predictive_model: PredictiveCodingNetwork
    circadian_model: CircadianPredictiveCodingNetwork
    losses: tuple[list[float], list[float], list[float]]
    traffic: tuple[list[LayerTraffic], list[LayerTraffic], list[LayerTraffic]]
    validation_accuracy: tuple[float | None, float | None, float | None]
    sleep: CircadianSleepSummary


def run_experiment(
    config: ExperimentConfig,
    adaptation_policy: NeuronAdaptationPolicy | None = None,
    *,
    checkpoint_store: ToyCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> ExperimentResult:
    """Train all three models on the same data and return comparable reports."""
    if (
        type(config.model_order) is not tuple
        or len(config.model_order) != len(TOY_MODEL_ORDER)
        or any(type(name) is not str for name in config.model_order)
        or set(config.model_order) != set(TOY_MODEL_ORDER)
    ):
        raise ValueError("model_order must be a permutation of the three toy models")
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("resume_from_checkpoint requires a checkpoint store")
    policy = adaptation_policy or NoOpNeuronAdaptationPolicy()
    # Why this: arbitrary external policies can carry state outside all three
    # models. The durable route has an explicit stateless policy boundary.
    if checkpoint_store is not None and type(policy) is not NoOpNeuronAdaptationPolicy:
        raise ValueError("toy checkpoint requires the stateless default adaptation policy")
    dataset = generate_two_cluster_dataset(
        sample_count=config.sample_count,
        noise_scale=config.noise_scale,
        seed=config.random_seed,
    )
    if config.protocol_id == TOY_VALIDATION_PROTOCOL:
        roles = split_training_validation(
            dataset,
            validation_fraction=config.validation_fraction,
            seed=config.random_seed + 17,
        )
        train_data = roles.train
        validation_data: LabeledData | None = roles.validation
        split_hashes = dict(roles.split_hashes)
    elif config.protocol_id == TOY_LEGACY_PROTOCOL:
        train_data = LabeledData(dataset.train_input, dataset.train_target)
        validation_data = None
        split_hashes = {}
    else:
        raise ValueError(f"Unknown toy benchmark protocol: {config.protocol_id}")

    if checkpoint_store is None:
        trained = _train_toy_models(
            config=config,
            policy=policy,
            train_data=train_data,
            validation_data=validation_data,
        )
    else:
        trained = _train_toy_models(
            config=config,
            policy=policy,
            train_data=train_data,
            validation_data=validation_data,
            checkpoint_store=checkpoint_store,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    final_test = (
        roles.test
        if config.protocol_id == TOY_VALIDATION_PROTOCOL
        else LabeledData(dataset.test_input, dataset.test_target)
    )

    backprop_accuracy = trained.backprop_model.compute_accuracy(final_test.input, final_test.target)
    predictive_coding_accuracy = trained.predictive_model.compute_accuracy(
        final_test.input, final_test.target
    )
    circadian_accuracy = trained.circadian_model.compute_accuracy(
        final_test.input, final_test.target
    )

    return ExperimentResult(
        backprop=ModelReport(
            loss_history=trained.losses[0],
            training_metric_id=NUMPY_BACKPROP_LOSS_ID,
            test_accuracy=backprop_accuracy,
            traffic_by_layer=trained.traffic[0],
            validation_accuracy=trained.validation_accuracy[0],
        ),
        predictive_coding=ModelReport(
            loss_history=trained.losses[1],
            training_metric_id=NUMPY_PC_ENERGY_ID,
            test_accuracy=predictive_coding_accuracy,
            traffic_by_layer=trained.traffic[1],
            validation_accuracy=trained.validation_accuracy[1],
        ),
        circadian_predictive_coding=ModelReport(
            loss_history=trained.losses[2],
            training_metric_id=NUMPY_CIRCADIAN_ENERGY_ID,
            test_accuracy=circadian_accuracy,
            traffic_by_layer=trained.traffic[2],
            validation_accuracy=trained.validation_accuracy[2],
        ),
        circadian_sleep=trained.sleep,
        protocol_id=config.protocol_id,
        split_hashes=split_hashes,
        comparison_scope=scope_for_hidden_dims(
            config.hidden_dims if config.hidden_dims is not None else (config.hidden_dim,)
        ),
        training_order=config.model_order,
    )


def _train_toy_models(
    *,
    config: ExperimentConfig,
    policy: NeuronAdaptationPolicy,
    train_data: LabeledData,
    validation_data: LabeledData | None,
    checkpoint_store: ToyCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> _ToyTrainingOutcome:
    """Train and inspect models with no reference to the final test role."""
    resolved_hidden_dims = list(config.hidden_dims) if config.hidden_dims is not None else None
    backprop_model = BackpropMLP(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=config.random_seed,
        hidden_dims=resolved_hidden_dims,
    )
    predictive_coding_model = PredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=config.random_seed + 1,
        hidden_dims=resolved_hidden_dims,
    )
    circadian_model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=config.random_seed + 2,
        circadian_config=config.circadian_config,
        hidden_dims=resolved_hidden_dims,
    )

    backprop_losses: list[float] = []
    predictive_coding_energies: list[float] = []
    circadian_energies: list[float] = []

    sleep_event_count = 0
    total_splits = 0
    total_prunes = 0
    hidden_dim_start = circadian_model.hidden_dim
    sleep_events: list[SleepEventTelemetry] = []

    data_digest = toy_data_digest(train_data, validation_data) if checkpoint_store else ""
    config_digest = toy_config_digest(config) if checkpoint_store else ""
    start_epoch = 1
    start_model_index = 0
    if resume_from_checkpoint:
        assert checkpoint_store is not None
        checkpoint = checkpoint_store.load()
        position = validate_toy_checkpoint(
            checkpoint,
            config_digest=config_digest,
            data_digest=data_digest,
            protocol_id=config.protocol_id,
            epoch_count=config.epoch_count,
            model_order=config.model_order,
            hidden_dims=tuple(resolved_hidden_dims or [config.hidden_dim]),
        )
        restore_circadian_checkpoint(
            circadian_model,
            checkpoint.combined,
            retry=None,
            protocol_id=config.protocol_id,
            config=circadian_model.config,
            data_digest=data_digest,
            expected_stage=position.stage,
        )
        backprop_model = deepcopy(checkpoint.backprop_model)
        predictive_coding_model = deepcopy(checkpoint.predictive_model)
        backprop_losses, predictive_coding_energies, circadian_energies = deepcopy(
            checkpoint.losses
        )
        sleep_event_count = checkpoint.sleep_event_count
        total_splits = checkpoint.total_splits
        total_prunes = checkpoint.total_prunes
        hidden_dim_start = checkpoint.hidden_dim_start
        sleep_events = list(checkpoint.sleep_events)
        if position.stage == "wake":
            start_epoch = position.completed_epoch + 1
            start_model_index = position.next_batch_index
        elif position.stage == "before_sleep":
            start_epoch = position.completed_epoch
            start_model_index = len(config.model_order)
        else:
            start_epoch = position.completed_epoch + 1

    for epoch_index in range(start_epoch, config.epoch_count + 1):
        model_start = start_model_index if epoch_index == start_epoch else 0
        for model_index in range(model_start, len(config.model_order)):
            model_name = config.model_order[model_index]
            if model_name == "backprop":
                backprop_step = backprop_model.train_epoch(
                    input_batch=train_data.input,
                    target_batch=train_data.target,
                    learning_rate=config.backprop_learning_rate,
                )
                backprop_losses.append(backprop_step.loss)
            elif model_name == "predictive_coding":
                predictive_coding_step = predictive_coding_model.train_epoch(
                    input_batch=train_data.input,
                    target_batch=train_data.target,
                    learning_rate=config.pc_learning_rate,
                    inference_steps=config.pc_inference_steps,
                    inference_learning_rate=config.pc_inference_learning_rate,
                )
                predictive_coding_energies.append(predictive_coding_step.energy)
            else:
                circadian_step = circadian_model.train_epoch(
                    input_batch=train_data.input,
                    target_batch=train_data.target,
                    learning_rate=config.circadian_learning_rate,
                    inference_steps=config.circadian_inference_steps,
                    inference_learning_rate=config.circadian_inference_learning_rate,
                )
                circadian_energies.append(circadian_step.energy)

            if checkpoint_store is not None and model_index + 1 < len(config.model_order):
                _save_toy_checkpoint(
                    checkpoint_store,
                    config,
                    config_digest,
                    data_digest,
                    backprop_model,
                    predictive_coding_model,
                    circadian_model,
                    (backprop_losses, predictive_coding_energies, circadian_energies),
                    sleep_event_count,
                    total_splits,
                    total_prunes,
                    hidden_dim_start,
                    sleep_events,
                    completed_epoch=epoch_index - 1,
                    stage="wake",
                    next_model_index=model_index + 1,
                )

        if checkpoint_store is not None and model_start < len(config.model_order):
            _save_toy_checkpoint(
                checkpoint_store,
                config,
                config_digest,
                data_digest,
                backprop_model,
                predictive_coding_model,
                circadian_model,
                (backprop_losses, predictive_coding_energies, circadian_energies),
                sleep_event_count,
                total_splits,
                total_prunes,
                hidden_dim_start,
                sleep_events,
                completed_epoch=epoch_index,
                stage="before_sleep",
                next_model_index=0,
            )

        # Why this: corrected component runs can attempt adaptive sleep on
        # any epoch; legacy runs keep their historical interval-only calls.
        adaptive_due = (
            circadian_model.config.sleep_mode == "components"
            and circadian_model.should_trigger_sleep()
        )
        decision = decide_sleep_attempt(
            sleep_mode=circadian_model.config.sleep_mode,
            completed_epochs=epoch_index,
            interval_epochs=config.circadian_sleep_interval,
            adaptive_due=adaptive_due,
            force_periodic=config.circadian_force_sleep,
        )
        if decision.attempted:
            sleep_policy = policy if config.circadian_use_policy_for_sleep else None
            attempt_started_at = perf_counter()
            sleep_result = circadian_model.sleep_event(
                adaptation_policy=sleep_policy,
                force_sleep=decision.force_sleep,
                epoch_progress=SleepEpochProgress(epoch_index, config.epoch_count),
            )
            sleep_events.append(
                describe_unguarded_numpy_sleep_decision(
                    circadian_model,
                    decision,
                    completed_epoch=epoch_index,
                    result=sleep_result,
                    attempt_seconds=perf_counter() - attempt_started_at,
                )
            )
            # Why this: legacy reports counted topology changes, while the
            # component route counts consolidation-only events as sleep.
            if circadian_model.config.sleep_mode == "legacy":
                performed_sleep = (
                    len(sleep_result.split_indices) > 0
                    or len(sleep_result.pruned_indices) > 0
                    or sleep_result.new_hidden_dim != sleep_result.old_hidden_dim
                )
            else:
                performed_sleep = sleep_result.performed
            if performed_sleep:
                sleep_event_count += 1
                total_splits += len(sleep_result.split_indices)
                total_prunes += len(sleep_result.pruned_indices)
        else:
            sleep_events.append(
                describe_unguarded_numpy_sleep_decision(
                    circadian_model, decision, completed_epoch=epoch_index
                )
            )

        if checkpoint_store is not None:
            _save_toy_checkpoint(
                checkpoint_store,
                config,
                config_digest,
                data_digest,
                backprop_model,
                predictive_coding_model,
                circadian_model,
                (backprop_losses, predictive_coding_energies, circadian_energies),
                sleep_event_count,
                total_splits,
                total_prunes,
                hidden_dim_start,
                sleep_events,
                completed_epoch=epoch_index,
                stage="after_sleep",
                next_model_index=0,
            )

    backprop_traffic = backprop_model.get_layer_traffic()
    predictive_coding_traffic = predictive_coding_model.get_layer_traffic()
    circadian_traffic = circadian_model.get_layer_traffic()

    backprop_model.apply_neuron_proposals(policy.propose(backprop_traffic))
    predictive_coding_model.apply_neuron_proposals(policy.propose(predictive_coding_traffic))
    circadian_model.apply_neuron_proposals(policy.propose(circadian_traffic))

    validation_accuracies: tuple[float | None, float | None, float | None] = (None, None, None)
    if validation_data is not None:
        validation_accuracies = (
            backprop_model.compute_accuracy(validation_data.input, validation_data.target),
            predictive_coding_model.compute_accuracy(validation_data.input, validation_data.target),
            circadian_model.compute_accuracy(validation_data.input, validation_data.target),
        )

    return _ToyTrainingOutcome(
        backprop_model=backprop_model,
        predictive_model=predictive_coding_model,
        circadian_model=circadian_model,
        losses=(backprop_losses, predictive_coding_energies, circadian_energies),
        traffic=(backprop_traffic, predictive_coding_traffic, circadian_traffic),
        validation_accuracy=validation_accuracies,
        sleep=CircadianSleepSummary(
            event_count=sleep_event_count,
            total_splits=total_splits,
            total_prunes=total_prunes,
            hidden_dim_start=hidden_dim_start,
            hidden_dim_end=circadian_model.hidden_dim,
            sleep_mode=circadian_model.config.sleep_mode,
            events=tuple(sleep_events),
        ),
    )


def _save_toy_checkpoint(
    store: ToyCheckpointStore,
    config: ExperimentConfig,
    config_digest: str,
    data_digest: str,
    backprop: BackpropMLP,
    predictive: PredictiveCodingNetwork,
    circadian: CircadianPredictiveCodingNetwork,
    losses: tuple[list[float], list[float], list[float]],
    sleep_event_count: int,
    total_splits: int,
    total_prunes: int,
    hidden_dim_start: int,
    sleep_events: list[SleepEventTelemetry],
    *,
    completed_epoch: int,
    stage: str,
    next_model_index: int,
) -> None:
    """Persist a fully matched cursor after a complete model update or sleep."""
    position = CircadianResumePosition(
        completed_epoch=completed_epoch,
        stage=stage,
        wake_batches=circadian.get_sleep_clocks().wake_batches,
        next_batch_index=next_model_index,
    )
    store.save(
        ToyRunnerCheckpoint(
            format_version=2,
            runner_config_digest=config_digest,
            data_digest=data_digest,
            backprop_model=deepcopy(backprop),
            predictive_model=deepcopy(predictive),
            losses=deepcopy(losses),
            sleep_event_count=sleep_event_count,
            total_splits=total_splits,
            total_prunes=total_prunes,
            hidden_dim_start=hidden_dim_start,
            sleep_events=tuple(sleep_events),
            combined=capture_circadian_checkpoint(
                circadian,
                retry=None,
                position=position,
                protocol_id=config.protocol_id,
                config=circadian.config,
                data_digest=data_digest,
            ),
        )
    )


def format_experiment_result(result: ExperimentResult) -> str:
    """Build a human-readable experiment summary."""
    bp_start = result.backprop.loss_history[0]
    bp_end = result.backprop.loss_history[-1]
    pc_start = result.predictive_coding.loss_history[0]
    pc_end = result.predictive_coding.loss_history[-1]
    cpc_start = result.circadian_predictive_coding.loss_history[0]
    cpc_end = result.circadian_predictive_coding.loss_history[-1]

    return "\n".join(
        [
            "Backprop vs Predictive Coding vs Circadian Predictive Coding",
            "------------------------------------------------------------",
            f"Protocol: {result.protocol_id}",
            f"Comparison scope: {result.comparison_scope.scope_id}. {result.comparison_scope.description}",
            (
                "Learning rules: "
                f"Backprop={result.comparison_scope.backprop_algorithm_id}, "
                f"PC={result.comparison_scope.predictive_algorithm_id}, "
                f"Circadian={result.comparison_scope.circadian_algorithm_id}"
            ),
            f"Training order: {result.training_order}",
            f"Split hashes: {result.split_hashes}",
            f"Backprop loss [{result.backprop.training_metric_id}]: {bp_start:.4f} -> {bp_end:.4f}",
            f"Backprop validation accuracy: {result.backprop.validation_accuracy}",
            f"Backprop test accuracy: {result.backprop.test_accuracy:.3f}",
            (
                "Predictive coding energy (training diagnostic "
                f"{result.predictive_coding.training_metric_id}): {pc_start:.4f} -> {pc_end:.4f}"
            ),
            f"Predictive coding validation accuracy: {result.predictive_coding.validation_accuracy}",
            f"Predictive coding test accuracy: {result.predictive_coding.test_accuracy:.3f}",
            (
                "Circadian predictive coding energy (training diagnostic "
                f"{result.circadian_predictive_coding.training_metric_id}): "
                f"{cpc_start:.4f} -> {cpc_end:.4f}"
            ),
            f"Circadian predictive coding validation accuracy: {result.circadian_predictive_coding.validation_accuracy}",
            f"Circadian predictive coding test accuracy: {result.circadian_predictive_coding.test_accuracy:.3f}",
            (
                "Circadian sleep: "
                f"mode={result.circadian_sleep.sleep_mode}, "
                f"events={result.circadian_sleep.event_count}, "
                f"splits={result.circadian_sleep.total_splits}, "
                f"prunes={result.circadian_sleep.total_prunes}, "
                f"hidden_dim={result.circadian_sleep.hidden_dim_start}"
                f"->{result.circadian_sleep.hidden_dim_end}"
            ),
            "",
            "Traffic snapshot (mean absolute activation):",
            "Backprop: "
            + _format_traffic_vector(result.backprop.traffic_by_layer[0].mean_abs_activation),
            "Predictive coding: "
            + _format_traffic_vector(
                result.predictive_coding.traffic_by_layer[0].mean_abs_activation
            ),
            "Circadian hidden: "
            + _format_traffic_vector(
                result.circadian_predictive_coding.traffic_by_layer[0].mean_abs_activation
            ),
            "Circadian chemical: "
            + _format_traffic_vector(
                result.circadian_predictive_coding.traffic_by_layer[1].mean_abs_activation
            ),
        ]
    )


def _format_traffic_vector(values: object) -> str:
    if hasattr(values, "tolist"):
        rounded = [f"{float(v):.3f}" for v in values.tolist()]
    else:
        rounded = [str(values)]
    return "[" + ", ".join(rounded) + "]"
