"""Generate a hardest-case training+inference dynamics GIF for README and docs site."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.backprop_mlp import BackpropMLP  # noqa: E402
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork  # noqa: E402
from src.core.predictive_coding import PredictiveCodingNetwork  # noqa: E402
from src.infra.datasets import DatasetSplit, generate_two_cluster_dataset_with_transform  # noqa: E402

Array = NDArray[np.float64]
DecisionMap = NDArray[np.float32]
LabelPrediction = NDArray[np.int8]

MODEL_ORDER = ["Backprop", "Predictive", "Circadian"]
MODEL_COLORS = {
    "Backprop": (46, 97, 173),
    "Predictive": (61, 157, 86),
    "Circadian": (189, 102, 36),
}


@dataclass(frozen=True)
class HardestModeConfig:
    seed: int = 7
    sample_count_phase_a: int = 500
    sample_count_phase_b: int = 500
    test_ratio: float = 0.25
    phase_b_train_fraction: float = 0.08
    hidden_dim: int = 8
    phase_a_epochs: int = 90
    phase_b_epochs: int = 120
    phase_a_noise: float = 0.8
    phase_b_noise: float = 1.2
    phase_b_rotation_degrees: float = 44.0
    phase_b_translation_x: float = 0.9
    phase_b_translation_y: float = -0.7
    sleep_interval_phase_a: int = 40
    sleep_interval_phase_b: int = 8
    snapshot_interval: int = 4
    decision_grid_size: int = 110
    latency_repeats: int = 10
    gif_duration_ms: int = 120


@dataclass(frozen=True)
class HardestModeSnapshot:
    epoch: int
    phase_name: str
    backprop_metric: float
    predictive_metric: float
    circadian_metric: float
    backprop_phase_b_accuracy: float
    predictive_phase_b_accuracy: float
    circadian_phase_b_accuracy: float
    backprop_latency_ms: float
    predictive_latency_ms: float
    circadian_latency_ms: float
    circadian_hidden_dim: int
    circadian_total_splits: int
    circadian_total_prunes: int
    decision_map: DecisionMap
    circadian_phase_b_predictions: LabelPrediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hardest-case train+inference dynamics GIF."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="docs/figures/hardest_mode_dynamics.gif",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--snapshot-interval", type=int, default=4)
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    return parser.parse_args()


def build_hardest_circadian_config() -> CircadianConfig:
    return CircadianConfig(
        use_reward_modulated_learning=False,
        split_threshold=0.25,
        prune_threshold=0.04,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        replay_steps=2,
        replay_memory_size=10,
        replay_learning_rate=0.04,
        replay_inference_steps=12,
        replay_inference_learning_rate=0.14,
    )


def sample_balanced_subset(
    input_batch: Array,
    target_batch: Array,
    fraction: float,
    seed: int,
) -> tuple[Array, Array]:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in (0, 1].")

    rng = np.random.default_rng(seed)
    total_count = input_batch.shape[0]
    subset_count = max(8, int(total_count * fraction))
    if subset_count >= total_count:
        return input_batch.copy(), target_batch.copy()

    target_flat = target_batch.reshape(-1)
    positive_indices = np.where(target_flat >= 0.5)[0]
    negative_indices = np.where(target_flat < 0.5)[0]
    if positive_indices.size == 0 or negative_indices.size == 0:
        picked = rng.choice(total_count, size=subset_count, replace=False)
        return input_batch[picked], target_batch[picked]

    positive_count = min(positive_indices.size, subset_count // 2)
    negative_count = min(negative_indices.size, subset_count - positive_count)
    if positive_count + negative_count < subset_count:
        remaining = subset_count - (positive_count + negative_count)
        if positive_indices.size - positive_count >= remaining:
            positive_count += remaining
        else:
            negative_count += remaining

    picked_positive = rng.choice(positive_indices, size=positive_count, replace=False)
    picked_negative = rng.choice(negative_indices, size=negative_count, replace=False)
    picked = np.concatenate([picked_positive, picked_negative])
    rng.shuffle(picked)
    return input_batch[picked], target_batch[picked]


def build_datasets(config: HardestModeConfig) -> tuple[DatasetSplit, DatasetSplit]:
    phase_a = generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_a,
        noise_scale=config.phase_a_noise,
        seed=config.seed,
        test_ratio=config.test_ratio,
    )
    phase_b_full = generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_b,
        noise_scale=config.phase_b_noise,
        seed=config.seed + 101,
        test_ratio=config.test_ratio,
        rotation_degrees=config.phase_b_rotation_degrees,
        translation=(config.phase_b_translation_x, config.phase_b_translation_y),
    )
    phase_b_train_input, phase_b_train_target = sample_balanced_subset(
        input_batch=phase_b_full.train_input,
        target_batch=phase_b_full.train_target,
        fraction=config.phase_b_train_fraction,
        seed=config.seed + 31,
    )
    phase_b = DatasetSplit(
        train_input=phase_b_train_input,
        train_target=phase_b_train_target,
        test_input=phase_b_full.test_input,
        test_target=phase_b_full.test_target,
    )
    return phase_a, phase_b


def create_decision_map(
    model: CircadianPredictiveCodingNetwork,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    grid_size: int,
) -> DecisionMap:
    x_values = np.linspace(x_bounds[0], x_bounds[1], grid_size)
    y_values = np.linspace(y_bounds[0], y_bounds[1], grid_size)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    probabilities = model.predict_proba(grid_points).reshape(grid_size, grid_size)
    return probabilities.astype(np.float32)


def measure_latency_ms(
    model_predictor: Callable[[Array], Array],
    input_batch: Array,
    repeats: int,
) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        model_predictor(input_batch)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / float(repeats)


def should_snapshot(epoch: int, total_epochs: int, phase_a_epochs: int, interval: int) -> bool:
    if epoch == 1 or epoch == phase_a_epochs or epoch == total_epochs:
        return True
    return (epoch % interval) == 0


def compute_bounds(phase_a: DatasetSplit, phase_b: DatasetSplit) -> tuple[tuple[float, float], tuple[float, float]]:
    joined = np.vstack(
        [
            phase_a.train_input,
            phase_a.test_input,
            phase_b.train_input,
            phase_b.test_input,
        ]
    )
    x_min = float(np.min(joined[:, 0])) - 0.4
    x_max = float(np.max(joined[:, 0])) + 0.4
    y_min = float(np.min(joined[:, 1])) - 0.4
    y_max = float(np.max(joined[:, 1])) + 0.4
    return (x_min, x_max), (y_min, y_max)


def collect_hardest_mode_snapshots(
    config: HardestModeConfig,
) -> tuple[list[HardestModeSnapshot], Array, Array, tuple[float, float], tuple[float, float]]:
    phase_a, phase_b = build_datasets(config)
    x_bounds, y_bounds = compute_bounds(phase_a, phase_b)

    backprop = BackpropMLP(input_dim=2, hidden_dim=config.hidden_dim, seed=config.seed)
    predictive = PredictiveCodingNetwork(input_dim=2, hidden_dim=config.hidden_dim, seed=config.seed + 1)
    circadian = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=config.seed + 2,
        circadian_config=build_hardest_circadian_config(),
    )

    snapshots: list[HardestModeSnapshot] = []
    total_splits = 0
    total_prunes = 0
    total_epochs = config.phase_a_epochs + config.phase_b_epochs

    for epoch in range(1, total_epochs + 1):
        in_phase_b = epoch > config.phase_a_epochs
        active_split = phase_b if in_phase_b else phase_a
        phase_name = "Phase B (Hard Drift)" if in_phase_b else "Phase A (Base)"

        backprop_step = backprop.train_epoch(
            input_batch=active_split.train_input,
            target_batch=active_split.train_target,
            learning_rate=0.12,
        )
        predictive_step = predictive.train_epoch(
            input_batch=active_split.train_input,
            target_batch=active_split.train_target,
            learning_rate=0.05,
            inference_steps=25,
            inference_learning_rate=0.2,
        )
        circadian_step = circadian.train_epoch(
            input_batch=active_split.train_input,
            target_batch=active_split.train_target,
            learning_rate=0.05,
            inference_steps=25,
            inference_learning_rate=0.2,
        )

        sleep_interval = (
            config.sleep_interval_phase_b if in_phase_b else config.sleep_interval_phase_a
        )
        if sleep_interval > 0:
            phase_epoch = epoch - config.phase_a_epochs if in_phase_b else epoch
            if (phase_epoch % sleep_interval) == 0:
                sleep_result = circadian.sleep_event(
                    force_sleep=True,
                    current_step=epoch,
                    total_steps=total_epochs,
                )
                total_splits += len(sleep_result.split_indices)
                total_prunes += len(sleep_result.pruned_indices)

        if not should_snapshot(
            epoch=epoch,
            total_epochs=total_epochs,
            phase_a_epochs=config.phase_a_epochs,
            interval=config.snapshot_interval,
        ):
            continue

        backprop_acc = backprop.compute_accuracy(phase_b.test_input, phase_b.test_target)
        predictive_acc = predictive.compute_accuracy(phase_b.test_input, phase_b.test_target)
        circadian_acc = circadian.compute_accuracy(phase_b.test_input, phase_b.test_target)
        backprop_latency = measure_latency_ms(
            model_predictor=backprop.predict_proba,
            input_batch=phase_b.test_input,
            repeats=config.latency_repeats,
        )
        predictive_latency = measure_latency_ms(
            model_predictor=predictive.predict_proba,
            input_batch=phase_b.test_input,
            repeats=config.latency_repeats,
        )
        circadian_latency = measure_latency_ms(
            model_predictor=circadian.predict_proba,
            input_batch=phase_b.test_input,
            repeats=config.latency_repeats,
        )

        decision_map = create_decision_map(
            model=circadian,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            grid_size=config.decision_grid_size,
        )
        predictions = circadian.predict_label(phase_b.test_input).reshape(-1).astype(np.int8)

        snapshots.append(
            HardestModeSnapshot(
                epoch=epoch,
                phase_name=phase_name,
                backprop_metric=backprop_step.loss,
                predictive_metric=predictive_step.energy,
                circadian_metric=circadian_step.energy,
                backprop_phase_b_accuracy=backprop_acc,
                predictive_phase_b_accuracy=predictive_acc,
                circadian_phase_b_accuracy=circadian_acc,
                backprop_latency_ms=backprop_latency,
                predictive_latency_ms=predictive_latency,
                circadian_latency_ms=circadian_latency,
                circadian_hidden_dim=circadian.hidden_dim,
                circadian_total_splits=total_splits,
                circadian_total_prunes=total_prunes,
                decision_map=decision_map,
                circadian_phase_b_predictions=predictions,
            )
        )

    return snapshots, phase_b.test_input, phase_b.test_target, x_bounds, y_bounds


def normalize_series(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    title: str,
    panel_box: tuple[int, int, int, int],
    series: dict[str, list[float]],
    frame_index: int,
    phase_b_start_index: int,
    total_frame_count: int,
    y_min: float,
    y_max: float,
    y_label: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = panel_box
    draw.rounded_rectangle([(left, top), (right, bottom)], radius=10, fill=(255, 255, 255), outline=(214, 220, 230), width=2)
    draw.text((left + 10, top + 8), title, fill=(20, 20, 26), font=font)
    chart_left = left + 48
    chart_right = right - 16
    chart_top = top + 26
    chart_bottom = bottom - 26
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top
    draw.rectangle([(chart_left, chart_top), (chart_right, chart_bottom)], outline=(220, 226, 236), width=1)

    for tick in range(5):
        ratio = tick / 4.0
        y = chart_bottom - int(ratio * chart_height)
        value = y_min + ratio * (y_max - y_min)
        draw.line([(chart_left, y), (chart_right, y)], fill=(236, 240, 246), width=1)
        draw.text((chart_left - 34, y - 6), f"{value:.2f}", fill=(108, 114, 124), font=font)

    if total_frame_count > 0:
        x_phase = chart_left + int((phase_b_start_index / total_frame_count) * chart_width)
        y = chart_top
        while y < chart_bottom:
            draw.line([(x_phase, y), (x_phase, min(y + 8, chart_bottom))], fill=(168, 171, 182), width=1)
            y += 14

    for model_name in MODEL_ORDER:
        values = series[model_name]
        points: list[tuple[int, int]] = []
        for idx in range(frame_index + 1):
            x = chart_left + int((idx / max(frame_index, 1)) * chart_width)
            normalized = (values[idx] - y_min) / max(y_max - y_min, 1e-8)
            y = chart_bottom - int(normalized * chart_height)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=MODEL_COLORS[model_name], width=3)
        if points:
            px, py = points[-1]
            draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=MODEL_COLORS[model_name], outline=(28, 28, 34))

    draw.text((chart_left, chart_bottom + 8), y_label, fill=(105, 110, 120), font=font)


def decision_map_to_image(decision_map: DecisionMap, width: int, height: int) -> Image.Image:
    prob = np.flipud(decision_map)
    red = (prob * 220.0 + 28.0).astype(np.uint8)
    blue = ((1.0 - prob) * 220.0 + 28.0).astype(np.uint8)
    green = (74.0 + 52.0 * (1.0 - np.abs(prob - 0.5) * 2.0)).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb).resize((width, height), resample=Image.Resampling.BILINEAR)


def map_point_to_panel(
    point: tuple[float, float],
    panel_box: tuple[int, int, int, int],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[int, int]:
    left, top, right, bottom = panel_box
    x_ratio = (point[0] - x_bounds[0]) / max(x_bounds[1] - x_bounds[0], 1e-8)
    y_ratio = (point[1] - y_bounds[0]) / max(y_bounds[1] - y_bounds[0], 1e-8)
    x = left + int(np.clip(x_ratio, 0.0, 1.0) * (right - left))
    y = bottom - int(np.clip(y_ratio, 0.0, 1.0) * (bottom - top))
    return x, y


def render_hardest_mode_gif(
    snapshots: list[HardestModeSnapshot],
    phase_b_test_input: Array,
    phase_b_test_target: Array,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    output_path: Path,
    frame_duration_ms: int,
) -> None:
    if not snapshots:
        raise ValueError("snapshots cannot be empty.")

    width = 1180
    height = 680
    training_panel = (30, 90, 620, 350)
    accuracy_panel = (30, 370, 620, 640)
    inference_panel = (650, 90, 1148, 610)
    font = ImageFont.load_default()

    normalized_objective = {
        "Backprop": normalize_series([snapshot.backprop_metric for snapshot in snapshots]),
        "Predictive": normalize_series([snapshot.predictive_metric for snapshot in snapshots]),
        "Circadian": normalize_series([snapshot.circadian_metric for snapshot in snapshots]),
    }
    inference_accuracy = {
        "Backprop": [snapshot.backprop_phase_b_accuracy for snapshot in snapshots],
        "Predictive": [snapshot.predictive_phase_b_accuracy for snapshot in snapshots],
        "Circadian": [snapshot.circadian_phase_b_accuracy for snapshot in snapshots],
    }
    phase_b_start_index = 0
    for index, snapshot in enumerate(snapshots):
        if "Phase B" in snapshot.phase_name:
            phase_b_start_index = index
            break

    phase_b_target = phase_b_test_target.reshape(-1)
    frames: list[Image.Image] = []
    for frame_index, snapshot in enumerate(snapshots):
        image = Image.new("RGB", (width, height), (244, 248, 253))
        draw = ImageDraw.Draw(image)
        draw.text((30, 20), "Hardest-Case Dynamics: Training + Inference", fill=(18, 20, 26), font=font)
        draw.text(
            (30, 40),
            f"Epoch {snapshot.epoch} | {snapshot.phase_name} | Circadian hidden={snapshot.circadian_hidden_dim} | splits={snapshot.circadian_total_splits}",
            fill=(78, 84, 95),
            font=font,
        )

        draw_line_panel(
            draw=draw,
            title="Training Signal (normalized objective)",
            panel_box=training_panel,
            series=normalized_objective,
            frame_index=frame_index,
            phase_b_start_index=phase_b_start_index,
            total_frame_count=max(len(snapshots) - 1, 1),
            y_min=0.0,
            y_max=1.0,
            y_label="Normalized objective value",
            font=font,
        )
        draw_line_panel(
            draw=draw,
            title="Inference Accuracy on Phase-B Test Set",
            panel_box=accuracy_panel,
            series=inference_accuracy,
            frame_index=frame_index,
            phase_b_start_index=phase_b_start_index,
            total_frame_count=max(len(snapshots) - 1, 1),
            y_min=0.0,
            y_max=1.0,
            y_label="Accuracy",
            font=font,
        )

        draw.rounded_rectangle(
            [(inference_panel[0], inference_panel[1]), (inference_panel[2], inference_panel[3])],
            radius=10,
            fill=(255, 255, 255),
            outline=(214, 220, 230),
            width=2,
        )
        draw.text(
            (inference_panel[0] + 10, inference_panel[1] + 8),
            "Inference View (Circadian decision map on Phase-B test domain)",
            fill=(20, 20, 26),
            font=font,
        )
        map_left = inference_panel[0] + 16
        map_top = inference_panel[1] + 30
        map_right = inference_panel[2] - 16
        map_bottom = inference_panel[1] + 450
        decision_image = decision_map_to_image(
            decision_map=snapshot.decision_map,
            width=map_right - map_left,
            height=map_bottom - map_top,
        )
        image.paste(decision_image, (map_left, map_top))
        draw.rectangle([(map_left, map_top), (map_right, map_bottom)], outline=(215, 220, 228), width=1)

        for index in range(phase_b_test_input.shape[0]):
            point = (float(phase_b_test_input[index, 0]), float(phase_b_test_input[index, 1]))
            px, py = map_point_to_panel(
                point=point,
                panel_box=(map_left, map_top, map_right, map_bottom),
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            true_label = int(phase_b_target[index] >= 0.5)
            predicted_label = int(snapshot.circadian_phase_b_predictions[index] >= 0.5)
            point_fill = (250, 250, 250) if true_label == 0 else (22, 22, 22)
            outline = (42, 182, 78) if true_label == predicted_label else (214, 68, 58)
            draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=point_fill, outline=outline, width=1)

        legend_y = map_bottom + 10
        draw.text((map_left, legend_y), "Inference metrics (Phase-B test):", fill=(30, 32, 38), font=font)
        draw.text(
            (map_left, legend_y + 16),
            (
                f"Backprop acc={snapshot.backprop_phase_b_accuracy:.3f}, latency={snapshot.backprop_latency_ms:.3f} ms | "
                f"Predictive acc={snapshot.predictive_phase_b_accuracy:.3f}, latency={snapshot.predictive_latency_ms:.3f} ms | "
                f"Circadian acc={snapshot.circadian_phase_b_accuracy:.3f}, latency={snapshot.circadian_latency_ms:.3f} ms"
            ),
            fill=(80, 86, 96),
            font=font,
        )
        draw.text(
            (30, height - 22),
            "Green outline=correct inference, Red outline=incorrect inference (circadian).",
            fill=(100, 106, 116),
            font=font,
        )
        frames.append(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame, *other_frames = frames
    first_frame.save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=other_frames,
        duration=frame_duration_ms,
        loop=0,
    )


def main() -> None:
    args = parse_args()
    config = HardestModeConfig(
        seed=args.seed,
        snapshot_interval=args.snapshot_interval,
        gif_duration_ms=args.gif_duration_ms,
    )
    snapshots, phase_b_test_input, phase_b_test_target, x_bounds, y_bounds = collect_hardest_mode_snapshots(config)
    output_path = Path(args.output_path)
    render_hardest_mode_gif(
        snapshots=snapshots,
        phase_b_test_input=phase_b_test_input,
        phase_b_test_target=phase_b_test_target,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        output_path=output_path,
        frame_duration_ms=config.gif_duration_ms,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
