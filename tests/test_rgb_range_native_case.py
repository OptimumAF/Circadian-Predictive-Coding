"""Zero-model guards for the fixed range learner's native reconstruction boundary."""

from copy import deepcopy
from dataclasses import asdict
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra.maniskill_continuous import ACTION_UNITS


case = importlib.import_module("artifacts.runs.rgb-range-native-20261019.native_case")


def reference():
    """Synthetic metadata only; no accepted report or learner is opened or created."""
    report = json.loads(
        json.dumps(
            dict(
                complete=True,
                expected_standardization_checked=True,
                input_normalization=dict(
                    definition="pick_cube_train_prefix21_fixed_rgb_range_v1",
                    recipe=asdict(pick_cube_rgb_recipe()),
                    fitted_prefix_dimensions=21,
                    fixed_range_dimensions=768,
                    rgb_mean=0.5,
                    rgb_scale=0.5,
                    range_validation_rows=88,
                    terminal_range_validation_rows=3,
                ),
                policy=dict(
                    spec=dict(
                        observation_dim=789,
                        action_units=ACTION_UNITS,
                        lower_bounds=(-1.0,) * 4,
                        upper_bounds=(1.0,) * 4,
                    ),
                    hidden_dim=64,
                    seed=17,
                    learning_rate=0.01,
                    momentum=0.9,
                    training_role="train",
                    binary_gripper_axis=None,
                    observation_standardization=dict(
                        means=(0.0,) * 21 + (0.5,) * 768,
                        scales=(1.0,) * 21 + (0.5,) * 768,
                        fitted_rows=29,
                    ),
                ),
                progress=dict(
                    updates_completed=24,
                    update_attempts=24,
                    replay_examples_completed=0,
                    next_tick=48,
                ),
                settings=dict(
                    phase_updates=8,
                    phases=["warm", "new", "warm_return"],
                    max_training_updates=24,
                    promotion_calls=0,
                    simulator_calls=0,
                    probe_reads=0,
                    final_reads=0,
                    guard_reads=0,
                ),
                source_episodes=dict(
                    warm=dict(
                        episode_id="pick-cube-task-teacher-17", context="pick-cube", role="train"
                    ),
                    new=dict(
                        episode_id="pick-cube-task-teacher-19",
                        context="pick-cube-train-19",
                        role="train",
                    ),
                ),
                events=[dict(index=index) for index in range(24)],
                applied_updates=[dict(update_number=index + 1) for index in range(24)],
                checkpoints=[
                    dict(
                        updates=updates, training_examples=rows, state_exact=True, owners_exact=True
                    )
                    for updates, rows in ((0, 0), (8, 232), (16, 472), (24, 704))
                ],
                work=dict(
                    fit_calls=1,
                    fit_rows=29,
                    fit_feature_values=609,
                    fixed_range_dimensions=768,
                    optimizer_updates=24,
                    training_rows=704,
                    unique_source_rows=59,
                    repeated_training_rows=645,
                    replay_budget_rows=0,
                ),
            )
        )
    )
    report["metrics"] = {
        name: 0.0
        for name in (
            "warm_acquisition_gain",
            "new_acquisition_gain",
            "warm_retention_loss",
            "warm_return_recovery",
            "warm_return_deficit",
            "new_retention_loss",
        )
    }
    for point in report["checkpoints"]:
        point["contexts"] = {
            name: dict(
                episode_id=source["episode_id"],
                residual=dict(residuals=dict(overall=dict(normalized_half_mse=0.0))),
            )
            for name, source in report["source_episodes"].items()
        }
    rows = 0
    for index in range(24):
        phase = "warm" if index < 8 else "new" if index < 16 else "warm_return"
        context = "new" if phase == "new" else "warm"
        rows += 30 if context == "new" else 29
        receipt = dict(
            sample_id=f"mlp-batch-{index}",
            episode_id=report["source_episodes"][context]["episode_id"],
            event_id=f"mlp-label-{index}",
            actor_version="actor-0",
            learner_version="candidate-0",
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
            update_number=index + 1,
            diagnostic=dict(
                definition="continuous_bc_half_mean_active_axis_normalized_squared_error_v1",
                value=0.0,
            ),
        )
        report["applied_updates"][index] = receipt
        report["events"][index] = dict(
            index=index,
            phase=phase,
            context=context,
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            deferred_exact=True,
            update_attempted=True,
            applied=[deepcopy(receipt)],
            success_validated=True,
            candidate_updates=index + 1,
            candidate_training_examples=rows,
        )
    return report


def test_should_accept_exact_range_binding_without_changing_or_owning_report():
    report = reference()
    original = deepcopy(report)
    assert case.require_range_normalization(report) is None
    assert report == original
    assert report["policy"]["observation_standardization"]["fitted_rows"] == 29


@pytest.mark.parametrize(
    "name",
    [
        "definition",
        "recipe",
        "fitted_prefix_dimensions",
        "fixed_range_dimensions",
        "rgb_mean",
        "rgb_scale",
        "range_validation_rows",
        "terminal_range_validation_rows",
    ],
)
def test_should_refuse_each_missing_normalization_qualification(name):
    report = reference()
    del report["input_normalization"][name]
    with pytest.raises(ValueError, match="normalization"):
        case.require_range_normalization(report)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "definition",
        "prefix_float",
        "rgb_float",
        "mean",
        "scale",
        "training_validation_rows",
        "terminal_validation_rows",
        "recipe_extra",
        "recipe_camera",
        "recipe_equal_wrong_type",
        "complete_false",
        "complete_integer",
    ],
)
def test_should_refuse_changed_or_untyped_normalization_metadata(damage):
    report = reference()
    metadata = report["input_normalization"]
    if damage == "missing":
        del report["input_normalization"]
    elif damage == "extra":
        metadata["selected_floor"] = 0.5
    elif damage == "definition":
        metadata["definition"] = "empirical_rgb"
    elif damage == "prefix_float":
        metadata["fitted_prefix_dimensions"] = 21.0
    elif damage == "rgb_float":
        metadata["fixed_range_dimensions"] = 768.0
    elif damage == "mean":
        metadata["rgb_mean"] = 0.0
    elif damage == "scale":
        metadata["rgb_scale"] = 1.0
    elif damage == "training_validation_rows":
        metadata["range_validation_rows"] = 88.0
    elif damage == "terminal_validation_rows":
        metadata["terminal_range_validation_rows"] = True
    elif damage == "recipe_extra":
        metadata["recipe"]["scale"] = 0.5
    elif damage == "recipe_camera":
        metadata["recipe"]["camera"] = "other_camera"
    elif damage == "recipe_equal_wrong_type":
        metadata["recipe"]["feature_dim"] = 789.0
    else:
        report["complete"] = False if damage == "complete_false" else 1
    with pytest.raises(ValueError, match="normalization"):
        case.require_range_normalization(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_dim", 64.0),
        ("hidden_dim", 32),
        ("seed", True),
        ("seed", 19),
        ("learning_rate", 0.02),
        ("momentum", 0.8),
        ("training_role", "inner_guard"),
        ("training_role", "probe_train"),
        ("training_role", "probe_test"),
        ("training_role", "final_test"),
        ("binary_gripper_axis", 3),
    ],
)
def test_should_refuse_changed_default_policy_or_non_train_authority(field, value):
    report = reference()
    report["policy"][field] = value
    with pytest.raises(ValueError, match="policy"):
        case.require_range_normalization(report)


@pytest.mark.parametrize(
    "damage",
    [
        "extra_policy",
        "missing_policy",
        "wrong_dimension",
        "dimension_float",
        "wrong_units",
        "bounds",
        "equal_integer_bounds",
        "extra_spec",
    ],
)
def test_should_require_exact_policy_and_four_axis_native_controller_schema(damage):
    report = reference()
    policy, spec = report["policy"], report["policy"]["spec"]
    if damage == "extra_policy":
        policy["tuned_scale"] = 0.5
    elif damage == "missing_policy":
        del policy["binary_gripper_axis"]
    elif damage == "wrong_dimension":
        spec["observation_dim"] = 42
    elif damage == "dimension_float":
        spec["observation_dim"] = 789.0
    elif damage == "wrong_units":
        spec["action_units"][-1] = "metres"
    elif damage == "bounds":
        spec["upper_bounds"][-1] = 0.9
    elif damage == "equal_integer_bounds":
        spec["lower_bounds"] = [-1] * 4
    else:
        spec["control_mode"] = "pd_joint_pos"
    with pytest.raises(ValueError):
        case.require_range_normalization(report)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "means_short",
        "scales_short",
        "rows",
        "float_rows",
        "boolean_rows",
        "nonfinite_mean",
        "nonfinite_scale",
        "zero_scale",
        "negative_scale",
        "boolean_coordinate",
        "suffix_mean",
        "suffix_scale",
    ],
)
def test_should_refuse_invalid_full_statistics_or_changed_rgb_suffix(damage):
    report = reference()
    stats = report["policy"]["observation_standardization"]
    if damage == "missing":
        del stats["means"]
    elif damage == "extra":
        stats["refitted_on_b"] = False
    elif damage == "means_short":
        stats["means"].pop()
    elif damage == "scales_short":
        stats["scales"].pop()
    elif damage == "rows":
        stats["fitted_rows"] = 30
    elif damage == "float_rows":
        stats["fitted_rows"] = 29.0
    elif damage == "boolean_rows":
        stats["fitted_rows"] = True
    elif damage == "nonfinite_mean":
        stats["means"][0] = float("nan")
    elif damage == "nonfinite_scale":
        stats["scales"][0] = float("inf")
    elif damage == "zero_scale":
        stats["scales"][0] = 0.0
    elif damage == "negative_scale":
        stats["scales"][0] = -1.0
    elif damage == "boolean_coordinate":
        stats["means"][0] = False
    elif damage == "suffix_mean":
        stats["means"][21] = 0.4
    else:
        stats["scales"][-1] = 0.4
    with pytest.raises(ValueError):
        case.require_range_normalization(report)


@pytest.mark.parametrize(
    "damage",
    [
        "settings",
        "phase_order",
        "extra_settings",
        "progress",
        "progress_bool",
        "source_context",
        "source_role",
        "source_id",
        "fit_work",
        "work_float",
        "replay_work",
        "event_count",
        "event_container",
        "receipt_count",
        "checkpoint_count",
        "checkpoint_updates",
        "checkpoint_rows",
        "checkpoint_bool_counter",
        "checkpoint_state",
        "checkpoint_owner",
    ],
)
def test_should_refuse_changed_recipe_exposures_counters_work_or_sources_before_reconstruction(
    damage,
):
    report = reference()
    if damage == "settings":
        report["settings"]["max_training_updates"] = 25
    elif damage == "phase_order":
        report["settings"]["phases"] = ["warm", "warm_return", "new"]
    elif damage == "extra_settings":
        report["settings"]["selected_seed"] = 17
    elif damage == "progress":
        report["progress"]["updates_completed"] = 23
    elif damage == "progress_bool":
        report["progress"]["replay_examples_completed"] = False
    elif damage == "source_context":
        report["source_episodes"]["new"]["context"] = "pick-cube"
    elif damage == "source_role":
        report["source_episodes"]["new"]["role"] = "inner_guard"
    elif damage == "source_id":
        report["source_episodes"]["warm"]["episode_id"] = "changed-source"
    elif damage == "fit_work":
        report["work"]["fit_feature_values"] = 22881
    elif damage == "work_float":
        report["work"]["training_rows"] = 704.0
    elif damage == "replay_work":
        report["work"]["replay_budget_rows"] = 59
    elif damage == "event_count":
        report["events"].pop()
    elif damage == "event_container":
        report["events"] = tuple(report["events"])
    elif damage == "receipt_count":
        report["applied_updates"].append(dict(update_number=25))
    elif damage == "checkpoint_count":
        report["checkpoints"].pop()
    elif damage == "checkpoint_updates":
        report["checkpoints"][1]["updates"] = 7
    elif damage == "checkpoint_rows":
        report["checkpoints"][2]["training_examples"] = 473
    elif damage == "checkpoint_bool_counter":
        report["checkpoints"][0]["updates"] = False
    elif damage == "checkpoint_state":
        report["checkpoints"][2]["state_exact"] = 1
    else:
        report["checkpoints"][-1]["owners_exact"] = False
    with pytest.raises(ValueError):
        case.require_range_normalization(report)


@pytest.fixture
def saved_reference(monkeypatch, tmp_path):
    stage = tmp_path / "rgb-range-native-20261019"
    stage.mkdir()
    prior = tmp_path / "rgb-range-normalization-20261019"
    prior.mkdir()
    path = prior / "case.json"
    baseline = tmp_path / "visual-mlp-native-20261019"
    baseline.mkdir()
    (baseline / "rollout-audit.json").write_text(
        json.dumps(
            dict(
                complete=True,
                rollouts=[
                    dict(seed=seed, qualified=True, sparse_return=0.0, final_success=False)
                    for seed in (17, 19)
                ],
            )
        ),
        encoding="utf8",
    )
    monkeypatch.setattr(case, "stage", stage)
    return stage, path


def save_reference(path, report=None, *, complete=True):
    path.write_text(
        json.dumps(
            dict(complete=complete, range=reference() if report is None else report),
            allow_nan=False,
        ),
        encoding="utf8",
    )


@pytest.mark.parametrize(
    "damage",
    [
        "metadata",
        "statistics",
        "policy",
        "complete",
        "receipt",
        "objective",
        "event_tick",
        "endpoint_bool",
        "metric_bool",
    ],
)
def test_should_reject_bad_reference_before_teacher_io_reconstruction_or_native(
    saved_reference, monkeypatch, damage
):
    _, path = saved_reference
    report = reference()
    if damage == "metadata":
        report["input_normalization"]["rgb_scale"] = 1.0
    elif damage == "statistics":
        report["policy"]["observation_standardization"]["scales"][-1] = 1.0
    elif damage == "policy":
        report["policy"]["training_role"] = "probe_train"
    elif damage == "receipt":
        report["applied_updates"][0] = {}
    elif damage == "objective":
        report["applied_updates"][0]["diagnostic"]["definition"] = "wrong_loss"
    elif damage == "event_tick":
        report["events"][0]["arrived_at"] = 2
    elif damage == "endpoint_bool":
        report["checkpoints"][0]["contexts"]["warm"]["residual"]["residuals"]["overall"][
            "normalized_half_mse"
        ] = True
    elif damage == "metric_bool":
        report["metrics"]["warm_acquisition_gain"] = True
    else:
        report["complete"] = False
    save_reference(path, report)
    monkeypatch.setattr(
        case, "_read_teacher", lambda *_a, **_k: pytest.fail("bad reference read teacher")
    )
    monkeypatch.setattr(
        case,
        "run_continuous_learnability",
        lambda *_a, **_k: pytest.fail("bad reference reconstructed learner"),
    )
    monkeypatch.setattr(case, "rollout", lambda *_a, **_k: pytest.fail("bad reference ran native"))
    with pytest.raises(ValueError):
        case.run_case({})


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_should_refuse_invalid_receipt_loss_before_any_reconstruction(value):
    report = reference()
    report["applied_updates"][0]["diagnostic"]["value"] = value
    with pytest.raises(ValueError):
        case.require_range_normalization(report)


def fake_sources(monkeypatch):
    warm = SimpleNamespace(transitions=tuple(range(29)), spec=SimpleNamespace(observation_dim=789))
    new = SimpleNamespace(transitions=tuple(range(30)), spec=warm.spec)
    reads = []

    def read(path, seed, **options):
        reads.append((path.name, seed, options))
        return warm if seed == 17 else new

    monkeypatch.setattr(case, "_read_teacher", read)
    return warm, new, reads


def test_should_forward_validated_fixed_statistics_and_recipe_to_only_reconstruction(
    saved_reference, monkeypatch
):
    _, path = saved_reference
    save_reference(path)
    warm, new, reads = fake_sources(monkeypatch)
    calls = []

    class BoundaryReached(RuntimeError):
        pass

    def reconstruct(*args, **options):
        calls.append((args, options))
        raise BoundaryReached("zero-SGD reconstruction spy")

    monkeypatch.setattr(case, "run_continuous_learnability", reconstruct)
    monkeypatch.setattr(
        case, "rollout", lambda *_a, **_k: pytest.fail("native before reconstruction")
    )
    audit = {}
    with pytest.raises(BoundaryReached):
        case.run_case(audit)
    assert reads == [
        ("teacher-visual.json", 17, dict(visual_observation=True)),
        ("teacher-b.json", 19, dict(visual_observation=True)),
    ]
    assert len(calls) == 1 and calls[0][0] == (warm, new)
    stats = calls[0][1]["expected_standardization"]
    assert stats.fitted_rows == 29 and stats.means[21:] == stats.scales[21:] == (0.5,) * 768
    assert calls[0][1]["rgb_range_recipe"] == pick_cube_rgb_recipe()
    assert audit["training_attempted"] is True and "training_returned" not in audit


@pytest.mark.parametrize("damage", ["metadata", "statistics", "role", "typed_dimension"])
def test_should_gate_returned_report_before_quantitative_verifier_or_native(
    saved_reference, monkeypatch, damage
):
    stage, path = saved_reference
    save_reference(path)
    fake_sources(monkeypatch)
    reconstructed = reference()
    if damage == "metadata":
        reconstructed.pop("input_normalization")
    elif damage == "statistics":
        reconstructed["policy"]["observation_standardization"]["means"][-1] = 0.0
    elif damage == "role":
        reconstructed["policy"]["training_role"] = "inner_guard"
    else:
        reconstructed["policy"]["spec"]["observation_dim"] = 789.0
    result = SimpleNamespace(report=reconstructed, candidate=object())
    monkeypatch.setattr(case, "run_continuous_learnability", lambda *_a, **_k: result)
    monkeypatch.setattr(
        case,
        "verify_reproduction",
        lambda a, b: (
            None if a is b else pytest.fail("bad normalization reached38quantity verifier")
        ),
    )
    monkeypatch.setattr(
        case, "rollout", lambda *_a, **_k: pytest.fail("bad normalization ran native")
    )
    audit = {}
    with pytest.raises(ValueError):
        case.run_case(audit)
    assert audit["training_returned"] is True
    assert audit["training_progress"] == reconstructed["progress"]
    assert audit["training_applied_updates"] == reconstructed["applied_updates"]
    assert json.loads((stage / "training-report.json").read_text(encoding="utf8")) == reconstructed
    assert "reproduction_passed" not in audit


def test_should_reach_existing_quantitative_verifier_only_after_both_metadata_gates(
    saved_reference, monkeypatch
):
    _, path = saved_reference
    save_reference(path)
    fake_sources(monkeypatch)
    reconstructed, observed = reference(), []
    monkeypatch.setattr(
        case,
        "run_continuous_learnability",
        lambda *_a, **_k: SimpleNamespace(report=reconstructed, candidate=object()),
    )

    class VerifierReached(RuntimeError):
        pass

    def verify(actual, previous):
        if actual is previous:
            return
        observed.append((actual, previous))
        raise VerifierReached("zero-model quantitative boundary")

    monkeypatch.setattr(case, "verify_reproduction", verify)
    monkeypatch.setattr(
        case, "rollout", lambda *_a, **_k: pytest.fail("native before quantitative verification")
    )
    audit = {}
    with pytest.raises(VerifierReached):
        case.run_case(audit)
    assert len(observed) == 1
    assert observed[0][0] is reconstructed and observed[0][1] == reference()
    assert audit["training_returned"] is True and "reproduction_passed" not in audit


@pytest.mark.parametrize(
    "name",
    [
        "policy-rollout.json",
        "training-report.json",
        "rollout-audit.json",
        "a-rollout.json",
        "b-rollout.json",
    ],
)
def test_should_preserve_existing_evidence_before_reading_reference_or_training(
    saved_reference, monkeypatch, name
):
    stage, _ = saved_reference
    existing = stage / name
    existing.write_text("preserved original attempt", encoding="utf8")
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(stage / "policy-rollout.json")])
    monkeypatch.setattr(case, "run_case", lambda *_: pytest.fail("existing evidence admitted work"))
    with pytest.raises(FileExistsError):
        case.main()
    assert existing.read_text(encoding="utf8") == "preserved original attempt"


def test_should_reject_summary_alias_before_admitting_any_work(saved_reference, monkeypatch):
    stage, _ = saved_reference
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(stage / "a-rollout.json")])
    monkeypatch.setattr(case, "run_case", lambda *_: pytest.fail("summary alias admitted work"))
    with pytest.raises(ValueError, match="fixed summary"):
        case.main()
    assert list(stage.iterdir()) == []
