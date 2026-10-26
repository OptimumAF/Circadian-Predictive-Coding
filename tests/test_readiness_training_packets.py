"""Fixed information-balanced packets preserve original causal TRAIN rows.

Two cached BP D12/H2/A4 phase1 pipelines, balanced and unchanged uniform A:
2 initializers64 normal values;6 SGD90rows,0 fits/serving/native. Including
read-only checkpoints:70 forwards730rows/22 backwards250rows/836 gradient
values6688B/54 normalizations570rows6840coords/33360 selected dense MAC.
Every other case uses literal records, early refusal or cached owned reports.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_continuous_robot_learning as robot
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.core import pick_cube_action_history as history
from src.core import readiness_training_packets as packets
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from test_bp_binary_gripper_robot_learning import assert_state, raw


RECIPE = PickCubeTaskSpaceHistoryRecipe()
STAGES = ("align", "descend", "grasp", "lift", "settle")
# Recorded order deliberately differs from the stage grouping: packet rows must
# carry the original causal tail, not a new trajectory stitched in stage order.
WARM_STAGES = (
    "align",
    "lift",
    "descend",
    "align",
    "lift",
    "grasp",
    "descend",
    "lift",
    "align",
    "descend",
    "lift",
    "settle",
    "align",
    "lift",
    "descend",
)
EXPECTED = (
    (0, 3, 8, 2, 6, 9, 5, 5, 5, 1, 4, 7, 11, 11, 11),
    (12, 0, 3, 14, 2, 6, 5, 5, 5, 10, 13, 1, 11, 11, 11),
    (8, 12, 0, 9, 14, 2, 5, 5, 5, 4, 7, 10, 11, 11, 11),
)
UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")


def command(stage):
    if stage in ("grasp", "settle"):
        return (0.0, 0.0, 0.0, -1.0)
    return (0.25, -0.5, 0.125, 1.0 if stage in ("align", "descend") else -1.0)


def binding(seed=17, labels=WARM_STAGES, *, encoded=True):
    source = ContinuousEpisode(
        f"pick-cube-readiness-teacher-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        ContinuousActionSpec(42, UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(
                index,
                raw(index, float(seed == 19) / 8),
                command(stage),
                float(index == len(labels) - 1),
                raw(index + 1, float(seed == 19) / 8),
                index == len(labels) - 1,
                False,
            )
            for index, stage in enumerate(labels)
        ),
    )
    return ReadinessStageBinding(
        history.action_history_training_episode(source, RECIPE) if encoded else source, labels
    )


def pair():
    return binding(), binding(19, STAGES)


def forbidden(*args, **options):
    pytest.fail("stage-balanced refusal/stub admitted extra IO, fit or model work")


def test_should_cycle_recorded_rows_in_fixed_three_per_stage_packets_without_reordering_source():
    original = binding()
    before = asdict(original)
    plan = packets.prepare_readiness_packets(original, 3)
    assert plan.indices == EXPECTED
    assert asdict(original) == before
    assert plan.binding.episode == original.episode
    for selected in plan.indices:
        assert tuple(original.stages[index] for index in selected) == tuple(
            stage for stage in STAGES for _ in range(3)
        )
        # A repeated rare-stage row retains its actual preceding command, which
        # generally differs from the preceding selected row's command.
        assert selected[6:9] == (5, 5, 5)
        assert all(
            plan.binding.episode.transitions[index].observation[-5:]
            == original.episode.transitions[index].observation[-5:]
            for index in selected
        )


@pytest.mark.parametrize("updates,unique", [(1, 11), (3, 15)])
def test_should_report_actual_unique_rows_and_repeated_exposure_separately(updates, unique):
    metadata = packets.readiness_packet_metadata(
        packets.prepare_readiness_packets(binding(), updates)
    )
    assert metadata["definition"] == "readiness_stage_balanced_a15_packets_v1"
    assert metadata["source_rows"] == metadata["rows_per_packet"] == 15
    assert metadata["source_stages"] == WARM_STAGES and metadata["stage_order"] == STAGES
    assert metadata["stage_source_rows"] == dict(align=4, descend=4, grasp=1, lift=5, settle=1)
    assert metadata["rows_per_stage"] == 3 and metadata["planned_packets"] == updates
    assert metadata["unique_training_rows_actual"] == unique
    assert metadata["repeated_training_rows"] == 15 * updates - unique
    assert metadata["planned_stage_exposures"] == dict.fromkeys(STAGES, 3 * updates)
    assert metadata["retained_index_numeric_bytes"] == 15 * updates * 8
    assert metadata["role"] == "train" and metadata["context"] == "pick-cube"
    assert metadata["source_episode_id"] == "pick-cube-readiness-teacher-17"


def test_should_detach_frozen_plan_binding_and_metadata_and_revalidate_corruption():
    source = binding()
    plan = packets.prepare_readiness_packets(source, 3)
    metadata = packets.readiness_packet_metadata(plan)
    with pytest.raises(FrozenInstanceError):
        setattr(plan, "indices", ())
    object.__setattr__(source.episode.transitions[0], "reward", 99.0)
    assert plan.binding.episode.transitions[0].reward == 0.0
    metadata["stage_source_rows"]["align"] = 99
    assert packets.readiness_packet_metadata(plan)["stage_source_rows"]["align"] == 4
    object.__setattr__(plan, "indices", (tuple(reversed(EXPECTED[0])),))
    with pytest.raises(ValueError):
        packets.readiness_packet_metadata(plan)


@pytest.mark.parametrize("updates", [True, 3.0, 0, 385, 2**100])
def test_should_refuse_noninteger_or_unbounded_update_counts_without_packet_construction(updates):
    with pytest.raises(ValueError):
        packets.prepare_readiness_packets(binding(), updates)


@pytest.mark.parametrize("fault", ["role", "new_context", "short", "missing_stage"])
def test_should_require_complete_warm_train_fifteen_rows_with_all_five_recorded_stages(fault):
    source = binding()
    if fault == "role":
        object.__setattr__(source.episode, "role", "probe_train")
    elif fault == "new_context":
        source = binding(19)
    elif fault == "short":
        source = binding(labels=WARM_STAGES[:-1])
    else:
        source = binding(
            labels=tuple("lift" if stage == "settle" else stage for stage in WARM_STAGES)
        )
    with pytest.raises(ValueError):
        packets.prepare_readiness_packets(source, 3)


@pytest.mark.parametrize("fault", ["outer_list", "inner_list", "bool", "range", "rotation"])
def test_should_reject_aliases_and_forged_row_indices_instead_of_silently_repairing_them(fault):
    indices: Any = EXPECTED
    if fault == "outer_list":
        indices = list(indices)
    elif fault == "inner_list":
        indices = (list(indices[0]), *indices[1:])
    else:
        first = list(indices[0])
        first[0] = False if fault == "bool" else 15 if fault == "range" else 3
        indices = (tuple(first), *indices[1:])
    with pytest.raises(ValueError):
        packets.ReadinessPacketPlan(binding(), indices)


@pytest.fixture(scope="module")
def completed():
    records: dict[str, Any] = {}
    warm, new = pair()
    policy_factory = demo.ContinuousBCPolicy
    runtime_factory = demo.ActorShadowRuntime
    original_train = bp.NumpyContinuousBCLearner.train_batch
    original_apply = demo._apply_packet
    current = [""]

    def policy(*args, **options):
        options["hidden_dim"] = 2
        return policy_factory(*args, **options)

    def runtime(owner, **options):
        value = runtime_factory(owner, **options)
        records[current[0]].update(
            source=owner, initial=owner.snapshot_state(), runtime=value, budget=options["budget"]
        )
        return value

    def train(owner, features, targets):
        records[current[0]]["trains"].append((features.copy(), targets.copy()))
        return original_train(owner, features, targets)

    def apply(runtime, clock, budget, episode, arrays, index, expected_rows, event):
        records[current[0]]["packets"].append(
            (asdict(episode), arrays[0].copy(), arrays[1].copy(), arrays)
        )
        return original_apply(runtime, clock, budget, episode, arrays, index, expected_rows, event)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "ContinuousBCPolicy", policy)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", train)
        patch.setattr(demo, "_apply_packet", apply)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        for mode in ("balanced", "default"):
            current[0] = mode
            records[mode] = dict(trains=[], packets=[])
            options: dict[str, Any] = dict(
                phase_updates=1,
                a_only_control=True,
                action_history_recipe=RECIPE,
                diagnostic_stage_bindings=(warm, new),
            )
            if mode == "balanced":
                options["stage_balanced_packets"] = True
            result = demo.run_continuous_learnability(warm.episode, new.episode, **options)
            assert result.report["complete"] is True and result.candidate is not None
            records[mode]["result"] = result
    return dict(warm=warm, new=new, records=records)


def test_should_train_only_the_fifteen_selected_original_a_rows_and_targets_at_each_arrival(
    completed,
):
    warm = completed["warm"]
    record = completed["records"]["balanced"]
    report = record["result"].report
    assert len(record["trains"]) == len(record["packets"]) == 3
    for index, (indices, training, packet, event, receipt) in enumerate(
        zip(
            EXPECTED,
            record["trains"],
            record["packets"],
            report["events"],
            report["applied_updates"],
            strict=True,
        )
    ):
        expected_x = [list(warm.episode.transitions[i].observation) for i in indices]
        expected_y = [list(warm.episode.transitions[i].action) for i in indices]
        assert training[0].tolist() == packet[1].tolist() == expected_x
        assert training[1].tolist() == packet[2].tolist() == expected_y
        assert packet[0] == asdict(warm.episode)
        assert event["training_row_indices"] == list(indices)
        assert event["deferred_exact"] is event["success_validated"] is True
        assert receipt["episode_id"] == warm.episode.episode_id
        assert receipt["observed_at"] == 2 * index
        assert receipt["arrived_at"] == receipt["applied_at"] == 2 * index + 1
        assert event["candidate_training_examples"] == 15 * (index + 1)
    final = record["result"].candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (3, 45)
    assert_state(record["initial"], record["source"].snapshot_state())
    assert_state(record["initial"], record["runtime"].actor.snapshot_state().state)
    assert_state(final, record["runtime"].candidate_snapshot().state)
    assert (
        record["budget"].updates_completed == 3 and record["budget"].replay_examples_completed == 0
    )


def test_should_keep_b_read_only_and_report_selection_costs_separately_from_unchanged_ml_costs(
    completed,
):
    report = completed["records"]["balanced"]["result"].report
    warm, new = completed["warm"], completed["new"]
    assert report["training_packets"] == packets.readiness_packet_metadata(
        packets.prepare_readiness_packets(warm, 3)
    )
    assert report["settings"]["stage_balanced_packets"] is True
    work = report["work"]
    assert work["training_rows"] == 45 and work["optimizer_updates"] == 3
    assert work["unique_source_rows"] == 20 and work["unique_training_rows"] == 15
    assert work["repeated_training_rows"] == 30
    assert work["training_rows_by_context"] == dict(warm=45, new=0)
    assert work["diagnostic_rows_by_context"] == dict(warm=60, new=20)
    assert work["fit_calls"] == work["fit_rows"] == work["fit_feature_values"] == 0
    assert work["forward_passes"] == 35 and work["forward_rows"] == 365
    assert work["backward_passes"] == 11 and work["backward_rows"] == 125
    assert work["gradient_values"] == 418 and work["dense_multiply_accumulates"] == 16680
    assert type(work["stage_balanced_packets"]) is dict
    selection = work["stage_balanced_packets"]
    assert selection["source_index_selections"] == 45
    assert selection["retained_index_numeric_bytes"] == 360
    assert selection["gathered_packets"] == 3
    assert selection["gathered_feature_values"] == 540
    assert selection["gathered_target_values"] == 180
    assert selection["gathered_numeric_bytes"] == 5760
    assert len(report["checkpoints"]) == 4
    for checkpoint in report["checkpoints"]:
        assert checkpoint["contexts"]["new"]["episode_id"] == new.episode.episode_id
        summaries = checkpoint["contexts"]["new"]["residual"]["residuals"]
        assert summaries["overall"]["rows"] == 5
        assert tuple(group["label"] for group in summaries["stages"]) == STAGES
    assert asdict(warm) == asdict(binding()) and asdict(new) == asdict(binding(19, STAGES))


def test_should_keep_omitted_option_uniform_original_arrays_and_no_new_default_fields(completed):
    record, warm = completed["records"]["default"], completed["warm"]
    report = record["result"].report
    assert "training_packets" not in report
    assert "stage_balanced_packets" not in report["settings"]
    assert "stage_balanced_packets" not in report["work"]
    for features, targets in record["trains"]:
        assert features.tolist() == [list(row.observation) for row in warm.episode.transitions]
        assert targets.tolist() == [list(row.action) for row in warm.episode.transitions]
    assert all("training_row_indices" not in event for event in report["events"])
    assert all(packet[3] is record["packets"][0][3] for packet in record["packets"])
    balanced = completed["records"]["balanced"]["result"].report
    assert report["progress"] == balanced["progress"]
    assert report["work"]["training_rows"] == balanced["work"]["training_rows"]
    assert report["checkpoints"][0] == balanced["checkpoints"][0]


@pytest.mark.parametrize(
    "fault", ["metadata", "indices", "bool_index", "receipt", "unrequested", "clock", "work"]
)
def test_should_refuse_forged_selected_report_before_native_serving_from_cached_paid_result(
    completed, monkeypatch, fault
):
    warm, new = completed["warm"], completed["new"]
    original = completed["records"]["balanced"]["result"]
    admitted, state = robot._admit_candidate(
        original,
        warm.episode,
        new.episode,
        "bp",
        1,
        a_only_control=True,
        action_history_recipe=RECIPE,
        stage_balanced_packets=True,
        diagnostic_stage_bindings=(warm, new),
    )
    assert admitted is original.candidate
    assert_state(state, original.candidate.snapshot_state())
    report = deepcopy(original.report)
    selected = True
    if fault == "metadata":
        report["training_packets"]["planned_stage_exposures"]["grasp"] = 8
    elif fault in ("indices", "bool_index"):
        report["events"][0]["training_row_indices"][0] = 3 if fault == "indices" else False
    elif fault == "receipt":
        report["events"][1]["applied"][0]["episode_id"] = new.episode.episode_id
    elif fault == "clock":
        report["events"][0]["observed_at"] = False
    elif fault == "work":
        report["work"]["stage_balanced_packets"]["retained_index_numeric_bytes"] -= 8
    else:
        selected = False
    result = replace(original, report=report)
    monkeypatch.setattr(robot, "run_frozen_rollout", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        robot._admit_candidate(
            result,
            warm.episode,
            new.episode,
            "bp",
            1,
            a_only_control=True,
            action_history_recipe=RECIPE,
            stage_balanced_packets=selected,
            diagnostic_stage_bindings=(warm, new),
        )
    if selected:
        plan = packets.prepare_readiness_packets(warm, 3)
        with pytest.raises(RuntimeError):
            demo._require_training_packet_evidence(report, plan, 3)


@pytest.mark.parametrize("fault", ["bool", "a_only", "bindings", "replay", "probe", "rgb"])
def test_should_refuse_unsupported_selection_before_fit_policy_or_runtime(fault, monkeypatch):
    warm, new = pair()
    options: dict[str, Any] = dict(
        stage_balanced_packets=True,
        a_only_control=True,
        action_history_recipe=RECIPE,
        diagnostic_stage_bindings=(warm, new),
        phase_updates=1,
    )
    if fault == "bool":
        options["stage_balanced_packets"] = 1
    elif fault == "a_only":
        options["a_only_control"] = False
    elif fault == "bindings":
        options["diagnostic_stage_bindings"] = None
    else:
        options[
            {"replay": "replay_plan", "probe": "probe_plan", "rgb": "rgb_range_recipe"}[fault]
        ] = object()
    monkeypatch.setattr(demo, "ContinuousBCPolicy", forbidden)
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(warm.episode, new.episode, **options)


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_only_true_through_json_wrapper_and_preserve_false_call_shape(
    monkeypatch, selected
):
    calls = []
    report = {"complete": True, "sentinel": "unchanged_report"}

    def run(*args, **options):
        calls.append((args, options))
        return SimpleNamespace(report=report)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    assert (
        demo.run_continuous_learnability_demo("warm", "new", stage_balanced_packets=selected)
        is report
    )
    expected: dict[str, Any] = dict(expected_standardization=None)
    if selected:
        expected["stage_balanced_packets"] = True
    assert calls == [(("warm", "new"), expected)]


@pytest.mark.parametrize("omitted", ["readiness", "stages", "a_only", "visual"])
def test_should_refuse_cli_selection_scope_before_any_input_output_or_learning(
    tmp_path, monkeypatch, omitted
):
    monkeypatch.setattr(robot, "_read_readiness_teacher", forbidden)
    monkeypatch.setattr(robot, "_read_teacher", forbidden)
    monkeypatch.setattr(robot, "run_learning_rollout", forbidden)
    output = tmp_path / "absent.json"
    args = [
        "--output",
        str(output),
        "--warm-teacher",
        "missing-a",
        "--new-teacher",
        "missing-b",
        "--stage-balanced-packets",
    ]
    for name, flag in (
        ("readiness", "--readiness-teachers"),
        ("stages", "--readiness-stage-residuals"),
        ("a_only", "--a-only-control"),
    ):
        if name != omitted:
            args.append(flag)
    if omitted == "visual":
        args.append("--visual-observation")
    with pytest.raises(SystemExit) as error:
        robot.main(args)
    assert error.value.code == 2 and not output.exists()


def test_should_forward_selected_cli_bound_stages_and_original_raw_episodes_without_model_work(
    tmp_path, monkeypatch, capsys
):
    warm, new = binding(encoded=False), binding(19, STAGES, encoded=False)
    calls, reads = [], []

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        value = warm if seed == 17 else new
        return value.episode, {"stages": list(value.stages)}

    def run(*args, **options):
        calls.append((args, options))
        return dict(complete=True)

    monkeypatch.setattr(robot, "_read_readiness_teacher", read)
    monkeypatch.setattr(robot, "run_learning_rollout", run)
    output = tmp_path / "stub.json"
    assert (
        robot.main(
            [
                "--output",
                str(output),
                "--warm-teacher",
                "warm",
                "--new-teacher",
                "new",
                "--phase-updates",
                "1",
                "--readiness-teachers",
                "--readiness-stage-residuals",
                "--a-only-control",
                "--stage-balanced-packets",
                "--taskspace-history",
                "executed",
            ]
        )
        == 0
    )
    assert reads == [
        ("warm", 17, {"visual_observation": False}),
        ("new", 19, {"visual_observation": False}),
    ]
    assert len(calls) == 1
    args, options = calls[0]
    assert args == (output, warm.episode, new.episode)
    assert options == dict(
        learner_family="bp",
        phase_updates=1,
        visual_observation=False,
        a_only_control=True,
        stage_balanced_packets=True,
        action_history_recipe=RECIPE,
        diagnostic_stage_bindings=(warm, new),
    )
    assert '"complete": true' in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.parametrize("fault", ["none", "after_return_metadata", "after_return_clock"])
def test_should_pass_selected_api_options_once_and_recheck_owned_evidence_after_frozen_return(
    completed,
    tmp_path,
    monkeypatch,
    fault,
):
    """Exercise real API composition, with paid training/native boundaries cached."""
    warm, new = binding(encoded=False), binding(19, STAGES, encoded=False)
    raw_before = (asdict(warm), asdict(new))
    record = completed["records"]["balanced"]
    original = record["result"]
    cached = replace(original, report=deepcopy(original.report))
    before = cached.candidate.snapshot_state()
    training_calls, frozen_calls = [], []

    def train(*args, **options):
        training_calls.append((args, options))
        assert args == (completed["warm"].episode, completed["new"].episode)
        assert options == dict(
            phase_updates=1,
            learner_family="bp",
            a_only_control=True,
            stage_balanced_packets=True,
            action_history_recipe=RECIPE,
            diagnostic_stage_bindings=(completed["warm"], completed["new"]),
        )
        return cached

    def frozen(report, candidate, state, path, visual, **options):
        frozen_calls.append((candidate, path, visual, options))
        assert candidate is cached.candidate
        assert_state(state, before)
        assert options == {"action_history_recipe": RECIPE} and visual is False
        report["rollout_attempted"] = True
        report["candidate_unchanged"] = True
        if fault == "after_return_metadata":
            report["training"]["training_packets"]["planned_stage_exposures"]["settle"] = 8
        elif fault == "after_return_clock":
            # False equals the expected0 numerically; it is not a logical tick.
            report["training"]["events"][0]["observed_at"] = False

    monkeypatch.setattr(robot, "run_continuous_learnability", train)
    monkeypatch.setattr(robot, "_frozen_return", frozen)
    monkeypatch.setattr(robot, "run_frozen_rollout", forbidden)
    output = tmp_path / f"cached-api-{fault}.json"
    report = robot.run_learning_rollout(
        output,
        warm.episode,
        new.episode,
        learner_family="bp",
        phase_updates=1,
        a_only_control=True,
        action_history_recipe=RECIPE,
        diagnostic_stage_bindings=(warm, new),
        stage_balanced_packets=True,
    )
    assert len(training_calls) == len(frozen_calls) == 1
    assert report["rollout_attempted"] is report["candidate_unchanged"] is True
    assert report["stage_balanced_packets"] is True and report["training"] is not None
    assert report["complete"] is (fault == "none")
    assert ("error" in report) is (fault != "none")
    assert (asdict(warm), asdict(new)) == raw_before
    assert cached.report == original.report
    assert_state(before, cached.candidate.snapshot_state())
    assert_state(record["initial"], record["source"].snapshot_state())
    assert_state(record["initial"], record["runtime"].actor.snapshot_state().state)
    # Three explicit output files together remain within1MiB of temporary IO.
    assert output.stat().st_size <= 1024 * 1024 // 3
    assert json.loads(output.read_text(encoding="utf8"))["complete"] is (fault == "none")
    assert not output.with_name(output.name + ".rollout.json").exists()
