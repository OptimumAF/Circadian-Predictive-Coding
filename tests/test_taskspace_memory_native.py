"""Native demo boundaries with synthetic scene returns, never a simulator.

One genuine seven-row stream costs one nearest constructor/seven appends/seven
rows. All other streams and all four-scene endpoints are scalar doubles. No
prediction, fitting, teacher, neural initialization or SDK work is permitted.
"""

from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_taskspace_memory_demo as cli
from scripts import taskspace_memory_native as native
from src.adapters import numpy_nearest_continuous_bc as nearest
from src.core.pick_cube_train_context import pick_cube_training_context
from test_taskspace_memory_demo import bundle


SCENES = ("pick19", "pick17", "pick20", "push17")
COUNTS: defaultdict[str, int] = defaultdict(int)


def forbidden(*args, **kwargs):
    pytest.fail("native fixture attempted prediction, fitting or real SDK work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = COUNTS
    initialize = nearest.NumpyNearestContinuousBCLearner.__init__
    append = nearest.NumpyNearestContinuousBCLearner.train_batch

    def initialized(owner, policy):
        counts["initializations"] += 1
        return initialize(owner, policy)

    def appended(owner, features, targets):
        counts["append_attempts"] += 1
        counts["append_rows"] += len(features)
        result = append(owner, features, targets)
        counts["append_returns"] += 1
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", initialized)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "train_batch", appended)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", forbidden)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "fit", forbidden, raising=False)
        yield counts
    if os.environ.get("NATIVE_MEMORY_DEMO_FIXTURE_LEDGER"):
        with Path(os.environ["NATIVE_MEMORY_DEMO_FIXTURE_LEDGER"]).open(
            "x", encoding="utf8"
        ) as handle:
            json.dump(dict(counts), handle, sort_keys=True)
            handle.write("\n")
    assert counts["initializations"] == 1
    assert counts["append_attempts"] == counts["append_returns"] == 7
    assert counts["append_rows"] == 7
    assert counts["scene_attempts"] == 41 and counts["scene_returns"] == 40


def receipt(seed, *, push=False):
    context = f"push-cube-train-{seed}" if push else pick_cube_training_context(seed)
    transition = dict(observation=[0.0] * (12 if push else 42))
    value: dict[str, Any] = dict(
        complete=True,
        role="train",
        seed=seed,
        source_updates=7,
        source_training_rows=7,
        episode=dict(context=context, transitions=[transition]),
        final_success=False,
        any_success=False,
        sparse_return=0.0,
        prediction_attempts=1,
        prediction_returns=1,
        nearest_prediction_attempts=1,
        nearest_prediction_returns=1,
        nearest_distance_candidate_upper_bound=7,
        nearest_distance_candidates_completed=7,
        nearest_distance_feature_upper_bound=84,
        nearest_distance_feature_values_completed=84,
        nearest_selected_indices=[0],
        work=dict(completed_control_physics_steps=5, possible_failed_physics_steps=0),
    )
    if push:
        value.update(
            transitions=[transition],
            source_unchanged=True,
            unreturned_prediction_possible=False,
            unreturned_native_work_unknown=False,
            unvalidated_nearest_prediction_work_unknown=False,
            control_attempts=1,
            control_returns=1,
        )
    else:
        value.update(
            frozen_state_exact=True,
            original_owner_exact=True,
            action_history_exact=True,
            optimizer_updates=0,
        )
    return value


def install(monkeypatch, output, *, fault=None, genuine=False):
    """Only the final observer is simulated; scene IO still uses real files."""
    calls: list[str] = []
    holders: list[Any] = []
    owner = SimpleNamespace(state={"value": 0})

    if not genuine:

        def pin(readout):
            return (
                readout.state,
                deepcopy(readout.state),
                None,
                None,
                SimpleNamespace(completed_updates=7, training_examples=7),
            )

        def require(readout, birth):
            if readout.state is not birth[0] or readout.state != birth[1]:
                raise ValueError("controlled readout changed")

        def stream(warm, releases, observer, **options):
            assert options == dict(fixed_input=True, maximum_releases=6)
            assert len(releases) == 6
            assert observer(owner, dict(phase="warm", packet_index=None)) == {}
            if fault == "no-final":
                return dict(complete=True)
            metadata = dict(
                phase="after_release",
                packet_index=5,
                completed_updates=7,
                prototype_rows=7,
                known_episode_ids=[e.episode_id for e in (warm, *releases)],
            )
            observer(owner, metadata)
            return dict(complete=fault != "stream-incomplete", error="controlled stream failure")

        monkeypatch.setattr(native, "_pin_readout", pin)
        monkeypatch.setattr(native, "_require_readout", require)
        monkeypatch.setattr(native, "run_sequence", stream)

    def before_scene(scene):
        calls.append(scene)
        COUNTS["scene_attempts"] += 1
        if scene == "pick17":
            if fault == "prior-raw":
                holders[0].report["sparse_return"] = 9.0
            elif fault == "holder":
                holders[0].report = deepcopy(holders[0].report)
            elif fault == "file":
                (output.parent / f"{output.stem}-pick19.json").write_text(
                    "replaced", encoding="utf8"
                )
            elif fault == "model":
                owner.state = deepcopy(owner.state)
            elif fault == "main":
                output.write_text("external paid bytes", encoding="utf8")
        if scene == "pick19" and fault == "lost":
            raise RuntimeError("controlled endpoint lost return")

    def pick(path, *, learner, train_seed, action_history_recipe):
        scene = f"pick{train_seed}"
        before_scene(scene)
        assert action_history_recipe.mode == "executed"
        raw = receipt(train_seed)
        if scene == "pick19":
            if fault == "incomplete":
                raw["complete"] = False
            elif fault == "counter":
                raw["source_updates"] = 6
            elif fault == "role":
                raw["role"] = "probe_test"
        with path.open("x", encoding="utf8") as handle:
            json.dump(raw, handle)
        holder = SimpleNamespace(report=raw, corrections={} if fault == "corrections" else None)
        holders.append(holder)
        COUNTS["scene_returns"] += 1
        return holder

    def push(learner, *, label, seed):
        before_scene("push17")
        assert label == "portable-memory-demo" and seed == 17
        COUNTS["scene_returns"] += 1
        return receipt(seed, push=True)

    monkeypatch.setattr(native, "run_frozen_rollout", pick)
    monkeypatch.setattr(native, "roll_push", push)
    if fault == "current":
        read = native._read_capture

        def changed(path, maximum):
            result = read(path, maximum)
            if path.name == f"{output.stem}-pick19.json" and holders:
                holders[0].report["sparse_return"] = 8.0
            return result

        monkeypatch.setattr(native, "_read_capture", changed)
    return calls, holders


def test_should_run_only_final_checkpoint_in_fixed_order_and_accept_negative_scores(
    monkeypatch, tmp_path
):
    output = tmp_path / "native.json"
    payload = bundle()
    original = deepcopy(payload)
    calls, _ = install(monkeypatch, output)
    result = native.run_native_demo(payload, output)
    assert result["complete"] is True and result["error"] is None
    assert calls == list(SCENES)
    assert result["attempted"] == ["stream", *SCENES]
    assert result["returned"] == [*SCENES, "stream"]
    assert [row["scene"] for row in result["native"]] == list(SCENES)
    assert set(result["outcomes"]) == set(SCENES)
    assert all(not value["final_success"] for value in result["outcomes"].values())
    assert result["work"]["optimizer_updates"] == result["work"]["teacher_queries"] == 0
    assert result == json.loads(output.read_text(encoding="utf8"))
    assert payload == original


def test_should_compose_one_real_seven_row_stream_without_prediction(monkeypatch, tmp_path):
    output = tmp_path / "real-stream.json"
    calls, _ = install(monkeypatch, output, genuine=True)
    result = native.run_native_demo(bundle(), output)
    assert result["complete"] is True and calls == list(SCENES)
    stream = result["stream"]
    assert stream["experiment_state"] == dict(completed_updates=6, next_tick=18)
    assert stream["model_state"]["candidate_publications"] == 7
    assert stream["model_state"]["candidate_prototype_rows"] == 7
    assert len(stream["checkpoints"]) == 13 and len(stream["applied"]) == 6
    assert [row["applied_at"] for row in stream["applied"]] == [2, 5, 8, 11, 14, 17]
    assert stream["work"]["fit_attempts"] == 0


def test_should_accept_native_tuple_fields_after_json_storage(monkeypatch, tmp_path):
    output = tmp_path / "tuple-result.json"
    calls, _ = install(monkeypatch, output)
    pick = native.run_frozen_rollout

    def native_tuples(path, **options):
        holder = pick(path, **options)
        # Native dataclass reports retain tuples; JSON storage turns them into lists.
        holder.report["nearest_selected_indices"] = (0,)
        holder.report["episode"]["transitions"][0]["observation"] = (0.0,) * 42
        path.write_text(json.dumps(holder.report), encoding="utf8")
        return holder

    monkeypatch.setattr(native, "run_frozen_rollout", native_tuples)
    result = native.run_native_demo(bundle(), output)
    assert result["complete"] is True and calls == list(SCENES)
    assert json.loads(output.read_text(encoding="utf8"))["complete"] is True
    assert result["native"][0]["report"]["nearest_selected_indices"] == (0,)


@pytest.mark.parametrize("destination", ["main", "failure", "pick19", "missing-parent"])
def test_should_refuse_all_reserved_output_collisions_before_admission(
    monkeypatch, tmp_path, destination
):
    output = tmp_path / "native.json"
    selected = {
        "main": output,
        "failure": output.with_suffix(".failure.json"),
        "pick19": tmp_path / "native-pick19.json",
    }
    if destination == "missing-parent":
        output = tmp_path / "absent" / "native.json"
    else:
        selected[destination].write_text("preserve", encoding="utf8")
    monkeypatch.setattr(native, "_admit", forbidden)
    with pytest.raises(ValueError, match="fresh"):
        native.run_native_demo({}, output)
    if destination != "missing-parent":
        assert selected[destination].read_text(encoding="utf8") == "preserve"


def test_should_refuse_invalid_bundle_before_stream_and_output(monkeypatch, tmp_path):
    output = tmp_path / "native.json"
    payload = bundle()
    payload["training"][-1]["episode"]["role"] = "final"
    monkeypatch.setattr(native, "run_sequence", forbidden)
    with pytest.raises(ValueError):
        native.run_native_demo(payload, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "counter",
        "role",
        "corrections",
        "lost",
        "prior-raw",
        "holder",
        "file",
        "model",
        "current",
    ],
)
def test_should_stop_after_paid_refusal_and_preserve_original_return(monkeypatch, tmp_path, fault):
    output = tmp_path / "native.json"
    calls, _ = install(monkeypatch, output, fault=fault)
    result = native.run_native_demo(bundle(), output)
    assert result["complete"] is False and result["error"]
    assert calls == (
        ["pick19", "pick17"] if fault in {"prior-raw", "holder", "file", "model"} else ["pick19"]
    )
    assert "push17" not in result["attempted"]
    if fault == "lost":
        assert result["native"] == [] and "pick19" not in result["returned"]
        assert result["unreturned_stage_work_unknown"] is True
    elif fault == "current":
        assert result["pending_paid_return"]["report"]["sparse_return"] == 0.0
    else:
        assert result["native"][0]["report"]["sparse_return"] == 0.0
        assert result["returned"][0] == "pick19"
    captured = output.with_suffix(".failure.json")
    if not captured.exists():
        captured = output
    assert json.loads(captured.read_text(encoding="utf8"))["complete"] is False


def test_should_preserve_replaced_main_bytes_and_exclusively_capture_failure(monkeypatch, tmp_path):
    output = tmp_path / "native.json"
    calls, _ = install(monkeypatch, output, fault="main")
    result = native.run_native_demo(bundle(), output)
    assert result["complete"] is False and calls == ["pick19", "pick17"]
    assert output.read_text(encoding="utf8") == "external paid bytes"
    failure = json.loads(output.with_suffix(".failure.json").read_text(encoding="utf8"))
    assert failure["native"][0]["report"]["sparse_return"] == 0.0
    assert failure["pending_paid_return"]["scene"] == "pick17"


def test_should_refuse_postwrite_readback_mutation_before_next_scene(monkeypatch, tmp_path):
    output = tmp_path / "native.json"
    calls, holders = install(monkeypatch, output)
    read = native._read_capture
    changed = False

    def postwrite(path, maximum):
        nonlocal changed
        result = read(path, maximum)
        if path == output and not changed and holders:
            current = json.loads(result[0])
            if len(current["native"]) == 1:
                changed = True
                holders[0].report["sparse_return"] = 7.0
        return result

    monkeypatch.setattr(native, "_read_capture", postwrite)
    result = native.run_native_demo(bundle(), output)
    assert changed and result["complete"] is False
    assert calls == ["pick19"]
    assert result["native"][0]["report"]["sparse_return"] == 0.0


@pytest.mark.parametrize("target", ["raw", "outcome"])
def test_should_revalidate_paid_evidence_after_final_clock(monkeypatch, tmp_path, target):
    output = tmp_path / "native.json"
    calls, holders = install(monkeypatch, output)
    summaries: list[dict[str, Any]] = []
    summarize = native._summary

    def summary(raw, *, push):
        value = summarize(raw, push=push)
        summaries.append(value)
        return value

    def clock():
        if holders:
            if target == "raw":
                holders[0].report["sparse_return"] = 6.0
            else:
                summaries[0]["sparse_return"] = 6.0
        return 0.0

    monkeypatch.setattr(native, "_summary", summary)
    monkeypatch.setattr(native, "monotonic", clock)
    result = native.run_native_demo(bundle(), output)
    assert calls == list(SCENES) and result["complete"] is False
    assert result["native"][0]["report"]["sparse_return"] == 0.0
    assert result["outcomes"]["pick19"]["sparse_return"] == 0.0


@pytest.mark.parametrize("fault", ["no-final", "stream-incomplete"])
def test_should_require_complete_stream_and_exact_final_observation(monkeypatch, tmp_path, fault):
    output = tmp_path / "native.json"
    calls, _ = install(monkeypatch, output, fault=fault)
    result = native.run_native_demo(bundle(), output)
    assert result["complete"] is False and result["stream"]["complete"] is (fault == "no-final")
    assert calls == ([] if fault == "no-final" else list(SCENES))


def test_should_require_native_output_before_loading_bundle(monkeypatch):
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--native"])
    assert error.value.code == 2


@pytest.mark.parametrize("complete", [True, False])
def test_should_route_native_cli_once_without_reopening_owned_output(
    monkeypatch, tmp_path, capsys, complete
):
    output = tmp_path / "native.json"
    payload = {"synthetic": "CLI routing only"}
    calls: list[Path] = []
    monkeypatch.setattr(cli, "load_bundle", lambda path: payload)
    monkeypatch.setattr(cli, "run_demo", forbidden)

    def run(value, path):
        assert value is payload and path == output
        calls.append(path)
        output.write_text("native-owned capture", encoding="utf8")
        return dict(complete=complete, mode="native_fresh_rollout")

    monkeypatch.setattr(native, "run_native_demo", run)
    assert cli.main(["--native", "--output", str(output)]) == (0 if complete else 1)
    assert calls == [output]
    assert output.read_text(encoding="utf8") == "native-owned capture"
    assert json.loads(capsys.readouterr().out)["complete"] is complete
