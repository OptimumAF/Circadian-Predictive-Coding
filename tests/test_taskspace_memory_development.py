"""Bounded TRAIN-context development routing with scalar-only scene doubles.

No source learner, nearest query, model mathematics, teacher or SDK is created.
The existing native receipt validator checks synthetic paid-return metadata.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_taskspace_memory_demo as cli
from scripts import taskspace_memory_native as native


def forbidden(*args, **options):
    pytest.fail("refused development request reached admission, stream or native work")


def synthetic_receipt(task, seed):
    push = task == "push"
    transition = dict(observation=[0.0] * (12 if push else 42))
    value: dict[str, Any] = dict(
        complete=True,
        role="train",
        seed=seed,
        source_updates=7,
        source_training_rows=7,
        episode=dict(context=f"{task}-cube-train-{seed}", transitions=[transition]),
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


def install(monkeypatch, *, pick_unrepresented=False, fault=None):
    contexts: tuple[str, ...] = (
        "pick-cube-train-17",
        "pick-cube-train-19",
        "pick-cube-train-20",
        "push-cube-train-17",
        "push-cube-train-17",
        "push-cube-train-17",
        "pick-cube-train-19",
    )
    if pick_unrepresented:
        contexts = tuple(
            "pick-cube-train-17" if value == "pick-cube-train-19" else value for value in contexts
        )
    episodes = tuple(
        SimpleNamespace(episode_id=f"original-{index}", context=context, transitions=(object(),))
        for index, context in enumerate(contexts)
    )
    owner = SimpleNamespace(state={"unchanged": True})
    calls: list[tuple[str, int]] = []
    returned: list[dict[str, Any]] = []

    def admit(payload):
        return episodes, (), {"notice": "synthetic admitted metadata"}

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
            raise ValueError("synthetic source owner changed")

    def sequence(warm, releases, observer, **options):
        assert options == dict(fixed_input=True, maximum_releases=6)
        assert (warm, *releases) == episodes
        assert observer(owner, dict(phase="before_release", packet_index=5)) == {}
        observer(
            owner,
            dict(
                phase="after_release",
                packet_index=5,
                completed_updates=7,
                prototype_rows=7,
                known_episode_ids=[item.episode_id for item in episodes],
            ),
        )
        return dict(complete=True, experiment_state={"completed_updates": 6})

    def result(task, seed):
        calls.append((task, seed))
        if len(calls) == 2:
            if fault == "prior":
                returned[0]["sparse_return"] = 9.0
            elif fault == "model":
                owner.state = deepcopy(owner.state)
        raw = synthetic_receipt(task, seed)
        if len(calls) == 1:
            if fault == "role":
                raw["role"] = "probe_test"
            elif fault == "context":
                raw["episode"]["context"] = "push-cube-train-17"
            elif fault == "seed":
                raw["seed"] = True
            elif fault == "incomplete":
                raw["complete"] = False
        returned.append(raw)
        return raw

    def push(learner, *, label, seed):
        assert learner is owner and label == "portable-memory-demo"
        return result("push", seed)

    def pick(path, *, learner, train_seed, action_history_recipe):
        assert learner is owner and action_history_recipe.mode == "executed"
        raw = result("pick", train_seed)
        with path.open("x", encoding="utf8") as handle:
            json.dump(raw, handle)
        return SimpleNamespace(report=raw, corrections=None)

    monkeypatch.setattr(native, "_admit", admit)
    monkeypatch.setattr(native, "_pin_readout", pin)
    monkeypatch.setattr(native, "_require_readout", require)
    monkeypatch.setattr(native, "run_sequence", sequence)
    monkeypatch.setattr(native, "roll_push", push)
    monkeypatch.setattr(native, "run_frozen_rollout", pick)
    return calls, returned


@pytest.mark.parametrize("task", ["pick", "push"])
@pytest.mark.parametrize("seed", [17, 19, 20])
def test_should_admit_only_existing_train_descriptors_and_freeze_their_values(task, seed):
    scene = native.NativeScene(task, seed)
    assert (scene.task, scene.seed) == (task, seed)
    with pytest.raises(FrozenInstanceError):
        setattr(scene, "seed", 17)


@pytest.mark.parametrize(
    "task,seed",
    [("Pick", 17), ("push", True), ("push", 23), ("pick", 18), ("push", "19"), (None, 19)],
)
def test_should_refuse_unknown_tasks_reserved_seeds_and_coerced_descriptor_values(task, seed):
    with pytest.raises(ValueError):
        native.NativeScene(task, seed)


@pytest.mark.parametrize(
    "kind", ["empty", "list", "three", "duplicate", "none-item", "string", "bool"]
)
def test_should_refuse_invalid_scene_sequences_before_bundle_admission(monkeypatch, tmp_path, kind):
    scene = native.NativeScene("push", 19)
    options: dict[str, Any] = {
        "empty": (),
        "list": [scene],
        "three": (scene,) * 3,
        "duplicate": (scene, scene),
        "none-item": (None,),
        "string": "push19",
        "bool": True,
    }
    monkeypatch.setattr(native, "_admit", forbidden)
    with pytest.raises(ValueError):
        native.run_native_demo({}, tmp_path / "result.json", development_scenes=options[kind])
    assert list(tmp_path.iterdir()) == []


def test_should_revalidate_a_frozen_descriptor_corrupted_before_the_call(monkeypatch, tmp_path):
    scene = native.NativeScene("push", 19)
    object.__setattr__(scene, "seed", True)
    monkeypatch.setattr(native, "_admit", forbidden)
    with pytest.raises(ValueError):
        native.run_native_demo({}, tmp_path / "result.json", development_scenes=(scene,))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("task,seed", [("pick", 19), ("push", 17)])
def test_should_refuse_bank_represented_contexts_before_stream_or_output(
    monkeypatch, tmp_path, task, seed
):
    install(monkeypatch)
    monkeypatch.setattr(native, "run_sequence", forbidden)
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="context|represented"):
        native.run_native_demo({}, output, development_scenes=(native.NativeScene(task, seed),))
    assert not output.exists()


@pytest.mark.parametrize("tasks", [("push", "push"), ("pick", "push")])
def test_should_run_selected_unrepresented_train_contexts_once_and_accept_negative_scores(
    monkeypatch, tmp_path, tasks
):
    calls, _ = install(monkeypatch, pick_unrepresented=tasks[0] == "pick")
    selected = tuple(
        native.NativeScene(task, seed) for task, seed in zip(tasks, (19, 20), strict=True)
    )
    output = tmp_path / "development.json"
    report = native.run_native_demo({}, output, development_scenes=selected)
    assert report["complete"] is True and report["error"] is None
    assert report["definition"] == "taskspace_memory_development_v1"
    assert report["mode"] == "unrepresented_train_contexts"
    assert (
        report["classification"]
        == "fixed TRAIN resets unrepresented in admitted bank; no global unseen-data claim"
    )
    assert calls == list(zip(tasks, (19, 20), strict=True))
    names = [f"{task}{seed}" for task, seed in calls]
    assert report["attempted"] == ["stream", *names]
    assert report["returned"] == [*names, "stream"]
    assert report["development_scenes"] == [
        dict(
            task=task,
            seed=seed,
            scene=f"{task}{seed}",
            context=f"{task}-cube-train-{seed}",
            role="train",
        )
        for task, seed in calls
    ]
    assert set(report["outcomes"]) == set(names)
    assert all(not item["final_success"] for item in report["outcomes"].values())
    assert json.loads(output.read_text(encoding="utf8")) == report
    assert all(item["report"]["source_updates"] == 7 for item in report["native"])
    assert (tmp_path / "development-pick19.json").exists() is (tasks[0] == "pick")


@pytest.mark.parametrize("fault", ["role", "context", "seed", "incomplete", "prior", "model"])
def test_should_stop_after_invalid_or_corrupted_development_return_and_keep_paid_prefix(
    monkeypatch, tmp_path, fault
):
    calls, _ = install(monkeypatch, fault=fault)
    selected = (native.NativeScene("push", 19), native.NativeScene("push", 20))
    report = native.run_native_demo({}, tmp_path / "development.json", development_scenes=selected)
    assert report["complete"] is False and report["error"]
    assert calls == (
        [("push", 19), ("push", 20)] if fault in {"prior", "model"} else [("push", 19)]
    )
    assert report["returned"][0] == "push19"
    assert report["native"][0]["report"]["sparse_return"] == 0.0
    if fault in {"prior", "model"}:
        assert report["pending_paid_return"]["scene"] == "push20"


@pytest.mark.parametrize("collision", ["main", "selected-pick"])
def test_should_refuse_only_selected_scene_destination_collisions_without_overwriting(
    monkeypatch, tmp_path, collision
):
    output = tmp_path / "development.json"
    path = output if collision == "main" else tmp_path / "development-pick19.json"
    path.write_text("preserve", encoding="utf8")
    monkeypatch.setattr(native, "_admit", forbidden)
    with pytest.raises(ValueError):
        native.run_native_demo({}, output, development_scenes=(native.NativeScene("pick", 19),))
    assert path.read_text(encoding="utf8") == "preserve"


@pytest.mark.parametrize(
    "arguments", [["--development-probe"], ["--development-probe", "--native"]]
)
def test_should_require_explicit_native_and_output_for_development_cli(monkeypatch, arguments):
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


def test_should_forward_fixed_push19_then_push20_cli_selection_without_default_native_work(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "development.json"
    payload = {"synthetic": "CLI metadata"}
    calls: list[Any] = []
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: payload)
    monkeypatch.setattr(cli, "run_demo", forbidden)

    def run(value, path, **options):
        assert value is payload and path == output
        assert tuple((item.task, item.seed) for item in options["development_scenes"]) == (
            ("push", 19),
            ("push", 20),
        )
        calls.append(options)
        output.write_text("native owns capture", encoding="utf8")
        return dict(complete=True, mode="unrepresented_train_contexts")

    monkeypatch.setattr(native, "run_native_demo", run)
    assert cli.main(["--native", "--development-probe", "--output", str(output)]) == 0
    assert len(calls) == 1 and json.loads(capsys.readouterr().out)["complete"] is True
    assert output.read_text(encoding="utf8") == "native owns capture"
