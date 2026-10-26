"""Owned live TRAIN packets; marked false labels never change guard/task truth."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts.ticket_controller_streams import (
    TicketStreamRecipe,
    LiveTrainingRecipe,
    LIVE_TRAINING_RECIPES,
    WITHHELD_RECIPES,
    build_stream,
    build_live_stream,
)
from scripts.ticket_serving_timeline import LiveServingPlan


@pytest.fixture(scope="module")
def prototypes():
    spec = demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
    _, _, batches = demo.context._teacher_batches(spec)
    return batches[0], batches[demo.context.BATCHES_PER_PHASE]


def mixed(count=2):
    return LiveTrainingRecipe(TicketStreamRecipe("live-test", ("new_contexts",) * count, 2), True)


def test_retained_noise_changes_only_owned_old_invoice_label_and_preserves_new_teacher(prototypes):
    new, old = prototypes
    before = (new.features.copy(), new.targets.copy(), old.features.copy(), old.targets.copy())
    clean = build_stream(mixed().recipe, new, old)
    noisy = build_live_stream(mixed(), new, old)
    ids = []
    for a, b in zip(clean, noisy, strict=True):
        assert b.role == "train" and b.features.shape == (12, 7) and b.targets.shape == (12, 1)
        assert np.array_equal(a.features, b.features) and np.array_equal(b.targets[:6], new.targets)
        assert np.flatnonzero(a.targets[:, 0] != b.targets[:, 0]).tolist() == [7]
        assert a.targets[7, 0] == demo.ACTION_IDS.index("inspect")
        assert b.targets[7, 0] == demo.ACTION_IDS.index("defer")
        assert not np.shares_memory(b.features, new.features) and not np.shares_memory(
            b.targets, old.targets
        )
        ids.extend(b.row_ids)
    assert len(ids) == len(set(ids)) == 24
    noisy[0].targets[0, 0] = 99
    assert noisy[1].targets[0, 0] != 99
    for a, b in zip(before, (new.features, new.targets, old.features, old.targets), strict=True):
        np.testing.assert_array_equal(a, b)


def test_old_recipe_schema_and_old_only_row1_corruption_are_unchanged(prototypes):
    new, old = prototypes
    assert set(asdict(WITHHELD_RECIPES[1])) == {
        "recipe_id",
        "contexts",
        "label_delay",
        "noisy_invoice",
    }
    original = build_stream(WITHHELD_RECIPES[1], new, old)
    live = build_live_stream(LiveTrainingRecipe(WITHHELD_RECIPES[1]), new, old)
    for a, b in zip(original, live, strict=True):
        assert a.row_ids == b.row_ids and a.context == b.context
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.targets, b.targets)
        assert b.targets[1, 0] == demo.ACTION_IDS.index("defer")
        assert b.targets[7, 0] == old.targets[1, 0] == demo.ACTION_IDS.index("inspect")


@pytest.mark.parametrize("role", ["probe_train", "inner_guard", "outer_selection", "final_test"])
def test_live_builder_never_relabels_nontrain_prototypes(role, prototypes):
    new, old = prototypes
    with pytest.raises(ValueError, match="role must be train"):
        build_live_stream(mixed(), replace(new, role=role), old)
    with pytest.raises(ValueError, match="role must be train"):
        build_live_stream(mixed(), new, replace(old, role=role))


def test_bad_retained_teacher_and_double_or_non_boolean_noise_are_refused(prototypes):
    new, old = prototypes
    bad = old.targets.copy()
    bad[1, 0] = demo.ACTION_IDS.index("route")
    with pytest.raises(ValueError, match="retained inspect"):
        build_live_stream(mixed(), new, replace(old, targets=bad))
    with pytest.raises(ValueError, match="two label corruptions"):
        LiveTrainingRecipe(WITHHELD_RECIPES[1], True)
    with pytest.raises(ValueError, match="boolean"):
        LiveTrainingRecipe(mixed().recipe, cast(Any, 1))
    with pytest.raises(ValueError, match="exact"):
        LiveTrainingRecipe(cast(Any, object()))
    broken = mixed()
    object.__setattr__(broken.recipe, "contexts", ("final_test",))
    with pytest.raises(ValueError, match="contexts"):
        build_live_stream(broken, new, old)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"live_training_recipe": mixed()},
        {"live_training_recipe": object(), "serving_plan": LiveServingPlan()},
        {
            "live_training_recipe": mixed(),
            "serving_plan": LiveServingPlan(),
            "reference_mode": "periodic",
        },
        {
            "live_training_recipe": mixed(),
            "serving_plan": LiveServingPlan(),
            "reference_mode": "drift",
            "stream_recipe": WITHHELD_RECIPES[0],
            "deferred_evaluation": True,
        },
        {
            "live_training_recipe": mixed(),
            "serving_plan": LiveServingPlan(2, ()),
            "reference_mode": "drift",
        },
    ],
)
def test_live_recipe_mode_failures_precede_source_or_model_access(kwargs, monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda p: pytest.fail("source access"))
    with pytest.raises(ValueError, match="live TRAIN|live serving|live plan"):
        demo.run_case(Path("unused"), **kwargs)


def test_comparative_recipe_still_requires_deferred_mode_even_when_live_is_requested(monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda p: pytest.fail("source access"))
    with pytest.raises(ValueError, match="deferred evaluation"):
        demo.run_case(Path("unused"), reference_mode="drift", stream_recipe=WITHHELD_RECIPES[0])
    with pytest.raises(ValueError, match="live serving"):
        demo.run_case(
            Path("unused"),
            reference_mode="drift",
            stream_recipe=WITHHELD_RECIPES[0],
            deferred_evaluation=True,
            serving_plan=LiveServingPlan(),
        )


def test_cli_recipe_requires_live_mode_and_keeps_existing_output(tmp_path, monkeypatch):
    calls = []

    def run(source, **kwargs):
        calls.append(kwargs)
        return {"complete": True}

    monkeypatch.setattr(demo, "run_case", run)
    output = tmp_path / "live.json"
    argv = [
        "--source",
        "unused",
        "--output",
        str(output),
        "--publish-during-serving",
        "--live-training-recipe",
        "live_new_retained_noise",
    ]
    assert demo.main(argv) == 0
    assert calls == [
        dict(
            reference_mode="drift",
            serving_plan=LiveServingPlan(),
            publish_on_drain=True,
            live_training_recipe=LIVE_TRAINING_RECIPES[1],
        )
    ]
    with pytest.raises(FileExistsError):
        demo.main(argv)
    with pytest.raises(SystemExit):
        demo.main(
            [
                "--source",
                "unused",
                "--output",
                str(tmp_path / "bad"),
                "--reference",
                "drift",
                "--live-training-recipe",
                "noisy_return",
            ]
        )
    assert not (tmp_path / "bad").exists()


@pytest.fixture(scope="module")
def native_result():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
        result = demo.run_case(
            Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
            reference_mode="drift",
            serving_plan=LiveServingPlan(8, (2,)),
            publish_on_drain=True,
            live_training_recipe=mixed(),
        )
    if os.environ.get("LIVE_TRAINING_RECIPE_NATIVE_REPORT"):
        Path(os.environ["LIVE_TRAINING_RECIPE_NATIVE_REPORT"]).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf8"
        )
    assert result["complete"], result.get("error")
    return result


def test_native_recipe_uses_causal_twelve_row_feedback_and_original_truth_guards(native_result):
    r = native_result
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=24, returned_updates=2, unreturned_work_unknown=False
    )
    assert (
        r["training_stream"]["row_counts"] == [12, 12] and r["training_stream"]["label_delay"] == 2
    )
    assert r["controller_work"]["prediction_rows"] == r["controller_work"]["feedback_rows"] == 24
    assert all(e["observation"]["feedback"] is None for e in r["controller_events"][:2])
    assert all(
        u["arrived_at"] - u["observed_at"] == 2 and u["arrived_at"] <= u["applied_at"]
        for u in r["applied_updates"]
    )
    assert r["candidate_checkpoint"]["state"]["completed_updates"] == 98
    assert r["candidate_checkpoint"]["state"]["training_examples"] == 600
    assert r["drain_publication"]["tick"] == 4 and r["drain_publication"]["completed"]
    assert r["training_corruption"]["rows_per_batch"] == [7]
    assert r["training_corruption"]["planned_corrupted_exposures"] == 2
    assert r["source_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["final_examples"] == r["reserved_probe_updates"] == r["replay_examples"] == 0
    old_guard = next(g for g in r["guards"] if g["task_id"] == "old")
    assert old_guard["targets"][1][0] == demo.ACTION_IDS.index("inspect")
    assert r["training_stream"]["first_targets"][7][0] == demo.ACTION_IDS.index("defer")
    assert all(g["role"] != "live_service" for g in r["guards"])


def test_mutated_original_live_recipe_stops_before_more_native_or_serving_work(monkeypatch):
    recipe = mixed()
    original = demo.choose_reference

    def mutate(policy, observation):
        object.__setattr__(recipe, "retained_invoice_noise", False)
        return original(policy, observation)

    monkeypatch.setattr(demo, "choose_reference", mutate)
    r = demo.run_case(
        Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
        reference_mode="drift",
        serving_plan=LiveServingPlan(8, (2,)),
        publish_on_drain=True,
        live_training_recipe=recipe,
    )
    assert not r["complete"] and "live TRAIN recipe changed" in r["error"]
    assert r["training_work"]["attempted_updates"] == r["training_work"]["attempted_rows"] == 0
    assert r["live_serving"]["records"] == []
