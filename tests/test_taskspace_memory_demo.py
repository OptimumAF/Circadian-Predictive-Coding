"""Portable CPU bundle admission and recorded replay, without new outcomes.

Three tiny seven-row streams: success, paid mismatch and malformed receipt.
Forecast3 nearest constructors/21 appends21 rows/6 predictions6 query rows/
42distance candidates504feature visits. All other cases are metadata/IO only.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.adapters import taskspace_memory_demo as demo
from src.adapters import continuous_stream_sequence as sequence
from src.adapters import numpy_nearest_continuous_bc as nearest
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.core.pick_cube_train_context import pick_cube_training_context

GROUPS = ("a", "b", "train20", "push", "corrections", "current_corrections", "b22")
SCENES = ("pick19", "pick17", "pick20", "push17")


def episode(name, context, *, count=1, corrective=False):
    history = ActionHistoryState()
    rows = []
    target = (0.125, 0.0, 0.0, -1.0)
    for index in range(count):
        before = (0.0,) * 7 + (*history.previous_action, history.closing_count / 6)
        history = advance_action_history(history, (0.25, 0.0, 0.0, 1.0) if corrective else target)
        after = (0.0,) * 7 + (*history.previous_action, history.closing_count / 6)
        rows.append(
            ContinuousTransition(index, before, target, 0.0, after, False, index == count - 1)
        )
    return asdict(ContinuousEpisode(name, context, "train", cube_taskspace_spec(), tuple(rows)))


def bundle():
    contexts = (
        pick_cube_training_context(17),
        "pick-cube-train-19",
        "pick-cube-train-20",
        "push-cube-train-17",
        "push-cube-train-17",
        "push-cube-train-17",
        "pick-cube-train-19",
    )
    value = dict(
        definition="taskspace_memory_demo_v1",
        provenance={"notice": "synthetic generated test data, no SDK assets"},
        training=[
            dict(
                group=name,
                episode=episode("training-" + name, context, corrective=name == "corrections"),
            )
            for name, context in zip(GROUPS, contexts, strict=True)
        ],
        replay=[
            dict(
                scene=scene,
                seed=seed,
                episode=episode("recorded-" + scene, context),
                selected_indices=[0],
                recorded_outcomes=dict(final_success=False, any_success=False, sparse_return=0.0),
            )
            for scene, seed, context in zip(
                SCENES,
                (19, 17, 20, 17),
                (
                    "pick-cube-train-19",
                    pick_cube_training_context(17),
                    "pick-cube-train-20",
                    "push-cube-train-17",
                ),
                strict=True,
            )
        ],
    )
    return json.loads(json.dumps(value))


def forbidden(*args, **options):
    pytest.fail("refused bundle reached constructor, fitter or real SDK work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts: defaultdict[str, int] = defaultdict(int)
    initialize, append, predict = (
        nearest.NumpyNearestContinuousBCLearner.__init__,
        nearest.NumpyNearestContinuousBCLearner.train_batch,
        nearest.NumpyNearestContinuousBCLearner.predict_with_work,
    )

    def initialized(owner, policy):
        counts["initializations"] += 1
        return initialize(owner, policy)

    def appended(owner, features, targets):
        counts["append_attempts"] += 1
        counts["append_rows"] += len(features)
        result = append(owner, features, targets)
        counts["append_returns"] += 1
        return result

    def predicted(owner, features):
        counts["prediction_attempts"] += 1
        counts["query_rows"] += len(features)
        result = predict(owner, features)
        counts["prediction_returns"] += 1
        counts["distance_candidates"] += result.distance_candidates
        counts["distance_feature_values"] += result.distance_feature_values
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", initialized)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "train_batch", appended)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", predicted)
        patch.setattr(sequence, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("TASKSPACE_MEMORY_DEMO_FIXTURE_LEDGER"):
        with Path(os.environ["TASKSPACE_MEMORY_DEMO_FIXTURE_LEDGER"]).open(
            "x", encoding="utf8"
        ) as output:
            json.dump(dict(counts), output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 3
    assert (
        counts["append_attempts"] == counts["append_returns"] == 21 and counts["append_rows"] == 21
    )
    assert (
        counts["prediction_attempts"] == counts["prediction_returns"] == counts["query_rows"] == 6
    )
    assert counts["distance_candidates"] == 42 and counts["distance_feature_values"] == 504


@pytest.fixture(scope="module")
def successful(ledger):
    payload = bundle()
    before = deepcopy(payload)
    result = demo.run_demo(payload)
    assert payload == before
    return result, payload


@pytest.mark.parametrize(
    "fault",
    [
        "payload",
        "definition",
        "training_order",
        "training_role",
        "training_spec",
        "training_duplicate",
        "training_count",
        "training_total",
        "training_rows",
        "replay_order",
        "replay_seed",
        "replay_context",
        "replay_role",
        "replay_history",
        "replay_float32",
        "indices_length",
        "index_bool",
        "index_range",
        "score_nonfinite",
        "score_mismatch",
    ],
)
def test_should_refuse_all_bad_bundle_scopes_before_any_memory_constructor(monkeypatch, fault):
    payload: Any = bundle()
    if fault == "payload":
        payload = [payload]
    elif fault == "definition":
        payload["definition"] = "unknown"
    elif fault == "training_order":
        payload["training"][0], payload["training"][1] = (
            payload["training"][1],
            payload["training"][0],
        )
    elif fault == "training_role":
        payload["training"][0]["episode"]["role"] = "final_test"
    elif fault == "training_spec":
        payload["training"][0]["episode"]["spec"]["action_units"][0] = "other"
    elif fault == "training_duplicate":
        payload["training"][-1]["episode"]["episode_id"] = payload["training"][0]["episode"][
            "episode_id"
        ]
    elif fault == "training_count":
        payload["training"].append(deepcopy(payload["training"][0]))
    elif fault == "training_total":
        for item, count in zip(payload["training"], (37, 37, 37, 37, 37, 37, 35), strict=True):
            item["episode"] = json.loads(
                json.dumps(
                    episode(item["episode"]["episode_id"], item["episode"]["context"], count=count)
                )
            )
    elif fault == "training_rows":
        item = payload["training"][0]
        item["episode"] = json.loads(
            json.dumps(episode("oversize", item["episode"]["context"], count=51))
        )
    elif fault == "replay_order":
        payload["replay"][0], payload["replay"][1] = payload["replay"][1], payload["replay"][0]
    elif fault == "replay_seed":
        payload["replay"][0]["seed"] = True
    elif fault == "replay_context":
        payload["replay"][0]["episode"]["context"] = "pick-cube-train-20"
    elif fault == "replay_role":
        payload["replay"][0]["episode"]["role"] = "probe_train"
    elif fault == "replay_history":
        payload["replay"][0]["episode"]["transitions"][0]["next_observation"][7] = 0.25
    elif fault == "replay_float32":
        payload["replay"][0]["episode"]["transitions"][0]["action"][0] = 0.123456789
    elif fault == "indices_length":
        payload["replay"][0]["selected_indices"] = []
    elif fault == "index_bool":
        payload["replay"][0]["selected_indices"] = [True]
    elif fault == "index_range":
        payload["replay"][0]["selected_indices"] = [7]
    elif fault == "score_nonfinite":
        payload["replay"][0]["recorded_outcomes"]["sparse_return"] = float("nan")
    else:
        payload["replay"][0]["recorded_outcomes"]["sparse_return"] = 1.0
    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", forbidden)
    with pytest.raises(ValueError):
        demo.run_demo(payload)


def test_should_load_one_owned_finite_bounded_json_bundle_without_learning(tmp_path):
    payload = bundle()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload))
    result = demo.load_bundle(path)
    assert result == payload and result is not payload
    result["provenance"]["notice"] = "changed owned return"
    assert json.loads(path.read_text()) == payload


def test_should_refuse_a_cyclic_python_bundle_before_constructing_a_learner(monkeypatch):
    payload = bundle()
    payload["provenance"]["cycle"] = payload
    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", forbidden)
    with pytest.raises(ValueError):
        demo.run_demo(payload)


@pytest.mark.parametrize("fault", ["oversize", "duplicate", "nonfinite", "list"])
def test_should_refuse_ambiguous_or_unbounded_bundle_json_before_demo(tmp_path, monkeypatch, fault):
    path = tmp_path / "bundle.json"
    raw = (
        b" " * (512 * 1024 + 1)
        if fault == "oversize"
        else b'{"definition":"first",' + json.dumps(bundle()).encode()[1:]
        if fault == "duplicate"
        else b'{"invalid":NaN}'
        if fault == "nonfinite"
        else b"[]"
    )
    path.write_bytes(raw)
    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", forbidden)
    with pytest.raises(ValueError):
        demo.load_bundle(path)


@pytest.mark.parametrize("output_selected", [False, True])
def test_should_route_default_or_selected_bundle_and_optional_fresh_output(
    tmp_path, monkeypatch, capsys, output_selected
):
    from scripts import run_taskspace_memory_demo as cli

    payload = {"fixture": "owned metadata"}
    result = dict(complete=True, mode="offline_recorded_replay", work=dict(simulator_controls=0))
    calls: list[Any] = []
    selected = tmp_path / "selected.json"
    output = tmp_path / "result.json"

    def load(path):
        calls.append(("load", path))
        return payload

    def run(value):
        calls.append(("run", value))
        assert value is payload
        return result

    monkeypatch.setattr(cli, "load_bundle", load)
    monkeypatch.setattr(cli, "run_demo", run)
    arguments = ["--bundle", str(selected), "--output", str(output)] if output_selected else []
    assert cli.main(arguments) == 0
    assert calls == [
        ("load", selected if output_selected else cli.DEFAULT_BUNDLE),
        ("run", payload),
    ]
    assert json.loads(capsys.readouterr().out) == result
    if output_selected:
        assert json.loads(output.read_text()) == result
    else:
        assert not output.exists()


def test_should_print_partial_demo_without_claiming_success(tmp_path, monkeypatch, capsys):
    from scripts import run_taskspace_memory_demo as cli

    report = dict(
        complete=False, error="paid recorded replay mismatch", mode="offline_recorded_replay"
    )
    monkeypatch.setattr(cli, "load_bundle", lambda path: {})
    monkeypatch.setattr(cli, "run_demo", lambda payload: report)
    assert cli.main(["--bundle", str(tmp_path / "input.json")]) == 1
    assert json.loads(capsys.readouterr().out) == report


def test_should_refuse_existing_output_before_bundle_read_or_model_work(tmp_path, monkeypatch):
    from scripts import run_taskspace_memory_demo as cli

    output = tmp_path / "result.json"
    output.write_text("preserve")
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    monkeypatch.setattr(cli, "run_demo", forbidden)
    with pytest.raises(ValueError):
        cli.main(["--output", str(output)])
    assert output.read_text() == "preserve"


def test_should_rebuild_only_seven_training_rows_and_match_all_four_recorded_batches(successful):
    result, payload = successful
    assert result["complete"] is True, result.get("error")
    assert result["mode"] == "offline_recorded_replay"
    assert [item["scene"] for item in result["replays"]] == list(SCENES)
    assert all(
        item["complete"] is True
        and item["selected_indices_exact"] is True
        and item["native_actions_exact"] is True
        for item in result["replays"]
    )
    for item in result["replays"]:
        assert item["prediction"]["selected_indices"] == [0]
        assert item["prediction"]["actions"] == [[0.125, 0.0, 0.0, -1.0]]
    work = result["work"]
    assert work["prediction_attempts"] == work["prediction_returns"] == 4
    assert work["query_rows_attempted"] == work["query_rows_returned"] == 4
    assert work["distance_candidate_upper_bound"] == work["distance_candidates_completed"] == 28
    assert work["distance_feature_upper_bound"] == work["distance_feature_values_completed"] == 336
    assert work["validated_replays"] == 4 and work["unknown_unreturned_prediction_work"] is False
    assert work["simulator_controls"] == 0
    stream = result["stream"]
    assert stream["complete"] is True and stream["fixed_input"] is True
    assert stream["model_state"]["source_publications"] == 1
    assert stream["model_state"]["candidate_publications"] == 7
    assert stream["model_state"]["candidate_prototype_rows"] == 7
    assert len(stream["applied"]) == 6 and len(stream["checkpoints"]) == 13
    assert stream["work"]["fit_attempts"] == stream["work"]["model_random_draws"] == 0
    assert stream["work"]["warm_append_rows"] + stream["work"]["runtime_append_rows"] == 7
    assert "learner" not in result and "candidate" not in result
    assert len(payload["training"]) == 7 and len(payload["replay"]) == 4


def test_should_keep_recorded_negative_scores_separate_and_preserve_corrective_student_history(
    successful,
):
    result, payload = successful
    assert result["recorded_outcomes"] == [
        dict(scene=item["scene"], seed=item["seed"], outcomes=item["recorded_outcomes"])
        for item in payload["replay"]
    ]
    assert all(item["outcomes"]["final_success"] is False for item in result["recorded_outcomes"])
    correction = payload["training"][4]["episode"]["transitions"][0]
    assert correction["action"][3] == -1.0 and correction["next_observation"][10] == 1.0
    original = payload["replay"][0]["recorded_outcomes"]["sparse_return"]
    try:
        payload["replay"][0]["recorded_outcomes"]["sparse_return"] = 99.0
        assert result["recorded_outcomes"][0]["outcomes"]["sparse_return"] == 0.0
    finally:
        payload["replay"][0]["recorded_outcomes"]["sparse_return"] = original


def test_should_keep_first_paid_prediction_and_all_ingestion_charges_before_refusing_recorded_selector():
    payload = bundle()
    payload["replay"][0]["selected_indices"] = [1]
    original = deepcopy(payload)
    result = demo.run_demo(payload)
    assert result["complete"] is False and result["error"]
    assert len(result["replays"]) == 1
    first = result["replays"][0]
    assert first["scene"] == "pick19" and first["complete"] is False
    assert first["prediction"]["selected_indices"] == [0]
    assert first["prediction"]["distance_candidates"] == 7
    assert first["selected_indices_exact"] is False
    assert result["work"]["prediction_attempts"] == result["work"]["prediction_returns"] == 1
    assert result["work"]["validated_replays"] == 0
    assert result["work"]["unknown_unreturned_prediction_work"] is False
    assert len(result["stream"]["applied"]) == 6
    assert result["stream"]["work"]["validated_updates"] == 6
    assert result["stream"]["complete"] is False
    assert payload == original


def test_should_distinguish_a_returned_invalid_receipt_from_an_unreturned_prediction(monkeypatch):
    predict = nearest.NumpyNearestContinuousBCLearner.predict_with_work

    def malformed(owner, features):
        returned = predict(owner, features)
        return replace(returned, distance_candidates=8)

    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", malformed)
    result = demo.run_demo(bundle())
    assert result["complete"] is False and len(result["replays"]) == 1
    assert result["replays"][0]["prediction"]["distance_candidates"] == 8
    work = result["work"]
    assert work["prediction_attempts"] == work["prediction_returns"] == 1
    assert work["unknown_unreturned_prediction_work"] is False
    assert work["unknown_unvalidated_prediction_work"] is True
    assert work["distance_candidate_upper_bound"] == 7
    assert work["distance_candidates_completed"] == 0
