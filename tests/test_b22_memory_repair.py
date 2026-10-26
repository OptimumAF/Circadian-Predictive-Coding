"""Six fixed-input releases preserve all prior rows and their original histories.

One complete220-row stream and one seven-row late observer refusal use two
nearest constructors/fourteen appends227rows. One detached final-bank query
batch has two rows/440candidates5280feature visits. No empirical fitting or neural work.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_stream_sequence as sequence
from src.adapters import numpy_nearest_continuous_bc as nearest
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history

NAMES = ("a", "b50", "train20", "push", "corrections", "current", "b22")
COUNTS = (15, 50, 15, 18, 50, 50, 22)


def episode(name, count, value):
    history = ActionHistoryState()
    rows = []
    command = (value, 0.0, 0.0, -1.0)
    for index in range(count):
        before = (
            0.0,
            float(index),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            *history.previous_action,
            history.closing_count / 6,
        )
        history = advance_action_history(history, command)
        after = (
            0.0,
            float(index + 1),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            *history.previous_action,
            history.closing_count / 6,
        )
        rows.append(
            ContinuousTransition(index, before, command, 0.0, after, False, index == count - 1)
        )
    context = "pick-cube-train-19" if name in ("b50", "b22") else "TRAIN-" + name
    return ContinuousEpisode(
        "synthetic-" + name, context, "train", cube_taskspace_spec(), tuple(rows)
    )


def records(*, tiny=False):
    return tuple(
        episode(name, 1 if tiny else count, index / 8)
        for index, (name, count) in enumerate(zip(NAMES, COUNTS, strict=True))
    )


def forbidden(*args, **options):
    pytest.fail("refused six-release inputs reached constructor, fitting or predictor")


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
        counts["query_attempts"] += 1
        counts["query_rows"] += len(features)
        result = predict(owner, features)
        counts["query_returns"] += 1
        counts["distance_candidates"] += result.distance_candidates
        counts["distance_feature_values"] += result.distance_feature_values
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", initialized)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "train_batch", appended)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", predicted)
        patch.setattr(sequence, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("B22_MEMORY_REPAIR_LEDGER"):
        with Path(os.environ["B22_MEMORY_REPAIR_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(dict(counts), output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 2
    assert counts["append_attempts"] == counts["append_returns"] == 14
    assert counts["append_rows"] == 227
    assert counts["query_attempts"] == counts["query_returns"] == 1 and counts["query_rows"] == 2
    assert counts["distance_candidates"] == 440 and counts["distance_feature_values"] == 5280
    assert counts["artifact_case_runs"] == counts["artifact_stream_doubles"] == 3
    assert counts["artifact_pick_doubles"] == counts["artifact_current_mutations"] == 3


@pytest.fixture(scope="module")
def completed(ledger):
    data = records()
    before = tuple(asdict(item) for item in data)
    captured: dict[str, Any] = dict(metadata=[])
    factory = sequence.ActorShadowRuntime

    def runtime(source, **options):
        value = factory(source, **options)
        captured.update(source=source, runtime=value, budget=options["budget"])
        return value

    def observe(readout, metadata):
        captured["metadata"].append(deepcopy(metadata))
        if metadata["phase"] == "after_release" and metadata["packet_index"] == 4:
            captured["prefix"] = readout
            captured["prefix_initial"] = readout.snapshot_state()
        elif metadata["phase"] == "after_release" and metadata["packet_index"] == 5:
            captured["final"] = readout
        # Only the callback's owned metadata changes, never parent evidence.
        metadata["tick"] = -99
        metadata["known_episode_ids"].append("observer-only")
        return dict(scope="metadata only; no prediction or new authority")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sequence, "ActorShadowRuntime", runtime)
        report = sequence.run_sequence(
            data[0], data[1:], observe, fixed_input=True, maximum_releases=6
        )
    assert tuple(asdict(item) for item in data) == before
    return report, captured, data


def test_should_append_all220_in_order_and_keep_six_independent_permanent_delayed_charges(
    completed,
):
    report, captured, data = completed
    assert report["complete"] is True, report
    assert report["release_scope"] == dict(
        maximum_releases=6,
        default_maximum_releases=3,
        scope="explicit fixed-input six-release sequence; no prior budget renewal",
    )
    assert [row["arrived_at"] for row in report["applied"]] == [2, 5, 8, 11, 14, 17]
    assert [row["observed_at"] for row in report["applied"]] == [0, 3, 6, 9, 12, 15]
    assert [row["episode_id"] for row in report["applied"]] == [
        item.episode_id for item in data[1:]
    ]
    assert report["experiment_state"] == dict(completed_updates=6, next_tick=18)
    assert (
        captured["budget"].budget.max_training_updates == captured["budget"].updates_completed == 6
    )
    assert captured["budget"].replay_examples_completed == 0
    assert report["model_state"] == dict(
        source_publications=1,
        candidate_publications=7,
        candidate_prototype_rows=220,
        durable_checkpoint=False,
    )
    assert len(report["events"]) == 18 and len(report["checkpoints"]) == 13
    assert report["work"]["validated_updates"] == 6 and report["work"]["runtime_append_rows"] == 205
    assert (
        report["work"]["fit_attempts"]
        == report["work"]["model_random_draws"]
        == report["work"]["optimizer_updates"]
        == 0
    )
    final = captured["final"].snapshot_state()
    assert final.prototype_features.tolist() == [
        list(row.observation) for item in data for row in item.transitions
    ]
    assert final.prototype_actions.tolist() == [
        list(row.action) for item in data for row in item.transitions
    ]
    assert report["checkpoints"][-1]["known_episode_ids"] == [item.episode_id for item in data]
    assert all(row["tick"] >= 0 for row in report["checkpoints"])


def test_should_preserve_original198_prefix_warm_source_actor_and_episode_resets(completed):
    report, captured, data = completed
    prefix, expected = captured["prefix"].snapshot_state(), captured["prefix_initial"]
    assert prefix.completed_updates == 6 and prefix.training_examples == 198
    np.testing.assert_array_equal(prefix.prototype_features, expected.prototype_features)
    np.testing.assert_array_equal(prefix.prototype_actions, expected.prototype_actions)
    final = captured["final"].snapshot_state()
    np.testing.assert_array_equal(final.prototype_features[:198], prefix.prototype_features)
    np.testing.assert_array_equal(final.prototype_actions[:198], prefix.prototype_actions)
    source = captured["source"].snapshot_state()
    actor = captured["runtime"].actor.snapshot_state().state
    assert source.training_examples == actor.training_examples == 15
    np.testing.assert_array_equal(source.prototype_features, actor.prototype_features)
    assert not np.shares_memory(prefix.prototype_features, final.prototype_features)
    assert not np.shares_memory(final.prototype_features, source.prototype_features)
    assert data[1].context == data[-1].context and data[1].episode_id != data[-1].episode_id
    assert data[-1].transitions[0].observation[7:] == (0.0,) * 5
    assert final.prototype_features[198, 7:].tolist() == [0.0] * 5
    assert report["memory"]["candidate_prototype_bytes"] == 28160
    assert report["memory"]["retained_episode_rows"] == 220
    assert "candidate" not in report and "learner" not in report


def test_should_keep_earliest_reset_tie_and_retrieve_new_b22_history_from_final_detached_bank(
    completed,
):
    _, captured, data = completed
    readout = captured["final"]
    before = readout.snapshot_state()
    query = np.asarray([data[-1].transitions[index].observation for index in (0, 1)], np.float64)
    result = readout.predict_with_work(query)
    assert result.selected_indices == (0, 199)
    assert result.actions.tolist() == [
        list(data[0].transitions[0].action),
        list(data[-1].transitions[1].action),
    ]
    assert result.distance_candidates == 440 and result.distance_feature_values == 5280
    after = readout.snapshot_state()
    np.testing.assert_array_equal(after.prototype_features, before.prototype_features)
    np.testing.assert_array_equal(after.prototype_actions, before.prototype_actions)


def test_should_keep_all_six_paid_updates_when_last_returned_readout_is_corrupted():
    data = records(tiny=True)

    def observe(readout, metadata):
        if metadata["phase"] == "after_release" and metadata["packet_index"] == 5:
            readout._state.prototype_actions[0, 0] = 0.875
        return dict(phase=metadata["phase"], index=metadata["packet_index"])

    report = sequence.run_sequence(data[0], data[1:], observe, fixed_input=True, maximum_releases=6)
    assert report["complete"] is False
    assert len(report["applied"]) == len(report["retained_applied_on_failure"]) == 6
    assert [row["arrived_at"] for row in report["applied"]] == [2, 5, 8, 11, 14, 17]
    assert report["experiment_state"] == dict(completed_updates=6, next_tick=18)
    assert report["work"]["runtime_append_returns"] == report["work"]["validated_updates"] == 6
    assert report["work"]["observer_attempts"] == report["work"]["observer_returns"] == 13
    assert report["work"]["observer_validated"] == 12
    assert report["checkpoints"][-1]["result"] == dict(phase="after_release", index=5)
    assert report["observed_native_on_failure"]["candidate"]["completed_updates"] == 7
    assert report["observed_native_on_failure"]["candidate"]["training_examples"] == 7
    assert report["observed_native_on_failure"]["source"]["training_examples"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "nonfixed",
        "bool",
        "float",
        "none",
        "seven",
        "over6",
        "role",
        "duplicate",
        "capacity",
        "spec",
        "five",
    ],
)
def test_should_refuse_bad_expanded_scope_or_inputs_before_any_constructor(monkeypatch, fault):
    data = records(tiny=True)
    maximum: Any = 6
    fixed = True
    if fault == "nonfixed":
        fixed = False
    elif fault == "bool":
        maximum = True
    elif fault == "float":
        maximum = 6.0
    elif fault == "none":
        maximum = None
    elif fault == "seven":
        maximum = 7
    elif fault == "over6":
        data = data + (episode("extra", 1, 0.875),)
    elif fault == "role":
        data = data[:-1] + (replace(data[-1], role="probe_test"),)
    elif fault == "duplicate":
        data = data[:-1] + (replace(data[-1], episode_id=data[1].episode_id),)
    elif fault == "capacity":
        data = tuple(
            episode(str(index), count, index / 8)
            for index, count in enumerate((37, 37, 37, 37, 37, 37, 35))
        )
    elif fault == "spec":
        item = data[-1]
        data = data[:-1] + (
            replace(
                item, spec=replace(item.spec, action_units=("other",) + item.spec.action_units[1:])
            ),
        )
    else:
        maximum = 5
    monkeypatch.setattr(sequence, "NumpyNearestContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        sequence.run_sequence(data[0], data[1:], fixed_input=fixed, maximum_releases=maximum)


@pytest.mark.parametrize("fault", ["source_read", "saved_read", "holder"])
def test_should_keep_current_paid_birth_and_refuse_getter_corruption_before_accepting_first_scene(
    tmp_path, monkeypatch, ledger, fault
):
    import runpy
    import sys
    from types import SimpleNamespace

    # The artifact performs only definitions at this import name. Isolate its
    # repository-path insertion so it cannot persist into another fixture.
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = Path(__file__).resolve().parents[1] / "artifacts/runs/b22-memory-repair-20261021/case.py"
    run = runpy.run_path(str(path), run_name="fixture")["run"]
    scope = run.__globals__
    data = records(tiny=True)
    bank = tuple(zip(NAMES[:-1], data[:-1], strict=True))
    receipt = {"sources": [{"name": "controlled-input"}]}
    features = [list(row.observation) for item in data for row in item.transitions]
    actions = [list(row.action) for item in data for row in item.transitions]

    class LiteralArray:
        def __init__(self, rows):
            self.rows = rows

        def tolist(self):
            return deepcopy(self.rows)

    owner = object()
    state = SimpleNamespace(
        prototype_features=LiteralArray(features), prototype_actions=LiteralArray(actions)
    )
    pin = (None, None, None, None, None, state)
    current: dict[str, Any] = {}
    calls: list[int] = []
    saved_reads: list[Path] = []
    summary = dict(final_success=False, any_success=False, sparse_return=0.0)
    reference = {
        "readouts": [{}, {"scenes": {scene: deepcopy(summary) for scene in scope["SCENES"]}}]
    }

    def mutate():
        if current and not current.get("mutated"):
            current["mutated"] = True
            if fault == "holder":
                current["holder"].report = deepcopy(current["raw"])
            else:
                current["raw"]["birth"] = "changed-after-return"
            ledger["artifact_current_mutations"] += 1

    def source_read(original, work):
        assert original is receipt
        work["attempts"] += 1
        work["returns"] += 1
        work["bytes"] += 1
        if fault in ("source_read", "holder"):
            mutate()

    def read_saved(path, work):
        saved_reads.append(path)
        work["attempts"] += 1
        mutate()
        # A matching saved file must not authenticate a changed live return.
        raw = json.dumps(current["raw"], sort_keys=True).encode()
        path.write_bytes(raw)
        work["returns"] += 1
        work["bytes"] += len(raw)
        return raw

    def native_pick(path, *, learner, train_seed, action_history_recipe):
        assert learner is owner and action_history_recipe.mode == "executed"
        calls.append(train_seed)
        ledger["artifact_pick_doubles"] += 1
        raw = dict(complete=True, birth="original-paid-return", **summary)
        path.write_text(json.dumps(raw))
        holder = SimpleNamespace(report=raw, corrections=None)
        current.update(raw=raw, holder=holder)
        return holder

    def stream(warm, releases, *, observer, fixed_input, maximum_releases):
        ledger["artifact_stream_doubles"] += 1
        assert warm is data[0] and releases == data[1:]
        assert fixed_input is True and maximum_releases == 6
        result: dict[str, Any] = {"complete": False}
        try:
            observer(
                owner,
                dict(
                    phase="after_release",
                    packet_index=5,
                    tick=17,
                    prototype_rows=220,
                    completed_updates=7,
                ),
            )
        except Exception as error:
            result["error"] = str(error)
        return result

    def require(readout, expected, **options):
        assert readout is owner and expected is pin

    replacements = dict(
        STAGE=tmp_path,
        read_pick19_inputs=lambda **options: (data[-1], data[1], data[1], bank, (0,), receipt),
        _read=lambda *args: json.dumps(reference).encode(),
        _require_sources=source_read,
        _read_saved=read_saved,
        _pin_push_nearest=lambda learner: pin,
        _require_push_nearest=require,
        _summary=lambda raw, **options: deepcopy(summary),
        run_sequence=stream,
        run_frozen_rollout=native_pick,
        roll_push=forbidden,
        _require_native=forbidden,
        _pick_reset=forbidden,
    )
    for name, value in replacements.items():
        monkeypatch.setitem(scope, name, value)
    ledger["artifact_case_runs"] += 1
    result = run()
    assert result["complete"] is False
    assert calls == [19] and result["native"] == []
    assert result["pending_paid_return"] == dict(
        scene="pick19", report=dict(complete=True, birth="original-paid-return", **summary)
    )
    assert result["returned"] == ["admission", "pick19", "stream"]
    assert result["unreturned_stage_work_unknown"] is False
    assert "current paid native return changed" in result["stream"]["error"]
    assert len(saved_reads) == (1 if fault == "saved_read" else 0)
    assert json.loads((tmp_path / "run.json").read_text()) == json.loads(json.dumps(result))
