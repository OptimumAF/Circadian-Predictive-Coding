"""Explicit cue-relative TRAIN fitting through the existing delayed stream.

Only one case constructs native models: two updates/108 rows and twelve
predictions/108 rows. All other cases inspect admission, projection and CLI
behavior without fitting a model, collecting trajectories or accessing final data.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_constructive_fit as fit
from src.adapters import minigrid_probe_support as support
from src.adapters import minigrid_train_catalogue as catalogue
from src.adapters.cue_relative_torch_feature_learner import CueRelativeTorchFeatureSnapshot
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.core.categorical_input_bounds import categorical_input_bounds
from src.core.minigrid_cue_relative_encoding import CUE_RELATIVE_HISTORY_ENCODING_ID
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.core.minigrid_public_history import encode_public_history_episode
from src.shared.torch_runtime import require_torch
from test_minigrid_constructive_fit import assert_raw_scores, capture_stream
from test_minigrid_train_fit_bounds import histories, raw_sources
from test_torch_feature_learner import assert_tree
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.fixture(autouse=True)
def observed_native_work(monkeypatch, record_property):
    work = dict(updates=0, training_rows=0, predictions=0, prediction_rows=0)
    train, predict = TorchFeatureLearner.train_batch, TorchFeatureLearner.predict

    def counted_train(owner, features, targets):
        work["updates"] += 1
        work["training_rows"] += len(features)
        assert work["updates"] <= 2 and work["training_rows"] <= 108
        return train(owner, features, targets)

    def counted_predict(owner, features):
        work["predictions"] += 1
        work["prediction_rows"] += len(features)
        assert work["predictions"] <= 12 and work["prediction_rows"] <= 108
        return predict(owner, features)

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", counted_train)
    monkeypatch.setattr(TorchFeatureLearner, "predict", counted_predict)
    yield work
    for name, value in work.items():
        record_property(f"cue_relative_fit_native_{name}", value)


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("_learner", "AttributionStream", "ProcessRssSampler"):
        monkeypatch.setattr(fit, name, forbidden)


def relative_oracle(history):
    """Independent public integer oracle; targets/context never enter the map."""
    result = list(canonicalize_public_history(tuple(float(value) for value in history)))
    pixels = [
        tuple(round((history[i + j] + 1) * scale / 2) for j, scale in enumerate((10, 5, 2)))
        for i in range(0, 147, 3)
    ]
    cues = [color for object_code, color, _ in pixels if object_code == 6]
    assert len(cues) == 1 and cues[0] in (0, 2)
    for offset in (0, 148):
        for position in range(49):
            index = offset + 3 * position
            object_code = round((history[index] + 1) * 5)
            if object_code == 5:
                color = round((history[index + 1] + 1) * 2.5)
                assert color in (0, 2)
                result[index + 1] = 2 * 2 / 5 - 1 if color == cues[0] else -1.0
    return tuple(result)


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_bool_recipe_before_sources_or_resources(no_resources, monkeypatch, flag):
    monkeypatch.setattr(fit, "summarize_train_fit_bounds", forbidden)
    with pytest.raises(ValueError):
        fit.describe_protocol(cue_relative=flag)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), updates=2, cue_relative=flag)
    with pytest.raises(ValueError):
        catalogue.compose_catalogue_training(histories(), 1, cue_relative=flag)
    with pytest.raises(ValueError):
        support.describe_train_fit_bounds(cue_relative=flag)
    with pytest.raises(ValueError):
        support.summarize_train_fit_bounds(histories(), cue_relative=flag)


def test_should_refuse_unqualified_gradient_reader_before_any_resource(no_resources, monkeypatch):
    monkeypatch.setattr(fit, "summarize_train_fit_bounds", forbidden)
    monkeypatch.setattr(fit, "diagnose_prior_gradients", forbidden)
    with pytest.raises(ValueError):
        fit.describe_protocol(cue_relative=True, cue_turn_gradients=True)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), cue_relative=True, cue_turn_gradients=True)


def test_should_keep_default_protocol_and_explicitly_name_distinct_recipe(no_resources):
    original = fit.describe_protocol(updates=128)
    assert original == fit.describe_protocol(updates=128, cue_relative=False)
    relative = fit.describe_protocol(updates=128, cue_relative=True)
    assert relative["definition"] != original["definition"]
    assert relative["method"] == "public_cue_relative_bp"
    assert relative["encoding_id"] == CUE_RELATIVE_HISTORY_ENCODING_ID
    for key in (
        "feature_dim",
        "hidden_dim",
        "num_classes",
        "seed",
        "learning_rate",
        "momentum",
        "objective",
        "updates",
        "checkpoint_updates",
        "label_delay",
        "training_only",
        "heldout",
        "replay_rows",
        "probe_updates",
        "final_episodes",
        "training_rows",
        "score_forward_calls",
        "score_rows",
        "max_wall_seconds",
    ):
        assert relative[key] == original[key]
    assert relative["checkpoint_updates"] == [0, 128]
    assert relative["training_rows"] == 6912
    assert support.describe_train_fit_bounds() == support.describe_train_fit_bounds(
        cue_relative=False
    )
    declaration = support.describe_train_fit_bounds(cue_relative=True)
    assert declaration["definition"] != support.TRAIN_BOUNDS_DEFINITION
    assert declaration["encoding_id"] == CUE_RELATIVE_HISTORY_ENCODING_ID
    json.dumps(relative, allow_nan=False)


def test_should_validate_relative_schema_but_transport_unprojected_owned_current_rows():
    sources = histories()
    saved = pickle.dumps(sources)
    original = catalogue.compose_catalogue_training(sources, 2)
    relative = catalogue.compose_catalogue_training(sources, 2, cue_relative=True)
    raw = np.array([s.observation for e in sources for s in e.transitions])
    mapped = np.array([relative_oracle(tuple(row)) for row in raw])
    assert not np.array_equal(raw, mapped)
    assert relative.source_spans == original.source_spans
    assert relative.row_exposures == original.row_exposures == 108
    assert relative.validation_projection_rows == 108
    assert relative.input_array_bytes == original.input_array_bytes
    for before, after in zip(original.training, relative.training, strict=True):
        np.testing.assert_array_equal(after.features, raw)
        np.testing.assert_array_equal(before.features, after.features)
        np.testing.assert_array_equal(before.targets, after.targets)
        assert after.row_ids == before.row_ids
        assert not np.shares_memory(before.features, after.features)
    assert not np.shares_memory(relative.training[0].features, relative.training[1].features)
    assert pickle.dumps(sources) == saved


def test_should_compute_new_bounds_from_independent_mapping_without_input_or_rng_changes():
    sources = histories()
    saved = pickle.dumps(sources)
    rng = pickle.dumps((random.getstate(), np.random.get_state()))
    result = support.summarize_train_fit_bounds(sources, cue_relative=True)
    raw = np.array([s.observation for e in sources for s in e.transitions])
    targets = np.array([[s.action] for e in sources for s in e.transitions], dtype=np.float64)
    projected = np.array([relative_oracle(tuple(row)) for row in raw])
    assert result["canonical"] == asdict(categorical_input_bounds(projected, targets, 7))
    assert result["raw"] == asdict(categorical_input_bounds(raw, targets, 7))
    assert result["protocol"] == support.describe_train_fit_bounds(cue_relative=True)
    assert result["definition"] == result["protocol"]["definition"]
    assert result["encoding_id"] == CUE_RELATIVE_HISTORY_ENCODING_ID
    assert result["work"]["native_updates"] == result["work"]["native_score_forwards"] == 0
    assert result["work"]["admission_projection_rows"] == 108
    assert result["work"]["bound_projection_rows"] == 54
    assert pickle.dumps(sources) == saved
    assert pickle.dumps((random.getstate(), np.random.get_state())) == rng


@pytest.mark.parametrize("fault", ["bad_last_next_key", "missing_initial_cue", "bad_role"])
def test_should_admit_complete_relative_current_and_next_before_models(
    no_resources, monkeypatch, fault
):
    sources = histories()
    if fault == "bad_role":
        sources = (*sources[:-1], replace(sources[-1], role="final_test"))
    elif fault == "missing_initial_cue":
        # Replace every initial Ball with floor, keeping the complete history causal.
        def remove_cue(view):
            values = list(view)
            values[3] = values[148 + 3] = -0.4  # Public object3 floor at cell1.
            return tuple(values)

        source = sources[-1]
        sources = (
            *sources[:-1],
            replace(
                source,
                transitions=tuple(
                    replace(
                        s,
                        observation=remove_cue(s.observation),
                        next_observation=remove_cue(s.next_observation),
                    )
                    for s in source.transitions
                ),
            ),
        )
    else:
        step = sources[-1].transitions[-1]
        values = list(step.next_observation)
        values[148 + 7] = -0.6  # Valid public green1, unsupported key color.
        object.__setattr__(step, "next_observation", tuple(values))
    monkeypatch.setattr(support, "categorical_input_bounds", forbidden)
    with pytest.raises(ValueError):
        catalogue.compose_catalogue_training(sources, 1, cue_relative=True)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(sources, updates=2, cue_relative=True)


def test_should_describe_explicit_cli_without_reading_sources(monkeypatch, capsys):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", "--constructive-bp", "--cue-relative", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == fit.describe_protocol(cue_relative=True)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--cue-relative", "--describe"],
        ["--bounds-only", "--cue-relative", "--describe"],
        ["--constructive-bp", "--cue-relative", "--cue-turn-gradients", "--describe"],
    ],
)
def test_should_refuse_incompatible_cli_before_source_access(monkeypatch, arguments):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_dispatch_raw_saved_sources_to_only_explicit_constructive_recipe(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "sources.json"
    source.write_text(json.dumps(dict(training_sources=[asdict(e) for e in raw_sources()])))
    before, calls = source.read_bytes(), []

    def diagnose(episodes, **options):
        calls.append((episodes, options))
        return dict(status="complete", proof="dispatch only")

    monkeypatch.setattr(cli, "diagnose_train_fit", diagnose)
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--constructive-bp",
            "--cue-relative",
            "--fit-updates",
            "2",
            "--source",
            str(source),
        ],
    )
    cli.main()
    assert len(calls) == 1
    assert calls[0][1] == dict(updates=2, max_wall_seconds=90, cue_relative=True)
    assert calls[0][0] == tuple(encode_public_history_episode(e) for e in raw_sources())
    assert source.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_should_run_two_relative_updates_and_read_fixed_points_through_original_stream(
    monkeypatch, observed_native_work
):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    captured = capture_stream(monkeypatch)
    sources = histories()
    saved = pickle.dumps(sources)
    rng = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
    original_score, states = fit._score_point, []

    def score(source, state, point, episodes, report, guard):
        assert captured["stream"]._finished is True
        assert captured["stream"].completed_updates == 2
        assert type(state) is CueRelativeTorchFeatureSnapshot
        assert state.policy.encoding_id == CUE_RELATIVE_HISTORY_ENCODING_ID
        fit._counter(state, point, cue_relative=True)
        forged = SimpleNamespace(
            policy=state.policy, completed_updates=point, training_examples=54 * point
        )
        old = InvariantTorchFeatureSnapshot(
            InvariantTorchFeaturePolicy(state.policy.native_policy), state.native_state
        )
        wrong_clock = deepcopy(state)
        object.__setattr__(wrong_clock.native_state, "completed_updates", point + 1)
        for invalid in (
            forged,
            old,
            wrong_clock,
        ):
            with pytest.raises(ValueError):
                fit._counter(invalid, point, cue_relative=True)
        with pytest.raises(ValueError):
            fit._counter(state, point)
        states.append(deepcopy(state))
        return original_score(source, state, point, episodes, report, guard)

    monkeypatch.setattr(fit, "_score_point", score)
    result = fit.diagnose_train_fit(sources, updates=2, max_wall_seconds=10, cue_relative=True)
    assert result["status"] == "complete", result["reason"]
    assert result["definition"] == result["protocol"]["definition"]
    assert result["protocol"]["method"] == "public_cue_relative_bp"
    assert observed_native_work == dict(
        updates=2, training_rows=108, predictions=12, prediction_rows=108
    )
    assert (
        result["charged_updates"]
        == result["committed_updates"]
        == result["native_completed_updates"]
        == 2
    )
    assert result["native_training_rows"] == 108
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert [s.completed_updates for s in states] == [0, 2]
    stream, initial = captured["stream"], captured["initial"]
    assert type(initial) is CueRelativeTorchFeatureSnapshot
    assert_tree(torch, captured["source"].snapshot_state(), initial)
    assert_tree(torch, stream._runtime.actor.snapshot_state().state, initial)
    raw = np.array([s.observation for e in sources for s in e.transitions])
    for packet in captured["training"]:
        np.testing.assert_array_equal(packet.features, raw)
        assert packet.role == "train"
    assert captured["heldout"] == ()
    assert captured["options"]["training_only"] is True
    assert captured["options"]["label_delay"] == 1
    budget = captured["options"]["budget"]
    assert budget.updates_completed == 2 and budget.budget.max_training_updates == 2
    assert result["budget"]["started_at"] == budget.started_at
    assert result["budget"]["limits"] == asdict(budget.budget)
    receipts = result["applied_updates"]
    assert [r["update_number"] for r in receipts] == [1, 2]
    assert all(
        r["arrived_at"] == r["observed_at"] + 1 and r["applied_at"] >= r["arrived_at"]
        for r in receipts
    )
    assert result["work"]["score_forward_calls_completed"] == 12
    assert result["work"]["score_rows_completed"] == 108
    assert result["work"]["score_work_may_be_unobserved"] is False
    assert result["work"]["training_work_may_be_unobserved"] is False
    assert_raw_scores(result)
    assert result["fit_gap_outcomes"] == support.fit_gap_outcomes(
        result["train_fit_bounds"], result["outcomes"]
    )
    assert pickle.dumps(sources) == saved
    assert random.getstate() == rng[0] and pickle.dumps(np.random.get_state()) == rng[1]
    assert torch.equal(torch.get_rng_state(), rng[2])


class RelativeScriptedLearner:
    """Typed outer checkpoint with an explicit scripted native payload, no Torch."""

    def __init__(self):
        from src.adapters.cue_relative_torch_feature_learner import CueRelativeTorchFeaturePolicy
        from src.adapters.torch_feature_learner import TorchFeaturePolicy
        from test_minigrid_learning_pilot import Learner

        self.policy = CueRelativeTorchFeaturePolicy(
            TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7)
        )
        self.script = Learner(self.policy)

    def snapshot_state(self):
        return CueRelativeTorchFeatureSnapshot(self.policy, self.script.snapshot_state())

    def restore_state(self, state):
        assert type(state) is CueRelativeTorchFeatureSnapshot and state.policy == self.policy
        self.script.restore_state(state.native_state)

    def fork(self):
        result = RelativeScriptedLearner()
        result.restore_state(self.snapshot_state())
        return result

    def train_batch(self, features, targets):
        return self.script.train_batch(features, targets)

    def predict(self, features):
        return self.script.predict(features)


@pytest.mark.parametrize("fault", ["train_error", "wall", "score_error"])
def test_should_keep_relative_paid_work_and_partial_scores_when_later_work_fails(
    monkeypatch, fault, observed_native_work
):
    from test_minigrid_learning_pilot import Learner

    captured = capture_stream(monkeypatch)
    monkeypatch.setattr(Learner, "work", {})
    monkeypatch.setattr(fit, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    clock = [1.0]
    monkeypatch.setattr(fit, "monotonic", lambda: clock[0])

    def learner(*, cue_relative=False):
        assert cue_relative is True
        return RelativeScriptedLearner()

    monkeypatch.setattr(fit, "_learner", learner)
    if fault == "score_error":

        def prediction(owner):
            if len(Learner.work["predicts"]) == 2:
                raise RuntimeError("injected second relative TRAIN read failure")

        Learner.work["on_predict"] = prediction
    else:

        def after_train(owner):
            if fault == "train_error":
                raise RuntimeError("injected paid relative native failure")
            clock[0] = 1000.0

        Learner.work["on_train"] = after_train
    sources = histories()
    saved = pickle.dumps(sources)
    result = fit.diagnose_train_fit(sources, updates=2, max_wall_seconds=10, cue_relative=True)
    assert result["status"] == "incomplete" and result["reason"]
    assert result["fit_witness"] is False and "outcomes" not in result
    assert result["protocol"]["encoding_id"] == CUE_RELATIVE_HISTORY_ENCODING_ID
    assert result["native_completed_updates"] == (2 if fault == "score_error" else 1)
    assert result["native_training_rows"] == (108 if fault == "score_error" else 54)
    assert result["charged_updates"] == {"train_error": 0, "wall": 1, "score_error": 2}[fault]
    assert len(result["applied_updates"]) == result["charged_updates"]
    assert result["work"]["training_work_may_be_unobserved"] is (fault != "score_error")
    assert result["budget"]["started_at"] == 1.0
    if fault == "score_error":
        assert len(result["readouts"]) == 1 and len(result["readouts"][0]["scores"]) == 1
        assert result["work"]["score_forward_calls_attempted"] == 2
        assert result["work"]["score_forward_calls_completed"] == 1
        assert result["work"]["score_work_may_be_unobserved"] is True
    else:
        assert result["readouts"] == []
    initial = captured["initial"]
    assert pickle.dumps(captured["source"].snapshot_state()) == pickle.dumps(initial)
    assert pickle.dumps(captured["stream"]._runtime.actor.snapshot_state().state) == pickle.dumps(
        initial
    )
    assert pickle.dumps(sources) == saved
    assert observed_native_work == dict(
        updates=0, training_rows=0, predictions=0, prediction_rows=0
    )
