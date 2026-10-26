"""One shared tiny fixed MLP sequence; direct_reference must run once globally.

Each pytest invocation requesting ``completed`` executes exactly24 runtime SGD.
The separately selected direct_reference test adds24 standalone SGD. CLI/refusal
checks never execute a model update. Each partial-failure case adds exactly1 SGD
and consumes one further prospectively allocated runtime slot per invocation.
The final-empty-poll regression adds24 SGD and a fourth runtime slot. Thus a
full invocation uses50 runtime SGD/four slots, plus24 direct SGD only when the
direct_reference test is explicitly admitted. Runner owns aggregate cost.
"""

from dataclasses import asdict, replace
import json
import sys

import numpy as np
import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_learnability_demo as demo
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization


def episode(name="A", *, role="train"):
    observations = (
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        if name == "A"
        else [(10.0, 1.0), (11.0, 1.0), (12.0, 1.0), (13.0, 1.0)]
    )
    actions = [(-2.0, -1.0), (6.0, 1.0)] if name == "A" else [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0)]
    declared = ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0))
    return ContinuousEpisode(
        "synthetic-" + name,
        "context-" + name,
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                action,
                0.0,
                observations[index + 1],
                index == len(actions) - 1,
                False,
            )
            for index, action in enumerate(actions)
        ),
    )


def same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(first, group), getattr(second, group), strict=True):
            np.testing.assert_array_equal(left, right)


@pytest.fixture(scope="module")
def completed():
    """ONE24-update run shared by all complete-state assertions in this process."""
    warm, new = episode(), episode("B")
    captured = dict(
        fits=[], trains=[], runtimes=[], polls=[], endpoints={}, actors=[], diagnostics=[]
    )
    original_runtime = demo.ActorShadowRuntime
    original_fit = demo.fit_training_standardization
    original_train = NumpyContinuousBCLearner.train_batch

    with pytest.MonkeyPatch.context() as patch:

        def fit(record):
            captured["fits"].append(record)
            return original_fit(record)

        def train(owner, features, targets):
            captured["trains"].append((owner, len(features)))
            return original_train(owner, features, targets)

        def runtime(source, **options):
            owner = original_runtime(source, **options)
            captured["runtimes"].append((source, owner, options))
            captured["endpoints"][0] = owner.candidate_snapshot().state
            original_ready = owner.train_ready

            def ready(*args, **kwargs):
                before = owner.candidate_snapshot().state
                previous = options["budget"].updates_completed
                result = original_ready(*args, **kwargs)
                after = owner.candidate_snapshot().state
                current = options["budget"].updates_completed
                captured["polls"].append((options["clock"].now(), previous, current, result))
                if not result:
                    same_state(before, after)
                if result and current in (8, 16, 24):
                    captured["endpoints"][current] = after
                return result

            original_actor_predict = owner.actor.predict

            def actor_predict(features):
                result = original_actor_predict(features)
                captured["actors"].append((features.copy(), result))
                return result

            patch.setattr(owner, "train_ready", ready)
            patch.setattr(owner.actor, "predict", actor_predict)
            return owner

        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", train)
        originals = asdict(warm), asdict(new)
        report = demo.run_continuous_learnability_demo(
            warm,
            new,
            expected_standardization=ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2),
        )
        assert (asdict(warm), asdict(new)) == originals
    return report, captured, warm, new


def test_should_use_one_frozen_a_fit_default_policy_and_private_24_update_sequence(completed):
    report, captured, warm, new = completed
    assert report["complete"] is True and "error" not in report
    assert report["schema_version"] == 1 and report["protocol_id"] == "continuous_mlp_train_aba_v1"
    assert report["scope"] == "saved_train_imitation"
    assert captured["fits"] == [warm]
    assert report["expected_standardization_checked"] is True
    policy = report["policy"]
    assert policy["seed"] == 17 and policy["hidden_dim"] == 64
    assert policy["learning_rate"] == 0.01 and policy["momentum"] == 0.9
    assert policy["training_role"] == "train" and policy["binary_gripper_axis"] is None
    assert policy["observation_standardization"] == dict(
        means=(1.0, 0.0), scales=(1.0, 1.0), fitted_rows=2
    )
    assert report["source_episodes"] == {
        "warm": dict(episode_id=warm.episode_id, context=warm.context, role="train"),
        "new": dict(episode_id=new.episode_id, context=new.context, role="train"),
    }
    source, runtime, options = captured["runtimes"][0]
    assert len(captured["runtimes"]) == 1
    assert options["max_experiences"] == 24 and options["max_consolidations"] == 0
    assert options["budget"].budget.max_training_updates == 24
    assert options["budget"].updates_completed == 24
    assert options["budget"].replay_examples_completed == 0
    assert len(captured["trains"]) == 24
    assert all(owner is runtime._candidate for owner, _ in captured["trains"])
    assert [rows for _, rows in captured["trains"]] == [2] * 8 + [3] * 8 + [2] * 8
    same_state(captured["endpoints"][0], source.snapshot_state())
    same_state(captured["endpoints"][0], runtime.actor.snapshot_state().state)
    assert runtime.actor.version == "actor-0" and source.snapshot_state().completed_updates == 0
    assert report["progress"] == dict(
        update_attempts=24, updates_completed=24, replay_examples_completed=0, next_tick=48
    )
    assert report["settings"] == dict(
        phase_updates=8,
        phases=["warm", "new", "warm_return"],
        max_training_updates=24,
        promotion_calls=0,
        simulator_calls=0,
        probe_reads=0,
        final_reads=0,
        guard_reads=0,
    )
    json.dumps(report, allow_nan=False)


def test_should_wait_for_each_label_validate_every_receipt_and_preserve_all_losses(completed):
    report, captured, warm, new = completed
    assert len(report["events"]) == 24 and len(captured["polls"]) == 49
    assert len(report["applied_updates"]) == 24
    rows = 0
    for index, event in enumerate(report["events"]):
        record = new if 8 <= index < 16 else warm
        phase = "warm" if index < 8 else "new" if index < 16 else "warm_return"
        rows += len(record.transitions)
        assert event["index"] == index and event["phase"] == phase
        assert event["context"] == ("new" if record is new else "warm")
        assert event["observed_at"] == index * 2 and event["arrived_at"] == index * 2 + 1
        assert (
            event["deferred_exact"]
            is event["update_attempted"]
            is event["success_validated"]
            is True
        )
        assert (
            event["candidate_updates"] == index + 1 and event["candidate_training_examples"] == rows
        )
        (receipt,) = event["applied"]
        assert report["applied_updates"][index] == receipt
        assert (
            receipt["sample_id"] == f"mlp-batch-{index}"
            and receipt["event_id"] == f"mlp-label-{index}"
        )
        assert receipt["episode_id"] == record.episode_id and receipt["update_number"] == index + 1
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
        assert receipt["observed_at"] == index * 2
        assert receipt["arrived_at"] == receipt["applied_at"] == index * 2 + 1
        assert (
            receipt["diagnostic"]["definition"]
            == "continuous_bc_half_mean_active_axis_normalized_squared_error_v1"
        )
        assert np.isfinite(receipt["diagnostic"]["value"])
        deferred, applied = captured["polls"][2 * index : 2 * index + 2]
        assert deferred[:3] == (2 * index, index, index) and deferred[3] == ()
        assert applied[:3] == (2 * index + 1, index, index + 1) and len(applied[3]) == 1
    assert captured["polls"][-1][:3] == (47, 24, 24) and captured["polls"][-1][3] == ()


def test_should_keep_fixed_endpoint_order_owned_diagnostics_and_stable_actor_predictions(completed):
    report, captured, warm, new = completed
    assert [row["updates"] for row in report["checkpoints"]] == [0, 8, 16, 24]
    assert [row["training_examples"] for row in report["checkpoints"]] == [0, 16, 40, 56]
    assert len(captured["actors"]) == 8
    for endpoint, row in enumerate(report["checkpoints"]):
        assert row["state_exact"] is row["owners_exact"] is True
        assert list(row["contexts"]) == ["warm", "new"]
        for context, record, offset in (("warm", warm, 0), ("new", new, 1)):
            result = row["contexts"][context]
            assert result["episode_id"] == record.episode_id
            conditioning = result["conditioning"]
            residual = result["residual"]
            assert (
                conditioning["standardization_calls"] == 1
                and conditioning["standardization_applied"] is True
            )
            assert (
                conditioning["standardization_rows"]
                == residual["prediction_rows"]
                == len(record.transitions)
            )
            assert conditioning["standardization_values"] == len(record.transitions) * 2
            assert residual["prediction_calls"] == 1
            overall = residual["residuals"]["overall"]
            assert overall["active_axes"] == 2 and len(overall["axes"]) == 2
            (stage,) = residual["residuals"]["stages"]
            assert stage["label"] == "all" and tuple(stage["row_indices"]) == tuple(
                range(len(record.transitions))
            )
            assert stage["summary"] == overall
            assert conditioning["conditioning"]["objective_loss"] == pytest.approx(
                overall["normalized_half_mse"], abs=1e-14
            )
            expected = report["checkpoints"][0]["contexts"][context]["actor_predictions"]
            assert result["actor_predictions"] == expected
            actual = captured["actors"][endpoint * 2 + offset][1]
            assert actual.actor_version == "actor-0"
            np.testing.assert_array_equal(actual.prediction, expected)


def test_should_report_signed_metrics_without_requiring_a_favorable_learning_result(completed):
    report, _, _, _ = completed
    values = [
        {
            key: value["residual"]["residuals"]["overall"]["normalized_half_mse"]
            for key, value in row["contexts"].items()
        }
        for row in report["checkpoints"]
    ]
    assert report["metrics"] == dict(
        warm_acquisition_gain=values[0]["warm"] - values[1]["warm"],
        new_acquisition_gain=values[1]["new"] - values[2]["new"],
        warm_retention_loss=values[2]["warm"] - values[1]["warm"],
        warm_return_recovery=values[2]["warm"] - values[3]["warm"],
        warm_return_deficit=values[3]["warm"] - values[1]["warm"],
        new_retention_loss=values[3]["new"] - values[2]["new"],
    )


def test_should_account_training_diagnostics_actor_normalization_memory_and_exposures(completed):
    report, _, _, _ = completed
    work = report["work"]
    expected = dict(
        fit_calls=1,
        fit_rows=2,
        fit_feature_values=4,
        optimizer_updates=24,
        training_rows=56,
        unique_source_rows=5,
        repeated_training_rows=51,
        replay_budget_rows=0,
        conditioning_calls=8,
        conditioning_rows=20,
        residual_calls=8,
        residual_rows=20,
        actor_prediction_calls=8,
        actor_prediction_rows=20,
        normalization_calls=48,
        normalization_rows=116,
        normalization_values=232,
        forward_passes=56,
        forward_rows=136,
        backward_passes=32,
        backward_rows=76,
        gradient_values=10304,
        gradient_numeric_bytes=82432,
        dense_multiply_accumulates=64000,
        parameter_values_per_owner=322,
        model_numeric_bytes_per_owner=5152,
        statistics_numeric_bytes_per_owner=32,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=1792,
        residual_prediction_bytes=320,
        actor_prediction_bytes=320,
    )
    for key, value in expected.items():
        assert work[key] == value, key


def test_should_match_one_direct_reference_every_endpoint_full_momentum_and_all_24_losses(
    completed,
):
    """Budget: exactly24 direct SGD; select this test once across all gates."""
    report, captured, warm, new = completed
    source = captured["runtimes"][0][0]
    direct = NumpyContinuousBCLearner(source.policy)
    same_state(captured["endpoints"][0], direct.snapshot_state())
    expected_losses = []
    for index in range(24):
        record = new if 8 <= index < 16 else warm
        diagnostic = direct.train_episode(record)
        expected_losses.append(diagnostic.value)
        if index + 1 in (8, 16, 24):
            same_state(captured["endpoints"][index + 1], direct.snapshot_state())
    observed_losses = [event["applied"][0]["diagnostic"]["value"] for event in report["events"]]
    assert observed_losses == expected_losses


@pytest.mark.parametrize("failure", ["corrupt_receipt", "postcommit_throw"])
def test_should_stop_after_one_success_and_preserve_the_permanent_receipt(monkeypatch, failure):
    """Budget: exactly1 real SGD per case, each charged as a runtime slot."""
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(source, **options):
        owner = original(source, **options)
        captured.append((source, owner, options, source.snapshot_state()))
        ready = owner.train_ready

        def changed_receipt(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                assert options["budget"].updates_completed == 1
                if failure == "postcommit_throw":
                    raise RuntimeError("synthetic postcommit return failed")
                return (replace(applied[0], arrived_at=applied[0].arrived_at + 1),)
            return applied

        monkeypatch.setattr(owner, "train_ready", changed_receipt)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    report = demo.run_continuous_learnability_demo(episode(), episode("B"))
    source, owner, options, initial = captured[0]
    assert report["complete"] is False
    assert (
        "receipt/state/budget"
        if failure == "corrupt_receipt"
        else "synthetic postcommit return failed"
    ) in report["error"]
    assert report["progress"]["update_attempts"] == report["progress"]["updates_completed"] == 1
    assert len(report["events"]) == len(report["checkpoints"]) == 1
    assert "work" not in report and "metrics" not in report
    event = report["events"][0]
    assert event["deferred_exact"] is event["update_attempted"] is True
    assert event["success_validated"] is False
    if failure == "corrupt_receipt":
        assert event["candidate_updates"] == 1
        assert event["applied"][0]["arrived_at"] == 2
    else:
        assert event["applied"] == [] and "candidate_updates" not in event
    assert len(report["applied_updates"]) == 1
    assert (
        report["applied_updates"][0]["arrived_at"]
        == report["applied_updates"][0]["applied_at"]
        == 1
    )
    assert np.isfinite(report["applied_updates"][0]["diagnostic"]["value"])
    assert owner.applied_updates[0].arrived_at == 1
    assert (
        options["budget"].updates_completed == 1
        and options["budget"].budget.max_training_updates == 24
    )
    same_state(initial, source.snapshot_state())
    same_state(initial, owner.actor.snapshot_state().state)


def test_should_refuse_final_empty_poll_candidate_mutation_and_keep_all_receipts(monkeypatch):
    """Budget: one further24-SGD runtime slot; no direct-reference calls."""
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(source, **options):
        owner = original(source, **options)
        initial = source.snapshot_state()
        captured.append((source, owner, options, initial))
        ready = owner.train_ready

        def mutate_empty(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if not applied and options["budget"].updates_completed == 24:
                state = owner._candidate.snapshot_state()
                state.parameters[0][0, 0] += 0.5
                owner._candidate.restore_state(state)
            return applied

        monkeypatch.setattr(owner, "train_ready", mutate_empty)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    report = demo.run_continuous_learnability_demo(episode(), episode("B"))
    source, owner, options, initial = captured[0]
    assert report["complete"] is False and "final empty poll" in report["error"]
    assert len(report["events"]) == len(report["applied_updates"]) == 24
    assert all(event["success_validated"] is True for event in report["events"])
    assert len(report["checkpoints"]) == 4 and all(
        row["state_exact"] is True for row in report["checkpoints"]
    )
    assert report["progress"]["updates_completed"] == options["budget"].updates_completed == 24
    assert report["progress"]["update_attempts"] == 24 and report["progress"]["next_tick"] == 48
    assert "metrics" not in report and "work" not in report
    assert all(np.isfinite(item["diagnostic"]["value"]) for item in report["applied_updates"])
    same_state(initial, source.snapshot_state())
    same_state(initial, owner.actor.snapshot_state().state)


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
@pytest.mark.parametrize("position", ["warm", "new"])
def test_should_refuse_every_non_train_role_before_fitting_or_constructing_models(
    monkeypatch, role, position
):
    warm = episode(role=role) if position == "warm" else episode()
    new = episode("B", role=role) if position == "new" else episode("B")
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("held-out role reached fitting"),
    )
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("held-out role constructed a learner"),
    )
    with pytest.raises(ValueError, match="TRAIN"):
        demo.run_continuous_learnability_demo(warm, new)


@pytest.mark.parametrize("bad", ["identity", "spec", "missing", "subclass"])
def test_should_refuse_invalid_source_bindings_before_fit_or_model(monkeypatch, bad):
    warm, new = episode(), episode("B")
    if bad == "identity":
        new = replace(new, episode_id=warm.episode_id)
    elif bad == "spec":
        new = replace(new, spec=ContinuousActionSpec(2, ("m", "m"), (-3.0, -1.0), (6.0, 1.0)))
    elif bad == "missing":
        new = None
    else:

        class DerivedEpisode(ContinuousEpisode):
            pass

        new = DerivedEpisode(new.episode_id, new.context, new.role, new.spec, new.transitions)
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("invalid inputs reached fitting"),
    )
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("invalid inputs constructed a learner"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability_demo(warm, new)


@pytest.mark.parametrize("bad", ["means", "rows", "type"])
def test_should_compare_one_a_fit_to_frozen_reference_before_any_model_or_sgd(monkeypatch, bad):
    reference = (
        ObservationStandardization((2.0, 0.0), (1.0, 1.0), 2)
        if bad == "means"
        else (
            ObservationStandardization((1.0, 0.0), (1.0, 1.0), 3)
            if bad == "rows"
            else {"means": [1.0, 0.0]}
        )
    )
    calls = []
    original = demo.fit_training_standardization

    def fit(record):
        calls.append(record)
        return original(record)

    monkeypatch.setattr(demo, "fit_training_standardization", fit)
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("mismatched reference admitted a model"),
    )
    with pytest.raises(ValueError, match="frozen standardization"):
        demo.run_continuous_learnability_demo(
            episode(), episode("B"), expected_standardization=reference
        )
    assert len(calls) == 1 and calls[0].episode_id == "synthetic-A"


@pytest.mark.parametrize("scenario", ["continuous-mlp-learnability", "continuous-nearest"])
@pytest.mark.parametrize("visual", [False, True])
def test_should_route_existing_cli_with_two_fixed_teacher_reads_without_model_work(
    monkeypatch, capsys, scenario, visual
):
    records = [object(), object()]
    reads, runners = [], []

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        return records[len(reads) - 1]

    def run(name):
        def execute(*args):
            runners.append((name, args))
            return dict(complete=True, route=name)

        return execute

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run("mlp"))
    monkeypatch.setattr(cli, "run_continuous_demo", run("nearest"))
    args = ["demo", "--scenario", scenario, "--warm-teacher", "A.json", "--new-teacher", "B.json"]
    if visual:
        args.append("--visual-observation")
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    options = {"visual_observation": True} if visual else {}
    assert reads == [("A.json", 17, options), ("B.json", 19, options)]
    assert runners == [
        ("mlp" if scenario == "continuous-mlp-learnability" else "nearest", tuple(records))
    ]
    result = json.loads(capsys.readouterr().out)
    if visual:
        assert result["input_observation"]["mode"] == "public_rgb"
        assert result["input_observation"]["per_file_byte_limit"] == 4194304
    else:
        assert "input_observation" not in result


@pytest.mark.parametrize("extra", [["--seed", "17"], ["--diagnose-units"], ["--seed", "23"]])
def test_should_refuse_seed_or_probe_selection_in_learning_cli_before_file_reads(
    monkeypatch, extra
):
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_a, **_k: pytest.fail("invalid learning CLI read a teacher")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_print_partial_learning_evidence_and_return_failure_status_without_retry(
    monkeypatch, capsys
):
    monkeypatch.setattr(cli, "_read_teacher", lambda *_a, **_k: object())
    calls = []
    partial = dict(
        complete=False,
        error="synthetic partial failure",
        applied_updates=[{"update_number": 1}],
        progress={"updates_completed": 1},
    )
    monkeypatch.setattr(
        cli, "run_continuous_learnability_demo", lambda *_: calls.append(True) or partial
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 1 and calls == [True]
    assert json.loads(capsys.readouterr().out) == partial


def test_should_preserve_original_stream_default_without_loading_teacher_files(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "run_demo",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"route": "original"},
    )
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_a, **_k: pytest.fail("default stream read continuous files")
    )
    monkeypatch.setattr(sys, "argv", ["demo"])
    cli.main()
    assert calls == [((23,), {})]
    assert json.loads(capsys.readouterr().out) == {"route": "original"}
