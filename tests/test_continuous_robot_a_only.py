"""A-only training admitted by the existing frozen robot composition.

Per platform: four true initializations/seven PC+BC constructor entries draw
11776 normal values; twelve native updates train24 A rows. Four fits use eight
A rows/336 values.32 conditioning/residual/actor calls each read80 A+B rows.
Four fake scenes use eight predictions/controls, four resets/closes/name/limit
reads and twelve joint samples. PC<=45 energy calls/36 latent proposals.
Every other training/native call is stubbed and uses owned RNG-free forks.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_learnability_demo as demo
from src.core.continuous_demonstration import ContinuousTransition
from src.infra import maniskill_continuous as native
from test_continuous_robot_learning import FAMILIES, assert_state, episode as previous_episode
from test_maniskill_joint_positions import StateEnvironment


def episodes(*, visual=False):
    dimension = 789 if visual else 42
    warm = previous_episode(dimension=dimension)
    new = previous_episode(19, dimension=dimension)
    first, last = new.transitions
    terminal = (13.0, 1.0) + (0.0,) * (dimension - 2)
    extra = ContinuousTransition(
        2, last.next_observation, (0.3, -0.4, 0.2, 1.0), 1.0, terminal, True, False
    )
    new = replace(new, transitions=(first, replace(last, reward=0.0, terminated=False), extra))
    return (
        replace(warm, episode_id="pick-cube-readiness-teacher-17"),
        replace(new, episode_id="pick-cube-readiness-teacher-19"),
    )


def forbidden(*args, **kwargs):
    pytest.fail("invalid or stubbed A-only request admitted additional training/native work")


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    folder = tmp_path_factory.mktemp("robot-a-only")
    warm, new = episodes()
    reports, results, environments = {}, {}, {}
    captures: dict[str, Any] = {}
    original_train, original_fit, original_runtime = (
        entry.run_continuous_learnability,
        demo.fit_training_standardization,
        demo.ActorShadowRuntime,
    )
    current = [""]
    with pytest.MonkeyPatch.context() as patch:

        def fit(record, **options):
            captures[current[0]]["fits"].append((asdict(record), options))
            return original_fit(record, **options)

        def runtime(source, **options):
            created = original_runtime(source, **options)
            captured = captures[current[0]]
            captured.update(
                source=source, runtime=created, initial=source.snapshot_state(), options=options
            )
            original_experience, original_label = created.record_experience, created.record_label

            def experience(value):
                captured["experiences"].append((value.episode_id, value.features.tolist()))
                return original_experience(value)

            def label(value):
                captured["labels"].append((value.episode_id, value.targets.tolist()))
                return original_label(value)

            patch.setattr(created, "record_experience", experience)
            patch.setattr(created, "record_label", label)
            return created

        def train(*args, **options):
            assert options == dict(learner_family=current[0], phase_updates=1, a_only_control=True)
            result = original_train(*args, **options)
            results[current[0]] = result
            assert result.candidate is not None
            captures[current[0]]["before_native"] = result.candidate.snapshot_state()
            return result

        def factory(**options):
            assert options == {}
            value = StateEnvironment()
            environments[current[0]] = value
            return value

        patch.setattr(entry, "run_continuous_learnability", train)
        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(native, "_make_environment", factory)
        patch.setattr(smoke, "version", lambda _: "fake-a-only-robot")
        for family in FAMILIES:
            current[0] = family
            captures[family] = dict(fits=[], experiences=[], labels=[])
            reports[family] = entry.run_learning_rollout(
                folder / f"{family}.json",
                warm,
                new,
                learner_family=family,
                phase_updates=1,
                a_only_control=True,
            )
    return reports, results, captures, environments, warm, new


@pytest.mark.parametrize("family", FAMILIES)
def test_should_train_only_a_packets_while_preserving_b_as_diagnostics_and_serve_exact_candidate(
    learned, family
):
    reports, results, captures, environments, warm, new = learned
    report, result, captured = reports[family], results[family], captures[family]
    assert report["complete"] is report["a_only_control"] is report["candidate_unchanged"] is True
    assert "A-only" in report["scope"]
    expected_features = [list(row.observation) for row in warm.transitions]
    expected_targets = [list(row.action) for row in warm.transitions]
    assert captured["experiences"] == [(warm.episode_id, expected_features)] * 3
    assert captured["labels"] == [(warm.episode_id, expected_targets)] * 3
    assert captured["fits"] == [(asdict(warm), {})]
    assert new.episode_id != warm.episode_id and len(new.transitions) == 3
    training = result.report
    assert training["settings"]["training_contexts"] == ["warm"] * 3
    assert training["settings"]["diagnostic_contexts"] == ["warm", "new"]
    for checkpoint in training["checkpoints"]:
        assert checkpoint["contexts"]["new"]["episode_id"] == new.episode_id
        assert checkpoint["contexts"]["new"]["residual"]["residuals"]["overall"]["rows"] == 3
    state = result.candidate.snapshot_state()
    assert state.completed_updates == 3 and state.training_examples == 6
    assert_state(captured["before_native"], state)
    assert_state(captured["initial"], captured["source"].snapshot_state())
    assert_state(captured["initial"], captured["runtime"].actor.snapshot_state().state)
    if family in ("neutral-cpc", "active-cpc"):
        assert state.wake_state.wake_updates == 3
    served = report["rollout"]
    assert served["source_updates"] == 3 and served["source_training_rows"] == 6
    assert served["joint_position_exact"] is True and served["optimizer_updates"] == 0
    environment = environments[family]
    assert (
        environment.reset_calls == [17]
        and len(environment.commands) == 2
        and environment.close_calls == 1
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_should_keep_a_only_clocks_charges_and_physical_diagnostic_work_distinct(learned, family):
    reports, results, captures, _, warm, _ = learned
    training, work = results[family].report, results[family].report["work"]
    assert work["training_rows"] == 6 and work["unique_source_rows"] == 5
    assert work["unique_training_rows"] == 2 and work["repeated_training_rows"] == 4
    if family == "bp":
        assert work["training_rows_by_context"] == dict(warm=6, new=0)
        assert work["diagnostic_rows_by_context"] == dict(warm=8, new=12)
    else:
        assert (
            work["conditioning_rows"]
            == work["residual_rows"]
            == work["actor_prediction_rows"]
            == 20
        )
    for index, (event, receipt) in enumerate(
        zip(training["events"], training["applied_updates"], strict=True)
    ):
        assert (
            event["context"] == "warm"
            and event["deferred_exact"] is event["success_validated"] is True
        )
        assert event["observed_at"] == receipt["observed_at"] == 2 * index
        assert (
            event["arrived_at"] == receipt["arrived_at"] == receipt["applied_at"] == 2 * index + 1
        )
        assert receipt["episode_id"] == warm.episode_id and receipt["update_number"] == index + 1
        assert event["candidate_training_examples"] == 2 * (index + 1)
    assert captures[family]["options"]["budget"].updates_completed == 3
    assert captures[family]["options"]["budget"].replay_examples_completed == 0
    if family != "bp":
        assert [row["training_rows"] for row in training["native_updates"]] == [2, 2, 2]
        assert work["native_update_counts"]["published_updates"] == 3
    json.dumps(reports[family], allow_nan=False)


@pytest.mark.parametrize("value", [None, 1, 0, "true", 0.0])
def test_should_refuse_non_boolean_mode_before_input_output_or_training(
    monkeypatch, tmp_path, value
):
    monkeypatch.setattr(entry, "_admit_inputs", forbidden)
    monkeypatch.setattr(entry, "run_continuous_learnability", forbidden)
    output = tmp_path / "absent.json"
    with pytest.raises(ValueError):
        entry.run_learning_rollout(output, *episodes(), a_only_control=value)
    assert not output.exists()


@pytest.mark.parametrize("selected", [None, False, True])
def test_should_omit_false_control_kwargs_and_default_only_report_fields(
    monkeypatch, tmp_path, selected
):
    calls = []

    def train(*args, **options):
        calls.append(options)
        return demo.ContinuousLearnabilityResult(
            dict(complete=False, error="retained partial report"), None
        )

    monkeypatch.setattr(entry, "run_continuous_learnability", train)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    options = {} if selected is None else dict(a_only_control=selected)
    report = entry.run_learning_rollout(
        tmp_path / "failure.json", *episodes(), phase_updates=1, **options
    )
    expected = dict(learner_family="bp", phase_updates=1)
    if selected is True:
        expected["a_only_control"] = True
    assert calls == [expected]
    assert report["complete"] is False and report["training"]["error"] == "retained partial report"
    assert ("a_only_control" in report) == (selected is True)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_mode",
        "false_mode",
        "integer_mode",
        "training_contexts",
        "diagnostic_contexts",
        "event_context",
        "event_observed",
        "event_arrival",
        "receipt_episode",
        "receipt_sample",
        "receipt_observed",
        "receipt_update",
        "candidate_rows",
        "work_warm",
        "work_new",
        "unique_rows",
        "repeated_rows",
        "total_rows",
        "checkpoint_rows",
        "float_rows",
        "bool_replay",
        "float_progress",
        "float_phase",
        "native_rows",
        "work_float",
        "work_bool",
        "checkpoint_float",
        "checkpoint_updates",
    ],
)
def test_should_retain_paid_training_but_refuse_forged_a_only_exposure_before_native(
    learned, monkeypatch, tmp_path, damage
):
    _, results, _, _, warm, new = learned
    family = (
        "bp" if damage in ("work_warm", "work_new", "work_float", "work_bool") else "active-cpc"
    )
    stored = results[family]
    report = deepcopy(stored.report)
    candidate = stored.candidate.fork()
    if damage == "missing_mode":
        del report["settings"]["a_only_control"]
    elif damage == "false_mode":
        report["settings"]["a_only_control"] = False
    elif damage == "integer_mode":
        report["settings"]["a_only_control"] = 1
    elif damage == "training_contexts":
        report["settings"]["training_contexts"][1] = "new"
    elif damage == "diagnostic_contexts":
        report["settings"]["diagnostic_contexts"] = ["warm"]
    elif damage.startswith("event_"):
        key, value = {
            "event_context": ("context", "new"),
            "event_observed": ("observed_at", 1),
            "event_arrival": ("arrived_at", 2),
        }[damage]
        report["events"][0][key] = value
    elif damage.startswith("receipt_"):
        key, value = {
            "receipt_episode": ("episode_id", new.episode_id),
            "receipt_sample": ("sample_id", "foreign-sample"),
            "receipt_observed": ("observed_at", 1),
            "receipt_update": ("update_number", True),
        }[damage]
        report["applied_updates"][0][key] = value
        report["events"][0]["applied"][0][key] = value
    elif damage == "candidate_rows":
        candidate.restore_state(replace(candidate.snapshot_state(), training_examples=7))
        report["checkpoints"][-1]["training_examples"] = 7
    elif damage == "checkpoint_rows":
        report["checkpoints"][-1]["training_examples"] = True
    elif damage == "checkpoint_float":
        report["checkpoints"][-1]["training_examples"] = 6.0
    elif damage == "checkpoint_updates":
        report["checkpoints"][-1]["updates"] = 3.0
    elif damage == "work_float":
        report["work"]["training_rows_by_context"] = dict(warm=6.0, new=0)
    elif damage == "work_bool":
        report["work"]["training_rows_by_context"] = dict(warm=6, new=False)
    elif damage == "float_rows":
        report["work"]["training_rows"] = 6.0
    elif damage == "bool_replay":
        report["progress"]["replay_examples_completed"] = False
    elif damage == "float_progress":
        report["progress"]["updates_completed"] = 3.0
    elif damage == "float_phase":
        report["settings"]["phase_updates"] = 1.0
    elif damage == "native_rows":
        report["native_updates"][1]["training_rows"] = 3
    else:
        key, value = {
            "work_warm": ("training_rows_by_context", dict(warm=5, new=0)),
            "work_new": ("training_rows_by_context", dict(warm=6, new=1)),
            "unique_rows": ("unique_training_rows", 3),
            "repeated_rows": ("repeated_training_rows", 5),
            "total_rows": ("training_rows", 7),
        }[damage]
        report["work"][key] = value
    returned = demo.ContinuousLearnabilityResult(report, candidate)
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: returned)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    output = tmp_path / "refused.json"
    observed = entry.run_learning_rollout(
        output, warm, new, learner_family=family, phase_updates=1, a_only_control=True
    )
    assert observed["complete"] is False and observed["rollout_attempted"] is False
    assert observed["training"] == report and observed["rollout"] is None
    assert output.exists() and not output.with_name(output.name + ".rollout.json").exists()


def test_should_refuse_an_unrequested_a_only_result_on_default_aba_path(
    learned, monkeypatch, tmp_path
):
    _, results, _, _, warm, new = learned
    stored = results["bp"]
    # Equal A/B lengths make aggregate row counts ambiguous; the mode declaration
    # must still prevent an A-only result from satisfying an A/B/A request.
    new = replace(previous_episode(19), episode_id=new.episode_id)
    returned = demo.ContinuousLearnabilityResult(deepcopy(stored.report), stored.candidate.fork())
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: returned)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    report = entry.run_learning_rollout(tmp_path / "unrequested.json", warm, new, phase_updates=1)
    assert report["complete"] is False and report["rollout_attempted"] is False
    assert report["training"] == returned.report
    assert "a_only_control" not in report


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("visual", [False, True])
def test_should_forward_cli_control_only_when_selected_and_preserve_readiness_reader_binding(
    monkeypatch, tmp_path, control, visual
):
    warm, new = episodes(visual=visual)
    reads, calls = [], []

    def reader(path, seed, **options):
        reads.append((seed, options))
        return (warm if seed == 17 else new), dict(retained="metadata")

    def run(output, given_warm, given_new, **options):
        calls.append((given_warm, given_new, options))
        return dict(complete=False, training=dict(paid=1))

    monkeypatch.setattr(entry, "_read_readiness_teacher", reader)
    monkeypatch.setattr(entry, "_read_teacher", forbidden)
    monkeypatch.setattr(entry, "run_learning_rollout", run)
    flags = (["--a-only-control"] if control else []) + (["--visual-observation"] if visual else [])
    code = entry.main(
        [
            "--output",
            str(tmp_path / "cli.json"),
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
            "--readiness-teachers",
            "--phase-updates",
            "1",
            *flags,
        ]
    )
    assert code == 1 and reads == [
        (17, dict(visual_observation=visual)),
        (19, dict(visual_observation=visual)),
    ]
    expected = dict(learner_family="bp", phase_updates=1, visual_observation=visual)
    if control:
        expected["a_only_control"] = True
    assert calls == [(warm, new, expected)]
