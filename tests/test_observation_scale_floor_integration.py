"""Selected frozen scales through existing streams, private updates and fake IO.

Four phase-one family fixtures, one initializer2944normals each, one A fit2rows
each, three stream updates6rows each; eight private updates16rows per family.
Each uses two fake controls/predictions. PC/CPC work is native, not a BP estimate.
No simulator, probe, final evaluation, promotion, floor selection or new framework.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_correction_demo as corrections
from src.adapters import continuous_learnability_demo as demo
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.adapters.numpy_continuous_pc import NumpyContinuousPCLearner
from src.core.observation_standardization import ScaleFlooredObservationStandardization
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.infra import maniskill_continuous as native
from test_continuous_robot_learning import assert_state
from test_state_quaternion_corrections import batch
from test_state_quaternion_robot_learning import QuaternionEnvironment, episodes


FLOORS = (2.0,) * 42  # Synthetic contract oracle, not a proposed physical study recipe.
FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def forbidden(*args, **kwargs):
    pytest.fail("refused request reached fitting/model/native work")


@pytest.fixture(scope="module", params=FAMILIES)
def learned(request, tmp_path_factory):
    family = request.param
    directory = tmp_path_factory.mktemp(f"scale-floor-{family}")
    environment = QuaternionEnvironment()
    captured: dict[str, Any] = dict(fits=[], normalized=[])
    fit, run = demo.fit_training_standardization, entry.run_continuous_learnability

    def fit_once(value, **options):
        captured["fits"].append((asdict(value), deepcopy(options)))
        return fit(value, **options)

    def train(*args, **options):
        result = run(*args, **options)
        captured["result"] = result
        assert result.candidate is not None
        captured["before_native"] = result.candidate.snapshot_state()
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "fit_training_standardization", fit_once)
        patch.setattr(entry, "run_continuous_learnability", train)
        patch.setattr(native, "_make_environment", lambda **options: environment)
        patch.setattr(smoke, "version", lambda _: "fake-scale-floor")
        report = entry.run_learning_rollout(
            directory / "case.json",
            *episodes(),
            learner_family=family,
            phase_updates=1,
            a_only_control=True,
            canonical_state_quaternions=True,
            scale_floors=FLOORS,
        )
    return family, report, captured, environment


def test_should_use_one_train_fit_and_same_selected_transform_through_frozen_return(learned):
    family, report, captured, environment = learned
    assert report["complete"] is True, report.get("error")
    owner = captured["result"].candidate
    policy = owner.policy.base if family != "bp" else owner.policy
    fitted = policy.observation_standardization
    assert type(fitted) is ScaleFlooredObservationStandardization
    assert fitted.scale_floors == fitted.effective_scales == FLOORS
    assert fitted.scales[0] == 0.5 and fitted.scales[22] == 1.0
    assert fitted.means[22:26] == (0.5, -0.5, 0.5, -0.5)
    assert len(captured["fits"]) == 1
    assert captured["fits"][0][1] == dict(scale_floors=FLOORS)
    assert captured["fits"][0][0]["role"] == "train"
    assert report["training"]["work"]["statistics_numeric_bytes_per_owner"] == 1344
    assert report["scale_floors"] == FLOORS
    assert "scale_floor" in report["protocol_id"]
    returned = report["rollout"]
    returned_policy = returned["learner_policy"]
    if family != "bp":
        returned_policy = returned_policy["base"]
    assert returned_policy["observation_standardization"] == asdict(fitted)
    assert returned["optimizer_updates"] == 0 and returned["prediction_returns"] == 2
    assert len(environment.commands) == 2 and environment.close_calls == 1
    assert returned["frozen_state_exact"] is report["candidate_unchanged"] is True
    assert_state(captured["before_native"], owner.snapshot_state())
    assert report["training"]["settings"]["final_reads"] == 0
    assert report["training"]["settings"]["probe_reads"] == 0


def test_should_preserve_selected_statistics_and_source_through_private_corrections(
    learned,
    monkeypatch,
):
    family, _, captured, _ = learned
    source = captured["result"].candidate.fork()
    before = source.snapshot_state()
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    seen = []
    method_owner = NumpyContinuousBCLearner if family == "bp" else NumpyContinuousPCLearner
    original = (
        NumpyContinuousBCLearner.train_batch
        if family == "bp"
        else NumpyContinuousPCLearner.train_batch_with_work
    )

    def train(owner, features, targets):
        policy = owner.policy if family == "bp" else owner.policy.base
        seen.append(policy.observation_standardization)
        return original(owner, features, targets)

    monkeypatch.setattr(
        method_owner, "train_batch" if family == "bp" else "train_batch_with_work", train
    )
    result = corrections.run_continuous_correction_update(
        source,
        batch(),
        now=3,
        observation_projection=state_quaternion_projection(),
    )
    assert result.report["complete"] is True, result.report.get("error")
    assert result.candidate is not None and len(seen) == 8
    policy = before.policy if family == "bp" else before.policy.base
    assert all(value == policy.observation_standardization for value in seen)
    assert_state(before, source.snapshot_state())
    after = result.candidate.snapshot_state()
    assert after.policy == before.policy and after.completed_updates == 11
    assert after.training_examples == 22
    assert result.report["work"]["statistics_numeric_bytes_per_owner"] == 1344
    assert result.report["work"]["fit_calls"] == 0
    assert len(result.report["applied_updates"]) == 8
    assert result.report["progress"]["next_tick"] == 4


@pytest.mark.parametrize(
    "floors,visual",
    [
        ((), False),
        ((1.0,) * 41, False),
        ((float("nan"),) * 42, False),
        ((True,) * 42, False),
        (FLOORS, True),
    ],
)
def test_should_refuse_bad_native_floor_request_before_source_io(
    monkeypatch, tmp_path, floors, visual
):
    monkeypatch.setattr(entry, "_admit_inputs", forbidden)
    monkeypatch.setattr(entry, "run_continuous_learnability", forbidden)
    with pytest.raises(ValueError):
        entry.run_learning_rollout(
            tmp_path / "absent.json", *episodes(), scale_floors=floors, visual_observation=visual
        )
    assert not (tmp_path / "absent.json").exists()


def test_should_refuse_ignored_floor_request_before_native_with_paid_training_report(
    learned,
    monkeypatch,
    tmp_path,
):
    _, _, captured, _ = learned
    result = captured["result"]
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: result)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    observed = entry.run_learning_rollout(
        tmp_path / "changed-floor.json",
        *episodes(),
        learner_family=learned[0],
        phase_updates=1,
        a_only_control=True,
        canonical_state_quaternions=True,
        scale_floors=(3.0,) * 42,
    )
    assert observed["complete"] is False and observed["rollout_attempted"] is False
    assert "requested scale floors" in observed["error"]
    assert observed["training"] == result.report


def test_should_refuse_unrequested_selected_floor_before_native(learned, monkeypatch, tmp_path):
    family, _, captured, _ = learned
    result = captured["result"]
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: result)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    observed = entry.run_learning_rollout(
        tmp_path / "unrequested.json",
        *episodes(),
        learner_family=family,
        phase_updates=1,
        a_only_control=True,
        canonical_state_quaternions=True,
    )
    assert observed["complete"] is False and observed["rollout_attempted"] is False
    assert "unrequested scale floors" in observed["error"]
    assert observed["training"] == result.report


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_cli_floors_only_when_selected(monkeypatch, tmp_path, selected):
    warm, new = episodes()
    calls = []
    monkeypatch.setattr(
        entry, "_read_teacher", lambda path, seed, **options: warm if seed == 17 else new
    )

    def run(*args, **options):
        calls.append(options)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_learning_rollout", run)
    args = [
        "--output",
        str(tmp_path / "case.json"),
        "--warm-teacher",
        "warm.json",
        "--new-teacher",
        "new.json",
    ]
    if selected:
        args += ["--observation-scale-floors", *["2"] * 42]
    assert entry.main(args) == 0
    assert calls[0].get("scale_floors") == (FLOORS if selected else None)
    assert ("scale_floors" in calls[0]) is selected


@pytest.mark.parametrize("values", [["0"] * 42, ["nan"] * 42, ["1"] * 41])
def test_should_refuse_cli_floors_before_reading_teachers(monkeypatch, tmp_path, values):
    monkeypatch.setattr(entry, "_read_teacher", forbidden)
    with pytest.raises(SystemExit) as refusal:
        entry.main(
            [
                "--output",
                str(tmp_path / "case.json"),
                "--warm-teacher",
                "warm.json",
                "--new-teacher",
                "new.json",
                "--observation-scale-floors",
                *values,
            ]
        )
    assert refusal.value.code == 2


def test_should_keep_default_runner_dispatch_without_new_keyword(monkeypatch):
    calls = []

    def run(*args, **options):
        calls.append(options)
        return demo.ContinuousLearnabilityResult({}, None)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    demo.run_continuous_learnability_demo(*episodes())
    assert calls == [dict(expected_standardization=None)]
    demo.run_continuous_learnability_demo(*episodes(), scale_floors=FLOORS)
    assert calls[-1] == dict(expected_standardization=None, scale_floors=FLOORS)


def saved_floor_inputs():
    from test_state_control_diagnosis import inputs

    teacher, learned = inputs()
    metadata = learned["training"]["policy"]["observation_standardization"]
    metadata.update(scale_floors=[5.0] * 42, effective_scales=[5.0] * 42)
    learned["rollout"]["learner_policy"] = deepcopy(learned["training"]["policy"])
    return teacher, learned


def test_should_reconstruct_saved_selected_divisors_without_silently_using_empirical_scales():
    from scripts import diagnose_learned_robot_control as diagnostic

    teacher, learned = saved_floor_inputs()
    before = deepcopy((teacher, learned))
    report = diagnostic.diagnose_control(teacher, learned, state_observation=True)
    assert report["nearest_state42_normalized_rms"] == pytest.approx(
        [0.0, 0.2 / 42**0.5, 0.0, 0.4 / 42**0.5]
    )
    assert report["work"]["optimizer_updates"] == report["work"]["fits"] == 0
    assert (teacher, learned) == before


@pytest.mark.parametrize("fault", ["missing_floor", "changed_effective", "bool_effective"])
def test_should_refuse_corrupt_saved_selected_metadata_before_normalization(monkeypatch, fault):
    from scripts import diagnose_learned_robot_control as diagnostic

    teacher, learned = saved_floor_inputs()
    metadata = learned["training"]["policy"]["observation_standardization"]
    if fault == "missing_floor":
        del metadata["scale_floors"]
    else:
        metadata["effective_scales"][0] = True if fault == "bool_effective" else 6.0
    learned["rollout"]["learner_policy"] = deepcopy(learned["training"]["policy"])
    monkeypatch.setattr(diagnostic, "standardize_observations", forbidden)
    with pytest.raises(ValueError):
        diagnostic.diagnose_control(teacher, learned, state_observation=True)
