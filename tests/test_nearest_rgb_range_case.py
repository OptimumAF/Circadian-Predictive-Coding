"""Metadata/stub checks for one frozen nearest RGB-range control.

No real arrays, policies, models, fits, normal draws, predictions, gradients,
prototype appends, simulator calls, or accepted artifact reads in this module.
Main-path dependencies are explicit local stubs; only immutable metadata and
bounded temporary JSON outputs are real. The executable runner owns all checks.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.observation_standardization import ObservationStandardization

case = importlib.import_module("artifacts.runs.nearest-rgb-range-control-20261020.native_case")


@dataclass(frozen=True)
class Spec:
    observation_dim: int = 789
    action_units: tuple[str, ...] = (
        "normalized_root_dx",
        "normalized_root_dy",
        "normalized_root_dz",
        "normalized_gripper",
    )
    lower_bounds: tuple[float, ...] = (-1.0,) * 4
    upper_bounds: tuple[float, ...] = (1.0,) * 4


@dataclass(frozen=True)
class Policy:
    spec: Spec
    observation_standardization: ObservationStandardization
    max_prototypes: int = 256
    training_role: str = "train"


def statistics() -> dict[str, Any]:
    return {
        "means": [float(index) / 10 for index in range(21)] + [0.25] * 768,
        "scales": [1.0 + float(index) / 100 for index in range(21)] + [0.00001] * 768,
        "fitted_rows": 29,
    }


def metadata() -> dict[str, Any]:
    return {
        "spec": asdict(Spec()),
        "observation_standardization": statistics(),
        "max_prototypes": 256,
        "training_role": "train",
    }


def endpoint(seed: int, success: int, *, rows: int = 2, prototypes: int = 29) -> dict[str, Any]:
    return {
        "seed": seed,
        "complete": True,
        "steps": rows,
        "sparse_return": float(success),
        "any_success": bool(success),
        "final_success": bool(success),
        "nearest_work": {
            "nearest_prediction_returns": rows,
            "nearest_distance_candidates_completed": rows * prototypes,
            "nearest_distance_feature_values_completed": rows * prototypes * 789,
        },
    }


def pipeline() -> dict[str, Any]:
    return {
        "complete": True,
        "source_updates": 1,
        "source_training_rows": 29,
        "source_state_exact": True,
        "collection": endpoint(19, 0),
        "rollouts": [endpoint(17, 1, prototypes=31), endpoint(19, 0, prototypes=31)],
        "update": {
            "complete": True,
            "policy": metadata(),
            "protocol_id": "nearest_continuous_correction_1_v1",
            "progress": {"updates_completed": 1},
            "work": {
                "training_rows": 2,
                "optimizer_updates": 0,
                "source_prototypes": 29,
                "candidate_prototypes": 31,
            },
        },
        "guard_reads": 0,
        "probe_reads": 0,
        "final_reads": 0,
        "promotion_calls": 0,
    }


def control() -> dict[str, Any]:
    return {
        "complete": True,
        "source_unchanged": True,
        "pipeline": pipeline(),
        "producer": {"python": "3.12.3", "numpy": "1.26.4"},
        "metrics": {
            "b_after_minus_before": {
                key: 0 for key in ("sparse_return", "any_success", "final_success")
            },
            "a_reference_minus_after": {
                key: 0 for key in ("sparse_return", "any_success", "final_success")
            },
        },
        "work": {"controls": 6, "predictions": 6, "prototype_appends": 2},
    }


def test_should_keep_original_public_prefix_and_use_fixed_rgb_half_range() -> None:
    raw = statistics()
    before = deepcopy(raw)
    result = case.range_statistics(raw)
    assert type(result) is ObservationStandardization
    assert result.means == tuple(raw["means"][:21]) + (0.5,) * 768
    assert result.scales == tuple(raw["scales"][:21]) + (0.5,) * 768
    assert result.fitted_rows == 29 and raw == before
    raw["means"][0] = 99.0
    assert result.means[0] == 0.0


@pytest.mark.parametrize("damage", ["short", "fitted_rows", "bool_rows", "zero_prefix_scale"])
def test_should_refuse_invalid_original_statistics(damage: str) -> None:
    raw = statistics()
    if damage == "short":
        raw["means"].pop()
        raw["scales"].pop()
    elif damage == "fitted_rows":
        raw["fitted_rows"] = 28
    elif damage == "bool_rows":
        raw["fitted_rows"] = True
    else:
        raw["scales"][0] = 0.0
    with pytest.raises(ValueError):
        case.range_statistics(raw)


def test_should_accept_and_own_a_complete_negative_empirical_control() -> None:
    raw, policy = control(), metadata()
    before = deepcopy((raw, policy))
    result = case.validate_control(raw, policy)
    assert (raw, policy) == before
    assert result["views"]["b_before"]["sparse_return"] == 0.0
    assert result["views"]["a_after"]["sparse_return"] == 1.0
    assert result["views"]["b_after"]["sparse_return"] == 0.0
    raw["pipeline"]["collection"]["sparse_return"] = 99.0
    assert result["views"]["b_before"]["sparse_return"] == 0.0


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "source_rows",
        "policy",
        "outcome",
        "metric",
        "bool_source_counter",
        "incomplete_view",
        "incomplete_update",
        "nonfinite_work",
    ],
)
def test_should_refuse_an_unmatched_or_changed_empirical_control(damage: str) -> None:
    raw = control()
    if damage == "incomplete":
        raw["complete"] = False
    elif damage == "source_rows":
        raw["pipeline"]["source_training_rows"] = 30
    elif damage == "policy":
        raw["pipeline"]["update"]["policy"]["observation_standardization"]["scales"][22] = 0.5
    elif damage == "outcome":
        raw["pipeline"]["rollouts"][1]["final_success"] = True
    elif damage == "bool_source_counter":
        raw["pipeline"]["source_updates"] = True
    elif damage == "incomplete_view":
        raw["pipeline"]["collection"]["complete"] = False
    elif damage == "incomplete_update":
        raw["pipeline"]["update"]["complete"] = False
    elif damage == "nonfinite_work":
        raw["work"]["controls"] = float("inf")
    else:
        raw["metrics"]["b_after_minus_before"]["sparse_return"] = 1
    with pytest.raises(ValueError):
        case.validate_control(raw, metadata())


@pytest.mark.parametrize("negative", [False, True])
def test_should_preserve_all_six_signed_metrics_and_nine_raw_endpoint_differences(
    negative: bool,
) -> None:
    saved = case.validate_control(control(), metadata())
    values = {
        "b_after_minus_before": -1 if negative else 0,
        "a_reference_minus_after": 1 if negative else 0,
    }
    metrics = {group: {key: value for key in case.METRICS} for group, value in values.items()}
    views = {
        "b_before": endpoint(19, int(negative)),
        "a_after": endpoint(17, int(not negative)),
        "b_after": endpoint(19, 0),
    }
    before = deepcopy((metrics, views, saved))
    result = case.report_comparison(metrics, views, saved)
    assert result["metric_delta"] == {
        group: {key: value for key in case.METRICS} for group, value in values.items()
    }
    assert result["endpoint_delta"] == {
        "b_before": {key: int(negative) for key in case.METRICS},
        "a_after": {key: -int(negative) for key in case.METRICS},
        "b_after": {key: 0 for key in case.METRICS},
    }
    assert "not matched compute" in result["qualification"]
    assert (metrics, views, saved) == before


@pytest.fixture
def stub_main(monkeypatch, tmp_path):
    calls: dict[str, Any] = {"fits": [], "constructors": [], "appends": [], "pipelines": []}
    teacher = {"qualified": True, "role": "train", "seed": 17, "candidate_episode": {"stub": "A29"}}
    prior = {"complete": True, "role": "train", "learner_policy": metadata()}
    saved = control()
    episode = SimpleNamespace(role="train", spec=Spec(), transitions=(None,) * 29)

    def read(path):
        if path == case.TEACHER:
            return deepcopy(teacher)
        if path == case.POLICY:
            return deepcopy(prior)
        return deepcopy(saved)

    def fit(record, **options):
        calls["fits"].append((record, options))
        return case.range_statistics(statistics())

    class Model:
        def __init__(self, policy):
            calls["constructors"].append(policy)
            self.policy = policy
            self.rows = 0

        def train_episode(self, record):
            calls["appends"].append(record)
            self.rows = 29
            return SimpleNamespace(value=29)

        def snapshot_state(self):
            return SimpleNamespace(
                policy=self.policy, completed_updates=1, training_examples=self.rows
            )

    def run(owner, **options):
        calls["pipelines"].append((owner, options))
        result = pipeline()
        result["update"]["policy"] = asdict(owner.policy)
        return result

    monkeypatch.setattr(case, "STAGE", tmp_path)
    monkeypatch.setattr(
        case, "sys", SimpleNamespace(version_info=(3, 12, 3), version="3.12.3 stub")
    )
    monkeypatch.setattr(case, "np", SimpleNamespace(__version__="1.26.4"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(case, "read", read)
    monkeypatch.setattr(case, "_episode", lambda raw: episode)
    monkeypatch.setattr(case, "NearestContinuousBCPolicy", Policy)
    monkeypatch.setattr(case, "NumpyNearestContinuousBCLearner", Model)
    monkeypatch.setattr(case, "fit_training_standardization", fit)
    monkeypatch.setattr(case, "run_corrective_adaptation", run)
    monkeypatch.setattr(case, "_same_state", lambda first, second: True)
    monkeypatch.setattr(case, "measure_peak_rss", lambda: 1)
    return tmp_path, calls, saved, episode


def test_should_count_one_returned_fit_before_constructing_and_running_control(stub_main) -> None:
    directory, calls, _, episode = stub_main
    assert case.main() == 0
    report = json.loads((directory / "native-result.json").read_text(encoding="utf8"))
    assert report["complete"] is True
    assert (
        report["statistics_fits_attempted"]
        == report["statistics_fits_completed"]
        == report["fits"]
        == 1
    )
    assert (
        len(calls["fits"])
        == len(calls["constructors"])
        == len(calls["appends"])
        == len(calls["pipelines"])
        == 1
    )
    assert calls["fits"][0][0] is episode
    assert calls["fits"][0][1]["rgb_range_recipe"] is not None


def test_should_retain_returned_wrong_fit_but_refuse_model_initialization(
    stub_main, monkeypatch
) -> None:
    directory, calls, _, _ = stub_main
    actual = case.fit_training_standardization

    def wrong(record, **options):
        valid = actual(record, **options)
        return ObservationStandardization(valid.means, (1.0,) * 789, 29)

    monkeypatch.setattr(case, "fit_training_standardization", wrong)
    assert case.main() == 1
    report = json.loads((directory / "native-result.json").read_text(encoding="utf8"))
    assert report["complete"] is False
    assert (
        report["statistics_fits_attempted"]
        == report["statistics_fits_completed"]
        == report["fits"]
        == 1
    )
    assert calls["constructors"] == calls["appends"] == calls["pipelines"] == []


def test_should_retain_incomplete_paid_pipeline_without_comparison(stub_main, monkeypatch) -> None:
    directory, calls, _, _ = stub_main
    returned = pipeline()
    returned.update(complete=False, error="synthetic paid cleanup failure")

    def run(owner, **options):
        calls["pipelines"].append((owner, options))
        return returned

    monkeypatch.setattr(case, "run_corrective_adaptation", run)
    assert case.main() == 1
    report = json.loads((directory / "native-result.json").read_text(encoding="utf8"))
    assert report["pipeline"] == json.loads(json.dumps(returned))
    assert report["pipeline"]["update"]["progress"]["updates_completed"] == 1
    assert report["complete"] is False
    assert "comparison" not in report


def test_should_retain_paid_pipeline_with_wrong_returned_normalization_policy(
    stub_main, monkeypatch
) -> None:
    directory, calls, _, _ = stub_main
    returned = pipeline()  # Empirical statistics differ from this range policy.

    def run(owner, **options):
        calls["pipelines"].append((owner, options))
        assert asdict(owner.policy) != returned["update"]["policy"]
        return returned

    monkeypatch.setattr(case, "run_corrective_adaptation", run)
    assert case.main() == 1
    report = json.loads((directory / "native-result.json").read_text(encoding="utf8"))
    assert report["complete"] is False
    assert report["pipeline"] == json.loads(json.dumps(returned))
    assert report["statistics_fits_completed"] == report["fits"] == 1
    assert report["setup_append_returns"] == 1
    assert (
        len(calls["fits"])
        == len(calls["constructors"])
        == len(calls["appends"])
        == len(calls["pipelines"])
        == 1
    )
    assert "comparison" not in report


def test_should_refuse_changed_saved_control_before_any_fit(stub_main) -> None:
    directory, calls, saved, _ = stub_main
    saved["complete"] = False
    assert case.main() == 1
    report = json.loads((directory / "native-result.json").read_text(encoding="utf8"))
    assert report["complete"] is False
    assert report["statistics_fits_attempted"] == report["statistics_fits_completed"] == 0
    assert calls["fits"] == calls["constructors"] == calls["appends"] == calls["pipelines"] == []


@pytest.mark.parametrize("existing", ["native-case", "native-result.json"])
def test_should_refuse_preexisting_one_use_output_before_fit(stub_main, existing: str) -> None:
    directory, calls, _, _ = stub_main
    path = directory / existing
    if existing == "native-case":
        path.mkdir()
    else:
        path.write_text("retained", encoding="utf8")
    with pytest.raises(ValueError, match="already exist"):
        case.main()
    assert calls["fits"] == calls["constructors"] == calls["appends"] == calls["pipelines"] == []
    if existing == "native-result.json":
        assert path.read_text(encoding="utf8") == "retained"
