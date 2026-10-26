"""Actual literal full stream to four typed development requests, default off."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_catalog_decision as binding_adapter
import src.adapters.clm_head_learner as native
import src.app.clm_temporal_sequence as stream_app
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder, text_recipe
from src.core.clm_catalog import CATALOG_OBJECTIVE
from src.core.clm_head import CLMHeadPolicy, FrozenEncoderIdentity
from src.core.temporal_decision_sequence import temporal_examples
from src.core.typed_decision import ProbabilityDecisionResult
from src.infra.clm_local_loader import LoadedCLM
from src.shared.process_memory import ProcessRssSegment
from test_clm_catalog_learning import parts
from test_clm_frozen_encoder import FakeTokenizer, backbone


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def json_record(value):
    return json.loads(json.dumps(asdict(value)))


class LiteralSampler:
    def __init__(self, *, interval_seconds):
        self.interval_seconds = interval_seconds

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 1024 * 1024

    def snapshot(self):
        return ProcessRssSegment(1, self.sample(), self.sample(), 1, self.interval_seconds)


def execute(torch, *, typed=True, mode="none", fail_head=False, drift=False):
    states, actions = study.feature_texts(include_catalog_support=mode != "none")
    mapping = {text: [index + 1] for index, text in enumerate(states + actions)}
    model, tokenizer = backbone(torch), FakeTokenizer(mapping)
    state, action, scale = parts(torch)
    identity = FrozenEncoderIdentity("fake", "literal", "literal", "last", text_recipe(2048), 2)
    loaded = LoadedCLM(model, tokenizer, state, action, scale, identity, {})
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict()), scale.detach().clone()
    caller_rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    handles: dict[str, Any] = {"encoders": []}
    observations = []
    real_stream, real_encoder = stream_app.run_temporal_stream, FrozenCLMEncoder.__init__
    real_bind = binding_adapter.CLMCatalogPredictorBinding.bind

    def loader(*args, **kwargs):
        return loaded

    def observed_encoder(self, *args, **kwargs):
        real_encoder(self, *args, **kwargs)
        handles["encoders"].append(self)

    def observed_stream(learner, *args, **kwargs):
        handles["learner"] = learner
        return real_stream(learner, *args, **kwargs)

    def fail(module, args, output):
        raise RuntimeError("typed action head return lost")

    def bound(self):
        predictor = real_bind(self)
        predictor._action_head.register_forward_hook(fail)
        return predictor

    def save():
        learner = handles.get("learner")
        record = report.get("typed_development", {})
        observations.append(
            (
                None if learner is None else learner.completed_updates,
                len(record.get("predictions", ())),
                "checkpoints" in report,
            )
        )
        if drift and len(record.get("predictions", ())) == 1 and not handles.get("drifted"):
            assert learner is not None
            handles["drifted"] = True
            learner._bundle[0].training = not learner._bundle[0].training

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "load_local_clm", loader)
        patch.setattr(study, "ProcessRssSampler", LiteralSampler)
        patch.setattr(stream_app, "run_temporal_stream", observed_stream)
        patch.setattr(FrozenCLMEncoder, "__init__", observed_encoder)
        if fail_head:
            patch.setattr(binding_adapter.CLMCatalogPredictorBinding, "bind", bound)
        kwargs = dict(catalog_training=True, catalog_support=mode, save=save)
        if typed:
            kwargs["typed_development_readout"] = True
        if fail_head:
            with pytest.raises(RuntimeError, match="typed action head return lost"):
                study.run_case(Path("fixture-model"), Path("fixture-head"), report, **kwargs)
        elif drift:
            with pytest.raises(ValueError, match="live source changed"):
                study.run_case(Path("fixture-model"), Path("fixture-head"), report, **kwargs)
        else:
            study.run_case(Path("fixture-model"), Path("fixture-head"), report, **kwargs)
    return dict(
        report=report,
        handles=handles,
        model=model,
        tokenizer=tokenizer,
        loaded=loaded,
        original=original,
        rng=caller_rng,
        observations=observations,
    )


@pytest.fixture(scope="module")
def completed(torch):
    return execute(torch)


def test_should_run_twelve_catalog_updates_then_four_actual_typed_requests(completed):
    report = completed["report"]
    assert report["complete"] is True and report["sequence_complete"] is True
    assert report["training_objective"] == CATALOG_OBJECTIVE
    assert report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 36
    paid = report["learner_work"]
    assert paid["train_attempts"] == paid["commits"] == paid["backward_returns"] == 12
    assert (
        paid["catalog_candidate_rows_attempted"] == 36
        and paid["catalog_logit_values_attempted"] == 108
    )
    assert paid["snapshot_attempts"] == paid["snapshot_returns"] == 1
    assert paid["choice_prediction_returns"] == 5
    assert all(
        update == 12
        for update, predictions, scored in completed["observations"]
        if predictions or scored
    )
    record = report["typed_development"]
    examples = temporal_examples("after_return", "outer_selection")
    assert record["complete"] is True
    assert record["requests"] == [json_record(example.request) for example in examples]
    assert len(record["predictions"]) == 4
    assert [row["request_id"] for row in record["predictions"]] == [
        example.request.request_id for example in examples
    ]
    for raw, example in zip(record["predictions"], examples, strict=True):
        assert [pair[0] for pair in raw["probabilities"]] == [
            candidate.candidate_id for candidate in example.request.allowed_candidates
        ]


def test_should_observe_batch_to_request_probability_parity_and_complete_binding_costs(completed):
    report = completed["report"]
    record = report["typed_development"]
    cached = report["raw_readouts"][-1]["probabilities"]
    differences: list[float] = []
    for raw, expected in zip(record["predictions"], cached, strict=True):
        observed = [pair[1] for pair in raw["probabilities"]]
        assert observed == pytest.approx(expected, abs=2e-7)
        differences.extend(abs(a - b) for a, b in zip(observed, expected, strict=True))
    assert record["max_abs_cached_difference"] == max(differences)
    work = record["binding_work"]
    assert work["binding_attempts"] == work["binding_returns"] == 1
    assert work["snapshot_attempts"] == work["snapshot_returns"] == 1
    assert work["guard_capture_attempts"] == work["guard_capture_returns"] == 3
    assert work["head_copy_attempts"] == work["head_copy_returns"] == 2
    assert (
        work["head_parameter_elements_attempted"] == work["head_parameter_elements_returned"] == 8
    )
    assert work["scale_copy_elements_attempted"] == work["scale_copy_elements_returned"] == 2
    guard = record["source_guard_work"]
    assert guard["capture_attempts"] == guard["capture_returns"] == 6
    assert 2 * guard["tensor_elements_copied"] == 3 * work["snapshot_tensor_elements_copied"]
    receipt = record["binding_receipt"]
    assert receipt["work"] == work and receipt["objective"] == CATALOG_OBJECTIVE
    assert receipt["source_completed_updates"] == receipt["source_generation"] == 12
    assert receipt["source_training_examples"] == 36
    assert receipt["identity"] == report["identity"]
    assert (
        receipt["policy"] == record["source_policy"] == vars(completed["handles"]["learner"].policy)
    )
    assert receipt["instructions"] == study.INSTRUCTIONS
    actor = record["predictor_work"]
    assert actor["request_attempts"] == actor["request_returns"] == 4
    assert actor["encoder_returns"] == 4 and actor["head_forward_returns"] == 8
    assert actor["projected_rows"] == 16 and actor["unreturned_work_unknown"] is False


def test_should_keep_one_encoder_born_with_five_forwards_and_preserve_loaded_controls(
    torch, completed
):
    case, report = completed, completed["report"]
    encoders = case["handles"]["encoders"]
    assert len(encoders) == 1 and encoders[0]._limits[2] == 5
    assert [len(call["ids"]) for call in case["model"].calls] == [12, 4, 4, 4, 4]
    assert (
        report["encoder_work"]["forward_attempts"] == report["encoder_work"]["forward_returns"] == 5
    )
    before = report["typed_development"]["encoder_work_before"]
    after = report["typed_development"]["encoder_work_after"]
    assert before["forward_returns"] == 1 and after["forward_returns"] == 5
    assert after["sequences_returned"] - before["sequences_returned"] == 16
    assert after == report["encoder_work"]
    paid = encoders[0].work
    with pytest.raises(ValueError, match="allowance exhausted"):
        encoders[0].embed_texts(("invoice",), ("tail",))
    assert encoders[0].work == paid
    state, action, scale = case["original"]
    assert native._equal(case["loaded"].state_head.state_dict(), state)
    assert native._equal(case["loaded"].action_head.state_dict(), action)
    assert torch.equal(case["loaded"].logit_scale, scale)
    assert torch.equal(torch.get_rng_state(), case["rng"])
    assert report["control_unchanged"] is report["caller_rng_unchanged"] is True
    serialized = json.loads(json.dumps(report, allow_nan=False))
    assert serialized["typed_development"] == report["typed_development"]


@pytest.fixture(scope="module")
def default_completed(torch):
    return execute(torch, typed=False)


def test_should_keep_default_off_output_snapshot_and_encoder_budget_unchanged(default_completed):
    report = default_completed["report"]
    assert report["complete"] is True and "typed_development" not in report
    assert (
        report["learner_work"]["snapshot_attempts"]
        == report["learner_work"]["snapshot_returns"]
        == 0
    )
    assert report["encoder_work"]["forward_returns"] == 1
    assert len(default_completed["handles"]["encoders"]) == 1
    assert default_completed["handles"]["encoders"][0]._limits[2] == 1
    assert report["cached_typed_decodes"] == 88


@pytest.mark.parametrize(
    "fault", ("partial", "identity", "objective", "work", "receipt", "existing")
)
def test_should_refuse_invalid_live_source_report_before_binding(
    default_completed, monkeypatch, fault
):
    case = default_completed
    report = dict(case["report"])
    if fault == "partial":
        report["sequence_complete"] = False
    elif fault == "identity":
        report["identity"] = {**report["identity"], "revision": "foreign"}
    elif fault == "objective":
        report["training_objective"] = native.OBJECTIVE
    elif fault == "work":
        report["learner_work"] = {**report["learner_work"], "snapshot_attempts": 1}
    elif fault == "receipt":
        report["stream"] = deepcopy(report["stream"])
        report["stream"]["receipts"][-1]["diagnostic"]["definition"] = native.OBJECTIVE
    else:
        report["typed_development"] = {"complete": False}

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted live source report constructed a binding")

    monkeypatch.setattr(binding_adapter, "CLMCatalogPredictorBinding", forbidden)
    source, frozen = case["handles"]["learner"], case["handles"]["encoders"][0]
    before, encoded = source.work, frozen.work
    with pytest.raises(ValueError):
        study.read_typed_return(source, frozen, report, before_step=lambda: None, save=lambda: None)
    assert source.work == before and frozen.work == encoded


@pytest.mark.parametrize(
    "fault",
    (
        "partial",
        "missing",
        "snapshot",
        "request",
        "candidate_order",
        "prediction",
        "binding_work",
        "binding_receipt",
        "actor_work",
        "encoder_prefix",
        "encoder_suffix",
        "difference",
        "guard_count",
        "guard_copy",
        "guard_missing",
        "policy_updates",
        "policy_rows",
        "policy_extra",
        "source_policy",
        "encoder_before_missing",
        "encoder_after_missing",
        "encoder_before_extra",
        "encoder_after_extra",
        "encoder_unsafe_before",
        "encoder_unsafe_after",
    ),
)
def test_should_admit_bound_typed_record_before_accepting_extra_snapshot_or_scoring(
    completed, monkeypatch, fault
):
    report = dict(completed["report"])
    report["learner_work"] = dict(report["learner_work"])
    report["typed_development"] = deepcopy(report["typed_development"])
    record = report["typed_development"]
    if fault == "partial":
        record["complete"] = False
    elif fault == "missing":
        report.pop("typed_development")
    elif fault == "snapshot":
        report["learner_work"]["snapshot_returns"] = 2
    elif fault == "request":
        record["requests"][0]["query"] = "selected alternative"
    elif fault == "candidate_order":
        record["predictions"][-1]["probabilities"].reverse()
    elif fault == "prediction":
        record["predictions"][-1]["probabilities"][0][1] += 0.1
    elif fault == "binding_work":
        record["binding_work"]["snapshot_returns"] = 0
    elif fault == "binding_receipt":
        record["binding_receipt"]["source_training_examples"] = 44
    elif fault == "actor_work":
        record["predictor_work"]["request_returns"] = 3
    elif fault == "encoder_prefix":
        record["encoder_work_before"]["forward_returns"] = 2
    elif fault == "encoder_suffix":
        record["encoder_work_after"]["forward_returns"] = 4
    elif fault == "difference":
        record["max_abs_cached_difference"] += 0.01
    elif fault == "guard_count":
        record["source_guard_work"]["capture_returns"] = 5
    elif fault == "guard_copy":
        record["source_guard_work"]["tensor_elements_copied"] += 1
    elif fault == "guard_missing":
        record.pop("source_guard_work")
    elif fault == "policy_updates":
        record["binding_receipt"]["policy"]["max_updates"] = 13
    elif fault == "policy_rows":
        record["binding_receipt"]["policy"]["max_rows"] = 3
    elif fault == "policy_extra":
        record["binding_receipt"]["policy"]["undeclared"] = True
    elif fault == "source_policy":
        record["source_policy"]["learning_rate"] = 0.002
    elif fault == "encoder_before_missing":
        record["encoder_work_before"].pop("tokenization_attempts")
    elif fault == "encoder_after_missing":
        record["encoder_work_after"].pop("tokenization_attempts")
    elif fault == "encoder_before_extra":
        record["encoder_work_before"]["undeclared"] = 0
    elif fault == "encoder_after_extra":
        record["encoder_work_after"]["undeclared"] = 0
    elif fault == "encoder_unsafe_before":
        record["encoder_work_before"]["token_ids_visited"] = 2**53 + 1
        record["encoder_work_after"]["token_ids_visited"] = 2**53 + 17
    else:
        record["encoder_work_after"]["token_ids_visited"] = 2**53 + 1

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound extra snapshot exposed cached scores")

    monkeypatch.setattr(study, "_score_cached", forbidden)
    with pytest.raises(ValueError):
        study.score_sequence(report)


@pytest.fixture(scope="module")
def paid_failure(torch):
    return execute(torch, mode="route", fail_head=True)


def test_should_preserve_paid_completed_training_and_partial_typed_head_failure(paid_failure):
    report = paid_failure["report"]
    assert report["complete"] is False and report["sequence_complete"] is True
    assert report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 44
    assert report["learner_work"]["catalog_candidate_rows_attempted"] == 36
    assert report["learner_work"]["catalog_logit_values_attempted"] == 132
    assert (
        report["learner_work"]["snapshot_attempts"]
        == report["learner_work"]["snapshot_returns"]
        == 1
    )
    assert "checkpoints" not in report
    record = report["typed_development"]
    assert record["complete"] is False and record["predictions"] == []
    assert record["binding_work"]["binding_returns"] == 1
    assert (
        record["source_guard_work"]["capture_attempts"]
        == record["source_guard_work"]["capture_returns"]
        == 2
    )
    assert record["binding_receipt"]["source_training_examples"] == 44
    assert record["predictor_work"]["head_forward_attempts"] == 2
    assert record["predictor_work"]["head_forward_returns"] == 1
    assert record["predictor_work"]["request_returns"] == 0
    assert record["predictor_work"]["unreturned_work_unknown"] is True
    assert record["encoder_work_before"]["forward_returns"] == 1
    assert (
        record["encoder_work_after"]["forward_returns"]
        == report["encoder_work"]["forward_returns"]
        == 2
    )
    assert [len(call["ids"]) for call in paid_failure["model"].calls] == [13, 4]
    source, frozen = paid_failure["handles"]["learner"], paid_failure["handles"]["encoders"][0]
    before, encoded = source.work, frozen.work
    with pytest.raises(ValueError):
        study.read_typed_return(source, frozen, report, before_step=lambda: None, save=lambda: None)
    with pytest.raises(ValueError):
        study.score_sequence(report)
    assert source.work == before and frozen.work == encoded


@pytest.mark.parametrize(
    "kwargs",
    (
        {"typed_development_readout": True},
        {
            "typed_development_readout": True,
            "catalog_training": True,
            "compare_catalog_support": True,
        },
        {"typed_development_readout": True, "catalog_training": True, "reserved_task_probes": True},
        {"typed_development_readout": 1, "catalog_training": True},
    ),
)
def test_should_refuse_invalid_typed_run_api_before_loading(monkeypatch, kwargs):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported typed mode reached model loader")

    monkeypatch.setattr(study, "load_local_clm", forbidden)
    with pytest.raises(ValueError):
        study.run_case(Path("fixture"), Path("fixture-head"), {}, **kwargs)


def test_should_refuse_typed_comparison_cache_before_admitting_any_arm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported comparison reached catalog work admission")

    monkeypatch.setattr(study, "_require_catalog_completed", forbidden)
    with pytest.raises(ValueError, match="typed development.*recipe comparison"):
        study._require_comparison_arm("repeat", {"typed_development": {"complete": True}})


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], {}),
        (
            ["--catalog-training", "--typed-development-readout"],
            {"catalog_training": True, "typed_development_readout": True},
        ),
        (
            ["--catalog-training", "--catalog-support", "route", "--typed-development-readout"],
            {
                "catalog_training": True,
                "catalog_support": "route",
                "typed_development_readout": True,
            },
        ),
    ],
)
def test_should_route_only_explicit_typed_cli_flags(monkeypatch, tmp_path, arguments, expected):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-temporal",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            *arguments,
        ],
    )
    calls = []

    def case(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True, fixture="metadata only")
        save()

    monkeypatch.setattr(study, "run_case", case)
    study.main()
    assert calls == [expected]


@pytest.mark.parametrize(
    "arguments",
    (
        ["--typed-development-readout"],
        ["--typed-development-readout", "--catalog-training", "--compare-catalog-support"],
        ["--typed-development-readout", "--catalog-training", "--reserved-task-probes"],
    ),
)
def test_should_refuse_invalid_typed_cli_before_runner(monkeypatch, tmp_path, arguments):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-temporal",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            *arguments,
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported typed CLI reached runner")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2


@pytest.fixture(scope="module")
def callback_drift(torch):
    return execute(torch, drift=True)


def test_should_refuse_callback_source_drift_before_the_second_typed_request(callback_drift):
    case, report = callback_drift, callback_drift["report"]
    assert report["sequence_complete"] is True and report["complete"] is False
    assert case["handles"]["drifted"] is True
    assert report["learner_work"]["commits"] == 12
    assert (
        report["learner_work"]["snapshot_attempts"]
        == report["learner_work"]["snapshot_returns"]
        == 1
    )
    record = report["typed_development"]
    assert record["complete"] is False and len(record["predictions"]) == 1
    assert (
        record["predictor_work"]["request_attempts"]
        == record["predictor_work"]["request_returns"]
        == 1
    )
    assert record["predictor_work"]["encoder_returns"] == 1
    assert record["predictor_work"]["unreturned_work_unknown"] is False
    assert (
        record["source_guard_work"]["capture_attempts"]
        == record["source_guard_work"]["capture_returns"]
        == 3
    )
    assert record["encoder_work_after"]["forward_returns"] == 2
    assert [len(call["ids"]) for call in case["model"].calls] == [12, 4]
    source, frozen = case["handles"]["learner"], case["handles"]["encoders"][0]
    paid, encoded = source.work, frozen.work
    with pytest.raises(ValueError):
        study.read_typed_return(source, frozen, report, before_step=lambda: None, save=lambda: None)
    with pytest.raises(ValueError):
        study.score_sequence(report)
    assert source.work == paid and frozen.work == encoded


@pytest.mark.parametrize("failure", ("source_mutation", "save_exception"))
def test_should_keep_four_returned_choices_partial_when_final_publication_fails(
    monkeypatch, failure
):
    """Control-flow doubles only; native binding/provenance is tested by full fixtures."""

    class Source:
        identity = FrozenEncoderIdentity(
            "control-flow", "literal", "literal", "last", "no-native", 2
        )
        policy = CLMHeadPolicy(projection_dim=2, max_updates=12, max_rows=4)
        completed_updates = generation = 12
        training_examples = 36

        def __init__(self):
            self._bundle = object()
            self._work = {"snapshot_attempts": 0, "snapshot_returns": 0}
            self.value = 1

        @property
        def work(self):
            return dict(self._work)

        def _capture(self):
            return SimpleNamespace(value=self.value)

        def _check(self):
            pass

    removed = []

    class Encoder:
        def __init__(self):
            self._work = {"forward_returns": 1}
            self._model = SimpleNamespace(
                register_forward_pre_hook=lambda *args, **kwargs: SimpleNamespace(
                    remove=lambda: removed.append(True)
                )
            )

        @property
        def work(self):
            return dict(self._work)

    class Predictor:
        def __init__(self, frozen):
            self.frozen = frozen
            self.calls = 0

        @property
        def work(self):
            return {"request_attempts": self.calls, "request_returns": self.calls}

        def __call__(self, request):
            self.calls += 1
            self.frozen._work["forward_returns"] += 1
            probabilities = tuple(
                (candidate.candidate_id, 1 / 3) for candidate in request.allowed_candidates
            )
            return ProbabilityDecisionResult(
                request.request_id, probabilities[0][0], probabilities, reported_confidence=0.0
            )

    class Binding:
        def __init__(self, source, frozen, **kwargs):
            self.source, self.frozen = source, frozen
            self.work = {"binding_attempts": 1, "binding_returns": 1}
            self.receipt = {"fixture_only": "no native source admission"}

        def bind(self):
            self.source._work["snapshot_attempts"] += 1
            self.source._work["snapshot_returns"] += 1
            return Predictor(self.frozen)

    source, frozen = Source(), Encoder()
    report = dict(
        sequence_complete=True,
        identity=asdict(source.identity),
        learner_work=source.work,
        stream={"training_pairs": 36},
        raw_readouts=[
            dict(
                phase="after_return",
                completed_updates=12,
                probabilities=[[1 / 3] * 3 for _ in range(4)],
            )
        ],
    )
    saves = []

    def save():
        record = report["typed_development"]
        saves.append(len(record["predictions"]))
        assert record["complete"] is False
        if "binding_work" in record and len(record["predictions"]) == 4:
            if failure == "source_mutation":
                source.value = 2
            else:
                raise RuntimeError("final save did not return")

    monkeypatch.setattr(study, "_require_catalog_completed", lambda report: None)
    monkeypatch.setattr(binding_adapter, "CLMCatalogPredictorBinding", Binding)
    expected = ValueError if failure == "source_mutation" else RuntimeError
    message = "live source changed" if failure == "source_mutation" else "final save did not return"
    with pytest.raises(expected, match=message):
        study.read_typed_return(source, frozen, report, before_step=lambda: None, save=save)
    record = report["typed_development"]
    assert record["complete"] is False and message in record["error"]
    assert len(record["predictions"]) == 4 and saves == [1, 2, 3, 4, 4]
    assert record["predictor_work"] == {"request_attempts": 4, "request_returns": 4}
    assert record["binding_work"]["binding_returns"] == 1
    assert report["learner_work"] == {"snapshot_attempts": 1, "snapshot_returns": 1}
    assert record["encoder_work_after"] == {"forward_returns": 5}
    assert record["source_guard_work"]["capture_returns"] == (
        6 if failure == "source_mutation" else 5
    )
    assert removed == [True]
