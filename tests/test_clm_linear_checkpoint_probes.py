"""Cold probe control through the existing stream; no pretrained evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn
import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_linear_checkpoint_probes as linear
from src.adapters.clm_catalog_learner import CLMCatalogLearner
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder, text_recipe
from src.adapters.clm_head_learner import _equal
from src.adapters.clm_linear_fit_control import fit_clm_linear_control, predict_clm_linear_control
from src.app.clm_temporal_sequence import run_temporal_stream
from src.core.clm_head import CLMHeadPolicy, FrozenEncoderIdentity
from src.core.clm_probe import FrozenProbeChoiceBatch
from src.core.temporal_decision_sequence import CHECKPOINTS
from test_clm_catalog_learning import parts
from test_clm_reserved_probe import features

literal_cli_capture: dict[str, Any] | None = None


def plan():
    identity, values = features(torch)
    base, training, reads = study.build_reserved_probe_batches(
        identity, values, catalog_training=True
    )
    benchmark = study.build_catalog_training_batches(identity, base)
    return identity, values, benchmark, training, reads


def source(identity):
    return CLMCatalogLearner(
        *parts(torch),
        identity=identity,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=12, max_rows=4),
    )


@pytest.fixture(scope="module")
def completed():
    identity, values, (batches, choices), training, reads = plan()
    learner = source(identity)
    report: dict[str, Any] = {}
    caller = torch.get_rng_state().clone()
    initial = values.clone()
    observations = []
    hook = study.make_reserved_probe_hook(
        training, reads, report, catalog_training=True, linear_fit_control=True
    )

    def checkpoint(phase, owner):
        captured = owner._capture()
        work = owner.work
        hook(phase, owner)
        assert _equal(vars(captured), vars(owner._capture())) and work == owner.work
        observations.append((phase, owner.completed_updates))

    run_temporal_stream(
        learner, batches, choices, report, catalog_training=True, on_checkpoint=checkpoint
    )
    assert torch.equal(caller, torch.get_rng_state()) and torch.equal(initial, values)
    return report, learner, observations


def test_should_run_one_real_fit_per_checkpoint_and_preserve_benchmark(completed):
    report, learner, observations = completed
    assert observations == [(phase, n) for n, phase in CHECKPOINTS]
    assert learner.completed_updates == 12 and learner.training_examples == 36
    assert report["learner_work"]["commits"] == 12 and len(report["stream"]["receipts"]) == 12
    assert "reserved_probes" not in report and report["pending_linear_probe"] is None
    assert len(report["linear_reserved_probes"]) == 4
    for row in report["linear_reserved_probes"]:
        assert row["complete"] and row["optimizer_steps"] == 0
        assert row["linear_fit"]["rows_returned"] == 2 and row["linear_fit"]["attempts"] == 1
        assert row["initialization"] == "cold_raw_features_no_source_weights"
        assert [len(r["probabilities"]) for r in row["readouts"]] == [2, 2, 4]
        assert all(r["work"]["optimizer_steps"] == 0 for r in row["readouts"])
        assert row["fit_summary"]["squared_error"] < 1e-24


def test_should_score_all_cached_endpoints_only_after_original_stream(completed):
    report, _, _ = completed
    scored = study.score_reserved_probes(report)
    assert scored["cached_probability_rows"] == 32 and len(scored["checkpoints"]) == 4
    assert "not warm plasticity/retention" in scored["scope"]
    assert all(tuple(r["groups"]) == linear.GROUPS for r in scored["checkpoints"])
    assert "after_minus_before" not in json.dumps(scored)
    partial = deepcopy(report)
    partial["sequence_complete"] = False
    with pytest.raises(ValueError, match="finish all"):
        study.score_reserved_probes(partial)


@pytest.mark.parametrize("fault", ["probabilities", "work", "role", "fit", "proof", "phase"])
def test_should_validate_all_cached_records_before_any_metric(completed, fault):
    report = deepcopy(completed[0])
    last = report["linear_reserved_probes"][-1]
    if fault == "probabilities":
        last["readouts"][0]["probabilities"][0] = [0.2, 0.3, 0.5]
    elif fault == "work":
        last["readouts"][0]["work"]["rows_returned"] = 1
    elif fault == "role":
        last["readouts"][2]["role"] = "probe_test"
    elif fault == "fit":
        last["linear_fit"]["attempts"] = 2
    elif fault == "proof":
        report["linear_probe_proof"] = "[]"
    else:
        last["phase"] = "after_A"
    with pytest.raises(ValueError):
        linear.score_linear_records(
            report, score_group=lambda *a: pytest.fail("scored corrupt input")
        )


@pytest.mark.parametrize(
    "fault",
    [
        "feature_data",
        "target",
        "prefix",
        "prefix_owner",
        "source",
        "rng",
        "fit_cost",
        "lost_fit",
        "lost_read",
        "fit_model",
        "fit_owner",
    ],
)
def test_should_refuse_changed_or_uncertain_paid_work_without_retry(monkeypatch, fault):
    identity, _, _, training, reads = plan()
    learner = source(identity)
    report: dict[str, Any] = {}
    fired = False
    captured: list[Any] = []

    def save():
        nonlocal fired
        pending = report.get("pending_linear_probe")
        if fired or not pending or "linear_fit" not in pending:
            return
        fired = True
        if fault == "feature_data":
            training["baseline"][0].states.data.add_(0.01)
        elif fault == "target":
            object.__setattr__(training["baseline"][1], "candidate_indices", (0, 0))
        elif fault == "source":
            learner._work["commits"] += 1
        elif fault == "rng":
            state = torch.get_rng_state()
            state[0] ^= 1
            torch.set_rng_state(state)
        elif fault == "fit_owner":
            object.__setattr__(captured[0], "model", deepcopy(captured[0].model))
        elif fault == "fit_model":
            model = captured[0].model
            object.__setattr__(
                captured[0], "model", replace(model, squared_error=model.squared_error + 0.01)
            )

    hook = study.make_reserved_probe_hook(
        training, reads, report, catalog_training=True, linear_fit_control=True, save=save
    )
    if fault in ("fit_model", "fit_owner"):
        original_fit = linear.fit_clm_linear_control

        def remember(*args):
            value = original_fit(*args)
            captured.append(value)
            return value

        monkeypatch.setattr(linear, "fit_clm_linear_control", remember)
    if fault in ("prefix", "prefix_owner"):
        hook("baseline", learner)
        if fault == "prefix":
            report["linear_reserved_probes"][0]["readouts"][0]["probabilities"][0] = [0.2, 0.3, 0.5]
        else:
            report["linear_reserved_probes"][0] = deepcopy(report["linear_reserved_probes"][0])
    if fault in ("fit_cost", "lost_fit"):
        original = linear.fit_clm_linear_control

        def corrupt(*args):
            value = original(*args)
            if fault == "lost_fit":
                raise RuntimeError("lost fit return")
            args[2]["linear_fit"]["attempts"] = 2
            return value

        monkeypatch.setattr(linear, "fit_clm_linear_control", corrupt)
    if fault == "lost_read":
        original_read = linear.predict_clm_linear_control

        def lost(*args):
            original_read(*args)
            raise RuntimeError("lost read return")

        monkeypatch.setattr(linear, "predict_clm_linear_control", lost)
    caller = torch.get_rng_state().clone()
    try:
        with pytest.raises((ValueError, RuntimeError)):
            hook("baseline", learner)
        with pytest.raises(ValueError):
            hook("baseline", learner)
    finally:
        torch.set_rng_state(caller)
    if fault not in ("prefix", "prefix_owner"):
        assert report["failed_linear_probe"]["complete"] is False
        assert report["failed_linear_probe"]["linear_fit"]["attempts"] >= 1
        assert "preserved_paid_prefix" in report["failed_linear_probe"]


def test_should_admit_read_only_old_task_role_without_deployment_or_final():
    _, _, _, training, reads = plan()
    batch, targets = training["baseline"]
    fitted = fit_clm_linear_control(batch, targets, {})
    p = predict_clm_linear_control(fitted, reads["baseline"]["old_task"], batch.candidate_ids, {})
    assert p.shape == (4, 3) and fitted.role == "probe_train"
    with pytest.raises(ValueError):
        FrozenProbeChoiceBatch(batch.identity, batch.states, batch.actions, "final_test")
    from src.adapters.clm_linear_decision import CLMLinearDecisionPredictor

    # Sentinel owner: the role must fail before encoder/action access or work.
    encoder = FrozenCLMEncoder.__new__(FrozenCLMEncoder)
    with pytest.raises(ValueError, match="disposable probe fit"):
        CLMLinearDecisionPredictor(encoder, fitted, None, instructions="x")


@pytest.mark.parametrize(
    "option",
    [
        "probe_rehearsal",
        "probe_marker_control",
        "probe_train_readouts",
        "probe_state_only",
        "train_state_only",
        "train_evidence_memory",
        "paired_replay_candidate",
        "typed_development_readout",
        "compare_catalog_support",
    ],
)
def test_should_refuse_conflicting_api_options_before_loader(monkeypatch, option):
    monkeypatch.setattr(study, "load_local_clm", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="linear probes require"):
        study.run_case(
            Path("unused"),
            Path("unused"),
            {},
            catalog_training=True,
            reserved_task_probes=True,
            probe_linear_fit_control=True,
            **{option: True},
        )


@pytest.mark.parametrize("value", [0, 1, None])
def test_should_refuse_nonboolean_selection_before_source_access(value):
    with pytest.raises(ValueError):
        study.run_case(Path("unused"), Path("unused"), {}, probe_linear_fit_control=value)


def test_should_refuse_cli_without_required_catalog_before_output(monkeypatch, tmp_path):
    path = tmp_path / "unused.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "study",
            "--model-dir",
            "x",
            "--head",
            "x",
            "--output",
            str(path),
            "--probe-linear-fit-control",
        ],
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not path.exists()


@pytest.mark.parametrize("timing", ["before", "after"])
def test_should_refuse_new_native_mode_at_paid_boundary(timing):
    identity, _, _, training, reads = plan()
    learner = source(identity)
    report: dict[str, Any] = {}

    def save():
        pending = report["pending_linear_probe"]
        if timing == "before" or "linear_fit" in pending:
            report["probe_parameter_scope"] = "state_only"

    hook = study.make_reserved_probe_hook(
        training, reads, report, catalog_training=True, linear_fit_control=True, save=save
    )
    with pytest.raises(ValueError, match="native probe/replay"):
        hook("baseline", learner)
    assert ("linear_fit" in report["failed_linear_probe"]) == (timing == "after")


def test_should_refuse_mixed_cached_modes_before_any_metric(completed):
    report = deepcopy(completed[0])
    report["training_replay_consent"] = True
    with pytest.raises(ValueError, match="native probe/replay"):
        linear.score_linear_records(report, score_group=lambda *a: pytest.fail("scored"))


def test_should_refuse_boolean_source_clock_before_fit(monkeypatch):
    identity, _, _, training, reads = plan()
    learner = source(identity)
    report: dict[str, Any] = {}
    hook = study.make_reserved_probe_hook(
        training, reads, report, catalog_training=True, linear_fit_control=True
    )
    monkeypatch.setattr(type(learner), "completed_updates", property(lambda self: False))
    with pytest.raises(ValueError, match="source/order"):
        hook("baseline", learner)
    assert report["pending_linear_probe"] is None


class LiteralProbeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        x = input_ids.float()
        value = torch.stack(
            (x / 18, torch.where(input_ids % 2 == 1, 1.0, -1.0) * (x + 1) / 19), dim=-1
        )
        return SimpleNamespace(last_hidden_state=value)


def test_should_complete_actual_public_cli_with_literal_loader(monkeypatch, tmp_path):
    global literal_cli_capture
    texts, actions = study.feature_texts(include_reserved_probe=True)
    lookup = {s: i + 1 for i, s in enumerate(texts + actions)}

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, *, add_special_tokens):
            return dict(input_ids=[lookup[text]])

    head1, head2, scale = parts(torch)
    loaded = SimpleNamespace(
        model=LiteralProbeBackbone().eval(),
        tokenizer=Tokenizer(),
        state_head=head1,
        action_head=head2,
        logit_scale=scale.detach(),
        identity=FrozenEncoderIdentity("literal", "fixed", "fixed", "last", text_recipe(2048), 2),
    )
    calls = []

    def load(*args, **kwargs):
        calls.append(True)
        return loaded

    monkeypatch.setattr(study, "load_local_clm", load)
    real_threads = torch.set_num_threads
    threads = []

    def one_thread(n):
        threads.append(n)
        real_threads(1)

    monkeypatch.setattr(torch, "set_num_threads", one_thread)
    path = tmp_path / "actual.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "study",
            "--model-dir",
            "x",
            "--head",
            "x",
            "--output",
            str(path),
            "--catalog-training",
            "--reserved-task-probes",
            "--probe-linear-fit-control",
        ],
    )
    study.main()
    saved = json.loads(path.read_text())
    assert saved["complete"] and len(calls) == 1
    assert len(saved["reserved_probe_scores"]["checkpoints"]) == 4
    assert saved["reserved_probe_scores"]["cached_probability_rows"] == 32
    assert saved["stream"]["completed_updates"] == 12
    assert threads == [2]
    literal_cli_capture = dict(
        definition="literal_cli_fixture_not_native_v1",
        requested_threads=threads,
        actual_fixture_threads=1,
        report=saved,
    )
    with pytest.raises(ValueError, match="refusing"):
        study.main()
    assert len(calls) == 1
