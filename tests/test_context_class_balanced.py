"""Current released TRAIN balancing, unchanged defaults and paid failure prefixes."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_workflow_native as native
from src.adapters import torch_array_weighted as weighted
from src.core.experience import Experience, ExperiencePermissions, ExperienceRole, LabelArrival


@pytest.mark.parametrize(
    "options",
    [
        dict(class_balanced_bp=1),
        dict(method="pc"),
        dict(method="perceptron"),
        dict(context_return=False),
        dict(replay="recent"),
        dict(policy_change=True),
        dict(separate_context_heads=True),
        dict(private_adaptation=True),
        dict(paired_serving="joint"),
    ],
)
def test_should_refuse_invalid_options_before_data_or_models(options, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("read data"))
    kwargs = dict(context_return=True, method="bp", class_balanced_bp=True)
    kwargs.update(options)
    with pytest.raises(ValueError, match="class-balanced"):
        entry.run_example(**kwargs)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--context-return"],
        ["--context-return", "--method", "pc"],
        ["--context-return", "--method", "bp", "--replay", "coverage"],
    ],
)
def test_should_refuse_cli_before_reserving_output(flags, tmp_path, monkeypatch):
    path = tmp_path / "refused.json"
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("ran"))
    with pytest.raises(SystemExit) as error:
        entry.main(["--output", str(path), "--class-balanced-bp", *flags])
    assert error.value.code == 2 and not path.exists()


@pytest.mark.parametrize("diagnostics", [False, True])
def test_should_forward_cli_and_keep_exclusive_output(diagnostics, tmp_path, monkeypatch):
    path = tmp_path / "one.json"
    seen = []

    def run(**kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    flags = ["--output", str(path), "--context-return", "--method", "bp", "--class-balanced-bp"]
    if diagnostics:
        flags.append("--record-train-fit")
    assert entry.main(flags) == 0
    assert seen == [
        dict(
            context_return=True,
            method="bp",
            class_balanced_bp=True,
            **({"record_train_fit": True} if diagnostics else {}),
        )
    ]
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(seen) == 1


def test_should_use_current_arrived_batch_with_original_budget_and_diagnostics(monkeypatch):
    holder = {}
    original_stream = entry.AttributionStream
    original_weighted = weighted.train_class_weighted_batch
    observed = []

    def capture(*args, **kwargs):
        stream = original_stream(*args, **kwargs)
        holder["stream"] = stream
        return stream

    def train(learner, features, targets, weights):
        stream = holder["stream"]
        inbox = stream._runtime._inbox
        ready = inbox._ready(inbox._last_time)
        assert len(ready) == 1
        source, label = inbox._experiences[ready[0]], inbox._labels[ready[0]]
        assert source.observed_at + 1 == label.arrived_at <= inbox._last_time
        assert np.array_equal(targets, label.targets)
        assert np.array_equal(features, source.features)
        counts = np.bincount(targets[:, 0].astype(int), minlength=6)
        expected = tuple(6 / (4 * int(c)) if c else 1.0 for c in counts)
        assert weights == expected
        observed.append(weights)
        return original_weighted(learner, features, targets, weights)

    monkeypatch.setattr(entry, "AttributionStream", capture)
    monkeypatch.setattr(weighted, "train_class_weighted_batch", train)
    r = entry.run_example(
        context_return=True, method="bp", class_balanced_bp=True, record_train_fit=True
    )
    assert r["complete"], r.get("error")
    rows = r["class_balanced_updates"]
    assert len(rows) == len(observed) == 12
    assert [x["source_update"] for x in rows] == list(range(1, 13))
    assert all(x["complete"] and x["role"] == "train" for x in rows)
    assert all(x["class_weights"] == w for x, w in zip(rows, observed, strict=True))
    assert all(x["rows"] == sum(x["class_counts"]) == 6 for x in rows)
    assert all(
        x["diagnostic"] == receipt["diagnostic"]
        for x, receipt in zip(rows, r["applied_updates"], strict=True)
    )
    assert rows[0]["class_counts"] == rows[3]["class_counts"] == rows[8]["class_counts"]
    assert rows[0]["class_counts"] != rows[4]["class_counts"]
    assert r["training_work"]["attempted_rows"] == 72
    assert r["training_work"]["returned_updates"] == r["budget"]["updates_completed"] == 12
    assert r["class_balanced_work"] == dict(
        attempted_updates=12, returned_updates=12, unreturned_work_unknown=False
    )
    assert r["budget"]["limits"]["max_wall_seconds"] == 30
    assert r["budget"]["limits"]["max_process_rss_bytes"] == 512 * 1024**2
    assert all(
        r[x]
        for x in (
            "source_unchanged",
            "actor_unchanged",
            "numpy_rng_unchanged",
            "torch_rng_unchanged",
        )
    )
    assert all(
        r[x] == 0
        for x in ("final_examples", "probe_updates", "replay_examples", "external_actions")
    )
    assert r["stream"]["score_forward_calls"] == 0
    assert len(r["episodes"]) == 24 and len(r["train_fit_readouts"]) == 4
    assert len(entry._encode_report(r)) <= 512 * 1024


def make_pair():
    report: dict[str, Any] = {}
    journal = native.ClassBalancedUpdates(report)
    x = np.eye(6, 9, dtype=np.float64)
    y = np.arange(6, dtype=np.float64).reshape(6, 1)

    def admit(features):
        index = len(journal.pins)
        source = Experience(
            f"batch-{index}",
            "episode",
            index,
            "actor",
            features,
            "train",
            ExperiencePermissions(training=True),
        )
        label = LabelArrival(f"label-{index}", f"batch-{index}", "episode", index + 1, "actor", y)
        journal.admit(source, label, index + 1, "learner")

    balanced = native.counted_learner("bp", 9, 6, admit, lambda: None, balanced=journal)
    ordinary = native.counted_learner("bp", 9, 6, lambda x: None, lambda: None)
    return report, journal, balanced, ordinary, x, y


def test_should_preserve_default_and_all_one_continuation_in_factory():
    report, journal, balanced, ordinary, x, y = make_pair()
    for _ in range(2):
        ordinary.train_batch(x, y)
        diagnostic = balanced.train_batch(x, y)
        assert diagnostic.definition == weighted.TORCH_ARRAY_WEIGHTED_CE_ID
        assert native.same_state(balanced.snapshot_state(), ordinary.snapshot_state())
    assert all(row["class_weights"] == (1.0,) * 6 for row in report["class_balanced_updates"])
    journal.finish()
    assert report["class_balanced_work"]["returned_updates"] == 2


@pytest.mark.parametrize("fault", ["value", "row", "list"])
def test_should_refuse_changed_or_replaced_paid_prefix_before_next_update(fault):
    report, journal, learner, _, x, y = make_pair()
    learner.train_batch(x, y)
    original = deepcopy(report["class_balanced_updates"])
    if fault == "value":
        report["class_balanced_updates"][0]["class_counts"] = (0,) * 6
    elif fault == "row":
        report["class_balanced_updates"][0] = deepcopy(original[0])
    else:
        report["class_balanced_updates"] = deepcopy(original)
    with pytest.raises(ValueError, match="paid evidence changed"):
        learner.train_batch(x, y)
    assert learner.completed_updates == 1
    journal.finish()
    assert report["preserved_class_balanced_updates"] == original
    assert report["class_balanced_work"]["attempted_updates"] == 1
    assert not report["complete"]


def test_should_keep_lost_return_and_unpublished_optimizer_work(monkeypatch):
    report, journal, learner, _, x, y = make_pair()
    torch = native.require_torch()
    before = learner.snapshot_state()
    original = torch.optim.SGD.step

    def lost(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("lost optimizer return")

    monkeypatch.setattr(torch.optim.SGD, "step", lost)
    with pytest.raises(RuntimeError, match="lost optimizer"):
        learner.train_batch(x, y)
    journal.finish()
    assert native.same_state(before, learner.snapshot_state())
    assert report["class_balanced_work"] == dict(
        attempted_updates=1, returned_updates=0, unreturned_work_unknown=True
    )
    assert not report["class_balanced_updates"][0]["complete"]


@pytest.mark.parametrize("role", ["final_test", "probe_train"])
def test_should_refuse_other_roles_before_weighted_update(role, monkeypatch):
    original = entry.prepare_training

    def data(**kwargs):
        spec, batches = original(**kwargs)
        return spec, (replace(batches[0], role=role), *batches[1:])

    monkeypatch.setattr(entry, "prepare_training", data)
    monkeypatch.setattr(weighted, "train_class_weighted_batch", lambda *a: pytest.fail("weighted"))
    r = entry.run_example(context_return=True, method="bp", class_balanced_bp=True)
    assert not r["complete"] and r["class_balanced_work"]["attempted_updates"] == 0
    assert r["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize("effect", ["rate", "gate", "composite"])
def test_should_refuse_controller_effect_before_model_update(effect):
    _, _, learner, _, x, y = make_pair()
    with pytest.raises(ValueError, match="no .*controller"):
        if effect == "rate":
            learner.train_batch_with_rate_multiplier(x, y, 2)
        elif effect == "gate":
            learner.train_batch_with_gate_setting(x, y, "active")
        else:
            learner.train_batch_with_rate_and_gate(x, y, 2, "active")
    assert learner.completed_updates == 0


@pytest.mark.parametrize("when", ["first", "later", "same_value"])
def test_should_bind_original_and_preceding_causal_receipt(when):
    from src.core.experience import AppliedExperience
    from src.core.learner_ports import TrainingDiagnostic

    r: dict[str, Any] = {}
    journal = native.ClassBalancedUpdates(r)
    diagnostic = TrainingDiagnostic(weighted.TORCH_ARRAY_WEIGHTED_CE_ID, 1.0)
    receipt = AppliedExperience("s", "e", "label", "actor", "learner", 0, 1, 1, 1, diagnostic)
    source_receipt = asdict(receipt)
    del source_receipt["diagnostic"]
    row = dict(
        complete=True, source_update=1, diagnostic=asdict(diagnostic), source_receipt=source_receipt
    )
    journal.rows.append(row)
    journal.pins.append((row, deepcopy(row)))
    if when != "first":
        journal.require_receipts((receipt,))
    with pytest.raises(ValueError, match="causal receipt"):
        journal.require_receipts(
            (
                replace(receipt)
                if when == "same_value"
                else replace(receipt, event_id="replacement"),
            )
        )


@pytest.mark.parametrize("fault", ["unarrived", "probe", "final", "unconsented"])
def test_should_refuse_unarrived_or_nonordinary_train_admission(fault):
    r: dict[str, Any] = {}
    journal = native.ClassBalancedUpdates(r)
    role: ExperienceRole = (
        "probe_train" if fault == "probe" else "final_test" if fault == "final" else "train"
    )
    source = Experience(
        "s",
        "e",
        0,
        "actor",
        np.zeros((6, 9)),
        role,
        ExperiencePermissions(training=role in ("train", "probe_train") and fault != "unconsented"),
    )
    label = LabelArrival("label", "s", "e", 1, "actor", np.zeros((6, 1)), role)
    with pytest.raises(ValueError, match="arrived TRAIN"):
        journal.admit(source, label, 0 if fault == "unarrived" else 1, "learner")
    assert journal.attempts == 0 and journal.pending is None
