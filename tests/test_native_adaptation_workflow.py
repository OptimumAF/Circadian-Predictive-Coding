"""NEW bounded native-composition tests; literal ports/files, never a simulator."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import taskspace_memory_native_adaptation as native
from scripts import run_taskspace_memory_demo as cli
from scripts import push_cube_reserved_probe_inputs as inputs
from src.adapters import numpy_retained_nearest_bc as retained
from src.adapters import continuous_nearest_probe as near
from src.adapters import continuous_stream_sequence as sequence
from src.adapters import continuous_probe as probe
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from test_taskspace_memory_demo import bundle
from test_retained_adaptation_workflow import plan
from test_taskspace_memory_native import receipt
from test_retained_nearest_memory import _process_peak_bytes

WORK: dict[str, int] = {}
PRIOR: dict[str, int] = {}
CAPS = dict(
    initializers=160,
    updates=2048,
    incoming_rows=8192,
    forks=8192,
    predictions=8192,
    query_rows=16384,
    candidates=524288,
    coordinates=6291456,
    snapshots=262144,
    validators=1048576,
    copy_bytes=2 * 1024**3,
    hash_bytes=4 * 1024**3,
    residual_rows=16384,
    endpoints=256,
    native_calls=256,
)
FIFO = ReplayRetentionPolicy("recent_fifo")


def charge(key, count=1):
    WORK[key] = WORK.get(key, 0) + count
    assert WORK[key] + PRIOR.get(key, 0) <= CAPS[key], key


@pytest.fixture(scope="module", autouse=True)
def bounded_literal_work():
    PRIOR.update(json.loads(os.environ.get("NATIVE_ADAPT_PRIOR", "{}")))

    def wrap(method, kind):
        def call(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("incoming_rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                candidates = rows * len(owner._state.prototype_features)
                for key, count in (
                    ("query_rows", rows),
                    ("candidates", candidates),
                    ("coordinates", candidates * 12),
                ):
                    charge(key, count)
            return method(owner, *args, **kwargs)

        return call

    validate, ids = retained.retained_snapshot, retained._ids
    residual = probe.diagnostics.continuous_bc_residuals

    def validated(state, policy, rule):
        charge("validators")
        charge("copy_bytes", state.prototype_features.nbytes + state.prototype_actions.nbytes)
        return validate(state, policy, rule)

    def hashed(features, actions):
        charge("hash_bytes", features.nbytes + actions.nbytes + 24 * len(features))
        return ids(features, actions)

    def residuals(*args, **kwargs):
        charge("residual_rows", len(args[0]))
        return residual(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("literal workflow attempted model RNG")

    with pytest.MonkeyPatch.context() as patch:
        for name, kind in (
            ("__init__", "initializers"),
            ("train_batch", "updates"),
            ("fork", "forks"),
            ("predict_with_work", "predictions"),
            ("snapshot_state", "snapshots"),
        ):
            patch.setattr(
                retained.NumpyRetainedNearestBCLearner,
                name,
                wrap(getattr(retained.NumpyRetainedNearestBCLearner, name), kind),
            )
        for module in (retained, near, sequence):
            patch.setattr(module, "retained_snapshot", validated)
        patch.setattr(retained, "_ids", hashed)
        patch.setattr(probe.diagnostics, "continuous_bc_residuals", residuals)
        patch.setattr(np.random, "default_rng", forbidden)
        yield
    peak = _process_peak_bytes()
    receipt = os.environ.get("NATIVE_ADAPT_RECEIPT")
    if receipt:
        with Path(receipt).open("x") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate={k: WORK.get(k, 0) + PRIOR.get(k, 0) for k in CAPS},
                    peak_bytes=peak,
                ),
                handle,
                indent=2,
            )
    assert peak <= 512 * 1024**2


def install(monkeypatch, output, *, returning=True, genuine=False, fault=None):
    """Real private stream only for two integration cases; other cases use scalar ports."""
    calls, raws, holders = [], [], []
    source = SimpleNamespace(state={"version": 1})
    if not genuine:
        birth = deepcopy(source.state)

        def pin(owner):
            return SimpleNamespace(
                state=SimpleNamespace(
                    completed_updates=9 if returning else 8,
                    training_examples=26 if returning else 25,
                    retained_rows=2,
                )
            )

        def require(owner, original):
            if owner.state != birth:
                raise RuntimeError("literal readonly source changed")

        monkeypatch.setattr(native, "pin_retained_native", pin)
        monkeypatch.setattr(native, "require_retained_native", require)

        def workflow(payload, **options):
            def check(operation=None):
                options["check_boundary"]()
                if operation is not None:
                    operation()
                options["check_boundary"]()

            phases = ("before_probe", "after_probe") + (("after_return",) if returning else ())
            for phase in phases:
                charge("endpoints")
                options["endpoint_observer"](source, phase, check)
            return dict(complete=True, adaptation={"complete": True, "paid": 2 if returning else 1})

        monkeypatch.setattr(native, "run_demo", workflow)

    def make(child, seed, probe23=False, push=False):
        charge("native_calls")
        name = "push17" if push else "probe23" if probe23 else f"pick{seed}"
        calls.append(name)
        if fault == "lost" and len(calls) == 2:
            raise RuntimeError("native call did not return")
        state = native.pin_retained_native(child).state
        raw = receipt(17 if probe23 else seed, push=push)
        raw.update(
            seed=seed,
            source_updates=state.completed_updates,
            source_training_rows=state.training_examples,
            nearest_prototype_rows=state.retained_rows,
            nearest_distance_candidate_upper_bound=state.retained_rows,
            nearest_distance_candidates_completed=state.retained_rows,
            nearest_distance_feature_upper_bound=state.retained_rows * 12,
            nearest_distance_feature_values_completed=state.retained_rows * 12,
        )
        if probe23:
            binding = pick_cube_probe_binding("probe_train")
            raw.update(role="probe_train", probe_train_seed=23, scoring_only=True)
            raw["episode"].update(
                context=binding.context, episode_id="pick-cube-probe-train-smoke-23"
            )
            if fault in ("role", "context", "reset", "authority", "work"):
                if fault == "role":
                    raw["role"] = "final"
                elif fault == "context":
                    raw["episode"]["context"] = "other"
                elif fault == "reset":
                    raw["probe_train_seed"] = 24
                elif fault == "authority":
                    raw["scoring_only"] = False
                else:
                    raw["nearest_distance_candidates_completed"] += 1
        raws.append(raw)
        return raw

    def pick(path, *, learner, action_history_recipe, **options):
        assert action_history_recipe.mode == "executed"
        raw = make(
            learner,
            options.get("probe_train_seed", options.get("train_seed")),
            probe23="probe_train_seed" in options,
        )
        with path.open("x") as handle:
            json.dump(raw, handle)
        holder = SimpleNamespace(report=raw, corrections=None)
        holders.append(holder)
        return holder

    def push(child, *, label, seed):
        return make(child, seed, push=True)

    monkeypatch.setattr(native, "run_frozen_rollout", pick)
    monkeypatch.setattr(native, "roll_push", push)
    read = native._read_capture
    damaged = False

    def capture(path, maximum):
        nonlocal damaged
        value = read(path, maximum)
        if (
            not damaged
            and path == output
            and raws
            and json.loads(value[0]).get("outcomes", {}).get("pick17") is not None
        ):
            if fault in (
                "raw",
                "holder",
                "main-bytes",
                "main-owner",
                "pick-bytes",
                "pick-owner",
                "source",
            ):
                damaged = True
                if fault == "raw":
                    raws[0]["source_updates"] += 1
                elif fault == "holder":
                    holders[0].report = deepcopy(raws[0])
                elif fault == "source":
                    source.state["version"] = 2
                else:
                    target = (
                        output
                        if fault.startswith("main")
                        else output.parent / f"{output.stem}-pick17.json"
                    )
                    if fault.endswith("owner"):
                        saved = target.read_bytes()
                        target.rename(target.with_suffix(".original"))
                        target.write_bytes(saved)
                    else:
                        target.write_text("{}")
        return value

    monkeypatch.setattr(native, "_read_capture", capture)
    return calls, raws, holders


def run(output, returning=True):
    return native.run_native_adaptation(
        bundle(),
        plan(),
        output,
        retention=FIFO,
        max_prototypes=2,
        return_task="pick17" if returning else None,
        input_receipts=[{"role": "probe_train"}, {"role": "probe_test"}],
    )


@pytest.mark.parametrize("returning", [False, True])
def test_real_private_stream_observes_only_final_endpoint_and_keeps_source_costs(
    tmp_path, monkeypatch, returning
):
    output = tmp_path / "native.json"
    calls, _, _ = install(monkeypatch, output, returning=returning, genuine=True)
    result = run(output, returning)
    assert result["complete"], result.get("error")
    assert calls == [x[0] for x in native.SCENES]
    assert result["observed_native_phase"] == ("after_return" if returning else "after_probe")
    assert result["cpu_endpoint_phases"] == ["before_probe", "after_probe"] + (
        ["after_return"] if returning else []
    )
    assert result["unobserved_native_phases"] == result["cpu_endpoint_phases"][:-1]
    assert result["workflow"]["native_outcomes"] is None
    assert result["workflow"]["stream"]["model_state"]["candidate_training_examples"] == 7
    assert len(result["native"]) == 5 and result["attempted"] == result["returned"] == calls
    assert all(
        not v["any_success"] for v in result["outcomes"].values()
    )  # Negative scores are complete.
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))


@pytest.mark.parametrize(
    "fault", ["raw", "holder", "main-bytes", "main-owner", "pick-bytes", "pick-owner", "source"]
)
def test_prior_paid_evidence_corruption_refuses_before_second_scene(tmp_path, monkeypatch, fault):
    output = tmp_path / "native.json"
    calls, _, _ = install(monkeypatch, output, fault=fault)
    result = run(output)
    assert not result["complete"] and calls == ["pick17"]
    assert len(result["native"]) == 1 and result["native"][0]["report"]["source_updates"] == 9
    assert output.with_suffix(".failure.json").exists()


@pytest.mark.parametrize("fault", ["role", "context", "reset", "authority", "work"])
def test_reserved_receipt_failure_retains_paid_return_without_admitting_more_scenes(
    tmp_path, monkeypatch, fault
):
    output = tmp_path / "native.json"
    calls, _, _ = install(monkeypatch, output, fault=fault)
    result = run(output)
    assert not result["complete"] and calls == ["pick17", "probe23"]
    assert result["returned"] == calls and len(result["native"]) == 2
    assert list(result["outcomes"]) == ["pick17"]


def test_lost_native_return_stays_unknown_and_original_workflow_costs_are_preserved(
    tmp_path, monkeypatch
):
    output = tmp_path / "native.json"
    calls, _, _ = install(monkeypatch, output, fault="lost")
    result = run(output)
    assert not result["complete"] and calls == ["pick17", "probe23"]
    assert result["returned"] == ["pick17"] and result["unreturned_native_work_unknown"]
    assert len(result["native"]) == 1


@pytest.mark.parametrize("index", range(6))
def test_every_output_collision_refuses_before_original_inputs(tmp_path, monkeypatch, index):
    output = tmp_path / "native.json"
    paths = native.native_adaptation_outputs(output)
    paths[index].write_text("preserve")
    monkeypatch.setattr(native, "_plan", lambda *a: pytest.fail("collision reached input work"))
    with pytest.raises(ValueError):
        run(output)


def argv(output):
    return [
        "--adaptation-probe",
        "train.json",
        "test.json",
        "--retention",
        "recent_fifo",
        "--max-prototypes",
        "2",
        "--return-task",
        "pick17",
        "--native",
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("index", range(6))
def test_cli_admits_all_native_destinations_before_loading_pair(tmp_path, monkeypatch, index):
    output = tmp_path / "native.json"
    paths = native.native_adaptation_outputs(output)
    paths[index].write_text("old")
    monkeypatch.setattr(
        inputs, "load_reserved_probe_inputs", lambda *a: pytest.fail("pair read before collision")
    )
    with pytest.raises(ValueError):
        cli.main(argv(output))


def test_cli_native_dispatch_keeps_pair_receipts_and_does_not_use_cpu_output_writer(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "native.json"
    p = plan()
    seen = []
    monkeypatch.setattr(
        inputs,
        "load_reserved_probe_inputs",
        lambda *a: (replace(p, diagnostics=True), [{"role": "probe_train"}]),
    )
    monkeypatch.setattr(cli, "_load_default_bundle", bundle)
    monkeypatch.setattr(
        cli, "run_demo", lambda *a, **k: pytest.fail("native dispatched to CPU writer")
    )

    def execute(payload, recipe, path, **options):
        seen.append(options)
        assert recipe.diagnostics is False and recipe.updates == 1 and path == output
        result = {
            "complete": True,
            "native": ["literal final endpoint"],
            "input_receipts": options["input_receipts"],
        }
        path.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(native, "run_native_adaptation", execute)
    assert cli.main(argv(output)) == 0
    assert seen[0]["return_task"] == "pick17" and seen[0]["max_prototypes"] == 2
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())


def test_late_capture_mutation_restores_workflow_and_native_paid_births(tmp_path, monkeypatch):
    output = tmp_path / "native.json"
    calls, raws, _ = install(monkeypatch, output)
    opened = Path.open
    damaged = False

    def changed(path, *args, **kwargs):
        nonlocal damaged
        if not damaged and path == output and args and args[0] == "wb" and len(calls) == 5:
            damaged = True
            raws[0]["source_updates"] = 0
            raise OSError("late main capture mutated paid evidence")
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed)
    result = run(output)
    assert not result["complete"] and len(calls) == 5
    assert result["native"][0]["report"]["source_updates"] == 9
    assert output.with_suffix(".failure.json").exists()
