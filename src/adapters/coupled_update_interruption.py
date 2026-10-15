"""Engineering interruption after native computation, before checkpoint delivery.

Inputs are a fresh owned directory and fixed train fixtures. Outputs retain full
A and terminal authority/host evidence. The surviving parent keeps serving A;
this demo neither retries uncertain work nor implements managed-bundle recovery.
"""

from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import sys
import sysconfig
import time
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.coupled_checkpoint_interruption import FEATURES, LIMITS, _bytes, _model, _plain
from src.adapters.coupled_update_completion import BINDING, _known
from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.recovery_coordination import RecoveryCosts
from src.infra.bounded_json_child import BoundedJsonChild
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_recovery_observer import WindowsRecoveryObserver

CASES = ("pc", "neutral_cpc", "active_cpc")


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_update_interruption_engineering_v1",
        cases=list(CASES),
        native_training_calls=9,
        native_energy_evaluations=36,
        preparatory_training_calls=6,
        admitted_child_updates=3,
        owned_child_launches=3,
        actor_predictions=12,
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        training=dict(role="train", features=[list(row) for row in FEATURES], targets=[[1], [0]]),
        external_training_settings=dict(
            learning_rate=0.02, inference_steps=2, inference_learning_rate=0.05
        ),
        binding=asdict(BINDING),
        limits=asdict(LIMITS),
        limitations="Engineering pause telemetry is not completion authority. Surviving Windows parent; no uncertain retry, actor promotion, full managed-bundle or coordinator-loss recovery. Fixture bindings are not authentication.",
    )


def _child(stack: ExitStack) -> BoundedJsonChild:
    executable = getattr(sys, "_base_executable", None)
    if type(executable) is not str or not Path(executable).is_file():
        raise ValueError("native interruption requires the existing physical base interpreter")
    repository = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(repository), sysconfig.get_path("purelib")))
    child = stack.enter_context(
        BoundedJsonChild(
            [
                executable,
                "-u",
                "-B",
                "-X",
                "utf8",
                "-m",
                "src.adapters.coupled_update_worker",
                "--pause-before-result",
            ],
            cwd=repository,
            env=env,
        )
    )
    if child.receive() != dict(event="hello", pid=child.pid, version=list(sys.version_info[:3])):
        raise ValueError("native interruption hello differs from owned physical worker")
    return child


def _case(directory: Path, family: str) -> dict[str, Any]:
    actor = _model(family)
    x, y = np.asarray(FEATURES, dtype=np.float64), np.array([[1.0], [0.0]])
    for _ in range(2):
        actor.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05)
    a = actor.snapshot_state()
    codec = CoupledCheckpointCodec()
    raw_a = codec.encode(a, binding=BINDING, limits=LIMITS)
    digest_a = sha256(raw_a).hexdigest()
    directory.mkdir()
    path_a, path_b = directory / "actor-a.json", directory / "candidate-b.json"
    BoundedCheckpointFileStore(path_a, codec).save(a, binding=BINDING, limits=LIMITS)
    predictions = dict(before=actor.predict_proba(x.copy()).tolist())
    callbacks = dict(prepare=0, validate=0, refused_prepare=0, refused_validate=0)
    records: dict[str, Any] = {}
    observations: dict[str, Any] = {}
    responses = []
    with ExitStack() as stack:
        child = _child(stack)
        api = WindowsRecoveryApi()
        anchor = stack.enter_context(WindowsProcessHandle.pin(os.getpid(), api=api))
        worker = stack.enter_context(WindowsProcessHandle.pin(child.pid, api=api))
        observer = WindowsRecoveryObserver(anchor)
        initial = observer.observe(worker)
        observations["initial"] = asdict(initial)
        known = _known(initial, anchor.identity, worker.identity, digest_a, f"interrupted-{family}")
        journal = SqliteRecoveryJournal.create(directory / "authority.sqlite", known)
        coordinator = stack.enter_context(
            RecoveryCoordinator(known, journal, observer, anchor, worker)
        )
        records["initial"] = asdict(journal.read())
        exit_codes = []

        def prepare():
            callbacks["prepare"] += 1
            pending = journal.read()
            usage = pending.metadata.usage
            if not pending.metadata.uncertain_work or (
                usage.updates_admitted,
                usage.updates_completed,
                usage.copied_bytes,
                usage.checkpoint_attempts,
            ) != (1, 0, 4 * LIMITS.max_encoded_bytes, 1):
                raise AssertionError("native work was not charged before child dispatch")
            records["pending"] = asdict(pending)
            child.send(
                dict(
                    op="update",
                    binding=asdict(BINDING),
                    limits=asdict(LIMITS),
                    expected_sha256=digest_a,
                    payload_utf8=raw_a.decode(),
                )
            )
            response = child.receive()
            if (
                response != dict(event="computed", training_calls=1)
                or type(response["training_calls"]) is not int
            ):
                raise ValueError("worker did not reach the exact pre-result engineering pause")
            responses.append(response)
            observations["computed_live"] = asdict(observer.observe(worker))
            if observations["computed_live"]["previous_owner_ended"] is not False:
                raise AssertionError("native worker ended before the controlled interruption")
            predictions["pending"] = actor.predict_proba(x.copy()).tolist()
            # Why here: killing before prepare returns exercises the coordinator's
            # pending-work guard; telemetry never reaches checkpoint validation.
            exit_codes.append(child.kill())
            observations["ended"] = asdict(observer.observe(worker))
            if (
                worker.is_ended() is not True
                or observations["ended"]["previous_owner_ended"] is not True
            ):
                raise AssertionError("owned registered native worker did not end")
            return response

        def validate(response):
            callbacks["validate"] += 1
            raise AssertionError("interrupted worker reached checkpoint validation")

        try:
            coordinator.execute_native_update(
                RecoveryCosts(
                    updates=1, copy_bytes=4 * LIMITS.max_encoded_bytes, checkpoint_attempts=1
                ),
                prepare,
                validate,
            )
        except ValueError as error:
            native_failure = str(error)
            if native_failure != "coordinator registered process is ended or unsupported":
                raise
        else:
            raise AssertionError("interrupted native update was completed")
        records["result"] = asdict(journal.read())
        predictions["after_result"] = actor.predict_proba(x.copy()).tolist()

        def refused_prepare():
            callbacks["refused_prepare"] += 1
            raise AssertionError("uncertain work was dispatched again")

        def refused_validate(value):
            callbacks["refused_validate"] += 1
            raise AssertionError("uncertain work was validated again")

        try:
            coordinator.execute_native_update(
                RecoveryCosts(updates=1), refused_prepare, refused_validate
            )
        except ValueError as error:
            final_refusal = str(error)
            if (
                final_refusal
                != "coordinator previously failed; independent reconciliation required"
            ):
                raise
        else:
            raise AssertionError("interrupted update was retried")
        records["terminal"] = asdict(journal.read())
        if not coordinator.terminal_confirmed or not records["terminal"]["metadata"]["stopped"]:
            raise AssertionError("interrupted native authority was not terminally recorded")
        predictions["after_refusal"] = actor.predict_proba(x.copy()).tolist()
        identity, child_pid, exit_code = asdict(worker.identity), child.pid, child.returncode
        if exit_code != exit_codes[0] or exit_code is None or child.stderr:
            raise AssertionError("interrupted worker was not reaped cleanly")

    terminal = records["terminal"]["metadata"]
    usage = terminal["usage"]
    if (
        not terminal["uncertain_work"]
        or terminal["manifest"] != asdict(known.metadata.manifest)
        or (
            usage["updates_admitted"],
            usage["updates_completed"],
            usage["copied_bytes"],
            usage["checkpoint_attempts"],
            usage["grants"],
        )
        != (1, 0, 4 * LIMITS.max_encoded_bytes, 1, 0)
    ):
        raise AssertionError("native interruption lost original authority or spent charges")
    if callbacks != dict(prepare=1, validate=0, refused_prepare=0, refused_validate=0):
        raise AssertionError("native interruption crossed the refused callback boundary")
    if (
        _bytes(path_a) != raw_a
        or path_b.exists()
        or any(p != predictions["before"] for p in predictions.values())
    ):
        raise AssertionError("native interruption changed actor A or published B")
    after = actor.snapshot_state()
    np.testing.assert_equal(asdict(after), asdict(a))
    return dict(
        family=family,
        path_a=str(path_a),
        path_b=str(path_b),
        checkpoint_a=dict(digest=digest_a, payload_utf8=raw_a.decode(), snapshot=asdict(a)),
        actor_after=asdict(after),
        predictions=predictions,
        response=responses[0],
        authority=records,
        observations=observations,
        callbacks=callbacks,
        native_failure=native_failure,
        final_attempt_refusal=final_refusal,
        terminal_confirmed=True,
        child_pid=child_pid,
        worker_identity=identity,
        exit_code=exit_code,
    )


def run_example(directory: Path) -> dict[str, Any]:
    if sys.platform != "win32":
        raise OSError("native interruption example requires Windows retained-process observation")
    directory = directory.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = [_case(directory / family, family) for family in CASES]
    if time.monotonic() - started > 60:
        raise RuntimeError("native interruption example exceeded60s")
    np.testing.assert_equal(
        rows[0]["actor_after"]["parameters"], rows[1]["actor_after"]["parameters"]
    )
    remaining = [str(path) for path in directory.glob("*/.*.tmp")]
    retained = sum(path.stat().st_size for path in directory.glob("*/*") if path.is_file())
    if remaining or retained > 64 * 1024**2:
        raise AssertionError("native interruption left an orphan or exceeded storage bounds")
    return _plain(
        dict(
            **describe_example(),
            directory=str(directory),
            rows=rows,
            remaining_temporary_files=remaining,
            retained_file_bytes=retained,
        )
    )
