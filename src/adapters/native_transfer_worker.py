"""Fixed trusted physical native coordinator for the transfer example.

Adopts authority only over the supervisor-owned channel, then delegates charged
restore/update/completion to NativeTransferSession. Strict old/new chronology and
code-object budgets precede work. No child spawning, disk authority bootstrap,
general service or automatic retry.
Opt-in replacement interruption pauses after native mutation and discloses only
pending-authority telemetry; no train result can reach session checkpoint IO.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import threading

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from src.adapters.native_transfer_protocol import BINDING, LIMITS
from src.adapters.native_transfer_session import NativeTransferSession
from src.adapters.native_transfer_pause import PausedNativeModel
from src.adapters.coupled_checkpoint_worker import _model
from src.adapters.windows_transfer_worker import _emit, _fields, _identity
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianPredictiveCodingNetwork as CPC,
)
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork as PC,
    _updated_parameters,
)
from src.core.coupled_energy import evaluate_coupled, predict_coupled
from src.core.coupled_pc_scaling import step_coupled_scaling
from src.core.coupled_scaling_structure import split_coupled_scaling_width
from src.core.coupled_inference import infer_coupled
from src.infra.bounded_json_child import _json_object
from src.infra.recovery_authority_codec import decode_authority, encode_authority
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_recovery_observer import WindowsRecoveryObserver


def _budget(role, *, pause_after_train=False):
    if (
        type(role) is not str
        or role not in ("old", "new")
        or type(pause_after_train) is not bool
        or (pause_after_train and role != "new")
    ):
        raise ValueError("native transfer budget requires old/new role and a new-role pause flag")
    counts = dict(
        constructions=0,
        train_calls=0,
        native_sgd=0,
        raw_scaling_sgd=0,
        energy=0,
        predictions=0,
        split=0,
        guarded_inference=0,
        popen=0,
    )
    limits = dict(
        constructions=4,
        train_calls=1,
        native_sgd=1,
        raw_scaling_sgd=0,
        energy=4,
        predictions=1 if role == "old" or pause_after_train else 2,
        split=0,
        guarded_inference=0,
        popen=0,
    )
    codes = {
        _updated_parameters.__code__: "native_sgd",
        step_coupled_scaling.__code__: "raw_scaling_sgd",
        evaluate_coupled.__code__: "energy",
        predict_coupled.__code__: "predictions",
        split_coupled_scaling_width.__code__: "split",
        infer_coupled.__code__: "guarded_inference",
        subprocess.Popen.__init__.__code__: "popen",
    }

    def profile(frame, event, arg):
        if event != "call":
            return
        code, model, key = frame.f_code, frame.f_locals.get("self"), None
        if (code is PC.__init__.__code__ and type(model) is PC) or (
            code is CPC.__init__.__code__ and type(model) is CPC
        ):
            key = "constructions"
        elif (code is PC.train_epoch.__code__ and type(model) is PC) or (
            code is CPC.train_epoch.__code__ and type(model) is CPC
        ):
            key = "train_calls"
        elif code in codes:
            key = codes[code]
        if key is not None:
            counts[key] += 1
            if counts[key] > limits[key]:
                raise RuntimeError(f"native transfer {key} budget exhausted BEFORE work")
        path = code.co_filename.replace("\\", "/")
        if path.endswith(
            ("/src/app/serving_promotion.py", "/src/app/promotion_guard_evaluation.py")
        ):
            raise RuntimeError("native transfer actor/guard budget is zero BEFORE work")

    return counts, profile


def _request():
    raw = sys.stdin.buffer.readline(65537)
    if len(raw) > 65536 or not raw.endswith(b"\n"):
        raise ValueError("native transfer requires one bounded complete command line")
    return _json_object(raw)


def _adopt(request, own, worker, last, *, pause_after_train=False, counts=None):
    if type(pause_after_train) is not bool:
        raise ValueError("pause_after_train must be an exact boolean")
    _fields(request, {"op", "authority_utf8", "journal_path", "binding", "limits"})
    _fields(request["binding"], {"source_sha256", "policy_sha256"})
    _fields(
        request["limits"], {"max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension"}
    )
    if type(request["authority_utf8"]) is not str or type(request["journal_path"]) is not str:
        raise ValueError("native transfer adoption requires exact strings")
    binding, limits = CodecBinding(**request["binding"]), CodecLimits(**request["limits"])
    if binding != BINDING or limits != LIMITS:
        raise ValueError("native transfer controls differ from frozen local protocol")
    known = decode_authority(request["authority_utf8"].encode("utf-8"))
    if (
        known.anchor != own.identity
        or known.worker != worker.identity
        or known.metadata.observed_ns < last.now_ns
    ):
        raise ValueError(
            "native transfer known authority differs from physical retained observations"
        )
    # Fresh observer carries the independently known cross-process RSS peak.
    observer = WindowsRecoveryObserver(own, peak_rss_bytes=known.metadata.usage.peak_rss_bytes)
    if not pause_after_train:
        return NativeTransferSession(
            known, request["journal_path"], observer, own, worker, binding=binding, limits=limits
        )
    if type(counts) is not dict:
        raise ValueError("paused native transfer adoption requires actual work counters")

    def factory(snapshot):
        # Factory invocation occurs only inside a charged restore/update after
        # session assignment. Read this session's actual pending journal record.
        return PausedNativeModel(
            _model(snapshot), lambda: session.authority(), counts, emit=_emit, wait=_request
        )

    session = NativeTransferSession(
        known,
        request["journal_path"],
        observer,
        own,
        worker,
        binding=binding,
        limits=limits,
        model_factory=factory,
    )
    return session


def _serve(stack, own, api, role, counts, *, pause_after_train=False):
    if type(pause_after_train) is not bool or (pause_after_train and role != "new"):
        raise ValueError("pause_after_train requires an exact boolean and the new role")
    _emit(dict(event="hello", pid=own.identity.pid, version=list(sys.version_info[:3])))
    worker = last = session = None
    restored = updated = exhausted = False
    for _ in range(8):
        request = _request()
        if request == {"op": "exit"} and (session is None or exhausted):
            return
        if request == {"op": "crash"} and role == "old" and updated:
            os._exit(17)
        if request.get("op") == "observe" and worker is None and session is None:
            _fields(request, {"op", "worker", "peak_rss_bytes"})
            identity = _identity(request["worker"])
            worker = stack.enter_context(
                WindowsProcessHandle.pin(identity.pid, expected=identity, api=api)
            )
            observer = WindowsRecoveryObserver(own, peak_rss_bytes=request["peak_rss_bytes"])
            last = observer.observe(worker)
            if last.previous_owner_ended is not False:
                raise ValueError("native transfer worker must be live")
            _emit(dict(event="observed", observation=asdict(last)))
        elif (
            request.get("op") == "adopt"
            and worker is not None
            and last is not None
            and session is None
        ):
            session = (
                _adopt(request, own, worker, last, pause_after_train=True, counts=counts)
                if pause_after_train
                else _adopt(request, own, worker, last)
            )
            stack.callback(session.close)
            _emit(
                dict(
                    event="adopted",
                    authority_utf8=encode_authority(session.authority()).decode("utf-8"),
                    budget=dict(counts),
                )
            )
        elif (
            request.get("op") == "restore"
            and role == "new"
            and session is not None
            and not restored
            and not updated
        ):
            _fields(request, {"op", "source", "expected_sha256"})
            if type(request["source"]) is not str:
                raise ValueError("native restore source must be a native string")
            result = session.restore(Path(request["source"]), request["expected_sha256"])
            restored = True
            _emit(dict(result, budget=dict(counts)))
        elif (
            request.get("op") == "update"
            and session is not None
            and not updated
            and (role == "old" or restored)
        ):
            _fields(
                request,
                {"op", "source", "target", "expected_source_sha256", "expected_target_sha256"},
            )
            if type(request["source"]) is not str or type(request["target"]) is not str:
                raise ValueError("native update paths must be native strings")
            result = session.update(
                Path(request["source"]),
                Path(request["target"]),
                request["expected_source_sha256"],
                request["expected_target_sha256"],
            )
            updated = True
            _emit(dict(result, budget=dict(counts)))
        elif (
            request == {"op": "exhaust"}
            and role == "new"
            and session is not None
            and updated
            and not exhausted
        ):
            result = session.refuse_update()
            exhausted = True
            _emit(dict(result, budget=dict(counts)))
        else:
            raise ValueError("unsupported native transfer command chronology")
    raise ValueError("native transfer command limit exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("old", "new"), required=True)
    parser.add_argument("--pause-after-train", action="store_true")
    args = parser.parse_args()
    if args.pause_after_train and args.role != "new":
        parser.error("--pause-after-train is valid only with --role new")
    if sys.platform != "win32":
        raise OSError("native physical transfer requires Windows")
    counts, profile = (
        _budget(args.role, pause_after_train=True) if args.pause_after_train else _budget(args.role)
    )
    sys.setprofile(profile)
    threading.setprofile(profile)
    try:
        api = WindowsRecoveryApi()
        with ExitStack() as stack:
            own = stack.enter_context(WindowsProcessHandle.pin(api.current_pid(), api=api))
            if args.pause_after_train:
                _serve(stack, own, api, args.role, counts, pause_after_train=True)
            else:
                _serve(stack, own, api, args.role, counts)
        return 0
    finally:
        sys.setprofile(None)
        threading.setprofile(None)


if __name__ == "__main__":
    raise SystemExit(main())
