"""Public constructor key controls with explicit setup and lookup phases."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import FrameType
from typing import Any, NoReturn

import pytest

from provider_construction_profile import ProviderConstructionProfile
from test_provider_mixed_construction import Parts, TYPE_DICT, ZERO


HOOK_NAMES = ("len", "hash", "eq", "repr", "getattribute")
HOOKS = dict.fromkeys(HOOK_NAMES, 0)
HASH_VALUE = 0
SUBCASES = ("key128", "key129", "size512", "size513", "foreign", "subclass")
NAMES = (
    "test_construction_keys_preserves_module_metadata_boundaries",
    "test_construction_keys_preserves_native_class_boundaries",
    "test_construction_keys_reproves_pending_frame_after_prior_hook",
)
FRAME_LOCALS = TYPE_DICT.__get__(FrameType, type)["f_locals"]
FRAME_CODE = TYPE_DICT.__get__(FrameType, type)["f_code"]
FRAME_GLOBALS = TYPE_DICT.__get__(FrameType, type)["f_globals"]
TYPE_NAME = TYPE_DICT.__get__(type, type)["__name__"]
EXCEPTION_ARGS = TYPE_DICT.__get__(BaseException, type)["args"]


class HostileKey(str):
    def __len__(self):
        HOOKS["len"] += 1
        return str.__len__(self)

    def __hash__(self):
        HOOKS["hash"] += 1
        return HASH_VALUE

    def __eq__(self, other):
        HOOKS["eq"] += 1
        return other is self

    def __repr__(self):
        HOOKS["repr"] += 1
        return "hostile-key"

    def __getattribute__(self, name):
        HOOKS["getattribute"] += 1
        return object.__getattribute__(self, name)


def reset_hooks():
    for name in HOOK_NAMES:
        HOOKS[name] = 0


def safe_error(error):
    arguments = EXCEPTION_ARGS.__get__(error, BaseException)
    return dict(
        type=TYPE_NAME.__get__(type(error), type),
        message=arguments[0] if arguments and type(arguments[0]) is str else None,
    )


def choose_key(mode, authority_name):
    global HASH_VALUE
    if mode == "foreign":
        return object()
    if mode == "subclass":
        # A colliding key exposes authority lookups made before key validation.
        HASH_VALUE = hash(authority_name)
        return str.__new__(HostileKey, "h" * 128)
    return "k" * (129 if mode == "key129" else 128)


def fixture_counts(parts_count, owner_receivers=0):
    return dict(
        parts=parts_count,
        source_compilations=parts_count,
        source_module_bodies=parts_count,
        source_class_bodies=parts_count,
        record_class_bodies=parts_count,
        record_constructor_bodies=0,
        helper_record_constructor_bodies=0,
        owner_class_witnesses=parts_count,
        record_class_witnesses=parts_count,
        owner_receivers=owner_receivers,
        retained_code_witnesses=6 * parts_count,
        retained_function_witnesses=4 * parts_count,
        retained_module_witnesses=3 * parts_count,
        caller_paths=parts_count,
        callback_paths=parts_count,
        policy_rows=parts_count,
    )


def metadata_case(tmp_path, mode, *, native_class):
    kit = Parts()
    prior = (sys.getprofile(), threading.getprofile())
    namespace = TYPE_DICT.__get__(kit.owner_type, type) if native_class else kit.record_ns
    baseline_size = len(namespace)
    reset_hooks()
    key = choose_key(mode, "__dict__" if native_class else "Record")
    construction = HOOKS.copy()
    reset_hooks()
    intern_calls = 0
    intern_returned_original_is: bool | None = None
    if native_class and type(key) is str:
        original_key = key
        key = sys.intern(original_key)
        intern_calls = 1
        intern_returned_original_is = key is original_key
        del original_key
    intern = HOOKS.copy()
    assert not any(intern.values())
    reset_hooks()
    padding = []
    class_attempts = class_created = 0
    setup_error = None
    selected = kit.policy
    selected_role = kit.role
    target_class = kit.owner_type
    added = False
    insertion_prefix = dict.fromkeys(HOOK_NAMES, 0)
    descriptor_effects = dict.fromkeys(HOOK_NAMES, 0)
    try:
        if native_class and mode in ("foreign", "subclass"):
            class_attempts = 1
            try:
                class_fields = {
                    name: value
                    for name, value in namespace.items()
                    if name not in ("__dict__", "__weakref__")
                }
                class_fields[key] = None
                target_class = type("KeyOwner", (object,), class_fields)
                class_created = 1
                namespace = TYPE_DICT.__get__(target_class, type)
                insertion_prefix = HOOKS.copy()
                reset_hooks()
                descriptor = namespace["__dict__"]
                selected_role = replace(kit.role, owner_type=target_class, descriptor=descriptor)
                selected = replace(kit.policy, role=selected_role)
                descriptor_effects = HOOKS.copy()
                reset_hooks()
            except Exception as error:
                setup_error = safe_error(error)
        else:
            if native_class:
                type.__setattr__(target_class, key, None)
            else:
                namespace[key] = None
            added = True
        if setup_error is None and mode in ("size512", "size513"):
            target_size = 512 if mode == "size512" else 513
            padding = ["pad_" + str(index) for index in range(target_size - len(namespace))]
            for name in padding:
                if native_class:
                    type.__setattr__(target_class, name, None)
                else:
                    namespace[name] = None
        insertion = {name: insertion_prefix[name] + HOOKS[name] for name in HOOK_NAMES}
        reset_hooks()
        retained = setup_error is None and any(candidate is key for candidate in namespace)
        retention = HOOKS.copy()
        reset_hooks()
        namespace_size = len(namespace)
        declaration_attempts = declaration_accepts = declaration_refusals = current_checks = 0
        guard_result = "recipe-unusable"
        error_record = None
        if retained:
            declaration_attempts = 1
            try:
                guard = kit.profile(tmp_path / "keys-breach.json", (selected,), selected_role)
            except ValueError as error:
                declaration_refusals = 1
                guard_result = "refused"
                error_record = safe_error(error)
            else:
                declaration_accepts = 1
                guard_result = "accepted"
                guard.require_current_enrollments()
                current_checks = 1
                assert guard.counts == ZERO and guard.enrollment_counts == (0,)
            if mode in ("key128", "size512"):
                assert guard_result == "accepted"
            else:
                assert guard_result == "refused"
        elif mode not in ("foreign", "subclass"):
            pytest.fail("native string recipe was not retained")
        validation = HOOKS.copy()
        assert not any(validation.values())
    finally:
        reset_hooks()
        for name in padding:
            if native_class:
                type.__delattr__(target_class, name)
            else:
                del namespace[name]
        if added:
            if native_class:
                type.__delattr__(target_class, key)
            else:
                del namespace[key]
        kit.record_ns["Record"] = kit.record_type
        cleanup = HOOKS.copy()
    assert sys.getprofile() is prior[0] and threading.getprofile() is prior[1]
    assert (
        len(TYPE_DICT.__get__(kit.owner_type, type) if native_class else kit.record_ns)
        == baseline_size
    )
    assert not (tmp_path / "keys-breach.json").exists()
    return dict(
        mode=mode,
        baseline_size=baseline_size,
        namespace_size=namespace_size,
        padding_keys=len(padding),
        key_retained_is=retained,
        intern_calls=intern_calls,
        intern_returned_original_is=intern_returned_original_is,
        native_class_attempts=class_attempts,
        native_class_created=class_created,
        additional_role_rows=int(selected_role is not kit.role),
        additional_policy_rows=int(selected is not kit.policy),
        setup_error=setup_error,
        guard_result=guard_result,
        declaration_attempts=declaration_attempts,
        declaration_accepts=declaration_accepts,
        declaration_refusals=declaration_refusals,
        current_checks=current_checks,
        error=error_record,
        hostile_phases=dict(
            construction=construction,
            intern=intern,
            insertion=insertion,
            descriptor=descriptor_effects,
            retention=retention,
            validation=validation,
            cleanup=cleanup,
        ),
        restored=True,
        restored_hooks=True,
    )


def metadata_work(tmp_path, *, native_class, name):
    rows = [metadata_case(tmp_path, mode, native_class=native_class) for mode in SUBCASES]
    record = dict(
        test_name=name,
        kind="class" if native_class else "module",
        subcases=rows,
        fixtures=fixture_counts(6),
        class_foreign_recipes_qualified=all(row["key_retained_is"] for row in rows),
        registered_attempts=0,
        registered_bodies=0,
        callback_calls=0,
        provider_calls=0,
        factory_calls=0,
        delegate_calls=0,
        workers=0,
        restored_hooks=True,
    )
    (tmp_path / "keys-work.json").write_text(json.dumps(record), encoding="utf8")


def test_construction_keys_preserves_module_metadata_boundaries(tmp_path):
    metadata_work(tmp_path, native_class=False, name=NAMES[0])


def test_construction_keys_preserves_native_class_boundaries(tmp_path):
    metadata_work(tmp_path, native_class=True, name=NAMES[1])


class KeyRecordingProfile(ProviderConstructionProfile):
    directory: Path
    kit: Any
    hook_evidence: dict
    denied: bool

    def snapshot(self):
        hook = dict(self.hook_evidence, validation=HOOKS.copy())
        offending = (
            sum(
                self.kit.effects["counts"][name] for name in ("constructor", "provider", "delegate")
            )
            if self.denied
            else 0
        )
        return dict(
            counts=self.counts,
            body_labels=list(self.kit.effects["bodies"]),
            side_effects=self.kit.effects["counts"].copy(),
            enrollment_counts=list(self.enrollment_counts),
            hook=hook,
            fixtures=fixture_counts(1, 1),
            offending_bodies=offending,
        )

    def _abort(
        self, reason: str, label: str = "profile", attempt: int = 0, limit: int = 0
    ) -> NoReturn:
        try:
            raw = json.dumps(self.snapshot()).encode("utf8")
            assert len(raw) <= 4096
            (self.directory / "keys-counts.json").write_bytes(raw)
        finally:
            super()._abort(reason, label, attempt, limit)


def run_frame_child(mode, directory):
    assert mode in SUBCASES
    kit = Parts(capacity=1)
    evidence: dict[str, Any] = {}
    frame_code = kit.prepared.classes[0].constructor_code
    constructor_globals = kit.loaded.source.namespace
    fired = False

    def prior_hook(frame, event, argument):
        nonlocal fired
        if fired or event != "call" or FRAME_CODE.__get__(frame, FrameType) is not frame_code:
            return
        fired = True
        assert type(frame) is FrameType
        assert FRAME_GLOBALS.__get__(frame, FrameType) is constructor_globals
        assert FRAME_LOCALS is TYPE_DICT.__get__(FrameType, type)["f_locals"]
        mapping = FRAME_LOCALS.__get__(frame, FrameType)
        assert mapping.get("value") is kit.initial_value
        original_self = mapping.get("self")
        assert type(original_self) is kit.owner_type
        reset_hooks()
        key = choose_key(mode, "self")
        construction = HOOKS.copy()
        reset_hooks()
        mapping[key] = None
        padding = []
        if mode in ("size512", "size513"):
            padding = [
                "pad_" + str(index)
                for index in range((512 if mode == "size512" else 513) - len(mapping))
            ]
            for name in padding:
                mapping[name] = None
        insertion = HOOKS.copy()
        reset_hooks()
        retained = any(candidate is key for candidate in mapping)
        padding_retained = sum(any(candidate is name for candidate in mapping) for name in padding)
        retention = HOOKS.copy()
        reset_hooks()
        assert retained and padding_retained == len(padding)
        self_retained = mapping.get("self") is original_self
        arg_retained = mapping.get("value") is kit.initial_value
        authority_reads = HOOKS.copy()
        reset_hooks()
        evidence.update(
            fired=True,
            native_frame_code_is=True,
            native_frame_globals_is=True,
            native_getter_is=True,
            original_self_is=self_retained,
            original_arg_is=arg_retained,
            key_retained_is=retained,
            namespace_size=len(mapping),
            padding_keys=len(padding),
            padding_retained_is=padding_retained,
            hostile_phases=dict(
                construction=construction,
                insertion=insertion,
                retention=retention,
                authority_reads=authority_reads,
            ),
            validation=HOOKS.copy(),
        )

    original_main, original_thread = sys.getprofile(), threading.getprofile()
    sys.setprofile(prior_hook)
    try:
        guard = kit.profile(directory / "keys-breach.json", kind=KeyRecordingProfile)
        guard.kit, guard.directory, guard.hook_evidence = kit, directory, evidence
        guard.denied = mode not in ("key128", "size512")
        with guard.measure():
            owner, result = kit.factory(kit.initial_value)
            assert result is kit.initial_value and owner.value is kit.initial_value
        assert sys.getprofile() is prior_hook and threading.getprofile() is original_thread
        guard.require_current_enrollments()
        assert fired and guard.enrollment_counts == (1,)
        assert not any(HOOKS.values())
        snapshot = guard.snapshot()
        snapshot["hook"]["validation"] = HOOKS.copy()
    except Exception:
        (directory / "caught").write_text("unexpected", encoding="ascii")
        raise
    finally:
        sys.setprofile(original_main)
        threading.setprofile(original_thread)
    assert sys.getprofile() is original_main and threading.getprofile() is original_thread
    if mode not in ("key128", "size512"):
        (directory / "continued").write_text("unexpected", encoding="ascii")
    record = dict(mode=mode, exit=0, snapshot=snapshot, receipt=None, restored_hooks=True)
    (directory / "keys-child.json").write_text(json.dumps(record), encoding="utf8")


def run_owned_child(command, environment):
    hidden = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    child = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=hidden,
    )
    try:
        stdout, stderr = child.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=1.5,
                        check=True,
                        creationflags=hidden,
                    )
                except (subprocess.TimeoutExpired, OSError):
                    child.kill()
            else:
                child.kill()
        finally:
            try:
                child.communicate(timeout=1.5)
            except subprocess.TimeoutExpired:
                child.kill()
        raise
    assert child.returncode is not None
    return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def test_construction_keys_reproves_pending_frame_after_prior_hook(tmp_path):
    prior = (sys.getprofile(), threading.getprofile())
    records = []
    for mode in SUBCASES:
        directory = tmp_path / mode
        directory.mkdir()
        environment = os.environ.copy()
        environment.update(
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(Path(__file__).resolve().parent)
        )
        code = "from pathlib import Path; import sys; from test_provider_construction_keys import run_frame_child; run_frame_child(sys.argv[1], Path(sys.argv[2]))"
        child = run_owned_child(
            [sys.executable, "-B", "-c", code, mode, str(directory)], environment
        )
        assert not child.stdout and not child.stderr
        assert len(child.stdout) <= 4096 and len(child.stderr) <= 4096
        for filename in ("caught", "continued", "offending_body"):
            assert not (directory / filename).exists()
        accepted = mode in ("key128", "size512")
        assert child.returncode == (0 if accepted else 73)
        if accepted:
            raw = (directory / "keys-child.json").read_bytes()
            assert len(raw) <= 4096 and not (directory / "keys-breach.json").exists()
            saved = json.loads(raw)
            snapshot, receipt, receipt_bytes = saved["snapshot"], None, 0
            assert saved["restored_hooks"]
        else:
            raw = (directory / "keys-counts.json").read_bytes()
            assert len(raw) <= 4096
            snapshot = json.loads(raw)
            breach = (directory / "keys-breach.json").read_bytes()
            assert len(breach) <= 1024
            receipt, receipt_bytes = json.loads(breach), len(breach)
            assert receipt == dict(
                reason="measurement_failure", counter="profile", attempt=0, limit=0
            )
        assert not any(snapshot["hook"]["validation"].values())
        assert (
            snapshot["hook"]["key_retained_is"]
            and snapshot["hook"]["original_self_is"]
            and snapshot["hook"]["original_arg_is"]
        )
        assert snapshot["offending_bodies"] == 0
        records.append(
            dict(
                mode=mode,
                exit=child.returncode,
                snapshot=snapshot,
                receipt=receipt,
                receipt_bytes=receipt_bytes,
                stdout_bytes=len(child.stdout),
                stderr_bytes=len(child.stderr),
                registered_attempts=sum(snapshot["counts"].values()),
                registered_bodies=len(snapshot["body_labels"]),
                caught=0,
                continued=0,
                parent_restored_hooks=True,
            )
        )
    assert sys.getprofile() is prior[0] and threading.getprofile() is prior[1]
    record = dict(
        test_name=NAMES[2],
        kind="pending-frame",
        children=records,
        fixtures=fixture_counts(6, 6),
        restored_hooks=True,
        workers=0,
    )
    (tmp_path / "keys-work.json").write_text(json.dumps(record), encoding="utf8")
