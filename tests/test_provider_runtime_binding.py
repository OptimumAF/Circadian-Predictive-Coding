"""Independent runtime binding controls using bounded stand-ins only.

Why separate controls: retained callback identity and retained code identity must
both survive admission and active measurement. No original provider is imported.
The single positive worker is joined; six disposable negative children must exit
before a stand-in body, exception handler, or continuation can write a marker.
"""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import cast

import pytest

from provider_entry_profile import ProviderEntryProfile
from provider_runtime_binding import (
    ProviderRuntimeBindingProfile,
    RuntimeBinding,
    require_current_bindings,
)


def assert_refused(namespace, slot, callback, token, owner_type=None, raw_call=None):
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings(
            (RuntimeBinding(namespace, slot, callback, token, owner_type, raw_call),)
        )


def assert_rows_refused(rows):
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings(rows)


def test_should_accept_retained_function_alias_c_method_clock_and_delegate():
    def provider():
        raise AssertionError("validation must not dispatch")

    class Receiver:
        def fetch(self):
            raise AssertionError("validation must not dispatch")

    class Clock:
        def __call__(self):
            raise AssertionError("validation must not dispatch")

    method = cast(MethodType, Receiver().fetch)
    native = cast(BuiltinFunctionType, time.monotonic)
    clock = Clock()
    raw_call = Clock.__dict__["__call__"]
    namespace: dict[str, object] = {
        "provider": provider,
        "alias": provider,
        "native": native,
        "method": method,
        "clock": clock,
        "delegate": provider,
    }
    rows = (
        RuntimeBinding(namespace, "provider", provider, provider.__code__),
        RuntimeBinding(namespace, "alias", provider, provider.__code__),
        RuntimeBinding(namespace, "native", native, native),
        RuntimeBinding(namespace, "method", method, method.__func__.__code__),
        RuntimeBinding(namespace, "clock", clock, raw_call.__code__, Clock, raw_call),
        RuntimeBinding(namespace, "delegate", provider, provider.__code__),
    )
    require_current_bindings(rows)
    assert namespace["provider"] is namespace["alias"] is namespace["delegate"]
    with pytest.raises(FrozenInstanceError):
        setattr(rows[0], "slot", "changed")


def test_should_accept_one_and_sixty_four_rows_and_maximum_namespace_shape():
    def provider():
        raise AssertionError("validation must not dispatch")

    namespace: dict[str, object] = {f"key{index}": provider for index in range(511)}
    slot = "s" * 128
    namespace[slot] = provider
    row = RuntimeBinding(namespace, slot, provider, provider.__code__)
    require_current_bindings((row,))
    require_current_bindings((row,) * 64)


def test_should_refuse_foreign_callback_even_with_same_or_equal_code():
    def provider():
        raise AssertionError("validation must not dispatch")

    namespace: dict[str, object] = {"callback": provider}
    row = RuntimeBinding(namespace, "callback", provider, provider.__code__)
    same_code = FunctionType(provider.__code__, provider.__globals__)
    equal_code = FunctionType(provider.__code__.replace(), provider.__globals__)
    assert same_code.__code__ is provider.__code__
    assert equal_code.__code__ == provider.__code__
    assert equal_code.__code__ is not provider.__code__
    for foreign in (same_code, equal_code):
        namespace["callback"] = foreign
        with pytest.raises((TypeError, ValueError)):
            require_current_bindings((row,))
    namespace["callback"] = provider
    assert_refused(namespace, "callback", provider, equal_code.__code__)


def test_should_refuse_changed_code_missing_slot_and_explicit_delegate():
    def provider():
        raise AssertionError("validation must not dispatch")

    original_code = provider.__code__
    namespace: dict[str, object] = {"callback": provider, "delegate": provider}
    rows = (
        RuntimeBinding(namespace, "callback", provider, original_code),
        RuntimeBinding(namespace, "delegate", provider, original_code),
    )
    provider.__code__ = original_code.replace()
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings(rows)
    provider.__code__ = original_code
    del namespace["callback"]
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings(rows)
    namespace["callback"] = provider
    namespace["delegate"] = FunctionType(original_code, provider.__globals__)
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings(rows)


def test_should_refuse_rebound_method_changed_method_code_and_native_token():
    class Receiver:
        def fetch(self):
            raise AssertionError("validation must not dispatch")

    receiver = Receiver()
    method = cast(MethodType, receiver.fetch)
    code = method.__func__.__code__
    namespace: dict[str, object] = {"callback": method}
    row = RuntimeBinding(namespace, "callback", method, code)
    namespace["callback"] = MethodType(method.__func__, receiver)
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings((row,))
    namespace["callback"] = method
    method.__func__.__code__ = code.replace()
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings((row,))
    method.__func__.__code__ = code
    namespace["callback"] = time.monotonic
    assert_refused(namespace, "callback", time.monotonic, len)


def test_should_refuse_changed_callable_class_raw_call_and_raw_code():
    class Clock:
        def __call__(self):
            raise AssertionError("validation must not dispatch")

    class OtherClock:
        def __call__(self):
            raise AssertionError("validation must not dispatch")

    clock = Clock()
    raw_call = Clock.__dict__["__call__"]
    code = raw_call.__code__
    namespace: dict[str, object] = {"clock": clock}
    row = RuntimeBinding(namespace, "clock", clock, code, Clock, raw_call)
    object.__setattr__(clock, "__class__", OtherClock)
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings((row,))
    object.__setattr__(clock, "__class__", Clock)
    setattr(Clock, "__call__", OtherClock.__call__)
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings((row,))
    setattr(Clock, "__call__", raw_call)
    raw_call.__code__ = code.replace()
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings((row,))
    raw_call.__code__ = code
    assert_refused(namespace, "clock", clock, code)
    assert_refused(namespace, "clock", clock, code, OtherClock, raw_call)
    assert_refused(namespace, "clock", clock, code, Clock, OtherClock.__call__)


def test_should_refuse_malformed_rows_namespaces_slots_tokens_and_witnesses():
    def provider():
        raise AssertionError("validation must not dispatch")

    class ForeignRow(RuntimeBinding):
        pass

    class ForeignDict(dict):
        pass

    class ForeignTuple(tuple):
        pass

    class ForeignStr(str):
        pass

    namespace: dict[str, object] = {"callback": provider}
    row = RuntimeBinding(namespace, "callback", provider, provider.__code__)
    for rows in ((), (row,) * 65, [row], ForeignTuple((row,)), (object(),)):
        assert_rows_refused(rows)
    with pytest.raises((TypeError, ValueError)):
        require_current_bindings((ForeignRow(namespace, "callback", provider, provider.__code__),))
    non_string_keys: dict[object, object] = dict.fromkeys(range(512))
    mixed_keys: dict[object, object] = non_string_keys.copy()
    mixed_keys["callback"] = provider
    malformed_namespaces: list[object] = [None, [], ForeignDict(namespace), mixed_keys]
    for invalid_namespace in malformed_namespaces:
        assert_refused(invalid_namespace, "callback", provider, provider.__code__)
    for slot in (None, 7, "", "s" * 129, ForeignStr("callback")):
        assert_refused(namespace, slot, provider, provider.__code__)
    for invalid_token in (object(), provider, None, True):
        assert_refused(namespace, "callback", provider, invalid_token)
    assert_refused(namespace, "callback", provider, provider.__code__, type(provider), provider)
    assert_refused(namespace, "callback", provider, provider.__code__, None, provider)
    assert_refused(namespace, "callback", provider, provider.__code__, type(provider), None)
    overcapacity: dict[str, object] = {f"key{index}": provider for index in range(512)}
    overcapacity["callback"] = provider
    assert_refused(overcapacity, "callback", provider, provider.__code__)


def test_should_validate_all_exact_dictionary_keys_before_lookup_without_user_dispatch():
    events: list[str] = []

    class HostileKey(str):
        def __hash__(self):
            events.append("hash")
            return str.__hash__(self)

        def __eq__(self, other):
            events.append("equality")
            raise AssertionError("foreign equality dispatched")

        def __repr__(self):
            events.append("repr")
            raise AssertionError("foreign repr dispatched")

    def provider():
        events.append("body")

    for key in (HostileKey("callback"), 3, "k" * 129):
        namespace: dict[object, object] = {key: provider}
        events.clear()
        assert_refused(namespace, "callback", provider, provider.__code__)
        assert events == []
    namespace = {"callback": provider, HostileKey("unrelated"): provider}
    events.clear()
    assert_refused(namespace, "callback", provider, provider.__code__)
    assert events == []


def test_should_refuse_overcapacity_and_long_keys_in_callable_class_namespace():
    def provider(*args):
        raise AssertionError("validation must not dispatch")

    class LongKeyClock:
        __call__ = provider

    class WideClock:
        __call__ = provider

    setattr(LongKeyClock, "k" * 129, provider)
    for index in range(512):
        setattr(WideClock, f"key{index}", provider)
    for owner in (LongKeyClock, WideClock):
        callback = owner()
        namespace: dict[str, object] = {"callback": callback}
        assert_refused(namespace, "callback", callback, provider.__code__, owner, provider)


def test_should_refuse_descriptors_inheritance_custom_metaclass_and_foreign_attributes():
    events: list[str] = []

    def provider(*args):
        events.append("body")

    class Descriptor:
        def __get__(self, instance, owner):
            events.append("descriptor")
            raise AssertionError("descriptor dispatched")

    class DescriptorClock:
        __call__ = Descriptor()

    class PropertyClock:
        __call__ = property(provider)

    class StaticClock:
        __call__ = staticmethod(provider)

    class ParentClock:
        __call__ = provider

    class InheritedClock(ParentClock):
        pass

    class HostileMeta(type):
        def __getattribute__(self, name):
            events.append("metaclass_attribute")
            raise AssertionError("metaclass attribute dispatched")

        def __hash__(self):
            events.append("metaclass_hash")
            raise AssertionError("metaclass hash dispatched")

        def __eq__(self, other):
            events.append("metaclass_equality")
            raise AssertionError("metaclass equality dispatched")

        def __repr__(self):
            events.append("metaclass_repr")
            raise AssertionError("metaclass repr dispatched")

    class MetaClock(metaclass=HostileMeta):
        __call__ = provider

    class ForeignObject:
        def __getattribute__(self, name):
            events.append("foreign_attribute")
            raise AssertionError("foreign attribute dispatched")

        def __eq__(self, other):
            events.append("foreign_equality")
            raise AssertionError("foreign equality dispatched")

        def __hash__(self):
            events.append("foreign_hash")
            raise AssertionError("foreign hash dispatched")

        def __repr__(self):
            events.append("foreign_repr")
            raise AssertionError("foreign repr dispatched")

    for owner in (DescriptorClock, PropertyClock, StaticClock, InheritedClock, MetaClock):
        callback = owner()
        namespace: dict[str, object] = {"callback": callback}
        events.clear()
        assert_refused(namespace, "callback", callback, provider.__code__, owner, provider)
        assert events == []
    foreign = ForeignObject()
    namespace = {"callback": foreign}
    events.clear()
    assert_refused(namespace, "callback", foreign, provider.__code__)
    assert_refused(namespace, "callback", provider, foreign)
    assert events == []


def test_should_refuse_absent_entry_and_mutation_before_measure_installation(tmp_path):
    def provider():
        raise AssertionError("refused profile must not dispatch")

    def other():
        raise AssertionError("refused profile must not dispatch")

    namespace: dict[str, object] = {"callback": provider}
    rows = (RuntimeBinding(namespace, "callback", provider, provider.__code__),)
    for absent in (other, provider.__code__.replace()):
        with pytest.raises((TypeError, ValueError)):
            ProviderRuntimeBindingProfile(((absent, "calls", 1),), tmp_path / "breach", rows)
    guard = ProviderRuntimeBindingProfile(((provider, "calls", 1),), tmp_path / "breach", rows)
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    namespace["callback"] = other
    with pytest.raises((TypeError, ValueError)):
        with guard.measure():
            pytest.fail("changed pre-install callback admitted")
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert not (tmp_path / "breach").exists()


def test_should_count_main_and_joined_future_worker_with_all_binding_kinds(tmp_path):
    bodies: list[str] = []

    def provider():
        bodies.append("provider")

    def delegate():
        bodies.append("delegate")

    class Receiver:
        def fetch(self):
            bodies.append("method")

    class Clock:
        def __call__(self):
            bodies.append("clock")
            return delegate()

    method = cast(MethodType, Receiver().fetch)
    native = cast(BuiltinFunctionType, time.monotonic)
    clock = Clock()
    raw_call = Clock.__dict__["__call__"]
    namespace: dict[str, object] = {
        "provider": provider,
        "alias": provider,
        "native": native,
        "method": method,
        "clock": clock,
        "delegate": delegate,
    }
    rows = (
        RuntimeBinding(namespace, "provider", provider, provider.__code__),
        RuntimeBinding(namespace, "alias", provider, provider.__code__),
        RuntimeBinding(namespace, "native", native, native),
        RuntimeBinding(namespace, "method", method, method.__func__.__code__),
        RuntimeBinding(namespace, "clock", clock, raw_call.__code__, Clock, raw_call),
        RuntimeBinding(namespace, "delegate", delegate, delegate.__code__),
    )
    entries = (
        (provider, "provider", 3),
        (native, "native", 1),
        (method.__func__, "method", 1),
        (raw_call, "clock", 1),
        (delegate, "delegate", 1),
    )
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    guard = ProviderRuntimeBindingProfile(entries, tmp_path / "breach", rows)
    assert isinstance(guard, ProviderEntryProfile)
    with guard.measure():
        provider()
        alias = provider
        alias()
        time.monotonic()
        method()
        clock()
        worker = threading.Thread(target=provider, daemon=False)
        worker.start()
        # Why no timeout here: timed join uses threading's retained monotonic C
        # alias and would consume the same registered native entry allowance.
        worker.join()
        assert not worker.is_alive()
    assert bodies == ["provider", "provider", "method", "clock", "delegate", "provider"]
    assert guard.counts == {"provider": 3, "native": 1, "method": 1, "clock": 1, "delegate": 1}
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert namespace["alias"] is provider and provider.__code__ is rows[0].token
    assert Clock.__dict__["__call__"] is raw_call and not (tmp_path / "breach").exists()
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            pytest.fail("spent runtime binding guard reused")
    (tmp_path / "work.json").write_text(
        json.dumps(dict(bodies=6, workers=1, joined=True, restored=True, counts=guard.counts)),
        encoding="utf8",
    )


def run_runtime_binding_negative(mode, directory):
    """Called only in six disposable children; every stand-in body is forbidden."""
    path = Path(directory)

    def provider():
        (path / "body").write_text("forbidden", encoding="utf8")

    def replacement(*args):
        (path / "foreign_body").write_text("forbidden", encoding="utf8")

    class Clock:
        def __call__(self):
            (path / "body").write_text("forbidden", encoding="utf8")

    namespace: dict[str, object] = {"callback": provider}
    callback: object = provider
    token = provider.__code__
    owner_type: type | None = None
    raw_call = None
    if mode == "raw_call":
        clock = Clock()
        callback = clock
        owner_type = Clock
        raw_call = Clock.__dict__["__call__"]
        token = raw_call.__code__
        namespace["callback"] = callback
    row = RuntimeBinding(namespace, "callback", callback, token, owner_type, raw_call)
    receipt = path / "breach.json"
    if mode == "receipt_existing":
        receipt.write_bytes(b"retained existing receipt")
    elif mode == "receipt_missing":
        receipt = path / "missing" / "breach.json"

    def prior(frame, event, arg):
        if mode == "prior_slot" and event == "call" and frame.f_code is token:
            namespace["callback"] = replacement

    if mode == "prior_slot":
        sys.setprofile(prior)
    guard = ProviderRuntimeBindingProfile(((token, "calls", 1),), receipt, (row,))
    try:
        with guard.measure():
            if mode in ("slot", "receipt_existing", "receipt_missing"):
                namespace["callback"] = replacement
            elif mode == "code":
                provider.__code__ = replacement.__code__
            elif mode == "raw_call":
                setattr(Clock, "__call__", replacement)
            if mode == "raw_call":
                clock()
            else:
                provider()
    except BaseException:
        (path / "caught").write_text("forbidden", encoding="utf8")
    (path / "continued").write_text("forbidden", encoding="utf8")


@pytest.mark.parametrize(
    "mode", ("slot", "code", "raw_call", "prior_slot", "receipt_existing", "receipt_missing")
)
def test_should_exit_73_on_active_binding_mutation_before_body_catch_or_continuation(
    mode, tmp_path
):
    args = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_provider_runtime_binding import run_runtime_binding_negative; "
        "run_runtime_binding_negative(sys.argv[2], sys.argv[3])",
        str(Path(__file__).resolve().parent),
        mode,
        str(tmp_path),
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, timeout=5)
    assert result.returncode == 73
    assert len(result.stdout) + len(result.stderr) <= 4096
    assert not any(
        (tmp_path / name).exists() for name in ("body", "foreign_body", "caught", "continued")
    )
    receipt = None
    if mode == "receipt_existing":
        raw = (tmp_path / "breach.json").read_bytes()
        assert len(raw) <= 1024 and raw == b"retained existing receipt"
    elif mode == "receipt_missing":
        assert not (tmp_path / "missing").exists()
        assert not (tmp_path / "breach.json").exists()
    else:
        raw = (tmp_path / "breach.json").read_bytes()
        assert len(raw) <= 1024
        receipt = json.loads(raw)
        assert receipt == dict(
            reason="callback_binding_changed", counter="profile", attempt=0, limit=0
        )
    (tmp_path / "child.json").write_text(
        json.dumps(
            dict(
                mode=mode,
                exit=result.returncode,
                stdout_bytes=len(result.stdout),
                stderr_bytes=len(result.stderr),
                receipt=receipt,
                bodies=0,
                caught=0,
                continued=0,
            )
        ),
        encoding="utf8",
    )
