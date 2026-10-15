"""Independent native owner dictionary controls with finite stand-in dispatch.

Why native witnesses: a callback may remain valid in a detached dictionary while
the owner's actual storage has changed. These controls retain the original native
descriptor and dictionary, and exercise that gap without original providers.
"""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import BuiltinFunctionType, GetSetDescriptorType
from typing import cast

import pytest

from provider_owner_namespace_profile import (
    OwnerNamespaceBinding,
    ProviderOwnerNamespaceProfile,
    require_current_namespaces,
)
from provider_runtime_binding import (
    ProviderRuntimeBindingProfile,
    RuntimeBinding,
    require_current_bindings,
)


def assert_namespace_refused(owner, owner_type, descriptor, namespace):
    with pytest.raises((TypeError, ValueError)):
        require_current_namespaces(
            (OwnerNamespaceBinding(owner, owner_type, descriptor, namespace),)
        )


def assert_namespace_rows_refused(rows):
    with pytest.raises((TypeError, ValueError)):
        require_current_namespaces(rows)


def test_should_accept_native_owner_aliases_multiple_owners_and_finite_boundaries():
    class Owner:
        pass

    first, second = Owner(), Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    first_namespace: dict[str, object] = descriptor.__get__(first, Owner)
    second_namespace: dict[str, object] = descriptor.__get__(second, Owner)
    for index in range(511):
        first_namespace[f"key{index}"] = None
    first_namespace["k" * 128] = None
    first_row = OwnerNamespaceBinding(first, Owner, descriptor, first_namespace)
    second_row = OwnerNamespaceBinding(second, Owner, descriptor, second_namespace)
    alias = first
    alias_row = OwnerNamespaceBinding(alias, Owner, descriptor, first_namespace)
    require_current_namespaces((first_row,))
    require_current_namespaces((first_row, alias_row, second_row))
    require_current_namespaces((first_row,) * 64)
    assert alias is first and first_namespace is not second_namespace
    assert descriptor.__get__(first, Owner) is first_namespace
    assert descriptor.__get__(second, Owner) is second_namespace
    with pytest.raises(FrozenInstanceError):
        setattr(first_row, "namespace", {})


def test_should_read_native_dictionary_without_hostile_instance_lookup():
    events: list[str] = []

    class HostileOwner:
        def __getattribute__(self, name):
            events.append("getattribute")
            raise AssertionError("instance attribute hook dispatched")

        def __getattr__(self, name):
            events.append("getattr")
            raise AssertionError("instance fallback hook dispatched")

        def __eq__(self, other):
            events.append("equality")
            raise AssertionError("instance equality dispatched")

        def __hash__(self):
            events.append("hash")
            raise AssertionError("instance hash dispatched")

        def __repr__(self):
            events.append("repr")
            raise AssertionError("instance repr dispatched")

    owner = HostileOwner()
    descriptor = cast(GetSetDescriptorType, HostileOwner.__dict__["__dict__"])
    namespace: dict[str, object] = descriptor.__get__(owner, HostileOwner)
    namespace["retained"] = owner
    events.clear()
    require_current_namespaces((OwnerNamespaceBinding(owner, HostileOwner, descriptor, namespace),))
    assert descriptor.__get__(owner, HostileOwner) is namespace
    assert events == []


def test_should_refuse_wrong_owner_class_descriptor_and_dictionary_without_dispatch():
    class Owner:
        pass

    class OtherOwner:
        pass

    owner, other = Owner(), Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    other_descriptor = cast(GetSetDescriptorType, OtherOwner.__dict__["__dict__"])
    namespace: dict[str, object] = descriptor.__get__(owner, Owner)
    assert_namespace_refused(other, Owner, descriptor, namespace)
    assert_namespace_refused(owner, OtherOwner, descriptor, namespace)
    assert_namespace_refused(owner, Owner, other_descriptor, namespace)
    assert_namespace_refused(owner, Owner, descriptor, namespace.copy())
    for invalid_descriptor in (
        None,
        object(),
        property(),
        Owner.__dict__,
        type.__dict__["__dict__"],
    ):
        assert_namespace_refused(owner, Owner, invalid_descriptor, namespace)


def test_should_refuse_dictionary_and_exact_class_swaps_with_retained_original_witnesses():
    class Owner:
        pass

    class OtherOwner:
        pass

    def provider():
        raise AssertionError("validation must not dispatch")

    owner = Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    namespace: dict[str, object] = descriptor.__get__(owner, Owner)
    namespace["callback"] = provider
    row = OwnerNamespaceBinding(owner, Owner, descriptor, namespace)
    binding = RuntimeBinding(namespace, "callback", provider, provider.__code__)
    object.__setattr__(owner, "__dict__", {"callback": provider})
    # Why check both: callback identity alone still accepts detached storage.
    require_current_bindings((binding,))
    assert_namespace_rows_refused((row,))
    object.__setattr__(owner, "__dict__", namespace)
    require_current_namespaces((row,))
    object.__setattr__(owner, "__class__", OtherOwner)
    assert_namespace_rows_refused((row,))


def test_should_refuse_malformed_rows_and_non_native_namespace_types():
    class Owner:
        pass

    class ForeignTuple(tuple):
        pass

    class ForeignDict(dict):
        pass

    class ForeignRow(OwnerNamespaceBinding):
        pass

    owner = Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    namespace: dict[str, object] = descriptor.__get__(owner, Owner)
    row = OwnerNamespaceBinding(owner, Owner, descriptor, namespace)
    malformed_rows: list[object] = [(), (row,) * 65, [row], ForeignTuple((row,)), (object(),)]
    for rows in malformed_rows:
        assert_namespace_rows_refused(rows)
    assert_namespace_rows_refused((ForeignRow(owner, Owner, descriptor, namespace),))
    invalid_namespaces: list[object] = [None, [], ForeignDict(namespace), Owner.__dict__]
    for invalid_namespace in invalid_namespaces:
        assert_namespace_refused(owner, Owner, descriptor, invalid_namespace)
    for invalid_class in (None, object(), Owner()):
        assert_namespace_refused(owner, invalid_class, descriptor, namespace)


def test_should_refuse_hostile_dictionary_keys_before_native_slot_lookup():
    events: list[str] = []

    class Owner:
        pass

    class HostileKey(str):
        def __hash__(self):
            events.append("hash")
            return str.__hash__(self)

        def __eq__(self, other):
            events.append("equality")
            raise AssertionError("foreign key equality dispatched")

        def __repr__(self):
            events.append("repr")
            raise AssertionError("foreign key repr dispatched")

    owner = Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    namespace: dict[object, object] = descriptor.__get__(owner, Owner)
    for key in (HostileKey("callback"), 3, "k" * 129):
        namespace.clear()
        namespace[key] = owner
        events.clear()
        assert_namespace_refused(owner, Owner, descriptor, namespace)
        assert events == []
    namespace.clear()
    for index in range(513):
        namespace[f"key{index}"] = None
    events.clear()
    assert_namespace_refused(owner, Owner, descriptor, namespace)
    assert events == []


def test_should_refuse_property_inherited_foreign_and_custom_metaclass_descriptors():
    events: list[str] = []

    class Owner:
        pass

    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])

    def hostile_getter(instance):
        events.append("property")
        raise AssertionError("property dispatched")

    PropertyOwner = type("PropertyOwner", (), {"__dict__": property(hostile_getter)})

    class InheritedOwner(Owner):
        __slots__ = ()

    ForeignDescriptorOwner = type("ForeignDescriptorOwner", (), {"__dict__": descriptor})

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

    class MetaOwner(metaclass=HostileMeta):
        pass

    for owner_type in (PropertyOwner, InheritedOwner, ForeignDescriptorOwner):
        owner = owner_type()
        events.clear()
        assert_namespace_refused(owner, owner_type, descriptor, {})
        assert events == []
    meta_owner = MetaOwner()
    meta_class_namespace = type.__getattribute__(MetaOwner, "__dict__")
    meta_descriptor = cast(GetSetDescriptorType, meta_class_namespace["__dict__"])
    meta_namespace = meta_descriptor.__get__(meta_owner, MetaOwner)
    events.clear()
    assert_namespace_refused(meta_owner, MetaOwner, meta_descriptor, meta_namespace)
    assert events == []
    assert_namespace_refused(PropertyOwner(), PropertyOwner, PropertyOwner.__dict__["__dict__"], {})
    assert events == []


def test_should_refuse_hostile_row_descriptors_and_unbounded_owner_class_keys():
    events: list[str] = []

    class ForeignRow(OwnerNamespaceBinding):
        def __getattribute__(self, name):
            events.append("row_attribute")
            raise AssertionError("foreign row attribute dispatched")

    foreign_row = object.__new__(ForeignRow)
    assert_namespace_rows_refused((foreign_row,))
    assert events == []

    class LongKeyOwner:
        pass

    class WideOwner:
        pass

    setattr(LongKeyOwner, "k" * 129, None)
    for index in range(512):
        setattr(WideOwner, f"key{index}", None)
    for owner_type in (LongKeyOwner, WideOwner):
        owner = owner_type()
        descriptor = cast(GetSetDescriptorType, owner_type.__dict__["__dict__"])
        namespace = descriptor.__get__(owner, owner_type)
        assert_namespace_refused(owner, owner_type, descriptor, namespace)


def test_should_refuse_uncovered_callback_namespaces_and_detached_storage_before_install(tmp_path):
    class Owner:
        pass

    def provider():
        raise AssertionError("refused profile must not dispatch")

    owner = Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    namespace: dict[str, object] = descriptor.__get__(owner, Owner)
    namespace["callback"] = provider
    rows = (OwnerNamespaceBinding(owner, Owner, descriptor, namespace),)
    binding = RuntimeBinding(namespace, "callback", provider, provider.__code__)
    uncovered: dict[str, object] = {"callback": provider}
    uncovered_binding = RuntimeBinding(uncovered, "callback", provider, provider.__code__)
    entries = ((provider, "calls", 1),)
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    with pytest.raises((TypeError, ValueError)):
        ProviderOwnerNamespaceProfile(entries, tmp_path / "breach", (uncovered_binding,), rows)
    guard = ProviderOwnerNamespaceProfile(entries, tmp_path / "breach", (binding,), rows)
    object.__setattr__(owner, "__dict__", {"callback": provider})
    with pytest.raises((TypeError, ValueError)):
        ProviderOwnerNamespaceProfile(entries, tmp_path / "breach", (binding,), rows)
    with pytest.raises((TypeError, ValueError)):
        with guard.measure():
            pytest.fail("detached pre-install dictionary admitted")
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert not (tmp_path / "breach").exists()


def test_should_count_actual_owner_dispatch_and_joined_future_worker_then_restore_profiles(
    tmp_path,
):
    bodies: list[str] = []

    class Owner:
        pass

    def provider():
        bodies.append("provider")

    def delegate():
        bodies.append("delegate")

    first, second = Owner(), Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    first_namespace: dict[str, object] = descriptor.__get__(first, Owner)
    second_namespace: dict[str, object] = descriptor.__get__(second, Owner)
    native = cast(BuiltinFunctionType, time.monotonic)
    first_namespace.update(callback=provider, alias=provider, native=native)
    second_namespace["callback"] = delegate
    namespaces = (
        OwnerNamespaceBinding(first, Owner, descriptor, first_namespace),
        OwnerNamespaceBinding(second, Owner, descriptor, second_namespace),
    )
    bindings = (
        RuntimeBinding(first_namespace, "callback", provider, provider.__code__),
        RuntimeBinding(first_namespace, "alias", provider, provider.__code__),
        RuntimeBinding(first_namespace, "native", native, native),
        RuntimeBinding(second_namespace, "callback", delegate, delegate.__code__),
    )
    entries = ((provider, "calls", 3), (delegate, "delegate", 1), (native, "native", 1))
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    guard = ProviderOwnerNamespaceProfile(entries, tmp_path / "breach", bindings, namespaces)
    assert isinstance(guard, ProviderRuntimeBindingProfile)
    with guard.measure():
        getattr(first, "callback")()
        getattr(first, "alias")()
        getattr(second, "callback")()
        getattr(first, "native")()
        worker = threading.Thread(target=getattr(first, "callback"), daemon=False)
        worker.start()
        # Why no timed join: threading's monotonic alias shares the C entry token.
        worker.join()
        assert not worker.is_alive()
    assert bodies == ["provider", "provider", "delegate", "provider"]
    assert guard.counts == {"calls": 3, "delegate": 1, "native": 1}
    assert descriptor.__get__(first, Owner) is first_namespace
    assert descriptor.__get__(second, Owner) is second_namespace
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert not (tmp_path / "breach").exists()
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            pytest.fail("spent owner namespace profile reused")
    (tmp_path / "work.json").write_text(
        json.dumps(dict(bodies=4, workers=1, joined=True, restored=True, counts=guard.counts)),
        encoding="utf8",
    )


def run_owner_namespace_negative(mode, directory):
    """Only six disposable children invoke this bounded negative dispatch."""
    path = Path(directory)

    class Owner:
        pass

    class OtherOwner:
        pass

    def provider():
        (path / "body").write_text("forbidden", encoding="utf8")

    def replacement():
        (path / "foreign_body").write_text("forbidden", encoding="utf8")

    owner = Owner()
    descriptor = cast(GetSetDescriptorType, Owner.__dict__["__dict__"])
    namespace: dict[str, object] = descriptor.__get__(owner, Owner)
    namespace["callback"] = provider
    namespaces = (OwnerNamespaceBinding(owner, Owner, descriptor, namespace),)
    bindings = (RuntimeBinding(namespace, "callback", provider, provider.__code__),)
    receipt = path / "breach.json"
    if mode == "receipt_existing":
        receipt.write_bytes(b"retained existing receipt")
    elif mode == "receipt_missing":
        receipt = path / "missing" / "breach.json"

    def prior(frame, event, arg):
        if mode == "prior_dict" and event == "call" and frame.f_code is provider.__code__:
            object.__setattr__(owner, "__dict__", {"callback": replacement})

    if mode == "prior_dict":
        sys.setprofile(prior)
    guard = ProviderOwnerNamespaceProfile(((provider, "calls", 1),), receipt, bindings, namespaces)
    try:
        with guard.measure():
            if mode in ("dict", "receipt_existing", "receipt_missing"):
                object.__setattr__(owner, "__dict__", {"callback": replacement})
            elif mode == "class":
                object.__setattr__(owner, "__class__", OtherOwner)
            elif mode == "callback":
                namespace["callback"] = replacement
            getattr(owner, "callback")()
    except BaseException:
        (path / "caught").write_text("forbidden", encoding="utf8")
    (path / "continued").write_text("forbidden", encoding="utf8")


@pytest.mark.parametrize(
    "mode", ("dict", "class", "prior_dict", "callback", "receipt_existing", "receipt_missing")
)
def test_should_exit_73_before_actual_owner_body_catch_or_continuation_on_mutation(mode, tmp_path):
    args = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_provider_owner_namespace_profile import run_owner_namespace_negative; "
        "run_owner_namespace_negative(sys.argv[2], sys.argv[3])",
        str(Path(__file__).resolve().parent),
        mode,
        str(tmp_path),
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, timeout=5)
    assert result.returncode == 73 and len(result.stdout) + len(result.stderr) <= 4096
    assert not any(
        (tmp_path / name).exists() for name in ("body", "foreign_body", "caught", "continued")
    )
    receipt = None
    if mode == "receipt_existing":
        raw = (tmp_path / "breach.json").read_bytes()
        assert len(raw) <= 1024 and raw == b"retained existing receipt"
    elif mode == "receipt_missing":
        assert not (tmp_path / "missing").exists() and not (tmp_path / "breach.json").exists()
    else:
        raw = (tmp_path / "breach.json").read_bytes()
        assert len(raw) <= 1024
        receipt = json.loads(raw)
        reason = "callback_binding_changed" if mode == "callback" else "owner_namespace_changed"
        assert receipt == dict(reason=reason, counter="profile", attempt=0, limit=0)
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
