"""Borrowed constructor recipient identity uses the existing local copy window."""

from contextlib import nullcontext
from copy import deepcopy
import gc
from typing import Any
from weakref import ref

import pytest

from src.core.native_model_copy import ModelCopyLimits, begin_model_copy, observe_model_copies


class Model:
    pass


class Recipient:
    pass


@pytest.mark.parametrize("fail", (False, True))
def test_should_borrow_actual_recipient_synchronously_and_release_it_even_after_failure(fail):
    source = Model()
    source_ref = ref(source)
    receiver = Recipient()
    receiver_ref = ref(receiver)
    saved = []
    observed = []

    def observer(stage, read, lookup):
        actual = read()
        assert actual.source is source_ref()
        if stage == "before_copy":
            assert actual.target is actual.recipient is None
        else:
            assert actual.recipient is receiver_ref()
            assert actual.target is lookup(source_ref())
            observed.append(ref(actual.target))
            saved.append(read)
            if fail:
                raise RuntimeError("recipient observer failed")

    with pytest.raises(RuntimeError, match="recipient observer failed") if fail else nullcontext():
        with observe_model_copies(source, observer, ModelCopyLimits(1, 2, 8)):
            window = begin_model_copy(source)
            assert window is not None
            memo: dict[int, Any] = {}
            target = deepcopy(source, memo)
            try:
                window.copied(target, memo, recipient=receiver)
            finally:
                window.end()
    assert len(saved) == len(observed) == 1
    with pytest.raises(ValueError, match="callback"):
        saved[0]()
    del receiver, target, memo, window, source
    gc.collect()
    assert receiver_ref() is None and observed[0]() is None


def test_should_keep_existing_model_copy_callers_compatible_without_a_recipient():
    source = Model()
    stages = []

    def observer(stage, read, lookup):
        actual = read()
        assert actual.recipient is None
        assert actual.source is source
        if stage == "copied":
            assert actual.target is lookup(source)
        stages.append(stage)

    with observe_model_copies(source, observer, ModelCopyLimits(1, 2, 8)):
        window = begin_model_copy(source)
        assert window is not None
        memo: dict[int, Any] = {}
        target = deepcopy(source, memo)
        try:
            window.copied(target, memo)
        finally:
            window.end()
    assert stages == ["before_copy", "copied"]
