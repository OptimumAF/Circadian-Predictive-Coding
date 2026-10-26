"""Real profile refusal must bypass ordinary per-arm failure handling."""

import ast
from pathlib import Path
import sys
from time import monotonic
from typing import Any


def test_should_stop_before_another_arm_when_actual_profile_budget_raises():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/r28-removal-confirmation-20261017/native_case.py"
    )
    tree = ast.parse(path.read_text())
    cap_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeCapExceeded"
    )
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    functions = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.FunctionDef) and n.name in ("refuse", "boundary", "profile")
    ]
    module = ast.Module(body=[cap_class, *functions], type_ignores=[])
    counts = {"train": dict(attempted_calls=0, returned_calls=0, attempted_rows=0, returned_rows=0)}
    violations: list[str] = []
    namespace: dict[str, Any] = {
        "monotonic": monotonic,
        "started": monotonic(),
        "caps": {"train": (0, 0)},
        "counts": counts,
        "violations": violations,
    }
    exec(compile(module, str(path), "exec"), namespace)
    fake: dict[str, Any] = {}
    exec(
        compile(
            "def train_batch(features):\n    return 1\n",
            "/local/src/adapters/coupled_learners.py",
            "exec",
        ),
        fake,
    )
    reached: list[int | str] = []
    caught = None
    try:
        sys.setprofile(namespace["profile"])
        for arm in range(2):
            reached.append(arm)
            try:
                fake["train_batch"]([0])
            except Exception:
                reached.append("ordinary handler would continue")
    except BaseException as error:
        caught = error
        assert sys.getprofile() is None
    finally:
        sys.setprofile(None)
    assert type(caught) is namespace["NativeCapExceeded"]
    assert reached == [0] and violations == ["train pre-entry budget"]
    assert counts["train"]["attempted_calls"] == 0
    # The actual worker must preserve evidence at its outer boundary.
    outer = next(n for n in main.body if isinstance(n, ast.Try) and n.handlers)
    assert any(
        isinstance(h.type, ast.Name) and h.type.id == "BaseException" for h in outer.handlers
    )
