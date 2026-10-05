"""Complete current input ports for the fixed findings publication use case.

Outer adapters provide unchanged current outcome/cost and matrix bundle readers
and a complete development-ledger reader. This module defines their small typed
interface only; IO, validators, models, source selection and scoring are external.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


JsonBody = dict[str, Any]
BundleParts = tuple[JsonBody, JsonBody, JsonBody]


@dataclass(frozen=True)
class FindingsReaders:
    outcome_costs: Callable[[], BundleParts]
    matrix: Callable[[], BundleParts]
    development: Callable[[], JsonBody]
