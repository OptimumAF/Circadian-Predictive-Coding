"""Original inbox shape/dtype and finite serialization allowances.

No payload access, source attestation, admission, model operation or IO.
"""

from dataclasses import dataclass

NUMERIC_DTYPES = ("|i1", "|u1") + tuple(
    endian + kind
    for endian in ("<", ">")
    for kind in ("i2", "i4", "i8", "u2", "u4", "u8", "f2", "f4", "f8")
)
INTEGER_FIELDS = {
    "input_dim",
    "max_batch_rows",
    "max_records",
    "max_identifier_bytes",
    "max_candidate_ids",
}
DTYPE_FIELDS = {"feature_dtypes", "target_dtypes"}


@dataclass(frozen=True)
class InboxCodecPolicy:
    input_dim: int
    max_batch_rows: int
    max_records: int
    max_identifier_bytes: int
    max_candidate_ids: int
    feature_dtypes: tuple[str, ...] = NUMERIC_DTYPES
    target_dtypes: tuple[str, ...] = NUMERIC_DTYPES

    def __post_init__(self) -> None:
        if vars(self).keys() != INTEGER_FIELDS | DTYPE_FIELDS:
            raise ValueError("inbox codec policy fields differ from complete schema")
        for name in INTEGER_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError("inbox allowances require bounded positive exact integers")
        for name in DTYPE_FIELDS:
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or len(values) > len(NUMERIC_DTYPES)
                or any(type(value) is not str or value not in NUMERIC_DTYPES for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("original dtype policy requires supported unique immutable names")
