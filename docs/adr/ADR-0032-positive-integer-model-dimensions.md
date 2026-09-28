# ADR-0032: Validate model dimensions before parameter allocation

## Context

NumPy and Torch model constructors checked some dimensions only with
`<= 0`, then passed values to a random tensor allocation. NaN,
infinity, and fractional widths could therefore produce a late backend
`TypeError`; explicit NumPy `hidden_dims` silently truncated finite
fractions through `int(value)`. Boolean widths were also accepted as
integers. These exceptions were inconsistent with the field-named
training input and topology contracts.

## Decision

The five NumPy backprop/ordinary PC/circadian and Torch PC/circadian
head constructors use one pure dimension validator before allocation.
It accepts positive `numbers.Integral` values, including NumPy integer
scalars, and rejects nonfinite floats, fractions, booleans, zero, and
negative values with a field-named `ValueError`. NumPy explicit hidden
layer entries are checked individually; circadian minimum and maximum
widths and Torch output class count are included. Existing ordering and
random draws for valid integer dimensions are unchanged.

## Evidence and limits

The first `input_dim=NaN` backprop fixture failed with NumPy's late
allocation `TypeError`. `tests/test_constructor_dimensions.py` now
passes invalid-value cases for each relevant field, valid `np.int64`
widths, and repeated seeded multilayer construction. The development
log records the full quality gate. This validation concerns model
dimensions; seed interpretation and benchmark-level configuration have
their existing separate boundaries.
