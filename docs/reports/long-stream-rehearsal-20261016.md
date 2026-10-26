# Long-stream rehearsal: representation survives, exposure stays sparse

The fixed64-update A16→B32→A16 run completed for BP, PC, neutral CPC and active
CPC. Each receives64eight-row wake batches and42single-row replay calls. The
existing16row/384array-byte recent/reservoir policy and rotation stay unchanged.

All eleven recorded B events retain at least2first-A rows. During return, all five
events retain at least4B rows. At final wake64 the exact buffer has2first-A,
4B and10return-A rows. No future or final rows enter replay selection. Four plans
are identical and PC-neutral full reports match.

**Retaining an example does not mean replaying it.** Seven B events expose no
first-A row; total first-A exposure during B is4rows/3distinct IDs. Three return
events expose no B row; total B exposure is2rows/2distinct IDs. These gaps remain
part of the result, alongside earlier starvation findings.

| Method | B gain ↑ | A loss during B ↓ | A return gain ↑ |
| --- | ---: | ---: | ---: |
| BP | .349937 | .139676 | .223477 |
| PC/neutral CPC | .157041 | .084644 | .119478 |
| Active CPC | .131408 | .062758 | .078547 |

All methods learn B, forget some A and recover when A returns. BP starts phase B
with a better A loss and ends with lower B48/A64 absolute BCE. Active CPC's smaller
forgetting does not establish an overall advantage. All absolute levels and
accuracy fields are retained; this is one exploratory stream, not a causal replay
comparison or generalization result.

Cost:424calls/2216row exposures/32scores/1908coupled+212BP evaluations,
1.2916297s/RSS42856448B, one joined sampler. Equal exposure/caps are not equal
computation. The result qualifies R2.3's original bounded implementation criterion;
universal retention and every-event old exposure are not claimed.

Independent SOURCE/GO/ACTUAL accepted60qualified focused cases and five-path
static gates. The broader63-case invocation hit the no-probe guard after those
60cases; that failure and the three excluded historical probe cases are disclosed.
Full evidence, commands, raw levels and retention/exposure CSVs:
`artifacts/runs/r23-long-stream-20261016/RESULTS.md`.
