# Results — fmts greybox norew

Registration: [FMTS-GNR1](../../docs/fmts2026/PREREG_GREYBOX_NOREW_20260913.md).
Seeds 0/1/2; parent return `399a60ce7190`. **Audited 2026-09-14**, return `15757e8`, execution `5e2d34e`.
Linux saved 6 array replays. Local source/artifact and independent saved-array audit passed; no local real-record replay.
[Full audit](../../docs/fmts2026/audits/m7_gnr1_20260914/REVIEW_ZH.md).

## Checklist before trusting these numbers
- [x] Parent 12-run saved-array audit completed; matched result files available.
- [x] Only `aW1/aW2` differ at initialization and are frozen near zero.
- [x] Seven local tests passed, including one synthetic new-arm smoke and two inference-array replays.
- [x] Test disabled; parent index bytes and input hashes enforced; oracle caveats retained.
- [x] Three formal Linux seeds returned and audited; all 24k updates, no failures.

## Numbers
| run | method | metric | mean | std | seeds | notes |
|-----|--------|--------|------|-----|-------|-------|
| parent v0.2 | greybox_steady_none | H18 MAE °C | 0.969283 | 0.008483 | 0,1,2 | rewetting on |
| GNR1 | greybox_steady_none_norew | H18 MAE °C | 0.973432 | 0.008449 | 0,1,2 | no-rewet |

## Observations
- Parent greybox endpoint responses are +0.00448/+0.03305°C. They do not validate the intended cooling response.
- Synthetic smoke proves implementation/replay only; its numbers are not scientific evidence.
- New H18 endpoint responses: −0.000280/−0.002809°C. Seed-level endpoint means and day-block intervals are negative, but amplitude remains weak and uncalibrated.
- Valve1 seed0 has 62 negative and 2 zero endpoints; other seed/valve endpoints all negative. No unsupported windows excluded.
- All paired MAE day-block intervals versus old greybox cross zero: no demonstrated prediction improvement.

## Decision
- Use as a matched no-rewet descriptive reference, retaining the old greybox. Do not claim correct plant gain recovery or continue searching.
- Related tests rerun: 24 passed in 31.84 s; test remains locked. See audit for small-signal floating-point/replay limitations.
