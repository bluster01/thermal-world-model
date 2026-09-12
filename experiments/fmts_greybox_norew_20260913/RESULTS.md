# Results — fmts greybox norew

Registration: [FMTS-GNR1](../../docs/fmts2026/PREREG_GREYBOX_NOREW_20260913.md).
Seeds 0/1/2; parent return `399a60ce7190`. **Released for Linux by user push request; formal result pending.**

## Checklist before trusting these numbers
- [x] Parent 12-run saved-array audit completed; matched result files available.
- [x] Only `aW1/aW2` differ at initialization and are frozen near zero.
- [x] Seven local tests passed, including one synthetic new-arm smoke and two inference-array replays.
- [x] Test disabled; parent index bytes and input hashes enforced; oracle caveats retained.
- [ ] Three formal Linux seeds returned and audited.

## Numbers
| run | method | metric | mean | std | seeds | notes |
|-----|--------|--------|------|-----|-------|-------|
| parent v0.2 | greybox_steady_none | H18 MAE °C | 0.969283 | 0.008483 | 0,1,2 | rewetting on |
| GNR1 pending | greybox_steady_none_norew | H18 MAE °C | pending | pending | 0,1,2 | not run |

## Observations
- Parent greybox endpoint responses are +0.00448/+0.03305°C. They do not validate the intended cooling response.
- Synthetic smoke proves implementation/replay only; its numbers are not scientific evidence.

## Decision
- Prepare Linux execution. Do not assume no-rewet succeeds; do not replace the parent results before audit.
