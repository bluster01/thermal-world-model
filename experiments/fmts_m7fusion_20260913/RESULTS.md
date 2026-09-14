# FMTS-M7R1 results — audited 2026-09-14

Return `15757e8`, execution `5e2d34e`: three seeds completed 24k updates each.
Linux saved 6 prediction/response and 9 health replays, no failures. Local source,
artifact and saved-array checks passed; no local real-record inference replay.
[Full audit](../../docs/fmts2026/audits/m7_gnr1_20260914/REVIEW_ZH.md).

M7 H18 MAE **0.509880 ± 0.012142°C** (sample SD, three seeds). This is 22.1% lower
than old token, but 15.1% higher than GRU. Paired wins versus GRU: **0/3**.
Not MAE-eligible; retain `fusion_gru_norew` under the frozen rule.

Retained parent: GRU H18 0.443139°C, old token H18 0.654311°C, persistence
0.638755°C. Real histories confirm old token seed0/2 active-state saturation of
100%; M7 saturation is 0–3.91%, with nontrivial history sensitivity. This is a
bundled encoder/head repair, not evidence isolating either component.

M7 mean H18 valve deltas are −0.131365/−0.101275°C; all 64 endpoint windows per
seed/valve are negative, none excluded. Model responses are not plant gain truth.
Post-v0.2 exploratory supplement, test locked, no further search authorized.
Related tests rerun: 24 passed in 31.84 s. Parent results/paper snapshot retained.
