# Phase 3.5-MS3-D implementation plan

## Step 1: Freeze the diagnostic matrix

- Output: `configs/phase3_5/ms3d_asymmetry_diagnosis.json` with source, split, side mapping, event thresholds, endpoints, bootstrap settings and no-test contract.
- Test: reject changed source SHA, non-validation split, incomplete A/B mapping and invalid thresholds.

## Step 2: Implement event extraction and response metrics

- Output: `src/phase35/ms3d.py` with pure event detection, stability classification, signed response and daily aggregation functions.
- Test: synthetic A/B arrays cover held/non-held SP, gaps, boundary events, expected sign, near-zero division and same-side separation.

## Step 3: Implement the local runner

- Output: `experiments/phase3_5/ms3d_asymmetry_diagnosis.py` loading the existing MS3 caches and checkpoint replay artifact, restricted to validation.
- Artifacts: manifest, event JSONL, event CSV and `summary_validation.json` under `results/phase3_5/ms3d_asymmetry_diagnosis/`.
- Test: CLI fixture verifies artifact schemas, source pins and `test_accessed=false`.

## Step 4: Execute and audit

- Output: `docs/PHASE35_MS3D_SUPERVISOR_AUDIT_2026-08-11.md` mapping each claim to event/day support and uncertainty.
- Test: independently recompute headline A/B effects from event CSV; inspect rejection funnel, opening/closing support and paired dates.

## Step 5: Migrate project state

- Output: TODO, README, context snapshot, methods traceability and experiment registry agree on the diagnosis and next authorized action.
- Test: status CLI, focused tests, complete `tests/phase35` regression, compile and `git diff --check` all pass before push.
