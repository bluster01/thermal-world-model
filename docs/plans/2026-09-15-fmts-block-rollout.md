# FMTS 180-second block-feedback implementation plan

**Goal:** export measured history and compare 1,200-second main-steam-temperature trajectories using existing checkpoints, without training.

**Architecture:** a small inference wrapper around the unchanged FMTS v0.2 / GNR1 adapters; common validation-origin eligibility; complete 18-step target feedback; independently checked saved artifacts and numerical replay.

**Tech stack:** Python, PyTorch, NumPy, pytest; existing canonical record and IAPWS surrogate.

1. Register the user-confirmed block18 protocol, source identities, information policy and exclusions before execution.
2. Test block advancement, target non-leakage, remaining-context updates, tail handling and common eligibility.
3. Implement fixed-checkpoint inference, history export, original-H18 replay gate and saved-array audit. Do not modify frozen model files.
4. Verify locally with engineering fixtures and checkpoint identity checks only. Exact-record inference belongs to Linux.
5. Save Linux instructions and pending status. Audit returned evidence before revising figures or paper.

Historical exp_048/054/058 use first-step feedback with stride 1. They establish lineage, not this run's protocol: the author explicitly chose **stride 18, complete predicted blocks** on 2026-09-15. A fixed H18 output head does not prevent recursive forecasting.

Existing dirty manuscript, figures, results and status changes are preserved. No new model search, no locked-test or extension scoring, no changed response probe, no public upload of raw history/trajectories, no automatic paper verdict.
