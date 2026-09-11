# FMTS main-steam comparison experiment

Current protocol: **0.2**. Blackbox and both hybrid observers are registered to
use 23 past variables (including nine boundary extensions). The revised input,
training, raw-array exports and read-only replay audit are implemented.
See [Linux execution](RUN_LINUX.md) and [verification](VERIFICATION.md).

This package implements the prospective development and evaluation protocol in
`docs/fmts2026/PREREG_THREE_MODEL_MAIN_STEAM_20260911.md`.

The package does **not** revise matrix v0.7 or its verdicts.  It introduces one
experimental initial-state observer while freezing the physical transition,
action path, neural closure and boundary model.  The locked chronological test
split remains disabled until a separate author decision.

## Two-stage decision

1. **Observer development on validation only.** Compare the registered GRU
   hybrid and token-cross-attention hybrid.  The token candidate advances only
   under the rule in `spec.py`; response-curve appearance is not a selector.
2. **Final three-family comparison.** Compare the saved action-conditioned
   iTransformer, the true steady/no-closure grey box, and the hybrid selected in
   stage 1 on the identical frozen manifests.  GRU remains a disclosed ablation.

Linux is the intended full-training executor.  Local work is limited to code,
contract tests, synthetic smoke tests and immutable-spec generation.

## Files

- `spec.py`: executable frozen choices and selection rule.
- `prepare.py`: records hashes and serialized specs without reading canonical data.
- `smoke.py`: synthetic, past-only shape/identity checks.
- `smoke_pipeline.py`: complete synthetic four-arm training and replay.
- `data.py`: common quality-filtered windows and train-only normalization.
- `models.py`: rich-history blackbox and fusion adapters.
- `run.py`: frozen four-arm/three-seed training and raw-array export.
- `audit.py`: hash, pairing, checkpoint and optional numerical replay checks.
- `experiment_state.json`: machine-readable lifecycle state.
- `CHANGELOG.md`: chronological human audit trail.
- `RUN_LINUX.md`: fail-closed executor handoff; generated after local verification.
