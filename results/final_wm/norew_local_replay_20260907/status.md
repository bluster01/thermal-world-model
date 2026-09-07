# Local checkpoint replay status

Status: STOPPED_NUMERICAL_REPLAY_MISMATCH. No automatic retry or tolerance amendment.

User authorized inference-only replay, explicitly excluding retraining. Three arms × three seeds were planned; only physics_only seed 0 prediction has run. No response probe, optimizer, leakage training, Linux dispatch, commit or push occurred.

Input hashes and 30 saved T1 artifact hashes match the original manifest. Current src/final_wm and matrix_spec.py match the original manifest commit. A synthetic fixture verified prefix isolation and test sampling rejection. The source canonical has 707709 rows; only the numeric prefix of 636937 train/validation rows was decoded, with original row indices preserved. Test partition labels were read as metadata; test numeric values were not decoded.

Before inference protocol.json froze per-element float32 atol=0.002, rtol=0.0001 for MAE/NLL/CRPS. This is a newly selected conservative numerical replay gate, NOT the original scientific R1 threshold and NOT evidence that the original experiment failed.

physics_only seed 0: saved aggregate MAE 1.749002695°C, replay 1.748900056°C, signed mean difference -0.000102582°C. Per-element max MAE difference 0.007672191°C; 369/4608 elements exceed the recorded tolerance. NLL and CRPS also exceed tolerance. UTC day ids match exactly. See stopped_diagnostics.json for full summaries.

The aggregate difference is small relative to the prior +0.283368°C configuration contrast, but this alone does not establish elementwise replay identity or explain the discrepancy. Cross-platform CUDA/kernel/numerical behavior is a candidate explanation, not a verified cause. No favorable scientific verdict is inferred.

Next decision: design an explicit numerical-equivalence diagnostic (original environment information and controlled execution settings), preserve this failed attempt, and record any revised criterion with an independent justification before another run. Do not raise the threshold merely above the observed maximum. Missing no-closure/norew checkpoints remain unresolved and no training is initiated.
