# Norew paired evidence: Linux inference-only replay

Scope: physics_only, closure_cons, closure_cons_norew; seeds 0/1/2. Reuse original checkpoints. No model training, leakage-probe training, canonical rebuild, parameter search, reserved-test numeric access, threshold changes or automatic retries. Full R1 is NOT adjudicated by this batch.

Use the original v0.7 Python/CUDA environment. The script fixes matmul precision to highest and records the actual CUDA/cuDNN/TF32 flags. It checks original input/checkpoint/metric hashes, reads only train/validation numeric rows with original indices, compares saved per-element prediction metrics, then performs frozen dual-valve H18/H60 response probes with complete support masks. Windows numerical mismatch remains recorded; its tolerance has not been loosened.

The package directory must contain inputs/ and sideA/ exactly as in the original execution package. Output must not already exist. The branch is codex/norew-inference-replay; checkout may be done in a separate worktree to preserve current work.

```bash
git fetch origin codex/norew-inference-replay
git worktree add --detach /home/bluster/thermal-world-model-norew-replay origin/codex/norew-inference-replay
cd /home/bluster/thermal-world-model-norew-replay
python -u experiments/final_wm/replay_norew_pairs.py \
  --package /home/bluster/final_wm_v07_full_reissue_v1 \
  --out /home/bluster/final_wm_norew_paired_replay_20260907
```

Run the lines sequentially and stop on any error; do not continue after a failed fetch/worktree/cd. Use the same activated environment as the original batch. Do not automatically install or change dependencies.

Return the entire new output directory (protocol.json, isolation.json, nine run JSONs, nine prediction tensors, summary.json) plus stdout/stderr, git rev-parse HEAD and environment details. An incomplete run returns its existing output and error, without retry. Unsupported interventions retain their masks and diagnostic outputs; they cannot produce a supported-response conclusion. Missing leakage is deliberate, not zero leakage.

No-closure/norew is not a registered checkpoint in this original package. This batch does not train that missing arm and cannot quantify learning recovery on the norew path without it.

## User-relayed diagnostic evidence (not independently verified files)

User reports Linux physics_only seed0 replay has elementwise max difference 0 and identical day ids; high precision-mode experiment has a smaller discrepancy than Windows. These reports support checking in the source environment but do not identify the precise Windows divergence kernel. Identical day ids alone do not prove identical sample indices; the script now explicitly records reconstructed canonical indices.

Windows maximum MAE discrepancy: sample index 180 (zero-based), H15, first future index 549643, target index 549657, history start 549547. Target timestamp 2026-02-25T16:57:50Z. Saved 1.773406982421875, replay 1.7810791730880737. Windows script used highest; recorded torch 2.5.1+cu121, GPU RTX 4070 Laptop. Later environment inspection reports CUDA 12.1, cuDNN 90100, CUDA matmul TF32 false and cuDNN TF32 true; the latter flags were not captured at the original Windows run, so distinguish reconstruction from contemporaneous logging.
