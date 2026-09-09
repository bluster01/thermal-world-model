# Linux private data preparation task

Task: `thermal_world_model_tenth_prepare_data_v1`. Run after the published preflight succeeds, using the repository revision containing this task and its expected identity. This task prepares and verifies data only; it does not start training or an optimizer. Linux uses its own private files. The Git repository carries code, task state and the explicitly public receipts.

Resolve these three paths locally. The output directory must be new, and its parent must already exist. Do not put the private directory inside a public Git checkout.

```bash
export CANONICAL_PATH='/private/local/canonical_sideA_v2.npz'
export RAW_MERGED_PATH='/private/local/all_merged_10s.csv'
export PRIVATE_OUTPUT_DIR='/private/local/tenth_v1'
```

The canonical file must match the SHA recorded in `task_prepare_data_v1.json`. The merged CSV header must match its specified header SHA. The preparer enforces both before substantive raw-prefix parsing, checks the actual old training boundary, and stops at the selected timestamp cutoff. It reads no old validation/test measurements. Do not substitute a similar export or a different time interval when these checks fail.

From the checked-out repository root, run the following in order. Preserve all logs locally: the preparer and saved-array audit may print private paths or data statistics. A failed command stops the chain.

```bash
set -e
python -m pytest analysis/world_model_plant/test_prepare_tenth.py analysis/world_model_plant/test_compare_receipts.py -q
python -m analysis.world_model_plant.prepare_tenth \
  --canonical "$CANONICAL_PATH" \
  --raw "$RAW_MERGED_PATH" \
  --output "$PRIVATE_OUTPUT_DIR" \
  > "${PRIVATE_OUTPUT_DIR}.prepare.local.log" 2>&1
python -m analysis.world_model_plant.verify_tenth \
  --dataset "$PRIVATE_OUTPUT_DIR" \
  --output "$PRIVATE_OUTPUT_DIR/saved_array_audit.json" \
  > "${PRIVATE_OUTPUT_DIR}.verify.local.log" 2>&1
python -m analysis.world_model_plant.compare_receipts \
  --receipt "$PRIVATE_OUTPUT_DIR/public_receipt.json" \
  --expected experiments/world_model_plant/expected_data_identity_v1.json \
  --output "$PRIVATE_OUTPUT_DIR/public_comparison.json"
```

Success requires both the saved-array verifier and comparator to exit zero, with comparison status `DATA_RECEIPT_MATCH`. The comparator loads only public JSON and the current preparer source; it does not load plant arrays. Its equality check covers the dataset identity, canonical/header/raw-prefix/timestamp hashes, exact counts and split ranges, 10-second timing, 64/128 window dimensions, all 39 stable array hashes, command status `UNCONFIRMED`, and the false timestamp/upstream-causality certification flags. A mismatch produces fixed allowlisted mismatch codes without echoing values or paths and exits nonzero. Stop on a mismatch; changing data selection or launching training is outside this task.

The pinned preparer identity permits only the exact source in pure LF or pure CRLF form. These two byte hashes are derived from the same UTF-8 source. Git's newline conversion therefore does not count as a different model/data program. Other source changes and mixed line endings fail. The reference receipt's original raw script hash, current checkout raw script hash and newly observed receipt raw script hash are recorded separately. A Windows script hash need not equal the Linux script hash. The NPZ ZIP container hash is not the cross-machine data comparison criterion; the 39 dtype/shape/C-order array hashes are.

Return files on branch `codex/linux-thermal-data-20260909` under `coordination/thermal_world_model_tenth_prepare_data_v1/`, using an explicit two-file allowlist:

- `public_receipt.json`
- `public_comparison.json`

For comparison failure, return only the sanitized `public_comparison.json`; keep private diagnostic files on Linux. For an earlier preparation or local-verification failure that produced no comparison, record a generic task failure in the coordination state without copying the exception, machine path, data values or private logs. Do not retry by changing the interval or source identity.

Never copy the NPZ, CSV, full manifest, saved-array audit, preparation logs, statistics, actual time ranges, tag values or weights into Git. This task stops after publishing its data receipt. The separate training task must still specify anchors, parameter fitting, boundary forecasts, measurement masks and actuator semantics.
