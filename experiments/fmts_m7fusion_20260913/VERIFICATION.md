# FMTS-M7R1 engineering verification — 2026-09-13

## Implemented and checked

- M7 RevIN/per-variable TCN/VariableAttention modules reused, fixed patch math
  checked against the legacy module. No dependency on global patch dimensions.
- All three seeds preserve the GRU template's non-observer state_dict tensors
  exactly before training, including transition/closure/boundary/observation.
- Independent 11-row state head, exact zero-head anchor, bounded correction,
  no state-query shared scalar head; absolute-history levels reach context.
- Gradients reach all nine historical extensions after activating the head;
  inherited rich-input/no-future data tests remain in the targeted suite.
- Synthetic four-arm parent + one M7 seed with two updates: checkpoints reload,
  prediction/response replay (2 sets), old/new observer health replay (3 sets),
  paired comparison, original indices bytes, hash-tamper rejection, no overwrite.
- Local M7 suite: **7 passed** (12.08 s). This includes the smoke above.
- Combined M7/GNR1/rich-pipeline/mainsteam suites: **24 passed** (33.34 s).
- Full `tests/final_wm`: **226 passed, 1 failed** (197.09 s).

## Existing full-suite failure, not concealed

`tests/final_wm/test_jepa.py::test_registry_closes_jepa_batches_after_linux_audit`
expects global `linux_authorized_gate=final_world_model_pipeline`, while the
existing registry says `thermal_world_model_tenth_prototype`. A standalone rerun
fails identically. `git diff HEAD --exit-code -- tests/final_wm/test_jepa.py
configs/phase3_5/experiment_registry.json` is empty/success: neither file was
changed by this work. Do not alter the historical global queue to make this test
green. New FMTS jobs have their own explicit versioned state files and runbooks.

All parent v0.2 source SHA-256 values still match the returned identity. The new
package adds files; it does not edit frozen parent model/train/evaluation source.
The executable M7 spec equals its frozen JSON. Imported legacy modules and local
runner/audit sources are separately fingerprinted in each execution identity.

## Scientific limits

No real M7/GNR1 training or locked-test evaluation was performed locally. These
checks do not show restored prediction accuracy, healthy long-run optimization,
correct valve response, convergence or plant calibration. Linux must return the
six registered real fits and both explicit replay audits. Read-only original
observer health diagnoses use real validation histories on Linux, not the local
synthetic histories from the initial audit.
