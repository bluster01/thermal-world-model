# FMTS-CORE1 local verification — 2026-09-14

## Completed verification

Working tree: `C:/Users/14020/Documents/ChatGPT/thermal-world-model-fmts-release`,
HEAD `15757e8`. Existing frozen GNR/GRU scientific source was not edited.

1. Initial test collection failed because the new package did not yet exist
   (expected red test).
2. CORE1 contract tests: **10 passed in 25.82 s**.
3. After provenance/index/identity tightening, related regression:

```powershell
python -B -m pytest tests/final_wm/test_fmts_core.py tests/final_wm/test_fmts_m7fusion.py tests/final_wm/test_fmts_greybox_norew.py tests/final_wm/test_fmts_rich_pipeline.py tests/final_wm/test_fmts_mainsteam_experiment.py -q -p no:cacheprovider
```

Result: **34 passed in 58.68 s**.

Tests cover exact synthetic C11 identity, unchanged/frozen physical weights,
four observer/closure call contracts, invalid variant rejection, a small-response
replay failure that absolute-temperature allclose would miss, one history-derived
initial state per counterfactual pair, model state traces, metadata-only preflight,
and end-to-end saved-array plus record-backed replay.
Artifact-only audit correctly stays `complete=false`.

The end-to-end fixture uses tiny synthetic parent/GNR models with the existing
two-update setup. CORE1 itself performs **zero** optimizer updates. No plant
training, new real-record inference, locked-test scoring or extension scoring
was run in this task. Tests are engineering evidence, not scientific outcomes.

## Provenance finding and earlier blocked synchronization

Source-matched archived Phase3.5 manifests, actual block-fitting code and nonempty
saved results establish prior use of the March-to-May candidate. See
`analysis/fmts_greybox_core_20260914/EXPOSURE_REVIEW_ZH.md` and
`temporal_candidate_metadata.json` in that directory.

After the 34-pass run, metadata preflight gained the distinct blocker
`HISTORICAL_EXPOSURE_CONFIRMED` plus one test asserting that schema/model/window
freeze cannot convert known exposure into independent-test eligibility.
This is an additional **11th CORE1 test**, not part of the recorded 34 passes.
It changes no CORE1 inference/scientific gate.

The requested copy into the release worktree and final regression were rejected
by the permission system before the shell command started. At that earlier point:

- The latest main-workspace metadata guard and documentation are saved, but have
  **not** been synchronized into release by that command or test-verified.
- No actual candidate metadata preflight invocation has run; the saved metadata
  is an audit input, not a claimed machine execution receipt.
- `ready_for_linux=false` until latest files are synchronized and tested.
- Nothing was committed, pushed, dispatched to Linux or marked scientifically
  complete. No statement that greybox response was recovered is warranted.

The main workspace contains unrelated dirty work. Its existing source, README
and TODO were not overwritten. Proposed release README/TODO additions are saved
as `analysis/fmts_greybox_core_20260914/release_{README,TODO}.staging.md` only;
the actual release README/TODO still hold their prior content.

## Resumed completion after permission update

The user then explicitly requested a Linux-executed evidence supplement before
paper revision, and the environment permissions were updated. Only task-specific
new files were synchronized. README/TODO were checked against their preserved
SHA256 before applying the prepared additions, preserving their prior audit work.

The same five-file regression command passed **35 tests in 43.08 s**, including
all 11 CORE1 contracts. The runner then gained per-cell progress messages and
preservation of finite arrays before a reference-gate failure; scientific logic,
data, thresholds and budgets were unchanged. Final regression passed
**35 tests in 44.00 s**. No source under frozen `src/final_wm`, parent FMTS or
GNR/M7 Python modules was changed.

JSON-only preflight actually ran and saved
`analysis/fmts_greybox_core_20260914/temporal_preflight.json`: historical exposure
confirmed; schema/checkpoint/window declarations incomplete; scoring disabled;
targets_read=false. Original candidate contract is preserved as
`source_contract.snapshot.json`. Shared ai-research provenance lesson was logged.

Package is ready for release through the existing origin/main Linux handoff.
No Linux execution/start/result receipt exists. Its actual science is 15
REAL-record inference cells, zero training, followed by 30 array / 15 trace
replays. This remains distinct from the local synthetic software checks.
Paper numbers, plots and verdict remain unchanged pending return and author audit.
