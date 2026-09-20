# Local verification — 2026-09-19

## 2026-09-20: H128 and nominal-plan screen

- 36 tests passed in42.11s. Added: exact hold initialization including valve bounds, causal nominal-plan prefix, frozen-policy gradients, nominal temperature identity, all110 paired response invariance between held and predicted nominal plans, and nonzero long-tail loss gradients.
- Round3 tiny smoke: one policy fit and six temperature fits complete, twelve checkpoint evaluations complete, zero failures. Each fit sees16 windows and2 updates. Reports/plots generated; these are wiring checks only.
- H128 pack retains all20,371 train origins and every existing train/selector prefix, all evaluation rows and normalization exactly. SHA256 `6f97d5cabc0f0834b05d920baf2337fe32046e63a246b06fe3ce89374ff76c8e`;24,530,842 bytes.
- Formal round3 fits have not run locally. Returned1/3 baseline and round2 findings are in RETURN33_AND_ROUND3.md; the Linux command is at the top of README.md.

## After first return: 1/3 expansion and response ablations

- 32 tests pass (25.07s): original suite plus nested sampling, protected nominal identity, prefix/topology/sign preservation across reference refresh, full110 scenarios in both modes, finite-difference action gradient and frozen response-teacher behavior.
- Default1/3 baseline smoke: all11 rows complete, no failures. Only16 training samples and2 updates per fitted model.
- Round2 smoke: all6 rows complete, no failures; explicitly supplied returned1/10 parents and matching old data to check wiring before1/3 parent fits exist. This is not a1/3 performance result.
- Packed20,371 training windows; all6,111 previous starts retained. Selector, reporting banks/starts and normalization exactly equal to old pack. SHA256 `ddf42e52acfb1bf31465703c5c6f735b4337a8d696c2f13ccd1320e970c09389`.
- Confirmed runner rejects mixing1/3 data with1/10 parent checkpoints before execution.
- Formal1/3 baseline and continuation fits remain for Linux. Initial returned results analyzed from saved forecasts and response arrays; see ROUND2.md and round1_analysis/.

## Initial baseline implementation

- Python / PyTorch 2.5.1+cu121, CPU one thread, NumPy1.26.4.
- 27 focused pytest cases pass: shape/backward/checkpoint for every trained family; left/right time alignment; future-temperature isolation; exact block feedback construction; R4 wrapper/zero-adapter identity; causal versus unconstrained action prefix; original R4 reachable/zero-response paths; fixed nominal plan across blocks; metrics arithmetic; full110-shape paired probe suite; dose/symmetry/interaction aggregation; float64 gradient versus finite difference.
- Tiny real-data smoke completes all11 model rows: persistence, nine fits, anchored composition. Each fit sees16 training windows for one epoch/two updates; selector8, reporting4, response2 origins/four scenarios. Outputs and plots generated, no failures. These are implementation checks, not model-performance results.
- Full110-scenario / block+native R4 behavior exercised independently on a small deterministic fixture. Pairing and float64 remove float32-scale spurious pre-onset/upstream effects. Response measurements do not require re-training.
- Pack:6,111 training windows,128 selector,256 reporting; SHA256 recorded in adjacent data JSON. Includes five temperatures and eight control/boundary variables, with all required endpoint controls for correct feedback.
- Default6-epoch seed11 experiment has NOT run locally. Linux owns that execution. No new test/extension evaluation, planner optimization or field control has been launched.

Entry points and interpretation are in README.md. This experiment is isolated from the existing mainline and historical side-experiment result directories.
