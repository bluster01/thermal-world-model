# Local verification — 2026-09-19

- Python / PyTorch 2.5.1+cu121, CPU one thread, NumPy1.26.4.
- 27 focused pytest cases pass: shape/backward/checkpoint for every trained family; left/right time alignment; future-temperature isolation; exact block feedback construction; R4 wrapper/zero-adapter identity; causal versus unconstrained action prefix; original R4 reachable/zero-response paths; fixed nominal plan across blocks; metrics arithmetic; full110-shape paired probe suite; dose/symmetry/interaction aggregation; float64 gradient versus finite difference.
- Tiny real-data smoke completes all11 model rows: persistence, nine fits, anchored composition. Each fit sees16 training windows for one epoch/two updates; selector8, reporting4, response2 origins/four scenarios. Outputs and plots generated, no failures. These are implementation checks, not model-performance results.
- Full110-scenario / block+native R4 behavior exercised independently on a small deterministic fixture. Pairing and float64 remove float32-scale spurious pre-onset/upstream effects. Response measurements do not require re-training.
- Pack:6,111 training windows,128 selector,256 reporting; SHA256 recorded in adjacent data JSON. Includes five temperatures and eight control/boundary variables, with all required endpoint controls for correct feedback.
- Default6-epoch seed11 experiment has NOT run locally. Linux owns that execution. No new test/extension evaluation, planner optimization or field control has been launched.

Entry points and interpretation are in README.md. This experiment is isolated from the existing mainline and historical side-experiment result directories.
