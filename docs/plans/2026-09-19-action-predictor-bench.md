# Action predictor quick benchmark implementation plan

**Goal:** Compare predictive accuracy and action behavior across black-box and structured predictors, with a small portable dataset and one Linux command.

**Architecture:** One isolated experiment directory; shared data, training loop, forecast metrics and response probes. Reuse the original R4 implementation as two local source snapshots. No framework, service, dependency on untracked experiments, or automatic follow-up training.

**Tech stack:** Python, NumPy, PyTorch, matplotlib; pytest for focused checks.

1. Package 10% of chronological training-window candidates, train-only normalization, and fixed selector/evaluation banks. Keep test/extension untouched. Include the small compressed pack so Linux does not need Windows source paths.
2. Implement persistence; direct no-action/action black boxes; recurrent GRU and latent SSM; compact action-interleaved Transformer; R4, R4+MLP and direct-reference R4. Add a no-fit anchored combination of trained direct and R4 predictors.
3. Train all learned models with identical samples, target weights, epochs and optimizer settings; record cost and parameters. H64 history, H32 supervision, H128/H512 evaluation. Default seed11, configurable seeds11/23/37.
4. Evaluate common block32 rollout, native continuous extrapolation where supported, held-boundary stress, per-step/per-channel errors, incremental errors, and early/late validation groups. No true future temperature enters rollout histories.
5. Probe signed multi-dose steps, pulse/release, ramp, sine, smooth varying input, dual valves, early/mid/late onset, block-edge onsets and pre-window action effects. Record response curves, actual dosage, pre-onset leakage, timing, gains, recovery, reachability and boundedness; no maximize-response score.
6. Test time alignment, target isolation, exact rollout-history construction, R4 snapshot parity, anchored identity, prefix behavior and a tiny all-model train/save/reload/evaluate smoke.
7. Write concise Linux instructions, run local checks, commit only this experiment/plan and push codex/action-predictor-bench. Full fits run on Linux by the user.

Status: ready_for_linux after local verification. Nine learned models (including the equal-parameter attention fusion control), persistence and anchored combination. 27 focused tests pass; all11 entries complete the tiny real-data train/reload/evaluate smoke. User explicitly authorized implementation and git push. Experimental candidates remain exploratory; no formal causal gate or paper workflow is introduced.
