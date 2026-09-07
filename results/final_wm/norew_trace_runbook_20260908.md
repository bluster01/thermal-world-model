# Linux: existing-checkpoint trajectory export only

User explicitly selected: only reevaluate existing models; do not train new models. This batch exports missing response time series for publication plots. It does not train the missing no-closure/norew comparator, run leakage training, tune parameters, change sample support, or access reserved test values.

Prerequisite: completed paired return at /home/bluster/final_wm_norew_paired_replay_20260907 (a34481f return). Reuse original package and original Linux environment. The existing replay worktree must be clean before switching. Execute sequentially, stop on any error; no retry or new output suffix after failure without review.

```bash
set -euo pipefail
cd /home/bluster/thermal-world-model-norew-replay
git fetch origin codex/norew-inference-replay
git switch --detach origin/codex/norew-inference-replay
test ! -e /home/bluster/final_wm_norew_traces_20260908
python -u experiments/final_wm/export_norew_response_traces.py \
  --package /home/bluster/final_wm_v07_full_reissue_v1 \
  --paired /home/bluster/final_wm_norew_paired_replay_20260907 \
  --out /home/bluster/final_wm_norew_traces_20260908
```

Frozen scope: Side A only, 3 existing arms x 3 seeds x 2 valves x H18/H60 = 36 cells, each on its original 64 windows with history 96, valve +0.05 clamped at 1, constant oracle boundary. No window selection/replacement. Preserve all five temperature channels, full base/step trajectories, raw per-window differences, actions, frozen boundaries, indices, day ids, both support masks. H18 and H60 sample sets are distinct as before; never splice them into a single trajectory.

Every cell must reproduce its earlier terminal response point and CI within 1e-6 numerical tolerance and support masks exactly before acceptance. This is a repeat-output check, not a new scientific acceptance criterion. Store 36 trace .pt files with SHA-256, nine run JSONs, protocol.json, isolation.json and summary.json. Return the whole output directory, stdout/stderr and git rev-parse HEAD. Record failure output unchanged. No training or remote execution was performed by the local preparation task.

Pointwise day-bootstrap bands are descriptive curve uncertainty, not simultaneous confidence bands and not measured plant response truth. Every curve must retain its support limitation. No-closure/norew remains missing by the user's explicit no-new-training decision. Full R1 verdict remains unassessed for this export batch.
