# RM3-A capacity-matched and local/terminal Pareto design

## Objective

RM3 showed a repeatable terminal-MAE advantage for P5 over P3/P4, but P5 has about 40% more state elements and worse local-drop MAE. RM3-A is a validation-only identification experiment for that ambiguity. It is not a new architecture search and does not open test or MS4.

## Architecture decision

Use bidirectional capacity matching instead of only enlarging the baselines:

| New candidate | Base architecture | State elements | Question |
|---|---|---:|---|
| A0 | P3 paired-free, d=77 | 120,928 | Does a large free Gate-C model close the P5 terminal gap? |
| A1 | P4 scheduled A1, d=77 | 121,103 | Does a large A1 Gate-C model close the P5 terminal gap? |
| A2 | P5 joint-latent, d=52 | 83,649 | Does small P5 retain an advantage over original P3/P4? |
| A3 | P5 d=64, local/terminal 0.35/0.35 | 122,301 | Can local error improve without losing terminal gain? |
| A4 | P5 d=64, local/terminal 0.50/0.30 | 122,301 | Where is the stronger local-anchor Pareto boundary? |

Original P3/P4/P5 runs are immutable audited references and are not rerun. Five new candidates over two folds and three seeds produce 30 new runs.

## Loss and selection

Every candidate remains full-multitask. Balanced loss assigns 0.25 each to valve, Tin, local drop and terminal. A3 and A4 vary only these component weights. Checkpoint selection uses the declared weighted validation loss on selector anchors; terminal/local reporting uses disjoint reporting anchors.

There is no single composite champion. Capacity-matched terminal and local differences are paired by fold/seed. The Pareto table reports terminal MAE, local MAE and selector score together. An architecture direction is retained only if its terminal advantage survives both directions of capacity matching. A loss profile is retained only as a trade-off point; no arbitrary scalar converts local degradation into terminal improvement.

## Operational and claim boundary

Hermes executes only the frozen 30 new runs. It must return all five per-run artifacts including checkpoint and NPZ; `.gitignore` explicitly permits both. Existing run directories are refused, not resumed. Worktree, source cache, matrix, code and parent-audit hashes are pinned.

RM3-A may distinguish capacity, architecture and loss-weight effects under observed-policy validation. It cannot establish unique plant gain, measured spray-flow physics, arbitrary valve intervention, independent test performance or closed-loop readiness.

## Alternatives rejected

- Enlarge only P3/P4: rejects because it cannot test whether P5 works at matched small capacity.
- Add more architectures or response operators: rejects because RM3-A addresses a specific confound, not method discovery.
- Rank with one weighted terminal/local score: rejects because the exchange rate is not scientifically justified.
