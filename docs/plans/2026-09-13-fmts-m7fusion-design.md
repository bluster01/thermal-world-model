# FMTS M7-encoder fusion and no-rewet reference design

**Goal:** Improve the new fusion observer without changing the physical action
path, and test whether refitting the no-rewet greybox changes its valve response.
User authorized implementation on 2026-09-13 after the token diagnostic and
historical M7 encoder review. No paper verdict or locked-test access is authorized.

## Alternatives and decision

1. Shared-token-head-only repair: smallest change, but does not reuse the M7
   encoder requested by the author.
2. **Selected:** M7 historical encoder family with independent anchor-relative
   state readout; one bundled candidate, no claim to isolate encoder/head effects.
3. Direct temperature residual head: would change the physical output/action
   contract and obscure attribution; out of scope.

The candidate is called `fusion_m7var_norew`, not an iTransformer reproduction or
the unchanged `fusion_token_xattn_norew`. Reuse M7's actual RevIN, per-variable TCN
and VariableAttention building blocks. A local fixed patch implementation avoids
the historical module's global-config patch dimensions. Dimensions: 23 histories,
96 steps, patch16/stride8, d64, two TCN blocks, four-head variable attention.
Flatten all 23 variable representations. Preserve physical-normalized historical
mean/std alongside the RevIN shape representation; this prevents loss of absolute
levels relevant to physical initial-state estimation. Context MLP 1518→256→128
with GELU/dropout0.1 and non-affine output LayerNorm. The existing independent
11-state linear heads, zero initialization, pressure features and 0.1×state-scale
tanh bounds are retained from the GRU observer. Only four slow states are active.
This is an encoder adaptation, not reuse of old temperature heads or weights.

Keep physical transition, conservative no-rewet closure, data/windows/optimizer/
loss/checkpoint selector identical to the parent GRU. Do not mutate any files in
the parent source fingerprint set. A separately identified observer subclass uses
the existing vector-context posterior interface; checkpoint metadata explicitly
records the effective M7 implementation rather than calling it a GRU model.

## Scope and outputs

- FMTS-M7R1: one new fusion arm × seeds 0/1/2, full parent 24000-update cap.
- FMTS-GNR1: the already registered no-rewet grey arm × seeds 0/1/2, unchanged.
- Retain 12 parent results, same indices and input hashes. No blackbox/GRU/token
  reruns, search, sign constraints, new data splits or locked test.
- Before M7 training, save read-only health diagnostics on the six original
  observer checkpoints using their actual fixed validation histories.
- During M7 training, log observer gradient norms and periodic health summaries;
  export full best-checkpoint health arrays. Saturation is diagnostic, not a
  post-hoc checkpoint selector or a claim of plant response fidelity.
- Compare M7 predictions/response to all four parent arms. GNR1 keeps its own
  single-change comparison. Full paired plots and the paper are updated only
  after both real result packages have returned and been audited.

No-rewet is a testable response-recovery intervention, not a promise of correct
global transient signs or a calibrated plant response. Failures remain visible.
