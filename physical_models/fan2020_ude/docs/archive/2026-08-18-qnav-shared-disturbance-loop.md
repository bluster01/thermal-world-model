# Q32-S Shared-Disturbance Deviation Loop

## Question

Q32-R shows that a state-dependent residual can cancel or reverse the explicit action response, but its 16 sampled points were all dry and it was open loop. Q32-S asks one narrower solution question: if the baseline and counterfactual worlds share the same learned disturbance trajectory, does incremental closed-loop authority improve in both genuinely wet and dry states?

## Frozen design

- Reuse the frozen Q32 `h_now` checkpoint for each fold; no training.
- Use development rows only. Select four evenly spaced wet and four dry points in each evaluation fold without stability filtering.
- Start baseline and controlled worlds from the same warmed state.
- Track the paired deviation `T_controlled - T_baseline = +0.5°C`; this isolates incremental action authority from nominal trajectory drift and is not presented as a full plant setpoint test.
- Apply the existing PI schedule, valve rate limit, and training-only state-specific valve-to-W coupling.
- Compare only three modes:
  - `physical`: residual disabled in both worlds;
  - `live`: each world evaluates the residual on its own evolving state;
  - `shared`: the baseline records its residual outputs and the controlled world replays exactly that sequence.
- Report point-level tracking error, tail variation, saturation, reversals, valve range and baseline drift. Threshold counts are diagnostics, not PASS/FAIL gates.

## Interpretation boundary

Improvement of `shared` over `live` would support a practical disturbance/action separation inside this simulator. It would not establish a causal plant model, validate arbitrary interventions, or replace real wet/dry event evidence. Failure would reject this particular replay solution without undoing the Q32-R mechanism result.

## Linux execution boundary

Linux runs the one frozen inference command printed by `--dry-run` and returns raw artifacts plus the log. It must not train, retry, alter the matrix, repair code, select examples, make a figure, or update conclusions. Any error is returned unchanged for supervisor repair.
