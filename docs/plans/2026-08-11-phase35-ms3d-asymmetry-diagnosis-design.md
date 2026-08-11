# Phase 3.5-MS3-D A/B response asymmetry diagnosis design

> Frozen on: 2026-08-11
> Scope: local, validation-only, no training, no test access

## 1. Question and boundary

MS3-D does not retry MS3 and does not open MS4. It asks why the same joint response model learned a much weaker standardized A-side terminal-temperature response than B. The diagnostic separates three observed links:

```text
secondary-superheat SP -> actual valve feedback -> local spray temperature drop -> terminal temperature
```

The A-labelled SP/valve is paired with the right thermal train; the B-labelled SP/valve is paired with the left thermal train. This wiring is fixed from site knowledge and the MS3 data contract. The analysis is an observational cascade-PID response diagnosis, not `do(valve)`, an open-loop plant gain, or an independent model test.

## 2. Alternatives considered

1. A strict 600 s pre-event steady-state window is physically cleanest, but the validation-only feasibility scan yields only about 1 A and 4 B candidates. It is retained only as a support diagnostic.
2. The selected primary layer is the intersection of a strict 60 s screen and a moderate 180 s screen. These thresholds were already documented before MS3-D in `PHASE35_STEADY_STATE_ANALYSIS_2026-08-09.md`; the current analysis therefore does not tune them against response outcomes.
3. SP-held events that fail the primary steady-state layer remain a dynamic-operation secondary stratum. They are never pooled with the primary layer to improve a result.

## 3. Frozen data and event contract

- Source: the MS3 v1.1 `all_merged_10s.csv` contract, SHA-256 `85a3f92648d5f88a4543f500859b200207fb55a32555900ca88f7c339c4e4da6`.
- Split: chronological validation rows only, about 2026-03-16 to 2026-04-13. Test rows and outcomes are not evaluated.
- SP onset: one-step signed change with `1.0 <= |delta SP| <= 3.0 degC`.
- Held requirement: the new SP remains within `max(0.1 degC, 0.05*|delta SP|)` for the complete 600 s response horizon.
- Continuity: the 180 s pre-window and 600 s response window must contain only 10 s transitions.
- Independence filter: accepted onsets on the same side are at least 600 s apart.
- Operating screen: pre-event mean load at least 250 MW, pressure at least 10 MPa, and terminal temperature in 500--600 degC.
- Primary dual-steady layer: the 60 s ranges satisfy load <=5 MW, pressure <=0.2 MPa and terminal temperature <=1 degC, and the 180 s ranges satisfy load <=10 MW, pressure <=0.5 MPa and terminal temperature <=2 degC.
- Clean-chain subgroup: additionally requires actual-valve pre-range <=1 percentage point over 60 s and <=2 points over 180 s.
- The other loop's SP and valve motion are reported as contamination features. They are not screened using post-event temperature.

All rejection reasons are counted. Events above 3 degC, non-held events and dynamic-operation events remain in the audit funnel rather than silently disappearing.

## 4. Response definitions

The baseline is the median of the last 60 s before onset. For SP sign `s = sign(delta SP)` and horizon `h` in 60, 180, 300 and 600 s:

```text
V_h = -s * (valve_h - valve_0)
D_h = -s * ((Tin-Tout)_h - (Tin-Tout)_0)
T_h =  s * (terminal_h - terminal_0)
```

Positive values therefore mean the expected cascade direction: a higher temperature SP closes the spray valve, reduces the local spray drop and raises terminal temperature, with the signs reversed for a lower SP. Each horizon endpoint is the median of its final 30 s.

Responses are also normalized by `|delta SP|`. The local valve-to-drop ratio is reported only when `|delta valve| >= 0.5` percentage point; near-zero doses are never divided. Opening and closing events are reported separately as support and sensitivity diagnostics.

## 5. Statistics and checkpoint alignment

The top-level analysis unit is UTC day. Event measurements are first aggregated within side and day. A/B contrasts use only dates supporting both sides and a paired day bootstrap; event rows and 10 s samples are not treated as independent `n`. Report point effects, 95% bootstrap intervals, event/day counts and direction rates. No p-value language is used.

The MS3 checkpoint diagnostic contributes the constant raw-valve `+5%` H600 response for each side and seed. It is compared with the empirical cascade in stages:

- `SP->valve`: whether B has a more persistent controller/actuator response;
- `valve->local drop`: whether a side-specific near-plant response is visible;
- `SP->terminal`: whether the complete closed loop differs;
- standardized checkpoint response: whether the learned terminal response asymmetry exceeds the empirical stage asymmetry.

The supervisor label is conservative:

- `FIELD_A_WEAK_SUPPORTED` only if paired-day evidence shows consistently weaker A response through the physical chain;
- `MODEL_A_RESPONSE_ABSORPTION_COMPATIBLE` when the checkpoint asymmetry is large but the empirical local/terminal A response is not demonstrably weaker;
- otherwise `INCONCLUSIVE_ASYMMETRY_DIAGNOSIS`.

The second label is a diagnosis for a new response-identification protocol, not proof of a neural-network mechanism.

## 6. Artifacts and stop rules

The local run must write a config-pinned manifest, event JSONL, event metrics CSV, validation summary and Supervisor audit. It must assert the source/matrix contract, validation bounds, no test evaluation and exact A/B cross mapping. MS3 remains FAIL regardless of this result. Linux remains unauthorized until the local audit explicitly freezes a new protocol.
