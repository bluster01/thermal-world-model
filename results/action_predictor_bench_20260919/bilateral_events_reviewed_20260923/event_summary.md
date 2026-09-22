# Reviewed natural actions: train only

60s detection, 64-row history, 128-step future; all controls anchored at their matching endpoint.
Opening and closing are separate. Matched differences remain observational, not physical-response ground truth.

| Valve | Candidates | Matched independent-spaced events | Opening | Closing |
|---|---:|---:|---:|---:|
| u1A | 251 | 139 | 63 | 76 |
| u1B | 471 | 205 | 79 | 126 |
| u2A | 1475 | 216 | 119 | 97 |
| u2B | 1851 | 182 | 83 | 99 |

Control pool: 919; unmatched: 0.
All prefeatures use anchor-310s through anchor only. Event/control 1920s windows do not overlap.
Previous v1 event report is superseded; its counts and response curves are not accepted for calibration.
