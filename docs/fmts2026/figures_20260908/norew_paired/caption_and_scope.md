# Figure 2 candidate: paired forecast and action-response evidence

Files: fig2_paired_direction.pdf (vector), .svg (editable vector), .png (300 dpi preview). Grayscale diagnostic also included. Generator: experiments/final_wm/plot_norew_paired_evidence.py. Sources: results/final_wm/norew_paired_audit_20260908/{prediction,response,paired_mae_ci}.csv, independently audited from return a34481f.

## Proposed caption

**Prediction accuracy and action-response direction for matched model configurations.** (a) Validation five-temperature mean absolute error over the 18-step prediction horizon; small points show three seeds and vertical marks their arithmetic mean. All configurations use hybrid initialization and oracle boundaries. Physics-only denotes the transition without a neural closure; its hybrid observer remains learned. (b–e) Final-steam temperature response to a +0.05 increase in one normalized valve position (capped at 1), with the initial future boundary held constant. Points denote the equal-day mean response averaged over the last ten rollout steps; bars denote 95% UTC-day block-bootstrap intervals (1,000 resamples). Marker shape identifies the seed; colour and row identify the configuration. Each cell uses 64 frozen validation windows. Open markers indicate at least one unsupported baseline or intervention step; filled markers indicate all steps are supported. No samples are excluded. Eleven of twelve cells per configuration contain unsupported steps, so the response results are flagged direction diagnostics and do not establish plant response fidelity. Disabling rewetting increases the three-seed window-weighted prediction MAE by 0.283°C (17.7%) relative to the conservative closure configuration; all twelve no-rewetting response intervals lie below zero.

## What is and is not available

Available: matched prediction errors and 36 terminal-response points/intervals; original window identities and support masks independently verified. This figure is a candidate based entirely on real saved evidence; no paper source has been replaced automatically.

Not yet available: continuous response time series, because the first paired runner saved terminal summaries only. An authorized inference-only exporter now prepares all 36 existing cells without training; its outputs will support direct trajectory plots and independent response-CI recomputation.

Unavailable under current authorization: the no-closure + no-rewetting trained comparator. User explicitly declines new training, so do not draw a learning-recovery arrow within the norew path. Existing physics_only → closure_cons comparison belongs to the original rewetting structure.

No plant intervention ground truth is supplied. Further computation of these checkpoints cannot establish field response magnitude/time-course accuracy; this remains a claim boundary rather than an experiment automatically queued for execution.
