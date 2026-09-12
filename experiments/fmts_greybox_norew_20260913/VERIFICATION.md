# Local verification — 2026-09-13

Executed on the preparation host, not the plant record:

```text
python -m pytest tests/final_wm/test_fmts_greybox_norew.py tests/final_wm/test_fmts_rich_pipeline.py tests/final_wm/test_fmts_mainsteam_experiment.py -q
17 passed in 23.59s
```

Seven GNR1 tests cover three seeded spec deltas, initial parameter identity,
end-to-end synthetic smoke plus checkpoint replay, formal/smoke separation,
and rejection of changed input bytes. Ten retained parent tests also pass.

- Scientific spec differences: only `arm` and `closure_mode`.
- Initial tensor differences: only `base.transition.raw.aW1/aW2`.
- These gains are `-30` in raw space and frozen; no neural closure is activated.
- Synthetic new arm: seed 0, two optimizer updates; both prediction and response arrays replayed.
- Original index file is byte-identical; paired targets, days, support masks and doses match.
- Existing output paths are rejected. No hidden resume or failed-seed replacement.
- Old v0.2 24-array paper audit still passes after implementation; frozen parent source was not edited.

No formal GNR1 training, locked-test access, commit, push or Linux dispatch occurred.
The full GNR1 result and scientific interpretation remain pending.
