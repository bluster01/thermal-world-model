# Private 1/10 plant prototype inputs

Run preparation on the Linux worker using its own local source files. Do not upload the source CSV, resulting NPZ, full local manifest, local audit report, weights or plant statistics to a public repository. `public_receipt.json` is intentionally limited to hashes, counts and software identifiers.

```bash
python -m pytest analysis/world_model_plant/test_prepare_tenth.py -q
python -m analysis.world_model_plant.prepare_tenth \
  --canonical /private/input/canonical_sideA_v2.npz \
  --raw /private/input/all_merged_10s.csv \
  --output /private/output/tenth_v1
python -m analysis.world_model_plant.verify_tenth \
  --dataset /private/output/tenth_v1 \
  --output /private/output/tenth_v1/saved_array_audit.json
```

Only NumPy and Python standard-library modules are required for preparation/audit; pytest is used for authored fixtures. The output directory must not exist. The canonical full-file hash is checked, but only the selected timestamp prefix and old-training-boundary split metadata are decoded; original observations and held-out arrays are never loaded. Raw CSV records are read only through the selected time cutoff, without future interpolation. If the cutoff has no source record, only the following timestamp field is inspected, without its numeric fields.

The denominator is the existing 530,779-row training split. Its earliest `floor(N/10)=53,077` continuous 10-second gridpoints form this prototype. They are partitioned chronologically into `[0,42461)` for prototype training and `[42461,53077)` for prototype validation. These are both inside the old training split. Prototype validation is not a new independent reserved test set.

NPZ uses numeric/bool arrays only and can be loaded with `numpy.load(path, allow_pickle=False)`. Core arrays:

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| `timestamps_epoch_s` | N | int64 | Encoded source time mapped to epoch seconds |
| `observations` | N,2,5 | float32 | Five temperatures, degrees Celsius |
| `positions` | N,2,2 | float32 | Measured actual valve feedback, fraction; unmodified outliers retained |
| `boundary` | N,2,7 | float32 | kg/s, t/h, MPa, Celsius, Celsius, MPa, t/h |
| `command_candidates_unconfirmed` | N,2,2 | float32 | Main-controller-output candidates, original units UNCONFIRMED; excluded from model inputs |
| `raw_values` | N,26 | float64 | Selected raw numeric columns before unit conversion, missing export rows/cells NaN |
| `raw_exported_record_mask` | N | bool | Whether a source CSV record exists at that gridpoint |
| `prototype_split` | N | int8 | 0 prototype train, 1 prototype validation |

Side axis is `[A_left, B_right]`; stage 1 feedback uses the same side, stage 2 uses the crossed side. Observation order is `sh1_inlet_temp, sh1_outlet_temp, sh2_inlet_temp, sh2_outlet_temp, final_outlet_temp`. Boundary order is `steam_flow, coal_command, separator_pressure, separator_temperature, feedwater_temperature, outlet_pressure, spray_flow_total`. The `coal_command` compatibility name refers to recorded uncorrected coal quantity, not a confirmed executable command. Full local manifest binds channel names, raw indices and conversion factors.

Each core group has `_present_mask`, `_available_mask`, `_age_s` and `_source_timestamp_epoch_s` arrays of the same shape. `present` means this exported cell is finite at this tick; `available` means some finite value is causally available by this tick. Missing updates only carry the last finite exported value, with positive age. Unknown values remain NaN, age infinity and source timestamp the minimum int64 value. There is no clipping or zero fill.

Use `observations_present_mask` for online measurement assimilation and target losses. A held value is not a fresh observation or a fresh target. Anchor construction may use available values under its declared age limit and must record ages. Export age does not certify sensor update time or network arrival; the upstream production of the merged CSV remains unverified for causality. Its encoded timezone is preserved computationally, without certifying the DCS's actual timezone.

Window arrays are `{A,B}_{train,prototype_validation}_{basic,property_input}_window_starts`, dtype int64. For a start `s`, history is `[s,s+64)`, the anchor is at `s+63`, and future targets are `[s+64,s+192)`. All 192 gridpoints must belong to their own prototype split. The saved-array audit also rejects a retained window if a held required value originates before that split. These indices do not prescribe whether a recorded position snapshot acts over the preceding or following interval; a model runner must separately declare that timing assumption.

`basic` requires finite observations, positions and the first six boundaries, each age ≤30 seconds. `property_input` further requires the current table's input ranges, positive steam flow, nonnegative coal, ordered pressures, and positions in [0,1]. Raw values are retained even when windows are excluded. These checks do not certify the composed thermodynamic query domain, phase validity, steady-state anchor reachability, identifiable physical parameters or plant safety. The runner must separately validate its anchors and physics queries.

Recorded future positions and future boundaries support an explicitly observational conditional/oracle diagnostic. They are not known future forecasts or intervention truth. For the world-model adapter, exclude total spray-flow W from every history mask and use a declared zero placeholder for its ignored future slot. A deployment-facing branch needs independently generated boundary forecasts and a confirmed command-to-position contract.

`public_receipt.json` contains individual stable array hashes. The hash is SHA256 of canonical ASCII JSON `{dtype: dtype.str, shape: [...]}` (sorted keys, no spaces), one newline byte, and array bytes in C order. Compare these hashes across machines; do not rely on ZIP container metadata remaining identical. The private `manifest.json` additionally contains raw source-prefix identity, local paths, data statistics, selected times, mapping and all denominators. Keep it private.

No model, optimizer, checkpoint or historical test set is used by these scripts. Successful preparation is not a forecast, causal-identification, actuator-identification or closed-loop-control result.
