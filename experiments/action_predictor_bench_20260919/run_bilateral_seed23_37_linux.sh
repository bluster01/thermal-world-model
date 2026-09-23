#!/usr/bin/env bash
# Planned paired-seed replication. Preserve all seven bilateral main arms.
# G2 has a fixed seed11 parent and only two calibrated gains; do not pretend
# different optimization seeds are independent parent-model replications.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS="${THREADS:-1}"
export MKL_NUM_THREADS="${THREADS:-1}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
OUT="${OUT:-results/action_predictor_bench_20260919/bilateral33_seed23_37}"
mkdir -p "$OUT"
git rev-parse HEAD
"$PYTHON" -m experiments.action_predictor_bench_20260919.accept_bilateral --output "$OUT/data_acceptance.json"
"$PYTHON" -m experiments.action_predictor_bench_20260919.bilateral \
  --output "$OUT" --device "${DEVICE:-cpu}" --threads "${THREADS:-1}" \
  --arms S0_A S0_B S1 S2 S3 R3 P3 --seeds 23 37 \
  --batch-size 128 --min-epochs 12 --max-epochs 90 \
  --learning-rate 0.001 --min-lr 0.0001 --min-delta 0.002 \
  --lr-patience 4 --stop-patience 6 --response-windows 32 --resume
printf 'Return the entire directory, including ignored arrays and weights: %s\n' "$OUT"
