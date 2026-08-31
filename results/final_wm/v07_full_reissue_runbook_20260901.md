# Final-WM v0.6 / v0.7 全量 Linux 冻结执行单（2026-09-01）

> 授权批次：`final_wm_v07_full_reissue_v1`。只执行本单；不改代码、阈值、seed、数据、预算或命令；不重试失败 run；不访问 test；不写论文结论。

## 1. 唯一命令

```bash
set -euo pipefail

REPO=/home/bluster/projectA/thermal-world-model
DATA_ROOT='/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10'
RUN_ROOT=/home/bluster/final_wm_v07_full_reissue_v1
INPUT_ROOT="$RUN_ROOT/inputs"
PROP="$REPO/artifacts/final_wm/iapws_surrogate.npz"
PROP_INPUT="$INPUT_ROOT/iapws_surrogate.npz"

cd "$REPO"
git pull --ff-only origin main
test -z "$(git status --porcelain)"
python -m pytest tests/final_wm -q
python -m compileall -q src/final_wm experiments/final_wm
test -f "$PROP"

mkdir -p "$INPUT_ROOT"
cp "$PROP" "$PROP_INPUT"

python experiments/final_wm/build_canonical_v2.py \
  --side A --data-root "$DATA_ROOT" \
  --v1 artifacts/final_wm/canonical_sideA.npz \
  --out "$INPUT_ROOT/canonical_sideA_v2.npz"

python experiments/final_wm/build_canonical_v2.py \
  --side B --data-root "$DATA_ROOT" \
  --v1 artifacts/final_wm/canonical_sideB.npz \
  --out "$INPUT_ROOT/canonical_sideB_v2.npz"

for SIDE in A B; do
  OUT="$RUN_ROOT/side${SIDE}"
  RECORD="$INPUT_ROOT/canonical_side${SIDE}_v2.npz"

  python experiments/final_wm/run_matrix.py \
    --phase dsyn --device cuda --out "$OUT"

  python - "$OUT/dsyn_verdict.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p["verdict"] == "PASS" and p["quick"] is False
assert sorted(x["seed"] for x in p["per_seed"]) == [0, 1, 2]
assert all(x["n_perturbed"] > 0 and x["parameter_delta_l2"] > 0 for x in p["per_seed"])
PY

  python experiments/final_wm/run_matrix.py \
    --phase matrix --device cuda \
    --record "$RECORD" --side "$SIDE" \
    --properties-npz "$PROP_INPUT" \
    --out "$OUT"

  python experiments/final_wm/audit_manifest.py \
    --manifest "$OUT/manifest.json"
done

git status --short
```

## 2. 固定规模

- 每侧 39 个训练 run：D-SYN 3、O1 9、T1 15、B1 3、J1 9；R1 复用 T1 权重不训练。
- 每个训练 spec 显式 `epochs=120 / patience=20`；seeds 仅 `0,1,2`。
- R1 只审 `closure_cons_norew`，双阀 H18/H60；任一步越出逐样本支持域则 R1=`INCOMPLETE`，不得改成外推判决。
- A/B 侧分别生成 summary/manifest，禁止聚合。

## 3. 成功与停止条件

1. 两份 canonical meta 均为 v2.2；raw quality、接线连续性、对齐门全部通过。
2. D-SYN 每侧 3/3 完整，teacher 扰动数量与 L2 距离均大于零，且总判决 PASS；否则立即停止该侧，不进入真实矩阵。
3. matrix 不因科学 `REJECTED/MIXED/INCOMPLETE` 返回而补跑；只要命令正常闭合并如实写证据即可。
4. `manifest.json` 独立校验通过；任何哈希、run、seed、summary 或 test-lock 缺口均停止。
5. 进程异常、CUDA OOM、文件缺失或非零退出：保留原输出和 stderr，停止，不自动重试。

## 4. 回传清单

- `$RUN_ROOT/inputs/canonical_side{A,B}_v2_meta.json`；NPZ 可带外回传，但 manifest 所引路径/哈希必须可复核。
- `$RUN_ROOT/side{A,B}/manifest.json`、`matrix_summary_side{A,B}.json`、`dsyn_verdict.json`、`r1_report.json`、`ledger.jsonl`、`checkpoints/`、`metrics/`。
- `git rev-parse HEAD`、执行命令日志、GPU/驱动/Python/PyTorch 环境摘要。

回传后只由本地执行 `audit_manifest.py` 和判决复算；注册表的 `results_returned/audited` 在复核前保持 false，论文 verdict 不自动升级。
