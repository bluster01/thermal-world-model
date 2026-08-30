# Linux 执行单：JEPA-B 系列 v1

## 授权范围

只执行 `configs/final_wm/jepa_b_series_v1.json` 中按序冻结的
`c0 → b1 → b2 → b3 → b3_shuffle → b4`。这是侧 A、seed0、validation-only
探索批；test、补种子、重试、搜索和论文 verdict 均未授权。

## 拉取与只读核对

```bash
git fetch origin main
git status --short
git pull --ff-only origin main
git rev-parse HEAD
python experiments/phase3_5/experiment_status.py --check --json
sha256sum configs/final_wm/jepa_b_series_v1.json
```

要求：

- tracked 工作树为空；Linux 自有 `artifacts/`、checkpoint 等 gitignore 文件可存在；
- registry 输出 `active_gate=jepa_b_series`、`linux_authorized_gate=jepa_b_series`、
  `active_status=ready_for_linux`；
- matrix SHA-256 必须为 `b664c06272318775ad5aa89cc93c337c09a72806e5b16340552d536c66224751`；
- `artifacts/final_wm/canonical_sideA_v2.npz` 必须为 canonical v2.2，且
  `artifacts/final_wm/iapws_surrogate.npz` 存在。

任一项不符即停止，不自行修矩阵或数据。

## 执行

先跑不训练的合同门：

```bash
python experiments/final_wm/run_jepa_b.py --sanity
```

只有 `b1/b2/b3/b3_shuffle/b4` 五项 `exact=true` 才可继续：

```bash
python experiments/final_wm/run_jepa_b.py --queue
```

顺序执行，禁止并行 worker。runner 自带完整臂复用和半臂拒绝：已有完整、同 commit+
matrix hash 的 report 可跳过；出现半臂 ledger/checkpoint 而无 report 时不得自动重跑。

## 回传

提交并 push（checkpoint 默认不入 Git）：

- `results/final_wm/jepa_b_series_v1/sanity_report.json`
- 六臂各自的 `ledger.jsonl` 与 `report.json`
- `results/final_wm/jepa_b_series_v1/report.json`

回传前确认 root report 为 6/6 臂且 `paper_verdict_upgraded=false`。若某臂 OOM、NaN、
中断或质量门失败，如实回传错误/已完成 ledger；不要改 batch、维度、权重、stride、seed、
epoch、patience、评估窗或方向判据，也不要补跑 seeds 1/2。
