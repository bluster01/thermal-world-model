# Linux 执行单：JEPA-B5 v1

## 授权范围

只执行 `configs/final_wm/jepa_b5_series_v1.json` 的 `c0 → b5`（侧 A、seed0、
validation-only 探索批）。test、补种子、重试、搜索和论文 verdict 未授权。

## 前置（B 系列 v1 已回传）

- `results/final_wm/jepa_b_series_v1/` 6/6 报告已 push（`cde385e` 含方向门
  原轨迹口径修复）。
- registry 已切换 `active_gate/linux_authorized_gate = jepa_b5`。

## 只读核对

```bash
git fetch origin main && git pull --ff-only origin main
git rev-parse HEAD
python experiments/phase3_5/experiment_status.py --check --json
sha256sum configs/final_wm/jepa_b5_series_v1.json  # 必须 28dcb4b6…ee884
```

## 执行

```bash
python experiments/final_wm/run_jepa_b.py --sanity       # b5 exact=true 才继续
python experiments/final_wm/run_jepa_b.py --queue        # c0 → b5，禁止并行
```

## 回传

提交 push：`results/final_wm/jepa_b5_series_v1/sanity_report.json`、
`c0/`、`b5/` 的 ledger.jsonl + report.json、root `report.json`。
确认 root 2/2 臂 + `paper_verdict_upgraded=false`。
OOM/NaN/中断原样回传，不改任何冻结参数。
