# Phase 3.5 Task 2 证据闭合: cache 生成记录 (审计 P0 回应)

> 2026-08-09 | 回应 docs/PHASE3_5_LINUX_REVIEW_2026-08-09.md Task 2 HOLD

## 生成命令 (逐字)

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/phase3_5/prepare_data.py \
  --input "/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/merged_data/A侧主汽温全数据4.csv" \
  --output "/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_A.npz" \
  --side A
# exit 0; stdout: {"rows_scanned": 55000000}{"rows_scanned": 60000000}{"rows_scanned": 65000000}{"rows_scanned": 70000000}{"cache": ".../cache_A.npz", "grid_rows": 714087}

python experiments/phase3_5/prepare_data.py \
  --input "/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/merged_data/B侧主汽温全数据4.csv" \
  --output "/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_B.npz" \
  --side B
# exit 0; stdout: {"cache": ".../cache_B.npz", "grid_rows": 714087}
```

## 产物

| 文件 | 内容 |
|---|---|
| `results/phase3_5/cache_evidence/cache_A.manifest.json` | A 侧 manifest 全文 |
| `results/phase3_5/cache_evidence/cache_B.manifest.json` | B 侧 manifest 全文 |

cache 本体 (58.8 MB / 58.8 MB .npz) 位于数据机 `/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/`, 未入 Git (体积与可复现性考虑); manifest 含 source SHA256、行数、时间范围、grid 大小与每列 staleness 统计。

## 关键 fingerprint

| 项 | A | B |
|---|---|---|
| source SHA256 | 5618fd974a36d194... | bff8c7f11608edc5... |
| raw rows | 70,020,906 | 71,204,795 |
| grid rows | 714,087 | 714,087 |
| grid start / end (ns) | 1766478330e9 / 1773619190e9 | 同 |
| SP updates / gap_med | 122,475 / 61.2 s | 117,760 / 61.2 s |
| valve updates / gap_med | 1,980,771 / 3.6 s | 2,018,476 / 3.5 s |
| target updates / gap_med | 1,446,924 / 4.5 s | 1,724,145 / 3.8 s |

## pytest 日志 (Task 3 证据)

```bash
python -m pytest tests/phase35 -q
# 25 passed in 6.41s
python -m compileall -q src/phase35 experiments/phase3_5
# (无输出, exit 0)
```

完整环境: Python 3.11.15, NumPy 2.3.5, Pandas 3.0.2, PyTorch 2.11.0+cu130, NVIDIA GB10; git HEAD 61601c8 时生成 cache, 工作树干净。

## 备注

- 审计指出 "仓库没有 manifest": 已补齐 (此前 cache 输出到数据机, 未复制回仓库 — 流程疏漏, 已修复)。
- 42-run 训练 (commit 4f8d89a) 使用上述 cache 生成, SHA256 可复现。
