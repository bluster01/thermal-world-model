# V0–V4 冻结协议 Linux 执行报告 (2026-08-09)

> 审计: docs/PHASE35_SEGMENTED_IDENTIFICATION_REVIEW_2026-08-09.md (codex 远端审计)
> 执行: Linux (DGX Spark), 冻结脚本不调阈值不筛结果
> 数据: cleaned_data/all_merged_10s.csv (10s 网格, 2025-12-24~2026-05-11, 1,192,329 行)

## 1. 结论 (Supervisor 判定一致)

**V0–V4 全部执行, 无一通过放行门槛。85%/74% 与 0s 峰值确认仅为 exploratory pilot,
交叉拓扑物理结论 NOT VERIFIED。E3 保持 INCONCLUSIVE, E4 保持 BLOCKED。**

| Gate | 结果 | 关键数字 |
|---|---|---|
| V0 | **INCONCLUSIVE** | A=7 事件/5 日, B=6 事件/5 日 (<30/8); 12 open/1 close → 无双向 support; held-step=2 |
| V1 | **INCONCLUSIVE** | paired area CI 含 0: A=[-0.27,+0.35], B=[-8.35,+0.74] (日块 bootstrap 2000) |
| V2 | **INCONCLUSIVE** | held-step 仅 2 个, 主分析不可行; trajectory ARX n 过小 (A=7/B=6) |
| V3 | **NOT PASSED** | 同侧系数不占优 (b_oth>b_same 全部块); 双侧共线; R²=1.0 自回归主导 |
| V4 | **NOT PASSED** | B 错侧面积(4.53)>真配对(1.88); 分层 close 格空 (A 无 close, B 仅 1) |

## 2. V0 事件漏斗 (3% 主阈值)

- 阀位粗筛 (|Δ|≥1.5%) → 稳态门禁 (负荷/压力/温度/阀位 range) → 剂量≥3% → 另一阀安静 → 独立间隔
- 严格门禁下事件骤减至 A=7/B=6, held-step 仅 2 个
- **原因**: 闭环 PID 持续回调 + 稳态窗苛刻 (600s 内 5 条件同时满足) + 双向 common support 不存在
- 敏感性: 2% → A=13/B=23; 5% → A=0/B=2 (全 open 为主)

## 3. P0 问题修正对照

| P0 | 修正 |
|---|---|
| P0-A 阶跃符号 | ✅ 剂量 = median(post30-60s) - median(pre60s) 保留符号 |
| P0-B 非可辨识阶跃 | ✅ held-step 门禁 (剂量完成后 300s 保持); trajectory ARX 次分析 |
| P0-C 伪脉冲响应 | ✅ 废弃; V1 用有符号温降响应面积, V2/V3 用 ARX |
| P0-D 无验证集 | ✅ dev(12-24~03-31)/val(04)/rob(05) 切分; dev 选阶, val 单次评估 |

## 4. 产物清单 (审计 §9)

```
results/phase35_segmented_v2/
  run_manifest.json          ✅ (git SHA 625b4f7, CSV SHA256 85a3f9..., 阈值/切分/环境/命令)
  data_audit.json            ✅ (行数/时间范围/缺口)
  event_manifest.jsonl       ✅ (13 事件, A=7/B=6)
  leading_response_by_event.csv  ✅ (每事件 R_left/R_right 轨迹+面积)
  leading_summary.json       ✅
  blocked_model_scores.json  ✅ (V2 ARX + V3 blocked + dev→val/rob NRMSE)
  parameter_health.json      ⚠️ V2 ARX 参数 (b/a1) 含于 blocked_model_scores.json; 无独立文件
  placebo_summary.json       ✅ (错移/日内置换/错侧/分层)
  bootstrap_summary.json     ✅ (日块 CI)
  console.log                ✅ (各脚本 stdout 存 /tmp/v0..v4_out.txt)
```

## 5. 对 5C/论文的影响

1. **事件不足是数据特性, 不是实现缺陷**: 闭环 PID 下 held-step 大剂量事件稀缺,
   现有观测数据无法满足 V0 门槛 → 分段辨识在当前数据上只能 exploratory
2. **V0-V4 脚本即 5C 的验证骨架**: 冻结协议已代码化 (experiments/phase3_5/segmented_v2/),
   未来新时间块 (未查看) 可原样重跑, 无需改阈值
3. **论文表述边界** (审计 §10): 只能写 "分段分析发现与交叉回路一致的探索性信号,
   但现有闭环观测和事件支持不足以确认物理拓扑"
4. **confirmatory 证据路径**: 未来新时间块 / 现场预注册小幅阶跃试验 (唯一能解锁双向
   common support 的途径)

## 6. 复现命令

```bash
PY=/home/bluster/anaconda3/envs/Alloftime/bin/python3
CSV=/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/cleaned_data/all_merged_10s.csv
OUT=results/phase35_segmented_v2
$PY experiments/phase3_5/segmented_v2/v0_events.py --csv $CSV --out-dir $OUT
$PY experiments/phase3_5/segmented_v2/v1_leading.py --csv $CSV --events $OUT/event_manifest.jsonl --out-dir $OUT
$PY experiments/phase3_5/segmented_v2/v23_models.py --csv $CSV --events $OUT/event_manifest.jsonl --out-dir $OUT
$PY experiments/phase3_5/segmented_v2/v4_placebo.py --csv $CSV --events $OUT/event_manifest.jsonl --out-dir $OUT
$PY experiments/phase3_5/segmented_v2/gen_manifest.py
$PY experiments/phase3_5/segmented_v2/gen_bootstrap.py
```
