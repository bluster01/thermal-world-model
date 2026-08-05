# Linux 代理 TODO (2026-08-05 晚)

> 上游审查已完成并推送 (commit 5c954bc)。以下按优先级排列。
> 脚本均自带门禁断言 (L0 动作编码往返 / g(x,0)=0 不变量), 不过门直接 assert 失败, 不会跑出脏数。

---

## P0 — 纯推理, 最快出结论 (CPU 即可, ~10min)

- [ ] `python experiments/phase3_feedforward/exp_103_protocol_recheck.py`
  - 先跑这个。它会:
    1. 断言 `build_action` 与训练取法逐元素相同 (不过门直接崩)
    2. 复现 bug: 展示旧(二阶差分) vs 新(一阶差分) 动作序列差异
    3. 用修正编码重测 4 个已有 ckpt (M5DSP / M7DSP / M9DSP60 / M9DSP18), 新旧口径并列
  - **看输出最后的"判决"表**: 如果 `ratio > 1.5` 或方向提升 >10%, 说明 exp_100-102 的 FAIL 结论可能翻转 → 后续策略会变

---

## P1 — 通宵训练 (GPU, 7 变体 × seed 0, ~8-12h)

- [ ] 先 smoke 确认环境 (4 epoch, ~5min):
      `python experiments/phase3_feedforward/exp_106_causal_arch.py --variant A1phys --seed 0 --smoke`

- [ ] 全 7 变体 seed 0:
      `python experiments/phase3_feedforward/exp_106_causal_arch.py --all --seeds 0`

- [ ] 如时间够, 关键 3 变体补 5 seeds (论文需 n≥5):
      ```
      python experiments/phase3_feedforward/exp_106_causal_arch.py --variant A1phys --seeds 0,1,2,3,4
      python experiments/phase3_feedforward/exp_106_causal_arch.py --variant B1glb  --seeds 0,1,2,3,4
      python experiments/phase3_feedforward/exp_106_causal_arch.py --variant B1flat --seeds 0,1,2,3,4
      ```

**关键对比** (跑完看 `summary.json` 末尾的汇总表):
- `B1glb` vs `B1flat`: 只差 head → "精度 vs 因果"权衡的直接证据 (**论文缺的那张图**)
- `A1phys` vs `A1mlp`: 物理先验的价值
- `*_cs` vs 无后缀: C1 增量累积的效果

---

## P2 — DiD 真值 (CPU, 一次性, ~30min)

- [ ] 建 `exp_104_did_groundtruth.py` (参考 `causal_eval.py` 中的 `did_response` / `match_controls`)
  - 事件只取 test 区间: `CE.select_events(raw, I_SP, I_LD, H=60, lo=n_val_end, W=W)`
  - 输出落 `results/cfe_groundtruth/did_response.json`
  - 产出: `R_true[k]`, `ci_lo[k]`, `ci_hi[k]`, `sgn_ceiling[k]`, `noise_floor[k]`, `r` (逐事件斜率)
  - **exp_106 的 `--gt` 参数默认读这个文件**; 没有它时 CFI 退化为 sign(ΔSP) 口径 (只能选 ckpt, 不能作最终结论)

- [ ] 跑完 P1 后用 DiD 真值重评:
      `python experiments/phase3_feedforward/exp_106_causal_arch.py --variant A1phys --seed 0 --epochs 0`
      (epochs=0 跳过训练, 仅用已有 ckpt 做 CFE 评测 — 如脚本支持; 否则手动调 `eval_causal(model, H, gt)`)

---

## P3 — 修旧脚本 (9 个, 按需)

按 `docs/causal_eval_framework.md` §4.1 逐行修复清单:

- [ ] 统一替换所有 `np.diff(raw41[s+W-1:s+W+H, I_DSP])` → `CE.build_action(raw41, s, W, H, I_DSP)`
- [ ] 优先级: `exp_097_fig_cases_v3.py` (主模型论文图) > `exp_100/101/102` (主结果) > 其余
- [ ] 修完重跑出图

---

## 注意事项

1. **OMP 冲突**: 如遇 `libiomp5md.dll already initialized`, 设 `export KMP_DUPLICATE_LIB_OK=TRUE`
2. **exp_106 训练前后各做一次 `g(x,0)=0` 断言** — 训练后失败说明动作分支学了 bias, 结果不可信
3. **DiD 真值就位前, gain/方向仍是 sign(ΔSP) 口径, 不作最终结论** (脚本会打印警告)
4. **事件区间**: 旧 exp_100-102 在全量数据选事件含训练集 → 已修 `select_events(lo=, W=)`, 重测时须传 `lo=n_val_end`
5. **不要改训练段**: 各脚本训练段语义正确 (一阶差分), 权重有效无需重训; 只改推理/评测/出图通路
