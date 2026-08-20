# final_wm 审计脚本（转正版）

本目录脚本是 2026-08-19/20 动作通道审计的转正产物：逻辑与 /tmp 运行版完全一致，
`results/final_wm/evidence_chain.md` 中的数字由这些脚本产出（数值溯源）。
全部从仓根运行（相对路径假定 cwd=仓根）。

## 证据链脚本（evidence_chain.md §1-4 数字来源）

| 脚本 | 产出 | 对应文档节 |
|---|---|---|
| `evidence_phys_ref.py` | dW/dv 回归、混合能量平衡参考、真实事件研究 | §2, §3 |
| `evidence_valve_audit.py` | 学习参数实值、喷水量对比、H1 拆解、长时反转定位(closure 开/关) | §4 |
| `evidence_rewet_ablation.py` | 再湿项 aW=0 消融 (根因实锤) | §4.3 |

## 论文配图数据脚本

| 脚本 | 产出 |
|---|---|
| `probe_step_response.py` | ±2% 阀位阶跃响应曲线 → fig3a |
| `eval_perchannel.py` | 官方 evaluate_windows per-channel MAE (H1/H6/H18) → fig3b |
| `eval_perchannel_bins.py` | per-channel MAE + persistence 基线 |
| `binning_v02.py` | v0.2 全通道负荷分箱比率与箱均值 → fig3c |
| `make_paper_figs.py` | 生成 fig1/fig2/fig3 三张 PDF |

## 运行注意

- 需 `~/anaconda3/envs/Alloftime/bin/python`（torch + CUDA），GPU 可用
- IAPWS 网格路径硬编码（env-specific）：`/home/bluster/.hermes/workspace/adhoc2_lumped_enthalpy/out/iapws_surrogate.npz`
- checkpoint 依赖 `artifacts/final_wm/checkpoints/t1_closure_steam_seed0.pt`
- `ledger_timing.py`：T1 各 run 训练耗时（wall_seconds），支撑执行报告 §3
- 事件研究/回归结论含闭环混杂警示（控制器响应扰动才动阀），引用时须带 caveat
