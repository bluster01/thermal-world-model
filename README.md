# Thermal World Model — 火电主汽温深度世界模型 + MPC

火电主汽温深度世界模型 (DWM) 与模型预测控制 (MPC) 闭环研究。目标期刊: Applied Energy（论文图表全英文）。

**数据**: 伊敏6号机，10s采样，40列（清洗后CSV全数值列），707K样本（495K训练/106K验证/106K测试）。
- 目标: 末级过热器出口汽温 (idx 29)
- SP: 二级减温调节阀设定 (idx 36，主汽温PI回路设定值)
- 动作: 一/二级减温调节门阀位 (idx 37/38，绝对阀位)

---

## 目录结构与路径约定

```
thermal-world-model/
├── config.py                  # 全局配置 (窗口/epoch/早停)  [注意: 在 src/ 下!]
├── src/                       # 核心代码 (config.py, 模型, 数据管线)
├── experiments/
│   ├── phase1_dynamics/       # Phase 1: 世界模型训练/消融/事件研究 (exp_001-028)
│   ├── phase2_mpc/            # Phase 2: DWM-MPC 主循环+评测 (exp_027-052)
│   └── phase3_feedforward/    # 路线B: SP前馈/PI辨识 (exp_033-040)
├── docs/                      # 设计+结果文档 (phase2_results.md = 最新结论)
├── results/exp_0XX_*/         # 实验结果 (JSON: agg+per_track; checkpoints)
├── figures/                   # 论文图表 (全英文)
├── data/  notebooks/
```

**运行约定**:
- 环境: `conda activate Alloftime`，**一律从仓库根目录运行**:
  `python experiments/phase2_mpc/exp_052_overlap_consistency.py`
- 脚本头部已注入根路径 + `root/src`（config.py 在 src/），迁移后 import 链路已验证
- **exp_050/051/052 已加 `__main__` 保护**：import 不会执行实验（曾因无保护导致 import 误跑覆盖结果）
- 新实验**必须**加 `if __name__ == '__main__':` 保护

---

## 项目进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 世界模型训练/消融/因果验证 | ✅ 完成 (M7 定稿) |
| Phase 2 | DWM-MPC 闭环 (路线A: 阀位) | ✅ 完成 (含边界跳变修复) |
| Phase 2.5 | 证据链: 保真度/基线/统计/鲁棒性/扰动 | ✅ 完成 |
| 路线B | SP前馈 (伊敏落地) | ⏸ 暂停 (结论: 动作通道二选一) |
| 论文 | Applied Energy 写作 | 📝 准备中 (图表+表格) |

### Phase 1 核心结论 (docs/phase1_report.md, phase1_conclusions_audit.md)
- **M7 定稿**: β-NLL(β=−0.3), RevIN+PerVarTCN+VarAttn+动作concat, H_OUT=18步(180s)
  - avg rollout MAE **0.295°C**, 单步0.077→18步0.488, 敏感性 sens −0.411, σ=0.74, 11工况全胜
- 物理时标: 主汽温对减温阀 **60-90s+大滞后**, 10min达−3.4°C（事件研究定稿）
- 绝对阀位 >> 差分阀位 (敏感性×32-130); 单步符号正则**有害**（压滞后成伪响应）
- 评测协议: 只扰未来动作首步（全步扰动测到PID共因统计伪正）
- ⚠️ 旧README的"动作响应≈0 / GRU broken"结论已被推翻（差分阀位+未训练GRU分支），勿引用

### Phase 2 核心结论 (docs/phase2_results.md §1-12)
- **M_STEP=6 多步执行**（60s动作段，对齐过程时标）: 单步10s执行时标错配→发散/假象
- 双协议: 无扰动世界 RMSE↓36-44%（理想）/ 扰动世界 ↓8.4%（主结果，保守）
- 公平协议: PID 也走 WM 闭环+同一扰动世界（"PID真实 vs MPC扰动"对比是假反转）
- 证据链: 保真度1200s MAE 1.2°C / ARX基线增量14.6% / Wilcoxon p<1e-4 / 噪声σ=0.5退化 MPC 179% vs PID 257%
- 虚拟世界必须注入过程扰动（WM=条件期望天然平滑，无扰动MPC呈直线）

### 边界跳变问题 (exp_050-052, §10-12) — 2026-08-03
- **根源**: 重规划边界不连续占动作TV 28-46%（m=6时~45%），基线viol_mpc 17次/10轨迹（边界跳变最大8.78% > 自身|Δa|≤5限幅）
- **"60s最优"不成立**: 扰动世界 RMSE/TV 单调改善到 m=18（满视野执行），MPC优势 −8%→−14%
- **hard5 = 免费修复**（默认推荐）: 首步钳制 a[0]∈a_last±5，viol 17→0，RMSE 2.040→2.034，jump_max 8.78→5.0
- **overlap = 平滑家族最优**（利用预测后半程）: 旧计划未执行段作软轨迹参考 J+=λ3·Σ|a_new[j]−a_old[M_STEP+j]|²，λ3=0.5: jump −76%/TV −52%，代价 RMSE +14%；优于惯性块(inert +17%)与融合(blend +31%)
- 组合建议: 默认 hard5；要强平滑用 ovl05_hard5

---

## 核心实验索引

| 实验 | 内容 | 结果位置 | 文档 |
|------|------|----------|------|
| exp_025_M7 | 世界模型定稿 | results/exp_025_M7/ | phase1_report.md |
| exp_027 | DWM-MPC 主循环 (grad/CEM) | results/exp_027_M7/ | phase2_results §1-7 |
| exp_042 | 多轨迹公平对比 (9/9全胜) | results/exp_042_trajectories.json | §8-9 |
| exp_044 | 保真度 1200s 闭环 | results/exp_044_fidelity/ | §8 |
| exp_045 | LTI-MPC 线性基线 | results/exp_045_lti/ | §8 |
| exp_046 | Wilcoxon 统计检验 | results/exp_046_stats/ | §8.3 |
| exp_048 | 预测长度扫描 + 鲁棒性 | results/exp_048_horizon/, exp_048_robustness.json | §8.4 |
| exp_049 | 扰动世界响应 | results/exp_049_disturbance.json | §9 |
| exp_050 | M_STEP×拼接模式扫描 | results/exp_050_mstep_dist/ | §10 |
| exp_051 | 边界修复三方案 (H18/H10) | results/exp_051_boundary_fix/, _H10/ | §11 |
| exp_052 | 重叠一致性 (用户想法) | results/exp_052_overlap/ | §12 |
| exp_033-040 | 路线B: PI辨识/SP前馈/联合 | results/exp_033~040/ | phase2_plan.md |

完整索引: `results/README.md`

---

## 实验管理规范（防混乱，2026-08-03 定）

1. **脚本按阶段放** `experiments/<phase>/`，严禁根目录堆脚本；命名 `exp_0XX_<描述>.py` 编号连续
2. **必须加** `if __name__ == '__main__':` 保护（import 不应有副作用）
3. **结果目录**: `results/exp_0XX_<描述>/`（JSON 含 agg + per_track）；**不同协议/参数用不同目录或文件后缀**（教训: exp_051 的 H=18 与 H=10 曾混在同一目录互相覆盖）
4. **流程**: 实验设计 → 留痕(git commit) → GitHub同步 → 跑实验 → 分析 → 讨论 → 下一步，每步 commit
5. 新结论必须更新 `docs/phase2_results.md` + 本 README 进度段
6. 论文图表全英文；中文术语首次出现括号标注英文

---

## 关键技术参数 (主协议)

- WM: M7, H_OUT=18, β-NLL fixed −0.3, RevIN, seed 42, 早停 patience=20
- MPC: grad planner, H_PLAN=10, α=0.5 (终端), λ1=0.1 (平滑), λ2=0.05 (锚定), E=30步 Adam, |Δa|≤5/10s
- 闭环: M_STEP=6 (60s块), DIST_AMP=0.3 随机游走扰动 (自相关0.9), 120步×10s=1200s, 10-50轨迹 seed42
- 修复: FIX_MODE ∈ {none, hard2, hard5, blend, inert05, inert025} + overlap(λ3) + HARD_DELTA

## 参考文献

- Differentiable World Models for Offline RL + MPC (arXiv, 2026.03)
- Graph Spatiotemporal World-Model-Driven Rolling MPC (Electronics, 2026)
- Differentiable Predictive Control / Neuromancer (PNNL)
- iTransformer-SST (Zhang et al., Sensors, 2026) — baseline
