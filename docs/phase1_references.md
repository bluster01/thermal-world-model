# Phase 1 论文引用参考列表

> 生成时间: 2026-08-04
> 来源: Phase 1 实验文档中提及的方法、模型、技术
> 用途: 论文参考文献定稿

---

## A. 核心方法引用 (World Model + MPC)

### 1. TD-MPC2 — 任务导向世界模型哲学
- **标题**: TD-MPC2: Scalable, Robust World Models for Continuous Control
- **作者**: Nicklas Hansen, Hao Su, Xiaolong Wang
- **会议**: ICLR 2024 (Spotlight)
- **arXiv**: 2310.16828
- **引用理由**: Phase 1 "单目标合理"结论的理论依据 — TD-MPC2 的 task-oriented 哲学：只预测控制相关变量，容量集中
- **BibTeX**:
```bibtex
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control},
  author={Hansen, Nicklas and Su, Hao and Wang, Xiaolong},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

### 2. DreamerV3 — 世界模型通用性
- **标题**: Mastering Diverse Control Tasks through World Models
- **作者**: Danijar Hafner, Jurgis Pašukonis, Jimmy Ba, Timothy Lillicrap
- **期刊**: Nature (2025)
- **arXiv**: 2301.04104
- **引用理由**: 世界模型在多样控制任务中的 SOTA，作为 world model for control 的背景参考
- **BibTeX**:
```bibtex
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pašukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  year={2025},
  publisher={Nature Publishing Group}
}
```

### 3. PETS — 概率集成 + 轨迹采样
- **标题**: Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
- **作者**: Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine
- **会议**: NeurIPS 2018
- **arXiv**: 1805.12114
- **引用理由**: Phase 2 中 PETS 式集成基线 (3×M7+CEM) 的方法来源；概率不确定性在 MBRL 中的应用
- **BibTeX**:
```bibtex
@inproceedings{chua2018pets,
  title={Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models},
  author={Chua, Kurtland and Calandra, Roberto and McAllister, Rowan and Levine, Sergey},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

### 4. Neuromancer / Differentiable Predictive Control
- **标题**: Differentiable Predictive Control (DPC)
- **作者**: Jan Drgona, Aaron Tuor, Draguna Vrabie
- **机构**: Pacific Northwest National Laboratory (PNNL)
- **开源**: github.com/pnnl/neuromancer
- **引用理由**: 可微 MPC 框架参考，将 ML 模型嵌入 MPC 的端到端控制策略
- **BibTeX**:
```bibtex
@misc{drgona2023neuromancer,
  title={Neuromancer: Differentiable Programming for Constraint-Based Control},
  author={Drgona, Jan and Tuor, Aaron and Vrabie, Draguna},
  year={2023},
  howpublished={\url{https://github.com/pnnl/neuromancer}}
}
```

---

## B. 模型架构组件引用

### 5. RevIN — 可逆实例归一化
- **标题**: Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift
- **作者**: Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, Jaegul Choo
- **会议**: ICLR 2022
- **引用理由**: 消融实验核心结论 — RevIN 不可或缺 (去掉后 MAE ×70)
- **BibTeX**:
```bibtex
@inproceedings{kim2022revin,
  title={Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift},
  author={Kim, Taesung and Kim, Jinhee and Tae, Yunwon and Park, Cheonbok and Choi, Jang-Ho and Choo, Jaegul},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

### 6. PatchTST — Patch 嵌入
- **标题**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **作者**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
- **会议**: ICLR 2023
- **arXiv**: 2211.14730
- **引用理由**: 模型架构中 Patch(16, stride 8) 的来源；消融 M2 (−patch) 验证其贡献
- **BibTeX**:
```bibtex
@inproceedings{nie2023patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

### 7. iTransformer — 倒置 Transformer
- **标题**: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- **作者**: Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long
- **会议**: ICLR 2024 (Spotlight)
- **arXiv**: 2310.06625
- **引用理由**: Baseline B4 (iTransformer) 的方法来源；VariableAttention 设计的灵感参考
- **BibTeX**:
```bibtex
@inproceedings{liu2024itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

### 8. β-NLL — 概率损失函数
- **标题**: On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks
- **作者**: Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, Georg Martius
- **会议**: ICLR 2022
- **arXiv**: 2203.09168
- **引用理由**: 概率头训练损失 β-NLL (β=−0.3) 的来源；exp_005/006/007 消融的核心方法
- **BibTeX**:
```bibtex
@inproceedings{seitzer2022betanll,
  title={On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks},
  author={Seitzer, Maximilian and Tavakoli, Arash and Antic, Dimitrije and Martius, Georg},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

---

## C. Baseline 方法引用

### 9. DLinear — 线性时序预测
- **标题**: Are Transformers Effective for Time Series Forecasting?
- **作者**: Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu
- **会议**: AAAI 2023
- **arXiv**: 2205.13504
- **引用理由**: Baseline B5 (DLinear) 的方法来源
- **BibTeX**:
```bibtex
@inproceedings{zeng2023dlinear,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  pages={11121--11128},
  year={2023}
}
```

### 10. Mamba — 选择性状态空间模型
- **标题**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **作者**: Albert Gu, Tri Dao
- **arXiv**: 2312.00752
- **引用理由**: Baseline B6 (Mamba) 的方法来源；论文中标注 aarch64 环境限制
- **BibTeX**:
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

### 11. N4SID — 线性子空间辨识
- **标题**: N4SID: Subspace Algorithms for the Identification of Combined Deterministic-Stochastic Systems
- **作者**: Peter Van Overschee, Bart De Moor
- **期刊**: Automatica, Vol. 30, No. 1, pp. 75-93, 1994
- **DOI**: 10.1016/0005-1098(94)90230-5
- **引用理由**: 线性 SSM baseline (exp_020) 的方法来源；证明线性模型长程发散 → 非线性 WM 必要性
- **BibTeX**:
```bibtex
@article{vanoverschee1994n4sid,
  title={N4SID: Subspace Algorithms for the Identification of Combined Deterministic-Stochastic Systems},
  author={Van Overschee, Peter and De Moor, Bart},
  journal={Automatica},
  volume={30},
  number={1},
  pages={75--93},
  year={1994}
}
```

---

## D. 项目内部参考 (已在 references.md 中)

### 12. Differentiable World Model for Offline RL
- **标题**: Differentiable World Model for Offline RL
- **arXiv**: 2603.22430 (2026)
- **引用理由**: 可微世界模型 + 梯度 MPC 的先驱工作；本项目方法论的直接参考
- **注**: 扩散模型作为 transition model，通过梯度反传做 MPC 优化

### 13. Graph Spatiotemporal World Model Rolling MPC
- **标题**: Graph Spatiotemporal World Model Rolling MPC
- **作者**: Junling Liu, Xiaojun Wang, Leilei Wang, Yu Song
- **期刊**: Electronics (2026)
- **引用理由**: 历史窗口 + 自回归展开训练 + 物理一致性约束 + MPC 显式嵌入的框架参考

---

## E. Phase 2 叙事重构新增引用 (objective mismatch + 工业WM定位)

### 14. Objective Mismatch in Model-based RL
- **标题**: Objective Mismatch in Model-based Reinforcement Learning
- **作者**: Alex Lambert, Amos Storkey, Joao Messias
- **会议**: ICML 2020 Workshop on Object-Oriented Learning
- **arXiv**: 2102.03023
- **引用理由**: §IV/V 核心理论框架 — 预测精度 ≠ 控制效用; 世界模型训练目标 (一步预测似然) 与下游目标 (闭环控制性能) 不相关
- **BibTeX**:
```bibtex
@inproceedings{lambert2020mismatch,
  title={Objective Mismatch in Model-based Reinforcement Learning},
  author={Lambert, Alex and Storkey, Amos and Messias, Joao},
  booktitle={ICML Workshop on Object-Oriented Learning},
  year={2020}
}
```

### 15. Predicting Closed-Loop Performance of Latent World Models
- **标题**: Predicting Closed-Loop Performance of Latent World Models: Offline Checkpoint Selection for MPC and Model-Based RL Under Non-Markovian Rewards in LunarLander
- **arXiv**: 2607.01736 (2026)
- **引用理由**: §V — 开环预测精度与闭环控制性能脱节的实证; validation loss 持续改善但闭环性能已崩溃
- **BibTeX**:
```bibtex
@article{2026closedloop,
  title={Predicting Closed-Loop Performance of Latent World Models},
  journal={arXiv preprint arXiv:2607.01736},
  year={2026}
}
```

### 16. Closing the Train-Test Gap in World Models
- **标题**: Closing the Train-Test Gap in World Models for Gradient-Based Planning
- **arXiv**: 2512.09929 (2025)
- **引用理由**: §V — 训练数据 (运行员自然轨迹) vs 规划动作分布的差异; compounding error on out-of-distribution trajectories
- **BibTeX**:
```bibtex
@article{2025traintest,
  title={Closing the Train-Test Gap in World Models for Gradient-Based Planning},
  journal={arXiv preprint arXiv:2512.09929},
  year={2025}
}
```

### 17. Actionable World Models for Industrial Process Control
- **标题**: Learning Actionable World Models for Industrial Process Control
- **会议**: IEEE SDS 2025
- **arXiv**: 2503.01411
- **引用理由**: §VI — 工业世界模型做预测+建议而非闭环控制的范式; JEPA + contrastive learning + disentangled representations; 注塑成型案例
- **BibTeX**:
```bibtex
@inproceedings{2025actionable,
  title={Learning Actionable World Models for Industrial Process Control},
  booktitle={IEEE International Conference on System, Design and Service},
  year={2025},
  note={arXiv:2503.01411}
}
```

### 18. How Should World Models Be Evaluated for Embodied Decision-Making
- **标题**: How Should World Models Be Evaluated for Embodied Decision-Making? A Decision-Making-Centric Position
- **arXiv**: 2606.15032 (2026)
- **引用理由**: §V — L0-L7 评估阶梯 (视觉合理性→策略优化效用); 开环 vs 闭环评估; interventional action fidelity
- **BibTeX**:
```bibtex
@article{2026eval,
  title={How Should World Models Be Evaluated for Embodied Decision-Making?},
  journal={arXiv preprint arXiv:2606.15032},
  year={2026}
}
```

### 19. Predictive but Not Plannable: RC-aux
- **标题**: Predictive but Not Plannable: RC-aux for Latent World Models
- **arXiv**: 2605.07278 (2026)
- **引用理由**: §V — 预测准确 ≠ 可规划; 时空 mismatch (短程训练 vs 长程搜索); reachability-correction auxiliary objective
- **BibTeX**:
```bibtex
@article{2026rcaux,
  title={Predictive but Not Plannable: RC-aux for Latent World Models},
  journal={arXiv preprint arXiv:2605.07278},
  year={2026}
}
```

### 20. Imagined Rollouts are Kinematic, Not Dynamic
- **标题**: Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure
- **arXiv**: 2607.05966 (2026)
- **引用理由**: §V — WM rollout 是运动学 (kinematic) 而非动力学 (dynamic); 长程 rollout 失败诊断; 摩擦不变性实验
- **BibTeX**:
```bibtex
@article{2026kinematic,
  title={Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure},
  journal={arXiv preprint arXiv:2607.05966},
  year={2026}
}
```

---

## F. 引用与 Phase 1 结论的对应关系

| Phase 1 结论 | 对应引用 | 消融/实验 |
|---|---|---|
| RevIN 不可或缺 | Kim et al. 2022 (RevIN) | M6 (−RevIN): MAE ×70 |
| β-NLL > MSE | Seitzer et al. 2022 (β-NLL) | exp_005/006/007 |
| Patch 嵌入有效 | Nie et al. 2023 (PatchTST) | M2 (−patch) |
| 单目标合理 | Hansen et al. 2024 (TD-MPC2) | 40列 vs 13列 |
| 线性模型长程发散 | Van Overschee & De Moor 1994 (N4SID) | exp_020 |
| DLinear baseline | Zeng et al. 2023 (DLinear) | B5 |
| Mamba baseline | Gu & Dao 2023 (Mamba) | B6 |
| iTransformer baseline | Liu et al. 2024 (iTransformer) | B4 |
| PETS 集成方法 | Chua et al. 2018 (PETS) | Phase 2: 3×M7+CEM |
| 世界模型通用性 | Hafner et al. 2025 (DreamerV3) | 背景参考 |
| 可微 MPC 框架 | Drgona et al. 2023 (Neuromancer) | 方法论参考 |
| 预测精度≠控制效用 | Lambert et al. 2020 (Objective Mismatch) | §IV/V: S3 因果反演 + 闭环不显著 |
| 开环≠闭环性能 | arXiv 2607.01736 (Closed-Loop Perf) | §V: validation loss 持续改善但闭环崩溃 |
| 训练vs规划分布差异 | arXiv 2512.09929 (Train-Test Gap) | §V: 小幅动作正确, 大幅阶跃反演 |
| 工业WM预测+建议范式 | arXiv 2503.01411 (Actionable WM) | §VI: 监督模式定位 |
| WM评估阶梯 | arXiv 2606.15032 (WM Evaluation) | §V: L0-L7, 开环vs闭环 |
| 预测≠可规划 | arXiv 2605.07278 (RC-aux) | §V: 时空mismatch |
| WM rollout≠动力学 | arXiv 2607.05966 (Kinematic) | §V: 长程rollout失败 |
