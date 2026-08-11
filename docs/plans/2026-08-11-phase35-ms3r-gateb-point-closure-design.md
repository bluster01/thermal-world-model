# MS3-R Gate B：点位路径闭合与 IV 可行性设计

## 目的与边界

Gate A 已说明双阀创新的代数秩不是当前首要瓶颈，但没有闭合路径特异性：错侧和上游 placebo 非零，A 回路在长时程存在 lead 污染，末温无法支持侧别归因。Gate B 因而是正式模型训练前最后一批点位实验，不训练世界模型，也不访问 test。

本批最多支持“历史与工况条件下的局部响应路径”。它不能证明开环 plant、有效喷水流量、任意 `do(valve)` 或有效工具变量。

## 冻结主问题

独立统计单位为 UTC 日。对每个短时程 `h∈{60,180}s`，同时残差化双阀创新 `U=[u_A,u_B]` 与双侧局部温降变化 `Y=[Δ(Tin−Tout)_A,Δ(Tin−Tout)_B]`，逐日估计 2×2 MIMO 响应矩阵

\[
\widehat B_{d,h}=(U_d^TU_d+\lambda I)^{-1}U_d^TY_d.
\]

行表示 A/B 阀门创新，列表示 A/B 局部温降。正对角线是预期正确路径，非对角线是错侧路径。

每个 UTC 日先对 60/180 s 两个矩阵取平均，再构造两个预注册的逐日配对对比：

\[
C^{spec}_{d,s}=B_{d,s,s}-|B_{d,s,\bar s}|,
\]

\[
C^{time}_{d,s}=B^{future}_{d,s,s}-|B^{lead}_{d,s,s}|.
\]

分别对 A/B 两个侧别组成两个 family。每个 family 用 Bonferroni 同时 97.5% bootstrap 区间控制 FWER=0.05；禁止比较两条分别计算的 95% CI。两个 family 的两侧下界均大于 0，且每侧至少 8 个 UTC 日，才允许 Supervisor 将点位闭合判为通过。代码不自动晋级。

## 次要诊断

- 保存 H60/H180/H300/H600 的逐日 MIMO future/lead 矩阵；H300/H600 不参与主门。
- 转换到 common/differential 坐标，检查可支持子空间，但不据此声称独立 plant 通道。
- 按 rolling fold、负荷 tertile、基线阀位 tertile、煤量/负荷 tertile、opening/closing 分层。全部是异质性诊断，不作数据驱动阈值筛选。
- 末温和上游 Tin 只作 downstream/placebo 图谱，不参与主门。

## SP-IV 可行性

IV 路线另用“不含当前 ΔSP”的历史预测器。工具变量为当前 ΔSP 的 cross-fit residual，内生动作是当前 Δvalve 的同口径 residual。报告 overall 与逐日 first stage、partial R²、另一阀门共动作、lead 和错侧 reduced form/2SLS。常规 `F>10` 不作为充分条件；这些诊断不能验证排除限制或把闭环 SP 自动升级为外生工具。

## 一次性远端合同

本地冻结配置、实现与合成测试后，Linux 才能获得一次批量授权。Linux 不改阈值、不补 seed、不写科学判决；用 `/usr/bin/time -v` 执行并完整返回日志、资源记录、JSON、NPZ 和 ledger。NPZ 必须含逐行 residual、逐日矩阵与逐日配对量，使 Supervisor 无需重建 cache 即可数值重放。缺件时只补传缺件，不重跑整批。

Gate B 审计完成后，才设计 measured-boundary latent MIMO 的正式模型消融：残差分支容量、局部中间监督、加法与工况调度喷水通道。Gate C/MS4 保持冻结。
