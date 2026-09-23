# 三任务统一模型首轮实验

实验标识：`unified_industrial_stage1_20260923`。首次发布时代码与执行协议已备好，**尚无 Linux 正式训练回执**。实际进度以本批领取回执和 Linux worker 生成的 `public_status.json` 为准。执行入口见 [RUN_LINUX.md](RUN_LINUX.md)，冻结参数以 [protocol.json](protocol.json) 为准。

本轮比较同一建模方式在 SCR 出入口、主汽温测点和再热汽温测点上的预测表现，并用慢状态消融检验储存记忆的作用。三个任务分别训练各自权重，共享架构、输入规则和训练协议；跨任务权重迁移留给后续独立实验。

## 训练矩阵

| 维度 | 配置 |
|---|---|
| 任务 | `scr`、`main_steam`、`reheat_steam` |
| 模型 | `gru_direct`、`ssm_rollout`、`itransformer`、`mechanism`、`mechanism_no_slow` |
| 随机种子 | 11、22、33 |
| 正式拟合数 | 3 × 5 × 3 = **45** |
| 历史输入 | 360 步，每步 10 秒，共 60 分钟 |
| 训练未来 | 60 步，共 10 分钟 |
| 延展评价 | 相同权重预测 180 步，共 30 分钟 |
| 执行平台 | 远端 Linux、CUDA；每次一个拟合 |

全部模型只使用截止起点的历史值、有效性、是否新到达和观测年龄。模型不读取未来测量阀位、边界、标签或标签掩码。监督标签只进入损失与评价。每路输出锚定自身最近有效历史值；全历史缺测时使用该目标训练中心，即标准化后的零。

H180 是 H60 训练后的延展评价。状态模型继续递推；直接模型使用训练时已有的连续 lead-query 读出，lead 始终按 `t / 60` 定义。第 61—180 步没有接受 H180 标签训练，也没有把真实未来观测回填进历史。

## 五个模型的实际实现

| 模型 | 历史编码与未来计算 |
|---|---|
| `gru_direct` | 6 步固定均值池化后的 GRU 观察器，加连续 lead-query 多步读出 |
| `ssm_rollout` | 同类 GRU 观察器，加具有正时间常数的对角状态递推；外源驱动由历史预测 |
| `itransformer` | 每变量的完整值、掩码、到达标记和年龄历史形成一个 token；两层变量注意力与连续 lead-query 读出；属于本地实现 |
| `mechanism` | GRU 恢复初态和条件参数；显式过程状态、每目标 3 个慢储存、外端口和传感器状态递推；输出只读取传感器状态 |
| `mechanism_no_slow` | 保留相同观察器、外端口和传感器，移除慢储存及对应交换 |

GRU 类模型的全部 360 步信息进入固定池化，最近有效目标锚点从未池化历史提取。`ssm_rollout` 的 SSM 位于未来解码器；其历史编码器仍是 GRU。`itransformer` 用于倒置变量注意力基线比较，没有使用官方 checkpoint，也未声明完成官方实验复现。各模型参数量会逐拟合记录。

热任务当前采用**每个目标自己的快状态—慢储存星形交换结构**。快慢状态在共同温度坐标中进行成对正交换，外部松弛端口提供热源和热汇，传感器具有正时间常数。节点之间通过共享历史观察器获得条件信息，尚未实现整条主汽或再热物理输运链；没有按测点名称强加再热连接顺序。容量为等效容量，内部收支属于这个明确的等效状态系统。

SCR 当前使用非负 NO 浓度代理。源、移除及同侧入口至出口传输均显式建模；反应移除允许 NO 浓度下降，因此四个测点的浓度之和不被当作守恒量。六组还原剂历史进入共同观察器，组别到 A/B 侧的未知路由没有被硬编码。浓度保持源数据量纲，不擅自转换为 ppm 或总 NOx。

## 训练与评价

45 个拟合使用全部合格训练起点。优化器为 AdamW，batch size 128，初始学习率 0.001，梯度裁剪 1.0。训练目标为逐目标等权的掩码标准化 MSE，并统一加入 lag 1、6、18 步增量损失，权重 0.2。标准化统计与数据切分随数据身份冻结。

每 epoch 在同一固定、按时间均匀取出的验证子集上选择模型，最多 4096 个起点。最多 80 epoch，至少 20 epoch，12 个 epoch 无有效改进时提前停止；4 个 epoch 平台期触发学习率下降。到达 epoch 上限记录为 `BUDGET_EXHAUSTED`，保留完整曲线。

全部 45 个拟合完成后，统一加载各自验证最佳权重，评价完整验证集和历史测试集的 H60/H180。历史测试不参与模型选择；已保存且身份核验通过的最终评价不重复计算。逐目标保存 MAE、RMSE、一步增量 MAE、指定端点误差、有效标签数及持续值基线。不同任务的物理量不直接混合求均值。运行器自动生成私有 `summary.json` 和 `SUMMARY_ZH.md`，报告三种子 mean ± sample std、逐目标指标和有慢状态相对无慢状态的配对差。

评价 NPZ 保存所有起点的标签计数、标准化平方误差和、物理尺度绝对误差和、持续值误差和，以及增量误差和与计数，支持后续同起点分析；另保存前三个 batch 的完整轨迹预览。逐起点统计覆盖所有评价窗口，完整轨迹预览只覆盖其标明的批次。

## 文件与执行边界

- [models.py](models.py)：模型和机理算子。
- [run.py](run.py)：Linux CUDA 预运行、45 次拟合、最终统一评价和续训。
- [prepare_linux.py](prepare_linux.py)：按源文件字节身份重建或核验私有数据包。
- [data_recipe.json](data_recipe.json)、[expected_data.json](expected_data.json)：公开的别名、规则与身份哈希。
- [requirements.txt](requirements.txt)：隔离环境依赖；同一矩阵及续训保留相同实际运行时版本。
- [test_models.py](test_models.py)、[test_runner.py](test_runner.py)、[test_prepare_linux.py](test_prepare_linux.py)、[test_summarize.py](test_summarize.py)、[test_execution.py](test_execution.py)：软件检查；可以在 CPU 上执行，不代表工业训练结果。
- `release_manifest.json`：本次发布源码清单，由发布步骤生成；worker 必须使用对应 release commit。

原始 CSV、私有数据包、数据路径、详细日志、真实轨迹、逐项指标和训练权重保留在 Linux 私有目录，目录必须位于所有 Git checkout 外。GitHub 传递公开代码、冻结配置及允许的领取/进度回执；Linux 仅向 `results/unified_industrial_stage1_20260923/` 的 `claim.json`、`public_status.json`、`return_receipt.json` 回传状态，通过 `origin/main` 普通 fast-forward 提交。首次领取和启动必须由远端 worker 实际完成；发布执行合同不等于实验已经运行。
