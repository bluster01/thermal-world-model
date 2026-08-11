# Phase 3.5-MS3 远端执行事故审计

> 日期：2026-08-11；性质：运行时兼容性缺陷，不是科学结果。

## 1. 已核实事实

- Linux commit `054c570` 使用冻结数据 SHA `85a3f926...e4da6`，preflight 为 171/171 tests、compile 与 12-run dry-run 全绿。
- cache builder 在 pandas 3.0.2 中把 `datetime64[us, UTC]` 直接 `astype(int64)`；代码却将整数解释为纳秒。
- 因而真实 10 s 间隔被读取为 0.01 s，1,192,328/1,192,328 transitions 全被判为 irregular，train/validation anchors 均为 0。
- runner 在第 1/12 个 run、首个 optimizer update 前抛错；没有 checkpoint、history、metrics、summary 或 test artifact，也没有 test 访问。
- 把远端 diff 按微秒解释后，得到 279×20 s、120 s、180 s、75,750 s，与本地独立扫描的 282 个异常间隔完全一致。因此没有证据指向源文件漂移或交叉 side mapping 错误。

## 2. 根因与影响边界

根因是 pandas 2/3 对 `to_datetime(..., utc=True)` 输出分辨率的默认行为不同，而项目在整数化前没有显式单位转换。旧 cache 的字段名虽为 `timestamps_ns`，实际值为微秒，必须整体作废并覆盖。

该缺陷只阻断样本窗口构造。由于训练尚未开始，它不改变或否定 MS5、MS3 模型架构、损失、split、seed、预算、指标和门禁，也不能被解释成 MS3 科学失败。

## 3. v1.1 修复合同

1. 所有 Phase3.5 CSV 时间戳统一经过 UTC 解析，再显式转换为 `datetime64[ns]` 后整数化；
2. MS3 matrix 明确冻结 `timestamp_storage_unit=ns`；
3. cache builder 在写盘前核对 1,192,329 行、起止纳秒、282 个异常 transition 和 75,750 s 最大缺口；
4. runner 同时核对上述 manifest 字段并从 cache 数组重算时间线，旧 v1 cache 或只改 metadata 的伪修复会在训练前 fail closed；
5. 增加以 `datetime64[us]` 为输入的跨版本回归测试，直接要求 10 s=`10,000,000,000 ns`。

协议标记升为 `phase3.5-ms3-v1.1`。这是数据表示修复，不是允许修改科学合同的新实验版本。

## 4. 远端恢复边界

Linux 拉取 v1.1 后必须重新执行完整 preflight 和 cache builder；builder 会覆盖两个错误 cache 与 manifest。确认时间线字段完全匹配后，才可执行原冻结 12-run validation。不得复用旧 cache、补超参数、改阈值、访问 test 或启动 MS4。旧 `run_failure_note.txt` 保留作为事故证据。
