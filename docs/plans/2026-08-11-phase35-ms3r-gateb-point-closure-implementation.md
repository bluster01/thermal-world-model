# MS3-R Gate B 实施清单

1. 冻结 validation-only 配置、父 Gate-A 审计 pin、主 family 与远端资源合同。
2. 实现共享 rolling cross-fit、逐日 2×2 MIMO、配对 bootstrap 与 common/differential 变换。
3. 实现工况不变性和 SP-IV feasibility；二者只输出诊断。
4. runner 写出完整 JSON、可独立重放 NPZ 与 SHA256 ledger；科学判决字段固定为 `null`。
5. replay 脚本只读回传产物复算日矩阵、配对量和 interval，不读取 cache、不训练。
6. 用合成双输入双输出数据测试正确路径、错侧、lead、秩亏防护、IV first stage 与协议拒绝项。
7. 运行专项与 Phase3.5 全回归；仅标记 `local_verified`，Linux 授权保持为空。
