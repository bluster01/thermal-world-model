# Linux 侧预领取说明：合并表源偏离版数据包（状态与身份）
批次：`unified_industrial_stage1_20260923`（release `d14a5a80`）· 生成：2026-09-24 · Linux 执行侧
**性质：预领取（pre-claim）工作说明，不是回执；未领取、未训练、未改动任何冻结源码。**

## 背景
Linux 本机不持有本批公开发布的 80 个原生逐点导出（按别名文件名/尺寸/字节哈希核验 0/80 命中，
且厂内网不可达）。经作者指示，改以项目本地**合并 10s 表**为源（sha256 `85a3f92648d5f88a4543f500859b200207fb55a32555900ca88f7c339c4e4da6`），
用**未修改的冻结准备代码**重建了一份偏离版数据包用于口径决策。

## 方法
1. 别名↔源列映射：按 recipe `filename_sha256` 逐别名核验，**80/80 命中**（哈希级，零猜测）；
2. 抽取为 80 个单通道源文件；冻结 `prepare.py`（`preparation_code_sha256` 核验 ✅）+
   冻结 recipe 配置（日历/切分/任务/规则/跨步口径全部沿用发布值）重建；
3. 产出 1,200,210 × 80 数据包（留在 Linux 私有存储）。

## 身份核验（vs `expected_data.json`）
- ✅：`time_ns`、`split`、`tasks`、`grid_rows`、`aliases`、`reader_config`、冻结代码哈希、`normalization.active`
- ❌：`values`/`valid`/`observed`/`flags`/`source_time_ns`；`normalization` center/scale/fit_count；
  origins 计数（明细见 `identity.json`）；源哈希 0/80（预期——源非原生导出）

## 影响要点（供口径判断）
- train/validation/historical_test 窗口数：多数 **±0.3%–0.9%**；最大项 scr f10m train **+64%**；
- `*_complete` 子集数值膨胀（最大 ~63×）——**但 runner 全链路未使用 complete 子集**（读码确认，train/评估均用非 complete 集 + 逐目标逐步掩码）；
- `observed`/fresh 语义不可从合并表复原（无原生时标）：约 **1–2%** (目标,步) 计分对变化；
- 整行缺失 **7,881/1,200,210**（0.66%：一块 ~21 小时 + 281 个小块）；
- 数值抽验（以本机原生拉取为参照，非本批真源）：同源通道中位差为 float32 舍入级；
  个别通道存在点位/采样级差异（中位 ~0.3、极值 ~19，成因未定，建议复核）。

## 请求（Codex 决策）
1. **接受合并表口径** → 按 `identity.json` 中哈希出修订 `expected_data`（注明语义偏离）→ Linux 领取并执行 45 拟合；
2. **不接受** → 原生导出送达 Linux 后按冻结全流程执行；
3. 或给出其他重建规则/配方。

完整私有明细（点位名映射、路径、逐点抽验、脚本、日志、数据包本体）在 Linux 私有存储，可经私有通道提供。
