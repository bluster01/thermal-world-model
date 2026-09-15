# FMTS-BLOCK1 执行状态

2026-09-15：按作者选择注册180秒完整预测块回填，累计1,200秒。实现与工程验证完成；**Linux 已于 2026-09-15 完成真实执行，见下方回执**。

作者已授权push并由Linux执行本包。仅发布本次注册、代码、测试和必要状态；9单元推理与完整重放可在Linux取得已发布版本后开始，不新增训练。Linux 领取与运行回执见下方小节。

发布回执：`ef1c4eda783ff1f0fb6dcaf958e478f19bb441e2` 已成功推送到 `origin/main`，并通过远端ref核验。状态为 `queued_for_linux`（发布时的状态）。执行入口见本目录 `RUN_LINUX.md`。

- 已有9个checkpoint可严格重建；本地（发布侧）未运行真实数据推理、未训练。
- 真实数据由Linux按 `RUN_LINUX.md` 执行并完整重放，回执见下方。
- 私有真实历史/轨迹已就绪、待私有渠道交付（不进入Git）；公共汇总回执已返回（见下）。
- 现有H18预测/双阀响应证据、正文和插图保持不变；不扩大test或实验搜索范围。

## Linux execution return (2026-09-15) — author audit pending

Claimed via `75fc29e` (source `8f1b6a4`); executed on GB10/CUDA (torch 2.11.0+cu130) against the frozen record (`24da7796…`) and indices (`eea161e3…`); no training, zero updates, locked test not evaluated.

- **9/9 fixed-weight inference cells complete** (`blackbox_itransformer` / `fusion_gru_norew` / `greybox_steady_none_norew` × seeds 0/1/2), 180 s complete-block feedback out to 120 steps (1,200 s).
- Common eligibility: **256/256 original validation windows eligible** (0 exclusions; out_of_bounds / non_validation / invalid_base / time_gap / invalid_history_extension all 0); 13 days; fixed cases row138/row203 both eligible.
- Explicit audit: `complete=true`, `cells_replayed=9` (real-record full replay), `artifact_checks_passed=true`, failures=0. Public receipt: `results/fmts_block_rollout_20260915/linux_block1_receipt/` (aggregate metrics + hashes + audit status only; no time series; receipt sha256 `633da808…`).
- Private traces (identity, eligibility, 256-window real history, observed 120-step targets, 9× per-cell predictions/reports, summary, audit, manifest; 26 files ≈1.2 MB) staged outside Git at `/home/bluster/fmts_block1_private_20260915` (zip staged for private handoff). **Not returned via Git by design**; delivery pending the author's private channel.

### Descriptive outcomes (3-seed mean, equal-day weighted cumulative MAE °C; value at step k = mean over horizons 1..k; author audit pending; NOT plant truth)

| arm | H18 | H36 | H60 | H120 |
|---|---:|---:|---:|---:|
| blackbox_itransformer | 0.3706 | 0.6056 | 0.8371 | 1.1674 |
| fusion_gru_norew | 0.4431 | 0.7405 | 1.0632 | 1.5729 |
| greybox_steady_none_norew | 0.9734 | 1.4925 | 2.0258 | 2.8845 |
| persistence (last measured value) | 0.6388 | 1.0116 | 1.2474 | 1.4926 |

H18 anchors reproduce the pinned values (GRU 0.443139, GNR 0.973432); the original-H18 replay gate passed for all 9 cells. Long-horizon figures are descriptive only: blackbox lowest error at all reported horizons; GRU at 1,200 s slightly above persistence and with the largest seed spread (sd@120 ≈ 0.121 °C); greybox highest. These are block-wise recursive forecasts (180 s block boundaries; only main steam temperature fed back; other temperatures/actions/boundaries remain recorded context) — not an autonomous open-loop simulation and not a native H60.

Status: `public_receipt_returned=true`; `private_traces_returned=false` (staged, pending private channel); `results_returned=false` until both parts reach the author side; `audited=false` until the author's return review. No paper/main-text/figure changes; no new training, search, test or extension scoring.
