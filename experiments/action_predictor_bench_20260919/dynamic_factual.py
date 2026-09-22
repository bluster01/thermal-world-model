"""Describe load-changing report windows using saved forecasts; no fitting.

Future MW only defines post-hoc groups. Forecasts replay recorded future valve
and coal/steam boundary inputs; they are not driven directly by future MW.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/action_predictor_bench_20260919"
GROUPS = ("rise", "fall", "vary", "steady")
ZH = dict(rise="升负荷", fall="降负荷", vary="变化/反转", steady="近稳")


def classify_load(load_mw):
    """Return ordered rise/fall/vary/steady labels for [N,64+>=128] MW.

    Net change compares history's final six points with H128's final six;
    |change| >= 10 MW has priority over future-H128 range >= 20 MW.
    """
    load = np.asarray(load_mw, dtype=np.float64)
    if load.ndim != 2 or load.shape[1] < 192:
        raise ValueError("load_mw must be [windows, >=192] with 64 history steps")
    if not np.isfinite(load[:, :192]).all():
        raise ValueError("Load grouping requires finite history and H128 load")
    change = load[:, 186:192].mean(1) - load[:, 58:64].mean(1)
    span = np.ptp(load[:, 64:192], axis=1)
    labels = np.full(len(load), "steady", dtype="U6")
    labels[span >= 20] = "vary"
    labels[change >= 10] = "rise"
    labels[change <= -10] = "fall"
    return labels


def select_probe_windows(labels, n=8):
    """Choose up to n evenly spaced original-order indices per group."""
    if n < 1:
        raise ValueError("n must be positive")
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, GROUPS).all():
        raise ValueError("Unexpected load labels")
    return {group: indices[np.linspace(0, len(indices) - 1,
            min(n, len(indices)), dtype=int)].tolist() if len(indices) else []
            for group in GROUPS for indices in [np.flatnonzero(labels == group)]}


def metrics(prediction, truth, history, indices):
    pred = np.asarray(prediction, dtype=np.float64)[indices, :, 4]
    true = np.asarray(truth, dtype=np.float64)[indices, :, 4]
    result = {"windows": len(indices)}
    for horizon in (32, 128, 512):
        valid = np.isfinite(pred[:, :horizon]).all(1) & np.isfinite(true[:, :horizon]).all(1)
        result[f"H{horizon}_valid_windows"] = int(valid.sum())
        result[f"H{horizon}_mae_C"] = (float(np.abs(pred[valid, :horizon]
            - true[valid, :horizon]).mean()) if valid.any() else None)
    # Use only future-to-future increments. A constant predictor must have zero
    # amplitude and undefined correlation, not inherit dynamics from history.
    dp = pred[:, 6:128] - pred[:, :122]
    dy = true[:, 6:128] - true[:, :122]
    valid = np.isfinite(dp) & np.isfinite(dy)
    a, b = dp[valid], dy[valid]
    result["increment60_valid_pairs"] = int(valid.sum())
    result["increment60_valid_windows"] = int(valid.all(1).sum())
    result["increment60_mae_C"] = float(np.abs(a - b).mean()) if len(a) else None
    result["increment60_corr"] = (float(np.corrcoef(a, b)[0, 1])
        if len(a) > 1 and np.std(a) > 1e-12 and np.std(b) > 1e-12 else None)
    result["increment60_amplitude_ratio"] = (float(np.sqrt(np.mean(a*a) / np.mean(b*b)))
        if len(a) and np.mean(b*b) > 1e-12 else None)
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representative_cases(labels, change):
    """Choose median-magnitude and maximum-magnitude load cases, never by MAE."""
    cases = []
    for group in ("rise", "fall"):
        indices = np.flatnonzero(labels == group)
        if not len(indices):
            continue
        ordered = indices[np.argsort(np.abs(change[indices]), kind="stable")]
        for tag, index in (("median", ordered[(len(ordered) - 1) // 2]),
                           ("largest", ordered[-1])):
            cases.append({"group": group, "choice": tag, "index": int(index),
                          "delta_MW": float(change[index])})
    return cases


def plots(folder, load, bank, arrays, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    curves = [
        ("Truth", None, "#161616", "solid"),
        ("SSM B block", "focused_B_short/block", "#1675bf", "solid"),
        ("SSM B native", "focused_B_short/native", "#1675bf", "dash"),
        ("SSM A block", "focused_A_short/block", "#809eb1", "solid"),
        ("D block (frozen response)", "focused_D_short/block", "#ae62be", "solid"),
        ("P0 continuous", "physical_P0_short/block", "#d95b24", "solid"),
    ]
    horizon = 256
    xx = np.arange(-63, horizon + 1) / 6
    fig, axes = plt.subplots(4, 2, figsize=(16, 11),
        gridspec_kw={"height_ratios": [1, 2, 1, 2]}, sharex="col")
    interactive = make_subplots(rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=.09, row_heights=[.28, .72])
    buttons = []
    traces_per_case = 1 + len(curves)
    for case_number, case in enumerate(cases):
        index = case["index"]
        rr, cc = 2 * (case_number // 2), case_number % 2
        title = f'{case["group"]} {case["choice"]}: window {index}, load change {case["delta_MW"]:+.1f} MW'
        axes[rr, cc].plot(xx, load[index, :64+horizon], color="#5b8c37", lw=1.7)
        axes[rr, cc].set_title(title, fontsize=11)
        axes[rr, cc].set_ylabel("Load (MW)")
        interactive.add_trace(go.Scatter(x=xx, y=load[index, :64+horizon],
            name="Recorded load (grouping only)", line=dict(color="#5b8c37"),
            visible=case_number == 0), row=1, col=1)
        for name, key, color, dash in curves:
            predicted = bank[index, 64:64+horizon, 4] if key is None else arrays[key][index, :horizon, 4]
            yy = np.concatenate([bank[index, :64, 4], predicted])
            axes[rr+1, cc].plot(xx, yy, label=name, color=color,
                linestyle="--" if dash == "dash" else "-", lw=1.4 if key else 2)
            interactive.add_trace(go.Scatter(x=xx, y=yy, name=name,
                line=dict(color=color, dash=dash, width=2 if key else 3),
                visible=case_number == 0), row=2, col=1)
        for axis in axes[rr:rr+2, cc]:
            axis.axvline(0, color="#555", linestyle=":", linewidth=1)
            axis.axvline(128/6, color="#999", linestyle=":", linewidth=.8)
            axis.grid(alpha=.18)
        axes[rr+1, cc].set_ylabel("Main steam temperature (C)")
        axes[rr+1, cc].set_xlabel("Minutes from forecast origin")
        visible = [False] * (len(cases) * traces_per_case)
        visible[case_number*traces_per_case:(case_number+1)*traces_per_case] = [True]*traces_per_case
        buttons.append(dict(label=title, method="update",
            args=[{"visible": visible}, {"title": title}]))
    axes[1, 0].legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle("Recorded operating dynamics: fixed load-based examples (no refitting)", fontsize=15)
    fig.text(.5, .01, "Forecasts replay recorded future valves and coal/steam boundaries; future MW only labels cases. "
             "Dotted lines: origin / H128 (21.3 min).", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, .965))
    fig.savefig(folder / "factual_cases.png", dpi=160)
    plt.close(fig)
    interactive.update_layout(height=790, template="plotly_white",
        title=buttons[0]["label"], margin=dict(t=145, b=75),
        updatemenus=[dict(buttons=buttons, x=0, y=1.18, xanchor="left")],
        legend=dict(orientation="h", y=-.14),
        annotations=[dict(x=0, y=1.10, xref="paper", yref="paper", showarrow=False,
            xanchor="left", text="Recorded future valves + coal/steam boundaries; future MW only labels cases. Click legend to hide/show models.")])
    interactive.update_xaxes(title_text="Minutes from forecast origin", row=2, col=1)
    interactive.update_yaxes(title_text="Load (MW)", row=1, col=1)
    interactive.update_yaxes(title_text="Main steam temperature (C)", row=2, col=1)
    interactive.add_vline(x=0, line_dash="dot", line_color="#555")
    interactive.add_vline(x=128/6, line_dash="dot", line_color="#999")
    interactive.write_html(folder / "factual_cases.html", include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "ssm_dynamic_20260922")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.output / "load_context.npz") as source:
        load, times = source["load_MW"], source["times"]
    with np.load(RESULTS / "focused33_seed11/evaluation_inputs.npz") as source:
        bank = source["bank"]
        np.testing.assert_array_equal(times, source["times"])
    if not (np.diff(times) >= 0).all():
        raise ValueError("Original report windows must be in chronological order")
    labels = classify_load(load)
    probes = select_probe_windows(labels)
    change = load[:, 186:192].mean(1) - load[:, 58:64].mean(1)
    span = np.ptp(load[:, 64:192], axis=1)
    cases = representative_cases(labels, change)
    np.savez_compressed(args.output / "factual_groups.npz", labels=labels,
        delta_load_MW=change, future_H128_range_MW=span, times=times,
        **{f"probe_{k}": np.asarray(v, dtype=np.int64) for k, v in probes.items()})
    rows, arrays, sources = {}, {}, []
    runs = [("focused", "focused33_seed11", ["A_short", "B_short", "D_short"]),
            ("physical", "physical33_seed11_to90", ["P0_short", "P1_short", "P2_short"]),
            ("full", "full33_seed11", None)]
    for prefix, run, requested in runs:
        root = RESULTS / run
        with np.load(root / "evaluation_inputs.npz") as source:
            np.testing.assert_array_equal(times, source["times"])
            np.testing.assert_array_equal(bank[:, :, :13], source["bank"][:, :, :13])
        names = requested or sorted(p.name for p in (root / "seed11").iterdir()
                                   if p.name.endswith("_short") or p.name == "persistence")
        for name in names:
            path = root / "seed11" / name / "forecasts.npz"
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
            with np.load(path) as source:
                for mode in ("block", "native"):
                    if f"{mode}_recorded" not in source:
                        continue
                    prediction = source[f"{mode}_recorded"]
                    if prediction.shape != (len(bank), 512, 5):
                        raise ValueError(f"Unexpected forecast shape: {path}")
                    key = f"{prefix}_{name}/{mode}"
                    rows[key] = {group: metrics(prediction, bank[:, 64:, :5], bank[:, :64, :5], indices)
                        for group, indices in [("all", np.arange(len(bank))),
                            *((g, np.flatnonzero(labels == g)) for g in GROUPS)]}
                    if prefix != "full":
                        arrays[key] = prediction
    counts = {g: int((labels == g).sum()) for g in GROUPS}
    report = {"protocol": "posthoc_recorded_input_replay", "seed": 11,
        "notes": ["Grouping uses recorded future MW only; no training or model inputs changed.",
                  "All groups use the same 256 report windows and 512-step predictions.",
                  "MAE is main temperature, channel 4, degrees C.",
                  "60s increments use a six-step lag and 122 pure-future pairs within H128; no history-to-future pairs.",
                  "Increment correlation pools valid window/time pairs; amplitude ratio is RMS(predicted increments)/RMS(true increments).",
                  "Windows can overlap: valid-pair counts are not independent samples.",
                  "All predictions replay recorded future valves and boundary inputs, not future MW as an independent control.",
                  "Physical block/native use identical continuous-state trajectories; these are not independent protocols.",
                  "One seed and post-hoc load groups describe performance; they do not identify causal load responses."],
        "groups": counts, "group_definition": "last-six H128 MW minus last-six history MW >=10 rise, <=-10 fall; otherwise future H128 range >=20 vary; else steady",
        "probe_windows": probes, "plot_cases": cases, "forecast_sources": sources,
        "load_context_sha256": sha256(args.output / "load_context.npz"), "metrics": rows}
    (args.output / "factual_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 真实负荷变化期间的已保存预测", "", "仅重算已有预测，无训练或选模。按未来真实 MW 对报告窗口做描述分组；模型使用真实未来阀门、煤量、汽量等边界回放，未来 MW 并未直接作为独立驱动输入。", "",
        "分组：最后 6 个历史点的 MW 均值与 H128 最后 6 点均值比较，≥+10 MW 为升、≤−10 MW 为降；余下未来 H128 极差≥20 MW 为变化/反转，其余近稳。H128=21.3 分钟，H512=85.3 分钟。", "",
        " | 分组 | 窗口数 |", "|---|---:|", *[f"| {ZH[g]} | {counts[g]} |" for g in GROUPS], "",
        "60 秒增量采用间隔 6 步的温度差，只计算 H128 内 122 个纯未来差值，不混入历史跨预测起点的差值。幅度比=预测增量 RMS / 真实增量 RMS；相关系数汇总全部有效位置，不把重叠窗口当独立样本。持值基线的增量幅度为 0，相关系数未定义。", "",
        "曲线按升/降负荷幅度的中位和最大窗口固定选取，没有按误差筛选。PNG 为静态总览；HTML 可切换案例、点击图例开关模型。未来 H128 仅用于分组，H256 曲线后半段可能发生新的负荷方向变化。", ""]
    for group in ("all", *GROUPS):
        lines += [f"## {ZH.get(group, '全部窗口')}", "",
            "| 模型/模式 | n | H32 MAE | H128 MAE | H512 MAE | 60s Δ MAE | Δ corr | Δ 幅度比 | 有效 Δ 对 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, groups in rows.items():
            m = groups[group]
            values = [m[k] for k in ("H32_mae_C", "H128_mae_C", "H512_mae_C",
                "increment60_mae_C", "increment60_corr", "increment60_amplitude_ratio")]
            lines.append(f"| {name} | {m['H128_valid_windows']} | " + " | ".join(
                "—" if value is None else f"{value:.4f}" for value in values)
                + f" | {m['increment60_valid_pairs']} |")
        lines.append("")
    lines += ["物理三臂的 block/native 是同一连续状态递推；不能当成两个独立实验。这里能看出动态预测的形状与误差，但不能由真实运行轨迹直接证明固定其他条件下的负荷因果响应。"]
    (args.output / "factual_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plots(args.output, load, bank, arrays, cases)
    print(json.dumps({"counts": counts, "probe_windows": probes, "plot_cases": cases,
                      "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
