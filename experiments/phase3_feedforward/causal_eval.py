#!/usr/bin/env python3
"""
causal_eval.py — CFE 因果保真度评测共享模块 (2026-08-05)
=========================================================
设计稿: docs/causal_eval_framework.md

存在理由 (L0/P0.1): exp_096-102 中训练与评测的动作构造是两份代码, 语义不一致
(训练=一阶差分 ΔSP, 评测=二阶差分), 导致所有动作增益数字失效。
本模块提供**唯一**的动作构造函数, 训练与评测必须共用。

提供:
  build_action()        动作序列唯一来源 (训练/评测共用)
  select_events()       SP 阶跃事件筛选 (处理组)
  select_controls()     平稳段候选 (对照组)
  match_controls()      CEM 粗化精确匹配
  did_response()        L1 DiD 真值 + bootstrap CI + split-half ceiling
  model_response()      do-probe: pred(real a) − pred(a=0)
  causal_metrics()      L2 指标 SGN/GAIN/SHAPE/TTP/MONO
  ModelWrapper          统一 [B,H,1] 动作接口 (DirectWM 与 TimeXerWM 通用)
"""
import numpy as np
import torch

# ===== 事件筛选默认参数 (与 exp_099/100/101/102 一致, 便于回溯对比) =====
THR_DSP = 1.0      # 阶跃阈值 |ΔSP| > 1.0
GAP = 60           # 事件最小间隔
LOAD_STABLE = 3.0  # 负荷稳定: ±20 步内 max|Δload| ≤ 3.0
SP_HOLD = 0.3      # SP 保持: 61 步内 max|SP−SP[o]| ≤ 0.3


# ---------------------------------------------------------------- L0 动作构造
def build_action(raw41, s, W, H, i_dsp, override=None):
    """动作序列的**唯一**来源。训练与评测必须都调用本函数。

    Args:
        raw41: [N, 41] 第 i_dsp 列 = np.diff(SP)  (已是一阶差分)
        s:     历史窗口起点; onset o = s + W
        W, H:  历史窗口长度 / 预测长度
        override: None=真实 ΔSP; ndarray[H]=指定序列; 标量 0=SP 保持基线

    Returns:
        ndarray[H] float32 — 未来 H 步的 ΔSP (一阶差分, 与训练同构)

    注意: 训练用 `train_raw[i+W : i+W+H, I_DSP]`, 本函数与之逐元素等价。
          曾经的 bug 写法 `np.diff(raw41[s+W-1 : s+W+H, I_DSP])` 是二阶差分, 已废弃。
    """
    if override is not None:
        a = np.asarray(override, dtype=np.float32)
        if a.ndim == 0:
            a = np.full(H, float(a), dtype=np.float32)
        assert a.shape == (H,), f"override 形状应为 ({H},), 实为 {a.shape}"
        return a
    return raw41[s + W: s + W + H, i_dsp].astype(np.float32)


def assert_train_eval_identity(train_raw, raw41, W, H, i_dsp, n_probe=200, seed=0):
    """P0.2 往返一致性: 断言评测通路的动作构造与训练取法逐元素相同。"""
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(train_raw) - W - H, size=n_probe)
    for i in idxs:
        a_train = train_raw[i + W: i + W + H, i_dsp].astype(np.float32)
        a_eval = build_action(raw41, i, W, H, i_dsp)
        assert np.allclose(a_train, a_eval), f"动作编码不一致 @ i={i}"
    return True


# ---------------------------------------------------------------- 事件筛选
def select_events(raw, i_sp, i_ld, H, thr=THR_DSP, gap=GAP,
                  load_stable=LOAD_STABLE, sp_hold=SP_HOLD, hold_len=61,
                  lo=None, hi=None, W=None):
    """处理组: SP 阶跃事件。返回 (onsets[list], dsp_vals[ndarray])。

    lo/hi: 限定 onset 落在 [lo, hi) 区间。**必须显式传入 test 区间**, 否则事件
           会横跨训练集 — exp_100/101/102 未做此过滤, 因果指标含训练集泄漏。
    W:     若给出, 额外要求 onset-W >= lo (历史窗口也不越界进训练集)。
    """
    N = len(raw)
    d = np.abs(np.diff(raw[:, i_sp]))
    onsets = []
    for i in np.where(d > thr)[0] + 1:
        if not onsets or i - onsets[-1] >= gap:
            onsets.append(int(i))
    keep = []
    for o in onsets:
        if o + max(H, hold_len) >= N or o < 1:
            continue
        if lo is not None and (o < lo or (W is not None and o - W < lo)):
            continue
        if hi is not None and o + max(H, hold_len) >= hi:
            continue
        if np.abs(np.diff(raw[max(0, o - 20):min(N, o + 20), i_ld])).max() > load_stable:
            continue
        if np.abs(raw[o:o + hold_len, i_sp] - raw[o, i_sp]).max() > sp_hold:
            continue
        keep.append(o)
    dsp_vals = np.array([raw[o, i_sp] - raw[o - 1, i_sp] for o in keep], dtype=np.float32)
    return keep, dsp_vals


def select_controls(raw, i_sp, i_ld, W, H, stride=37, load_stable=LOAD_STABLE,
                    quiet_dsp=0.2):
    """对照组候选: 窗口内 SP 全程平稳、负荷稳定的时点。

    quiet_dsp: [o-W, o+H] 内 max|ΔSP| ≤ quiet_dsp 视为无干预。
    stride: 抽样步长 (避免相邻高度重叠样本)。
    """
    N = len(raw)
    d = np.abs(np.diff(raw[:, i_sp], prepend=raw[0, i_sp]))
    out = []
    for o in range(W + 1, N - H - 1, stride):
        if d[o - W:o + H].max() > quiet_dsp:
            continue
        if np.abs(np.diff(raw[max(0, o - 20):min(N, o + 20), i_ld])).max() > load_stable:
            continue
        out.append(o)
    return out


def match_controls(raw, treats, controls, i_t, i_ld, n_match=20,
                   load_bin=25.0, trend_bin=0.15, trend_lag=6):
    """CEM 粗化精确匹配: 同负荷分箱 × 同 onset 前温度趋势分箱。

    Returns: dict[onset -> list[control_onset]] (可能为空 list, 由调用方剔除)
    """
    def feats(o):
        ld = raw[o - 1, i_ld]
        tr = raw[o - 1, i_t] - raw[o - 1 - trend_lag, i_t]
        return ld, tr

    ctrl_f = np.array([feats(c) for c in controls], dtype=np.float32)
    ctrl_key = np.stack([np.floor(ctrl_f[:, 0] / load_bin),
                         np.floor(ctrl_f[:, 1] / trend_bin)], 1)
    out = {}
    for o in treats:
        ld, tr = feats(o)
        key = np.array([np.floor(ld / load_bin), np.floor(tr / trend_bin)])
        hit = np.where((ctrl_key == key).all(1))[0]
        if len(hit) == 0:                     # 退化: 只匹配负荷箱
            hit = np.where(ctrl_key[:, 0] == key[0])[0]
        if len(hit) > n_match:                # 取趋势最接近的 n_match 个
            order = np.argsort(np.abs(ctrl_f[hit, 1] - tr))
            hit = hit[order[:n_match]]
        out[o] = [controls[j] for j in hit]
    return out


# ---------------------------------------------------------------- L1 DiD 真值
def _delta_traj(raw, o, i_t, H):
    """ΔT(k) = T(o+k) − T(o−1), k=0..H-1"""
    return raw[o:o + H, i_t] - raw[o - 1, i_t]


def did_response(raw, treats, dsp_vals, matched, i_t, H,
                 n_boot=2000, seed=0, min_ctrl=3):
    """L1: 逐事件 DiD 归一化响应 + 聚合真值 + bootstrap CI + split-half ceiling。

    r_i(k) = [ΔT_treat,i(k) − mean_j ΔT_ctrl,ij(k)] / ΔSP_i

    Returns dict:
        r          [n_ev, H]  逐事件归一化响应 (真值样本)
        onsets     [n_ev]     对应 onset
        dsp        [n_ev]     对应 ΔSP
        R_true     [H]        聚合真值
        ci_lo/hi   [H]        bootstrap 95% CI
        sgn_ceiling[H]        split-half 符号一致率上限
        gain_ceiling[H]       split-half 增益比上限 (理想 1.0, 实测 <1 反映噪声)
    """
    rng = np.random.default_rng(seed)
    r, ok_onsets, ok_dsp, r_half_a, r_half_b = [], [], [], [], []
    for o, ds in zip(treats, dsp_vals):
        cs = matched.get(o, [])
        if len(cs) < min_ctrl:
            continue
        dt_t = _delta_traj(raw, o, i_t, H)
        dt_c = np.stack([_delta_traj(raw, c, i_t, H) for c in cs])
        r.append((dt_t - dt_c.mean(0)) / ds)
        # split-half: 用一半对照做"伪处理", 另一半做对照 → 纯噪声下的指标上限
        perm = rng.permutation(len(cs))
        ha, hb = perm[:len(cs) // 2], perm[len(cs) // 2:]
        if len(ha) >= 1 and len(hb) >= 1:
            r_half_a.append((dt_c[ha].mean(0) - dt_c[hb].mean(0)) / ds)
            r_half_b.append((dt_t - dt_c[hb].mean(0)) / ds)
        ok_onsets.append(o)
        ok_dsp.append(ds)
    r = np.array(r, dtype=np.float32)
    R_true = r.mean(0)

    idx = np.arange(len(r))
    boots = np.stack([r[rng.choice(idx, len(idx), replace=True)].mean(0)
                      for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5], axis=0)

    # ceiling: 半数对照 vs 另半数对照 的"响应"应为 0 → 用 r_half_b 与 R_true 的符号一致率
    ra = np.array(r_half_a, dtype=np.float32)   # 纯噪声臂
    rb = np.array(r_half_b, dtype=np.float32)   # 半对照臂 (含真信号)
    sgn_ceiling = (np.sign(rb) == np.sign(R_true)[None, :]).mean(0)
    denom = np.where(np.abs(R_true) < 1e-8, np.nan, R_true)
    gain_ceiling = np.abs(rb.mean(0) / denom)
    noise_floor = np.abs(ra.mean(0))

    return dict(r=r, onsets=np.array(ok_onsets), dsp=np.array(ok_dsp, dtype=np.float32),
                R_true=R_true, ci_lo=ci_lo, ci_hi=ci_hi,
                sgn_ceiling=sgn_ceiling, gain_ceiling=gain_ceiling,
                noise_floor=noise_floor, n_ev=len(r))


# ---------------------------------------------------------------- 模型封装
class ModelWrapper:
    """统一动作接口: 一律传 [B, H, 1]。

    DirectWM.forward 内 `a_future.reshape(B, -1)` → [B, H]      ✅
    TimeXerWM.forward 内 `a_future.permute(0, 2, 1)` → [B, 1, H] ✅
    故 [B, H, 1] 对两种架构通用, 无需分支。
    """

    def __init__(self, model, raw, raw41, W, H, i_dsp, device):
        self.m = model.eval()
        self.raw, self.raw41 = raw, raw41
        self.W, self.H, self.i_dsp = W, H, i_dsp
        self.device = device
        self.N = len(raw)

    def predict(self, s, action_override=None, x_hist_override=None):
        """Returns ndarray[H] 物理空间温度预测, 越界返回 None。"""
        if s < 0 or s + self.W + self.H >= self.N:
            return None
        xh = self.raw[s:s + self.W] if x_hist_override is None else x_hist_override
        a = build_action(self.raw41, s, self.W, self.H, self.i_dsp, action_override)
        x = torch.from_numpy(np.ascontiguousarray(xh, dtype=np.float32)).unsqueeze(0).to(self.device)
        at = torch.from_numpy(a).reshape(1, self.H, 1).to(self.device)
        with torch.no_grad():
            mu, _ = self.m(x, at)
        return mu[0].detach().cpu().numpy()


def model_response(wrapper, onsets, dsp_vals, x_hist_fn=None):
    """do-probe: m_i(k) = [pred(real a) − pred(a=0)](k) / ΔSP_i

    x_hist_fn: 可选 callable(o) -> x_hist, 用于 L3 (LEAK 屏蔽 / DO_vs_SEE 换状态)。

    Returns: m [n, H], pred_real [n, H], kept_mask [len(onsets)]
    """
    m, p_real_all, keep = [], [], []
    for o, ds in zip(onsets, dsp_vals):
        s = o - wrapper.W
        xh = None if x_hist_fn is None else x_hist_fn(o)
        p1 = wrapper.predict(s, None, xh)
        p0 = wrapper.predict(s, 0.0, xh)
        if p1 is None or p0 is None:
            keep.append(False)
            continue
        m.append((p1 - p0) / ds)
        p_real_all.append(p1)
        keep.append(True)
    return (np.array(m, dtype=np.float32), np.array(p_real_all, dtype=np.float32),
            np.array(keep, dtype=bool))


# ---------------------------------------------------------------- L2 指标
def causal_metrics(m, r, R_true, sgn_ceiling, ks):
    """L2 指标。m/r: [n, H] 模型响应与逐事件真值 (已对齐同一组事件)。

    ks: 报告时刻索引 list[(k, label)]
    """
    mbar = m.mean(0)
    out = {'profile': {}}
    for k, lab in ks:
        sgn_pair = float((np.sign(m[:, k]) == np.sign(r[:, k])).mean())
        sgn_agg = float((np.sign(m[:, k]) == np.sign(R_true[k])).mean())
        ceil = float(sgn_ceiling[k])
        sgn_norm = (sgn_pair - .5) / (ceil - .5) if ceil > .5 else np.nan
        gain = float(mbar[k] / R_true[k]) if abs(R_true[k]) > 1e-8 else np.nan
        out['profile'][lab] = dict(
            k=int(k), sgn_pair=sgn_pair, sgn_agg=sgn_agg, sgn_ceiling=ceil,
            sgn_norm=float(sgn_norm), gain=gain,
            resp_model=float(mbar[k]), resp_true=float(R_true[k]))
    kmax = ks[-1][0]
    sl = slice(0, kmax + 1)
    denom = np.std(mbar[sl]) * np.std(R_true[sl])
    out['shape_corr'] = float(np.corrcoef(mbar[sl], R_true[sl])[0, 1]) if denom > 1e-12 else np.nan
    out['ttp_model'] = int(_ttp(mbar[sl]))
    out['ttp_true'] = int(_ttp(R_true[sl]))
    out['ttp_err'] = out['ttp_model'] - out['ttp_true']
    kmid = kmax // 2
    out['mono_model'] = float(mbar[kmax] / mbar[kmid]) if abs(mbar[kmid]) > 1e-8 else np.nan
    out['mono_true'] = float(R_true[kmax] / R_true[kmid]) if abs(R_true[kmid]) > 1e-8 else np.nan
    return out


def _ttp(curve):
    """达到末点值 50% 的首个时刻。"""
    end = curve[-1]
    if abs(end) < 1e-8:
        return len(curve) - 1
    hit = np.where(np.abs(curve) >= 0.5 * abs(end))[0]
    return hit[0] if len(hit) else len(curve) - 1


def cfi(mt, k_ref_label):
    """复合因果分 CFI ∈ (-inf, 1]。用于 P0.5 的 best-CFI checkpoint 选择。"""
    p = mt['profile'][k_ref_label]
    sgn = p['sgn_norm'] if np.isfinite(p['sgn_norm']) else 0.0
    g = p['gain']
    g_term = min(g, 1.0 / g) if (np.isfinite(g) and g > 0) else 0.0
    sh = mt['shape_corr'] if np.isfinite(mt['shape_corr']) else 0.0
    kmax = max(v['k'] for v in mt['profile'].values())
    ttp = 1.0 - min(abs(mt['ttp_err']) / max(kmax, 1), 1.0)
    return float(0.35 * np.clip(sgn, 0, 1) + 0.30 * g_term + 0.20 * np.clip(sh, 0, 1) + 0.15 * ttp)
