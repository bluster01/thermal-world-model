#!/usr/bin/env python3
"""
test_eval_protocol.py — 公平评测协议回归测试
=============================================
无需真实数据/权重: 用桩模块替换 exp_025 的数据与模型加载, 以解析式假世界验证协议逻辑。
覆盖 P0-A/B/C/D 四项修复的行为契约。

运行: python -m pytest tests/test_eval_protocol.py -v
     或 python tests/test_eval_protocol.py
"""
import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg  # noqa: E402

# ── 桩: 数据与列定义 ──────────────────────────────────────────────
W = cfg.WINDOW_SIZE
H_OUT = 18
N_FEAT = 40
N_ROWS = 500

NUMERIC_COLS = [f'col_{i}' for i in range(N_FEAT)]
TARGET_IDX = 10
VALVE_IDX = [20, 21]
SP_IDX = 36
LOAD_IDX, COAL_IDX, FLOW_IDX = 0, 4, 3
NUMERIC_COLS[TARGET_IDX] = '末级过热器出口汽温'
NUMERIC_COLS[VALVE_IDX[0]] = '一级减温调节门阀位'
NUMERIC_COLS[VALVE_IDX[1]] = '二级减温调节门阀位'
NUMERIC_COLS[LOAD_IDX] = '机组负荷'
NUMERIC_COLS[COAL_IDX] = '未校正总煤量'
NUMERIC_COLS[FLOW_IDX] = '主蒸汽流量'

_rng = np.random.default_rng(0)
test_raw = np.zeros((N_ROWS, N_FEAT), dtype=np.float32)
test_raw[:, TARGET_IDX] = 565.0 + _rng.normal(0, 0.3, N_ROWS)   # 主汽温
test_raw[:, SP_IDX] = 568.0                                      # 设定值 (恒定, 便于判定)
test_raw[:, VALVE_IDX[0]] = 20.0
test_raw[:, VALVE_IDX[1]] = 15.0
test_raw[:, LOAD_IDX] = 800.0 + _rng.normal(0, 20, N_ROWS)
test_raw[:, COAL_IDX] = 300.0 + _rng.normal(0, 10, N_ROWS)
test_raw[:, FLOW_IDX] = 2600.0 + _rng.normal(0, 50, N_ROWS)

_LOAD_MEAN = float(test_raw[:, LOAD_IDX].mean())


class FakeWM(nn.Module):
    """解析式假世界模型: 物理方向正确 (开阀→降温) + 负荷正向影响

    ŷ_k = y0 + gain·(a_k − a_prev)·ramp_k + load_gain·(load − load_mean) + bias
    可微 (MPC 需反传), probabilistic 时返回 lv。
    """

    def __init__(self, gain=-0.25, load_gain=0.02, bias=0.0,
                 probabilistic=True, log_var=-1.0):
        super().__init__()
        self.gain, self.load_gain, self.bias = gain, load_gain, bias
        self.probabilistic, self.log_var = probabilistic, log_var
        self.use_sp = False

    def forward(self, x_hist, a_flat, sp_fut=None):
        B = x_hist.shape[0]
        y0 = x_hist[:, -1, TARGET_IDX].unsqueeze(1)              # [B,1]
        a_prev = x_hist[:, -1, VALVE_IDX[1]].unsqueeze(1)        # [B,1]
        load = x_hist[:, -1, LOAD_IDX].unsqueeze(1)
        a2 = a_flat.view(B, H_OUT, 2)[:, :, 1]                   # [B,H] 二级阀
        ramp = torch.linspace(0.3, 1.0, H_OUT, device=x_hist.device).unsqueeze(0)
        mu = y0 + self.gain * (a2 - a_prev) * ramp \
            + self.load_gain * (load - _LOAD_MEAN) + self.bias
        lv = torch.full_like(mu, self.log_var) if self.probabilistic else None
        return mu, lv


_FAKE_SPECS = {
    'M0': dict(gain=-0.25, bias=0.0),
    'M8': dict(gain=-0.24, bias=0.1),
    'M9': dict(gain=-0.26, bias=-0.1),
    'M5': dict(gain=-0.25, bias=0.0, probabilistic=False),
    'M7': dict(gain=-0.25, bias=0.0, probabilistic=True),
}


def build_model(mid):
    return FakeWM(**_FAKE_SPECS.get(mid, {}))


_stub = types.ModuleType('experiments.phase1_dynamics.exp_025_unified_benchmark')
for _n in ('build_model', 'test_raw', 'VALVE_IDX', 'TARGET_IDX', 'H_OUT', 'NUMERIC_COLS'):
    setattr(_stub, _n, globals()[_n])
sys.modules['experiments.phase1_dynamics.exp_025_unified_benchmark'] = _stub

from experiments.phase2_mpc import eval_protocol as ep  # noqa: E402

DEV = ep.DEVICE


def _fake_world(ids=('M0', 'M8', 'M9')):
    w = ep.WorldSim.__new__(ep.WorldSim)
    w.ids = list(ids)
    w.models = [build_model(i).to(DEV).eval() for i in ids]
    return w


def _ctrl_model(mid):
    return build_model(mid).to(DEV).eval()


# ══════════════════════════════════════════════════════════════════
# 执行器
# ══════════════════════════════════════════════════════════════════
class TestActuator:
    def test_rate_limit(self):
        """速率限幅: 单步位移不得超过 rate"""
        a = ep.Actuator([20.0, 15.0], inertia=1.0, rate=5.0)
        u = a.step([20.0, 100.0])
        assert u[1] == pytest.approx(20.0), "首步应被限幅到 15+5"
        u = a.step([20.0, 100.0])
        assert u[1] == pytest.approx(25.0)

    def test_range_clip(self):
        a = ep.Actuator([20.0, 44.0], inertia=1.0, rate=5.0, lo=0.0, hi=45.0)
        assert a.step([20.0, 100.0])[1] == pytest.approx(45.0)
        a2 = ep.Actuator([20.0, 2.0], inertia=1.0, rate=5.0, lo=0.0, hi=45.0)
        assert a2.step([20.0, -50.0])[1] == pytest.approx(0.0)

    def test_inertia(self):
        """一阶惯性: K=0.5 时半步逼近"""
        a = ep.Actuator([20.0, 10.0], inertia=0.5, rate=100.0)
        assert a.step([20.0, 20.0])[1] == pytest.approx(15.0)

    def test_shared_by_both_controllers(self):
        """公平性契约: PID 与 MPC 必须用同一执行器类"""
        src = open(ep.__file__, encoding='utf-8').read()
        assert src.count('act = Actuator(') == 1, "run_episode 中应只有一处执行器构造"


# ══════════════════════════════════════════════════════════════════
# P0-C 扰动
# ══════════════════════════════════════════════════════════════════
class TestDisturbance:
    def test_energy_normalized_across_rho(self):
        """AR(1) 稳态方差归一化: 不同 rho 下扰动能量可比 (否则 rho 越大幅度越大)"""
        stds = []
        for rho in (0.0, 0.5, 0.9):
            d = ep.Disturbance(sigma=1.0, rho=rho, seed=1)
            stds.append(np.std([d.step() for _ in range(20000)]))
        assert max(stds) / min(stds) < 1.35, f"能量未归一化: {stds}"

    def test_physical_mode_touches_physical_cols_only(self):
        """physical 模式只改物理通道, 不直接改温度/阀位"""
        d = ep.Disturbance(sigma=0.5, rho=0.9, mode='physical', seed=2)
        row = test_raw[100].copy()
        out = d.apply_to_row(row, 1.0)
        assert out[TARGET_IDX] == row[TARGET_IDX], "physical 模式不应直接改温度"
        assert out[VALVE_IDX[1]] == row[VALVE_IDX[1]], "physical 模式不应改阀位"
        assert not np.allclose(out[ep.PHYS_DIST_IDX], row[ep.PHYS_DIST_IDX])
        assert d.apply_to_temp(565.0, 1.0) == 565.0, "physical 模式不加输出偏置"

    def test_output_mode_is_legacy_bias(self):
        d = ep.Disturbance(sigma=0.5, rho=0.9, mode='output', seed=3)
        row = test_raw[100].copy()
        assert np.allclose(d.apply_to_row(row, 1.0), row), "output 模式不改物理列"
        assert d.apply_to_temp(565.0, 2.0) == pytest.approx(567.0)

    def test_disabled_when_sigma_zero(self):
        d = ep.Disturbance(sigma=0.0, seed=4)
        assert d.step() == 0.0
        assert d.apply_to_temp(565.0, 9.0) == 565.0

    def test_physical_channels_resolved(self):
        assert len(ep.PHYS_DIST_IDX) == 3, "应解析出负荷/煤量/流量三个通道"


# ══════════════════════════════════════════════════════════════════
# P0-B 第三方评测世界
# ══════════════════════════════════════════════════════════════════
class TestWorldSim:
    def test_rejects_self_scoring(self):
        """世界与控制器重叠必须报错 (禁止自评分)"""
        with pytest.raises(AssertionError, match='自评分'):
            ep.WorldSim(['M0', 'M7'], controller_ids=['M7'])

    def test_ensemble_mean(self):
        world = _fake_world(('M0', 'M8', 'M9'))
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        a = torch.full((H_OUT, 2), 15.0, device=DEV)
        a[:, 0] = 20.0
        mu = world.predict(win, a)
        singles = [m(win, a.reshape(1, -1))[0][0] for m in world.models]
        assert torch.allclose(mu, torch.stack(singles).mean(0), atol=1e-5)

    def test_world_is_shared_across_controllers(self):
        """同一世界对象被不同控制器复用 → 温度可比"""
        world = _fake_world()
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        a = torch.full((H_OUT, 2), 15.0, device=DEV)
        assert torch.allclose(world.predict(win, a), world.predict(win, a))


# ══════════════════════════════════════════════════════════════════
# 成本函数
# ══════════════════════════════════════════════════════════════════
class TestCostConfig:
    def _args(self, mu_val=580.0, sig_val=1.0):
        mu = torch.full((10,), mu_val, device=DEV, requires_grad=False)
        sig = torch.full((10,), sig_val, device=DEV)
        tgt = torch.full((10,), 568.0, device=DEV)
        a = torch.full((10, 1), 15.0, device=DEV)
        return mu, sig, tgt, a, torch.tensor(15.0, device=DEV)

    def test_needs_sigma_only_for_cvar(self):
        assert not ep.CostConfig('a').needs_sigma()
        assert not ep.CostConfig('b').needs_sigma()
        assert ep.CostConfig('c').needs_sigma()
        assert not ep.CostConfig('d').needs_sigma()

    def test_overtemp_variant_penalizes_more(self):
        """(b) 超温口径在超温时代价必须高于 (a)"""
        mu, sig, tgt, a, al = self._args(mu_val=580.0)
        assert ep.CostConfig('b')(mu, sig, tgt, a, al) > ep.CostConfig('a')(mu, sig, tgt, a, al)

    def test_cvar_sees_sigma(self):
        """(c) CVaR 对 σ 敏感 — 这是概率模型的专属能力"""
        c = ep.CostConfig('c')
        mu, _, tgt, a, al = self._args(mu_val=572.0)
        lo = c(mu, torch.full((10,), 0.1, device=DEV), tgt, a, al)
        hi = c(mu, torch.full((10,), 3.0, device=DEV), tgt, a, al)
        assert hi > lo, "σ 增大应提高 CVaR 代价"

    def test_cost_a_ignores_sigma(self):
        c = ep.CostConfig('a')
        mu, _, tgt, a, al = self._args(mu_val=572.0)
        lo = c(mu, torch.full((10,), 0.1, device=DEV), tgt, a, al)
        hi = c(mu, torch.full((10,), 3.0, device=DEV), tgt, a, al)
        assert lo == hi, "口径 a 不应受 σ 影响 (确定性/概率模型同等对待)"

    def test_differentiable(self):
        for v in 'abcd':
            a = torch.full((10, 1), 15.0, device=DEV, requires_grad=True)
            mu = 565.0 - 0.25 * (a.squeeze(1) - 15.0)
            sig = torch.full((10,), 1.0, device=DEV)
            tgt = torch.full((10,), 568.0, device=DEV)
            ep.CostConfig(v)(mu, sig, tgt, a, torch.tensor(15.0, device=DEV)).backward()
            assert a.grad is not None and torch.isfinite(a.grad).all(), f"口径 {v} 梯度异常"


# ══════════════════════════════════════════════════════════════════
# P0-A 真闭环 PID
# ══════════════════════════════════════════════════════════════════
class TestPID:
    def test_responds_to_error_sign(self):
        """SP > PV (需升温) → 减小阀位; SP < PV → 增大阀位"""
        pid = ep.PIDController()
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        u_cold = pid.act(win, 575.0, None, np.array([20.0, 15.0]))[0][1]
        pid.reset()
        u_hot = pid.act(win, 555.0, None, np.array([20.0, 15.0]))[0][1]
        assert u_cold > u_hot, "PID 输出应随偏差单调"

    def test_anti_windup(self):
        """饱和时不得继续积分"""
        pid = ep.PIDController(kp=40.0, ki=8.0, u_hi=45.0)
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        for _ in range(50):
            pid.act(win, 520.0, None, np.array([20.0, 15.0]))
        assert abs(pid.I) <= 300.0 + 1e-6
        pid2 = ep.PIDController()
        for _ in range(50):
            pid2.act(win, 700.0, None, np.array([20.0, 15.0]))
        assert abs(pid2.I) < 1e-6, "持续饱和时积分应停止累积"

    def test_is_closed_loop_not_replay(self):
        """P0-A 契约: PID 动作依赖当前窗口状态, 而非历史录像"""
        pid = ep.PIDController()
        w1 = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        w2 = w1.clone()
        w2[0, -1, TARGET_IDX] += 10.0
        u1 = pid.act(w1, 568.0, None, np.array([20.0, 15.0]))[0][1]
        pid.reset()
        u2 = pid.act(w2, 568.0, None, np.array([20.0, 15.0]))[0][1]
        assert u1 != u2, "PID 必须对当前温度反馈, 否则就是录像回放"


# ══════════════════════════════════════════════════════════════════
# MPC
# ══════════════════════════════════════════════════════════════════
class TestMPC:
    def _mpc(self, mid='M7', variant='a', **kw):
        return ep.MPCController(_ctrl_model(mid), ep.CostConfig(variant),
                                h_plan=10, m_step=6, e_steps=10, **kw)

    def test_plan_respects_rate_limit(self):
        m = self._mpc()
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        sp = torch.full((H_OUT,), 540.0, device=DEV)
        cmds, n = m.act(win, 540.0, sp, np.array([20.0, 15.0]))
        assert n == 6 and cmds.shape == (6, 2)
        assert abs(cmds[0, 1] - 15.0) <= 5.0 + 1e-4, "首步超出速率约束"
        assert np.all(np.abs(np.diff(cmds[:, 1])) <= 5.0 + 1e-4)

    def test_plan_direction_is_physical(self):
        """需降温 (PV>SP) → MPC 应开大减温阀 (gain<0)"""
        m = self._mpc()
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        win[0, :, TARGET_IDX] = 575.0
        sp = torch.full((H_OUT,), 560.0, device=DEV)
        cmds, _ = m.act(win, 560.0, sp, np.array([20.0, 15.0]))
        assert cmds[0, 1] > 15.0, "过热时应开阀降温"

    def test_warm_start_length_stable(self):
        m = self._mpc()
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        sp = torch.full((H_OUT,), 568.0, device=DEV)
        for _ in range(3):
            m.act(win, 568.0, sp, np.array([20.0, 15.0]))
            assert len(m.a_init) == m.h_plan, "warm-start 长度必须稳定"

    def test_single_channel_by_default(self):
        """默认只控二级阀 — 与 PID 同执行通道 (公平)"""
        m = self._mpc()
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        sp = torch.full((H_OUT,), 560.0, device=DEV)
        cmds, _ = m.act(win, 560.0, sp, np.array([20.0, 15.0]))
        assert np.allclose(cmds[:, 0], 20.0), "一级阀不应被默认改动"

    def test_deterministic_model_with_cvar_is_guarded(self):
        """确定性模型 + CVaR: σ 为 None, 不得崩溃 (由调用方跳过)"""
        m = ep.MPCController(_ctrl_model('M5'), ep.CostConfig('c'),
                             h_plan=10, m_step=6, e_steps=5)
        win = torch.FloatTensor(test_raw[:W]).unsqueeze(0).to(DEV)
        sp = torch.full((H_OUT,), 568.0, device=DEV)
        cmds, _ = m.act(win, 568.0, sp, np.array([20.0, 15.0]))
        assert np.isfinite(cmds).all()


# ══════════════════════════════════════════════════════════════════
# P0-D 窗口回填 + 主循环
# ══════════════════════════════════════════════════════════════════
class TestRunEpisode:
    def test_valve_backfilled_into_window(self):
        """P0-D 契约: 窗口阀位列必须是执行器实际值, 不是历史真值"""
        world = _fake_world()
        pid = ep.PIDController()
        r = ep.run_episode(pid, world, start=0, n_steps=30, dist=None)
        assert len(r['temp']) == 30
        # 控制器把阀位推离历史常值 15.0
        assert not np.allclose(r['act'][:, 1], 15.0), "阀位未随控制变化"

    def test_temp_and_sp_aligned(self):
        world = _fake_world()
        r = ep.run_episode(ep.PIDController(), world, start=0, n_steps=24, dist=None)
        assert len(r['temp']) == len(r['sp']) == len(r['act']) == len(r['dist'])

    def test_physical_disturbance_moves_temperature(self):
        """P0-C: 物理通道扰动必须经世界模型传导到温度"""
        world = _fake_world()
        base = ep.run_episode(ep.PIDController(), world, 0, n_steps=30, dist=None)
        dist = ep.Disturbance(sigma=3.0, rho=0.9, mode='physical', seed=7)
        pert = ep.run_episode(ep.PIDController(), world, 0, n_steps=30, dist=dist)
        assert not np.allclose(base['temp'], pert['temp'], atol=1e-3), \
            "物理扰动未传导到温度 (通道或世界模型未响应负荷)"

    def test_same_disturbance_seed_same_sequence(self):
        """公平性: 相同 seed → 相同扰动实现 (跨方法配对检验前提)"""
        d1 = ep.Disturbance(sigma=0.3, rho=0.9, seed=11)
        d2 = ep.Disturbance(sigma=0.3, rho=0.9, seed=11)
        assert [d1.step() for _ in range(50)] == [d2.step() for _ in range(50)]

    def test_pid_and_mpc_same_loop(self):
        """PID 与 MPC 走同一主循环, 输出结构一致"""
        world = _fake_world()
        mpc = ep.MPCController(_ctrl_model('M7'), ep.CostConfig('a'),
                               h_plan=10, m_step=6, e_steps=5)
        a = ep.run_episode(ep.PIDController(), world, 0, n_steps=24, dist=None)
        b = ep.run_episode(mpc, world, 0, n_steps=24, dist=None)
        assert a['temp'].shape == b['temp'].shape
        assert set(a) == set(b)

    def test_controller_reset_between_episodes(self):
        """跨轨迹状态必须清零, 否则前一条轨迹污染后一条"""
        world = _fake_world()
        pid = ep.PIDController()
        ep.run_episode(pid, world, 0, n_steps=20, dist=None)
        i_after = pid.I
        ep.run_episode(pid, world, 0, n_steps=1, dist=None)
        assert pid.I != i_after or abs(pid.I) < abs(i_after) + 1e-9


# ══════════════════════════════════════════════════════════════════
# 指标口径
# ══════════════════════════════════════════════════════════════════
class TestMetrics:
    def _ep(self, temp):
        n = len(temp)
        return dict(temp=np.asarray(temp, float),
                    sp=np.full(n, 568.0),
                    act=np.tile([20.0, 15.0], (n, 1)),
                    dist=np.zeros(n))

    def test_overtemp_in_seconds(self):
        """超温以秒计 (1 步 = 10s) — 修 exp_027 的步数/秒混用"""
        m = ep.metrics(self._ep([570.0] * 5 + [580.0] * 3))
        assert m['overtemp_s'] == pytest.approx(3 * ep.DT)
        assert m['overtemp_int'] == pytest.approx(3 * 5.0 * ep.DT)

    def test_rmse_against_sp(self):
        m = ep.metrics(self._ep([568.0] * 10))
        assert m['rmse'] == pytest.approx(0.0, abs=1e-9)
        assert m['overtemp_s'] == 0.0

    def test_j_total_monotone_in_overtemp(self):
        lo = ep.metrics(self._ep([568.0] * 10))['j_total']
        hi = ep.metrics(self._ep([568.0] * 7 + [590.0] * 3))['j_total']
        assert hi > lo

    def test_keys_present(self):
        m = ep.metrics(self._ep([568.0] * 12))
        for k in ('rmse', 'iae', 'itae', 'tv', 'overtemp_s', 'overtemp_int', 'j_total'):
            assert k in m


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
