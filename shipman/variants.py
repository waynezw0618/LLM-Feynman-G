"""shipman.variants — 四变体 y/X 构造 (灰箱: 扣除已知推进/舵项) + 扩展真值标定.

变体定义 (全部 LGF 零改动, "特征即列"):
  V1 传统:   y = (m'+my')(v̇m+u'r)+xG'm'ṙ − Y_R        (Y 方程)
             y = (IzG'+xG'²m'+Jz')ṙ + xG'm'(v̇m+u'r) − N_R (N 方程)
             X 基础库 (Abkowitz 式 ~26 列)
  V2 分数阶: + D^α×{u,vm,r}, α∈{0.3,0.5,0.7,0.9} → +12 列
  V3 积分:   + conv_{vm,r}×K1(τ=5s)/K2(τ=50s) + 幂律 PW(β=0.3) → +5 列
  V4 组合:   V2 列 + V3 列

真值 (可验): V1=Y&Y2015 系数; V2/V3/V4 = 扩展真值模型 (mmg_sim.ext_forces 生成,
与 features 同实现 → 闭环). 扩展系数量级标定 ~10% 力占比.

噪声下特征先过 savgol_smooth. 灰箱 y 从仿真真值加速度构造 (无噪声口径);
含噪口径用 central_diff(savgol(信号)) — 由 rc_sm_prep 的 noise_mode 控制.
"""
from __future__ import annotations

import numpy as np

from .coeff_kvlcc2 import ShipCoeffs, kvlcc2
from .features import (central_diff, exp_conv, gl_deriv, powerlaw_conv,
                       savgol_smooth)
from .mmg_sim import simulate

# 基础特征库 (V1): (列名, 构造函数(x: dict of signals) -> ndarray)
BASE_TERMS = [
    ("vm",      lambda d: d["v"]),
    ("r",       lambda d: d["r"]),
    ("u",       lambda d: d["u"]),
    ("vmvm",    lambda d: d["v"] ** 2),
    ("vmr",     lambda d: d["v"] * d["r"]),
    ("rr",      lambda d: d["r"] ** 2),
    ("vm3",     lambda d: d["v"] ** 3),
    ("vm2r",    lambda d: d["v"] ** 2 * d["r"]),
    ("vmr2",    lambda d: d["v"] * d["r"] ** 2),
    ("r3",      lambda d: d["r"] ** 3),
    ("vm4",     lambda d: d["v"] ** 4),
    ("uabsu",   lambda d: d["u"] * np.abs(d["u"])),
    ("vmabsvm", lambda d: d["v"] * np.abs(d["v"])),
    ("rabsr",   lambda d: d["r"] * np.abs(d["r"])),
    ("uvr",     lambda d: d["u"] * d["v"] * d["r"]),
    ("uur",     lambda d: d["u"] ** 2 * d["r"]),
    ("uvm",     lambda d: d["u"] * d["v"] ** 2),
    ("d",       lambda d: d["delta"]),
    ("dd",      lambda d: d["delta"] ** 2),
    ("d3",      lambda d: d["delta"] ** 3),
    ("vmd",     lambda d: d["v"] * d["delta"]),
    ("rd",      lambda d: d["r"] * d["delta"]),
    ("u2",      lambda d: d["u"] ** 2),
    ("vm2d2",   lambda d: d["v"] ** 2 * d["delta"] ** 2),
    ("r2d2",    lambda d: d["r"] ** 2 * d["delta"] ** 2),
    ("vmrd",    lambda d: d["v"] * d["r"] * d["delta"]),
]

ALPHAS = (0.3, 0.5, 0.7, 0.9)
MEM_KERNELS = {"K1": 5.0, "K2": 50.0}      # τ (s)
PW_BETA = 0.3


def variant_features(out: dict, variant: str, dt: float, noise_rel: float = 0.0) -> tuple:
    """构造某变体的 (X 列, 列名列表). out = simulate() 返回的 dict.

    噪声>0 时信号先 savgol 平滑 (观测口径), 特征从平滑信号计算.
    分数阶/卷积特征的前 window 点为 NaN, 返回 valid 掩码 (调用方裁剪).
    """
    sig = {}
    for k in ("u", "v", "r", "delta", "psi"):
        x = out[k].copy()
        if noise_rel > 0:
            x = savgol_smooth(x, dt, win_s=8.0, poly=3)
        sig[k] = x
    d = dict(sig); d["v"] = sig["v"]        # v 即 vm

    cols, names = [], []
    n = len(sig["u"])
    nan_mask = np.zeros(n, dtype=bool)
    for nm, fn in BASE_TERMS:
        c = fn(d)
        cols.append(c); names.append(nm)

    if variant in ("v2", "v4"):
        for a in ALPHAS:
            for s in ("u", "v", "r"):
                c = gl_deriv(sig[s], dt, a, window=200)
                cols.append(c); names.append(f"d{s}_a{int(a*10):02d}")
                nan_mask |= ~np.isfinite(c)

    if variant in ("v3", "v4"):
        for kn, tau in MEM_KERNELS.items():
            for s in ("v", "r"):
                c = exp_conv(sig[s], dt, tau)
                cols.append(c); names.append(f"conv_{s}_{kn}")
        c = powerlaw_conv(sig["r"], dt, PW_BETA, window=400)
        cols.append(c); names.append("conv_r_PW")
        nan_mask |= ~np.isfinite(c)

    X = np.column_stack(cols)
    return X, names, ~nan_mask


def graybox_targets(s: ShipCoeffs, out: dict) -> dict:
    """灰箱目标: y_Y/y_N (扣除舵力; 推进只在 X 方程).

    使用仿真真值加速度 (vdot/rdot, 无微分误差). y_X 含扣除 X_P/X_R.
    返回 dict(yY, yN, yX).
    """
    u, vm, r = out["u"], out["v"], out["r"]
    vdot, rdot = out["vdot"], out["rdot"]
    m = s.m_prime; xG = s.xG / s.Lpp
    mT = m + s.mx; nT = m + s.my; IzT = s.Iz_prime + xG**2*m + s.Jz
    # 注意 out 的 vdot/rdot 已含 ts=U0/Lpp (1/s), 灰箱还原为 prime 力:
    ts = s.U0 / s.Lpp
    vdot_p = vdot / ts; rdot_p = rdot / ts
    yY = nT * vdot_p + mT * u * r + xG * m * rdot_p - out["YR"]
    yN = IzT * rdot_p + xG * m * (vdot_p + u * r) - out["NR"]
    yX = mT * (out["udot"] / ts) - nT * vm * r - xG * m * r * r - out["XP"] - out["XR"]
    return dict(yY=yY, yN=yN, yX=yX)


def calibrate_ext(s: ShipCoeffs, target_frac: float = 0.10, seed: int = 0) -> ShipCoeffs:
    """标定 V2/V3 扩展真值系数: 使新项 RMS / 总力 RMS ≈ target_frac (对 Y 方程).

    返回带 cDv/cDr/mem_A 的新 ShipCoeffs (原对象不改).
    """
    s2 = kvlcc2()
    # 用 name 等字段拷贝原船
    for f in s.__dataclass_fields__:
        setattr(s2, f, getattr(s, f))
    out = simulate(s, "zig1010", t_end=600.0, seed=seed)
    yY = graybox_targets(s, out)["yY"]
    yN = graybox_targets(s, out)["yN"]
    rmsY = float(np.sqrt(np.mean(yY ** 2))) + 1e-12
    rmsN = float(np.sqrt(np.mean(yN ** 2))) + 1e-12
    dt = 1.0
    gl_v = gl_deriv(out["v"], dt, 0.5, window=200)
    gl_r = gl_deriv(out["r"], dt, 0.5, window=200)
    m = np.isfinite(gl_v) & np.isfinite(gl_r)
    rms_glv = float(np.sqrt(np.mean(gl_v[m] ** 2))) + 1e-12
    rms_glr = float(np.sqrt(np.mean(gl_r[m] ** 2))) + 1e-12
    s2.cDv = target_frac * rmsY / rms_glv
    s2.alpha_v = 0.5
    s2.cDr = target_frac * rmsN / rms_glr
    s2.alpha_r = 0.5
    # V3 记忆核: 作用于 r (N 方程), 幅值 A 使 ∫K·r 的 RMS ≈ target_frac·rmsN
    conv = exp_conv(out["r"], dt, MEM_KERNELS["K1"])
    rms_c = float(np.sqrt(np.mean(conv ** 2))) + 1e-12
    A1 = target_frac * rmsN / rms_c
    s2.mem_A = ((A1, MEM_KERNELS["K1"]),)
    return s2


def build_cases(s: ShipCoeffs, variant: str, maneuvers, noise_rel=0.0, seed=0,
                eq: str = "Y", min_pts: int = 50) -> tuple:
    """构造某变体某方程的 (cases, var_names). eq ∈ {X, Y, N}.

    cases = [dict(X=(N,k) 特征阵, y=(N,) 目标, name=机动名)]; NaN 行已裁剪.
    """
    Xs, names = [], None
    ys = []
    for i, mn in enumerate(maneuvers):
        out = simulate(s, mn, noise_rel=noise_rel, seed=seed + i)
        X, names, valid = variant_features(out, variant, 1.0, noise_rel)
        tg = graybox_targets(s, out)[f"y{eq}"]
        if noise_rel > 0:
            # 含噪口径: 目标加速度也由 savgol+中心差分 (与特征同口径)
            sm_v = savgol_smooth(out["v"], 1.0); sm_r = savgol_smooth(out["r"], 1.0)
            from .features import central_diff as cd
            vdot_n = cd(sm_v, 1.0) / (s.U0 / s.Lpp)
            rdot_n = cd(sm_r, 1.0) / (s.U0 / s.Lpp)
            m = s.m_prime; xG = s.xG / s.Lpp
            mT = m + s.mx; nT = m + s.my; IzT = s.Iz_prime + xG**2*m + s.Jz
            if eq == "Y":
                tg = nT * vdot_n + mT * out["u"] * out["r"] + xG * m * rdot_n - out["YR"]
            elif eq == "N":
                tg = IzT * rdot_n + xG * m * (vdot_n + out["u"] * out["r"]) - out["NR"]
        keep = valid & np.isfinite(tg)
        keep[:min_pts] = False       # 丢弃起始瞬态/窗热身
        Xs.append((X[keep], tg[keep], mn))
    cases = [dict(X=x, y=y, name=mn) for x, y, mn in Xs]
    return cases, names


def true_coeffs(s: ShipCoeffs, eq: str) -> dict:
    """返回某方程的真值系数表 (参数恢复验收用)."""
    if eq == "Y":
        return {"vm": s.Yv, "r": s.Yr, "vm3": s.Yvvv, "vm2r": s.Yvvr,
                "vmr2": s.Yvrr, "r3": s.Yrrr, "cDv_frac": s.cDv}
    if eq == "N":
        return {"vm": s.Nv, "r": s.Nr, "vm3": s.Nvvv, "vm2r": s.Nvvr,
                "vmr2": s.Nvrr, "r3": s.Nrrr, "cDr_frac": s.cDr}
    return {"uabsu": -s.R0, "vmvm": s.Xvv, "vmr": s.Xvr, "rr": s.Xrr, "vm4": s.Xvvvv}
