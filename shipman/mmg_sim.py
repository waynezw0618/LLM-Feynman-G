"""shipman.mmg_sim — 3-DOF MMG 标准模型数值仿真器 (Yasukawa & Yoshimura 2015).

模型 (prime 无量纲, 状态 s=[u,v,r,ψ,x,y]):
    (m'+mx')·u̇ − (m'+my')·v·r − xG'·m'·r² = X_H + X_P + X_R (+X_ext)
    (m'+my')·v̇ + (m'+mx')·u·r + xG'·m'·ṙ  = Y_H + Y_R (+Y_ext)
    (Iz'+Jz')·ṙ + xG'·m'·(v̇+u·r)          = N_H + N_R (+N_ext)

其中 xG 项按 Y&Y2015 式(1); 为简化辨识目标构造, 取 xG=0 的船中坐标
(KVLCC2 xG=11.2m, xG'=0.035, 影响为二阶小量; 报告中如实标注该近似).

模块力 (全部 prime):
  X_H = -R0'·u|u| + Xvv·v² + Xvr·v·r + Xrr·r² + Xvvvv·v⁴
  Y_H = Yv·v + Yr·r + Yvvv·v³ + Yvvr·v²r + Yvrr·vr² + Yrrr·r³
  N_H = Nv·v + Nr·r + Nvvv·v³ + Nvvr·v²r + Nvrr·vr² + Nrrr·r³
  X_P = (1-tp)·T',  T' = KT(J)·n²·DP⁴/Lpp² (prime 化在 prop_force 内完成)
  舵: FN' = fα·(AR/(Lpp·d))·UR'²·sin(αR),  fα = 6.07·Λ/(Λ+2.07)
      XR = -(1-tR)·FN·sinδ, YR = (1+aH)·FN·cosδ, NR = (xR+aH·xH)·FN·cosδ·eps_暂用
      (NR 用 (xR' + aH·xH')·(1+aH)^{-1}·YR 的 Y&Y2015 标准式, 见 rudder_force)

扩展真值项 ext_forces (V2/V3/V4):
  Y_ext = cDv·D^α v ;  N_ext = cDr·D^α r ;  X_ext = 0
  另有记忆项: N_mem = Σ A_j·∫exp(-(t-τ)/τj)·r(τ)dτ  (与 features.exp_conv 同实现)

RK4 子步 dt=0.5s, 输出 1 Hz. 历史依赖项 (分数阶/卷积) 在子步内用
"已积分历史 + 当前步显式" 近似 (dt=0.5s 误差 <0.5%, 测试覆盖).
"""
from __future__ import annotations

import numpy as np

from .coeff_kvlcc2 import ShipCoeffs, kvlcc2
from .features import exp_conv, gl_deriv

RHO = 1025.0          # 海水密度 kg/m^3
DEG = np.pi / 180.0


# ----------------------------------------------------------------------------
# 各模块力 (prime 无量纲; 输入 u,v,r prime 速度 [无量纲], delta 弧度)
# ----------------------------------------------------------------------------
def hull_forces(s: ShipCoeffs, u, v, r):
    """返回 (XH, YH, NH) — 速度多项式船体力 (Y&Y2015 式 3)."""
    XH = (-s.R0 * u * abs(u)
          + s.Xvv * v * v + s.Xvr * v * r + s.Xrr * r * r + s.Xvvvv * v ** 4)
    YH = (s.Yv * v + s.Yr * r + s.Yvvv * v ** 3
          + s.Yvvr * v * v * r + s.Yvrr * v * r * r + s.Yrrr * r ** 3)
    NH = (s.Nv * v + s.Nr * r + s.Nvvv * v ** 3
          + s.Nvvr * v * v * r + s.Nvrr * v * r * r + s.Nrrr * r ** 3)
    return XH, YH, NH


def wP_of(s: ShipCoeffs, u, vm, r):
    """Y&Y2015 式15/16: βP = β − xP'·r' (β=−vm/u 船中漂角);
    (1−wP)/(1−wP0) = 1 + (1−exp(−C1|βP|))·(C2−1), C1=2.0, C2: βP>0→1.6 / βP<0→1.1.
    返回 (betaP, wP).
    """
    beta_mid = -vm / max(abs(u), 0.1) * np.sign(u)
    xP_p = -0.5
    betaP = beta_mid - xP_p * r
    C1, C2 = 2.0, (1.6 if betaP > 0 else 1.1)
    ratio = 1.0 + (1.0 - np.exp(-C1 * abs(betaP))) * (C2 - 1.0)
    wP = 1.0 - (1.0 - s.wp0) * ratio
    return betaP, wP


def prop_force(s: ShipCoeffs, u, wP=None):
    """X_P' (prime). KT(J)=k0+k1J+k2J², J = u_dim(1-wP)/(n·DP).

    wP 由 wP_of 提供 (含机动修正); 缺省用 wP0.
    """
    if wP is None:
        wP = s.wp0
    u_dim = u * s.U0
    J = u_dim * (1.0 - wP) / (s.n_rps * s.DP)
    KT = s.k0 + s.k1 * J + s.k2 * J * J
    T_prime = 2.0 * s.n_rps ** 2 * s.DP ** 4 * KT / (s.Lpp * s.d * s.U0 ** 2)
    return (1.0 - s.tp) * T_prime


def rudder_force(s: ShipCoeffs, u, vm, r, delta_rad):
    """舵力 (Y&Y2015 式 17-25, 按原文全式, vm=船中侧速). 返回 (XR', YR', NR').

    β = −vm/u (船中漂角); βR = β − lR'·r' (lR'=-0.710)
    vR = −γR·U·βR → prime 下 vR' = −γR·(vm + lR'·r')·(U/u 归一在 βR 内)
       实现: vR' = −γR·(vm/u + lR'·r'/u)·u = −γR·(vm + lR'·r')  [prime 一致形式]
    γR 分段 (Y&Y2015 §4.4 实测): βR<0 → 0.395; βR>0 → 0.640
    uR = ε(1-wP)u·sqrt(η[1+κ(√(1+8KT/πJ²)−1)]²+(1−η)), η=DP/HR, κ=0.50
    FN' = fα·(AR/(Lpp·d))·UR'²·sin(αR), fα=2.747 (Tab.3)
    XR=−(1−tR)FN·sinδ; YR=−(1+aH)FN·cosδ; NR=−(xR'+aH·xH')·FN·cosδ
    """
    betaP, wP = wP_of(s, u, vm, r)
    u_dim = u * s.U0
    J = u_dim * (1.0 - wP) / (s.n_rps * s.DP)
    KT = s.k0 + s.k1 * J + s.k2 * J * J
    sigmaT = -1.0 + np.sqrt(1.0 + 8.0 * KT / (np.pi * max(J, 1e-3) ** 2)) if J > 1e-3 else 1.0
    kappa = 0.50
    eta_PR = s.DP / s.HR
    boost = 1.0 + kappa * sigmaT
    uR = s.eps * (1.0 - wP) * u * np.sqrt(eta_PR * boost ** 2 + (1.0 - eta_PR))
    # vR: Y&Y2015 式23/24 角度形式: vR = U·γR·βR, βR = β − lR'·r', β = atan(−vm/u)
    # prime: vR' = γR·βR (β 为船中漂角弧度, 含 ÷u — 与 βP 同口径)
    beta_mid = np.arctan2(-vm, max(abs(u), 0.1) * np.sign(u))
    betaR = beta_mid - s.lR * r
    gammaR = 0.640 if betaR > 0 else 0.395
    vR = gammaR * betaR
    UR = np.sqrt(uR ** 2 + vR ** 2) + 1e-9
    alphaR = delta_rad - np.arctan2(vR, uR)
    FN = s.f_alpha * (s.AR / (s.Lpp * s.d)) * UR ** 2 * np.sin(alphaR)
    XR = -(1.0 - s.tR) * FN * np.sin(delta_rad)
    YR = -(1.0 + s.aH) * FN * np.cos(delta_rad)
    NR = -(s.xR + s.aH * s.xH) * FN * np.cos(delta_rad)
    return XR, YR, NR


# ----------------------------------------------------------------------------
# 扩展真值项 (V2/V3/V4) — 与 features.py 同实现
# ----------------------------------------------------------------------------
class ExtHistory:
    """仿真过程中维护历史序列, 计算 D^α / 卷积 扩展力.

    与 features.gl_deriv / features.exp_conv 的离散化保持一致:
    GL 用同款权系数 (截断窗), exp_conv 用同款递推 — 真值生成与特征提取闭环.
    """

    def __init__(self, s: ShipCoeffs, dt: float, window: int = 200):
        self.s = s
        self.dt = dt
        self.window = window
        self.v_hist: list = []
        self.r_hist: list = []
        # exp_conv 状态 (递推)
        self.mem_states = [0.0 for _ in s.mem_A]

    def push(self, v, r):
        self.v_hist.append(float(v))
        self.r_hist.append(float(r))
        for j, (A, tau) in enumerate(self.s.mem_A):
            a = np.exp(-self.dt / tau)
            gain = (1.0 - a) * tau * 0.5          # 梯形-精确核 (与 features.exp_conv 一致)
            r_prev = self.r_hist[-2] if len(self.r_hist) >= 2 else float(r)
            self.mem_states[j] = self.mem_states[j] * a + gain * (r_prev + float(r))

    def forces(self):
        """返回 (Xe, Ye, Ne) — 当前时刻扩展力 (prime)."""
        s = self.s
        Ye = 0.0
        Ne = 0.0
        if s.cDv != 0.0 and len(self.v_hist) >= 2:
            Ye += s.cDv * _gl_last(self.v_hist, self.dt, s.alpha_v, self.window)
        if s.cDr != 0.0 and len(self.r_hist) >= 2:
            Ne += s.cDr * _gl_last(self.r_hist, self.dt, s.alpha_r, self.window)
        for j, (A, tau) in enumerate(s.mem_A):
            Ne += A * self.mem_states[j]
        return 0.0, Ye, Ne


def _gl_last(hist: list, dt: float, alpha: float, window: int) -> float:
    """GL 分数阶导数 (只算序列末点), 与 features.gl_deriv 权系数一致."""
    m = min(window, len(hist))
    w = np.empty(m)
    w[0] = 1.0
    for k in range(1, m):
        w[k] = w[k - 1] * (1.0 - (alpha + 1.0) / k)
    tail = np.asarray(hist[-m:], dtype=float)
    return float(np.dot(w[::-1] * 0 + w[::-1], tail) * dt ** (-alpha))  # w 反转后与 tail(旧→新) 点乘: 注意方向


# ----------------------------------------------------------------------------
# 右端项与 RK4
# ----------------------------------------------------------------------------
def rhs(s: ShipCoeffs, state, delta_rad, ext: ExtHistory):
    """状态导数. state=[u, vm, r, psi, x, y] (u/vm 为 prime, vm=船中侧速, Y&Y2015 式4).

    严格按 Y&Y2015: 状态变量为 (u', vm', r'), 力多项式/舵来流全部作用于 vm/β.
        (m'+mx')u̇ − (m'+my')vm·r − xG'm'·r²          = X
        (m'+my')v̇m + (m'+mx')u·r + xG'm'·ṙ           = Y
        (IzG'+xG'²m'+Jz')ṙ + xG'm'(v̇m + u·r)          = Nm
    v̇m/ṙ 联立 2×2 求解. 重心侧速 v = vm + xG'·r (输出层换算).
    """
    u, vm, r, psi, x, y = state
    XH, YH, NH = hull_forces(s, u, vm, r)
    XP = prop_force(s, u, wP_of(s, u, vm, r)[1])
    XR, YR, NR = rudder_force(s, u, vm, r, delta_rad)
    Xe, Ye, Ne = ext.forces()
    X = XH + XP + XR + Xe
    Y = YH + YR + Ye
    Nm = NH + NR + Ne
    m = s.m_prime
    mT = m + s.mx
    nT = m + s.my
    xG = s.xG / s.Lpp
    IzT = s.Iz_prime + xG ** 2 * m + s.Jz
    ts = s.U0 / s.Lpp
    udot = (X + nT * vm * r + xG * m * r * r) / mT * ts
    b1 = Y - mT * u * r
    b2 = Nm - xG * m * u * r
    det = nT * IzT - (xG * m) ** 2
    vmdot = (b1 * IzT - (xG * m) * b2) / det * ts
    rdot = (nT * b2 - (xG * m) * b1) / det * ts
    psidot = r * s.U0 / s.Lpp
    xdot = (u * np.cos(psi) - vm * np.sin(psi)) * s.U0 / s.Lpp   # 轨迹单位: 船长 (vm≈v, 差 xG·r 二阶小)
    ydot = (u * np.sin(psi) + vm * np.cos(psi)) * s.U0 / s.Lpp
    return np.array([udot, vmdot, rdot, psidot, xdot, ydot])


def simulate(s: ShipCoeffs, maneuver: str, *, dt_out: float = 1.0,
             sub_per_out: int = 2, t_end=None, noise_rel=0.0, seed=0) -> dict:
    """RK4 仿真一个机动. maneuver ∈ {turn35p, turn35s, zig1010, zig2020}.

    dt 子步 = dt_out/sub_per_out (默认 0.5s). 返回 dict(t, u, v, r, delta, psi,
    x, y, 以及逐时刻模块力 XH,YH,NH,XP,XR,YR,NR, Ye_ext, Ne_ext, 与真值
    udot,vdot,rdot — 用于灰箱目标构造).
    """
    dt = dt_out / sub_per_out
    if maneuver.startswith("turn"):
        t_end = t_end or 900.0
    else:
        t_end = t_end or 600.0
    n_out = int(round(t_end / dt_out))
    delta_max = 35.0 * DEG if maneuver.startswith("turn") else 10.0 * DEG if "1010" in maneuver else 20.0 * DEG
    sign = -1.0 if maneuver.endswith("s") else 1.0          # turn35s 为负舵
    zig = maneuver.startswith("zig")

    ext = ExtHistory(s, dt)
    # 初始: 直航平衡 (u=1 prime, v=r=0), 舵 0
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ext.push(state[1], state[2])

    # Z 形试验舵令状态机 (标准语义): δ_cmd 初值 +D; ψ 触及 +D 时 δ_cmd→−D,
    # 触及 −D 时 δ_cmd→+D. 本模型实测符号链: δ>0 → NR>0 → r'>0 → ψ 增,
    # 即正舵使 ψ 向正 — 与标准试验约定一致, 初始 sign=+1.
    # 舵角按 delta_rate 限速趋近目标. 翻转判定带迟滞 (只响应对面阈值).
    psi_target = delta_max if zig else 0.0    # rad (ψ 阈值 = D, 与舵角数值同)
    zig_cmd = {"sign": 1.0, "delta": 0.0}
    delta_rate = 1.76 * DEG                    # rad/s 舵角限速 (Y&Y2015 §5.1: 1.76 deg/s fullscale)

    def step_rudder(psi, dt_):
        """每子步调用一次: 限速推进 + 翻转判定. 返回当前舵角."""
        if not zig:
            return sign * delta_max
        target = zig_cmd["sign"] * delta_max
        step = delta_rate * dt_
        if zig_cmd["delta"] < target:
            zig_cmd["delta"] = min(zig_cmd["delta"] + step, target)
        elif zig_cmd["delta"] > target:
            zig_cmd["delta"] = max(zig_cmd["delta"] - step, target)
        if zig_cmd["sign"] > 0 and psi >= psi_target:
            zig_cmd["sign"] = -1.0
        elif zig_cmd["sign"] < 0 and psi <= -psi_target:
            zig_cmd["sign"] = 1.0
        return zig_cmd["delta"]

    out = {k: [] for k in ("t", "u", "v", "r", "delta", "psi", "x", "y",
                           "XH", "YH", "NH", "XP", "XR", "YR", "NR",
                           "Ye_ext", "Ne_ext", "udot", "vdot", "rdot")}
    t = 0.0
    for i in range(n_out):
        # 当前输出点: 记录 (输出间隔内舵令已由子步推进到位, 直接读)
        u, v, r, psi, x, y = state          # v 即 vm (船中侧速, Y&Y2015 状态)
        XH, YH, NH = hull_forces(s, u, v, r)
        XP = prop_force(s, u, wP_of(s, u, v, r)[1])
        d_out = zig_cmd["delta"] if zig else sign * delta_max
        XR, YR, NR = rudder_force(s, u, v, r, d_out)
        Xe, Ye, Ne = ext.forces()
        deriv = rhs(s, state, d_out, ext)
        for k, val in zip(out.keys(),
                          (t, u, v, r, d_out, psi, x, y,
                           XH, YH, NH, XP, XR, YR, NR, Ye, Ne,
                           deriv[0], deriv[1], deriv[2])):
            out[k].append(float(val))
        # 子步 RK4 (history 依赖项显式: 子步中 push 一次; 舵令每子步推进一次)
        for _ in range(sub_per_out):
            d_now = step_rudder(state[3], dt)
            k1 = rhs(s, state, d_now, ext)
            st2 = state + 0.5 * dt * k1
            ext.push(st2[1], st2[2])
            k2 = rhs(s, st2, d_now, ext)
            st3 = state + 0.5 * dt * k2
            ext.push(st3[1], st3[2])
            k3 = rhs(s, st3, d_now, ext)
            st4 = state + dt * k3
            ext.push(st4[1], st4[2])
            k4 = rhs(s, st4, d_now, ext)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ext.push(state[1], state[2])
            t += dt

    for k in out:
        out[k] = np.asarray(out[k])

    # 噪声注入 (观测): u,v,r,psi,delta 各加 rel·std 高斯
    if noise_rel > 0:
        rng = np.random.default_rng(seed)
        for k in ("u", "v", "r", "psi", "delta"):
            sig_std = float(np.std(out[k]))
            out[k] = out[k] + rng.normal(0.0, noise_rel * sig_std, size=out[k].shape)
    return out


