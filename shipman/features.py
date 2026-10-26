"""shipman.features — 特征工程库: 数值微分 / GL 分数阶 / 卷积 / 弱式积分.

设计原则 (与 mmg_sim.ext_forces 共用实现):
- "真值生成"与"候选特征"必须同一实现 — 保证参数恢复闭环 (tests: variants_roundtrip).
- 全部函数接受等间隔采样的 1D 序列 + dt, 返回同长度序列 (前 window 点不可信,
  由调用方配合 valid 掩码裁剪).
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# 预处理与数值微分
# ----------------------------------------------------------------------------
def savgol_smooth(x: np.ndarray, dt: float, win_s: float = 10.0, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay 平滑 (窗长按秒换算为奇数点数). 窗不足时降级为不处理."""
    from scipy.signal import savgol_filter
    n = int(round(win_s / dt))
    if n % 2 == 0:
        n += 1
    if n <= poly + 1 or n >= len(x):
        return np.asarray(x, dtype=float).copy()
    return savgol_filter(np.asarray(x, dtype=float), n, poly)


def central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """中心差分 dx/dt (端点单侧差分)."""
    x = np.asarray(x, dtype=float)
    d = np.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


# ----------------------------------------------------------------------------
# 分数阶 (Grünwald–Letnikov)
# ----------------------------------------------------------------------------
def gl_weights(alpha: float, n: int) -> np.ndarray:
    """GL 权系数 w0=1, wk = w(k-1)·(1-(α+1)/k), 长度 n (截断窗)."""
    w = np.empty(n, dtype=float)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (1.0 - (alpha + 1.0) / k)
    return w


def gl_deriv(x: np.ndarray, dt: float, alpha: float, window: int = 200) -> np.ndarray:
    """Grünwald-Letnikov 分数阶导数 (截断窗): dt^{-α}·Σ_k w_k·x_{n-k}.

    alpha=1 时退化为加权后向差分 (w=[1,-1,0,...] 当窗足够长时逼近中心差分低阶精度).
    前 window 点置 NaN 由调用方裁剪.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = min(window, n)
    w = gl_weights(alpha, m) * (dt ** (-alpha))
    out = np.full(n, np.nan)
    # 向量化: 对每个 i≥m-1, out[i] = Σ w_k x_{i-k}
    xs = np.concatenate([np.full(m - 1, x[0]), x])   # 前填充, 保证前段也有值 (标记 NaN 区)
    for i in range(m - 1, n):
        out[i] = float(np.dot(w, xs[i:i + m][::-1]))
    return out


# ----------------------------------------------------------------------------
# 卷积 (记忆核)
# ----------------------------------------------------------------------------
def exp_conv(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """∫₀ᵗ exp(-(t-τ)/tau)·x(τ)dτ 递推实现 (梯形-精确核离散化, O(N), O(dt²)).

    c_n = a·c_{n-1} + τ(1-a)·(x_{n-1}+x_n)/2, a=e^{-dt/tau} — x 常数时稳态
    增益精确 = τ·x; x 线性时核内积分精确到二阶 (相位误差 O(dt²)).
    """
    x = np.asarray(x, dtype=float)
    a = np.exp(-dt / tau)
    gain = (1.0 - a) * tau * 0.5
    out = np.empty_like(x)
    c = 0.0
    x_prev = x[0]
    for i, xi in enumerate(x):
        c = c * a + gain * (x_prev + xi)
        out[i] = c
        x_prev = xi
    return out


def powerlaw_conv(x: np.ndarray, dt: float, beta: float, window: int = 500) -> np.ndarray:
    """幂律核卷积 ∫₀ᵗ (t-τ)^{-β}·x(τ)dτ (截断窗), β∈(0,1)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = min(window, n)
    # 核采样 k=1..m (跳过奇点 k=0, 用 k=0.5 中点近似)
    ks = np.arange(1, m + 1)
    kern = dt * (ks * dt) ** (-beta)
    out = np.full(n, np.nan)
    xs = np.concatenate([np.full(m - 1, x[0]), x])
    for i in range(m - 1, n):
        out[i] = float(np.dot(kern, xs[i - m + 1:i + 1][::-1]))
    return out


# ----------------------------------------------------------------------------
# 弱式积分 (路线 B: 测试函数积分, 消除导数)
# ----------------------------------------------------------------------------
def weakform_rows(t: np.ndarray, r: np.ndarray, n_test: int = 8):
    """生成 hat 测试函数 φ_l (紧支, 均匀布点).

    返回 (centers, phi_list): 每个 phi 为与 t 等长的 0/1 斜坡函数.
    """
    t = np.asarray(t, dtype=float)
    centers = np.linspace(t[0] + 0.15 * (t[-1] - t[0]),
                          t[-1] - 0.15 * (t[-1] - t[0]), n_test)
    phis = []
    width = 0.5 * (t[-1] - t[0]) / n_test
    for c in centers:
        phi = np.clip(1.0 - np.abs(t - c) / width, 0.0, None)
        phis.append(phi)
    return centers, phis
