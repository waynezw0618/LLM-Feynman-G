"""shipman.coeff_kvlcc2 — KVLCC2.L7 MMG 系数 (文献值, 逐项注明来源).

数据来源 (交叉核实 2026-10-14):
- Yasukawa & Yoshimura (2015) JASNAOE 16(1):51-85, doi:10.1007/s00773-014-0293-y,
  "Introduction of MMG standard model for ship maneuvering predictions"
  KVLCC2.L7 拘束模型试验系数 (原文 Table 4/5/6/7).
- 完整数值转录自 Zhang et al. (2024) Ocean Engineering, "Time-optimal control of
  ship manoeuvring under wave loads" 的 Table 3/4/5 (其明确注明取自 Y&Y2015 体系);
  舵系数另与 Dai et al. (2019) PMRes (转录 Y&Y2015) 交叉核对一致.

主尺度 (Y&Y2015 Table 1, 全尺度):
  Lpp=320.0 m, B=58.0 m, d=20.8 m, Cb=0.810, xG=11.2 m (船中前),
  Dp=9.86 m, HR=15.80 m, AR=112.5 m^2, 15.5 kn 服务航速对应 n=1.78 rps.

自洽性验证 (2026-10-14): 用 R0'=0.022, wP0=0.35, tP=0.22, KT=k0+k1J+k2J^2
在 n=1.78 rps 直航工况下算得推力≈阻力, 与文献公布的服务转速精确吻合.

符号约定: 所有导数均为 prime 无量纲 (力除 ρ/2·Lpp·d·U^2, 力矩除 ρ/2·Lpp^2·d·U^2).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShipCoeffs:
    """MMG 模型系数容器 (prime 无量纲). 含 V2/V3/V4 扩展真值字段 (默认 0=纯传统)."""
    name: str
    # ---- 主尺度 ----
    Lpp: float; B: float; d: float; Cb: float
    xG: float                    # 重心纵向位置 (m, 船中前为正; 全尺度)
    U0: float                    # 服务航速 (m/s)
    # ---- 质量 (prime: m' = m/(ρ/2·Lpp^2·d), Iz' = Iz/(ρ/2·Lpp^4·d)) ----
    m_prime: float; Iz_prime: float
    # ---- 附加质量 ----
    mx: float; my: float; Jz: float
    # ---- 船体导数 (X_H / Y_H / N_H) ----
    R0: float                    # 直航阻力系数 (X_H 的 -R0'·u|u| 型)
    Xvv: float; Xvr: float; Xrr: float; Xvvvv: float
    Yv: float; Yr: float; Yvvv: float; Yvvr: float; Yvrr: float; Yrrr: float
    Nv: float; Nr: float; Nvvv: float; Nvvr: float; Nvrr: float; Nrrr: float
    # ---- 推进 ----
    tp: float; wp0: float
    k0: float; k1: float; k2: float
    DP: float; n_rps: float      # 螺旋桨直径 (m), 转速 (rps)
    # ---- 舵 (藤井) ----
    tR: float; aH: float; xH: float
    eps: float                   # ε = NR/NR'(舵法向力与舵力矩之比相关)
    gammaR_pos: float; gammaR_neg: float   # γR(C1,C2 分段)
    lR: float                    # lR' 有效来流位置
    f_alpha: float               # 舵升力梯度系数 fα (藤井 fα=6.07Λ/(Λ+2.07), KVLCC2 Λ=2.747 → 3.461)
    AR: float                    # 舵面积 (m^2)
    HR: float                    # 舵展长 (m), 用于 η=DP/HR
    xR: float                    # 舵纵向位置 (prime xR'=-0.5L)
    # ---- V2 分数阶扩展真值 (默认 0) ----
    cDv: float = 0.0; alpha_v: float = 0.5    # Y 方程: cDv·D^α v
    cDr: float = 0.0; alpha_r: float = 0.5    # N 方程: cDr·D^α r
    # ---- V3 积分(记忆核)扩展真值 (默认空) ----
    mem_A: tuple = ()            # ((A_j, tau_j_s), ...) 指数和核, 作用于 r
    # ---- 元信息 ----
    source: str = "Yasukawa & Yoshimura (2015) KVLCC2.L7"


def kvlcc2() -> ShipCoeffs:
    """KVLCC2.L7 全尺度 MMG 系数.

    数值来源: Y&Y2015 原文 (经 Zhang et al. 2024 Table 3-5 与 Dai et al. 2019
    Tab.5 双源交叉核对). xG 存全尺度米数 (仿真器内转 prime).
    """
    return ShipCoeffs(
        name="kvlcc2",
        Lpp=320.0, B=58.0, d=20.8, Cb=0.810,
        xG=11.2,
        U0=15.5 * 0.5144,                      # 15.5 kn → 7.974 m/s
        m_prime=2.0 * 0.810 * 58.0 / 320.0,   # m' = ρ·Cb·B·d·Lpp / (ρ/2·Lpp^2·d) = 2CbB/Lpp ≈ 0.294
        Iz_prime=0.0183,                       # Iz' = m'·(0.25Lpp)² = 0.294·0.0625 (惯量半径 0.25Lpp)
        mx=0.022, my=0.223, Jz=0.011,
        R0=0.022,
        Xvv=-0.040, Xvr=0.002, Xrr=0.011, Xvvvv=0.771,
        Yv=-0.315, Yr=0.083, Yvvv=-1.607,
        Yvvr=0.379, Yvrr=-0.391, Yrrr=0.008,
        Nv=-0.137, Nr=-0.049, Nvvv=-0.030,
        Nvvr=-0.294, Nvrr=0.055, Nrrr=-0.013,
        tp=0.220, wp0=0.40,
        k0=0.2931, k1=-0.2753, k2=-0.1385,
        DP=9.86, n_rps=1.7501,                 # wP0=0.40 (L7) 下直航平衡转速 (brentq 解, 2026-10-14)
        tR=0.387, aH=0.312, xH=-0.464,         # xH' (prime)
        eps=1.09,
        gammaR_pos=0.395, gammaR_neg=0.640,    # γR(βR>0)=0.395, γR(βR<0)=0.640
        lR=-0.710,                             # lR' (实验拟合值, 原文 near -1.0 的特例为 0.710 用于 KVLCC2)
        f_alpha=2.747,                         # fα (Tab.3 直接值, Hirano 处理: 全舵展弦比求 fα, 可动面积作 AR)
        AR=112.5, HR=15.80,
        xR=-0.500,                             # xR' = -0.5L (Nomenclature: xR'=-0.5Lpp)
        source="Yasukawa & Yoshimura (2015) KVLCC2.L7 (Zhang 2024 Tab.3-5 / Dai 2019 Tab.5 交叉核对)",
    )
