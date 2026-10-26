"""shipman.seeds — G_INIT 语义种子注入 (MMG 领域先验的程序化替身).

机制 (零 lgf 内核改动): 包装 llm_ops.G_INIT — 原调用输出后追加 MMG 结构
种子 DSL 列表. 与 pipeline 的 wall_seeds 机制 (P1d-B2) 同型: G_INIT 候选
不足时程序化补种, 解决 heuristic 后端无领域知识 + 26-43 列空间冷启动难.

设计原则: 种子只含"结构" (系数为 C# 自由参数), 不偷答案 — 真值系数仍需
paramfit 拟合; 跨变体种子共享多项式骨架, 分数阶/记忆项种子按变体追加.
"""
from __future__ import annotations

from typing import List

import lgf.llm_ops as llm_ops

# MMG 船体力多项式骨架 (Y/N 方程同构; X 方程为阻尼型)
POLY6 = "C1*vm + C2*r + C3*vm3 + C4*vm2r + C5*vmr2 + C6*r3"
POLY4 = "C1*vm + C2*r + C3*vm3 + C4*r3"
LIN2 = "C1*vm + C2*r"
X_DAMP = "C1*uabsu + C2*vmvm + C3*vmr + C4*rr"
X_DAMP_V4 = "C1*uabsu + C2*vmvm + C3*vmr + C4*rr + C5*vm4"

# 分数阶种子 (V2/V4): 基础多项式 + D^α 项 (α 由列名承载, 系数待拟合)
FRAC_SEEDS = [
    "C1*vm + C2*r + C3*dv_a05",
    "C1*vm + C2*r + C3*dr_a05",
    "C1*vm + C2*r + C3*vm3 + C4*dv_a05",
    "C1*vm + C2*r + C3*vm3 + C4*dr_a05",
    "C1*vm + C2*r + C3*dv_a03 + C4*dv_a05 + C5*dv_a07 + C6*dv_a09",
]
# 记忆核种子 (V3/V4)
MEM_SEEDS = [
    "C1*vm + C2*r + C3*conv_r_K1",
    "C1*vm + C2*r + C3*conv_v_K1",
    "C1*vm + C2*r + C3*conv_r_K1 + C4*conv_r_K2",
    "C1*vm + C2*r + C3*conv_r_K1 + C4*conv_r_PW",
    "C1*vm + C2*r + C3*conv_v_K1 + C4*conv_r_K1",
]


def mmg_seed_dsls(variant: str, eq: str) -> List[str]:
    """按变体与方程返回种子 DSL 列表."""
    base = [POLY6, POLY4, LIN2]
    if eq == "X":
        base = [X_DAMP, X_DAMP_V4, LIN2.replace("vm", "u").replace("r", "uabsu")]
    out = list(base)
    if variant in ("v2", "v4"):
        out += [s for s in FRAC_SEEDS if (eq == "Y" and "dv_" in s) or (eq == "N" and "dr_" in s)]
        out += [s for s in FRAC_SEEDS if "dv_a03" in s]     # 阶数扫描种子 (Y/N 通用给 dv)
    if variant in ("v3", "v4"):
        out += MEM_SEEDS
    if variant == "v4":
        # 组合种子
        out += ["C1*vm + C2*r + C3*dv_a05 + C4*conv_r_K1",
                "C1*vm + C2*r + C3*vm3 + C4*dv_a05 + C5*conv_r_K1"]
    return out


_orig_G_INIT = None


def patch_ginit_with_mmg_seeds(variant: str, eq: str):
    """包装 llm_ops.G_INIT: 追加 MMG 种子. 幂等 (重复调用先还原)."""
    global _orig_G_INIT
    if _orig_G_INIT is None:
        _orig_G_INIT = llm_ops.G_INIT
    seeds = mmg_seed_dsls(variant, eq)

    def wrapped(llm, profile, templates, n=8, **kw):
        out = _orig_G_INIT(llm, profile, templates, n=n, **kw)
        try:
            return list(out) + list(seeds)
        except Exception:
            return list(seeds)

    llm_ops.G_INIT = wrapped
    # pipeline_lgf 里是 "from . import llm_ops" 模块引用 → 同一对象, patch 生效
    return wrapped
