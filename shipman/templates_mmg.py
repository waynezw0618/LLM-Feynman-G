"""shipman.templates_mmg — MMG 模板动态注册 (不改 lgf/templates.py).

用法: import shipman.templates_mmg  →  lgf.templates.TEMPLATE_REGISTRY 即含 mmg_* 键.
"""
from __future__ import annotations

import lgf.templates as T
from lgf.graph import Graph, var, add, mul, global_param as gpar
from shipman.coeff_kvlcc2 import ShipCoeffs, kvlcc2


def tpl_mmg_poly3(name: str = "mmg_poly3"):
    """Y_H 骨架: C1·v + C2·r + C3·v³ + C4·v²r + C5·vr² + C6·r³.

    var_names 约定: ["v", "r", ...] (由 variants 供给的特征列名).
    槽位: 0=v, 1=r (在 2 变量最小集上构造; rc_sm_run 在更长 var_names 下
    重建时按索引替换).
    """
    v, r = 0, 1
    terms = []
    for (idx, power, cname) in [(v, 1, "C1"), (r, 1, "C2")]:
        node = var(idx)
        for _ in range(power - 1):
            node = mul(node, var(idx))
        p = gpar(cname, 1.0)
        terms.append(mul_with_param(node, p))
    for (a, b, cname) in [(v, v, v, "C3"), (v, v, r, "C4"), (v, r, r, "C5"), (r, r, r, "C6")]:
        pass  # 三次项构造见下 (直接展开)
    # 三次项: C3 v³ + C4 v²r + C5 vr² + C6 r³
    v3 = mul(mul(var(v), var(v)), var(v))
    v2r = mul(mul(var(v), var(v)), var(r))
    vr2 = mul(mul(var(v), var(r)), var(r))
    r3 = mul(mul(var(r), var(r)), var(r))
    for node, cname in [(v3, "C3"), (v2r, "C4"), (vr2, "C5"), (r3, "C6")]:
        terms.append(mul_with_param(node, gpar(cname, 1.0)))
    g = Graph(add(*terms), ["v", "r"], name=name)
    return g, {"slots": [v, r], "kind": "mmg_poly3", "formula": "C1v+C2r+C3v^3+C4v^2r+C5vr^2+C6r^3"}


def mul_with_param(node, p):
    from lgf.graph import Node
    return Node("mul", children=[(node, p)])


def tpl_mmg_damp(name: str = "mmg_damp"):
    """X_H 骨架: C1·uabsu (直航阻尼) + C2·v² + C3·vr + C4·r² (var_names 含预计算列)."""
    # 列名约定 (variants 供给): uabsu, vv, vr, rr 为预计算列索引 0..3
    terms = []
    for idx, cname in [(0, "C1"), (1, "C2"), (2, "C3"), (3, "C4")]:
        terms.append(mul_with_param(var(idx), gpar(cname, 1.0)))
    g = Graph(add(*terms), ["uabsu", "vv", "vr", "rr"], name=name)
    return g, {"slots": [0, 1, 2, 3], "kind": "mmg_damp"}


def tpl_mmg_mem(name: str = "mmg_mem"):
    """V3 记忆骨架: C1·v + C2·r + C3·conv_r_K1 + C4·conv_r_K2 (预计算列)."""
    terms = []
    for idx, cname in [(0, "C1"), (1, "C2"), (2, "C3"), (3, "C4")]:
        terms.append(mul_with_param(var(idx), gpar(cname, 1.0)))
    g = Graph(add(*terms), ["v", "r", "conv_r_K1", "conv_r_K2"], name=name)
    return g, {"slots": [0, 1, 2, 3], " kind": "mmg_mem"}


def _register():
    T.TEMPLATE_REGISTRY.update({
        "mmg_poly3": tpl_mmg_poly3,
        "mmg_damp": tpl_mmg_damp,
    })


_register()
