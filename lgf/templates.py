"""lgf.templates — 模板库, 来自 GraphED + 经典本构/物理模型.

每个模板返回 (Graph, slots_meta) ; slots_meta 描述槽位填法 (来自哪个变量).
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple

from .graph import Graph, Node, Param, var, const, add, mul, sub, div, scale, power
from .graph import expn, logn, sinn, cosn, tanhn, sqrtn, invn, negn
from .graph import global_param as gpar, case_param as kpar


# 工具: 用某变量构造多项式
def _poly(node_var_index: int, degree: int, coeffs_init: List[float] = None) -> Node:
    coeffs_init = coeffs_init or [1.0] * (degree + 1)
    terms = []
    for k, c0 in enumerate(coeffs_init):
        p = gpar(f"_p{node_var_index}_{k}", init=c0)
        if k == 0:
            terms.append(scale(const(1.0), p))
        elif k == 1:
            terms.append(scale(var(node_var_index), p))
        else:
            terms.append(scale(power(var(node_var_index), k), p))
    return add(*terms) if len(terms) >= 2 else terms[0]


def tpl_saturating(slot_var: int, name: str = "sat") -> Tuple[Graph, Dict]:
    """y = C1*slot / (C2 + slot)."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 1.0)
    g = Graph(
        mul(scale(var(slot_var), pC1),
            div_node := div(const(1.0), add(scale(var(slot_var), gpar("C3", 1.0)), scale(const(1.0), pC2)))),
        [f"x{slot_var+1}"], name=name)
    # 简化: 把上面的 div_node 错误; 重写
    num = scale(var(slot_var), pC1)
    den = add(var(slot_var), scale(const(1.0), pC2))
    g = Graph(div(num, den), [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "saturating", "formula": "C1*slot/(C2+slot)"}


def tpl_esat(slot_var: int, name: str = "esat") -> Tuple[Graph, Dict]:
    """y = C1 * exp(1/(C2*slot + C3)).   论文 DIF 形式."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 0.5); pC3 = gpar("C3", 0.3)
    num = const(1.0)
    den = add(scale(var(slot_var), pC2), scale(const(1.0), pC3))
    inner = invn(den)
    g = Graph(scale(expn(inner), pC1), [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "esat"}


def tpl_hollomon(slot_var: int, name: str = "holl") -> Tuple[Graph, Dict]:
    """y = C1 * slot^C2  (借助 exp(C2*log(slot)) 实现)."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 0.3)
    # slot^C2 = exp(C2 * log(slot))
    inner = expn(scale(logn(var(slot_var)), pC2))
    g = Graph(scale(inner, pC1), [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "hollomon"}


def tpl_voce(slot_var: int, name: str = "voce") -> Tuple[Graph, Dict]:
    """y = C1 - C2*exp(-C3*slot)."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 0.5); pC3 = gpar("C3", 0.3)
    g = Graph(sub(scale(const(1.0), pC1),
                   scale(expn(scale(negn(var(slot_var)), pC3)), pC2)),
              [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "voce"}


def tpl_gauss(slot_var: int, name: str = "gauss") -> Tuple[Graph, Dict]:
    """y = C1 * exp(-C2*(slot - C3)^2)."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 1.0); pC3 = gpar("C3", 0.0)
    delta = sub(var(slot_var), scale(const(1.0), pC3))
    sq = mul(delta, delta)
    inside = negn(scale(sq, pC2))
    g = Graph(scale(expn(inside), pC1), [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "gauss"}


def tpl_logistic(slot_var: int, name: str = "logistic") -> Tuple[Graph, Dict]:
    """y = C1 / (1 + exp(-(C2*slot + C3)))."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 1.0); pC3 = gpar("C3", 0.0)
    arg = negn(add(scale(var(slot_var), pC2), scale(const(1.0), pC3)))
    den = add(const(1.0), expn(arg))
    g = Graph(div(scale(const(1.0), pC1), den),
              [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "logistic"}


def tpl_trig_sin(slot_var: int, name: str = "sin") -> Tuple[Graph, Dict]:
    """y = C1 * sin(C2*slot + C3)."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 1.0); pC3 = gpar("C3", 0.0)
    g = Graph(scale(sinn(add(scale(var(slot_var), pC2), scale(const(1.0), pC3))), pC1),
              [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "trig_sin"}


def tpl_lin(slot_var: int, name: str = "lin") -> Tuple[Graph, Dict]:
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 0.0)
    g = Graph(add(scale(var(slot_var), pC1), scale(const(1.0), pC2)),
              [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "linear"}


def tpl_rational(slot_var: int, name: str = "rational") -> Tuple[Graph, Dict]:
    """y = slot / (C1*slot + C2)."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 1.0)
    den = add(scale(var(slot_var), pC1), scale(const(1.0), pC2))
    g = Graph(div(var(slot_var), den), [f"x{slot_var+1}"], name=name)
    return g, {"slots": [slot_var], "kind": "rational"}


def tpl_mul2(var_i: int, var_j: int, name: str = "mul2") -> Tuple[Graph, Dict]:
    """y = C1 * var_i^C2 * var_j^C3  (借助 log 实现非整数幂)."""
    pC1 = gpar("C1", 1.0); pC2 = gpar("C2", 1.0); pC3 = gpar("C3", 1.0)
    term_i = expn(scale(logn(var(var_i)), pC2))
    term_j = expn(scale(logn(var(var_j)), pC3))
    g = Graph(scale(mul(term_i, term_j), pC1), [f"x{var_i+1}", f"x{var_j+1}"], name=name)
    return g, {"slots": [var_i, var_j], "kind": "mul2"}


# ===== 模板注册表 =====
TEMPLATE_REGISTRY: Dict[str, Callable[..., Tuple[Graph, Dict]]] = {
    "lin": tpl_lin,
    "sat": tpl_saturating,
    "esat": tpl_esat,
    "holl": tpl_hollomon,
    "voce": tpl_voce,
    "gauss": tpl_gauss,
    "logistic": tpl_logistic,
    "sin": tpl_trig_sin,
    "rational": tpl_rational,
    "mul2": tpl_mul2,
}


def list_templates() -> List[str]:
    return list(TEMPLATE_REGISTRY.keys())


def build(name: str, *args, **kwargs) -> Tuple[Graph, Dict]:
    if name not in TEMPLATE_REGISTRY:
        raise KeyError(f"unknown template: {name}")
    return TEMPLATE_REGISTRY[name](*args, **kwargs)