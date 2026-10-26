"""smoke test: lgf.graph + lgf.dsl 基础正确性."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgf.graph import (
    Graph, var as mk_var, const as mk_const, mul, add, sub, div,
    expn, logn, sinn, tanhn, sqrtn, invn, negn, scale, power,
    global_param as gpar, case_param as kpar,
)
from lgf.dsl import parse_dsl, graph_to_dsl
import numpy as np
import sympy as sp


def eval_graph(g: Graph, Xs: dict) -> np.ndarray:
    """Naive eval: use sp.lambdify. 仅做单元测试使用, 生产用 compiler."""
    expr = g.to_sympy()
    var_syms = [sp.Symbol(v, real=True) for v in g.var_names]
    param_syms = [sp.Symbol(p.name, real=True) for p in g.params()]
    p_vals = [p.init for p in g.params()]
    fn = sp.lambdify([*var_syms, *param_syms], expr, "numpy")
    return fn(*[np.asarray(Xs[v]) for v in g.var_names], *p_vals)


def test_construction():
    g = Graph(add(mk_const(1.0), scale(mk_var(0), gpar("C1", 0.7))), ["x1", "x2"])
    assert g.count_params() == 1
    print("[OK] construction: 1 + C1*x1")


def test_dsl_parse_simple():
    text = "1 + C1 * x1"
    g = parse_dsl(text, ["x1", "x2"])
    assert g.count_params() == 1
    print("[OK] parse_dsl simple")


def test_dsl_parse_poly():
    text = """
    P1(x1) = C1 * x1 + C2 * x1^2 + C3
    y = C4 * P1(x1) + C5 * exp(C6 * x2)
    """
    g = parse_dsl(text, ["x1", "x2"])
    assert g.count_params() == 6  # C1..C6
    print("[OK] parse_dsl polynomial")


def test_dsl_round_trip():
    """构造一些图, 打印 DSL, 再解析回来, 数值上等价."""
    cases = []
    # case 1: 简单线性
    g1 = Graph(add(scale(mk_var(0), gpar("C1", 1.5)),
               scale(mk_var(1), gpar("C2", 2.0))),
               ["x1", "x2"])
    cases.append(("linear", g1))
    # case 2: 乘 + sin
    g2 = Graph(add(mul(mk_var(0), mk_var(1)),
                   scale(sinn(mk_var(0)), gpar("C3", 0.3))),
               ["x1", "x2"])
    cases.append(("mul+sin", g2))
    # case 3: exp(inv)
    g3 = Graph(scale(expn(invn(add(scale(mk_var(0), gpar("C1", 0.5)),
                                    scale(mk_const(1.0), gpar("C2", 0.3))))),
                     gpar("C3", 1.2)),
               ["x1"])
    cases.append(("esat", g3))

    rng = np.random.default_rng(42)
    for name, g in cases:
        text = graph_to_dsl(g)
        g2 = parse_dsl(text, g.var_names)
        # 将原图的 param init 传给 g2 以保证数值等价
        mapping = {p.name: p.init for p in g.params()}
        g2.set_param_inits(mapping)
        Xs = {v: rng.standard_normal(20) for v in g.var_names}
        v1 = eval_graph(g, Xs)
        v2 = eval_graph(g2, Xs)
        diff = np.max(np.abs(v1 - v2))
        assert diff < 1e-9, f"{name} roundtrip diff={diff}"
        print(f"[OK] round-trip {name}: max diff = {diff:.2e}")


def test_dsl_reject_garbage():
    try:
        parse_dsl("UNKNOWN(x1)", ["x1"])
    except SyntaxError:
        print("[OK] reject unknown function")
        return
    raise AssertionError("should have raised")


def test_subst_var():
    # P1(x) = 2*x + 1, P1(x2) -> 2*x2 + 1
    base = Graph(add(scale(mk_var(0), gpar("C1", 2.0)),
                    scale(mk_const(1.0), gpar("C2", 1.0))), ["x"])
    g2 = base.substitute_var(0, mk_var(0))  # 同名
    # var_names 不变, 仅替换子节点
    assert g2.count_params() == 2
    print("[OK] substitute_var")


if __name__ == "__main__":
    test_construction()
    test_dsl_parse_simple()
    test_dsl_parse_poly()
    test_dsl_reject_garbage()
    test_subst_var()
    test_dsl_round_trip()
    print("\nALL OK")