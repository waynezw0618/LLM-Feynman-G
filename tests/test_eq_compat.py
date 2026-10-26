"""lgf.eq_compat 兼容层单测: 物理模板实例化 + EQ:/params[] 方言转换 + DSL 往返.

运行: python tests/test_eq_compat.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lgf import eq_compat as EC
from lgf.dsl import parse_dsl
from lgf.compiler import CompiledGraph

PASS = 0; FAIL = 0; FAILED = []


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"[PASS] {name}")
    except Exception as e:
        FAIL += 1; FAILED.append(name)
        import traceback; traceback.print_exc()
        print(f"[FAIL] {name}: {e}")


def _compilable(dsl: str, var_names):
    """DSL 必须能被 parse_dsl 解析且 CompiledGraph 可编译+可求值."""
    g = parse_dsl(dsl, list(var_names))
    cg = CompiledGraph(g, backend="numpy")
    theta = np.array([p.init for p in g.params()], dtype=float)
    X = np.random.default_rng(0).uniform(0.5, 1.5, (8, len(var_names)))
    y = cg.eval_case(theta[:cg.n_g + cg.n_c], X)
    assert y.shape == (8,), f"shape {y.shape}"


# --------------------------------------------------------------------------
# 1. 物理上下文模板
# --------------------------------------------------------------------------
def test_context_templates_instantiate():
    vars2 = ["m", "k"]
    sk = EC.context_aware_skeletons("harmonic oscillator frequency", vars2, 4)
    assert sk, "oscillator 未产出模板"
    for s in sk:
        _compilable(s, vars2)


def test_context_templates_energy_momentum():
    for ctx, vars_ in [("kinetic energy of a particle", ["m", "v"]),
                       ("momentum of a body", ["m", "v"]),
                       ("stefan boltzmann radiation", ["A", "T"]),
                       ("ideal gas pressure", ["P", "V", "n", "T"])]:
        sk = EC.context_aware_skeletons(ctx, vars_, 4)
        assert sk, f"{ctx} 未产出模板"
        for s in sk:
            _compilable(s, vars_)


def test_context_templates_separable():
    sk = EC.context_aware_skeletons("multiplicative separability f = g*h", ["a", "b"], 4)
    sk2 = EC.context_aware_skeletons("additive separab f ≈ g+h", ["a", "b"], 4)
    assert sk and sk2, "可分模板未产出"


def test_template_var_count_guard():
    # lens 需要 3 变量; 只给 1 个变量时应返回 None (被过滤)
    sk = EC.context_aware_skeletons("lens focal optics", ["x"], 4)
    assert sk == [], f"变量不足时应无模板, 得到 {sk}"


# --------------------------------------------------------------------------
# 2. EQ: 方言转换
# --------------------------------------------------------------------------
def test_eq_prefix_extraction():
    text = "some chat...\nEQ: params[0] * np.sqrt(v1 / v2)\nEQ: params[0] * v1 + params[1] * v2**2\n"
    lines = EC.extract_eq_lines(text)
    assert len(lines) == 2, lines
    assert "np.sqrt" in lines[0]


def test_eq_params_to_C():
    dsl = EC.eq_line_to_lgf_dsl("params[1] * np.exp(-v1 / params[2])", ["v1"])
    assert dsl is not None, "转换失败"
    assert "C2" in dsl and "C3" in dsl and "exp" in dsl, dsl
    _compilable(dsl, ["v1"])


def test_eq_sqrt_ratio():
    dsl = EC.eq_line_to_lgf_dsl("params[0] * np.sqrt(v1 / v2)", ["v1", "v2"])
    assert dsl and "sqrt" in dsl, dsl
    _compilable(dsl, ["v1", "v2"])


def test_eq_power_two():
    dsl = EC.eq_line_to_lgf_dsl("params[0] * v1 * v2**2", ["v1", "v2"])
    assert dsl and "^2" in dsl, dsl
    _compilable(dsl, ["v1", "v2"])


def test_eq_fractional_power_param():
    dsl = EC.eq_line_to_lgf_dsl("params[0] * (v1 ** params[1])", ["v1"])
    assert dsl and "log" in dsl and "exp" in dsl, dsl
    _compilable(dsl, ["v1"])


def test_eq_np_pi_literal():
    dsl = EC.eq_line_to_lgf_dsl("params[0] * np.pi * np.sqrt(v1 / v2)", ["v1", "v2"])
    assert dsl and "3.14159" in dsl, dsl
    _compilable(dsl, ["v1", "v2"])


def test_eq_brace_placeholders():
    dsl = EC.eq_line_to_lgf_dsl("params[0] * {v0} * {v1}", ["a", "b"])
    assert dsl and "a" in dsl and "b" in dsl, dsl
    _compilable(dsl, ["a", "b"])


def test_eq_batch_and_reject_unknown():
    text = ("EQ: params[0]*v1\n"
            "EQ: params[0] * bogus_var\n"
            "EQ: params[0] / (params[1]*v1 + params[2]*v2)\n")
    out = EC.parse_eq_response(text, ["v1", "v2"])
    assert len(out) == 2, f"应通过 2 条 (拒绝未知变量), 得到 {out}"
    for d in out:
        _compilable(d, ["v1", "v2"])


def test_eq_inverse_power():
    dsl = EC.eq_line_to_lgf_dsl("params[0] / (v1 ** 2)", ["v1"])
    assert dsl and "^2" in dsl, dsl
    _compilable(dsl, ["v1"])


def test_numeric_equivalence_original_vs_lgf():
    """数值等价: 同一 (v1,v2) 下, 原方言表达式 == 转换后 LGF DSL 的取值."""
    var_names = ["v1", "v2"]
    expr = "params[0] * v1 * v2**2 + params[1] * np.sqrt(v1 / v2)"
    dsl = EC.eq_line_to_lgf_dsl(expr, var_names)
    assert dsl is not None
    # 原表达式的 numpy 求值
    rng = np.random.default_rng(3)
    X = rng.uniform(0.5, 2.0, (20, 2))
    v1, v2 = X[:, 0], X[:, 1]
    p0, p1 = 1.7, 0.4
    y_ref = p0 * v1 * v2**2 + p1 * np.sqrt(v1 / v2)
    # LGF 侧
    g = parse_dsl(dsl, var_names)
    cg = CompiledGraph(g, backend="numpy")
    names = [p.name for p in g.params()]
    assert len(names) == 2, names
    # theta 顺序 = 参数遍历序; 显式按 C1/C2 赋值
    theta = np.zeros(2)
    for i, nm in enumerate(names):
        if nm.endswith("C1") or nm == "C1":
            theta[i] = p0
        else:
            theta[i] = p1
    y_lgf = cg.eval_case(theta, X)
    err = float(np.max(np.abs(y_ref - y_lgf)))
    assert err < 1e-8, f"数值不等价: max|err|={err}"


def main():
    tests = [
        ("物理模板: 振子实例化+可编译", test_context_templates_instantiate),
        ("物理模板: 能量/动量/辐射/气体", test_context_templates_energy_momentum),
        ("物理模板: 加性/乘性可分", test_context_templates_separable),
        ("物理模板: 变量数不足守卫", test_template_var_count_guard),
        ("EQ: 前缀抽取", test_eq_prefix_extraction),
        ("EQ: params[i] -> C#", test_eq_params_to_C),
        ("EQ: sqrt(比值)", test_eq_sqrt_ratio),
        ("EQ: 平方项", test_eq_power_two),
        ("EQ: 分数幂(params 指数)", test_eq_fractional_power_param),
        ("EQ: np.pi 数值化", test_eq_np_pi_literal),
        ("EQ: {v0} 花括号占位", test_eq_brace_placeholders),
        ("EQ: 批量+未知变量拒绝", test_eq_batch_and_reject_unknown),
        ("EQ: 负幂 -> inv", test_eq_inverse_power),
        ("数值等价: 原方言 vs LGF DSL", test_numeric_equivalence_original_vs_lgf),
    ]
    for nm, fn in tests:
        check(nm, fn)
    print(f"\n===== {PASS} passed, {FAIL} failed"
          + (f", failed={FAILED}" if FAILED else "") + " =====")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
