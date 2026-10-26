"""LGF 全模块单元测试套件.

覆盖: 编译器梯度 / 参数拟合恢复 / 先验检验 / GP遗传算子 / 模板 /
归因 / LLM算子(启发式后端) / 符号等价 / 多案例参数 / 端到端冒烟.

运行: python tests/test_lgf.py
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp

from lgf.graph import (Graph, is_within_complexity,
                       var as mk_var, const as mk_const, add as mk_add,
                       scale, expn, global_param as gpar)
from lgf.dsl import parse_dsl
from lgf.compiler import CompiledGraph
from lgf.paramfit import fit_joint, evaluate
from lgf.priors import run_priors
from lgf.templates import TEMPLATE_REGISTRY, build, list_templates
from lgf import gp as gp_ops
from lgf import llm_ops
from lgf import attribution
from lgf.benchmarks import REGISTRY, make_cases
from lgf.pipeline_lgf import run_lgf
from lgf.llm_ops import make_client

PASS = 0; FAIL = 0; FAILED = []


def check(name, fn):
    global PASS, FAIL
    t0 = time.time()
    try:
        fn()
        PASS += 1
        print(f"[PASS] {name}  ({time.time()-t0:.1f}s)")
    except Exception:
        FAIL += 1; FAILED.append(name)
        print(f"[FAIL] {name}")
        traceback.print_exc()


# ============================================================================
# 1. 编译器: 解析梯度 vs 数值有限差分
# ============================================================================
def _named_graph():
    """程序化构造: y = C1*x1 + C2*exp(C3*x2).
    注意: Graph 构造时参数会按遍历序重命名 (右子树先), 因此语义值携带在 Param.init 上.
    """
    A = gpar("A", 0.7); B = gpar("B", -1.3); C = gpar("C", 0.4)
    g = Graph(mk_add(scale(mk_var(0), A),
                     scale(expn(scale(mk_var(1), C)), B)),
              ["x1", "x2"])
    return g, (A, B, C)


def test_compiler_gradcheck():
    g, (A, B, C) = _named_graph()
    cg = CompiledGraph(g, backend="numpy")
    rng = np.random.default_rng(0)
    X = rng.uniform(0.2, 1.5, (12, 2))
    theta = np.array([p.init for p in g.params()], dtype=float)
    G = cg.grad_case(theta, X)                 # (P, N)
    eps = 1e-6
    for j in range(len(theta)):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        num = (cg.eval_case(tp, X) - cg.eval_case(tm, X)) / (2 * eps)
        err = np.max(np.abs(num - G[j]))
        assert err < 1e-5, f"grad mismatch param {j}: {err}"


def test_compiler_vs_numpy():
    g, (A, B, C) = _named_graph()
    cg = CompiledGraph(g, backend="numpy")
    X = np.array([[0.5, 1.0], [1.2, 0.3]])
    theta = np.array([p.init for p in g.params()], dtype=float)
    yp = cg.eval_case(theta, X)
    ref = 0.7 * X[:, 0] + (-1.3) * np.exp(0.4 * X[:, 1])
    assert np.max(np.abs(yp - ref)) < 1e-12, np.max(np.abs(yp - ref))


# ============================================================================
# 2. 参数拟合: 常数恢复
# ============================================================================
def test_paramfit_linear_recovery():
    # 真值 y = 2*x1 + 1.5*x2^2 (无噪声, 精确恢复; 用符号替换做语义校验,
    # 因为 Graph 构造会按遍历序重命名参数, 不能依赖源文本参数名)
    bm = REGISTRY["add_sep"]()
    cases = make_cases(bm, n_per_case=120, noise_rel=0.0, noise_abs=0.0, seed=0)
    g = parse_dsl("C1 * x1 + C2 * x2^2", ["x1", "x2"])
    r = fit_joint(g, cases, n_starts=6, maxiter=150, seed=0, time_budget_s=20)
    assert r.loss < 1e-8, f"loss={r.loss}"
    e = g.to_sympy()
    subs = {sp.Symbol(p.name, real=True): float(r.theta[i])
            for i, p in enumerate(g.params())}
    e2 = sp.expand(e.subs(subs))
    x1, x2 = sp.symbols("x1 x2", real=True)
    diff = sp.simplify(e2 - (2 * x1 + 1.5 * x2 ** 2))
    assert diff == 0, f"语义不等价: {e2}"


def test_paramfit_template_esat():
    # esat 模板拟合 bm_esat: y = 1.2*exp(1/(0.5x+0.3))
    bm = REGISTRY["esat"]()
    cases = make_cases(bm, n_per_case=150, noise_rel=0.01, seed=0)
    g, meta = build("esat", 0)
    r = fit_joint(g, cases, n_starts=8, maxiter=200, seed=0, time_budget_s=30)
    assert r.loss < 0.01, f"loss={r.loss}"
    m = evaluate(g, r.theta, cases)
    r2 = [v for k, v in m.items() if k.endswith("_r2")]
    assert min(r2) > 0.99, m


# ============================================================================
# 3. Feynman 先验: 可分性 / 对称性检测
# ============================================================================
def test_priors_multiplicative():
    bm = REGISTRY["mul_sep"]()     # y = 4*x1*x2^2
    cases = make_cases(bm, n_per_case=300, noise_rel=0.0, noise_abs=0.0, seed=0)
    rep, pc = run_priors(cases, bm.var_names, sym_tol=0.08, sep_tol=0.2)
    assert pc.multiplicative_partitions, f"未检出乘性可分: {rep['separability']}"
    assert not pc.additive_partitions, f"乘性函数被误判为加性可分: {rep['separability']}"


def test_priors_additive():
    bm = REGISTRY["add_sep"]()     # y = 2*x1 + 1.5*x2^2
    cases = make_cases(bm, n_per_case=300, noise_rel=0.0, noise_abs=0.0, seed=0)
    rep, pc = run_priors(cases, bm.var_names, sym_tol=0.08, sep_tol=0.2)
    assert pc.additive_partitions, f"未检出加性可分: {rep['separability']}"
    assert not pc.multiplicative_partitions, f"加性函数被误判为乘性可分: {rep['separability']}"


# ============================================================================
# 4. GP 遗传算子: 复杂度约束 + DSL 可序列化
# ============================================================================
def test_gp_random_valid():
    rng = np.random.default_rng(1)
    for _ in range(40):
        g = gp_ops.random_graph(rng, ["x1", "x2"], max_edges=12, max_params=6, max_depth=5)
        assert is_within_complexity(g, 12, 6, 5), g.complexity()
        g2 = parse_dsl(g.to_dsl(), ["x1", "x2"])   # round-trip
        assert g2 is not None


def test_gp_mutate_crossover_valid():
    rng = np.random.default_rng(2)
    var_names = ["x1", "x2"]
    pop = [gp_ops.random_graph(rng, var_names, 12, 6, 5) for _ in range(10)]
    for g in pop:
        g.loss = float(rng.uniform(0.1, 1.0))
    for _ in range(30):
        parent = pop[rng.integers(0, len(pop))]
        child = gp_ops.mutate(parent, rng, var_names,
                              max_edges=12, max_params=6, max_depth=5)
        assert is_within_complexity(child, 12, 6, 5), child.complexity()
        assert parse_dsl(child.to_dsl(), var_names) is not None
    a, b = pop[0], pop[1]
    c = gp_ops.crossover(a, b, rng, 12, 6, 5)
    if c is not None:
        assert is_within_complexity(c, 12, 6, 5), c.complexity()
        assert parse_dsl(c.to_dsl(), var_names) is not None


def test_gp_select_sorted():
    rng = np.random.default_rng(3)
    pop = [gp_ops.random_graph(rng, ["x1"], 8, 4, 4) for _ in range(8)]
    for i, g in enumerate(pop):
        g.loss = 8.0 - i
    surv = gp_ops.select(pop, keep_ratio=0.5)
    losses = [g.loss for g in surv]
    assert losses == sorted(losses)
    assert len(surv) == 4


# ============================================================================
# 5. 模板库: 全部可构建可求值
# ============================================================================
def test_templates_all():
    rng = np.random.default_rng(4)
    X = rng.uniform(0.5, 1.5, (10, 2))
    for name in list_templates():
        if name == "mul2":
            g, meta = build(name, 0, 1)      # mul2 需要两个变量下标
        else:
            g, meta = build(name, 0)
        cg = CompiledGraph(g, backend="numpy")
        n_par = cg.n_g + cg.n_c
        theta = rng.uniform(0.5, 1.5, n_par)
        yp = cg.eval_case(theta, X[:, :len(g.var_names)])
        assert np.all(np.isfinite(yp)), f"template {name} produced non-finite"


# ============================================================================
# 6. 误差归因
# ============================================================================
def test_attribution():
    bm = REGISTRY["esat"]()
    cases = make_cases(bm, n_per_case=100, noise_rel=0.01, seed=0)
    g = parse_dsl("C1 * x1 + C2", ["x1"])   # 故意用错误结构
    r = fit_joint(g, cases, n_starts=3, maxiter=50, seed=0, time_budget_s=10)
    rep = attribution.analyze(g, r.theta, cases, ["x1"])
    assert isinstance(rep, dict)
    assert len(rep.get("text_summary", "")) > 0
    # 节点归因条目存在且带相关度
    nodes = rep.get("per_node") or {}
    assert len(nodes) > 0, rep.keys()
    for v in nodes.values():
        assert "pearson" in v and "shape" in v, v


# ============================================================================
# 7. LLM 算子层 (启发式后端, 无需网络)
# ============================================================================
def test_llm_heuristic():
    client = make_client("heuristic")
    bm = REGISTRY["esat"]()
    cases = make_cases(bm, n_per_case=80, noise_rel=0.01, seed=0)
    from lgf.priors import data_profile
    profile = data_profile(cases, list(bm.var_names))
    from lgf.priors import PriorConstraints
    dsls = llm_ops.G_INIT(client, profile, list_templates(), n=4)
    ok = 0
    for d in dsls:
        g = llm_ops.parse_dsl_safely(d, list(bm.var_names))
        if g is not None:
            ok += 1
    assert ok >= 1, f"G_INIT 无可解析 DSL: {dsls}"
    # M_GUIDE
    g0 = None
    for d in dsls:
        cand = llm_ops.parse_dsl_safely(d, list(bm.var_names))
        if cand is not None and len(cand.params()) > 0:
            g0 = cand; break
    if g0 is None:
        g0 = parse_dsl("C1 * x1 + C2", list(bm.var_names))
    rep = {"text_summary": "节点(1,0)残差相关0.8, 形态: 弯曲增长"}
    muts = llm_ops.M_GUIDE(client, [(g0, 0.5)], rep)
    assert isinstance(muts, list)
    # apply_mutations: perturb_param (用图上实际存在的参数名)
    pname = g0.params()[0].name
    before = g0.params()[0].init
    g1 = llm_ops.apply_mutations(g0, [{"type": "perturb_param",
                                       "param_id": pname, "magnitude": 0.5}],
                                 np.random.default_rng(0))
    assert g1.params()[0].init != before


# ============================================================================
# 8. 符号等价判定
# ============================================================================
def test_sympy_equivalence():
    g1 = parse_dsl("(C1 + C2) * x1", ["x1"])
    g2 = parse_dsl("C1 * x1 + x1 * C2", ["x1"])
    e1 = sp.expand(g1.to_sympy())
    e2 = sp.expand(g2.to_sympy())
    assert sp.simplify(e1 - e2) == 0
    # 数值等价: 设定相同参数
    m = {p.name: 1.7 for p in g1.params()}
    g1.set_param_inits(m); g2.set_param_inits(m)
    f1 = sp.lambdify([sp.Symbol("x1", real=True)], g1.to_sympy(), "numpy")
    f2 = sp.lambdify([sp.Symbol("x1", real=True)], g2.to_sympy(), "numpy")
    x = np.linspace(0.1, 2, 20)
    assert np.max(np.abs(f1(x) - f2(x))) < 1e-12


# ============================================================================
# 9. 多案例: 边特征(case 参数)按案例独立恢复
# ============================================================================
def test_multicase_case_params():
    bm = REGISTRY["esat_mc"]()   # 3 案例: C1*exp(1/(C2*x1+C3)), 案例参数独立
    cases = make_cases(bm, n_per_case=100, noise_rel=0.01, seed=0)
    assert len(cases) >= 2, f"多案例基准应 >=2 案例, got {len(cases)}"
    # 全案例级参数模板: y = K1 * exp(1/(K2*x1 + K3)) —— K# 按案例独立
    g = parse_dsl("K1 * exp(1 / (K2 * x1 + K3))", ["x1"])
    r = fit_joint(g, cases, n_starts=6, maxiter=200, seed=0, time_budget_s=60)
    m = evaluate(g, r.theta, cases)
    nmse = [v for k, v in m.items() if k.endswith("_nmse")]
    assert max(nmse) < 0.05, f"多案例拟合失败: {m}"
    # 各案例的 K1 应不同 (构造时使用了不同参数)
    n_g = len([p for p in g.params() if p.kind == "global"])
    n_c = len([p for p in g.params() if p.kind == "case"])
    ks = [r.theta[n_g + ci * n_c] for ci in range(len(cases))]
    assert len(set(np.round(ks, 4))) >= 2, f"案例参数未分化: {ks}"


# ============================================================================
# 10. 端到端 pipeline 冒烟 (启发式LLM + 先验 + 归因全开)
# ============================================================================
def test_pipeline_full():
    bm = REGISTRY["esat"]()
    cases = make_cases(bm, n_per_case=80, noise_rel=0.02, seed=0)
    from lgf.config import LGFConfig
    cfg = LGFConfig(population=10, generations=4, gp_starts=1, gp_maxiter=15,
                    final_starts=3, final_maxiter=50, time_budget_s=120,
                    sub_sample=80, use_llm=True, use_priors=True,
                    use_attribution=True, llm_backend="heuristic", seed=0,
                    stagnation_window=3, llm_every=2)
    out = run_lgf(cases, bm.var_names, cfg, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "outputs", "test"),
        make_client("heuristic"), name="smoke")
    assert out["best"]["loss"] < 0.05, f"best loss {out['best']['loss']}"
    assert out["top_final"] and out["top_final"][0]["loss"] == out["best"]["loss"]
    assert len(out["log"]) >= 1


# ============================================================================
if __name__ == "__main__":
    t_all = time.time()
    check("compiler 梯度 vs 有限差分", test_compiler_gradcheck)
    check("compiler vs 手写 numpy", test_compiler_vs_numpy)
    check("参数拟合: 线性+幂恢复", test_paramfit_linear_recovery)
    check("参数拟合: esat 模板", test_paramfit_template_esat)
    check("先验: 乘性可分检测", test_priors_multiplicative)
    check("先验: 加性可分检测", test_priors_additive)
    check("GP: 随机图复杂度约束+DSL往返", test_gp_random_valid)
    check("GP: 变异/交叉保持约束", test_gp_mutate_crossover_valid)
    check("GP: 选择排序", test_gp_select_sorted)
    check("模板库全量可求值", test_templates_all)
    check("误差归因", test_attribution)
    check("LLM 启发式后端", test_llm_heuristic)
    check("符号等价", test_sympy_equivalence)
    check("多案例 case 参数分化", test_multicase_case_params)
    check("端到端 pipeline 冒烟", test_pipeline_full)
    print(f"\n===== {PASS} passed, {FAIL} failed, {time.time()-t_all:.1f}s =====")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
