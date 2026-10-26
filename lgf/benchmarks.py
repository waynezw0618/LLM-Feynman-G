"""lgf.benchmarks — 基准题库 (合成 + 真实).

每个 benchmark: dict(name, var_names, n_vars, truth_expr, generate, n_cases, ...).
generate(case_idx) 返回 (X, y) 数组对, 服从一定噪声/采样策略.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Benchmark:
    name: str
    var_names: List[str]
    truth: str                 # 人类可读形式
    truth_fn: Callable         # f(X) -> y 真实函数 (numpy)
    sample: Callable           # sample(n, rng) -> X, y, domain_meta
    domain: Dict[str, tuple]   # 每个变量的取值区间 (lo, hi)
    n_cases: int = 1
    extra_cases: Optional[List[Callable]] = None  # 多案例时的采样函数列表
    source: str = "synthetic"
    notes: str = ""


def _add_noise(y: np.ndarray, rel: float = 0.01, abs_: float = 0.01, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return y + rng.normal(0, rel * np.std(y) + abs_, size=y.shape)


# ===== 合成基准 =====
def bm_poly_add():
    """y = x1^2 + 3*x1*x2 + 2*x2^2."""
    var_names = ["x1", "x2"]
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]
        return x1**2 + 3 * x1 * x2 + 2 * x2**2
    def sample(n, rng):
        X = rng.uniform(-1, 1, (n, 2))
        return X, f(X)
    return Benchmark("poly_add", var_names, "x1^2 + 3*x1*x2 + 2*x2^2", f, sample,
                     {"x1": (-1, 1), "x2": (-1, 1)})


def bm_mul_separable():
    """y = 4*x1*x2^2 (乘性可分)."""
    var_names = ["x1", "x2"]
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]
        return 4.0 * x1 * x2 ** 2
    def sample(n, rng):
        X = rng.uniform(0.5, 2.0, (n, 2))
        return X, f(X)
    return Benchmark("mul_sep", var_names, "4*x1*x2^2", f, sample,
                     {"x1": (0.5, 2.0), "x2": (0.5, 2.0)})


def bm_add_separable():
    """y = 2*x1 + 1.5*x2^2 (加性可分)."""
    var_names = ["x1", "x2"]
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]
        return 2.0 * x1 + 1.5 * x2 ** 2
    def sample(n, rng):
        X = rng.uniform(-1, 1, (n, 2))
        return X, f(X)
    return Benchmark("add_sep", var_names, "2*x1 + 1.5*x2^2", f, sample,
                     {"x1": (-1, 1), "x2": (-1, 1)})


def bm_rational():
    """y = 2*x1 / (x2 + 3)."""
    var_names = ["x1", "x2"]
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]
        return 2.0 * x1 / (x2 + 3.0)
    def sample(n, rng):
        X = rng.uniform(-2, 2, (n, 2))
        return X, f(X)
    return Benchmark("rational", var_names, "2*x1/(x2+3)", f, sample,
                     {"x1": (-2, 2), "x2": (-2, 2)})


def bm_esat():
    """y = 1.2*exp(1/(0.5*x1 + 0.3)).  论文 DIF 形式."""
    var_names = ["x1"]
    def f(X):
        x = X[:, 0]
        return 1.2 * np.exp(1.0 / (0.5 * x + 0.3))
    def sample(n, rng):
        X = rng.uniform(0.05, 1.5, (n, 1))
        return X, f(X)
    return Benchmark("esat", var_names, "1.2*exp(1/(0.5*x1+0.3))", f, sample,
                     {"x1": (0.05, 1.5)})


def bm_voce():
    """y = 100 - 80*exp(-2*x1)."""
    var_names = ["x1"]
    def f(X):
        return 100.0 - 80.0 * np.exp(-2.0 * X[:, 0])
    def sample(n, rng):
        X = rng.uniform(0, 2, (n, 1))
        return X, f(X)
    return Benchmark("voce", var_names, "100 - 80*exp(-2*x1)", f, sample,
                     {"x1": (0, 2)})


def bm_gauss():
    """y = exp(-(x1-1)^2/(2*0.5^2)) / (x2*sqrt(2π))."""
    var_names = ["x1", "x2"]
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]
        return np.exp(-((x1 - 1.0) ** 2) / (2 * 0.5 ** 2)) / (x2 * np.sqrt(2 * np.pi))
    def sample(n, rng):
        X = rng.uniform(0.1, 3, (n, 2))
        return X, f(X)
    return Benchmark("gauss", var_names, "exp(-(x1-1)^2/(2*0.5^2))/(x2*sqrt(2π))", f, sample,
                     {"x1": (0.1, 3), "x2": (0.1, 3)})


def bm_hollomon():
    """y = 50*x1^0.3 (Hollomon 幂律)."""
    var_names = ["x1"]
    def f(X):
        return 50.0 * np.power(X[:, 0], 0.3)
    def sample(n, rng):
        X = rng.uniform(0.1, 2, (n, 1))
        return X, f(X)
    return Benchmark("hollomon", var_names, "50*x1^0.3", f, sample, {"x1": (0.1, 2)})


def bm_logistic():
    """y = 1/(1+exp(-(2*x1-1)))."""
    var_names = ["x1"]
    def f(X):
        return 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0] - 1.0)))
    def sample(n, rng):
        X = rng.uniform(-3, 3, (n, 1))
        return X, f(X)
    return Benchmark("logistic", var_names, "1/(1+exp(-(2*x1-1)))", f, sample, {"x1": (-3, 3)})


def bm_oscillator1():
    """线性阻尼振子: d²x/dt² + 2ζω0 dx/dt + ω0² x = 0.
    数据用 (x, v) -> a. 我们不假设物理模型, 仅生成数据."""
    var_names = ["x", "v"]
    omega0 = 2.0; zeta = 0.1
    def f(X):
        x, v = X[:, 0], X[:, 1]
        return -2 * zeta * omega0 * v - omega0 ** 2 * x
    def sample(n, rng):
        X = rng.uniform(-1, 1, (n, 2))
        return X, f(X)
    return Benchmark("OSC1_like", var_names, "-2*ζ*ω0*v - ω0^2*x", f, sample,
                     {"x": (-1, 1), "v": (-1, 1)}, notes="模拟线性阻尼振子 a(x,v)")


def bm_multicase_esat():
    """多案例版 esat: 同一形式 σ = C1*exp(1/(C2*ε + C3)), 不同案例只改 C1/C2."""
    var_names = ["x1"]
    true_params = [(1.2, 0.5, 0.3), (1.5, 0.4, 0.35), (1.0, 0.6, 0.25)]
    def make_f(p):
        C1, C2, C3 = p
        def f(X):
            return C1 * np.exp(1.0 / (C2 * X[:, 0] + C3))
        return f
    fns = [make_f(p) for p in true_params]
    def sample_for(idx, n, rng):
        X = rng.uniform(0.05, 1.5, (n, 1))
        return X, fns[idx](X)
    return Benchmark("esat_multicase", var_names,
                     "C1*exp(1/(C2*x1 + C3)), 案例参数独立",
                     fns[0],
                     lambda n, rng: sample_for(0, n, rng),
                     {"x1": (0.05, 1.5)},
                     n_cases=3,
                     extra_cases=[lambda n, rng: sample_for(1, n, rng),
                                   lambda n, rng: sample_for(2, n, rng)],
                     notes="测试多案例 case_param 发现 (K1/K2/K3)")


REGISTRY: Dict[str, Callable[[], Benchmark]] = {
    "poly_add": bm_poly_add,
    "mul_sep": bm_mul_separable,
    "add_sep": bm_add_separable,
    "rational": bm_rational,
    "esat": bm_esat,
    "voce": bm_voce,
    "gauss": bm_gauss,
    "hollomon": bm_hollomon,
    "logistic": bm_logistic,
    "osc1_like": bm_oscillator1,
    "esat_mc": bm_multicase_esat,
}


def list_benchmarks() -> List[str]:
    return list(REGISTRY.keys())


def make_cases(bm: Benchmark, n_per_case: int = 400, noise_rel: float = 0.01,
                noise_abs: float = 0.01, seed: int = 0) -> List[dict]:
    """生成多案例数据 (cases 列表, 每条 {'X','y','name'})."""
    rng = np.random.default_rng(seed)
    cases = []
    if bm.n_cases == 1:
        X, y = bm.sample(n_per_case, rng)
        y = _add_noise(y, noise_rel, noise_abs, rng)
        cases.append(dict(X=X, y=y, name="case0"))
    else:
        # case 0 用 bm.sample
        X, y = bm.sample(n_per_case, rng)
        y = _add_noise(y, noise_rel, noise_abs, rng)
        cases.append(dict(X=X, y=y, name="case0"))
        for ci, ext in enumerate(bm.extra_cases or []):
            X, y = ext(n_per_case, rng)
            y = _add_noise(y, noise_rel, noise_abs, rng)
            cases.append(dict(X=X, y=y, name=f"case{ci+1}"))
    return cases


# 默认 base_dir 可通过环境变量 LGF_REAL_DATA_DIR 覆盖, 不存在即返回 None,
# 不会因 D:\AI\LLM-Feynman 路径错误而导致 LGF 无法运行.
_DEFAULT_REAL_DATA_DIR = os.environ.get(
    "LGF_REAL_DATA_DIR",
    r"D:\AI\LLM-Feynman\LLM-Feynman-main\data",
)


def load_real_csv(name: str, base_dir: str = _DEFAULT_REAL_DATA_DIR,
                  max_rows: int = 1000) -> Optional[List[dict]]:
    """加载真实 CSV 数据集 (oscillator1/oscillator2/bactgrow 等).

    该函数是**可选**能力: base_dir 不存在或缺失对应文件时返回 None,
    LGF 主流程不依赖它.
    """
    fn = os.path.join(base_dir, name, "train.csv")
    if not os.path.exists(fn):
        # 尝试 test_id
        fn = os.path.join(base_dir, name, "test_id.csv")
        if not os.path.exists(fn):
            return None
    df = pd.read_csv(fn)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0).reset_index(drop=True)
    arr = df.to_numpy()
    X = arr[:, :-1]; y = arr[:, -1]
    var_names = list(df.columns[:-1])
    return [dict(X=X, y=y, name="case0")] + [], var_names


def load_real_csv_simple(name: str, base_dir: str = _DEFAULT_REAL_DATA_DIR,
                          max_rows: int = 1000) -> Optional[List[dict]]:
    """简化版: 返回 (cases, var_names).

    失败时返回 (None, None); 调用方需判空, 不影响 LGF 主流程.
    """
    fn_train = os.path.join(base_dir, name, "train.csv")
    fn_test = os.path.join(base_dir, name, "test_id.csv")
    fp = fn_train if os.path.exists(fn_train) else (fn_test if os.path.exists(fn_test) else None)
    if fp is None:
        return None, None
    df = pd.read_csv(fp)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0).reset_index(drop=True)
    arr = df.to_numpy()
    X = arr[:, :-1]; y = arr[:, -1]
    var_names = list(df.columns[:-1])
    return [dict(X=X.astype(float), y=y.astype(float), name="case0")], var_names