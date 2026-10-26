"""lgf.priors — AI-Feynman 风格物理先验检测 + 数据画像.

提供:
- data_profile: 输出数据摘要 (单调性, 范围, 渐近, 正性, ...) 供 LLM/启发式使用.
- SurrogateField: 用 torch MLP 拟合数据生成代理场, 用于结构检验.
- test_symmetry / test_additive_separable / test_multiplicative_separable /
  test_compositionality / dimensional_analysis.
- tests_to_constraints: 将检测结果编译为 GP 使用的过滤约束.
"""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .graph import Graph


# ============================================================================
# 数据画像 (供 LLM 与启发式使用)
# ============================================================================
def data_profile(cases: List[dict], var_names: Sequence[str]) -> Dict:
    n_vars = len(var_names)
    out: Dict = {
        "n_vars": n_vars,
        "var_names": list(var_names),
        "n_cases": len(cases),
        "per_case": [],
        "global": {},
    }
    all_y = []
    for ci, c in enumerate(cases):
        X = c["X"]; y = c["y"]
        info = {
            "name": c.get("name", f"case{ci}"),
            "N": int(X.shape[0]),
            "y_min": float(np.min(y)), "y_max": float(np.max(y)),
            "y_mean": float(np.mean(y)), "y_std": float(np.std(y)),
            "y_positive": bool(np.all(y > 0)) if not np.any(np.isnan(y)) else False,
            "monotone": {},
            "var_ranges": {},
        }
        for i, vn in enumerate(var_names):
            x = X[:, i]
            info["var_ranges"][vn] = [float(np.min(x)), float(np.max(x))]
            if len(x) > 2 and np.std(x) > 1e-12:
                # 单调性: Spearman 相关
                from scipy.stats import spearmanr
                rho, _ = spearmanr(x, y)
                info["monotone"][vn] = float(rho)
        all_y.append(y)
        out["per_case"].append(info)
    if all_y:
        yc = np.concatenate(all_y)
        out["global"]["y_min"] = float(np.min(yc))
        out["global"]["y_max"] = float(np.max(yc))
        out["global"]["y_positive"] = bool(np.all(yc > 0))
        out["global"]["y_range"] = float(np.ptp(yc))
    return out


def profile_summary(profile: Dict) -> str:
    """供 LLM prompt 使用的人类可读摘要."""
    lines = []
    lines.append(f"任务: 从 {profile['n_cases']} 组案例数据中发现方程.")
    lines.append(f"变量: {profile['var_names']}")
    for pc in profile["per_case"]:
        m = pc["monotone"]
        mons = ", ".join(f"{k}={v:+.2f}" for k, v in m.items())
        lines.append(f"  - {pc['name']}: N={pc['N']}, y∈[{pc['y_min']:.3g},{pc['y_max']:.3g}], "
                     f"y_mean={pc['y_mean']:.3g}, std={pc['y_std']:.3g}, "
                     f"{'y>0' if pc['y_positive'] else 'y可负'}, "
                     f"单调性: {mons}")
    g = profile["global"]
    lines.append(f"全局: y∈[{g['y_min']:.3g},{g['y_max']:.3g}], {'y>0' if g['y_positive'] else 'y可负'}")
    return "\n".join(lines)


# ============================================================================
# Surrogate: 训练一个 torch MLP 用作 a.a. AI-Feynman 的神经网络代理.
# ============================================================================
class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers = []
        d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class SurrogateField:
    """拟合后的代理场. 提供 evaluate / mixed_partial / invariance."""
    mlp: _MLP
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = (X - self.x_mean) / self.x_std
        with torch.no_grad():
            yp = self.mlp(torch.from_numpy(x).float()).numpy()
        return yp * self.y_std + self.y_mean

    def grad_x(self, X: np.ndarray) -> np.ndarray:
        """返回 (N, n_vars)."""
        x = (X - self.x_mean) / self.x_std
        xt = torch.from_numpy(x).float().requires_grad_(True)
        yp = self.mlp(xt)
        (g,) = torch.autograd.grad(yp.sum(), xt, create_graph=False)
        return g.numpy() * (self.y_std / self.x_std)

    def mixed_partial(self, X: np.ndarray, i: int, j: int) -> np.ndarray:
        """∂²f/∂xi∂xj 数值估计 (中心差分)."""
        h = 0.01 * max(1.0, float(np.std(X[:, i])) if X.shape[0] > 1 else 1.0)
        Xp = X.copy(); Xp[:, i] += h; Xp[:, j] += h
        Xn = X.copy(); Xn[:, i] -= h; Xn[:, j] -= h
        Xpi = X.copy(); Xpi[:, i] += h
        Xni = X.copy(); Xni[:, i] -= h
        Xpj = X.copy(); Xpj[:, j] += h
        Xnj = X.copy(); Xnj[:, j] -= h
        fpp = self.predict(Xp); fnn = self.predict(Xn)
        fpi = self.predict(Xpi); fni = self.predict(Xni)
        fpj = self.predict(Xpj); fnj = self.predict(Xnj)
        return (fpp - fpi - fpj + fni + fni - Xni.shape[0]*0) * 0  # placeholder, replaced below

    def mixed_partial_ij(self, X: np.ndarray, i: int, j: int) -> np.ndarray:
        """∂²f/∂xi∂xj 数值估计 (中心差分, 实际公式)."""
        h = 0.01 * max(1.0, float(np.std(X[:, i])) if X.shape[0] > 1 else 1.0)
        Xpp = X.copy(); Xpp[:, i] += h; Xpp[:, j] += h
        Xpn = X.copy(); Xpn[:, i] += h; Xpn[:, j] -= h
        Xnp = X.copy(); Xnp[:, i] -= h; Xnp[:, j] += h
        Xnn = X.copy(); Xnn[:, i] -= h; Xnn[:, j] -= h
        fpp = self.predict(Xpp); fpn = self.predict(Xpn)
        fnp = self.predict(Xnp); fnn = self.predict(Xnn)
        return (fpp - fpn - fnp + fnn) / (4.0 * h * h)


def fit_surrogate(X: np.ndarray, y: np.ndarray,
                  hidden: int = 64, depth: int = 2, epochs: int = 500,
                  lr: float = 1e-3, device: str = "cpu", seed: int = 0) -> SurrogateField:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_in = X.shape[1]
    mlp = _MLP(n_in, hidden=hidden, depth=depth).to(device)
    x_mean = X.mean(axis=0); x_std = X.std(axis=0) + 1e-9
    y_mean = float(y.mean()); y_std = float(y.std()) + 1e-9
    xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std
    xt = torch.from_numpy(xn).float()
    yt = torch.from_numpy(yn).float()
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    bs = min(512, X.shape[0])
    for ep in range(epochs):
        idx = torch.randperm(xt.shape[0])
        for s in range(0, xt.shape[0], bs):
            ii = idx[s:s+bs]
            yp = mlp(xt[ii])
            loss = ((yp - yt[ii]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return SurrogateField(mlp, x_mean, x_std, y_mean, y_std)


# ============================================================================
# 结构性检验
# ============================================================================
def test_symmetry(field: SurrogateField, X: np.ndarray, i: int, j: int,
                  n_samples: int = 200, tol: float = 0.05) -> Tuple[bool, float]:
    """f(x_i, x_j) ≈ f(x_j, x_i)? 在代理场上采样检验."""
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)]
    Xs_swap = Xs.copy()
    Xs_swap[:, [i, j]] = Xs_swap[:, [j, i]]
    f1 = field.predict(Xs); f2 = field.predict(Xs_swap)
    scale = np.std(np.concatenate([f1, f2])) + 1e-9
    rel = np.max(np.abs(f1 - f2)) / scale
    return (rel < tol, float(rel))


def test_additive_separable(field: SurrogateField, X: np.ndarray, i: int, j: int,
                            n_samples: int = 300, tol: float = 0.1) -> Tuple[bool, float]:
    """检验 ∂²f/∂xi∂xj 是否近 0 (加性可分必要条件).

    归一化: rel = ||∂²ij|| / (||∂²ij|| + ||∂²ii|| + ||∂²jj||), 有界于 [0,1].
    - 加性可分: mp→0 ⇒ rel→0;
    - 纯乘性 (mp 主导, 纯二阶导为 0): rel→1;
    - 一般非线性: 居中.
    """
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)]
    mp = field.mixed_partial_ij(Xs, i, j)
    h_ii = field.mixed_partial_ij(Xs, i, i)
    h_jj = field.mixed_partial_ij(Xs, j, j)
    mp_abs = float(np.mean(np.abs(mp)))
    scale = float(np.mean(np.abs(h_ii)) + np.mean(np.abs(h_jj)))
    rel = mp_abs / (mp_abs + scale + 1e-12)
    return (rel < tol, rel)


def test_multiplicative_separable(field: SurrogateField, X: np.ndarray, i: int, j: int,
                                  n_samples: int = 300, tol: float = 0.1) -> Tuple[bool, float]:
    """在 log|f| 上做加性可分检验, 即 f = g(x_i)*h(x_j) 的必要条件."""
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)]
    f = field.predict(Xs)
    if np.any(f <= 0):
        # 移位使正
        f_pos = f - np.min(f) + 1e-3
    else:
        f_pos = f
    logf = np.log(f_pos)
    # 手工 log-代理：直接对 logf 重新拟合简单代理
    try:
        sub = fit_surrogate(Xs, logf, epochs=300, seed=1)
    except Exception:
        return (False, 1.0)
    return test_additive_separable(sub, Xs, i, j, n_samples=n_samples, tol=tol)


def test_compositionality(field: SurrogateField, X: np.ndarray, i: int, j: int,
                          transforms: Optional[List[str]] = None,
                          tol: float = 0.15) -> Tuple[bool, Dict]:
    """试探可分性的变换变体: 在 (t_i(x_i), x_j) 上做加性可分性, 命中暗示乘性/广义组合性.
    返回 (any_found, info_dict)."""
    if transforms is None:
        transforms = ["id", "log", "sqrt", "inv"]
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], size=min(300, X.shape[0]), replace=False)]
    best = (False, "identity", tol)
    info = {}
    for t in transforms:
        try:
            xt = Xs.copy()
            if t == "id":
                pass
            elif t == "log":
                xt[:, i] = np.log(np.abs(Xs[:, i]) + 1e-3)
            elif t == "sqrt":
                xt[:, i] = np.sqrt(np.abs(Xs[:, i]))
            elif t == "inv":
                xt[:, i] = 1.0 / (np.abs(Xs[:, i]) + 1e-3)
            else:
                continue
            # 重新拟合代理以适配 xt
            sub = fit_surrogate(xt, field.predict(Xs), epochs=200, seed=1)
            ok, rel = test_additive_separable(sub, xt, i, j, tol=tol)
            info[t] = rel
            if rel < best[2]:
                best = (ok, t, rel)
        except Exception:
            info[t] = None
    return (best[0], {"transform": best[1], "rel": best[2], "per_transform": info})


# ===== 量纲分析 (Buckingham-Π 简化版) =====
def dimensional_analysis(var_units: Dict[str, Dict[str, float]],
                          target_unit: Dict[str, float]) -> Dict:
    """var_units: {var_name: {dim: exponent}}; target_unit: {dim: exponent}.
    返回: Π-组的指数矩阵 (行=Π-组, 列=变量). 求解 linear system:
          target = sum_k coeff_k * var_k   in log-space (sum of exponents).
    若解存在, 用最小二乘回代. 若严格整数解不可得, 给出软建议.
    """
    if not var_units or not target_unit:
        return {"ok": True}
    dims = sorted(target_unit.keys())
    var_names = list(var_units.keys())
    M = np.array([[var_units[v].get(d, 0.0) for d in dims] for v in var_names])
    rhs = np.array([target_unit.get(d, 0.0) for d in dims])
    # 解: 若维度被变量张成, 求最小二乘
    sol, res, rank, sv = np.linalg.lstsq(M.T, rhs, rcond=None)
    # 重构残差
    pred = M.T @ sol
    err = float(np.linalg.norm(pred - rhs))
    return {"variables": var_names, "dimensions": dims, "coeffs": sol.tolist(),
            "reconstruction_error": err, "ok": err < 1e-6}


# ===== 将先验检测结果编译为 GP 使用的约束 =====
@dataclass
class PriorConstraints:
    symmetric_pairs: List[Tuple[int, int]] = field(default_factory=list)
    additive_partitions: List[List[int]] = field(default_factory=list)
    multiplicative_partitions: List[List[int]] = field(default_factory=list)
    transforms: Dict[Tuple[int, int], str] = field(default_factory=dict)
    pi_groups: List[Dict] = field(default_factory=list)
    notes: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        def _str_keys(d: Dict) -> Dict:
            return {str(k): v for k, v in d.items()}
        return dict(symmetric_pairs=[list(p) for p in self.symmetric_pairs],
                    additive_partitions=[list(p) for p in self.additive_partitions],
                    multiplicative_partitions=[list(p) for p in self.multiplicative_partitions],
                    transforms={f"{i},{j}": t for (i, j), t in self.transforms.items()},
                    pi_groups=self.pi_groups,
                    notes={str(k): (_str_keys(v) if isinstance(v, dict) else v)
                           for k, v in self.notes.items()})


def tests_to_constraints(report: Dict, n_vars: int, sym_tol: float = 0.05,
                          sep_tol: float = 0.1) -> PriorConstraints:
    pc = PriorConstraints()
    sym = report.get("symmetry", {})
    for (i, j), (sym_ok, _) in sym.items():
        if sym_ok:
            pc.symmetric_pairs.append((i, j))
    sep = report.get("separability", {})
    for (i, j), info in sep.items():
        if info.get("additive_ok"):
            pc.additive_partitions.append([i, j])
        if info.get("multiplicative_ok"):
            pc.multiplicative_partitions.append([i, j])
        if info.get("compositional_transform"):
            pc.transforms[(i, j)] = info["compositional_transform"]
    pc.notes = {k: v for k, v in report.items() if k not in ("symmetry", "separability")}
    return pc


# ===== 高层入口: run_all_priors =====
def run_priors(cases: List[dict], var_names: Sequence[str],
               sym_tol: float = 0.05, sep_tol: float = 0.1,
               use_surrogate: bool = True,
               var_units: Optional[Dict] = None,
               target_unit: Optional[Dict] = None) -> Tuple[Dict, PriorConstraints]:
    """运行全部可执行的结构检验, 返回 (report, constraints)."""
    rep: Dict = {"symmetry": {}, "separability": {}, "compositionality": {}, "da": None, "profile": None}
    rep["profile"] = data_profile(cases, var_names)
    if use_surrogate:
        # 用所有案例数据训练一个全局代理
        Xs = np.concatenate([c["X"] for c in cases], axis=0)
        ys = np.concatenate([c["y"] for c in cases], axis=0)
        field = fit_surrogate(Xs, ys)
        n = len(var_names)
        for i, j in itertools.combinations(range(n), 2):
            sym_ok, sym_rel = test_symmetry(field, Xs, i, j, tol=sym_tol)
            rep["symmetry"][(i, j)] = (sym_ok, sym_rel)
            add_ok, add_rel = test_additive_separable(field, Xs, i, j, tol=sep_tol)
            mul_ok, mul_rel = test_multiplicative_separable(field, Xs, i, j, tol=sep_tol)
            comp_ok, comp_info = test_compositionality(field, Xs, i, j, tol=sep_tol)
            rep["separability"][(i, j)] = dict(additive_ok=add_ok, additive_rel=add_rel,
                                               multiplicative_ok=mul_ok, multiplicative_rel=mul_rel,
                                               compositional_ok=bool(comp_ok),
                                               compositional_transform=comp_info.get("transform"))
            rep["compositionality"][(i, j)] = comp_info
    if var_units is not None:
        rep["da"] = dimensional_analysis(var_units, target_unit or {})
    return rep, tests_to_constraints(rep, len(var_names))