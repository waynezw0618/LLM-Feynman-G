"""lgf.paramfit — 多启动 L-BFGS 参数拟合 (对齐 GraphED §2.3).

要点:
- 联合优化: full_theta = [globals, case_params_1, case_params_2, ...]
- 多案例归一化: 损失 = mean_c (MSE_c / var(y_c)), 避免尺度不一致淹没.
- 多启动: 默认 20 restarts (GP 阶段降为 2), 每个 start 用 L-BFGS-B.
- 数值失败 (NaN/Inf) 一律返回大罚值, 不让优化器跟踪.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .compiler import CompiledGraph
from .graph import Graph


@dataclass
class FitResult:
    success: bool
    loss: float
    theta: np.ndarray
    per_case: Dict[str, float]   # 'r2','nmse','rmse' per case name
    n_finite: int
    n_iters: int


def fit_joint(graph: Graph, cases: List[dict], *,
              n_starts: int = 20, maxiter: int = 200,
              init_range: float = 1.5, fit_tol: float = 1e-9,
              time_budget_s: float = 600.0, seed: int = 0,
              range_factor: float = 20.0,
              progress: bool = False) -> FitResult:
    """cases: list of {'X': np.ndarray (N,n_vars), 'y': np.ndarray (N,), 'name': str}.

    返回 FitResult. theta 长度 = n_g + n_c * len(cases), 顺序:
       [globals..., case_params_for_case_0..., case_params_for_case_1..., ...]

    V3-c 值域护栏 (2026-10-08): 拟合后若预测值超出 range_factor×max|y|
    (如 exp(C/yw) 的壁面奇点爆炸), loss 加大罚 (10 + 1e4×越界点占比),
    阻止"整体 NMSE 尚可但局部爆到 1e8"的结构混入精英层.
    range_factor<=0 关闭.

    V4 (2026-10-12) 两项修复:
    - 硬限时: 旧版只在 start 之间查表, 单次 L-BFGS 无上限 → 单 start 可跑
      数分钟, 整体 budget 形同虚设 (实测 lgf_mg/m elapsed 2834s ≫ budget 600s).
      现在 deadline 由 L-BFGS-B callback 强制, 到点中断当次 start, 已有最优保留.
    - 多 start 策略: [base 初值, 均匀扰动, 对数尺度扰动×(n-2)].
      对数尺度 (±10^U(-3,3) 随机符号) 治特征尺度悬殊时的 θ 尺度病
      (yw∈[0,180] 把 θ 压到 1e-7 量级的教训).
    """
    rng = np.random.default_rng(seed)
    cg = CompiledGraph(graph, backend="numpy")
    n_g = cg.n_g; n_c = cg.n_c
    n_cases = len(cases)
    P_total = n_g + n_c * n_cases
    if P_total == 0:
        # 无自由参数, 直接评估
        theta = np.zeros(0)
        loss, _, _, nfin = cg.eval_cases(theta, cases)
        return FitResult(True, loss, theta, _metrics(cg, theta, cases), nfin, 0)

    # base theta: from graph.param().init for globals; replicate for case params
    base_theta = np.zeros(P_total)
    graph_params = graph.params()
    for i, p in enumerate(graph_params):
        base_theta[i] = p.init
    # case params: replicate per case
    for ci in range(n_cases):
        for j in range(n_c):
            base_theta[n_g + ci*n_c + j] = graph_params[n_g + j].init

    best_loss = float("inf"); best_theta = base_theta.copy(); best_iters = 0
    t0 = time.time()
    deadline = t0 + max(0.5, float(time_budget_s))
    n_starts_eff = max(1, int(n_starts))
    n_starts_done = 0

    class _Deadline(Exception):
        pass

    def _cb(xk):
        if time.time() > deadline:
            raise _Deadline()

    def _try(x0):
        """单次 L-BFGS-B (deadline 中断); 返回 (fun, x, nit) 或 None."""
        try:
            res = minimize(_loss_grad, x0, args=(cg, cases),
                           jac=True, method="L-BFGS-B", callback=_cb,
                           options=dict(maxiter=maxiter, ftol=fit_tol))
            return (float(res.fun), np.asarray(res.x),
                    int(getattr(res, "nit", 0) or 0))
        except _Deadline:
            return None          # 到点中断, 已有 best 保留
        except Exception:
            return None

    for s in range(n_starts_eff):
        if time.time() - t0 > time_budget_s:
            break
        if s == 0:
            x0 = base_theta.copy()
        elif s == 1:
            x0 = base_theta + rng.uniform(-init_range, init_range, P_total)
        else:
            # 对数尺度 start: 随机符号 × 10^U(-3,3), 覆盖跨量级参数
            mag = 10.0 ** rng.uniform(-3.0, 3.0, P_total)
            sgn = np.where(rng.random(P_total) < 0.5, -1.0, 1.0)
            x0 = sgn * mag
            if s % 4 == 3:       # 每 4 个对数 start 里 1 个为 base 的缩放扰动
                x0 = base_theta * sgn * mag
        r = _try(x0)
        n_starts_done += 1
        if (r is not None and np.isfinite(r[0]) and r[0] < best_loss
                and np.all(np.isfinite(r[1]))):
            best_loss, best_theta, best_iters = r[0], r[1].copy(), r[2]

    # V4-b (2026-10-12) 抛光步: start 循环以 fit_tol (默认 1e-9) 提前停,
    # 远起点 (对数尺度) 收敛后系数误差可停在 ~1e-7 量级
    # (linear_recovery 测试实测 5.95e-8, 符号精确相等断言失败).
    # 预算允许时从 best_theta 再跑一次紧 ftol 的 L-BFGS.
    if np.isfinite(best_loss) and time.time() < deadline:
        try:
            res = minimize(_loss_grad, best_theta, args=(cg, cases),
                           jac=True, method="L-BFGS-B", callback=_cb,
                           options=dict(maxiter=max(200, maxiter),
                                        ftol=1e-14, gtol=1e-10))
            if (np.isfinite(res.fun) and float(res.fun) < best_loss
                    and np.all(np.isfinite(res.x))):
                best_loss = float(res.fun)
                best_theta = np.asarray(res.x).copy()
                best_iters = int(getattr(res, "nit", 0) or 0)
        except Exception:
            pass

    metrics = _metrics(cg, best_theta, cases)
    # V3-c: 值域护栏 — 越界占比 (取最差 case) 转为大罚值叠加到 loss
    if np.isfinite(best_loss) and range_factor and range_factor > 0:
        frac_bad = _max_frac_out_of_range(cg, best_theta, cases, range_factor)
        if frac_bad > 0.0:
            best_loss = float(best_loss + 10.0 + 1e4 * frac_bad)
    success = np.isfinite(best_loss) and best_loss < 1e7
    return FitResult(success=success, loss=float(best_loss),
                     theta=best_theta, per_case=metrics,
                     n_finite=_count_finite(cg, best_theta, cases),
                     n_iters=best_iters)


def _loss_grad(theta: np.ndarray, cg: CompiledGraph, cases) -> tuple:
    loss, grad, _, _ = cg.eval_cases(theta, cases)
    return loss, grad


def _metrics(cg: CompiledGraph, theta: np.ndarray, cases) -> Dict[str, float]:
    """返回按 case 名映射的度量字典."""
    out = {}
    n_cases = len(cases); n_g = cg.n_g; n_c = cg.n_c
    for ci, case in enumerate(cases):
        X = case["X"]; y = case["y"]
        name = case.get("name", f"case{ci}")
        tl = np.concatenate([theta[:n_g], theta[n_g + ci*n_c : n_g + (ci+1)*n_c]])
        try:
            yp = cg.eval_case(tl, X)
        except Exception:
            out[f"{name}_nmse"] = float("inf"); out[f"{name}_r2"] = -1e9
            out[f"{name}_rmse"] = float("inf"); continue
        if not np.all(np.isfinite(yp)):
            out[f"{name}_nmse"] = float("inf"); out[f"{name}_r2"] = -1e9
            out[f"{name}_rmse"] = float("inf"); continue
        w = case.get("w")
        if w is None:
            var_y = float(np.var(y)) + 1e-12
            mse = float(np.mean((yp - y) ** 2))
        else:
            w = np.asarray(w, dtype=float)
            sw = float(w.sum())
            mu = float(np.dot(w, y) / sw)
            var_y = float(np.dot(w, (y - mu) ** 2) / sw) + 1e-12
            mse = float(np.dot(w, (yp - y) ** 2) / sw)
        nmse = mse / var_y
        r2 = 1.0 - mse / var_y
        rmse = float(np.sqrt(mse))
        out[f"{name}_nmse"] = nmse
        out[f"{name}_r2"] = r2
        out[f"{name}_rmse"] = rmse
    return out


def _max_frac_out_of_range(cg: CompiledGraph, theta: np.ndarray, cases,
                           factor: float) -> float:
    """V3-c: 各 case 中 |pred| > factor×max|y| 的点占比 (返回最差 case).

    eval 抛异常或输出含非有限值 → 视为全越界 (1.0).
    """
    n_g = cg.n_g; n_c = cg.n_c
    worst = 0.0
    for ci, case in enumerate(cases):
        X = case["X"]; y = case["y"]
        tl = np.concatenate([theta[:n_g], theta[n_g + ci*n_c : n_g + (ci+1)*n_c]])
        try:
            yp = cg.eval_case(tl, X)
        except Exception:
            return 1.0
        if not np.all(np.isfinite(yp)):
            return 1.0
        thr = factor * (float(np.max(np.abs(y))) + 1e-12)
        worst = max(worst, float(np.mean(np.abs(yp) > thr)))
    return worst


def _count_finite(cg: CompiledGraph, theta: np.ndarray, cases) -> int:
    n_g = cg.n_g; n_c = cg.n_c; n = 0
    for ci, case in enumerate(cases):
        X = case["X"]; y = case["y"]
        tl = np.concatenate([theta[:n_g], theta[n_g + ci*n_c : n_g + (ci+1)*n_c]])
        try:
            yp = cg.eval_case(tl, X)
            n += int(np.sum(np.isfinite(yp)))
        except Exception:
            pass
    return n


def evaluate(graph: Graph, theta: np.ndarray, cases: List[dict]) -> Dict[str, float]:
    """公开: 给定拟合后的 theta, 返回度量字典."""
    cg = CompiledGraph(graph, backend="numpy")
    return _metrics(cg, theta, cases)