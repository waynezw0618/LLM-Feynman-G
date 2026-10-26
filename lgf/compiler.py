"""lgf.compiler — 将 Graph 编译为可数值求值 (numpy/torch) 的可调用对象.

设计:
- 基于 sympy: to_sympy() -> Expression; lambdify 出 forward 与各参量的 grad.
- 同一函数对所有 (globals + 单 case 的 case_params) 都适用, 每案例独立调用一次.
- 数值稳定性: 在求值中遇到 inf/nan 由 paramfit 损失加罚屏蔽.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .graph import Graph


class CompiledGraph:
    def __init__(self, graph: Graph, backend: str = "numpy"):
        assert backend in ("numpy", "torch")
        self.g = graph
        self.backend = backend
        self.expr = graph.to_sympy()
        self.var_syms = [sp.Symbol(n, real=True) for n in graph.var_names]
        self.global_syms = [sp.Symbol(n, real=True) for n in graph.global_names()]
        self.case_syms = [sp.Symbol(n, real=True) for n in graph.case_names()]
        n_g = len(self.global_syms); n_c = len(self.case_syms)
        # 符号顺序: vars, globals, case_params
        sym_order = [*self.var_syms, *self.global_syms, *self.case_syms]
        # forward
        if backend == "numpy":
            self._fwd = sp.lambdify(sym_order, self.expr, "numpy")
            # grads
            self._grads: List = []
            for s in (self.global_syms + self.case_syms):
                d = sp.diff(self.expr, s)
                self._grads.append(sp.lambdify(sym_order, d, "numpy"))
        else:
            import torch  # noqa
            self._fwd = sp.lambdify(sym_order, self.expr, "torch")
            self._grads = []
            for s in (self.global_syms + self.case_syms):
                d = sp.diff(self.expr, s)
                self._grads.append(sp.lambdify(sym_order, d, "torch"))
        # 参量分块大小
        self.n_g = n_g; self.n_c = n_c
        # 模块校验: 用 dummy 输入快速执行一次, 确认编译可用
        try:
            self._fwd(*([np.zeros(1)] * len(sym_order)))
        except Exception:
            # 不致命, 可能图使用了不常见算子
            pass

    # ----- 单案例评估 -----
    def eval_case(self, theta_local: Sequence[float], X: np.ndarray) -> np.ndarray:
        """X: (N, n_vars), theta_local: [globals..., case_params...]. 返回 (N,)."""
        cols = [X[:, i] for i in range(X.shape[1])]
        args = cols + list(theta_local)
        return np.asarray(self._fwd(*args), dtype=float)

    def grad_case(self, theta_local: Sequence[float], X: np.ndarray) -> np.ndarray:
        """返回 (P_local, N) 梯度矩阵, P_local = n_g + n_c."""
        cols = [X[:, i] for i in range(X.shape[1])]
        args = cols + list(theta_local)
        out = np.zeros((self.n_g + self.n_c, X.shape[0]))
        for j, fn in enumerate(self._grads):
            out[j] = np.asarray(fn(*args), dtype=float)
        return out

    def eval_cases(self, theta: Sequence[float], cases) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """对多个案例联合评估: theta 长度 = n_g + n_c * n_cases.
        返回 (loss, grad_full, per_case_pred, n_finite).
        """
        n_cases = len(cases)
        P_total = self.n_g + self.n_c * n_cases
        loss = 0.0
        grad = np.zeros(P_total)
        per_pred = []
        n_finite = 0
        for ci, case in enumerate(cases):
            X = case["X"]; y = case["y"]
            w = case.get("w")
            tl = np.concatenate([theta[:self.n_g],
                                  theta[self.n_g + ci*self.n_c : self.n_g + (ci+1)*self.n_c]])
            try:
                yp = self.eval_case(tl, X)
                gp = self.grad_case(tl, X)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                return (1e8, np.zeros(P_total), per_pred, 0)
            if not (np.all(np.isfinite(yp)) and np.all(np.isfinite(gp))):
                return (1e8, np.zeros(P_total), per_pred, 0)
            n_finite += X.shape[0]
            per_pred.append(yp)
            res = yp - y
            if w is None:
                # 常规: 逐案例 NMSE (系数空间)
                var_y = float(np.var(y)) + 1e-12
                lc = float(np.mean(res**2)) / var_y
                for j in range(self.n_g + self.n_c):
                    gsum = float(np.mean(2.0 * res * gp[j])) / var_y / n_cases
                    if j < self.n_g:
                        grad[j] += gsum
                    else:
                        grad[self.n_g + ci*self.n_c + (j - self.n_g)] += gsum
            else:
                # 加权 NMSE (应力空间): w_i = ||T_m,i||_F^2 时, 加权残差平方和
                # 恰等于该基张量对 b_ij 重建的应力空间贡献误差 (逐点精确).
                w = np.asarray(w, dtype=float)
                sw = float(w.sum())
                mu = float(np.dot(w, y) / sw)
                var_w = float(np.dot(w, (y - mu) ** 2) / sw) + 1e-12
                lc = float(np.dot(w, res**2) / sw) / var_w
                for j in range(self.n_g + self.n_c):
                    gsum = float(np.dot(w, 2.0 * res * gp[j]) / sw) / var_w / n_cases
                    if j < self.n_g:
                        grad[j] += gsum
                    else:
                        grad[self.n_g + ci*self.n_c + (j - self.n_g)] += gsum
            loss += lc / n_cases
        return (loss, grad, per_pred, n_finite)


def safe_eval(expr_callable, *args, fallback=1e8):
    try:
        y = expr_callable(*args)
        if not np.all(np.isfinite(y)):
            return fallback
        return y
    except Exception:
        return fallback