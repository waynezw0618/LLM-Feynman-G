# -*- coding: utf-8 -*-
"""rc_asm_true_pysr.py — 真 ASM 方法1: PySR 纯符号回归基线 (无 LLM, 无物理提示).

方法学定位 (用户 2026-10-09 指令):
  PySR = 纯 SR 对照臂 — 只给数据 + 算子集, 不注入任何物理上下文,
  不借用 LGF 的任何策略 (G_INIT/M_GUIDE/先验/种子), 与 LGF+M2.7 严格隔离.

数据: outputs/asm/dataset_trueasm_{variant}.npz
  行 = (工况, 壁面位置, 张量分量) 三元组; 17 特征
  [T1..T10, eta2, zeta2, s3, so2, s2o2, ret, roloc]; 目标 y = b_ij.
  常数全部为全局 (PySR 常数在跨工况堆叠数据上 BFGS 拟合, 天然全局).

加权: 逐行 w = 1/(N_c * var_c) → PySR 内部加权 MSE ∝ Σ_c MSE_c/var_c
  = 5 × loss_asm (与 LGF paramfit 的 mean_c(MSE_c/var_c) 同向同最优).

评估: rc_asm_true_eval (loss_asm + rho_a,case W&S eq.22 + 作弊检测);
  对整个 Pareto 前沿逐条评估, 按 loss_asm 选优 (内部 loss 只作参考).

用法:
  python -X utf8 scripts/rc_asm_true_pysr.py --variant v \
      [--niterations 60] [--maxsize 30] [--seed 0] [--populations 8]
输出: outputs/asm/runs_trueasm_pysr/{variant}/ (result.json + preds_best.npz)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_true_eval import (load_variant, eval_predictions, summarize,  # noqa: E402
                              RO_CASES)
from rc_asm_true_prep import FEAT_NAMES  # noqa: E402

OUT_DIR = os.path.join(REPO, "outputs", "asm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v", "g", "e"], required=True)
    ap.add_argument("--niterations", type=int, default=60)
    ap.add_argument("--maxsize", type=int, default=30)
    ap.add_argument("--populations", type=int, default=8)
    ap.add_argument("--population_size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unary", nargs="*", default=["exp", "log", "sqrt"])
    args = ap.parse_args()

    from pysr import PySRRegressor

    data = load_variant(args.variant)
    X = np.vstack([c["Xfit"] for c in data])
    y = np.concatenate([c["yfit"] for c in data])
    w = np.concatenate([
        np.full(6 * c["Nv"], 1.0 / (6.0 * c["Nv"] * c["var_c"])) for c in data])
    print(f"[pysr:{args.variant}] X={X.shape} y={y.shape} "
          f"rows/case={[6 * c['Nv'] for c in data]}")

    nested = {u: {v: 0 for v in args.unary} for u in args.unary}
    model = PySRRegressor(
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        maxsize=args.maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=list(args.unary),
        nested_constraints=nested,          # 禁 exp(log(x)) 类一元嵌套
        constraints={u: 15 for u in args.unary},   # 一元算子参数复杂度上限
        model_selection="best",
        random_state=args.seed,
        temp_equation_file=True,
        verbosity=1, progress=False,
    )

    run_dir = os.path.join(OUT_DIR, "runs_trueasm_pysr", args.variant)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    model.fit(X, y, weights=w, variable_names=FEAT_NAMES)
    elapsed = time.time() - t0
    print(f"[pysr:{args.variant}] fit done in {elapsed:.0f}s, "
          f"Pareto front size={len(model.equations_)}")

    # ---- 整条 Pareto 前沿逐条评估 ----
    pareto = []
    eqs = model.equations_                       # pandas DataFrame (pysr 2.x)
    n_eq = len(eqs)
    for idx in range(n_eq):
        row = eqs.iloc[idx]
        try:
            preds = [np.asarray(model.predict(c["Xfit"], index=idx),
                                dtype=float) for c in data]
            m = eval_predictions(preds, data)
        except Exception as e:                       # Julia eval 失败/非有限
            m = dict(loss_asm=float("inf"), rho_a=-1.0, per_case={},
                     error=f"{type(e).__name__}: {e}")
            preds = None
        rec = dict(index=idx, complexity=int(row["complexity"]),
                   internal_loss=float(row["loss"]),
                   sympy=str(row.get("sympy_format", row["equation"])),
                   equation=str(row["equation"]),
                   metrics=m)
        pareto.append(rec)
        print(f"[pysr:{args.variant}] eq{idx} cplx={int(row['complexity']):<3} "
              f"internal={float(row['loss']):.5f} :: "
              f"{summarize(m) if m.get('per_case') else m.get('error')}")

    finite = [r for r in pareto if np.isfinite(r["metrics"].get("loss_asm", np.inf))]
    best = min(finite, key=lambda r: r["metrics"]["loss_asm"]) if finite else None

    result = dict(
        variant=args.variant, method="pysr",
        config=dict(niterations=args.niterations, maxsize=args.maxsize,
                    populations=args.populations,
                    population_size=args.population_size,
                    unary=args.unary, seed=args.seed,
                    weights="1/(6*Nv_c*var_c)"),
        n_rows=int(X.shape[0]), feat_names=list(FEAT_NAMES),
        elapsed_s=round(elapsed, 1), pareto=pareto,
    )
    if best is not None:
        result["best"] = dict(
            index=best["index"], complexity=best["complexity"],
            internal_loss=best["internal_loss"],
            sympy=best["sympy"], equation=best["equation"],
            metrics=best["metrics"])
        # 存最优公式的逐工况预测 (免除后续 Julia 重算)
        preds = [np.asarray(model.predict(c["Xfit"], index=best["index"]),
                            dtype=float) for c in data]
        preds_file = os.path.join(run_dir, "preds_best.npz")
        np.savez_compressed(preds_file,
                            **{f"c{i}": p for i, p in enumerate(preds)})
        result["preds_file"] = preds_file
        print(f"[pysr:{args.variant}] BEST eq{best['index']} "
              f"cplx={best['complexity']}\n  {best['sympy']}\n  "
              f"{summarize(best['metrics'])}")

    out = os.path.join(run_dir, "result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[pysr:{args.variant}] -> {out}")


if __name__ == "__main__":
    main()
