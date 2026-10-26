"""一次性回填脚本: 给 outputs/benchmark/results.jsonl 里每条记录补 best_theta
   再重跑 make_plots() 重新出图.

仅用于 --quick 模式历史结果的视觉化修复. 正常流程由 scripts/run_benchmarks.py
自动把 best_theta 写入 rec.

用法: python scripts/_backfill_theta_and_replot.py
"""
import sys, os, json, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np

from lgf.benchmarks import REGISTRY, make_cases
from lgf.dsl import parse_dsl
from lgf.paramfit import fit_joint


def backfill_one(rec):
    bm_key = rec["benchmark"]
    if "best_theta" in rec and rec["best_theta"] is not None:
        return rec  # already filled
    bm = REGISTRY[bm_key]()
    train = make_cases(bm, n_per_case=80, noise_rel=0.02, seed=0)  # 与 quick 模式一致
    g = parse_dsl(rec["best_dsl"], list(bm.var_names))
    try:
        fr = fit_joint(g, train, n_starts=4, maxiter=80, time_budget_s=20.0, seed=0, progress=False)
        if np.all(np.isfinite(fr.theta)):
            rec["best_theta"] = [float(x) for x in fr.theta]
        else:
            rec["best_theta"] = [float("nan")] * max(1, g.n_params())
    except Exception:
        rec["best_theta"] = [float("nan")] * max(1, g.n_params())
    return rec


def main():
    jsonl_path = os.path.join(ROOT, "outputs", "benchmark", "results.jsonl")
    if not os.path.exists(jsonl_path):
        print("no results.jsonl found"); return
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for ln in f:
            try:
                records.append(json.loads(ln))
            except Exception:
                pass
    print(f"backfilling theta for {len(records)} records ...")
    out = []
    for r in records:
        try:
            r = backfill_one(r)
            out.append(r)
        except Exception:
            traceback.print_exc()
            out.append(r)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print("theta backfilled, re-running make_plots ...")

    # 借调 scripts/run_benchmarks.py 的 make_plots
    from scripts.run_benchmarks import make_plots
    make_plots(out)
    print("plots regenerated under outputs/benchmark/plots/")


if __name__ == "__main__":
    main()
