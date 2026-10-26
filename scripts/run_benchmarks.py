"""LGF 端到端基准验证.

- 9 个合成基准 (含多案例 esat_mc)
- 3 种配置消融: GP-only / +Priors / Full(+启发式LLM+归因)
- 训练/测试分离 (不同 seed), 指标: 测试集 R^2 / NMSE
- 输出: outputs/benchmark/results.jsonl + summary.json + plots/*.png

运行: python scripts/run_benchmarks.py [--quick]
"""
import sys, os, json, time, argparse, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from lgf.benchmarks import REGISTRY, make_cases
from lgf.config import LGFConfig
from lgf.pipeline_lgf import run_lgf
from lgf.llm_ops import make_client
from lgf.dsl import parse_dsl
from lgf.paramfit import evaluate

BENCHMARKS = ["esat", "voce", "hollomon", "logistic", "add_sep", "mul_sep",
              "rational", "gauss", "esat_mc"]

CONFIGS = {
    "GP-only": dict(use_llm=False, use_priors=False, use_attribution=False),
    "+Priors": dict(use_llm=False, use_priors=True, use_attribution=False),
    "Full":    dict(use_llm=True,  use_priors=True, use_attribution=True),
}

OUT_DIR = os.path.join(ROOT, "outputs", "benchmark")
PLOT_DIR = os.path.join(OUT_DIR, "plots")


def run_one(bm_key: str, cfg_name: str, quick: bool = False) -> dict:
    bm = REGISTRY[bm_key]()
    train = make_cases(bm, n_per_case=80 if quick else 200,
                       noise_rel=0.02, seed=0)
    test = make_cases(bm, n_per_case=80 if quick else 200,
                      noise_rel=0.02, seed=1000)
    base = dict(
        population=16 if quick else 24,
        generations=6 if quick else 15,
        gp_starts=1 if quick else 2, gp_maxiter=15 if quick else 25,
        final_starts=3 if quick else 8, final_maxiter=50 if quick else 150,
        time_budget_s=60 if quick else 150,
        sub_sample=80 if quick else 200,
        stagnation_window=4 if quick else 6, llm_every=2,
        seed=0, llm_backend="heuristic",
    )
    base.update(CONFIGS[cfg_name])
    cfg = LGFConfig(**base)
    client = make_client("heuristic")
    t0 = time.time()
    out = run_lgf(train, bm.var_names, cfg, OUT_DIR, client,
                  name=f"{bm_key}__{cfg_name.replace('+', '')}")
    elapsed = time.time() - t0

    best = out["best"]
    # 在独立测试集上评估 (重新解析 best DSL, theta 顺序与原一致)
    try:
        g = parse_dsl(best["dsl"], list(bm.var_names))
        m = evaluate(g, np.array(best["theta"]), test)
        r2s = [v for k, v in m.items() if k.endswith("_r2")]
        nmses = [v for k, v in m.items() if k.endswith("_nmse")]
        test_r2 = float(min(r2s)); test_nmse = float(max(nmses))
    except Exception as e:
        test_r2, test_nmse = float("nan"), float("nan")

    rec = dict(
        benchmark=bm_key, config=cfg_name,
        truth=bm.truth,
        best_dsl=best["dsl"], best_pretty=None,
        best_theta=[float(x) for x in best["theta"]],
        train_loss=best["loss"], test_r2=test_r2, test_nmse=test_nmse,
        complexity=best["complexity"], elapsed_s=round(elapsed, 1),
        found=bool(test_r2 > 0.99),
        log=[(e["gen"], e["best_loss"]) for e in out["log"]],
    )
    try:
        import sympy as sp
        rec["best_pretty"] = sp.sstr(sp.simplify(parse_dsl(
            best["dsl"], list(bm.var_names)).to_sympy()))
    except Exception:
        pass
    return rec


def make_plots(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---- 图1: 收敛曲线 (Full 配置) ----
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, bm_key in zip(axes.ravel(), BENCHMARKS):
        recs = [r for r in records if r["benchmark"] == bm_key]
        for r in recs:
            gen, loss = zip(*r["log"]) if r["log"] else ([], [])
            style = {"GP-only": ":", "+Priors": "--", "Full": "-"}[r["config"]]
            ax.semilogy(gen, loss, style, label=r["config"], linewidth=1.6)
        ax.set_title(bm_key, fontsize=10)
        ax.set_xlabel("generation", fontsize=8)
        ax.set_ylabel("best train loss", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("LGF convergence: ablation across benchmarks", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "convergence.png"), dpi=130)
    plt.close(fig)

    # ---- 图2: 测试 NMSE 分组柱状图 ----
    cfg_names = list(CONFIGS.keys())
    x = np.arange(len(BENCHMARKS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    for ci, cn in enumerate(cfg_names):
        vals = []
        for bk in BENCHMARKS:
            recs = [r for r in records if r["benchmark"] == bk and r["config"] == cn]
            vals.append(recs[0]["test_nmse"] if recs else np.nan)
        vals = np.array([v if np.isfinite(v) else 1.0 for v in vals], dtype=float)
        ax.bar(x + (ci - 1) * width, np.clip(vals, 1e-6, None), width, label=cn)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, rotation=20, fontsize=9)
    ax.axhline(0.01, color="crimson", ls="--", lw=1, label="success threshold (0.01)")
    ax.set_ylabel("test NMSE (max over cases, log scale)")
    ax.set_title("LGF ablation: test-set NMSE per benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ablation_nmse.png"), dpi=130)
    plt.close(fig)

    # ---- 图3: 最优模型 预测 vs 真值 (esat, Full) ----
    try:
        bm = REGISTRY["esat"]()
        train = make_cases(bm, n_per_case=200, noise_rel=0.02, seed=0)
        test = make_cases(bm, n_per_case=200, noise_rel=0.02, seed=1000)
        rec = [r for r in records if r["benchmark"] == "esat" and r["config"] == "Full"][0]
        g = parse_dsl(rec["best_dsl"], list(bm.var_names))
        from lgf.compiler import CompiledGraph
        cg = CompiledGraph(g)
        yp = cg.eval_case(np.array(rec["best_theta"])[:cg.n_g + cg.n_c], test[0]["X"])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(test[0]["X"][:, 0], test[0]["y"], s=12, alpha=0.5, label="data")
        order = np.argsort(test[0]["X"][:, 0])
        ax.plot(test[0]["X"][order, 0], yp[order], "r-", lw=1.8, label="LGF best")
        ax.set_xlabel("x1"); ax.set_ylabel("y")
        ax.set_title(f"esat (test R2={rec['test_r2']:.4f})\n{rec['best_pretty'] or rec['best_dsl'][:60]}",
                     fontsize=9)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "fit_esat.png"), dpi=130)
        plt.close(fig)
    except Exception:
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUT_DIR, "results.jsonl")
    done = set()
    if os.path.exists(jsonl_path):        # 断点续跑
        with open(jsonl_path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done.add((r["benchmark"], r["config"]))
                except Exception:
                    pass

    n_total = len(BENCHMARKS) * len(CONFIGS)
    n_done = 0
    t_all = time.time()
    for bk in BENCHMARKS:
        for cn in CONFIGS:
            if (bk, cn) in done:
                n_done += 1
                continue
            t0 = time.time()
            print(f"[{n_done+1}/{n_total}] {bk} x {cn} ...", flush=True)
            try:
                rec = run_one(bk, cn, quick=args.quick)
                status = "OK" if rec["found"] else "miss"
                print(f"    -> {status}  train={rec['train_loss']:.4g} "
                      f"testR2={rec['test_r2']:.4f}  ({time.time()-t0:.0f}s)\n"
                      f"       dsl: {rec['best_dsl'][:90]}", flush=True)
            except Exception:
                traceback.print_exc()
                rec = dict(benchmark=bk, config=cn, error=traceback.format_exc()[-400:])
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n_done += 1

    # 汇总
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for ln in f:
            try:
                records.append(json.loads(ln))
            except Exception:
                pass
    ok = [r for r in records if "error" not in r]
    print(f"\n===== {len(ok)}/{n_total} runs ok, total {time.time()-t_all:.0f}s =====")
    summary = dict(
        n_runs=len(ok), total_s=round(time.time() - t_all, 1),
        found_rate={cn: round(np.mean([r["found"] for r in ok if r["config"] == cn]), 3)
                    for cn in CONFIGS},
        mean_test_r2={cn: round(float(np.nanmean([r["test_r2"] for r in ok
                                                  if r["config"] == cn])), 4)
                      for cn in CONFIGS},
    )
    print(json.dumps(summary, indent=2))
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, records=ok), f, ensure_ascii=False,
                  indent=2, default=str)
    make_plots(ok)
    print("plots ->", PLOT_DIR)


if __name__ == "__main__":
    main()
