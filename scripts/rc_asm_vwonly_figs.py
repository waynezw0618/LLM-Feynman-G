# -*- coding: utf-8 -*-
"""rc_asm_vwonly_figs.py — vw-only (b23) 跑批汇总 + 剖面图 + 基线重算.

输出:
  outputs/asm/figs_vwonly/fig_vw_profiles.png   最优模型 vs DNS vs 旧 mix 单公式
  outputs/asm/figs_vwonly/fig_vw_nmse_bar.png   12 组合 + 2 基线的 loss_vw 柱状图
  outputs/asm/figs_vwonly/vwonly_summary.json   全部指标 (报告数据源)

基线重算口径: 旧 mix 单公式 (6 分量 loss) 的 preds_best.npz 取 vw 行,
用与 vw-only 完全相同的 NMSE 定义 (mse / var_vw_per_case) 重算, 保证公平.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rc_asm_vw_only import load_vw_data, eval_vw_predictions, VW_IDX
from rc_asm_true_mix import load_mix

ASM = REPO / "outputs" / "asm"
FIG_DIR = ASM / "figs_vwonly"
RO_CASES = [7.5, 15, 30, 75, 150]
METHODS = ["pysr", "lgf_go", "lgf_mg"]
VARIANTS = ["v", "g", "e", "m"]
VARIANT_LABEL = {"v": "v (W\u0304)", "g": "g (W\u0304+\u03a9)",
                 "e": "e (W\u0304+2\u03a9)", "m": "m (mix 43d)"}
METHOD_LABEL = {"pysr": "PySR", "lgf_go": "LGF-go", "lgf_mg": "LGF-mg"}


def nmse_vw(pred: np.ndarray, y: np.ndarray, var_c: float) -> float:
    p = np.asarray(pred, dtype=float).reshape(-1)
    if not np.all(np.isfinite(p)):
        return float("inf")
    return float(np.mean((p - y) ** 2) / (var_c + 1e-300))


def load_vwonly_result(method: str, variant: str):
    p = ASM / f"runs_vwonly_{method}" / variant / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    if method == "pysr":
        best = r.get("best")
        if best is None:
            return None
        return dict(result=r, best=best, metrics=best["metrics"],
                    expr=best.get("sympy", ""), complexity=best["complexity"])
    entries = [e for e in r.get("entries", []) if e.get("tag") == "best"]
    if not entries:
        return None
    e0 = entries[0]
    if "per_case" not in e0.get("metrics", {}):
        return None
    return dict(result=r, best=e0, metrics=e0["metrics"],
                expr=e0["dsl"], complexity=e0["complexity"])


def recompute_baseline(tag: str, data_vw: list):
    """旧 mix 单公式基线: 用 preds_best.npz 的 vw 行按统一口径重算."""
    d = ASM / f"runs_trueasm_{tag}"
    pf = d / "preds_best.npz"
    if not pf.exists():
        return None
    z = np.load(pf)
    per_case = {}
    nmses = []
    for i, case in enumerate(data_vw):
        p = z[f"c{i}"].reshape(-1)
        Nv = case["Nv"]
        s = VW_IDX * Nv
        pvw = p[s:s + Nv] if p.shape[0] == 6 * Nv else p
        m = nmse_vw(pvw, case["yfit"], case["var_c"])
        per_case[f"Ro{RO_CASES[i]:g}"] = m
        nmses.append(m)
    return dict(loss_vw=float(np.mean(nmses)), per_case=per_case)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # vw-only 数据 (任一变体的 yfit 都一样, 用 v 取真值剖面)
    data_v, _ = load_vw_data("v")
    data_m, _ = load_vw_data("m")
    # 基线: mix 数据结构 (43 列, 行展开 6 分量)
    data_mix = load_mix()
    y_truth = [c["yfit"].copy() for c in data_v]   # 与 m 相同? 校验
    for cv, cm in zip(data_v, data_m):
        assert np.allclose(cv["yfit"], cm["yfit"]), "v/m 数据 vw 行不一致!"
        assert np.allclose(cv["yfit"], data_mix[cv["i"]]["yfit"][5 * cv["Nv"]:6 * cv["Nv"]]), \
            "vw-only 与 mix 数据 vw 行不一致!"

    # ---- 收集 12 组合 ----
    combos = {}
    for method in METHODS:
        for var in VARIANTS:
            r = load_vwonly_result(method, var)
            if r is None:
                print(f"[miss] {method}/{var}")
                continue
            combos[f"{method}/{var}"] = r
            pc = " ".join(f"Ro{k[2:]}:{v['nmse']:.3f}"
                          for k, v in r["metrics"]["per_case"].items())
            print(f"[ok] {method}/{var}: loss_vw={r['metrics']['loss_vw']:.4f} | {pc}")

    # ---- 基线 ----
    baselines = {}
    for tag in ("mix_lgf_go", "mix_pysr"):
        b = recompute_baseline(tag, data_v)
        if b:
            baselines[tag] = b
            pc = " ".join(f"Ro{k[2:]}:{v:.3f}" for k, v in b["per_case"].items())
            print(f"[base] {tag}: loss_vw={b['loss_vw']:.4f} | {pc}")

    if not combos:
        print("尚无完成的组合, 退出")
        return

    # ---- summary json ----
    summary = dict(
        combos={k: dict(expr=v["expr"], complexity=v["complexity"],
                        metrics=v["metrics"],
                        elapsed_s=v["result"].get("elapsed_s"))
                for k, v in combos.items()},
        baselines=baselines)
    with open(FIG_DIR / "vwonly_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 排名 ----
    rank = sorted(combos.items(), key=lambda kv: kv[1]["metrics"]["loss_vw"])
    best_key, best_r = rank[0]
    print(f"\n[BEST] {best_key} loss_vw={best_r['metrics']['loss_vw']:.4f}")

    # ---- 图 1: 剖面对比 ----
    bm, bv = best_key.split("/")
    best_dir = ASM / f"runs_vwonly_{bm}" / bv
    zb = np.load(best_dir / "preds_best.npz")
    zmix = None
    if "mix_lgf_go" in baselines:
        zmix = np.load(ASM / "runs_trueasm_mix_lgf_go" / "preds_best.npz")

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for i, (ax, case) in enumerate(zip(axes, data_v)):
        yh = case["yh"]
        ax.plot(y_truth[i], yh, "ko", ms=3, label="DNS")
        pb = zb[f"c{i}"].reshape(-1)
        ax.plot(pb, yh, "r-", lw=1.6,
                label=f"vw-only best ({METHOD_LABEL[bm]}, {VARIANT_LABEL[bv]})")
        if zmix is not None:
            Nv = case["Nv"]
            pm = zmix[f"c{i}"].reshape(-1)
            pm = pm[5 * Nv:6 * Nv] if pm.shape[0] == 6 * Nv else pm
            ax.plot(pm, yh, "b--", lw=1.2, label="mix single-formula (6-comp loss)")
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(f"Ro$_\\tau$={RO_CASES[i]:g}", fontsize=10)
        ax.set_xlabel("b$_{23}$ = <vw>/(2k)", fontsize=9)
        if i == 0:
            ax.set_ylabel("y/h", fontsize=10)
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("vw-only retraining: b23 profiles vs DNS", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_vw_profiles.png", dpi=300)
    plt.close(fig)

    # ---- 图 2: loss_vw 柱状图 ----
    labels, vals, colors = [], [], []
    for (method, var) in [(m, v) for m in METHODS for v in VARIANTS]:
        k = f"{method}/{var}"
        if k not in combos:
            continue
        labels.append(f"{METHOD_LABEL[method]}\n{var}")
        vals.append(combos[k]["metrics"]["loss_vw"])
        colors.append("tab:red" if k == best_key else "tab:blue")
    for tag, b in baselines.items():
        labels.append(f"{tag.replace('mix_', 'mix\n')}")
        vals.append(b["loss_vw"])
        colors.append("tab:gray")

    fig, ax = plt.subplots(figsize=(12, 4))
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=colors)
    for x, v in zip(xs, vals):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs, labels, fontsize=7)
    ax.set_ylabel("loss_vw (mean NMSE over 5 Ro)")
    ax.set_title("vw-only (b23) retraining: 12 combos + baselines")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_vw_nmse_bar.png", dpi=300)
    plt.close(fig)

    print(f"\n-> {FIG_DIR}")


if __name__ == "__main__":
    main()
