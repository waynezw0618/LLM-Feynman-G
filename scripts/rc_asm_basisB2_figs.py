# -*- coding: utf-8 -*-
"""rc_asm_basisB2_figs.py — 方案B v2 全矩阵作图 (论文用).

输出 outputs/asm/t23basisB2_paper/figs/:
  fig1_basisB2_bar.png       变体×方法 重组 NMSE 对比
  fig2_basisB2_profiles.png  冠军公式 b23 剖面 (DNS vs 重建, 5 工况)
  fig3_basisB2_coef.png      系数函数 a_m(Θ) 靶标 vs SR 发现
  fig4_basisB2_heatmap.png   逐工况 NMSE 热图 (12 组)
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from lgf import dsl as lgfdsl          # noqa: E402
from lgf.compiler import CompiledGraph  # noqa: E402

OUT = REPO / "outputs" / "asm"
PAPER = OUT / "t23basisB2_paper"
FIGS = PAPER / "figs"
VARIANTS = ["v", "g", "e", "m"]
METHODS = ["pysr", "lgf_mg", "lgf_go"]
LAB = {"pysr": "PySR", "lgf_mg": "LGF-mg", "lgf_go": "LGF-go"}
BASE = ["eta2", "zeta2", "s2o2", "ret", "roloc", "yw"]
ROS = [7.5, 15, 30, 75, 150]
CB, CG, CE, CM = "#3b6fb6", "#2f9e6f", "#c9524a", "#8a5fbf"


TAGS = ("", "_ms12", "_v2", "_big", "_nophys")


def _split_refit_theta(run, tg):
    """把 refit.theta 按每条 dsl 的 C# 个数切开，返回逐靶标 theta 列表。"""
    r = run.get("refit") or {}
    flat = r.get("theta")
    if not flat:
        return None
    out, o = [], 0
    for j in sorted(tg):
        n = len({s for s in re.findall(r"C\d+", tg[j]["dsl"])})
        out.append(list(flat[o:o + n])); o += n
    return out if o == len(flat) else None


def eval_target(kind, dsl, theta, X):
    """按 method 重新求值系数公式."""
    if kind != "pysr":
        g = lgfdsl.parse_dsl(dsl, BASE)
        return np.asarray(CompiledGraph(g, backend="numpy").eval_case(theta, X),
                          dtype=float)
    import sympy
    s = dsl.replace("^", "**")
    syms = {n: sympy.Symbol(n) for n in BASE}
    ex = sympy.sympify(s, locals=syms)
    f = sympy.lambdify([syms[n] for n in BASE], ex, "numpy")
    with np.errstate(all="ignore"):
        v = f(*[X[:, i] for i in range(X.shape[1])])
    return np.nan_to_num(np.asarray(v, dtype=float), nan=0.0, posinf=0.0,
                         neginf=0.0)


def reconstruct(variant, method, tag="", use_refit=True, only_cases=None):
    """重组 b23 预测 (逐工况); use_refit=True 用联合重拟合终值常数."""
    d0 = np.load(OUT / f"basisB2_targets_{variant}.npz")
    J = int(d0["J"])
    p = OUT / "runs_basisB2" / f"{variant}_{method}{tag}" / "result.json"
    run = json.load(open(p, encoding="utf-8"))
    tg = {int(t["target"][1:]) - 1: t for t in run["targets"]}
    ths = None
    if use_refit:
        ths = _split_refit_theta(run, tg)
    preds, per_case = [], (run.get("refit") or {}).get("per_case") or run["per_case"]
    for i in range(5):
        if only_cases is not None and i not in only_cases:
            preds.append(None)
            continue
        X = d0[f"c{i}_X"].astype(float)
        U = d0[f"c{i}_U"].astype(float)
        acc = np.zeros(X.shape[0])
        for j in range(J):
            t = tg[j]
            th = ths[j] if ths else t.get("theta")
            pj = eval_target(method, t["dsl"], th, X)
            acc = acc + pj * t["scale"] * U[:, j]
        preds.append(acc)
    return preds, run, d0, per_case


def load_all():
    B, R = {}, {}
    for v in VARIANTS:
        p = OUT / f"basisB2_base_{v}.json"
        if p.exists():
            B[v] = json.load(open(p, encoding="utf-8"))
    for v in VARIANTS:
        for m in METHODS:
            p = OUT / "runs_basisB2" / f"{v}_{m}" / "result.json"
            if p.exists():
                R[f"{v}_{m}"] = json.load(open(p, encoding="utf-8"))
    return B, R


def _save(fig, name):
    """同时保存 PNG(300dpi) 与 PDF 矢量双套。"""
    fig.savefig(FIGS / f"{name}.png", dpi=300)
    fig.savefig(FIGS / f"{name}.pdf")


def fig1(B, R):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.2
    x = np.arange(len(VARIANTS))
    series = [("base", "Joint solution", "#9aa6b2", -1.5),
              ("pysr", "PySR", CB, -0.5), ("lgf_mg", "LGF-mg", CG, 0.5),
              ("lgf_go", "LGF-go", CE, 1.5)]
    for key, lab, col, off in series:
        vals = []
        for v in VARIANTS:
            if key == "base":
                vals.append(B.get(v, {}).get("loss_vw", np.nan))
            else:
                vals.append(R.get(f"{v}_{key}", {}).get("loss_vw", np.nan))
        ax.bar(x + off * width, vals, width, label=lab, color=col,
               edgecolor="white", linewidth=0.6)
        for xi, vv in zip(x + off * width, vals):
            if np.isfinite(vv):
                ax.text(xi, vv * 1.12, f"{vv:.3g}", ha="center", va="bottom",
                        fontsize=7.2, rotation=90)
    ax.axhline(0.0514, color="k", ls="--", lw=1.1)
    ax.text(len(VARIANTS) - 0.45, 0.0514 * 1.25, "Scalar SOTA 0.0514", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v} basis" for v in VARIANTS])
    ax.set_ylabel("Reconstruction NMSE of $b_{23}$ (log)")
    ax.set_title("Tensor-basis coefficient caliber: joint solution vs PySR vs LGF (lower is better)",
                 fontsize=11)
    ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    _save(fig, "fig1_basisB2_bar")
    plt.close(fig)


def fig2(B, R):
    # 终口径冠军：按 refit loss (无则退回 SR loss)，扫全部 tag（直接读 result.json）
    cands = []
    for dd in os.listdir(OUT / "runs_basisB2"):
        if "_" not in dd:
            continue
        p = OUT / "runs_basisB2" / dd / "result.json"
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        if not np.isfinite(d.get("loss_vw", np.inf)):
            continue
        v = dd[0]
        rest = dd[2:]
        tag = ""
        for t in TAGS:
            if t and rest.endswith(t):
                tag, rest = t, rest[:-len(t)]
                break
        if rest not in METHODS:
            continue
        rl = (d.get("refit") or {}).get("loss_vw")
        cands.append((rl if rl is not None else d["loss_vw"], dd, v, rest, tag))
    cands.sort(key=lambda t: t[0])
    _, best_key, bv, bm, btag = cands[0]
    py_key = f"{bv}_pysr"
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.1), sharey=False)
    preds, run, d0, per_case = reconstruct(bv, bm, btag)
    ppreds = reconstruct(bv, "pysr", "_ms12")[0] if f"{bv}_pysr_ms12" in R else reconstruct(bv, "pysr")[0]
    for i, ax in enumerate(axes):
        b23 = d0[f"c{i}_b23"].astype(float)
        src = "v" if bv == "m" else bv
        d = np.load(OUT / f"dataset_trueasm_{src}.npz")
        yh = d[f"c{i}_yh"][d[f"c{i}_valid"].astype(bool)].astype(float)
        o = np.argsort(yh)
        ax.plot(yh[o], b23[o], "k-", lw=1.6, label="DNS")
        lab = f"{LAB[bm]}{btag} final (joint refit)" if btag else f"{LAB[bm]} (G×T)"
        ax.plot(yh[o], preds[i][o], "--", color=CE, lw=1.4, label=lab)
        if ppreds is not None:
            ax.plot(yh[o], ppreds[i][o], ":", color=CB, lw=1.4, label="PySR")
        ax.set_title(f"Ro$_\\tau$={ROS[i]:g}  NMSE={per_case[f'Ro{ROS[i]:g}']:.3f}",
                     fontsize=9)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("$b_{23}$")
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Best {bv}-basis × {LAB[bm]}: $b_{{23}}$ profiles reconstructed from tensor-basis coefficient formulas",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig2_basisB2_profiles")
    plt.close(fig)
    return (cands[0][0], best_key, bv, bm, btag)


def fig3(B, R, best):
    _, best_key, bv, bm, btag = best
    d0 = np.load(OUT / f"basisB2_targets_{bv}.npz")
    J = int(d0["J"])
    run = json.load(open(OUT / "runs_basisB2" / best_key / "result.json",
                         encoding="utf-8"))
    tg = {int(t["target"][1:]) - 1: t for t in run["targets"]}
    ths = _split_refit_theta(run, tg)
    ncol = min(J, 5)
    nrow = int(np.ceil(J / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow),
                             squeeze=False)
    for j in range(J):
        ax = axes[j // ncol][j % ncol]
        t = tg[j]
        th = ths[j] if ths else t.get("theta")
        tt, tp = [], []
        for i in range(5):
            X = d0[f"c{i}_X"].astype(float)
            tt.append(d0[f"c{i}_b{j}"].astype(float))
            tp.append(eval_target(bm, t["dsl"], th, X) * t["scale"])
        tt = np.concatenate(tt); tp = np.concatenate(tp)
        lim = [np.percentile(tt, 0.5), np.percentile(tt, 99.5)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.scatter(tt, tp, s=3, alpha=0.28, color=CE)
        rr = np.corrcoef(tt, tp)[0, 1]
        nm = np.mean((tp - tt) ** 2) / (np.var(tt) + 1e-300)
        ax.set_title(f"$a_{{{j+1}}}$: r={rr:.3f}, NMSE={nm:.2e}", fontsize=9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(alpha=0.25)
        if j % ncol == 0:
            ax.set_ylabel("SR discovered")
        if j // ncol == nrow - 1:
            ax.set_xlabel("Joint-solution target")
    for k in range(J, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{bv}-basis × {LAB[bm]}: coefficient functions $a_m(\\Theta)$, joint target vs SR",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_basisB2_coef")
    plt.close(fig)


def fig4(R):
    keys = [f"{v}_{m}" for v in VARIANTS for m in METHODS if f"{v}_{m}" in R]
    mat = np.full((len(keys), 5), np.nan)
    for a, k in enumerate(keys):
        for i, ro in enumerate(ROS):
            mat[a, i] = R[k]["per_case"].get(f"Ro{ro:g}", np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(keys) + 1.6))
    im = ax.imshow(np.log10(np.clip(mat, 1e-6, None)), aspect="auto",
                   cmap="viridis")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{r:g}" for r in ROS])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{k.split('_')[0]} | {LAB[k.split('_',1)[1]]}"
                        for k in keys], fontsize=8.5)
    for a in range(len(keys)):
        for i in range(5):
            if np.isfinite(mat[a, i]):
                ax.text(i, a, f"{mat[a,i]:.3g}", ha="center", va="center",
                        fontsize=7.5,
                        color="w" if np.log10(max(mat[a, i], 1e-6)) > -1.5 else "k")
    ax.set_xlabel("Ro$_\\tau$")
    ax.set_title("Per-case NMSE ($\\log_{10}$ color)", fontsize=10)
    fig.colorbar(im, ax=ax, label="log$_{10}$ NMSE")
    fig.tight_layout()
    _save(fig, "fig4_basisB2_heatmap")
    plt.close(fig)


def fig5():
    """v2 对照轮: 原始 / 等复杂度 / LGF增强 / 知识消融 分组条形图."""
    groups = [
        ("PySR full", "pysr", ""),
        ("PySR matched", "pysr", "_ms12"),
        ("LGF-go legacy", "lgf_go", ""),
        ("LGF-go enh.", "lgf_go", "_v2"),
        ("LGF-go nophys", "lgf_go", "_nophys"),
        ("LGF-mg enh.", "lgf_mg", "_v2"),
        ("LGF-mg legacy", "lgf_mg", ""),
    ]
    variants = ["g", "e"]
    colors = {"pysr": CB, "lgf_go": CE, "lgf_mg": CM}
    fig, axes = plt.subplots(1, len(variants), figsize=(4.8 * len(variants), 4.4))
    if len(variants) == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        xs, ys, cs, hs, labs = [], [], [], [], []
        for gi, (glabel, m, tag) in enumerate(groups):
            p = OUT / "runs_basisB2" / f"{v}_{m}{tag}" / "result.json"
            if not p.exists():
                continue
            d = json.load(open(p, encoding="utf-8"))
            xs.append(gi)
            ys.append(d["loss_vw"])
            cs.append(colors[m])
            hs.append("//" if tag == "_ms12" else "")
            labs.append(glabel)
        if not xs:
            continue
        bars = ax.bar(xs, ys, color=cs, edgecolor="k", linewidth=0.5)
        for b, h in zip(bars, hs):
            b.set_hatch(h)
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.06, f"{y:.3g}", ha="center", fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=7.5, rotation=22, ha="right")
        ax.set_title(f"variant {v}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, ls=":")
    axes[0].set_ylabel("reconstruction NMSE  $b_{23}$")
    fig.suptitle("Tensor-basis coefficient caliber: fairness-matched comparison",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig5_basisB2_v2_groups")
    plt.close(fig)
    print("[figs] fig5 done")


def fig6():
    """refit 后终值对比: 全部组 × {g, e}, 附 SOTA 0.0514 参考线."""
    groups = [
        ("PySR full", "pysr", ""),
        ("PySR matched", "pysr", "_ms12"),
        ("LGF-go legacy", "lgf_go", ""),
        ("LGF-mg legacy", "lgf_mg", ""),
        ("LGF-go enh.", "lgf_go", "_v2"),
        ("LGF-mg enh.", "lgf_mg", "_v2"),
        ("LGF-go nophys", "lgf_go", "_nophys"),
        ("LGF-go big", "lgf_go", "_big"),
        ("LGF-mg big", "lgf_mg", "_big"),
    ]
    variants = ["g", "e"]
    colors = {"pysr": CB, "lgf_go": CE, "lgf_mg": CM}
    fig, axes = plt.subplots(1, len(variants), figsize=(6.2 * len(variants), 4.6))
    for ax, v in zip(np.atleast_1d(axes), variants):
        xs, ys, cs, labs = [], [], [], []
        for gi, (glabel, m, tag) in enumerate(groups):
            p = OUT / "runs_basisB2" / f"{v}_{m}{tag}" / "result.json"
            if not p.exists():
                continue
            d = json.load(open(p, encoding="utf-8"))
            rf = d.get("refit")
            if not rf or rf.get("loss_vw") is None:
                continue
            xs.append(gi)
            ys.append(rf["loss_vw"])
            cs.append(colors[m])
            labs.append(glabel)
        if not xs:
            continue
        bars = ax.bar(xs, ys, color=cs, edgecolor="k", linewidth=0.5)
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.06, f"{y:.3g}", ha="center", fontsize=8)
        ax.axhline(0.0514, color="crimson", ls="--", lw=1.2)
        ax.text(xs[-1] + 0.4, 0.054, "scalar SOTA 0.0514", color="crimson",
                fontsize=8, ha="right")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=8, rotation=22, ha="right")
        ax.set_title(f"variant {v} — joint-constant refit", fontsize=10)
        ax.grid(axis="y", alpha=0.3, ls=":")
    np.atleast_1d(axes)[0].set_ylabel("reconstruction NMSE  $b_{23}$ (refit)")
    fig.suptitle("Final caliber after joint refit of all SR formulas",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig6_basisB2_refit_final")
    plt.close(fig)
    print("[figs] fig6 done")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    B, R = load_all()
    fig1(B, R)
    best = fig2(B, R)
    fig3(B, R, best)
    fig4(R)
    try:
        fig5()
    except Exception as e:
        print(f"[figs] fig5 skipped: {e}")
    try:
        fig6()
    except Exception as e:
        print(f"[figs] fig6 skipped: {e}")
    json.dump({"best": best}, open(FIGS / "_meta.json", "w"), indent=2)
    print(f"[figs] 完成 -> {FIGS}  (best={best})")


if __name__ == "__main__":
    main()
