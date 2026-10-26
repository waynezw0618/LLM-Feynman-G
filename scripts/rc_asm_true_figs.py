# -*- coding: utf-8 -*-
"""rc_asm_true_figs.py — 真 ASM 论文图与最终方程汇总.

数据源 (全部已完成, 无 LLM/Julia 依赖):
  runs_trueasm_pysr/{v,g,e}/result.json          PySR best + pareto (含逐条 metrics)
  runs_trueasm_lgf/{v,g,e}_{mg,go}/result.json   LGF best (dsl + theta + metrics)
  runs_trueasm_lgf/*/trueasm_*_log.jsonl         LGF 逐代 best_graph (重拟合后评 rho_a)
  prep_report_trueasm.json                       基线 (cc10 等)

产出:
  outputs/asm/figs_trueasm/fig_profile_{v,g,e}.png       6 分量 × 5 工况 b_ij 剖面对比
  outputs/asm/figs_trueasm/fig_iter_lgf.png              LGF 6 run 逐代 loss_asm / mean rho_a
  outputs/asm/figs_trueasm/fig_iter_lgf_rho_cases.png    LGF 逐代 rho_a,case (W&S eq.22) 分面
  outputs/asm/figs_trueasm/fig_iter_pysr.png             PySR Pareto loss_asm / rho_a vs complexity
  outputs/asm/ASM_trueasm_equations.md                   9 个最优方程 (常数代入) + 特征使用

迭代图口径说明 (诚实声明, 写入 md):
  LGF 日志只存逐代 best_graph (无 theta) — 对每个去重后的图用与 final refit
  相同的设置 (n_starts=8, maxiter=150) 在全量行上重新拟合再评估,
  保证迭代曲线与最终模型同一评估口径; 末点 "final" 取 result.json best.
  PySR 未存逐迭代历史, 只有 Pareto 前沿 → 以 complexity 为横轴.

用法: python -X utf8 scripts/rc_asm_true_figs.py
"""
from __future__ import annotations
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_true_eval import (load_variant, eval_predictions, lgf_predict,  # noqa: E402
                              COMP6, RO_CASES)
from rc_asm_true_prep import FEAT_NAMES  # noqa: E402
from lgf import dsl as lgfdsl  # noqa: E402
from lgf.paramfit import fit_joint  # noqa: E402

OUT = os.path.join(REPO, "outputs", "asm")
FIG = os.path.join(OUT, "figs_trueasm")
VARIANTS = ["v", "g", "e"]
ARMS = ["mg", "go"]
RUNS_LGF = [f"{v}_{a}" for v in VARIANTS for a in ARMS]
VAR_LABEL = dict(v=r"$v$: $\mathbf{W}=\overline{W}$ (vorticity)",
                 g=r"$g$: GS, $\mathbf{W}=\overline{W}+\Omega$",
                 e=r"$e$: $\mathbf{W}=\overline{W}+2\Omega$")
COMP_TEX = {"uu": r"$b_{11}$", "vv": r"$b_{22}$", "ww": r"$b_{33}$",
            "uv": r"$b_{12}$", "uw": r"$b_{13}$", "vw": r"$b_{23}$"}
RO_COLORS = plt.cm.viridis(np.linspace(0.10, 0.92, len(RO_CASES)))
VAR_COLORS = {"v": "tab:blue", "g": "tab:green", "e": "tab:purple"}

FEATS = [f"T{m}" for m in range(1, 11)] + \
        ["eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc"]


# ---------------------------------------------------------------- utilities
def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pysr(v):
    return _load(os.path.join(OUT, "runs_trueasm_pysr", v, "result.json"))


def load_lgf(run):
    return _load(os.path.join(OUT, "runs_trueasm_lgf", run, "result.json"))


def lgf_best(j):
    for e in j.get("entries", []):
        if e.get("tag") == "best":
            return e
    return None


def feature_usage(formula: str):
    used = []
    for f in FEATS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(f)}(?![A-Za-z0-9_])", formula):
            used.append(f)
    return used


def subst_constants(dsl: str, theta, var_names):
    """DSL 中 C# 替换为拟合值 (按 global_names 顺序对应 theta)."""
    g = lgfdsl.parse_dsl(dsl, var_names)
    names = g.global_names()
    m = {n: float(t) for n, t in zip(names, np.asarray(theta, dtype=float))}
    def rep(mo):
        k = mo.group(0)
        return f"({m[k]:.6g})" if k in m else k
    return re.sub(r"\bC\d+\b", rep, dsl)


def load_preds(j):
    pf = j.get("preds_file")                     # PySR: 顶层
    if not pf:
        e = lgf_best(j)                          # LGF: entries[0] (tag=="best")
        pf = (e or {}).get("preds_file")
    if not pf or not os.path.exists(pf):
        return None
    z = np.load(pf)
    return [np.asarray(z[f"c{i}"], dtype=float) for i in range(len(RO_CASES))]


# ---------------------------------------------------------------- Q1: equations
def write_equations_md():
    var_names = list(FEAT_NAMES)
    lines = []
    lines.append("# 真 ASM 最优模型方程汇总（常数代入）\n")
    lines.append("生成: rc_asm_true_figs.py · 单位: 无量纲 (b_ij, S/W 已乘 k/eps)\n")
    lines.append("## 1. 建模输入特征（对问题 2 的回答）\n")
    lines.append("全部 17 个特征均作为候选输入提供给 PySR 与 LGF，由算法自选：\n")
    lines.append("- **Pope (1975) 十个对称无迹张量基**: T1..T10（在当前张量分量 (ij) 处取值，"
                 "S_ij 与 W_ij 已乘 k/eps 归一化）")
    lines.append("- **5 个标量不变量**: eta2=tr(S²), zeta2=tr(W²), s3=tr(S³), so2=tr(S²W), s2o2=tr(S²W²)")
    lines.append("- **湍流雷诺数**: ret=k²/(ν·eps)")
    lines.append("- **局部旋转数**: roloc=2Ωk/eps（带符号）\n")
    lines.append("结构上即 W&S (JCP 2016) eq.19: b_ij = Σ_m g_m(I, Re_t, Ro_loc)·T^(m)_ij，"
                 "g_m 的函数形式由符号回归发现，常数一律为跨 5 工况全局常数。\n")
    lines.append("注: s3、so2 在本二维平均流场恒为零；v 变体下 zeta2≡eta2。"
                 "ret/roloc 提供了但**没有任何一条最优公式选用**（详见下方表格与报告 §6）。\n")

    lines.append("## 2. 主对比表\n")
    lines.append("| 变体 | 方法 | loss_asm ↓ | rho_a ↑ | 最差工况 nmse | 复杂度 | 选用特征 |")
    lines.append("|---|---|---|---|---|---|---|")
    eq_blocks = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        pj = load_pysr(v)
        b = pj.get("best")
        if b:
            m = b["metrics"]
            feats = ", ".join(feature_usage(b["sympy"])) or "—(纯常数)"
            lines.append(f"| {v} | PySR | {m['loss_asm']:.4f} | {m['rho_a']:+.3f} "
                         f"| {m['worst_case_nmse']:.4f} | {b['complexity']} nodes | {feats} |")
            eq_blocks[v]["PySR"] = b
        for arm in ARMS:
            j = load_lgf(f"{v}_{arm}")
            e = lgf_best(j)
            if not e:
                continue
            m = e["metrics"]
            cx = e["complexity"]
            feats = ", ".join(feature_usage(e["dsl"])) or "—(纯常数)"
            lines.append(f"| {v} | LGF-{arm} ({j.get('backend')}) | {m['loss_asm']:.4f} "
                         f"| {m['rho_a']:+.3f} | {m['worst_case_nmse']:.4f} "
                         f"| {cx.get('edges', '?')} edges | {feats} |")
            eq_blocks[v][f"LGF-{arm}"] = (e, j)

    lines.append("\n## 3. 全部最优方程\n")
    for v in VARIANTS:
        lines.append(f"### 变体 {v} — {VAR_LABEL[v]}\n")
        b = eq_blocks[v].get("PySR")
        if b:
            lines.append(f"**PySR** (complexity={b['complexity']}, "
                         f"loss_asm={b['metrics']['loss_asm']:.4f}, "
                         f"rho_a={b['metrics']['rho_a']:+.3f}):\n")
            lines.append("```")
            lines.append("b_ij = " + b["sympy"])
            lines.append("```\n")
        for arm in ARMS:
            blk = eq_blocks[v].get(f"LGF-{arm}")
            if not blk:
                continue
            e, j = blk
            m = e["metrics"]
            sub = subst_constants(e["dsl"], e["theta"], var_names)
            lines.append(f"**LGF-{arm}** ({j.get('backend')}/{j.get('model')}, "
                         f"edges={e['complexity'].get('edges')}, "
                         f"loss_asm={m['loss_asm']:.4f}, rho_a={m['rho_a']:+.3f}):\n")
            lines.append("```")
            lines.append("b_ij = " + sub)
            lines.append("```\n")
            lines.append(f"<sub>原始 DSL: `{e['dsl']}` · theta="
                         f"`{[round(float(t), 6) for t in e['theta']]}`</sub>\n")

    lines.append("## 4. 迭代图口径说明（诚实声明）\n")
    lines.append("- **LGF**: 逐代日志只保存 best_graph（无 θ）。为保证与最终模型同一评估口径，"
                 "对每个去重后的逐代最优图用与 final refit 相同设置（n_starts=8, maxiter=150）"
                 "在全量行（5 工况 × 6 分量）上重新拟合后评估 loss_asm 与 rho_a,case；"
                 "末点 `final` 为 result.json 的最终最优模型（final refit 产物，可能与末代图不同）。")
    lines.append("- **LGF 曲线语义**: 粗线 = best-so-far 包络（loss 取累积最小、rho_a 取累积最大），"
                 "细淡线 = 逐代原始值。包络用于抑制逐代重拟合落入劣质局部极小造成的非单调上跳；"
                 "星号 = result.json 的 final refit 模型，位于末代 +1.5 处。")
    lines.append("- **PySR**: 未保存逐迭代历史，仅 Pareto 前沿（复杂度阶梯）→ "
                 "以 complexity 为横轴画 loss_asm / rho_a（等效的复杂度-精度权衡曲线）。")
    lines.append("- 所有 rho_a,case 均为 W&S (2016) eq.22 的离散实现：逐点 3×3 张量余弦对有效壁面点平均。\n")

    path = os.path.join(OUT, "ASM_trueasm_equations.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[figs] equations -> {path}")
    return path


# ---------------------------------------------------------------- Q3: profiles
def plot_profiles():
    for v in VARIANTS:
        data = load_variant(v)
        pj = load_pysr(v)
        methods = []
        p = load_preds(pj)
        if p is not None:
            methods.append(("PySR", "tab:blue", "-", p))
        for arm, color in (("mg", "tab:orange"), ("go", "tab:red")):
            j = load_lgf(f"{v}_{arm}")
            p = load_preds(j)
            if p is not None:
                methods.append((f"LGF-{arm}", color, "-", p))

        fig, axes = plt.subplots(6, 5, figsize=(15.5, 12.8), sharex=True,
                                 constrained_layout=True)
        for ci, case in enumerate(data):
            Nv = case["Nv"]
            for c, name in enumerate(COMP6):
                ax = axes[c, ci]
                yh = case["yh"]
                ax.plot(yh, case["b6v"][:, c], color="k", lw=1.8, zorder=5)
                for label, color, ls, preds in methods:
                    pred = preds[ci].reshape(6, Nv).T[:, c]
                    ax.plot(yh, pred, color=color, lw=1.1, ls=ls, alpha=0.9)
                ax.axhline(0.0, color="0.75", lw=0.5)
                if c == 0:
                    ax.set_title(f"Ro = {case['ro']:g}", fontsize=10)
                if c == 5:
                    ax.set_xlabel(r"$y/h$", fontsize=9)
                if ci == 0:
                    ax.set_ylabel(COMP_TEX[name], fontsize=10)
                ax.tick_params(labelsize=8)
        handles = [plt.Line2D([], [], color="k", lw=1.8, label="DNS")]
        handles += [plt.Line2D([], [], color=color, lw=1.3, label=label)
                    for label, color, _, _ in methods]
        fig.suptitle(f"Reynolds-stress anisotropy profiles — variant {VAR_LABEL[v]}",
                     fontsize=12)
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   frameon=False, fontsize=10)
        out = os.path.join(FIG, f"fig_profile_{v}.png")
        fig.savefig(out, dpi=170)
        plt.close(fig)
        print(f"[figs] profile {v} -> {out} ({len(methods)+1} curves)")


# ---------------------------------------------------------------- Q4: LGF iter
def refit_eval(dsl, data, cache):
    if dsl in cache:
        return cache[dsl]
    var_names = list(FEAT_NAMES)
    try:
        g = lgfdsl.parse_dsl(dsl, var_names)
        if g.case_names():
            cache[dsl] = None
            return None
        cases_fit = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}")
                     for c in data]
        fr = fit_joint(g, cases_fit, n_starts=8, maxiter=150, seed=0)
        if not np.isfinite(fr.loss) or fr.loss > 1e6:
            cache[dsl] = None
            return None
        preds, _ = lgf_predict(dsl, fr.theta, data)
        m = eval_predictions(preds, data)
        cache[dsl] = m
        return m
    except Exception as ex:
        print(f"[figs][WARN] refit failed: {type(ex).__name__}: {ex} :: {dsl[:80]}")
        cache[dsl] = None
        return None


def lgf_iter_curves():
    """每 run → dict(gens, loss, rho, rho_case[5], final=(loss, rho, rho_case[5]))."""
    out = {}
    cache = {}
    for v in VARIANTS:
        data = load_variant(v)  # refit 缓存按 DSL; 变体不同数据不同 → 分变体缓存
        cache = {}
        for arm in ARMS:
            run = f"{v}_{arm}"
            logp = os.path.join(OUT, "runs_trueasm_lgf", run,
                                f"trueasm_{run}_log.jsonl")
            gens, loss, rho, rho_case = [], [], [], [[] for _ in RO_CASES]
            with open(logp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    m = refit_eval(r["best_graph"], data, cache)
                    gens.append(r["gen"])
                    if m is None:
                        loss.append(np.nan); rho.append(np.nan)
                        for k in range(len(RO_CASES)):
                            rho_case[k].append(np.nan)
                    else:
                        loss.append(m["loss_asm"]); rho.append(m["rho_a"])
                        for k, key in enumerate(m["per_case"]):
                            rho_case[k].append(m["per_case"][key]["rho_a"])
            j = load_lgf(run)
            e = lgf_best(j)
            fin = None
            if e and e["metrics"].get("per_case"):
                mm = e["metrics"]
                fin = (mm["loss_asm"], mm["rho_a"],
                       [mm["per_case"][k]["rho_a"] for k in mm["per_case"]])
            out[run] = dict(gens=np.array(gens), loss=np.array(loss),
                            rho=np.array(rho),
                            rho_case=[np.array(x) for x in rho_case], final=fin)
            print(f"[figs] iter {run}: {len(gens)} gens, "
                  f"final loss={fin[0]:.4f} rho={fin[1]:+.3f}" if fin else
                  f"[figs] iter {run}: {len(gens)} gens, NO final")
    return out


def plot_iter_lgf(curves):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    prep = _load(os.path.join(OUT, "prep_report_trueasm.json"))
    for v in VARIANTS:
        for arm in ARMS:
            run = f"{v}_{arm}"
            c = curves[run]
            ls = "-" if arm == "go" else "--"
            # 细淡线 = 逐代重拟合值 (非单调, 重拟合噪声); 粗线 = best-so-far 包络
            for ax, arr, red in ((axes[0], c["loss"], np.fmin),
                                 (axes[1], c["rho"], np.fmax)):
                ax.plot(c["gens"], arr, ls=ls, color=VAR_COLORS[v], lw=0.7,
                        alpha=0.30)
                env = red.accumulate(np.asarray(arr, dtype=float))
                ax.plot(c["gens"], env, ls=ls, color=VAR_COLORS[v], lw=2.0,
                        label=f"{v}-{arm}")
            if c["final"] is not None:
                xf = c["gens"][-1] + 1.5
                axes[0].plot(xf, c["final"][0], marker="*", ms=11,
                             color=VAR_COLORS[v], mec="k", mew=0.4)
                axes[1].plot(xf, c["final"][1], marker="*", ms=11,
                             color=VAR_COLORS[v], mec="k", mew=0.4)
    # cc10 基线 (loss panel)
    for v, dy in zip(VARIANTS, (0.0, 0.035, 0.07)):
        b = prep[v]["baselines"]["cc10"]["rel_resid"]
        axes[0].axhline(b, color=VAR_COLORS[v], ls=":", lw=1.0, alpha=0.8)
        axes[0].text(0.99, b + dy, f"cc10-{v}", color=VAR_COLORS[v], fontsize=7.5,
                     ha="right", transform=axes[0].get_yaxis_transform())
    axes[0].set_yscale("log")
    axes[0].set_xlabel("generation"); axes[0].set_ylabel(r"loss$_{asm}$ (log)")
    axes[1].set_xlabel("generation"); axes[1].set_ylabel(r"mean $\rho_a$ (W&S eq.22)")
    axes[1].set_ylim(-0.2, 1.0)
    axes[0].set_xlim(-1, 43); axes[1].set_xlim(-1, 43)
    axes[0].text(41.3, axes[0].get_ylim()[1], "", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    star = plt.Line2D([], [], marker="*", color="0.4", ls="", ms=10,
                      label="final refit")
    handles.append(star); labels.append("final refit")
    fig.legend(handles, labels, loc="outside upper center", ncol=7, frameon=False,
               fontsize=8.5)
    out = os.path.join(FIG, "fig_iter_lgf.png")
    fig.savefig(out, dpi=170); plt.close(fig)
    print(f"[figs] -> {out}")


def plot_iter_lgf_cases(curves):
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.0), constrained_layout=True,
                             sharex=True, sharey=True)
    for r, v in enumerate(VARIANTS):
        for cix, arm in enumerate(ARMS):
            ax = axes[r, cix]
            cur = curves[f"{v}_{arm}"]
            for k, ro in enumerate(RO_CASES):
                ax.plot(cur["gens"], cur["rho_case"][k], color=RO_COLORS[k],
                        lw=0.9, alpha=0.85, label=f"Ro={ro:g}")
            ax.plot(cur["gens"], cur["rho"], color="k", lw=2.0, label="mean")
            if cur["final"] is not None:
                xf = cur["gens"][-1] + 1.5
                ax.plot(xf, cur["final"][1], marker="*", ms=12, color="k",
                        mec="k", mew=0.4)
                for k in range(len(RO_CASES)):
                    ax.plot(xf, cur["final"][2][k], marker="*", ms=8,
                            color=RO_COLORS[k], mec="k", mew=0.3)
            ax.axhline(0.0, color="0.75", lw=0.5)
            ax.set_ylim(-0.45, 1.0)
            armname = "M_GUIDE" if arm == "mg" else "G_INIT only"
            ax.set_title(f"{v} / {arm} ({armname})", fontsize=10)
            if r == 2:
                ax.set_xlabel("generation")
            if cix == 0:
                ax.set_ylabel(r"$\rho_{a,case}$")
            if r == 0 and cix == 0:
                ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower right")
    out = os.path.join(FIG, "fig_iter_lgf_rho_cases.png")
    fig.savefig(out, dpi=170); plt.close(fig)
    print(f"[figs] -> {out}")


# ---------------------------------------------------------------- Q4: PySR pareto
def plot_iter_pysr():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True,
                             sharex="col")
    for cix, v in enumerate(VARIANTS):
        pj = load_pysr(v)
        par = [r for r in pj["pareto"]
               if np.isfinite(r["metrics"].get("loss_asm", np.inf))]
        par.sort(key=lambda r: r["complexity"])
        cx = [r["complexity"] for r in par]
        lo = [r["metrics"]["loss_asm"] for r in par]
        rh = [r["metrics"]["rho_a"] for r in par]
        bi = pj.get("best", {}).get("index")
        axes[0, cix].plot(cx, lo, "-o", color=VAR_COLORS[v], lw=1.3, ms=4)
        axes[0, cix].set_yscale("log")
        axes[0, cix].set_title(VAR_LABEL[v], fontsize=10)
        axes[0, cix].set_ylabel(r"loss$_{asm}$ (log)")
        axes[1, cix].plot(cx, rh, "-o", color=VAR_COLORS[v], lw=1.3, ms=4,
                          label=r"mean $\rho_a$")
        # 逐工况 rho 的散点 (论文审稿常要 per-case 证据)
        for k, key in enumerate(par[0]["metrics"]["per_case"]):
            ry = [r["metrics"]["per_case"][key]["rho_a"] for r in par]
            axes[1, cix].scatter(cx, ry, s=8, color=RO_COLORS[k], alpha=0.55,
                                 zorder=1, label=fr"Ro={key[2:]}")
        for r in (0, 1):
            for rec in par:
                if rec["index"] == bi:
                    axes[r, cix].plot(rec["complexity"],
                                      rec["metrics"]["loss_asm"] if r == 0
                                      else rec["metrics"]["rho_a"],
                                      marker="*", ms=14, color="gold",
                                      mec="k", mew=0.5, zorder=6)
        axes[1, cix].set_xlabel("equation complexity (nodes)")
        axes[1, cix].set_ylabel(r"$\rho_a$ (W&S eq.22)")
        axes[1, cix].set_ylim(-0.2, 1.0)
        if cix == 0:
            axes[1, cix].legend(fontsize=6.5, ncol=2, frameon=False,
                                 loc="lower right")
    out = os.path.join(FIG, "fig_iter_pysr.png")
    fig.savefig(out, dpi=170); plt.close(fig)
    print(f"[figs] -> {out}")


def main():
    os.makedirs(FIG, exist_ok=True)
    write_equations_md()
    plot_profiles()
    curves = lgf_iter_curves()
    plot_iter_lgf(curves)
    plot_iter_lgf_cases(curves)
    plot_iter_pysr()
    print("[figs] ALL DONE")


if __name__ == "__main__":
    main()
