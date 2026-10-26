# -*- coding: utf-8 -*-
"""rc_asm_true_mix_figs.py — 补画 MIX 两个 run 的 b_ij 剖面图 (fig_profile_mix).

背景: fig_profile_{v,g,e}.png 只覆盖单变体 run; mix_lgf_go / mix_pysr
(43 维混合特征空间) 此前只有散点图 (fig_mix_compare) 与逐工况 rho 条形图,
没有 6 分量 x 5 工况的剖面对比. 本脚本补齐:

  outputs/asm/figs_trueasm/fig_profile_mix.png       6 分量 x 5 工况 (DNS vs LGF vs PySR)
  outputs/asm/figs_trueasm/fig_profile_mix_b12b23.png  b12/b23 聚焦 (2 x 5, 含逐工况 NMSE 标注)

数据口径 (与 rc_asm_true_mix_writeup.per_comp_metrics 一致):
  dataset_trueasm_mix.npz: c{i}_yh / c{i}_valid / c{i}_b6 / c{i}_ro
  preds_*.npz: c{i} 长度 6*Nv, 分量 j 占 [j*Nv, (j+1)*Nv) — 逐分量连续块.

用法: python -X utf8 scripts/rc_asm_true_mix_figs.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "asm")
FIG = os.path.join(OUT, "figs_trueasm")

CMP = ["b11", "b22", "b33", "b12", "b13", "b23"]
COMP_TEX = {"b11": r"$b_{11}$", "b22": r"$b_{22}$", "b33": r"$b_{33}$",
            "b12": r"$b_{12}$", "b13": r"$b_{13}$", "b23": r"$b_{23}$"}
COMP_ORDER = ["b11", "b12", "b13", "b22", "b23", "b33"]  # 行顺序 (按张量下标)

RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
RO_COLORS = plt.cm.viridis(np.linspace(0.10, 0.92, len(RO_CASES)))


def load_mix():
    ds = np.load(os.path.join(OUT, "dataset_trueasm_mix.npz"), allow_pickle=True)
    cases = []
    for i in range(5):
        valid = ds[f"c{i}_valid"].astype(bool)
        cases.append(dict(
            ro=float(ds[f"c{i}_ro"]),
            yh=np.asarray(ds[f"c{i}_yh"])[valid],
            b6=np.asarray(ds[f"c{i}_b6"])[valid],   # (Nv, 6) 按 CMP 顺序
            Nv=int(valid.sum()),
        ))
    # preds: (name, color) -> list of (Nv,6)
    plgf = np.load(os.path.join(OUT, "runs_trueasm_mix_lgf_go", "preds_best.npz"))
    with open(os.path.join(OUT, "runs_trueasm_mix_pysr", "result.json"),
              encoding="utf-8") as f:
        jp = json.load(f)
    ppsr = np.load(jp["preds_file"])

    def to_b6(z):
        out = []
        for i in range(5):
            Nv = cases[i]["Nv"]
            v = np.asarray(z[f"c{i}"], dtype=float).reshape(6, Nv).T
            out.append(v)
        return out

    methods = [("mix_lgf_go", "tab:red", to_b6(plgf)),
               ("mix_pysr", "tab:blue", to_b6(ppsr))]
    return cases, methods


def per_comp_nmse(cases, preds):
    """逐工况 x 逐分量 NMSE (用于 b12/b23 图的标注)."""
    nmse = []
    for i, case in enumerate(cases):
        yt, yp = case["b6"], preds[i]
        sse = np.sum((yp - yt) ** 2, axis=0)
        sst = np.sum(yt ** 2, axis=0)
        nmse.append(sse / np.maximum(sst, 1e-300))
    return np.array(nmse)  # (5, 6)


def plot_full(cases, methods):
    fig, axes = plt.subplots(6, 5, figsize=(15.5, 12.8), sharex=True,
                             constrained_layout=True)
    for ci, case in enumerate(cases):
        for r, comp in enumerate(COMP_ORDER):
            j = CMP.index(comp)
            ax = axes[r, ci]
            ax.plot(case["yh"], case["b6"][:, j], color="k", lw=1.8, zorder=5)
            for label, color, preds in methods:
                ax.plot(case["yh"], preds[ci][:, j], color=color, lw=1.1,
                        alpha=0.9)
            ax.axhline(0.0, color="0.75", lw=0.5)
            if r == 0:
                ax.set_title(f"Ro = {case['ro']:g}", fontsize=10)
            if r == 5:
                ax.set_xlabel(r"$y/h$", fontsize=9)
            if ci == 0:
                ax.set_ylabel(COMP_TEX[comp], fontsize=10)
            ax.tick_params(labelsize=8)
    handles = [plt.Line2D([], [], color="k", lw=1.8, label="DNS")]
    handles += [plt.Line2D([], [], color=c, lw=1.3, label=l)
                for l, c, _ in methods]
    fig.suptitle(r"Reynolds-stress anisotropy profiles — MIX (43-feature space)",
                 fontsize=12)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=10)
    out = os.path.join(FIG, "fig_profile_mix.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[mixfigs] -> {out}")


def plot_b12b23(cases, methods):
    nmse = {label: per_comp_nmse(cases, preds) for label, _, preds in methods}
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 5.6), sharex=True,
                             constrained_layout=True)
    for r, comp in enumerate(("b12", "b23")):
        j = CMP.index(comp)
        for ci, case in enumerate(cases):
            ax = axes[r, ci]
            ax.plot(case["yh"], case["b6"][:, j], color="k", lw=1.9, zorder=5)
            for k, (label, color, preds) in enumerate(methods):
                ax.plot(case["yh"], preds[ci][:, j], color=color, lw=1.2,
                        alpha=0.9)
            ax.axhline(0.0, color="0.75", lw=0.5)
            rms = float(np.sqrt(np.mean(case["b6"][:, j] ** 2)))
            txt = "RMS={:.3f}\n".format(rms) + "  ".join(
                "{}:{:.2f}".format(l.replace("mix_", ""), nmse[l][ci, j])
                for l, _, _ in methods)
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.2,
                    va="bottom", ha="left",
                    bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=1.6))
            if r == 0:
                ax.set_title(f"Ro = {case['ro']:g}", fontsize=10)
            if r == 1:
                ax.set_xlabel(r"$y/h$", fontsize=9)
            if ci == 0:
                ax.set_ylabel(COMP_TEX[comp] + "\n" +
                              (r"$(-\overline{uv}/k)$" if comp == "b12"
                               else r"$(-\overline{vw}/k)$"), fontsize=10)
            ax.tick_params(labelsize=8)
    handles = [plt.Line2D([], [], color="k", lw=1.9, label="DNS")]
    handles += [plt.Line2D([], [], color=c, lw=1.3, label=l)
                for l, c, _ in methods]
    fig.suptitle(r"MIX profiles — $b_{12}$ (primary shear) vs $b_{23}$ "
                 r"(secondary-flow, weakest)", fontsize=12)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=10)
    out = os.path.join(FIG, "fig_profile_mix_b12b23.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[mixfigs] -> {out}")


def main():
    os.makedirs(FIG, exist_ok=True)
    cases, methods = load_mix()
    plot_full(cases, methods)
    plot_b12b23(cases, methods)
    for label, _, preds in methods:
        n = per_comp_nmse(cases, preds)
        print(f"[mixfigs] {label} per-case NMSE matrix (5x6):")
        for ci, case in enumerate(cases):
            print("  Ro{:>5g}: ".format(case["ro"]) +
                  " ".join(f"{c}={n[ci, j]:.3f}" for j, c in enumerate(CMP)))
    print("[mixfigs] ALL DONE")


if __name__ == "__main__":
    main()
