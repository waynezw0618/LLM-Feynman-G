# -*- coding: utf-8 -*-
"""rc_asm_basisB_figs.py — 方案B (纯张量基 b23) 出图 → outputs/asm/t23basisB_paper/figs/."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "asm"
FIG = OUT / "t23basisB_paper" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

ROS = [7.5, 15, 30, 75, 150]
CHANS = [1, 5, 6, 10]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
CG, CE = "#1D9E75", "#534AB7"


def cancellation(variant):
    d = np.load(OUT / f"basisB_targets_{variant}.npz")
    rms_b, tot = [], []
    for i in range(5):
        b = d[f"c{i}_b23"]
        terms = [d[f"c{i}_a{m}"] * d[f"c{i}_T{m}t"] for m in CHANS]
        rms_b.append(float(np.sqrt(np.mean(b ** 2))))
        tot.append(float(sum(np.sqrt(np.mean(t ** 2)) for t in terms)))
    return np.array(rms_b), np.array(tot)


def fig1():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w = 0.36
    x = np.arange(5)
    for k, (v, c) in enumerate((("g", CG), ("e", CE))):
        rb, tot = cancellation(v)
        ax.bar(x + (k - 0.5) * w, tot / rb, w, color=c, label=f"{v} basis")
        for xi, val in zip(x + (k - 0.5) * w, tot / rb):
            ax.text(xi, val * 1.15, f"{val:.0f}", ha="center", fontsize=7.5, color=c)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"Ro{r:g}" for r in ROS])
    ax.set_ylabel(r"cancellation ratio  $\sum_m \|C_m\| / \|b_{23}\|$")
    ax.set_title("b23 tensor-basis decomposition is ill-conditioned", fontsize=10)
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig1_cancellation_ratio.png"); plt.close(fig)


def fig2():
    forms = ["scalar f(roloc,yw)\n5 param (current LGF)", "scalar 10-basis",
             "pure T, const a_m", "pure T, a_m(roloc)",
             "pure T, a_m(roloc,yw) 6b", "pure T, a_m(roloc,yw) 10b",
             "scalar + T correction", "structured joint (this work)\ng 0.0530 / e 0.0482"]
    vals = [0.0555, 0.0532, 3.175, 0.688, 0.0849, 0.0528, 0.0081, 0.0506]
    cols = ["#5F5E5A", "#888780", "#A32D2D", "#BA7517", "#BA7517", "#BA7517",
            "#1D9E75", "#639922"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(forms))[::-1]
    ax.barh(y, vals, 0.72, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v * 1.12, yi, f"{v:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(forms, fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(2e-3, 2e1)
    ax.set_xlabel("loss_vw  (NMSE of b23, all-5-Ro joint fit)")
    ax.set_title("In-sample accuracy by model form (g/e basis)", fontsize=10)
    fig.savefig(FIG / "fig2_form_comparison.png"); plt.close(fig)


def fig3(champion_only=False):
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.4), sharey=True)
    for k, v in enumerate(("g", "e")):
        d = np.load(OUT / f"basisB_targets_{v}.npz")
        for ax, i in zip(axes, range(5)):
            b23 = d[f"c{i}_b23"]
            rec = sum(d[f"c{i}_a{m}"] * d[f"c{i}_T{m}t"] for m in CHANS)
            yh = np.load(OUT / f"dataset_trueasm_{v}.npz")[f"c{i}_yh"][
                np.load(OUT / f"dataset_trueasm_{v}.npz")[f"c{i}_valid"].astype(bool)]
            o = np.argsort(yh)
            ax.plot(yh[o], b23[o], "k-", lw=1.4,
                    label="DNS $b_{23}$" if (i == 0 and k == 0) else None)
            ax.plot(yh[o], rec[o], "--", lw=1.3,
                    color=CG if v == "g" else CE,
                    label=f"G$\\times$T ({v} basis)" if i == 0 else None)
            ax.set_title(f"Ro$_{{\\tau}}$={ROS[i]:g}", fontsize=8.5)
            ax.set_xlabel("y/h")
    axes[0].set_ylabel("$b_{23}$")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Structured tensor-basis closure vs DNS", fontsize=10, y=1.06)
    fig.savefig(FIG / "fig3_profiles.png"); plt.close(fig)


def fig4():
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.2))
    for r, v in enumerate(("g", "e")):
        d = np.load(OUT / f"basisB_targets_{v}.npz")
        ds = np.load(OUT / f"dataset_trueasm_{v}.npz")
        for cix, m in enumerate(CHANS):
            ax = axes[r, cix]
            for i, c in enumerate((CG, CE, "#BA7517", "#A32D2D", "#888780")):
                yh = ds[f"c{i}_yh"][ds[f"c{i}_valid"].astype(bool)]
                a = d[f"c{i}_a{m}"]
                o = np.argsort(yh)
                ax.plot(yh[o], a[o], color=c, lw=1.2 if i < 2 else 0.9,
                        alpha=1.0 if i < 2 else 0.55,
                        label=f"Ro{ROS[i]:g}" if cix == 0 and r == 0 else None)
            ax.set_title(rf"$a_{{{m}}}(\Theta)$   ({v} basis)", fontsize=8.5)
            ax.set_xlabel("y/h")
            if cix == 0:
                ax.set_ylabel("coefficient")
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.suptitle("Jointly identified tensor-basis coefficients", fontsize=10, y=1.02)
    fig.savefig(FIG / "fig4_coefficients.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("[figs] ->", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("   ", p.name)
