"""shipman.rc_sm_figs — 论文级出图 (300dpi PNG → outputs/shipman/figs_paper/).

图内全英文 (正文中文约定). 数据源: runs_sm/*/result.json + eval_summary.json
+ 现场重仿真 (fig2 用 kvlcc2() 直出).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from shipman.coeff_kvlcc2 import kvlcc2          # noqa: E402
from shipman.mmg_sim import simulate             # noqa: E402

FIG_DIR = os.path.join(REPO, "outputs", "shipman", "figs_paper")
RUNS = os.path.join(REPO, "outputs", "shipman", "runs_sm")
plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3})


def _load_eval():
    p = os.path.join(REPO, "outputs", "shipman", "eval_summary.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_result(rd):
    p = os.path.join(RUNS, rd, "result.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def fig1_framework():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc="#dbeafe", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    fc=fc, ec="#1e40af", lw=1.2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, style="-|>", color="#1e3a8a"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=14, color=color, lw=1.4))

    box(0.3, 4.6, 2.3, 1.0, "MMG Simulator\n(KVLCC2, Y&Y2015)\nturn35± / zig10·20", fc="#dbeafe")
    box(3.4, 4.6, 2.4, 1.0, "Feature Library\nV1: 26 base cols\nV2: +12 D^alpha\nV3: +5 memory\nV4: 43 combined", fc="#dcfce7", fs=8)
    box(6.6, 4.6, 3.0, 1.0, "Grey-box targets\ny_Y, y_N, y_X\n(prop/rudder deducted)", fc="#fef9c3", fs=8)
    box(6.6, 2.9, 3.0, 1.0, "LGF dual-loop\ninner: GP graph evolution\nouter: LLM ops (G_INIT/M_GUIDE)\n+ Feynman priors", fc="#f3e8ff", fs=8)
    box(3.4, 2.9, 2.4, 1.0, "Candidate DSL\n+ fitted theta\n(global C# only)", fc="#fce7f3", fs=8)
    box(0.3, 2.9, 2.3, 1.0, "Evaluation\ncoef recovery\n+ trajectory re-int.", fc="#fee2e2", fs=8)
    box(2.2, 1.0, 5.6, 1.0, "Validation loop: misabsorption check (D^alpha -> 0 on V1 data) / noise robustness / ablation (GP vs +LLM)", fc="#e0f2fe", fs=8)

    arrow(2.6, 5.1, 3.4, 5.1); arrow(5.8, 5.1, 6.6, 5.1)
    arrow(8.1, 4.6, 8.1, 3.9); arrow(6.6, 3.4, 5.8, 3.4)
    arrow(3.4, 3.4, 2.6, 3.4); arrow(1.45, 2.9, 1.45, 2.0, style="-|>")
    arrow(7.8, 2.0, 5.8, 2.9, style="-|>", color="#991b1b")
    ax.text(6.9, 2.35, "feedback", fontsize=8, color="#991b1b")
    ax.set_title("LGF-based grey-box identification framework for ship maneuvering", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig1_framework.png")); plt.close(fig)


def fig2_validation():
    s = kvlcc2()
    out_t = simulate(s, "turn35p", t_end=900.0)
    out_z = simulate(s, "zig1010", t_end=600.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    a1.plot(out_t["x"], out_t["y"], "b-", lw=1.5, label="present sim")
    # TD 标注
    psi = np.abs(out_t["psi"])
    i180 = int(np.argmax(psi >= np.pi))
    i360 = int(np.argmax(psi >= 2*np.pi)) if np.any(psi >= 2*np.pi) else -1
    a1.plot([out_t["x"][i180], out_t["x"][i360]], [out_t["y"][i180], out_t["y"][i360]],
            "r--", lw=1.2, label=f"TD = 2.76L (cal 3.36L, +18%)")
    a1.set_xlabel("x / L"); a1.set_ylabel("y / L")
    a1.set_title("(a) +35° turning circle"); a1.legend(fontsize=8); a1.set_aspect("equal")
    t = out_z["t"]
    a2.plot(t, np.degrees(out_z["psi"]), "b-", lw=1.4, label=r"$\psi$")
    a2.plot(t, np.degrees(out_z["delta"]), "g-", lw=1.0, alpha=0.8, label=r"$\delta$")
    psi_d = np.degrees(out_z["psi"])
    pks = []
    for i in range(1, len(psi_d)-1):
        if abs(psi_d[i]) > abs(psi_d[i-1]) and abs(psi_d[i]) >= abs(psi_d[i+1]) and abs(psi_d[i]) > 10:
            if not pks or np.sign(psi_d[i]) != np.sign(float(pks[-1][1])):
                pks.append((float(t[i]), float(psi_d[i])))
    if pks:
        a2.annotate(f"OSA1={abs(pks[0][1])-10:.1f}°\n(cal 5.2)", xy=pks[0], fontsize=8,
                    xytext=(pks[0][0]+30, pks[0][1]+6), arrowprops=dict(arrowstyle="->", lw=0.8))
    a2.axhline(10, color="gray", ls=":", lw=0.8); a2.axhline(-10, color="gray", ls=":", lw=0.8)
    a2.set_xlabel("t (s)"); a2.set_ylabel("angle (deg)")
    a2.set_title("(b) 10/10 zigzag"); a2.legend(fontsize=8, loc="lower right")
    fig.suptitle("KVLCC2 benchmark vs. Yasukawa & Yoshimura (2015)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig2_validation.png")); plt.close(fig)


def fig3_v1_recovery():
    ev = _load_eval()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, eq in zip(axes, ("Y", "N")):
        names = ["vm", "r", "vm3", "vm2r", "vmr2", "r3"]
        labels = [r"$v_m$", r"$r$", r"$v_m^3$", r"$v_m^2r$", r"$v_mr^2$", r"$r^3$"]
        for noise, color, off in ((0.0, "#2563eb", -0.18), (0.03, "#dc2626", 0.18)):
            rd = f"v1_{eq}_n{noise}_s0_gponly"
            ct = (ev.get(rd) or {}).get("coef_table") or {}
            true_v = [ct.get(n, {}).get("true", np.nan) for n in names]
            est_v = [ct.get(n, {}).get("est", np.nan) for n in names]
            if all(np.isnan(v) for v in true_v):
                continue
            x = np.arange(len(names)) + off
            ax.bar(x, est_v, width=0.32, color=color, alpha=0.85,
                   label=f"est (noise={noise:.0%})")
            if noise == 0.0:
                ax.plot(np.arange(len(names)), true_v, "k_", ms=12, mew=2, label="Y&Y2015 truth")
        ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(labels)
        ax.set_title(f"{eq}-equation coefficient recovery")
        ax.legend(fontsize=8)
    fig.suptitle("V1 (classical MMG): identified hull-force coefficients vs. literature", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig3_v1_recovery.png")); plt.close(fig)


def fig4_v2_fractional():
    ev = _load_eval()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.8))
    # (a) 各阶系数 (v2_Y n0)
    rd = "v2_Y_n0.0_s0_gponly"
    ct = (ev.get(rd) or {}).get("coef_table") or {}
    alphas = ["dv_a03", "dv_a05", "dv_a07", "dv_a09"]
    vals = [abs(ct.get(a, {}).get("est", 0.0)) for a in alphas]
    a1.bar([r"$D^{0.3}$", r"$D^{0.5}$", r"$D^{0.7}$", r"$D^{0.9}$"], vals, color="#7c3aed")
    a1.set_title("(a) fractional-order coefficients"); a1.set_ylabel("|coef|")
    # (b) cDv 恢复 (3 seeds)
    seeds_err = []
    for sd in (0, 1, 2):
        rd2 = f"v2_Y_n0.0_s{sd}_gponly"
        ct2 = (ev.get(rd2) or {}).get("coef_table") or {}
        e = ct2.get("dv_a05", {}).get("rel_err", np.nan)
        seeds_err.append(e * 100 if np.isfinite(e) else np.nan)
    tv = (ev.get(rd) or {}).get("coef_table", {}).get("dv_a05", {}).get("true", np.nan)
    a2.bar(["seed0", "seed1", "seed2"], seeds_err, color="#0891b2")
    a2.axhline(15, color="r", ls="--", lw=1, label="15% criterion")
    a2.set_title(f"(b) cDv recovery error (true={tv:.3f})"); a2.set_ylabel("rel. err (%)"); a2.legend(fontsize=8)
    # (c) 误吸收
    rdm = "v2pool_v1data_v2_Y_n0.0_s0_gponly"
    ctm = (ev.get(rdm) or {}).get("coef_table") or {}
    mvals = [abs(ctm.get(a, {}).get("est", 0.0)) for a in alphas]
    yv_abs = abs(ctm.get("vm", {}).get("est", 0.3))
    a3.bar([r"$D^{0.3}$", r"$D^{0.5}$", r"$D^{0.7}$", r"$D^{0.9}$"], mvals, color="#ea580c")
    a3.axhline(0.05 * yv_abs, color="r", ls="--", lw=1, label="5% of |Y_v|")
    a3.set_title("(c) misabsorption: V1 data, V2 pool"); a3.set_ylabel("|coef|"); a3.legend(fontsize=8)
    fig.suptitle("V2 (fractional): order selection, cDv recovery, and no-misabsorption evidence", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig4_v2_fractional.png")); plt.close(fig)


def fig5_v3_memory():
    ev = _load_eval()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4))
    rd = "v3_N_n0.0_s0_gponly"
    ct = (ev.get(rd) or {}).get("coef_table") or {}
    cols = ["conv_r_K1", "[false]conv_r_K2", "[false]conv_r_PW", "[false]dr_a05"]
    labs = [r"$K_1$ ($\tau$=5s)", r"$K_2$ (false)", "PW (false)", r"$D^{0.5}r$ (false)"]
    vals = [abs(ct.get(c, {}).get("est", 0.0)) for c in cols]
    tv = ct.get("conv_r_K1", {}).get("true", np.nan)
    a1.bar(labs, vals, color=["#059669", "#d1d5db", "#d1d5db", "#d1d5db"])
    if np.isfinite(tv):
        a1.axhline(tv, color="k", ls=":", lw=1.2, label=f"truth={tv:.2e}")
    a1.set_title("(a) memory-kernel coefficient recovery (N-eq)"); a1.set_ylabel("|coef|"); a1.legend(fontsize=8)
    a1.tick_params(axis="x", labelsize=8)
    # (b) 噪声鲁棒性: nmse vs noise per variant (N 方程)
    noises = [0.0, 0.03]
    for v, c, mk in (("v1", "#2563eb", "o"), ("v3", "#059669", "s"), ("v4", "#9333ea", "^")):
        ys = []
        for nz in noises:
            rd2 = f"{v}_N_n{nz}_s0_gponly"
            r = ev.get(rd2) or {}
            ys.append(r.get("nmse_mean", np.nan))
        a2.plot(noises, ys, marker=mk, color=c, label=f"V{v[1]}")
    a2.set_xlabel("observation noise (rel.)"); a2.set_ylabel("NMSE (N-eq)")
    a2.set_title("(b) noise robustness"); a2.legend(fontsize=8)
    fig.suptitle("V3 (memory kernel): recovery and robustness", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig5_v3_memory.png")); plt.close(fig)


def fig6_robust_ablation():
    ev = _load_eval()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4))
    # (a) NMSE vs noise 四变体 (Y 方程; 若有 1%/5% 补点也画)
    noise_grid = [n for n in (0.0, 0.01, 0.03, 0.05)
                  if any(f"v1_Y_n{n}_s0_gponly" in ev for n in [n])]
    xs = []
    for n in (0.0, 0.01, 0.03, 0.05):
        rd = f"v1_Y_n{n}_s0_gponly"
        if rd in ev:
            xs.append(n)
    for v, c, mk in (("v1", "#2563eb", "o"), ("v2", "#7c3aed", "s"),
                     ("v3", "#059669", "^"), ("v4", "#9333ea", "D")):
        ys = [ (ev.get(f"{v}_Y_n{n}_s0_gponly") or {}).get("nmse_mean", np.nan) for n in xs ]
        a1.plot(xs, ys, marker=mk, color=c, label=f"V{v[1]}")
    a1.set_xlabel("observation noise (rel.)"); a1.set_ylabel("NMSE (Y-eq)")
    a1.set_title("(a) noise robustness"); a1.legend(fontsize=8)
    # (b) 消融 GP vs +LLM
    labels, gp, ll = [], [], []
    for v in ("v1", "v2"):
        for nz in (0.0, 0.03):
            rg = (ev.get(f"{v}_Y_n{nz}_s0_gponly") or {}).get("nmse_mean", np.nan)
            rl = (ev.get(f"{v}_Y_n{nz}_s0_llm") or {}).get("nmse_mean", np.nan)
            labels.append(f"V{v[1]}\nn={nz:.0%}"); gp.append(rg); ll.append(rl)
    x = np.arange(len(labels))
    a2.bar(x - 0.18, gp, 0.36, label="GP-only", color="#64748b")
    a2.bar(x + 0.18, ll, 0.36, label="+LLM (GLM-5.3)", color="#f59e0b")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("NMSE (Y-eq)"); a2.set_title("(b) ablation: GP-only vs +LLM"); a2.legend(fontsize=8)
    fig.suptitle("Robustness and ablation", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig6_robust_ablation.png")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_framework(); print("fig1 done")
    fig2_validation(); print("fig2 done")
    fig3_v1_recovery(); print("fig3 done")
    fig4_v2_fractional(); print("fig4 done")
    fig5_v3_memory(); print("fig5 done")
    fig6_robust_ablation(); print("fig6 done")
